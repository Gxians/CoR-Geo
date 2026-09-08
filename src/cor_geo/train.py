"""CoR-Geo optimization, scheduling, checkpointing, and training loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from cor_geo.datasets import (
    CrossViewDataset,
    ResizedRGBMemmapCache,
    SampleRequest,
    cache_request,
    collate_mixed_fov,
    read_manifest,
)
from cor_geo.losses import CoRGeoLoss
from cor_geo.mining import HardNegativeCandidateBank, load_compact_hard_pool, write_compact_hard_pool
from cor_geo.model import CoRGeoModel
from cor_geo.samplers import PlannedRankBatchSampler, build_epoch_plan, stage_for_epoch
from cor_geo.utils import (
    ExperimentConfig,
    append_jsonl,
    capture_random_state,
    configure_determinism,
    load_experiment_config,
    resolve_training_topology,
    restore_random_state,
    sha256_file,
    sha256_json,
    stable_seed,
    write_json,
    write_yaml,
)

"""Curriculum lookup, static AdamW groups, and exact per-group cosine schedules."""


def scheduled_lr(base_lr: float, min_lr: float, warmup: int, total: int, step_1based: int) -> float:
    """Return the configured linear-warmup plus cosine learning rate."""
    if not 1 <= step_1based <= total:
        raise ValueError(f"step_1based must be in [1, {total}], got {step_1based}")
    if step_1based <= warmup:
        return base_lr * step_1based / warmup
    progress = (step_1based - warmup) / (total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _named_parameters(module: nn.Module, prefix: str) -> list[tuple[str, nn.Parameter]]:
    return [(f"{prefix}.{name}", parameter) for name, parameter in module.named_parameters()]


def _split_weight_decay(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def _append_logical_groups(
    output: list[dict[str, Any]],
    logical_name: str,
    named_parameters: list[tuple[str, nn.Parameter]],
    group_config: dict[str, Any],
    default_weight_decay: float,
) -> None:
    decay, no_decay = _split_weight_decay(named_parameters)
    common = {
        "logical_name": logical_name,
        "base_lr": float(group_config["base_learning_rate"]),
        "min_lr": float(group_config["min_learning_rate"]),
        "active_from_epoch": int(group_config["active_from_epoch"]),
    }
    initial_lr = common["base_lr"] if common["active_from_epoch"] == 1 else 0.0
    if decay:
        output.append({"params": decay, "lr": initial_lr, "weight_decay": default_weight_decay, **common})
    if no_decay:
        output.append({"params": no_decay, "lr": initial_lr, "weight_decay": 0.0, **common})


def build_optimizer(model: CoRGeoModel, train_config: dict[str, Any]) -> torch.optim.AdamW:
    """Build static groups for the Content--Order encoder and shared DINO suffix."""
    optimizer_config = train_config["optimizer"]
    configured = optimizer_config["param_groups"]
    groups: list[dict[str, Any]] = []
    head_parameters = _named_parameters(model.content_order_encoder, "content_order_encoder")
    _append_logical_groups(
        groups,
        "content_order_encoder",
        head_parameters,
        configured["content_order_encoder"],
        float(optimizer_config["weight_decay"]),
    )
    for index in reversed(model.backbone.registered_indices):
        logical_name = f"dinov2_block_{index}"
        _append_logical_groups(
            groups,
            logical_name,
            _named_parameters(model.backbone.blocks[index], f"backbone.model.blocks.{index}"),
            configured[logical_name],
            float(optimizer_config["weight_decay"]),
        )
    _append_logical_groups(
        groups,
        "dinov2_final_norm",
        _named_parameters(model.backbone.model.norm, "backbone.model.norm"),
        configured["dinov2_final_norm"],
        0.0,
    )
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Optimizer parameter groups overlap")
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(ids) != expected_ids:
        raise ValueError("Optimizer parameter groups omit or add parameters")
    return torch.optim.AdamW(
        groups,
        betas=tuple(map(float, optimizer_config["betas"])),
        eps=float(optimizer_config["eps"]),
    )


@dataclass
class GroupCosineScheduler:
    """Stateless scheduler with one configured prefix and contiguous late restarts."""

    optimizer: torch.optim.Optimizer
    train_config: dict[str, Any]
    steps_per_epoch: int
    global_step: int = 0

    def __post_init__(self) -> None:
        computed_total = int(self.train_config["epochs"]) * self.steps_per_epoch
        configured_prefix = int(self.train_config["scheduler"]["content_order_encoder"]["cosine_end_global_step"])
        dino_prefix = int(self.train_config["scheduler"]["dinov2"]["cosine_end_global_step"])
        if configured_prefix != dino_prefix:
            raise ValueError("Content--Order and DINO primary cosine endpoints must match")
        if not 1 <= configured_prefix <= computed_total:
            raise ValueError(
                f"Primary cosine endpoint {configured_prefix} is outside " f"the {computed_total}-step training run"
            )
        self.total_steps = computed_total
        self.primary_total_steps = configured_prefix
        dino_starts = [
            int(group["active_from_epoch"])
            for group in self.optimizer.param_groups
            if int(group["active_from_epoch"]) > 1
        ]
        self.first_backbone_epoch = min(dino_starts) if dino_starts else 1
        self.prefix_steps = (self.first_backbone_epoch - 1) * self.steps_per_epoch
        configured_active_steps = self.train_config["scheduler"]["dinov2"].get("active_optimizer_steps")
        if (
            configured_active_steps is not None
            and int(configured_active_steps) != self.primary_total_steps - self.prefix_steps
        ):
            raise ValueError("Configured DINO active steps do not match manifest-derived steps")
        overrides = self.train_config["scheduler"].get("late_cosine_overrides", [])
        if not isinstance(overrides, list):
            raise ValueError("late_cosine_overrides must be a list")
        known_names = {str(group["logical_name"]) for group in self.optimizer.param_groups}
        expected_start_step = self.primary_total_steps + 1
        for override in overrides:
            start_epoch = int(override["epoch_start"])
            end_epoch = int(override["epoch_end"])
            first_step = (start_epoch - 1) * self.steps_per_epoch + 1
            last_step = end_epoch * self.steps_per_epoch
            if not 1 <= start_epoch <= end_epoch <= int(self.train_config["epochs"]):
                raise ValueError("Late cosine override range is outside training")
            if first_step != expected_start_step:
                raise ValueError("Late cosine overrides must be contiguous and gap-free")
            groups = override.get("groups")
            if not isinstance(groups, dict) or set(map(str, groups)) != known_names:
                raise ValueError("Every late cosine override must define every optimizer group")
            for values in groups.values():
                start_lr = float(values["start_lr"])
                end_lr = float(values["end_lr"])
                if not 0.0 < end_lr <= start_lr:
                    raise ValueError("Late cosine override requires start_lr >= end_lr > 0")
            expected_start_step = last_step + 1
        if expected_start_step != self.total_steps + 1:
            raise ValueError("Late cosine overrides must cover every step after the primary cosine")

    def _late_override_lr(
        self,
        logical_name: str,
        global_step_1based: int,
    ) -> float | None:
        overrides = self.train_config["scheduler"].get("late_cosine_overrides", [])
        for override in overrides:
            first_step = (int(override["epoch_start"]) - 1) * self.steps_per_epoch + 1
            last_step = int(override["epoch_end"]) * self.steps_per_epoch
            if not first_step <= global_step_1based <= last_step:
                continue
            values = override["groups"][logical_name]
            override_steps = last_step - first_step + 1
            offset = global_step_1based - first_step
            progress = 1.0 if override_steps == 1 else offset / (override_steps - 1)
            start_lr = float(values["start_lr"])
            end_lr = float(values["end_lr"])
            return end_lr + 0.5 * (start_lr - end_lr) * (1.0 + math.cos(math.pi * progress))
        return None

    def set_for_next_step(self) -> dict[str, float]:
        """Set every group LR immediately before the next optimizer step."""
        global_step_1based = self.global_step + 1
        if global_step_1based > self.total_steps:
            raise ValueError("Scheduler advanced beyond the configured training endpoint")
        head_warmup = int(self.train_config["scheduler"]["content_order_encoder"]["warmup_optimizer_steps"])
        dino_warmup = int(self.train_config["scheduler"]["dinov2"]["warmup_optimizer_steps_after_activation"])
        logical_values: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            logical_name = str(group["logical_name"])
            override_value = self._late_override_lr(
                logical_name,
                global_step_1based,
            )
            if override_value is not None:
                value = override_value
            elif global_step_1based > self.primary_total_steps:
                raise ValueError("A post-prefix step has no configured cosine override")
            elif int(group["active_from_epoch"]) == 1:
                value = scheduled_lr(
                    float(group["base_lr"]),
                    float(group["min_lr"]),
                    head_warmup,
                    self.primary_total_steps,
                    global_step_1based,
                )
            else:
                active_epoch = int(group["active_from_epoch"])
                prefix_steps = (active_epoch - 1) * self.steps_per_epoch
                active_total = self.primary_total_steps - prefix_steps
                if global_step_1based <= prefix_steps:
                    value = 0.0
                else:
                    active_step = global_step_1based - prefix_steps
                    value = scheduled_lr(
                        float(group["base_lr"]),
                        float(group["min_lr"]),
                        dino_warmup,
                        active_total,
                        active_step,
                    )
            group["lr"] = value
            logical_values[logical_name] = value
        return logical_values

    def step_completed(self) -> None:
        """Advance after exactly one optimizer step."""
        self.global_step += 1

    @property
    def backbone_active_step(self) -> int:
        """Return completed DINO update steps."""
        return max(0, self.global_step - self.prefix_steps)

    def group_active_step(self, active_from_epoch: int) -> int:
        """Return completed optimizer steps since one logical group was activated."""
        prefix = (int(active_from_epoch) - 1) * self.steps_per_epoch
        return max(0, self.global_step - prefix)

    def state_dict(self) -> dict[str, int]:
        """Serialize exact counters."""
        return {"global_step": self.global_step, "backbone_active_step": self.backbone_active_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore and cross-check exact counters."""
        global_step = int(state["global_step"])
        expected_active = max(0, global_step - self.prefix_steps)
        if int(state["backbone_active_step"]) != expected_active:
            raise ValueError("Backbone active-step counter is inconsistent")
        self.global_step = global_step


def optimizer_param_group_manifest(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return serializable logical optimizer group metadata."""
    return [
        {
            "logical_name": group["logical_name"],
            "base_lr": group["base_lr"],
            "min_lr": group["min_lr"],
            "active_from_epoch": group["active_from_epoch"],
            "weight_decay": group["weight_decay"],
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]


"""Epoch-boundary checkpoint persistence and strict resume validation."""


def _logical_stage(completed_epoch: int, resolved_config: dict[str, Any]) -> str:
    matches = [
        str(stage["name"])
        for stage in resolved_config["train"]["stages"]
        if int(stage["epoch_start"]) <= completed_epoch <= int(stage["epoch_end"])
    ]
    if len(matches) != 1:
        raise ValueError(f"Completed epoch {completed_epoch} belongs to {len(matches)} stages")
    return matches[0]


def _trainable_parameter_names(model: nn.Module) -> list[str]:
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def _atomic_torch_save(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: GroupCosineScheduler,
    completed_epoch: int,
    resolved_config: dict[str, Any],
    provenance: dict[str, Any],
    next_epoch_schedule_seed: int,
) -> dict[str, Any]:
    """Construct the complete fixed checkpoint payload."""
    underlying = model.module if hasattr(model, "module") else model
    frozen, active = underlying.backbone.approved_parameter_names()
    group_manifest = optimizer_param_group_manifest(optimizer)
    group_states = [
        {
            "logical_name": group["logical_name"],
            "base_lr": float(group["base_lr"]),
            "min_lr": float(group["min_lr"]),
            "current_lr": float(group["lr"]),
            "active_step": (
                scheduler.global_step
                if int(group["active_from_epoch"]) == 1
                else scheduler.group_active_step(int(group["active_from_epoch"]))
            ),
        }
        for group in optimizer.param_groups
    ]
    fine_tune_state = {
        "current_epoch": completed_epoch,
        "logical_stage": _logical_stage(completed_epoch, resolved_config),
        "update_start_epoch": underlying.backbone.finetuning["update_start_epoch"],
        "block_update_start_epochs": dict(underlying.backbone.block_start_epochs),
        "final_norm_update_start_epoch": underlying.backbone.final_norm_start_epoch,
        "permanently_frozen_parameter_names": sorted(frozen),
        "optimizer_registered_dino_parameter_names": sorted(active),
        "optimizer_param_groups": group_manifest,
        "parameter_group_states": group_states,
        "backbone_active_step": scheduler.backbone_active_step,
    }
    return {
        "model_state": underlying.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "optimizer_param_group_manifest": group_manifest,
        "backbone_finetune_state": fine_tune_state,
        "backbone_active_step": scheduler.backbone_active_step,
        "trainable_parameter_names_sha256": sha256_json(_trainable_parameter_names(underlying)),
        "completed_epoch": completed_epoch,
        "global_step": scheduler.global_step,
        "resolved_config": resolved_config,
        "resolved_config_sha256": sha256_json(resolved_config),
        "next_epoch_schedule_seed": next_epoch_schedule_seed,
        **capture_random_state(),
        **provenance,
    }


def save_epoch_checkpoint(
    payload: dict[str, Any],
    checkpoint_dir: str | Path,
    completed_epoch: int,
    retain_epoch: bool = True,
) -> tuple[Path, Path]:
    """Replace last.ckpt and optionally retain an immutable epoch_NNN checkpoint."""
    directory = Path(checkpoint_dir).resolve()
    epoch_path = directory / f"epoch_{completed_epoch:03d}.ckpt"
    last_path = directory / "last.ckpt"
    if retain_epoch:
        _atomic_torch_save(payload, epoch_path, overwrite=False)
    _atomic_torch_save(payload, last_path, overwrite=True)
    return (epoch_path if retain_epoch else last_path), last_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: GroupCosineScheduler | None = None,
    restore_rng: bool = True,
    expected_resolved_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly restore model and optional training state."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    # Checkpoints are created locally by this project and contain Python/NumPy RNG states.
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {resolved}")
    underlying = model.module if hasattr(model, "module") else model
    if payload.get("resolved_config_sha256") != sha256_json(payload["resolved_config"]):
        raise ValueError("Checkpoint resolved-config hash is inconsistent")
    if expected_resolved_config is not None and payload["resolved_config_sha256"] != sha256_json(
        expected_resolved_config
    ):
        raise ValueError("Checkpoint resolved config differs from the requested run config")
    expected_names_hash = sha256_json(_trainable_parameter_names(underlying))
    if payload.get("trainable_parameter_names_sha256") != expected_names_hash:
        raise ValueError("Trainable parameter names differ from the checkpoint")
    frozen, active = underlying.backbone.approved_parameter_names()
    fine_tune_state = payload["backbone_finetune_state"]
    saved_block_starts = fine_tune_state.get("block_update_start_epochs")
    if (
        saved_block_starts is not None
        and {int(index): int(epoch) for index, epoch in saved_block_starts.items()}
        != underlying.backbone.block_start_epochs
    ):
        raise ValueError("DINO block activation schedule changed")
    saved_norm_start = fine_tune_state.get("final_norm_update_start_epoch")
    if saved_norm_start is not None and int(saved_norm_start) != underlying.backbone.final_norm_start_epoch:
        raise ValueError("DINO final-norm activation epoch changed")
    if fine_tune_state["permanently_frozen_parameter_names"] != sorted(frozen):
        raise ValueError("Permanently frozen DINO parameter names changed")
    if fine_tune_state["optimizer_registered_dino_parameter_names"] != sorted(active):
        raise ValueError("Optimizer-registered DINO parameter names changed")
    completed_epoch = int(payload["completed_epoch"])
    if fine_tune_state["logical_stage"] != _logical_stage(completed_epoch, payload["resolved_config"]):
        raise ValueError("Checkpoint logical stage is inconsistent")
    underlying.load_state_dict(payload["model_state"], strict=True)
    underlying.set_train_epoch(completed_epoch)
    if optimizer is not None:
        if scheduler is None:
            raise ValueError("Restoring optimizer state requires the matching scheduler")
        expected_manifest = optimizer_param_group_manifest(optimizer)
        if payload["optimizer_param_group_manifest"] != expected_manifest:
            raise ValueError("Optimizer parameter-group manifest changed")
        optimizer.load_state_dict(payload["optimizer_state"])
        loaded_group_states = [
            {
                "logical_name": group["logical_name"],
                "base_lr": float(group["base_lr"]),
                "min_lr": float(group["min_lr"]),
                "current_lr": float(group["lr"]),
                "active_step": (
                    int(payload["global_step"])
                    if int(group["active_from_epoch"]) == 1
                    else max(
                        0,
                        int(payload["global_step"]) - (int(group["active_from_epoch"]) - 1) * scheduler.steps_per_epoch,
                    )
                ),
            }
            for group in optimizer.param_groups
        ]
        if fine_tune_state["parameter_group_states"] != loaded_group_states:
            raise ValueError("Checkpoint parameter-group learning-rate state is inconsistent")
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
        if scheduler.global_step != int(payload["global_step"]):
            raise ValueError("Checkpoint global-step fields disagree")
        if scheduler.backbone_active_step != int(payload["backbone_active_step"]):
            raise ValueError("Checkpoint backbone active-step fields disagree")
    if restore_rng:
        restore_random_state(payload)
    return payload


"""One- or two-GPU trainer for the registered CoR-Geo protocol."""


def initialize_distributed(train_config: dict[str, Any]) -> tuple[int, int, int]:
    """Initialize an equivalent one- or two-GPU process group."""
    if not torch.cuda.is_available():
        raise RuntimeError("CoR-Geo training requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or local_rank < 0 or world_size < 1:
        raise RuntimeError("The CoR-Geo device launcher did not initialize the training workers")
    resolve_training_topology(train_config, world_size)
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(
        backend=str(train_config["distributed"]["backend"]),
        timeout=timedelta(minutes=int(train_config["distributed"]["timeout_minutes"])),
    )
    return rank, local_rank, world_size


def _broadcast(value: Any, rank: int) -> Any:
    values = [value if rank == 0 else None]
    distributed.broadcast_object_list(values, src=0)
    return values[0]


def _assert_global_unique(local_indices: torch.Tensor, world_size: int) -> None:
    gathered = [torch.empty_like(local_indices) for _ in range(world_size)]
    distributed.all_gather(gathered, local_indices)
    values = torch.cat(gathered)
    if len(torch.unique(values)) != len(values):
        raise ValueError("A global mixed batch contains duplicate locations")


class _MiningSatelliteRows(Dataset[dict[str, Any]]):
    """Contiguous satellite shard used by one mining rank."""

    def __init__(
        self,
        dataset: CrossViewDataset,
        indices: list[int],
        source_epoch: int,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.source_epoch = int(source_epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.dataset.load_satellite(
            SampleRequest(
                index=self.indices[int(position)],
                epoch=self.source_epoch,
                fov_deg=360,
                orientation_mode="hard_mining",
                source_epoch=self.source_epoch,
            )
        )


class _MiningGroundRows(Dataset[dict[str, Any]]):
    """Open one panorama once and return the four source-epoch crops."""

    def __init__(
        self,
        dataset: CrossViewDataset,
        indices: list[int],
        source_epoch: int,
        fovs: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.source_epoch = int(source_epoch)
        self.fovs = tuple(map(int, fovs))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.dataset.load_mining_ground_bundle(
            self.indices[int(position)],
            self.source_epoch,
            self.fovs,
        )


def _descriptor_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
        persistent_workers=int(workers) > 0,
        drop_last=False,
    )


def _encode_satellite_directions(
    model: CoRGeoModel,
    dataset: CrossViewDataset,
    indices: list[int],
    source_epoch: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    progress_label: str,
) -> tuple[np.ndarray, list[str]]:
    """Export normalized satellite direction sequences for self-mining."""
    was_training = model.training
    model.eval()
    directions: list[np.ndarray] = []
    ids: list[str] = []
    started = perf_counter()
    cursor = 0
    loader = _descriptor_loader(
        _MiningSatelliteRows(dataset, indices, source_epoch),
        batch_size,
        workers,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["satellite"].to(device, dtype=torch.float32, non_blocking=True)
            directions.append(model.encode_satellite(images).direction.float().cpu().numpy())
            ids.extend(map(str, batch["satellite_id"]))
            cursor += len(images)
            if batch_index == 1 or batch_index % 100 == 0:
                rate = cursor / max(perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] satellite {cursor}/{len(indices)} " f"{rate:.1f} rows/s",
                    flush=True,
                )
    if was_training:
        model.train(True)
        model.set_train_epoch(source_epoch)
    if cursor != len(indices):
        raise RuntimeError("Satellite Structure export is incomplete")
    return np.concatenate(directions).astype(np.float32, copy=False), ids


def _encode_ground_directions(
    model: CoRGeoModel,
    dataset: CrossViewDataset,
    indices: list[int],
    source_epoch: int,
    fovs: list[int],
    device: torch.device,
    batch_size: int,
    workers: int,
    progress_label: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    """Export four direction sequences while opening every panorama once."""
    was_training = model.training
    model.eval()
    directions: dict[int, list[np.ndarray]] = {fov: [] for fov in fovs}
    validities: dict[int, list[np.ndarray]] = {fov: [] for fov in fovs}
    ids: list[str] = []
    started = perf_counter()
    cursor = 0
    loader = _descriptor_loader(
        _MiningGroundRows(dataset, indices, source_epoch, fovs),
        batch_size,
        workers,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            rows = len(batch["query_id"])
            for fov in fovs:
                images = batch[f"ground_{fov}"].to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                representation = model.encode_ground(images, fov)
                directions[fov].append(representation.direction.float().cpu().numpy())
                validities[fov].append(representation.valid.cpu().numpy())
            ids.extend(map(str, batch["query_id"]))
            cursor += rows
            if batch_index == 1 or batch_index % 100 == 0:
                rate = cursor / max(perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] ground {cursor}/{len(indices)} " f"{rate:.1f} rows/s",
                    flush=True,
                )
    if was_training:
        model.train(True)
        model.set_train_epoch(source_epoch)
    if cursor != len(indices):
        raise RuntimeError("Ground Structure export is incomplete")
    return (
        {fov: np.concatenate(parts).astype(np.float32, copy=False) for fov, parts in directions.items()},
        {fov: np.concatenate(parts).astype(np.bool_, copy=False) for fov, parts in validities.items()},
        ids,
    )


def _hard_source_epoch(epoch: int, train_config: dict[str, Any]) -> int:
    eligible = [int(value) for value in train_config["hard_mining"]["refresh_after_epochs"] if int(value) < int(epoch)]
    if not eligible:
        raise ValueError(f"Epoch {epoch} has no prior mining refresh")
    return max(eligible)


def _expected_pool_metadata(
    source_epoch: int,
    checkpoint_hash: str,
    provenance: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    hard = train_config["hard_mining"]
    return {
        "source_epoch": int(source_epoch),
        "source_checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": str(provenance["manifest_hashes"]["train"]),
        "model_config_sha256": str(provenance["model_config_sha256"]),
        "candidate_strategy": str(hard["candidate_strategy"]),
        "coarse_frequency_count": int(hard["coarse_frequency_count"]),
        "coarse_candidate_locations": int(hard["coarse_candidate_locations"]),
        "keep_negative_locations": int(hard["keep_negative_locations"]),
        "crop_orientation_contract": str(hard["crop_orientation_contract"]),
        "format": str(hard["storage_format"]),
    }


def _hard_pool_root(
    run_root: Path,
    source_epoch: int,
    checkpoint_hash: str,
) -> Path:
    return run_root / "artifacts" / "hard_negatives" / checkpoint_hash / f"epoch_{source_epoch}"


def _load_hard_pools(
    run_root: Path,
    epoch: int,
    fovs: list[int],
    provenance: dict[str, Any],
    train_config: dict[str, Any],
    manifest_size: int,
) -> dict[int, np.ndarray]:
    source_epoch = _hard_source_epoch(epoch, train_config)
    hard = train_config["hard_mining"]
    if hard.get("pool_source") != "current_cor_geo_model":
        raise ValueError("Hard pools must come from the current CoR-Geo run")
    checkpoint = run_root / "checkpoints" / f"epoch_{source_epoch:03d}.ckpt"
    checkpoint_hash = sha256_file(checkpoint)
    root = _hard_pool_root(run_root, source_epoch, checkpoint_hash)
    metadata_path = root / "refresh.metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Current CoR-Geo hard pool is missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        refresh_metadata = json.load(handle)
    expected = _expected_pool_metadata(
        source_epoch,
        checkpoint_hash,
        provenance,
        train_config,
    )
    for key, value in {
        **expected,
        "fovs": list(map(int, fovs)),
        "location_count": int(manifest_size),
    }.items():
        if refresh_metadata.get(key) != value:
            raise ValueError(f"Current CoR-Geo hard-pool metadata mismatch: {key}")
    return {
        fov: load_compact_hard_pool(
            root,
            fov,
            expected,
            manifest_size,
            int(train_config["hard_mining"]["keep_negative_locations"]),
        )
        for fov in fovs
    }


def _shard_bounds(length: int, rank: int, world_size: int) -> tuple[int, int]:
    return length * rank // world_size, length * (rank + 1) // world_size


def _generate_hard_pools(
    model: CoRGeoModel,
    dataset: CrossViewDataset,
    train_manifest: Any,
    config: ExperimentConfig,
    source_epoch: int,
    source_checkpoint: Path,
    run_root: Path,
    device: torch.device,
    provenance: dict[str, Any],
    rank: int,
    world_size: int,
) -> None:
    """Generate self-contained Top-64 hard pools for all four FoVs."""
    checkpoint_hash = _broadcast(
        sha256_file(source_checkpoint) if rank == 0 else None,
        rank,
    )
    hard = config.train["hard_mining"]
    output_root = _hard_pool_root(run_root, source_epoch, checkpoint_hash)
    complete = _broadcast(
        (output_root / "refresh.metadata.json").is_file() if rank == 0 else None,
        rank,
    )
    if complete:
        return
    staging_root = run_root / "artifacts" / ".hard_negative_staging" / checkpoint_hash / f"epoch_{source_epoch}"
    setup_error: str | None = None
    if rank == 0:
        try:
            if output_root.exists():
                shutil.rmtree(output_root)
            if staging_root.exists():
                shutil.rmtree(staging_root)
            staging_root.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    setup_error = _broadcast(setup_error, rank)
    if setup_error:
        raise RuntimeError(setup_error)
    distributed.barrier()

    location_count = len(train_manifest)
    shard_start, shard_stop = _shard_bounds(location_count, rank, world_size)
    query_indices = np.arange(shard_start, shard_stop, dtype=np.int32)
    index_list = query_indices.tolist()
    fovs = list(map(int, hard["mining_fovs"]))
    batch_size = int(hard["descriptor_export_batch_size_per_gpu"])
    workers = int(hard["descriptor_export_workers_per_rank"])
    rank_root = staging_root / f"rank_{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=False)

    satellite_direction, satellite_ids = _encode_satellite_directions(
        model,
        dataset,
        index_list,
        source_epoch,
        device,
        batch_size,
        workers,
        f"rank{rank}/mine",
    )
    expected_satellite_ids = train_manifest.iloc[shard_start:shard_stop]["satellite_id"].astype(str).tolist()
    if satellite_ids != expected_satellite_ids:
        raise ValueError("Satellite mining order differs from train manifest")
    np.save(
        rank_root / "satellite_direction.fp32.npy",
        satellite_direction,
        allow_pickle=False,
    )
    del satellite_direction
    distributed.barrier()
    satellite_direction = np.concatenate(
        [
            np.load(
                staging_root / f"rank_{shard_rank:02d}" / "satellite_direction.fp32.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for shard_rank in range(world_size)
        ]
    )
    if len(satellite_direction) != location_count:
        raise RuntimeError("Satellite mining shards do not cover the manifest")

    query_directions, query_validities, query_ids = _encode_ground_directions(
        model,
        dataset,
        index_list,
        source_epoch,
        fovs,
        device,
        batch_size,
        workers,
        f"rank{rank}/mine",
    )
    expected_query_ids = train_manifest.iloc[shard_start:shard_stop]["query_id"].astype(str).tolist()
    if query_ids != expected_query_ids:
        raise ValueError("Ground mining order differs from train manifest")
    common = _expected_pool_metadata(
        source_epoch,
        checkpoint_hash,
        provenance,
        config.train,
    )
    candidate_bank = HardNegativeCandidateBank(
        satellite_direction,
        device,
        int(hard["coarse_frequency_count"]),
    )
    del satellite_direction
    for fov in fovs:
        result = candidate_bank.mine(
            query_directions[fov],
            query_validities[fov],
            query_indices,
            coarse_keep=int(hard["coarse_candidate_locations"]),
            final_keep=int(hard["keep_negative_locations"]),
            search_chunk_size=int(hard["coarse_query_chunk_size"]),
            rerank_chunk_size=int(hard["rerank_query_chunk_size"]),
        )
        fov_root = rank_root / f"fov_{fov}"
        fov_root.mkdir(parents=True, exist_ok=False)
        np.save(fov_root / "query_indices.npy", query_indices, allow_pickle=False)
        np.save(
            fov_root / "negative_indices.npy",
            result.indices,
            allow_pickle=False,
        )
        np.save(
            fov_root / "negative_scores.npy",
            result.scores,
            allow_pickle=False,
        )
        print(
            f"[rank{rank}/mine] FoV {fov} Top-{int(hard['keep_negative_locations'])} ready",
            flush=True,
        )
    distributed.barrier()

    publish_error: str | None = None
    if rank == 0:
        try:
            for fov in fovs:
                merged_queries = np.concatenate(
                    [
                        np.load(
                            staging_root / f"rank_{shard_rank:02d}" / f"fov_{fov}" / "query_indices.npy",
                            allow_pickle=False,
                        )
                        for shard_rank in range(world_size)
                    ]
                )
                if not np.array_equal(
                    merged_queries,
                    np.arange(location_count, dtype=np.int32),
                ):
                    raise RuntimeError("Mining query shards do not cover the manifest")
                write_compact_hard_pool(
                    output_root,
                    fov,
                    np.concatenate(
                        [
                            np.load(
                                staging_root / f"rank_{shard_rank:02d}" / f"fov_{fov}" / "negative_indices.npy",
                                allow_pickle=False,
                            )
                            for shard_rank in range(world_size)
                        ]
                    ),
                    np.concatenate(
                        [
                            np.load(
                                staging_root / f"rank_{shard_rank:02d}" / f"fov_{fov}" / "negative_scores.npy",
                                allow_pickle=False,
                            )
                            for shard_rank in range(world_size)
                        ]
                    ),
                    common,
                )
            write_json(
                output_root / "refresh.metadata.json",
                {
                    **common,
                    "fovs": fovs,
                    "location_count": location_count,
                    "world_size": world_size,
                },
            )
            shutil.rmtree(staging_root)
        except Exception as error:
            publish_error = f"{type(error).__name__}: {error}"
    publish_error = _broadcast(publish_error, rank)
    if publish_error:
        raise RuntimeError(f"Hard-pool publication failed: {publish_error}")
    distributed.barrier()
    del candidate_bank, query_directions, query_validities
    torch.cuda.empty_cache()


def _loss_from_config(model_config: dict[str, Any]) -> CoRGeoLoss:
    loss = model_config["loss"]
    return CoRGeoLoss(
        info_nce_temperature=float(loss["info_nce_temperature"]),
        label_smoothing=float(loss["label_smoothing"]),
        symmetric=bool(loss["symmetric"]),
        order_retrieval_weight=float(loss["order_retrieval_weight"]),
    )


def train(
    config: ExperimentConfig,
    run_name: str,
    resume: str | Path | None = None,
) -> None:
    """Train one complete CoR-Geo run using the seed stored in the config."""
    seed = int(config.train["seed"])
    rank, local_rank, world_size = initialize_distributed(config.train)
    topology = resolve_training_topology(config.train, world_size)
    configure_determinism(seed)
    device = torch.device("cuda", local_rank)
    project_root = Path(config.paths["project_root"])
    dataset_name = str(config.dataset["dataset"])
    run_root = Path(config.paths["output_root"]) / dataset_name / run_name
    manifest_root = project_root / "data_manifests" / dataset_name
    manifest_splits = tuple(map(str, config.dataset["manifest_splits"]))
    manifest_paths = {split: manifest_root / f"{split}.parquet" for split in manifest_splits}
    for path in manifest_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train_manifest = read_manifest(manifest_paths["train"])
    global_batch = int(config.train["distributed"]["global_batch_size"])
    steps_per_epoch = len(train_manifest) // global_batch
    if steps_per_epoch <= 0:
        raise ValueError("Training manifest is smaller than one global batch")

    cache_root, require_train_cache = cache_request(config.train, "train")
    for split in map(
        str,
        config.train["dataset_cache"]["required_splits"],
    ):
        split_manifest = train_manifest if split == "train" else read_manifest(manifest_paths[split])
        ResizedRGBMemmapCache(
            config.train["dataset_cache"]["root"],
            split_manifest,
            int(config.model["input"]["ground_height"]),
            int(config.model["input"]["panorama_width"]),
            int(config.model["input"]["satellite_size"][0]),
        )

    dataset = CrossViewDataset(
        train_manifest,
        global_seed=seed,
        ground_height=int(config.model["input"]["ground_height"]),
        panorama_width=int(config.model["input"]["panorama_width"]),
        satellite_size=int(config.model["input"]["satellite_size"][0]),
        ground_widths=config.model["input"]["widths"],
        resized_cache_root=cache_root,
        require_resized_cache=require_train_cache,
        dataset_name=dataset_name,
    )
    model = CoRGeoModel(
        config.model,
        dinov2_root=config.paths["dinov2_root"],
        checkpoint_path=config.paths["checkpoints"]["dinov2_vitb14"],
    ).to(device)
    optimizer = build_optimizer(model, config.train)
    scheduler = GroupCosineScheduler(optimizer, config.train, steps_per_epoch)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=bool(config.train["distributed"]["find_unused_parameters"]),
    )
    loss_function = _loss_from_config(config.model).to(device)

    setup_error: str | None = None
    if rank == 0:
        try:
            if resume is None:
                if run_root.exists():
                    raise FileExistsError(f"Run already exists: {run_root}")
                run_root.mkdir(parents=True, exist_ok=False)
                write_yaml(run_root / "config_resolved.yaml", config.as_dict())
            elif not run_root.is_dir():
                raise FileNotFoundError(f"Resume run does not exist: {run_root}")
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    setup_error = _broadcast(setup_error, rank)
    if setup_error:
        raise RuntimeError(setup_error)
    distributed.barrier()

    provenance = {
        "manifest_hashes": {split: sha256_file(path) for split, path in manifest_paths.items()},
        "model_config_sha256": sha256_json(config.model),
        "training_topology": {
            "world_size": topology.world_size,
            "per_gpu_batch_size": topology.per_gpu_batch_size,
            "global_batch_size": topology.global_batch_size,
            "samples_per_fov_per_rank": topology.samples_per_fov_per_rank,
        },
    }

    start_epoch = 1
    if resume is not None:
        payload = load_checkpoint(
            resume,
            ddp_model,
            optimizer,
            scheduler,
            restore_rng=False,
            expected_resolved_config=config.as_dict(),
        )
        saved_topology = payload.get("training_topology")
        if saved_topology is not None and int(saved_topology["world_size"]) != world_size:
            raise ValueError(
                "Resume must use the checkpoint's original GPU topology; "
                "one- and two-GPU runs are protocol-equivalent but not bitwise-interchangeable"
            )
        restore_random_state(payload)
        start_epoch = int(payload["completed_epoch"]) + 1
        distributed.barrier()
    completed_epoch = start_epoch - 1
    final_epoch = int(config.train["epochs"])
    if not start_epoch <= final_epoch:
        raise ValueError(f"Training is already complete: start={start_epoch}, final={final_epoch}")

    fovs = list(map(int, config.train["mixed_fov_batch"]["fovs"]))
    refresh_epochs = set(map(int, config.train["hard_mining"]["refresh_after_epochs"]))
    if (resume is not None) and completed_epoch in refresh_epochs:
        retained_checkpoint = run_root / "checkpoints" / f"epoch_{completed_epoch:03d}.ckpt"
        if not retained_checkpoint.is_file():
            retained_checkpoint = run_root / "checkpoints" / "last.ckpt"
        _generate_hard_pools(
            model,
            dataset,
            train_manifest,
            config,
            completed_epoch,
            retained_checkpoint,
            run_root,
            device,
            provenance,
            rank,
            world_size,
        )
    sampler = PlannedRankBatchSampler(
        rank,
        world_size,
        topology.per_gpu_batch_size,
    )
    loader_config = config.train["dataloader"]
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_mixed_fov,
        num_workers=int(loader_config["num_workers_per_rank"]),
        pin_memory=bool(loader_config["pin_memory"]),
        persistent_workers=bool(loader_config["persistent_workers"]),
        prefetch_factor=int(loader_config["prefetch_factor"]),
    )
    metrics_path = run_root / "train_metrics.jsonl"
    retained_epochs = set(map(int, config.train["checkpoint"]["retained_epochs"]))
    for epoch in range(start_epoch, final_epoch + 1):
        ddp_model.train(True)
        model.set_train_epoch(epoch)
        stage = stage_for_epoch(config.train["stages"], epoch)
        hard_pools = None
        if rank == 0 and float(stage["hard_batch_fraction"]) > 0:
            hard_pools = _load_hard_pools(
                run_root,
                epoch,
                fovs,
                provenance,
                config.train,
                len(train_manifest),
            )
        plan = (
            build_epoch_plan(
                train_manifest,
                epoch,
                seed,
                config.train,
                hard_pools,
            )
            if rank == 0
            else None
        )
        plan = _broadcast(plan, rank)
        sampler.set_plan(plan, epoch)
        for step_in_epoch, batch in enumerate(loader, start=1):
            local_indices = batch["manifest_index"].to(device, non_blocking=True)
            if step_in_epoch == 1:
                _assert_global_unique(local_indices, world_size)
            local_counts = {fov: int((batch["fov_deg"] == fov).sum()) for fov in fovs}
            expected_per_fov = topology.samples_per_fov_per_rank
            if local_counts != {fov: expected_per_fov for fov in fovs}:
                raise ValueError(f"Rank-local FoV counts are invalid: {local_counts}")

            learning_rates = scheduler.set_for_next_step()
            optimizer.zero_grad(set_to_none=True)
            ground_by_fov = {
                int(fov): tensor.to(device, non_blocking=True) for fov, tensor in batch["ground_by_fov"].items()
            }
            positions_by_fov = {
                int(fov): tensor.to(device, non_blocking=True)
                for fov, tensor in batch["ground_positions_by_fov"].items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = ddp_model(
                    ground_by_fov,
                    positions_by_fov,
                    batch["satellite"].to(device, non_blocking=True),
                )
                losses = loss_function(output)
            losses.total.backward()

            batch_kind = plan[step_in_epoch - 1].kind
            gradient_config = config.train["gradient"]
            clip_norm = float(
                gradient_config["hard_batch_clip_norm"]
                if batch_kind == "hard"
                else gradient_config["normal_batch_clip_norm"]
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=clip_norm,
            )
            optimizer.step()
            scheduler.step_completed()
            if rank == 0:
                append_jsonl(
                    metrics_path,
                    {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": scheduler.global_step,
                        "batch_kind": batch_kind,
                        "fov_counts": {str(key): value for key, value in local_counts.items()},
                        "loss": float(losses.total.detach()),
                        "joint_retrieval_loss": float(losses.joint_retrieval),
                        "order_retrieval_loss": float(losses.order_retrieval),
                        "joint_in_batch_accuracy": float(losses.joint_in_batch_accuracy),
                        "order_in_batch_accuracy": float(losses.order_in_batch_accuracy),
                        "learning_rates": learning_rates,
                    },
                )
            del output, losses, ground_by_fov, positions_by_fov

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        epoch_checkpoint: Path | None = None
        if rank == 0:
            payload = checkpoint_payload(
                ddp_model,
                optimizer,
                scheduler,
                completed_epoch=epoch,
                resolved_config=config.as_dict(),
                provenance=provenance,
                next_epoch_schedule_seed=stable_seed(
                    seed,
                    dataset_name,
                    "train",
                    epoch + 1,
                    "unique_locations",
                ),
            )
            epoch_checkpoint, _ = save_epoch_checkpoint(
                payload,
                run_root / "checkpoints",
                epoch,
                retain_epoch=epoch in retained_epochs,
            )
        checkpoint_value = _broadcast(
            str(epoch_checkpoint) if rank == 0 else None,
            rank,
        )
        distributed.barrier()
        if epoch in refresh_epochs:
            _generate_hard_pools(
                model,
                dataset,
                train_manifest,
                config,
                epoch,
                Path(checkpoint_value),
                run_root,
                device,
                provenance,
                rank,
                world_size,
            )
    distributed.destroy_process_group()


# ---- command-line interface ----

"""Launch the configured one- or two-GPU CoR-Geo training protocol."""


# Must be configured before the first CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _parse_device_ids(value: str | None) -> tuple[int, ...]:
    """Resolve the public device list while keeping single-GPU as the default."""
    if value is None:
        return (0,)
    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise ValueError("--devices must be a comma-separated list such as 0 or 0,1")
    try:
        devices = tuple(int(field) for field in fields)
    except ValueError as error:
        raise ValueError("--devices must contain non-negative integer GPU IDs") from error
    if any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative integer GPU IDs")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate GPU IDs")
    if len(devices) not in (1, 2):
        raise ValueError("CoR-Geo training supports one or two devices")
    return devices


def _launch_workers(devices_arg: str | None) -> None:
    """Hide the torchrun rendezvous behind the public ``--devices`` option."""
    if "RANK" in os.environ:
        return
    devices = _parse_device_ids(devices_arg)
    if devices_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={len(devices)}",
        "-m",
        "cor_geo.train",
        *sys.argv[1:],
    ]
    os.execvpe(sys.executable, command, os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cvact", "cvusa"), required=True)
    parser.add_argument(
        "--devices",
        help="Comma-separated GPU IDs; defaults to one GPU (for example: 0,1)",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    args = parser.parse_args()
    _launch_workers(args.devices)
    config = load_experiment_config(
        f"configs/{args.dataset}.yaml",
        args.config,
    )
    train(
        config,
        args.run_name or f"cor_geo_{args.dataset}",
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
