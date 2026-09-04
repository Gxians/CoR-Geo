"""Epoch-boundary checkpoint persistence and strict resume validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cor_geo.engine.stage_scheduler import GroupCosineScheduler, optimizer_param_group_manifest
from cor_geo.reproducibility import capture_random_state, restore_random_state
from cor_geo.utils.hashing import sha256_json


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
    if expected_resolved_config is not None and payload[
        "resolved_config_sha256"
    ] != sha256_json(expected_resolved_config):
        raise ValueError(
            "Checkpoint resolved config differs from the requested run config"
        )
    expected_names_hash = sha256_json(_trainable_parameter_names(underlying))
    if payload.get("trainable_parameter_names_sha256") != expected_names_hash:
        raise ValueError("Trainable parameter names differ from the checkpoint")
    frozen, active = underlying.backbone.approved_parameter_names()
    fine_tune_state = payload["backbone_finetune_state"]
    saved_block_starts = fine_tune_state.get("block_update_start_epochs")
    if saved_block_starts is not None and {
        int(index): int(epoch) for index, epoch in saved_block_starts.items()
    } != underlying.backbone.block_start_epochs:
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
                        int(payload["global_step"])
                        - (int(group["active_from_epoch"]) - 1) * scheduler.steps_per_epoch,
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
