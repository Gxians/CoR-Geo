"""Curriculum lookup, static AdamW groups, and exact per-group cosine schedules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cor_geo.models.cor_geo_model import CoRGeoModel


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
        configured_prefix = int(
            self.train_config["scheduler"]["content_order_encoder"][
                "cosine_end_global_step"
            ]
        )
        dino_prefix = int(
            self.train_config["scheduler"]["dinov2"]["cosine_end_global_step"]
        )
        if configured_prefix != dino_prefix:
            raise ValueError(
                "Content--Order and DINO primary cosine endpoints must match"
            )
        if not 1 <= configured_prefix <= computed_total:
            raise ValueError(
                f"Primary cosine endpoint {configured_prefix} is outside "
                f"the {computed_total}-step training run"
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
        configured_active_steps = self.train_config["scheduler"]["dinov2"].get(
            "active_optimizer_steps"
        )
        if (
            configured_active_steps is not None
            and int(configured_active_steps) != self.primary_total_steps - self.prefix_steps
        ):
            raise ValueError("Configured DINO active steps do not match manifest-derived steps")
        overrides = self.train_config["scheduler"].get("late_cosine_overrides", [])
        if not isinstance(overrides, list):
            raise ValueError("late_cosine_overrides must be a list")
        known_names = {
            str(group["logical_name"]) for group in self.optimizer.param_groups
        }
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
                    raise ValueError(
                        "Late cosine override requires start_lr >= end_lr > 0"
                    )
            expected_start_step = last_step + 1
        if expected_start_step != self.total_steps + 1:
            raise ValueError(
                "Late cosine overrides must cover every step after the primary cosine"
            )

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
            return end_lr + 0.5 * (start_lr - end_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        return None

    def set_for_next_step(self) -> dict[str, float]:
        """Set every group LR immediately before the next optimizer step."""
        global_step_1based = self.global_step + 1
        if global_step_1based > self.total_steps:
            raise ValueError("Scheduler advanced beyond the configured training endpoint")
        head_warmup = int(
            self.train_config["scheduler"]["content_order_encoder"][
                "warmup_optimizer_steps"
            ]
        )
        dino_warmup = int(
            self.train_config["scheduler"]["dinov2"]["warmup_optimizer_steps_after_activation"]
        )
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
