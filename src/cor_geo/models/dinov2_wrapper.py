"""Shared DINOv2-B/14 wrapper with delayed final-block fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PatchFeatureOutput:
    """Normalized DINO patch grid consumed by CoR-Geo."""

    tokens: Tensor
    height: int
    width: int


def _unwrap_checkpoint(value: Any) -> dict[str, Tensor]:
    if not isinstance(value, dict):
        raise ValueError("DINO checkpoint must contain a state dictionary")
    for key in ("model", "state_dict", "teacher"):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            value = candidate
            break
    if not all(isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()):
        raise ValueError("DINO checkpoint state is not a string-to-tensor mapping")
    if value and all(key.startswith("module.") for key in value):
        value = {key.removeprefix("module."): tensor for key, tensor in value.items()}
    return value


class DINOv2Backbone(nn.Module):
    """Load one shared DINOv2 and expose only its normalized patch grid."""

    def __init__(
        self,
        config: dict[str, Any],
        dinov2_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_size = int(config["patch_size"])
        self.output_dim = int(config["output_dim"])
        self.finetuning = config["finetuning"]
        if model is None:
            if dinov2_root is None or checkpoint_path is None:
                raise ValueError("dinov2_root and checkpoint_path are required when no model is injected")
            root = Path(dinov2_root).expanduser().resolve()
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"DINOv2 repository is missing: {root}")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"DINOv2 checkpoint is missing: {checkpoint}")
            model = torch.hub.load(
                repo_or_dir=str(root),
                model=config["name"],
                source="local",
                pretrained=False,
            )
            state = _unwrap_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True))
            incompatible = model.load_state_dict(state, strict=bool(config["strict_checkpoint"]))
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(
                    f"DINO checkpoint mismatch: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        self.model = model
        if not hasattr(self.model, "blocks") or not hasattr(self.model, "norm"):
            raise ValueError("DINO model must expose blocks and norm")
        self.blocks = list(self.model.blocks)
        expected = int(config["expected_transformer_blocks"])
        if len(self.blocks) != expected:
            raise ValueError(f"Expected {expected} DINO blocks, found {len(self.blocks)}")
        self.registered_indices = tuple(map(int, self.finetuning["trainable_blocks"]))
        expected_suffix = tuple(range(self.registered_indices[0], expected))
        if self.registered_indices != expected_suffix:
            raise ValueError(f"Trainable DINO blocks must be a contiguous suffix: {self.registered_indices}")
        configured_starts = self.finetuning.get("block_update_start_epochs")
        if configured_starts is None:
            default_start = int(self.finetuning["update_start_epoch"])
            configured_starts = {index: default_start for index in self.registered_indices}
        self.block_start_epochs = {int(index): int(epoch) for index, epoch in configured_starts.items()}
        if set(self.block_start_epochs) != set(self.registered_indices):
            raise ValueError("Every registered DINO block needs exactly one update-start epoch")
        if any(epoch < 1 for epoch in self.block_start_epochs.values()):
            raise ValueError("DINO block update-start epochs must be positive")
        self.final_norm_start_epoch = int(
            self.finetuning.get("final_norm_update_start_epoch", self.finetuning["update_start_epoch"])
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for index in self.registered_indices:
            for parameter in self.blocks[index].parameters():
                parameter.requires_grad_(True)
        if bool(self.finetuning["train_final_norm"]):
            for parameter in self.model.norm.parameters():
                parameter.requires_grad_(True)
        self.current_epoch = 1
        self.active_indices: tuple[int, ...] = ()
        self.updates_enabled = False
        self.norm_updates_enabled = False
        self.set_train_epoch(1)

    def train(self, mode: bool = True) -> DINOv2Backbone:
        """Keep the frozen prefix in eval while activating only approved modules."""
        super().train(mode)
        self.model.eval()
        if mode and self.updates_enabled:
            for index in self.active_indices:
                self.blocks[index].train(True)
        if mode and self.norm_updates_enabled:
            self.model.norm.train(True)
        return self

    def set_train_epoch(self, epoch: int) -> None:
        """Activate exactly the registered DINO blocks scheduled for this epoch."""
        if epoch < 1:
            raise ValueError("epoch must be 1-based")
        self.current_epoch = epoch
        self.active_indices = tuple(
            index for index in self.registered_indices if epoch >= self.block_start_epochs[index]
        )
        self.updates_enabled = bool(self.active_indices)
        self.norm_updates_enabled = bool(self.finetuning["train_final_norm"]) and epoch >= self.final_norm_start_epoch
        self.train(self.training)

    def approved_parameter_names(self) -> tuple[set[str], set[str]]:
        """Return permanently frozen and optimizer-registered parameter names."""
        active_prefixes = tuple(f"blocks.{index}." for index in self.registered_indices)
        active: set[str] = set()
        frozen: set[str] = set()
        for name, _ in self.model.named_parameters():
            if name.startswith(active_prefixes) or name.startswith("norm."):
                active.add(name)
            else:
                frozen.add(name)
        return frozen, active

    def _frozen_forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            output = self.model.forward_features(images)
        if isinstance(output, dict) and "x_norm_patchtokens" in output:
            return output["x_norm_patchtokens"]
        if isinstance(output, Tensor):
            return self._split_special_tokens(output)
        raise ValueError("DINO forward_features did not return normalized patch tokens")

    def _active_forward(self, images: Tensor) -> Tensor:
        if not hasattr(self.model, "prepare_tokens_with_masks"):
            raise ValueError("DINO model must expose prepare_tokens_with_masks for partial fine-tuning")
        if not self.active_indices:
            raise RuntimeError("Active DINO forward requested without an active transformer block")
        first_active = self.active_indices[0]
        with torch.no_grad():
            tokens = self.model.prepare_tokens_with_masks(images, None)
            for block in self.blocks[:first_active]:
                tokens = block(tokens)
            tokens = tokens.detach()
        for block in self.blocks[first_active:]:
            tokens = block(tokens)
        return self._split_special_tokens(self.model.norm(tokens))

    def _split_special_tokens(self, tokens: Tensor) -> Tensor:
        register_count = int(getattr(self.model, "num_register_tokens", 0))
        return tokens[:, 1 + register_count :]

    def encode_patches(self, images: Tensor) -> PatchFeatureOutput:
        """Return normalized patch tokens in BHWC layout."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError(f"Image shape must be divisible by patch size {self.patch_size}")
        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        use_active = self.training and self.updates_enabled
        patch_tokens = (
            self._active_forward(images) if use_active else self._frozen_forward(images)
        )
        expected_tokens = height * width
        if patch_tokens.shape != (images.shape[0], expected_tokens, self.output_dim):
            raise ValueError(
                f"Unexpected patch token shape {tuple(patch_tokens.shape)}, "
                f"expected {(images.shape[0], expected_tokens, self.output_dim)}"
            )
        tokens = patch_tokens.reshape(images.shape[0], height, width, self.output_dim)
        return PatchFeatureOutput(
            tokens=tokens,
            height=height,
            width=width,
        )
