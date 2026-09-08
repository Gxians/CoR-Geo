"""DINOv2 patch backbone and the complete CoR-Geo model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cor_geo.content_order import BilinearSquareRaySampler, ConservativeAngularResampler, ContentOrderEncoder

# ---- src/cor_geo/models/dinov2_wrapper.py ----

"""Shared DINOv2-B/14 wrapper with delayed final-block fine-tuning."""


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
        patch_tokens = self._active_forward(images) if use_active else self._frozen_forward(images)
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


# ---- src/cor_geo/models/cor_geo_model.py ----

"""Content--Order direction model used by CoR-Geo."""


@dataclass(frozen=True)
class GroundRepresentation:
    """Ground directions, Order subspace, and the valid-FoV mask."""

    direction: Tensor
    order_direction: Tensor
    valid: Tensor


@dataclass(frozen=True)
class SatelliteRepresentation:
    """Satellite directions and their Order subspace."""

    direction: Tensor
    order_direction: Tensor


@dataclass(frozen=True)
class CoRGeoOutput:
    """Paired Content+Order representations."""

    ground: GroundRepresentation
    satellite: SatelliteRepresentation


class CoRGeoModel(nn.Module):
    """Shared DINOv2 with explicit Content and Order direction subspaces."""

    def __init__(
        self,
        model_config: dict[str, Any],
        dinov2_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        backbone_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = model_config
        backbone_config = model_config["backbone"]
        architecture = model_config["architecture"]
        if architecture["name"] != "column_ray_content_order":
            raise ValueError("Unexpected architecture for the CoR-Geo Content--Order model")
        self.angular_bins = int(architecture["angular_bins"])
        self.direction_dim = int(architecture["direction_dim"])
        self.backbone = DINOv2Backbone(
            backbone_config,
            dinov2_root=dinov2_root,
            checkpoint_path=checkpoint_path,
            model=backbone_model,
        )
        column_config = architecture["content_order_encoder"]
        self.ground_sequence_length = int(column_config["ground_sequence_length"])
        self.satellite_sequence_length = int(column_config["satellite_sequence_length"])
        expected_ground_height = int(model_config["input"]["ground_height"]) // int(backbone_config["patch_size"])
        if self.ground_sequence_length != expected_ground_height:
            raise ValueError("Column sequence length must equal the ground DINO patch height")
        if self.ground_sequence_length != self.satellite_sequence_length:
            raise ValueError("Ground columns and satellite rays must share one length")
        if column_config.get("shared_across_modalities") is not True:
            raise ValueError("Ground and satellite must share one Content--Order encoder")
        self.content_order_encoder = ContentOrderEncoder(
            input_dim=int(backbone_config["output_dim"]),
            hidden_dim=int(column_config["hidden_dim"]),
            content_dim=int(column_config["content_dim"]),
            order_dim=int(column_config["order_dim"]),
            joint_order_weight=float(column_config["joint_order_weight"]),
        )
        if self.direction_dim != self.content_order_encoder.hidden_dim:
            raise ValueError("direction_dim must equal the Content--Order encoder hidden_dim")
        resampling = architecture["ground_angular_resampling"]
        source_bins = {int(key): int(value) for key, value in resampling["source_patch_columns"].items()}
        target_bins = {int(key): int(value) for key, value in resampling["target_direction_bins"].items()}
        if set(source_bins) != {360, 180, 90, 70} or set(target_bins) != set(source_bins):
            raise ValueError("Ground angular resampling must define all four FoVs")
        self.ground_angular_resamplers = nn.ModuleDict(
            {str(fov): ConservativeAngularResampler(source_bins[fov], target_bins[fov]) for fov in source_bins}
        )
        self.satellite_ray_sampler = BilinearSquareRaySampler(
            input_size=int(model_config["input"]["satellite_size"][0]) // int(backbone_config["patch_size"]),
            angular_bins=self.angular_bins,
            sequence_length=self.satellite_sequence_length,
        )
        score_config = model_config["score"]
        if (
            score_config.get("name") != "fov_masked_cyclic_hard_max"
            or score_config.get("shift_reduction") != "hard_max"
        ):
            raise ValueError("CoR-Geo requires cyclic Hard-Max scoring")

    def set_train_epoch(self, epoch: int) -> None:
        self.backbone.set_train_epoch(epoch)

    def _pad_ground_representation(
        self, direction: Tensor, order_direction: Tensor, valid: Tensor
    ) -> GroundRepresentation:
        angle_count = direction.shape[1]
        if angle_count > self.angular_bins:
            raise ValueError(f"Ground direction count {angle_count} exceeds {self.angular_bins}")
        padded_direction = direction.new_zeros(direction.shape[0], self.angular_bins, direction.shape[-1])
        padded_order_direction = order_direction.new_zeros(
            order_direction.shape[0], self.angular_bins, order_direction.shape[-1]
        )
        padded_valid = torch.zeros(direction.shape[0], self.angular_bins, device=direction.device, dtype=torch.bool)
        padded_direction[:, :angle_count] = direction
        padded_order_direction[:, :angle_count] = order_direction
        padded_valid[:, :angle_count] = valid
        return GroundRepresentation(
            padded_direction,
            padded_order_direction,
            padded_valid,
        )

    def encode_ground(self, images: Tensor, fov_deg: int) -> GroundRepresentation:
        backbone = self.backbone.encode_patches(images)
        key = str(int(fov_deg))
        if key not in self.ground_angular_resamplers:
            raise ValueError(f"Unsupported ground FoV: {fov_deg}")
        tokens = self.ground_angular_resamplers[key](backbone.tokens)
        columns = tokens.flip(dims=(1,)).permute(0, 2, 1, 3)
        if columns.shape[2] != self.ground_sequence_length:
            raise RuntimeError("Ground backbone returned an unexpected vertical token length")
        encoded = self.content_order_encoder(columns)
        valid = torch.ones(encoded.direction.shape[:2], dtype=torch.bool, device=encoded.direction.device)
        return self._pad_ground_representation(encoded.direction, encoded.order_direction, valid)

    def encode_satellite(self, images: Tensor) -> SatelliteRepresentation:
        backbone = self.backbone.encode_patches(images)
        columns = self.satellite_ray_sampler(backbone.tokens)
        encoded = self.content_order_encoder(columns)
        direction = encoded.direction
        if direction.shape[1] != self.angular_bins:
            raise RuntimeError("Satellite ray sampler returned an unexpected angular width")
        return SatelliteRepresentation(direction, encoded.order_direction)

    @staticmethod
    def _restore_row_order(
        representations: list[GroundRepresentation],
        positions: list[Tensor],
    ) -> GroundRepresentation:
        if not representations:
            raise ValueError("At least one ground FoV group is required")
        concatenated_positions = torch.cat(positions)
        expected = torch.arange(len(concatenated_positions), device=concatenated_positions.device)
        if not torch.equal(torch.sort(concatenated_positions).values, expected):
            raise ValueError("Ground FoV positions must cover every local batch row once")
        order = torch.argsort(concatenated_positions)
        return GroundRepresentation(
            torch.cat([value.direction for value in representations]).index_select(0, order),
            torch.cat([value.order_direction for value in representations]).index_select(0, order),
            torch.cat([value.valid for value in representations]).index_select(0, order),
        )

    def forward(
        self,
        ground_by_fov: dict[int, Tensor],
        ground_positions_by_fov: dict[int, Tensor],
        satellite: Tensor,
    ) -> CoRGeoOutput:
        if set(ground_by_fov) != set(ground_positions_by_fov):
            raise ValueError("Ground tensors and row-position mappings differ")
        representations: list[GroundRepresentation] = []
        positions: list[Tensor] = []
        for fov in sorted(ground_by_fov, reverse=True):
            representation = self.encode_ground(ground_by_fov[fov], fov)
            position = ground_positions_by_fov[fov].to(device=representation.direction.device, dtype=torch.long)
            if len(position) != len(representation.direction):
                raise ValueError(f"FoV {fov} position count differs from its ground batch")
            representations.append(representation)
            positions.append(position)
        ground = self._restore_row_order(representations, positions)
        satellite_representation = self.encode_satellite(satellite)
        if len(ground.direction) != len(satellite_representation.direction):
            raise ValueError("Ground and satellite local-batch sizes differ")
        if not torch.isfinite(ground.direction).all() or not torch.isfinite(satellite_representation.direction).all():
            raise FloatingPointError("Non-finite Content+Order descriptor")
        if (
            not torch.isfinite(ground.order_direction).all()
            or not torch.isfinite(satellite_representation.order_direction).all()
        ):
            raise FloatingPointError("Non-finite Order-only descriptor")
        return CoRGeoOutput(
            ground=ground,
            satellite=satellite_representation,
        )
