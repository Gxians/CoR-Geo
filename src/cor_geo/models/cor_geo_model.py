"""Content--Order direction model used by CoR-Geo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cor_geo.models.content_order_encoder import (
    BilinearSquareRaySampler,
    ConservativeAngularResampler,
    ContentOrderEncoder,
)
from cor_geo.models.dinov2_wrapper import DINOv2Backbone


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
        expected_ground_height = int(model_config["input"]["ground_height"]) // int(
            backbone_config["patch_size"]
        )
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
            {
                str(fov): ConservativeAngularResampler(source_bins[fov], target_bins[fov])
                for fov in source_bins
            }
        )
        self.satellite_ray_sampler = BilinearSquareRaySampler(
            input_size=int(model_config["input"]["satellite_size"][0])
            // int(backbone_config["patch_size"]),
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
        padded_direction = direction.new_zeros(
            direction.shape[0], self.angular_bins, direction.shape[-1]
        )
        padded_order_direction = order_direction.new_zeros(
            order_direction.shape[0], self.angular_bins, order_direction.shape[-1]
        )
        padded_valid = torch.zeros(
            direction.shape[0], self.angular_bins, device=direction.device, dtype=torch.bool
        )
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
        valid = torch.ones(
            encoded.direction.shape[:2], dtype=torch.bool, device=encoded.direction.device
        )
        return self._pad_ground_representation(
            encoded.direction, encoded.order_direction, valid
        )

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
            torch.cat([value.order_direction for value in representations]).index_select(
                0, order
            ),
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
            position = ground_positions_by_fov[fov].to(
                device=representation.direction.device, dtype=torch.long
            )
            if len(position) != len(representation.direction):
                raise ValueError(f"FoV {fov} position count differs from its ground batch")
            representations.append(representation)
            positions.append(position)
        ground = self._restore_row_order(representations, positions)
        satellite_representation = self.encode_satellite(satellite)
        if len(ground.direction) != len(satellite_representation.direction):
            raise ValueError("Ground and satellite local-batch sizes differ")
        if not torch.isfinite(ground.direction).all() or not torch.isfinite(
            satellite_representation.direction
        ).all():
            raise FloatingPointError("Non-finite Content+Order descriptor")
        if not torch.isfinite(ground.order_direction).all() or not torch.isfinite(
            satellite_representation.order_direction
        ).all():
            raise FloatingPointError("Non-finite Order-only descriptor")
        return CoRGeoOutput(
            ground=ground,
            satellite=satellite_representation,
        )
