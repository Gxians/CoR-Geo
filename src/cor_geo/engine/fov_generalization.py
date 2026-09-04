"""Parameter-free geometry for evaluation at unseen fields of view."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from cor_geo.models.content_order_encoder import ConservativeAngularResampler


@dataclass(frozen=True)
class EvaluationFovGeometry:
    """Resolved crop, patch-grid, and canonical direction sizes for one FoV."""

    fov_deg: int
    physical_crop_width: int
    aligned_input_width: int
    source_patch_columns: int
    target_direction_bins: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-serializable provenance."""
        return {key: int(value) for key, value in asdict(self).items()}


def resolve_evaluation_fov_geometry(
    fovs: Sequence[int],
    model_config: dict[str, Any],
) -> dict[int, EvaluationFovGeometry]:
    """Resolve unseen FoVs without changing any learned model parameter.

    The panorama is first cropped at its exact physical angular width.  The
    crop is then rounded *up* to a ViT patch multiple, matching the registered
    90-degree (189->196) and 70-degree (147->154) preprocessing rule.  The
    canonical direction count remains one 10-degree bin per direction.
    """
    input_config = model_config["input"]
    architecture = model_config["architecture"]
    patch_size = int(model_config["backbone"]["patch_size"])
    panorama_width = int(input_config["panorama_width"])
    angular_bins = int(architecture["angular_bins"])
    registered_widths = {
        int(key): int(value) for key, value in input_config["widths"].items()
    }
    resampling = architecture["ground_angular_resampling"]
    registered_sources = {
        int(key): int(value)
        for key, value in resampling["source_patch_columns"].items()
    }
    registered_targets = {
        int(key): int(value)
        for key, value in resampling["target_direction_bins"].items()
    }
    resolved: dict[int, EvaluationFovGeometry] = {}
    for raw_fov in fovs:
        fov = int(raw_fov)
        if not 0 < fov <= 360:
            raise ValueError(f"FoV must be in (0, 360], got {fov}")
        physical_numerator = panorama_width * fov
        if physical_numerator % 360:
            raise ValueError(
                f"FoV {fov} does not map to an integer crop at panorama width "
                f"{panorama_width}"
            )
        direction_numerator = angular_bins * fov
        if direction_numerator % 360:
            raise ValueError(
                f"FoV {fov} does not map to an integer number of canonical "
                f"direction bins for angular_bins={angular_bins}"
            )
        physical_width = physical_numerator // 360
        aligned_width = (
            registered_widths[fov]
            if fov in registered_widths
            else ((physical_width + patch_size - 1) // patch_size) * patch_size
        )
        if aligned_width % patch_size:
            raise ValueError(f"Aligned width for FoV {fov} is not patch-compatible")
        source_columns = aligned_width // patch_size
        target_bins = direction_numerator // 360
        if fov in registered_sources and registered_sources[fov] != source_columns:
            raise ValueError(f"Registered source-column geometry differs for FoV {fov}")
        if fov in registered_targets and registered_targets[fov] != target_bins:
            raise ValueError(f"Registered target-bin geometry differs for FoV {fov}")
        resolved[fov] = EvaluationFovGeometry(
            fov_deg=fov,
            physical_crop_width=physical_width,
            aligned_input_width=aligned_width,
            source_patch_columns=source_columns,
            target_direction_bins=target_bins,
        )
    if len(resolved) != len(tuple(fovs)):
        raise ValueError("Evaluation FoVs must be unique")
    return resolved


def register_evaluation_resamplers(
    model: nn.Module,
    geometries: dict[int, EvaluationFovGeometry],
    device: torch.device,
) -> None:
    """Attach fixed resamplers for unseen FoVs after strict checkpoint load."""
    for fov, geometry in geometries.items():
        key = str(int(fov))
        if key in model.ground_angular_resamplers:
            existing = model.ground_angular_resamplers[key]
            if (
                int(existing.source_bins) != geometry.source_patch_columns
                or int(existing.target_bins) != geometry.target_direction_bins
            ):
                raise ValueError(f"Existing resampler geometry differs for FoV {fov}")
            continue
        resampler = ConservativeAngularResampler(
            geometry.source_patch_columns,
            geometry.target_direction_bins,
        ).to(device)
        resampler.eval()
        model.ground_angular_resamplers[key] = resampler
