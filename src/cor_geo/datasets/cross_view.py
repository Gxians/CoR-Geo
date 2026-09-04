"""PyTorch dataset backed by immutable CVACT or CVUSA pair manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from cor_geo.datasets.panorama_crop import (
    CropSpec,
    hard_mining_crop_spec,
    random_crop_spec,
    train_crop_spec,
)
from cor_geo.datasets.resized_cache import ResizedRGBMemmapCache
from cor_geo.datasets.transforms import GroundTransform, SatelliteTransform


@dataclass(frozen=True)
class SampleRequest:
    """Index plus deterministic transform context passed through a batch sampler."""

    index: int
    epoch: int
    fov_deg: int
    orientation_mode: Literal["train", "eval", "hard_mining", "hard_train"] = "train"
    source_epoch: int | None = None


class CrossViewDataset(Dataset[dict[str, Any]]):
    """Load one paired panorama/satellite row with protocol-compliant transforms."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        global_seed: int,
        ground_height: int = 224,
        panorama_width: int = 756,
        satellite_size: int = 378,
        ground_widths: Mapping[int, int] | None = None,
        resized_cache_root: str | Path | None = None,
        require_resized_cache: bool = False,
        dataset_name: str | None = None,
    ) -> None:
        required = {
            "split",
            "query_id",
            "query_path",
            "satellite_id",
            "satellite_path",
        }
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        self.manifest = manifest.reset_index(drop=True).copy()
        manifest_datasets = (
            sorted(set(self.manifest["dataset"].astype(str)))
            if "dataset" in self.manifest.columns
            else [str(dataset_name or "cvact")]
        )
        if len(manifest_datasets) != 1:
            raise ValueError(f"Manifest must contain one dataset: {manifest_datasets}")
        self.dataset_name = str(dataset_name or manifest_datasets[0])
        if self.dataset_name != manifest_datasets[0]:
            raise ValueError("Configured dataset differs from the manifest")
        self.global_seed = global_seed
        self.panorama_width = panorama_width
        if ground_widths is None:
            ground_widths = {
                fov: panorama_width * fov // 360 for fov in (360, 180, 90, 70)
            }
        self.ground_transform = GroundTransform(
            ground_height,
            panorama_width,
            self.dataset_name,
            {int(key): int(value) for key, value in ground_widths.items()},
        )
        self.satellite_transform = SatelliteTransform(
            satellite_size,
            self.dataset_name,
        )
        self.resized_cache: ResizedRGBMemmapCache | None = None
        if resized_cache_root is not None:
            try:
                self.resized_cache = ResizedRGBMemmapCache(
                    resized_cache_root,
                    self.manifest,
                    ground_height,
                    panorama_width,
                    satellite_size,
                )
            except FileNotFoundError:
                if require_resized_cache:
                    raise
        elif require_resized_cache:
            raise ValueError("A required resized cache root was not provided")

    def __len__(self) -> int:
        return len(self.manifest)

    def _ground_image(
        self,
        index: int,
        row: pd.Series,
    ) -> tuple[Image.Image, bool]:
        if self.resized_cache is not None:
            return self.resized_cache.ground_image(index), True
        path = Path(str(row["query_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        return Image.open(path), False

    def _satellite_image(
        self,
        index: int,
        row: pd.Series,
    ) -> tuple[Image.Image, bool]:
        if self.resized_cache is not None:
            return self.resized_cache.satellite_image(index), True
        path = Path(str(row["satellite_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        return Image.open(path), False

    @staticmethod
    def _normalize_request(request: int | SampleRequest | tuple[int, int, int]) -> SampleRequest:
        if isinstance(request, int):
            return SampleRequest(request, epoch=1, fov_deg=360)
        if isinstance(request, tuple):
            return SampleRequest(*request)
        return request

    def _request_context(
        self,
        request: int | SampleRequest | tuple[int, int, int],
    ) -> tuple[SampleRequest, pd.Series, CropSpec, bool]:
        request = self._normalize_request(request)
        row = self.manifest.iloc[request.index]
        query_id = str(row["query_id"])
        if request.orientation_mode == "train":
            crop = train_crop_spec(
                self.global_seed,
                request.epoch,
                query_id,
                request.fov_deg,
                self.panorama_width,
                self.dataset_name,
            )
            training = True
        elif request.orientation_mode in {"hard_mining", "hard_train"}:
            if request.source_epoch is None:
                raise ValueError("source_epoch is required for hard-mining crops")
            crop = hard_mining_crop_spec(
                self.global_seed,
                int(request.source_epoch),
                query_id,
                request.fov_deg,
                self.panorama_width,
                self.dataset_name,
            )
            training = request.orientation_mode == "hard_train"
        else:
            raise ValueError(f"Unsupported orientation mode: {request.orientation_mode}")
        return request, row, crop, training

    def load_ground(
        self,
        request: int | SampleRequest | tuple[int, int, int],
    ) -> dict[str, Any]:
        """Load only the ground branches for descriptor export without satellite I/O."""
        request, row, crop, training = self._request_context(request)
        query_id = str(row["query_id"])
        image, pre_resized = self._ground_image(request.index, row)
        with image:
            ground = self.ground_transform(
                image,
                crop,
                query_id,
                self.global_seed,
                request.epoch,
                request.fov_deg,
                training,
                pre_resized=pre_resized,
            )
        return {
            "ground": ground,
            "query_id": query_id,
            "orientation_u32": torch.tensor(crop.orientation_u32, dtype=torch.int64),
            "center_px": torch.tensor(crop.center_px, dtype=torch.int64),
            "fov_deg": torch.tensor(request.fov_deg, dtype=torch.int64),
            "manifest_index": torch.tensor(request.index, dtype=torch.int64),
        }

    def load_satellite(
        self,
        request: int | SampleRequest | tuple[int, int, int],
    ) -> dict[str, Any]:
        """Load only the satellite branch for shared multi-FoV descriptor export."""
        request = self._normalize_request(request)
        row = self.manifest.iloc[request.index]
        training = request.orientation_mode in {"train", "hard_train"}
        satellite_id = str(row["satellite_id"])
        image, pre_resized = self._satellite_image(request.index, row)
        with image:
            satellite = self.satellite_transform(
                image,
                satellite_id,
                self.global_seed,
                request.epoch,
                request.fov_deg,
                training,
                pre_resized=pre_resized,
            )
        return {
            "satellite": satellite,
            "satellite_id": satellite_id,
            "manifest_index": torch.tensor(request.index, dtype=torch.int64),
        }

    def load_mining_ground_bundle(
        self,
        index: int,
        source_epoch: int,
        fovs: tuple[int, ...],
    ) -> dict[str, Any]:
        """Open a panorama once using the exact crops later replayed by hard batches."""
        if not fovs:
            raise ValueError("At least one mining FoV is required")
        row = self.manifest.iloc[int(index)]
        query_id = str(row["query_id"])
        crops = {
            int(fov): hard_mining_crop_spec(
                self.global_seed,
                int(source_epoch),
                query_id,
                int(fov),
                self.panorama_width,
                self.dataset_name,
            )
            for fov in fovs
        }
        if len({crop.orientation_u32 for crop in crops.values()}) != 1:
            raise ValueError("All mining FoVs for one query must share one orientation")
        image, pre_resized = self._ground_image(index, row)
        with image:
            grounds = {
                f"ground_{fov}": self.ground_transform(
                    image,
                    crop,
                    query_id,
                    self.global_seed,
                    int(source_epoch),
                    int(fov),
                    training=False,
                    pre_resized=pre_resized,
                )
                for fov, crop in crops.items()
            }
        reference = crops[int(fovs[0])]
        return {
            **grounds,
            "query_id": query_id,
            "orientation_u32": torch.tensor(reference.orientation_u32, dtype=torch.int64),
            "manifest_index": torch.tensor(index, dtype=torch.int64),
        }

    def load_random_evaluation_ground_bundle(
        self,
        index: int,
        fovs: tuple[int, ...],
        roll_angles_deg: dict[int, int],
    ) -> dict[str, Any]:
        """Produce independently oriented random FoV crops for evaluation."""
        if not fovs:
            raise ValueError("At least one evaluation FoV is required")
        if set(map(int, fovs)) != set(map(int, roll_angles_deg)):
            raise ValueError("Every FoV requires its own random roll angle")
        row = self.manifest.iloc[int(index)]
        query_id = str(row["query_id"])
        split = str(row["split"])
        if split not in {"val", "test"}:
            raise ValueError(
                "Random evaluation supports only val and test"
            )
        crops = {
            int(fov): random_crop_spec(
                int(roll_angles_deg[int(fov)]),
                int(fov),
                self.panorama_width,
            )
            for fov in fovs
        }
        image, pre_resized = self._ground_image(index, row)
        with image:
            grounds = {
                f"ground_{fov}": self.ground_transform(
                    image,
                    crop,
                    query_id,
                    self.global_seed,
                    epoch=1,
                    fov_deg=int(fov),
                    training=False,
                    pre_resized=pre_resized,
                )
                for fov, crop in crops.items()
            }
        return {
            **grounds,
            **{
                f"orientation_u32_{fov}": torch.tensor(
                    crop.orientation_u32,
                    dtype=torch.int64,
                )
                for fov, crop in crops.items()
            },
            "query_id": query_id,
            **{
                f"random_roll_angle_deg_{fov}": torch.tensor(
                    int(roll_angles_deg[int(fov)]),
                    dtype=torch.int64,
                )
                for fov in fovs
            },
            "manifest_index": torch.tensor(index, dtype=torch.int64),
        }

    def __getitem__(self, request: int | SampleRequest | tuple[int, int, int]) -> dict[str, Any]:
        request = self._normalize_request(request)
        ground_output = self.load_ground(request)
        satellite_output = self.load_satellite(request)
        output = {
            **ground_output,
            **satellite_output,
        }
        return output


def collate_mixed_fov(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-width ground images into four same-width sub-batches.

    Satellite tensors and metadata retain the original sample order. Row
    positions let the model restore that order after encoding each FoV bucket.
    """
    if not samples:
        raise ValueError("Cannot collate an empty mixed-FoV batch")
    fovs = [int(sample["fov_deg"]) for sample in samples]
    supported = (360, 180, 90, 70)
    unexpected = set(fovs) - set(supported)
    if unexpected:
        raise ValueError(f"Unsupported FoVs in mixed batch: {sorted(unexpected)}")
    ground_by_fov: dict[int, torch.Tensor] = {}
    positions_by_fov: dict[int, torch.Tensor] = {}
    for fov in supported:
        positions = [index for index, value in enumerate(fovs) if value == fov]
        if not positions:
            continue
        ground_by_fov[fov] = torch.stack([samples[index]["ground"] for index in positions])
        positions_by_fov[fov] = torch.tensor(positions, dtype=torch.int64)
    fixed_keys = (
        "satellite",
        "orientation_u32",
        "center_px",
        "fov_deg",
        "manifest_index",
    )
    output: dict[str, Any] = {
        key: default_collate([sample[key] for sample in samples])
        for key in fixed_keys
    }
    output.update(
        {
            "ground_by_fov": ground_by_fov,
            "ground_positions_by_fov": positions_by_fov,
            "query_id": [str(sample["query_id"]) for sample in samples],
            "satellite_id": [str(sample["satellite_id"]) for sample in samples],
        }
    )
    return output
