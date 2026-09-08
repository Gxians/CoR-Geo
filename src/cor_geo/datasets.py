"""CVACT/CVUSA indexing, preprocessing, cropping, and optional resized caches."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision

from cor_geo.utils import load_yaml, resolve_project_path, sha256_file, sha256_json, stable_seed, write_json

"""Deterministic orientation generation and circular panorama cropping."""


UINT32_RANGE = 2**32


@dataclass(frozen=True)
class CropSpec:
    """A reproducible finite-FoV crop definition."""

    orientation_u32: int
    center_px: int
    width: int
    panorama_width: int

    @property
    def center_deg(self) -> float:
        """Return the continuous protocol orientation in degrees."""
        return 360.0 * self.orientation_u32 / UINT32_RANGE

    @property
    def interval(self) -> tuple[int, int]:
        """Return the unwrapped half-open pixel interval."""
        return self.center_px - self.width // 2, self.center_px + (self.width + 1) // 2


def orientation_to_center_px(orientation_u32: int, panorama_width: int) -> int:
    """Map a uint32 orientation to a panorama pixel using integer arithmetic."""
    if not 0 <= orientation_u32 < UINT32_RANGE:
        raise ValueError(f"orientation_u32 out of range: {orientation_u32}")
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    return orientation_u32 * panorama_width // UINT32_RANGE


def center_px_to_orientation(center_px: int, panorama_width: int) -> int:
    """Encode an exact panorama pixel center as the smallest matching uint32."""
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    center = int(center_px) % int(panorama_width)
    return (center * UINT32_RANGE + panorama_width - 1) // panorama_width


def random_crop_spec(
    roll_angle_deg: int,
    fov_deg: int,
    panorama_width: int,
) -> CropSpec:
    """Apply an integer right-roll followed by a left-edge FoV crop."""
    if not 0 <= int(roll_angle_deg) <= 359:
        raise ValueError("Roll angle must be an integer in [0, 359]")
    if not 0 < int(fov_deg) <= 360:
        raise ValueError(f"FoV must be in (0, 360], got {fov_deg}")
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    if int(fov_deg) * int(panorama_width) % 360:
        raise ValueError(f"FoV {fov_deg} does not map to an exact integer width at " f"panorama width {panorama_width}")
    roll_pixels = int(roll_angle_deg) * int(panorama_width) // 360
    width = int(fov_deg) * int(panorama_width) // 360
    start_px = (-roll_pixels) % int(panorama_width)
    center_px = (start_px + width // 2) % int(panorama_width)
    orientation = center_px_to_orientation(center_px, int(panorama_width))
    return CropSpec(orientation, center_px, width, int(panorama_width))


def train_crop_spec(
    global_seed: int,
    epoch: int,
    query_id: str,
    fov_deg: int,
    panorama_width: int,
    dataset_name: str = "cvact",
) -> CropSpec:
    """Build a deterministic epoch-specific training crop."""
    orientation = stable_seed(
        global_seed,
        str(dataset_name),
        "train",
        epoch,
        query_id,
        "train_orientation",
    )
    width = panorama_width * fov_deg // 360
    if width * 360 != panorama_width * fov_deg:
        raise ValueError(f"FoV {fov_deg} does not map to an exact integer width")
    return CropSpec(orientation, orientation_to_center_px(orientation, panorama_width), width, panorama_width)


def hard_mining_crop_spec(
    global_seed: int,
    source_epoch: int,
    query_id: str,
    fov_deg: int,
    panorama_width: int,
    dataset_name: str = "cvact",
) -> CropSpec:
    """Return the exact crop shared by pool export and later hard batches."""
    orientation = stable_seed(
        global_seed,
        str(dataset_name),
        "train",
        source_epoch,
        query_id,
        "hard_mining_orientation",
    )
    width = panorama_width * fov_deg // 360
    if width * 360 != panorama_width * fov_deg:
        raise ValueError(f"FoV {fov_deg} does not map to an exact integer width")
    return CropSpec(
        orientation,
        orientation_to_center_px(orientation, panorama_width),
        width,
        panorama_width,
    )


def circular_crop(image: Image.Image, center_px: int, width: int) -> Image.Image:
    """Crop a horizontal interval with circular wrap and no padding."""
    if width <= 0 or width > image.width:
        raise ValueError(f"Crop width must be in [1, {image.width}], got {width}")
    center = center_px % image.width
    start = center - width // 2
    indices = [(start + offset) % image.width for offset in range(width)]
    if indices == list(range(indices[0], indices[0] + width)):
        return image.crop((indices[0], 0, indices[0] + width, image.height))
    left_width = image.width - (start % image.width)
    first = image.crop((start % image.width, 0, image.width, image.height))
    second = image.crop((0, 0, width - left_width, image.height))
    output = Image.new(image.mode, (width, image.height))
    output.paste(first, (0, 0))
    output.paste(second, (first.width, 0))
    return output


"""Deterministic image transformations with operation-specific seeds."""


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def resize_ground_base(
    image: Image.Image,
    height: int,
    panorama_width: int,
) -> Image.Image:
    """Return the deterministic RGB panorama stored by the input cache."""
    return vision.resize(
        image.convert("RGB"),
        [int(height), int(panorama_width)],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def resize_satellite_base(image: Image.Image, size: int) -> Image.Image:
    """Return the deterministic RGB satellite image stored by the input cache."""
    return vision.resize(
        image.convert("RGB"),
        [int(size), int(size)],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _rng(*items: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*items))


def _probability(seed_items: tuple[object, ...], probability: float) -> bool:
    return bool(_rng(*seed_items).random() < probability)


def _uniform(seed_items: tuple[object, ...], low: float, high: float) -> float:
    return float(_rng(*seed_items).uniform(low, high))


def _color_jitter(
    image: Image.Image,
    key: tuple[object, ...],
    amount: float,
    hue: float,
    probability: float,
) -> Image.Image:
    if not _probability((*key, "color_jitter_apply"), probability):
        return image
    brightness = _uniform((*key, "brightness"), 1.0 - amount, 1.0 + amount)
    contrast = _uniform((*key, "contrast"), 1.0 - amount, 1.0 + amount)
    saturation = _uniform((*key, "saturation"), 1.0 - amount, 1.0 + amount)
    hue_factor = _uniform((*key, "hue"), -hue, hue)
    image = vision.adjust_brightness(image, brightness)
    image = vision.adjust_contrast(image, contrast)
    image = vision.adjust_saturation(image, saturation)
    return vision.adjust_hue(image, hue_factor)


def _gaussian_blur(
    image: Image.Image,
    key: tuple[object, ...],
    sigma_range: tuple[float, float],
    probability: float,
) -> Image.Image:
    if not _probability((*key, "gaussian_blur_apply"), probability):
        return image
    sigma = _uniform((*key, "gaussian_blur_sigma"), *sigma_range)
    return vision.gaussian_blur(image, kernel_size=[5, 5], sigma=[sigma, sigma])


def _normalize(image: Image.Image) -> torch.Tensor:
    return vision.normalize(vision.to_tensor(image), mean=_MEAN, std=_STD)


def _random_erasing(tensor: torch.Tensor, key: tuple[object, ...]) -> torch.Tensor:
    if not _probability((*key, "random_erasing_apply"), 0.1):
        return tensor
    height, width = tensor.shape[-2:]
    generator = _rng(*key, "random_erasing_geometry")
    area = height * width
    for _ in range(10):
        target = area * float(generator.uniform(0.02, 0.08))
        aspect = float(np.exp(generator.uniform(np.log(0.3), np.log(3.3))))
        erase_h = int(round(np.sqrt(target * aspect)))
        erase_w = int(round(np.sqrt(target / aspect)))
        if 0 < erase_h < height and 0 < erase_w < width:
            top = int(generator.integers(0, height - erase_h + 1))
            left = int(generator.integers(0, width - erase_w + 1))
            output = tensor.clone()
            output[:, top : top + erase_h, left : left + erase_w] = 0.0
            return output
    return tensor


@dataclass(frozen=True)
class GroundTransform:
    """Resize, circularly crop, augment, and normalize a panorama."""

    height: int = 378
    panorama_width: int = 756
    dataset_name: str = "cvact"
    target_widths: Mapping[int, int] = field(default_factory=lambda: {360: 756, 180: 378, 90: 196, 70: 154})

    def __call__(
        self,
        image: Image.Image,
        crop: CropSpec,
        query_id: str,
        global_seed: int,
        epoch: int,
        fov_deg: int,
        training: bool,
        pre_resized: bool = False,
    ) -> torch.Tensor:
        if pre_resized:
            if image.mode != "RGB" or image.size != (
                self.panorama_width,
                self.height,
            ):
                raise ValueError("Cached ground image has the wrong mode or size")
        else:
            image = resize_ground_base(
                image,
                self.height,
                self.panorama_width,
            )
        image = circular_crop(image, crop.center_px, crop.width)
        target_width = int(self.target_widths[int(fov_deg)])
        if image.width != target_width:
            # Crop at the exact physical FoV first, then make only the small
            # patch-alignment resize required by ViT-B/14 (189->196, 147->154).
            image = vision.resize(
                image,
                [self.height, target_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        key = (global_seed, self.dataset_name, "train", epoch, query_id, fov_deg)
        if training:
            image = _color_jitter(image, key, amount=0.2, hue=0.05, probability=0.8)
            image = _gaussian_blur(image, key, sigma_range=(0.1, 2.0), probability=0.1)
        tensor = _normalize(image)
        return _random_erasing(tensor, key) if training else tensor


@dataclass(frozen=True)
class SatelliteTransform:
    """Resize, deterministically augment, and normalize a satellite image."""

    size: int = 378
    dataset_name: str = "cvact"

    def __call__(
        self,
        image: Image.Image,
        satellite_id: str,
        global_seed: int,
        epoch: int,
        fov_deg: int,
        training: bool,
        pre_resized: bool = False,
    ) -> torch.Tensor:
        if pre_resized:
            if image.mode != "RGB" or image.size != (self.size, self.size):
                raise ValueError("Cached satellite image has the wrong mode or size")
        else:
            image = resize_satellite_base(image, self.size)
        key = (
            global_seed,
            self.dataset_name,
            "train",
            epoch,
            satellite_id,
            fov_deg,
        )
        if training:
            image = _color_jitter(image, key, amount=0.15, hue=0.03, probability=0.5)
            image = _gaussian_blur(image, key, sigma_range=(0.1, 1.5), probability=0.1)
        return _normalize(image)


"""Deterministic uint8 RGB memmaps for paired panorama datasets."""


CACHE_FORMAT = "cor_geo_paired_resized_rgb_uint8_memmap_v1"
FINGERPRINT_COLUMNS = (
    "split",
    "query_id",
    "query_path",
    "satellite_id",
    "satellite_path",
)


def _single_split(manifest: pd.DataFrame) -> str:
    values = sorted(set(manifest["split"].astype(str)))
    if len(values) != 1:
        raise ValueError(f"A resized cache must contain exactly one split: {values}")
    return values[0]


def manifest_cache_fingerprint(manifest: pd.DataFrame) -> str:
    """Hash path/identity records independently of the caller's row order."""
    missing = set(FINGERPRINT_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"Cache fingerprint columns are missing: {sorted(missing)}")
    fingerprint_frame = manifest.loc[:, FINGERPRINT_COLUMNS].copy()
    fingerprint_frame.insert(
        0,
        "dataset",
        manifest["dataset"].astype(str) if "dataset" in manifest else "cvact",
    )
    rows = fingerprint_frame.astype(str).sort_values(["query_id", "satellite_id"]).to_dict(orient="records")
    return sha256_json(rows)


def cache_request(
    train_config: dict[str, Any],
    split: str,
) -> tuple[Path | None, bool]:
    """Resolve the active cache root and whether this split must be cached."""
    config = train_config.get("dataset_cache")
    if not config or not bool(config.get("enabled", False)):
        return None, False
    if str(config.get("format")) != CACHE_FORMAT:
        raise ValueError(f"Unsupported dataset cache format: {config.get('format')}")
    root = Path(str(config["root"])).expanduser().resolve()
    required = str(split) in set(map(str, config.get("required_splits", [])))
    return root, required


class _ResizeRows(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        ground_height: int,
        panorama_width: int,
        satellite_size: int,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.ground_height = int(ground_height)
        self.panorama_width = int(panorama_width)
        self.satellite_size = int(satellite_size)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(index)]
        query_path = Path(str(row["query_path"]))
        satellite_path = Path(str(row["satellite_path"]))
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        if not satellite_path.is_file():
            raise FileNotFoundError(satellite_path)
        with Image.open(query_path) as image:
            ground = np.array(
                resize_ground_base(
                    image,
                    self.ground_height,
                    self.panorama_width,
                ),
                dtype=np.uint8,
                copy=True,
            )
        with Image.open(satellite_path) as image:
            satellite = np.array(
                resize_satellite_base(image, self.satellite_size),
                dtype=np.uint8,
                copy=True,
            )
        return {
            "ground": ground,
            "satellite": satellite,
            "query_id": str(row["query_id"]),
        }


class ResizedRGBMemmapCache:
    """Validated lazy memmaps aligned to an arbitrary manifest row order."""

    def __init__(
        self,
        root: str | Path,
        manifest: pd.DataFrame,
        ground_height: int,
        panorama_width: int,
        satellite_size: int,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = _single_split(manifest)
        datasets = sorted(set(manifest["dataset"].astype(str))) if "dataset" in manifest.columns else ["cvact"]
        if len(datasets) != 1:
            raise ValueError(f"A resized cache must contain one dataset: {datasets}")
        self.dataset_name = datasets[0]
        self.split_root = self.root / self.split
        self.ground_path = self.split_root / "ground.rgb_u8.npy"
        self.satellite_path = self.split_root / "satellite.rgb_u8.npy"
        self.ids_path = self.split_root / "query_ids.npy"
        metadata_path = self.split_root / "metadata.json"
        for path in (
            self.ground_path,
            self.satellite_path,
            self.ids_path,
            metadata_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "format": CACHE_FORMAT,
            "dataset": self.dataset_name,
            "split": self.split,
            "row_count": len(manifest),
            "manifest_fingerprint": manifest_cache_fingerprint(manifest),
            "ground_shape": [
                len(manifest),
                int(ground_height),
                int(panorama_width),
                3,
            ],
            "satellite_shape": [
                len(manifest),
                int(satellite_size),
                int(satellite_size),
                3,
            ],
            "dtype": "uint8",
            "resize": "torchvision_pil_bilinear_antialias_rgb_v1",
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"Paired-image cache metadata mismatch: {mismatches}")
        for key, path in (
            ("ground_file_bytes", self.ground_path),
            ("satellite_file_bytes", self.satellite_path),
            ("query_ids_file_bytes", self.ids_path),
        ):
            if int(metadata.get(key, -1)) != path.stat().st_size:
                raise ValueError(f"Paired-image cache file size mismatch: {path}")
        cached_ids = np.load(self.ids_path, allow_pickle=False).astype(str).tolist()
        if len(cached_ids) != len(set(cached_ids)) or len(cached_ids) != len(manifest):
            raise ValueError("Cached query IDs are duplicated or incomplete")
        cache_index = {value: index for index, value in enumerate(cached_ids)}
        manifest_ids = manifest["query_id"].astype(str).tolist()
        if set(cache_index) != set(manifest_ids):
            raise ValueError("Cached and manifest query ID sets differ")
        self.manifest_to_cache = np.asarray(
            [cache_index[value] for value in manifest_ids],
            dtype=np.int64,
        )
        ground = np.load(self.ground_path, mmap_mode="r", allow_pickle=False)
        satellite = np.load(self.satellite_path, mmap_mode="r", allow_pickle=False)
        if (
            list(ground.shape) != expected["ground_shape"]
            or ground.dtype != np.uint8
            or list(satellite.shape) != expected["satellite_shape"]
            or satellite.dtype != np.uint8
        ):
            raise ValueError("Paired-image cache array shape or dtype is invalid")
        del ground, satellite
        self._ground: np.ndarray | None = None
        self._satellite: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_ground"] = None
        state["_satellite"] = None
        return state

    def _ground_array(self) -> np.ndarray:
        if self._ground is None:
            self._ground = np.load(
                self.ground_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._ground

    def _satellite_array(self) -> np.ndarray:
        if self._satellite is None:
            self._satellite = np.load(
                self.satellite_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._satellite

    def ground_image(self, manifest_index: int) -> Image.Image:
        cache_index = int(self.manifest_to_cache[int(manifest_index)])
        array = np.array(self._ground_array()[cache_index], copy=True)
        return Image.fromarray(array, mode="RGB")

    def satellite_image(self, manifest_index: int) -> Image.Image:
        cache_index = int(self.manifest_to_cache[int(manifest_index)])
        array = np.array(self._satellite_array()[cache_index], copy=True)
        return Image.fromarray(array, mode="RGB")


def build_resized_cache(
    manifest: pd.DataFrame,
    root: str | Path,
    ground_height: int,
    panorama_width: int,
    satellite_size: int,
    workers: int,
    batch_size: int,
    progress_every: int = 2048,
    *,
    _lock_acquired: bool = False,
) -> Path:
    """Build one split atomically, or strictly validate and reuse it."""
    if workers < 0 or batch_size <= 0 or progress_every <= 0:
        raise ValueError("Invalid cache build worker, batch, or progress setting")
    split = _single_split(manifest)
    datasets = sorted(set(manifest["dataset"].astype(str))) if "dataset" in manifest.columns else ["cvact"]
    if len(datasets) != 1:
        raise ValueError(f"A resized cache must contain one dataset: {datasets}")
    dataset_name = datasets[0]
    ordered = manifest.sort_values("query_id").reset_index(drop=True)
    resolved_root = Path(root).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    if not _lock_acquired:
        lock_path = resolved_root / f".{split}.build.lock"
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            return build_resized_cache(
                manifest,
                resolved_root,
                ground_height,
                panorama_width,
                satellite_size,
                workers,
                batch_size,
                progress_every,
                _lock_acquired=True,
            )
    final_root = resolved_root / split
    if final_root.exists():
        ResizedRGBMemmapCache(
            resolved_root,
            manifest,
            ground_height,
            panorama_width,
            satellite_size,
        )
        print(f"[cache/{split}] validated existing cache: {final_root}", flush=True)
        return final_root
    temporary = Path(tempfile.mkdtemp(prefix=f".{split}.building.", dir=resolved_root))
    try:
        count = len(ordered)
        ground = np.lib.format.open_memmap(
            temporary / "ground.rgb_u8.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(count, int(ground_height), int(panorama_width), 3),
        )
        satellite = np.lib.format.open_memmap(
            temporary / "satellite.rgb_u8.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(count, int(satellite_size), int(satellite_size), 3),
        )
        loader_arguments: dict[str, Any] = {
            "dataset": _ResizeRows(
                ordered,
                ground_height,
                panorama_width,
                satellite_size,
            ),
            "batch_size": int(batch_size),
            "shuffle": False,
            "num_workers": int(workers),
            "pin_memory": False,
            "persistent_workers": int(workers) > 0,
            "drop_last": False,
        }
        if int(workers) > 0:
            loader_arguments["prefetch_factor"] = 2
        loader = DataLoader(**loader_arguments)
        cursor = 0
        started = perf_counter()
        for batch in loader:
            rows = len(batch["query_id"])
            stop = cursor + rows
            ground[cursor:stop] = batch["ground"].numpy()
            satellite[cursor:stop] = batch["satellite"].numpy()
            cursor = stop
            if cursor in (rows, count) or cursor % int(progress_every) < rows:
                rate = cursor / max(perf_counter() - started, 1.0e-9)
                print(
                    f"[cache/{split}] {cursor}/{count} pairs " f"({rate:.1f} pairs/s)",
                    flush=True,
                )
        if cursor != count:
            raise RuntimeError(f"Cache build stopped at {cursor}/{count}")
        ground.flush()
        satellite.flush()
        del ground, satellite
        np.save(
            temporary / "query_ids.npy",
            np.asarray(ordered["query_id"].astype(str).tolist(), dtype=str),
            allow_pickle=False,
        )
        files = {
            "ground_file_bytes": (temporary / "ground.rgb_u8.npy").stat().st_size,
            "satellite_file_bytes": (temporary / "satellite.rgb_u8.npy").stat().st_size,
            "query_ids_file_bytes": (temporary / "query_ids.npy").stat().st_size,
        }
        write_json(
            temporary / "metadata.json",
            {
                "format": CACHE_FORMAT,
                "dataset": dataset_name,
                "split": split,
                "row_count": count,
                "manifest_fingerprint": manifest_cache_fingerprint(ordered),
                "ground_shape": [
                    count,
                    int(ground_height),
                    int(panorama_width),
                    3,
                ],
                "satellite_shape": [
                    count,
                    int(satellite_size),
                    int(satellite_size),
                    3,
                ],
                "dtype": "uint8",
                "resize": "torchvision_pil_bilinear_antialias_rgb_v1",
                **files,
            },
        )
        try:
            temporary.replace(final_root)
        except OSError:
            if not final_root.exists():
                raise
            ResizedRGBMemmapCache(
                resolved_root,
                manifest,
                ground_height,
                panorama_width,
                satellite_size,
            )
            shutil.rmtree(temporary, ignore_errors=True)
            print(
                f"[cache/{split}] reused concurrently published cache: {final_root}",
                flush=True,
            )
            return final_root
        ResizedRGBMemmapCache(
            resolved_root,
            manifest,
            ground_height,
            panorama_width,
            satellite_size,
        )
        print(f"[cache/{split}] published: {final_root}", flush=True)
        return final_root
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


"""CVACT/CVUSA annotation parsing, immutable manifests, and strict audits."""


MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "query_id",
    "query_path",
    "satellite_id",
    "satellite_path",
    "source_mat_struct",
    "source_mat_index",
]


def project_relative_path(path: str | Path, project_root: str | Path) -> str:
    """Return a portable POSIX path and reject files outside the repository."""
    resolved_root = Path(project_root).expanduser().resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Dataset path must be inside the repository: {resolved_path}") from error
    return relative.as_posix()


def _resolve_manifest_image_path(value: object, project_root: Path) -> str:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError("Manifest image paths must be repository-relative")
    resolved = resolve_project_path(relative, project_root)
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest image path escapes the repository: {relative}") from error
    return str(resolved)


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("pyarrow 17.0.0 is required for immutable Parquet manifests") from error


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path, overwrite: bool = False) -> Path:
    """Atomically write a Parquet dataframe."""
    _require_parquet_engine()
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {resolved}")
    descriptor, name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".parquet", dir=resolved.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest and resolve its repository-relative image paths."""
    _require_parquet_engine()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    frame = pd.read_parquet(resolved, engine="pyarrow")
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns {missing}: {resolved}")
    manifest_roots = [parent for parent in resolved.parents if parent.name == "data_manifests"]
    if len(manifest_roots) != 1:
        raise ValueError(f"Manifest must live under data_manifests/: {resolved}")
    project_root = manifest_roots[0].parent
    for column in ("query_path", "satellite_path"):
        frame[column] = frame[column].map(lambda value: _resolve_manifest_image_path(value, project_root))
    return frame


def parquet_row_count(path: str | Path) -> int:
    """Read a Parquet row count from file metadata without assuming a data-column name."""
    _require_parquet_engine()
    import pyarrow.parquet as parquet

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    metadata = parquet.ParquetFile(resolved).metadata
    if metadata is None:
        raise ValueError(f"Parquet metadata is unavailable: {resolved}")
    return int(metadata.num_rows)


def _load_annotations(dataset_root: Path, dataset_config: dict[str, Any]) -> dict[str, list[tuple[int, str]]]:
    mat_path = dataset_root / dataset_config["mat_file"]
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    values = loadmat(mat_path, simplify_cells=True)
    pano_ids = np.asarray(values["panoIds"]).reshape(-1)
    output: dict[str, list[tuple[int, str]]] = {}
    for split, split_config in dataset_config["splits"].items():
        struct = split_config["mat_struct"]
        field = split_config["mat_index_field"]
        indices = np.asarray(values[struct][field]).reshape(-1)
        if len(indices) != split_config["expected_annotated_count"]:
            raise ValueError(
                f"{split} annotation count is {len(indices)}, expected {split_config['expected_annotated_count']}"
            )
        rows: list[tuple[int, str]] = []
        for raw_index in indices:
            matlab_index = int(raw_index)
            if not 1 <= matlab_index <= len(pano_ids):
                raise ValueError(f"MATLAB index out of bounds in {split}: {matlab_index}")
            rows.append((matlab_index, str(pano_ids[matlab_index - 1])))
        output[split] = rows
    return output


def _load_exclusions(project_root: Path, dataset_config: dict[str, Any]) -> list[dict[str, Any]]:
    del project_root
    exclusions = dataset_config.get("exclusions", [])
    if not isinstance(exclusions, list):
        raise ValueError("exclusions must be a list")
    return exclusions


def _paths_for_id(
    dataset_root: Path,
    split_config: dict[str, Any],
    patterns: dict[str, str],
    query_id: str,
) -> tuple[Path, Path]:
    substitutions = {
        "image_root": split_config["image_root"],
        "query_id": query_id,
        "satellite_id": query_id,
    }
    return (
        dataset_root / patterns["query"].format(**substitutions),
        dataset_root / patterns["satellite"].format(**substitutions),
    )


def _verify_image(path: Path) -> tuple[str, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            if mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"}:
                raise ValueError(f"Unsupported PIL mode {mode}")
            image.verify()
            return path.suffix.lower(), size
    except Exception as error:
        raise ValueError(f"Unreadable or non-RGB-compatible image: {path}") from error


def _ids_from_directory(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ids: set[str] = set()
    for path in directory.iterdir():
        if path.is_file() and path.name.endswith(suffix):
            ids.add(path.name[: -len(suffix)])
    return ids


def _validate_split_sets(frames: dict[str, pd.DataFrame]) -> None:
    if not {"train", "val"}.issubset(frames):
        raise ValueError("Pair manifests must contain train and val splits")
    sets = {split: set(frame["query_id"].astype(str)) for split, frame in frames.items()}
    if sets["train"] & sets["val"]:
        raise ValueError("train and val IDs must be disjoint")
    if "test" in sets:
        if sets["train"] & sets["test"]:
            raise ValueError("train IDs must be disjoint from test")
        if not sets["val"] < sets["test"]:
            raise ValueError("CVACT val must be a strict subset of test")


def build_cvact_manifests(
    dataset_config_path: str | Path,
    paths_config_path: str | Path | dict[str, Any],
    overwrite: bool = False,
    verify_images: bool = True,
    progress_every: int = 10_000,
) -> dict[str, Path]:
    """Create all pair manifests after exhaustive CVACT integrity checks."""
    dataset_config = load_yaml(dataset_config_path)
    paths_config = paths_config_path if isinstance(paths_config_path, dict) else load_yaml(paths_config_path)
    project_root = Path(paths_config["project_root"]).expanduser().resolve()
    dataset_root = resolve_project_path(
        paths_config["datasets"][dataset_config["root_key"]],
        project_root,
    )
    output_root = project_root / "data_manifests" / "cvact"
    annotations = _load_annotations(dataset_root, dataset_config)
    exclusions = _load_exclusions(project_root, dataset_config)
    exclusion_keys = {(row["split"], str(row["query_id"])) for row in exclusions}
    if len(exclusion_keys) != len(exclusions):
        raise ValueError("Duplicate exclusion entries are forbidden")
    applied: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    dimensions: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    verified_images = 0
    for split, annotated_rows in annotations.items():
        split_config = dataset_config["splits"][split]
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for matlab_index, query_id in annotated_rows:
            if query_id in seen:
                raise ValueError(f"Duplicate query_id in {split}: {query_id}")
            seen.add(query_id)
            query_path, satellite_path = _paths_for_id(
                dataset_root,
                split_config,
                dataset_config["path_patterns"],
                query_id,
            )
            if (split, query_id) in exclusion_keys:
                exclusion = next(row for row in exclusions if row["split"] == split and row["query_id"] == query_id)
                if int(exclusion["source_mat_index"]) != matlab_index:
                    raise ValueError(f"Exclusion source index mismatch for {query_id}")
                applied.append(
                    {
                        **exclusion,
                        "query_exists": query_path.is_file(),
                        "satellite_exists": satellite_path.is_file(),
                    }
                )
                continue
            if not query_path.is_file() or not satellite_path.is_file():
                raise FileNotFoundError(f"Unapproved missing pair in {split}: {query_id}")
            if verify_images:
                for image_path in (query_path, satellite_path):
                    extension, size = _verify_image(image_path)
                    extensions[extension] += 1
                    dimensions[f"{size[0]}x{size[1]}"] += 1
                    verified_images += 1
                    if progress_every > 0 and verified_images % progress_every == 0:
                        print(f"verified_images={verified_images}", flush=True)
            records.append(
                {
                    "dataset": "cvact",
                    "split": split,
                    "query_id": query_id,
                    "query_path": project_relative_path(query_path, project_root),
                    "satellite_id": query_id,
                    "satellite_path": project_relative_path(
                        satellite_path,
                        project_root,
                    ),
                    "source_mat_struct": split_config["mat_struct"],
                    "source_mat_index": matlab_index,
                }
            )
        frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS).sort_values("query_id").reset_index(drop=True)
        if len(frame) != split_config["expected_final_count"]:
            raise ValueError(f"{split} final count is {len(frame)}, expected {split_config['expected_final_count']}")
        if not frame["query_id"].is_unique or not frame["satellite_id"].is_unique:
            raise ValueError(f"IDs must be unique within {split}")
        if not (frame["query_id"] == frame["satellite_id"]).all():
            raise ValueError(f"query_id/satellite_id mapping is not one-to-one in {split}")
        frames[split] = frame
    if set(exclusion_keys) != {(row["split"], row["query_id"]) for row in applied}:
        raise ValueError("Every configured exclusion must match an annotation")
    _validate_split_sets(frames)
    test_ids = set(frames["test"]["query_id"])
    test_root = dataset_root / dataset_config["splits"]["test"]["image_root"]
    query_files = _ids_from_directory(test_root / "streetview", "_grdView.jpg")
    satellite_files = _ids_from_directory(test_root / "satview_polish", "_satView_polish.jpg")
    if query_files != test_ids or satellite_files != test_ids:
        raise ValueError(
            "ANU_data_test file IDs must exactly equal valSetAll; "
            f"query extra/missing={len(query_files - test_ids)}/{len(test_ids - query_files)}, "
            f"satellite extra/missing={len(satellite_files - test_ids)}/{len(test_ids - satellite_files)}"
        )
    output_paths = {split: output_root / f"{split}.parquet" for split in frames}
    for split, frame in frames.items():
        write_parquet_atomic(frame, output_paths[split], overwrite=overwrite)
    exclusions_path = output_root / "exclusions_applied.parquet"
    write_parquet_atomic(pd.DataFrame(applied), exclusions_path, overwrite=overwrite)
    inventory = {
        "dataset_root": project_relative_path(dataset_root, project_root),
        "path_storage": "repository_relative_posix",
        "annotated_counts": {split: len(rows) for split, rows in annotations.items()},
        "final_counts": {split: len(frame) for split, frame in frames.items()},
        "image_dimensions": dict(sorted(dimensions.items())),
        "extensions": dict(sorted(extensions.items())),
        "test_query_file_count": len(query_files),
        "test_satellite_file_count": len(satellite_files),
    }
    inventory_path = output_root / "inventory.json"
    write_json(inventory_path, inventory, overwrite=overwrite)
    metadata = {
        "protocol_version": "cvact_v2.1",
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "paths_config_sha256": sha256_json(paths_config),
        "manifest_sha256": {split: sha256_file(path) for split, path in output_paths.items()},
        "exclusions_sha256": sha256_file(exclusions_path),
        "inventory_sha256": sha256_json(inventory),
    }
    metadata_path = output_root / "manifest_metadata.json"
    write_json(metadata_path, metadata, overwrite=overwrite)
    return {**output_paths, "exclusions": exclusions_path, "inventory": inventory_path, "metadata": metadata_path}


def _resolve_csv_path(dataset_root: Path, raw_path: str) -> Path:
    """Resolve one dataset-relative CSV path without allowing root escape."""
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"CVUSA CSV path must be dataset-relative: {raw_path}")
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"CVUSA CSV path escapes the dataset root: {raw_path}") from error
    return resolved


def _cvusa_id(path_value: str) -> str:
    stem = Path(str(path_value)).stem
    if not stem.isdigit() or len(stem) != 7:
        raise ValueError(f"CVUSA IDs must be seven decimal digits: {path_value}")
    return stem


def build_cvusa_manifests(
    dataset_config_path: str | Path,
    paths_config_path: str | Path | dict[str, Any],
    overwrite: bool = False,
    verify_images: bool = True,
    progress_every: int = 10_000,
) -> dict[str, Path]:
    """Create CVUSA train/val pair manifests from the official 19zl split CSVs.

    Source CSV row order is deliberately retained for stable gallery indexing.
    """
    dataset_config = load_yaml(dataset_config_path)
    if dataset_config.get("dataset") != "cvusa":
        raise ValueError("build_cvusa_manifests requires a CVUSA dataset config")
    paths_config = paths_config_path if isinstance(paths_config_path, dict) else load_yaml(paths_config_path)
    project_root = Path(paths_config["project_root"]).expanduser().resolve()
    dataset_root = resolve_project_path(
        paths_config["datasets"][dataset_config["root_key"]],
        project_root,
    )
    output_root = project_root / "data_manifests" / "cvusa"
    expected_sizes = {
        key: tuple(map(int, value)) for key, value in dataset_config["expected_source_image_sizes"].items()
    }
    frames: dict[str, pd.DataFrame] = {}
    dimensions: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    csv_hashes: dict[str, str] = {}
    verified_images = 0
    for split in ("train", "val"):
        split_config = dataset_config["splits"][split]
        csv_path = dataset_root / str(split_config["csv_file"])
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        csv_hashes[split] = sha256_file(csv_path)
        source = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
        if source.shape[1] != 3:
            raise ValueError(f"{split} CSV must contain satellite, ground, annotation columns")
        if len(source) != int(split_config["expected_count"]):
            raise ValueError(f"{split} CSV count is {len(source)}, expected {split_config['expected_count']}")
        records: list[dict[str, Any]] = []
        for row_index, values in enumerate(source.itertuples(index=False, name=None), start=1):
            satellite_raw, query_raw, annotation_raw = map(str, values)
            satellite_id = _cvusa_id(satellite_raw)
            query_id = _cvusa_id(query_raw)
            annotation_id = _cvusa_id(annotation_raw)
            if satellite_id != query_id or query_id != annotation_id:
                raise ValueError(f"CVUSA row {split}:{row_index} does not form an ID-matched triplet")
            satellite_path = _resolve_csv_path(dataset_root, satellite_raw)
            query_path = _resolve_csv_path(dataset_root, query_raw)
            annotation_path = _resolve_csv_path(dataset_root, annotation_raw)
            for role, image_path in (
                ("query", query_path),
                ("satellite", satellite_path),
                ("annotation", annotation_path),
            ):
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if verify_images:
                    extension, size = _verify_image(image_path)
                    if size != expected_sizes[role]:
                        raise ValueError(
                            f"Unexpected CVUSA {role} size at {image_path}: " f"{size}, expected {expected_sizes[role]}"
                        )
                    extensions[f"{role}:{extension}"] += 1
                    dimensions[f"{role}:{size[0]}x{size[1]}"] += 1
                    verified_images += 1
                    if progress_every > 0 and verified_images % progress_every == 0:
                        print(f"verified_images={verified_images}", flush=True)
            records.append(
                {
                    "dataset": "cvusa",
                    "split": split,
                    "query_id": query_id,
                    "query_path": project_relative_path(query_path, project_root),
                    "satellite_id": satellite_id,
                    "satellite_path": project_relative_path(
                        satellite_path,
                        project_root,
                    ),
                    # Retain the established manifest schema while recording
                    # the CSV source rather than pretending it came from MAT.
                    "source_mat_struct": str(split_config["csv_file"]),
                    "source_mat_index": row_index,
                }
            )
        frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS).reset_index(drop=True)
        if frame["query_id"].duplicated().any() or frame["satellite_id"].duplicated().any():
            raise ValueError(f"Duplicate CVUSA IDs in {split}")
        if not (frame["query_id"] == frame["satellite_id"]).all():
            raise ValueError(f"Broken CVUSA ground/satellite bijection in {split}")
        frames[split] = frame
    _validate_split_sets(frames)

    referenced_ids = set(pd.concat([frames["train"], frames["val"]])["satellite_id"])
    satellite_directory = dataset_root / str(dataset_config["satellite_inventory_root"])
    disk_ids = {path.stem for path in satellite_directory.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"}
    if dataset_config.get("strict_satellite_file_equality", False) and disk_ids != referenced_ids:
        raise ValueError(
            "CVUSA CSV union must exactly equal the bingmap/19 inventory; "
            f"extra/missing={len(disk_ids - referenced_ids)}/{len(referenced_ids - disk_ids)}"
        )

    output_paths = {split: output_root / f"{split}.parquet" for split in frames}
    for split, frame in frames.items():
        write_parquet_atomic(frame, output_paths[split], overwrite=overwrite)
    inventory = {
        "dataset_root": project_relative_path(dataset_root, project_root),
        "path_storage": "repository_relative_posix",
        "source_csv_counts": {split: len(frame) for split, frame in frames.items()},
        "source_csv_sha256": csv_hashes,
        "source_image_dimensions": dict(sorted(dimensions.items())),
        "source_extensions": dict(sorted(extensions.items())),
        "referenced_satellite_count": len(referenced_ids),
        "satellite_inventory_count": len(disk_ids),
        "manifest_order": "source_csv_row_order",
        "model_input_geometry": "shared with CVACT in configs/default.yaml",
    }
    inventory_path = output_root / "inventory.json"
    write_json(inventory_path, inventory, overwrite=overwrite)
    metadata = {
        "protocol_version": str(dataset_config["protocol_version"]),
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "paths_config_sha256": sha256_json(paths_config),
        "source_csv_sha256": csv_hashes,
        "manifest_sha256": {split: sha256_file(path) for split, path in output_paths.items()},
        "inventory_sha256": sha256_json(inventory),
    }
    metadata_path = output_root / "manifest_metadata.json"
    write_json(metadata_path, metadata, overwrite=overwrite)
    return {**output_paths, "inventory": inventory_path, "metadata": metadata_path}


def validate_manifest_frames(frames: dict[str, pd.DataFrame], expected_counts: dict[str, int]) -> None:
    """Validate loaded frames without changing them."""
    for split, frame in frames.items():
        if len(frame) != expected_counts[split]:
            raise ValueError(f"Unexpected {split} manifest count: {len(frame)}")
        if frame["query_id"].duplicated().any() or frame["satellite_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {split} manifest")
        if not (frame["query_id"].astype(str) == frame["satellite_id"].astype(str)).all():
            raise ValueError(f"Broken query/satellite bijection in {split}")
        missing_paths = [path for path in _paths(frame) if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"{split} manifest has missing paths, first={missing_paths[0]}")
    _validate_split_sets(frames)


def _paths(frame: pd.DataFrame) -> Iterable[str]:
    yield from frame["query_path"].astype(str)
    yield from frame["satellite_path"].astype(str)


def load_manifest_metadata(path: str | Path) -> dict[str, Any]:
    """Load generated JSON metadata."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid manifest metadata: {resolved}")
    return value


"""PyTorch dataset backed by immutable CVACT or CVUSA pair manifests."""


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
            ground_widths = {fov: panorama_width * fov // 360 for fov in (360, 180, 90, 70)}
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
            raise ValueError("Random evaluation supports only val and test")
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
    output: dict[str, Any] = {key: default_collate([sample[key] for sample in samples]) for key in fixed_keys}
    output.update(
        {
            "ground_by_fov": ground_by_fov,
            "ground_positions_by_fov": positions_by_fov,
            "query_id": [str(sample["query_id"]) for sample in samples],
            "satellite_id": [str(sample["satellite_id"]) for sample in samples],
        }
    )
    return output


def main() -> None:
    """Prepare portable manifests and optional resized caches."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", choices=("cvact", "cvusa"), required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    shared = load_yaml(args.config)
    dataset_path = Path("configs") / f"{args.dataset}.yaml"
    project_root = Path(shared["paths"]["project_root"]).expanduser().resolve()
    manifest_root = project_root / "data_manifests" / args.dataset
    expected_splits = tuple(map(str, load_yaml(dataset_path)["manifest_splits"]))
    manifests_exist = all((manifest_root / f"{split}.parquet").is_file() for split in expected_splits)
    if args.overwrite or not manifests_exist:
        builder = build_cvact_manifests if args.dataset == "cvact" else build_cvusa_manifests
        builder(
            dataset_path,
            shared["paths"],
            overwrite=bool(args.overwrite),
            verify_images=not bool(args.skip_image_verification),
        )
    else:
        print(f"Reusing manifests in {manifest_root}")

    if args.skip_cache:
        return
    cache_config = dict(shared["train"]["dataset_cache"])
    cache_root = str(cache_config["root"]).format(dataset=args.dataset)
    model_input = shared["model"]["input"]
    for split in ("train", "val"):
        manifest = read_manifest(manifest_root / f"{split}.parquet")
        build_resized_cache(
            manifest,
            cache_root,
            int(model_input["ground_height"]),
            int(model_input["panorama_width"]),
            int(model_input["satellite_size"][0]),
            workers=int(args.workers),
            batch_size=16,
        )


if __name__ == "__main__":
    main()
