"""Deterministic uint8 RGB memmaps for paired panorama datasets."""

from __future__ import annotations

import fcntl
import json
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cor_geo.datasets.transforms import resize_ground_base, resize_satellite_base
from cor_geo.utils.hashing import sha256_json
from cor_geo.utils.io import write_json

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
    rows = (
        fingerprint_frame
        .astype(str)
        .sort_values(["query_id", "satellite_id"])
        .to_dict(orient="records")
    )
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


def cache_metadata_path(root: str | Path, split: str) -> Path:
    return Path(root).expanduser().resolve() / str(split) / "metadata.json"


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
        datasets = (
            sorted(set(manifest["dataset"].astype(str)))
            if "dataset" in manifest.columns
            else ["cvact"]
        )
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
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
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
    datasets = (
        sorted(set(manifest["dataset"].astype(str)))
        if "dataset" in manifest.columns
        else ["cvact"]
    )
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
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{split}.building.", dir=resolved_root)
    )
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
                    f"[cache/{split}] {cursor}/{count} pairs "
                    f"({rate:.1f} pairs/s)",
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
            "satellite_file_bytes": (
                temporary / "satellite.rgb_u8.npy"
            ).stat().st_size,
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
