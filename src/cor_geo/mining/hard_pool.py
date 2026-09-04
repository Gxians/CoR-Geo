"""Atomic persistence for compact ranked hard-negative pools."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cor_geo.utils.io import write_json


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def write_compact_hard_pool(
    epoch_root: str | Path,
    fov_deg: int,
    negative_indices: np.ndarray,
    negative_scores: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """Atomically publish one dense [query, rank] hard pool."""
    root = Path(epoch_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"fov_{int(fov_deg)}"
    if target.exists():
        raise FileExistsError(target)
    indices = np.asarray(negative_indices, dtype=np.int32)
    scores = np.asarray(negative_scores, dtype=np.float32)
    if indices.ndim != 2 or scores.shape != indices.shape:
        raise ValueError("Hard-pool indices and scores must have equal 2-D shape")
    if not np.isfinite(scores).all():
        raise ValueError("Hard-pool scores must be finite")
    temporary = Path(tempfile.mkdtemp(prefix=f".fov_{int(fov_deg)}.", dir=root))
    try:
        _save_array(temporary / "negative_indices.i32.npy", indices)
        _save_array(temporary / "negative_scores.fp32.npy", scores)
        write_json(
            temporary / "metadata.json",
            {
                **metadata,
                "fov_deg": int(fov_deg),
                "shape": list(indices.shape),
                "indices_dtype": "int32",
                "scores_dtype": "float32",
            },
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def load_compact_hard_pool(
    epoch_root: str | Path,
    fov_deg: int,
    expected_metadata: dict[str, Any],
    manifest_size: int,
    keep_negative_locations: int,
) -> np.ndarray:
    """Validate and memory-map one pool in manifest-index order."""
    root = Path(epoch_root).expanduser().resolve() / f"fov_{int(fov_deg)}"
    metadata_path = root / "metadata.json"
    indices_path = root / "negative_indices.i32.npy"
    scores_path = root / "negative_scores.fp32.npy"
    if not all(path.is_file() for path in (metadata_path, indices_path, scores_path)):
        raise FileNotFoundError(f"Incomplete hard pool: {root}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    for key, expected in {**expected_metadata, "fov_deg": int(fov_deg)}.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Hard-pool metadata mismatch for FoV {fov_deg}: {key}")
    expected_shape = (int(manifest_size), int(keep_negative_locations))
    if tuple(map(int, metadata.get("shape", []))) != expected_shape:
        raise ValueError(f"Hard-pool metadata shape mismatch for FoV {fov_deg}")
    indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    if indices.dtype != np.int32 or scores.dtype != np.float32:
        raise ValueError(f"Hard-pool dtype mismatch for FoV {fov_deg}")
    if indices.shape != expected_shape or scores.shape != expected_shape:
        raise ValueError(f"Hard-pool array shape mismatch for FoV {fov_deg}")
    if np.any(indices < 0) or np.any(indices >= manifest_size):
        raise ValueError(f"Hard-pool indices are out of range for FoV {fov_deg}")
    positives = np.arange(manifest_size, dtype=np.int32)[:, None]
    if np.any(indices == positives):
        raise ValueError(f"Hard pool contains positive locations for FoV {fov_deg}")
    if np.any(np.diff(np.sort(np.asarray(indices), axis=1), axis=1) == 0):
        raise ValueError(f"Hard pool contains duplicate negatives for FoV {fov_deg}")
    if not np.isfinite(scores).all():
        raise ValueError(f"Hard-pool scores are non-finite for FoV {fov_deg}")
    return indices
