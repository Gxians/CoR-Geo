"""Auditable random-crop schedules for public evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cor_geo.utils.hashing import sha256_json


def _schedule_hash(
    schedule: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
) -> str:
    return sha256_json(
        {
            "protocol": "independent_random_roll_then_left_crop_v1",
            "split": split,
            "query_ids": schedule["query_id"].astype(str).tolist(),
            "roll_angles_deg": {
                str(fov): schedule[f"roll_angle_deg_{fov}"].astype(int).tolist()
                for fov in fovs
            },
        }
    )


def validate_random_crop_schedule(
    schedule: pd.DataFrame,
    manifest: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], pd.DataFrame, str]:
    """Validate, canonicalize, and hash a query--FoV heading schedule."""
    expected_columns = [
        "split",
        "query_id",
        *(f"roll_angle_deg_{fov}" for fov in fovs),
    ]
    if list(schedule.columns) != expected_columns:
        raise ValueError(
            f"Random crop schedule columns are {list(schedule.columns)}, "
            f"expected {expected_columns}"
        )
    if len(schedule) != len(manifest):
        raise ValueError("Random crop schedule length differs from the manifest")
    if schedule["split"].astype(str).tolist() != [split] * len(manifest):
        raise ValueError("Random crop schedule split differs from the request")
    query_ids = manifest["query_id"].astype(str).tolist()
    if schedule["query_id"].astype(str).tolist() != query_ids:
        raise ValueError("Random crop schedule query order differs from the manifest")

    canonical = pd.DataFrame({"split": split, "query_id": query_ids})
    angles: dict[int, np.ndarray] = {}
    for fov in fovs:
        column = f"roll_angle_deg_{fov}"
        numeric = pd.to_numeric(schedule[column], errors="raise").to_numpy()
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain finite integer degrees")
        if ((numeric < 0) | (numeric > 359)).any():
            raise ValueError(f"{column} must lie in [0, 359]")
        values = numeric.astype(np.int16)
        canonical[column] = values
        angles[int(fov)] = values
    return angles, canonical, _schedule_hash(canonical, split, fovs)


def draw_random_crop_schedule(
    manifest: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
    rng: np.random.Generator | None = None,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, str]:
    """Draw one independent integer heading for every query--FoV pair."""
    generator = np.random.default_rng() if rng is None else rng
    schedule = pd.DataFrame(
        {
            "split": split,
            "query_id": manifest["query_id"].astype(str),
            **{
                f"roll_angle_deg_{fov}": generator.integers(
                    0,
                    360,
                    size=len(manifest),
                    dtype=np.int16,
                )
                for fov in fovs
            },
        }
    )
    return validate_random_crop_schedule(schedule, manifest, split, fovs)


def load_random_crop_schedule(
    path: str | Path,
    manifest: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], pd.DataFrame, str]:
    """Load and validate a previously recorded Parquet crop schedule."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return validate_random_crop_schedule(
        pd.read_parquet(resolved, engine="pyarrow"),
        manifest,
        split,
        fovs,
    )
