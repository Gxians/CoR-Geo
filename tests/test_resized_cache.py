from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cor_geo.datasets.cross_view import CrossViewDataset, SampleRequest
from cor_geo.datasets.resized_cache import (
    CACHE_FORMAT,
    ResizedRGBMemmapCache,
    build_resized_cache,
    cache_request,
)


def _manifest(tmp_path, split: str = "train", count: int = 3) -> pd.DataFrame:
    rows = []
    for index in range(count):
        ground_path = tmp_path / f"ground_{index}.jpg"
        satellite_path = tmp_path / f"satellite_{index}.jpg"
        generator = np.random.default_rng(index)
        ground = generator.integers(0, 256, (56, 1008, 3), dtype=np.uint8)
        satellite = generator.integers(0, 256, (56, 56, 3), dtype=np.uint8)
        Image.fromarray(ground).save(ground_path, quality=91)
        Image.fromarray(satellite).save(satellite_path, quality=91)
        rows.append(
            {
                "split": split,
                "query_id": f"query_{index}",
                "query_path": str(ground_path),
                "satellite_id": f"query_{index}",
                "satellite_path": str(satellite_path),
            }
        )
    return pd.DataFrame(rows)


def test_resized_cache_preserves_training_tensors_and_row_mapping(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    cache_root = tmp_path / "cache"
    build_resized_cache(
        manifest,
        cache_root,
        ground_height=28,
        panorama_width=504,
        satellite_size=28,
        workers=0,
        batch_size=2,
        progress_every=1,
    )
    reversed_manifest = manifest.iloc[::-1].reset_index(drop=True)
    original = CrossViewDataset(
        reversed_manifest,
        global_seed=42,
        ground_height=28,
        panorama_width=504,
        satellite_size=28,
    )
    cached = CrossViewDataset(
        reversed_manifest,
        global_seed=42,
        ground_height=28,
        panorama_width=504,
        satellite_size=28,
        resized_cache_root=cache_root,
        require_resized_cache=True,
    )
    for index, fov in enumerate((360, 180, 90)):
        request = SampleRequest(index, epoch=3, fov_deg=fov)
        original_row = original[request]
        cached_row = cached[request]
        assert torch.equal(cached_row["ground"], original_row["ground"])
        assert torch.equal(cached_row["satellite"], original_row["satellite"])
    ResizedRGBMemmapCache(
        cache_root,
        reversed_manifest,
        ground_height=28,
        panorama_width=504,
        satellite_size=28,
    )


def test_required_cache_and_config_resolution_are_strict(tmp_path) -> None:
    manifest = _manifest(tmp_path, count=1)
    with pytest.raises(FileNotFoundError):
        CrossViewDataset(
            manifest,
            global_seed=42,
            ground_height=28,
            panorama_width=504,
            satellite_size=28,
            resized_cache_root=tmp_path / "missing",
            require_resized_cache=True,
        )
    root, required = cache_request(
        {
            "dataset_cache": {
                "enabled": True,
                "format": CACHE_FORMAT,
                "root": str(tmp_path / "cache"),
                "required_splits": ["train", "val"],
            }
        },
        "train",
    )
    assert root == (tmp_path / "cache").resolve()
    assert required is True
