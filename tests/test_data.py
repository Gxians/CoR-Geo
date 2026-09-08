"""Portable configuration, cropping, manifest, and metric tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from cor_geo.datasets import (
    MANIFEST_COLUMNS,
    build_cvusa_manifests,
    orientation_to_center_px,
    project_relative_path,
    random_crop_spec,
    read_manifest,
    write_parquet_atomic,
)
from cor_geo.evaluate import retrieval_metrics
from cor_geo.samplers import PlannedRankBatchSampler, build_epoch_plan
from cor_geo.train import _parse_device_ids
from cor_geo.utils import (
    ConfigError,
    load_experiment_config,
    load_yaml,
    resolve_training_topology,
    validate_experiment_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["cvact", "cvusa"])
def test_public_configuration_is_valid_and_portable(dataset: str) -> None:
    config = load_experiment_config(ROOT / "configs" / f"{dataset}.yaml", ROOT / "configs" / "default.yaml")
    assert config.dataset["dataset"] == dataset
    assert config.model["name"] == "cor_geo"
    assert config.train["dataset_cache"]["root"].startswith(".cache/cor_geo/")
    paths = load_yaml(ROOT / "configs" / "default.yaml")["paths"]
    assert paths["project_root"] == "."
    assert all(not Path(value).is_absolute() for value in paths["datasets"].values())


def test_validation_accepts_structurally_valid_experiment_variants() -> None:
    config = load_experiment_config(ROOT / "configs" / "cvact.yaml", ROOT / "configs" / "default.yaml")
    config.model["architecture"]["content_order_encoder"]["joint_order_weight"] = 0.25
    config.model["loss"]["label_smoothing"] = 0.05
    hard = config.train["hard_mining"]
    hard["coarse_candidate_locations"] = 256
    hard["keep_negative_locations"] = 32
    hard["sampling_rank_range"] = [1, 32]
    validate_experiment_config(config)


def test_one_and_two_gpu_topologies_preserve_the_global_protocol() -> None:
    train = load_yaml(ROOT / "configs" / "default.yaml")["train"]
    one = resolve_training_topology(train, 1)
    two = resolve_training_topology(train, 2)
    assert (one.per_gpu_batch_size, one.global_batch_size, one.samples_per_fov_per_rank) == (64, 64, 16)
    assert (two.per_gpu_batch_size, two.global_batch_size, two.samples_per_fov_per_rank) == (32, 64, 8)
    with pytest.raises(ConfigError, match="one or two GPUs"):
        resolve_training_topology(train, 4)


def test_public_device_option_defaults_to_one_gpu_and_accepts_two() -> None:
    assert _parse_device_ids(None) == (0,)
    assert _parse_device_ids("0") == (0,)
    assert _parse_device_ids("0,1") == (0, 1)
    assert _parse_device_ids(" 2, 3 ") == (2, 3)
    for invalid in ("", "0,", "-1", "0,0", "0,1,2", "gpu0"):
        with pytest.raises(ValueError):
            _parse_device_ids(invalid)


def test_one_gpu_batch_matches_the_two_contiguous_rank_shards() -> None:
    train = deepcopy(load_yaml(ROOT / "configs" / "default.yaml")["train"])
    train["stages"][0]["steps_per_epoch"] = 2
    manifest = pd.DataFrame({"dataset": ["cvact"] * 128})
    plan = build_epoch_plan(manifest, epoch=1, global_seed=42, train_config=train)

    one_topology = resolve_training_topology(train, 1)
    one_sampler = PlannedRankBatchSampler(0, 1, one_topology.per_gpu_batch_size)
    one_sampler.set_plan(plan, epoch=1)
    one_batches = list(one_sampler)

    two_topology = resolve_training_topology(train, 2)
    rank_samplers = [PlannedRankBatchSampler(rank, 2, two_topology.per_gpu_batch_size) for rank in (0, 1)]
    for sampler in rank_samplers:
        sampler.set_plan(plan, epoch=1)
    rank_batches = [list(sampler) for sampler in rank_samplers]

    for batch_index, one_batch in enumerate(one_batches):
        reconstructed = rank_batches[0][batch_index] + rank_batches[1][batch_index]
        assert [(item.index, item.fov_deg) for item in one_batch] == [
            (item.index, item.fov_deg) for item in reconstructed
        ]


def test_cvusa_manifest_builder_hashes_a_resolved_paths_mapping(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    dataset_root = project_root / "data" / "CVUSA"
    for relative in ("splits", "streetview/panos", "streetview/annotations", "bingmap/19"):
        (dataset_root / relative).mkdir(parents=True, exist_ok=True)

    rows = {
        "train": ("0000001", "train-19zl.csv"),
        "val": ("0000002", "val-19zl.csv"),
    }
    for _, (identifier, csv_name) in rows.items():
        satellite = dataset_root / "bingmap" / "19" / f"{identifier}.jpg"
        query = dataset_root / "streetview" / "panos" / f"{identifier}.jpg"
        annotation = dataset_root / "streetview" / "annotations" / f"{identifier}.png"
        Image.new("RGB", (3, 3)).save(satellite)
        Image.new("RGB", (4, 2)).save(query)
        Image.new("RGB", (4, 2)).save(annotation)
        (dataset_root / "splits" / csv_name).write_text(
            f"bingmap/19/{identifier}.jpg,streetview/panos/{identifier}.jpg,"
            f"streetview/annotations/{identifier}.png\n",
            encoding="utf-8",
        )

    dataset_config = {
        "dataset": "cvusa",
        "root_key": "cvusa",
        "protocol_version": "test",
        "manifest_splits": ["train", "val"],
        "splits": {
            "train": {"csv_file": "splits/train-19zl.csv", "expected_count": 1},
            "val": {"csv_file": "splits/val-19zl.csv", "expected_count": 1},
        },
        "expected_source_image_sizes": {"query": [4, 2], "satellite": [3, 3], "annotation": [4, 2]},
        "satellite_inventory_root": "bingmap/19",
        "strict_satellite_file_equality": True,
    }
    config_path = project_root / "configs" / "cvusa.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")
    paths = {"project_root": str(project_root), "datasets": {"cvusa": "data/CVUSA"}}

    outputs = build_cvusa_manifests(config_path, paths)
    with outputs["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert len(metadata["paths_config_sha256"]) == 64
    assert Path(outputs["train"]).is_file()
    assert Path(outputs["val"]).is_file()


def test_random_crop_keeps_a_common_left_boundary_for_all_fovs() -> None:
    roll_angle = 137
    panorama_width = 756
    expected_start = (-(roll_angle * panorama_width // 360)) % panorama_width
    crops = {fov: random_crop_spec(roll_angle, fov, panorama_width) for fov in (360, 180, 90, 70, 270, 120)}
    assert [crops[fov].width for fov in (360, 180, 90, 70)] == [756, 378, 189, 147]
    assert all(crop.interval[0] % panorama_width == expected_start for crop in crops.values())
    assert all(
        orientation_to_center_px(crop.orientation_u32, panorama_width) == crop.center_px for crop in crops.values()
    )


def test_manifest_paths_remain_repository_relative(tmp_path: Path) -> None:
    image = ROOT / "data" / "CVACT" / "example.jpg"
    assert project_relative_path(image, ROOT) == "data/CVACT/example.jpg"

    manifest_path = tmp_path / "data_manifests" / "cvact" / "val.parquet"
    row = pd.DataFrame(
        [["cvact", "val", "query", "data/CVACT/query.jpg", "query", "data/CVACT/satellite.jpg", "valSet", 1]],
        columns=MANIFEST_COLUMNS,
    )
    write_parquet_atomic(row, manifest_path)
    loaded = read_manifest(manifest_path)
    assert loaded.loc[0, "query_path"] == str(tmp_path / "data" / "CVACT" / "query.jpg")

    with pytest.raises(ValueError, match="inside the repository"):
        project_relative_path(tmp_path / "outside.jpg", ROOT)


def test_recall_at_one_percent_uses_the_configured_floor_boundary() -> None:
    metrics = retrieval_metrics([88, 89, 90], 8884, (1, 5, 10), "floor")
    assert metrics["R@1%_threshold_floor"] == 88
    assert metrics["R@1%_threshold_ceil"] == 89
    assert metrics["R@1%"] == pytest.approx(1 / 3)
