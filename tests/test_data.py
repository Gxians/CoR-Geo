"""Portable configuration, cropping, manifest, and metric tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
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
from cor_geo.evaluate import (
    _checkpoint_path,
    _load_checkpoint_payload,
    _next_evaluation_root,
    _resolve_evaluation_crop_schedule,
    _resolve_run_dir,
    _resolved_config_from_checkpoint,
    draw_random_crop_schedule,
    retrieval_metrics,
    validate_random_crop_schedule,
)
from cor_geo.samplers import PlannedRankBatchSampler, build_epoch_plan
from cor_geo.train import (
    _append_validation_metrics,
    _macro_recall_at_1,
    _parse_device_ids,
    _update_best_validation,
    save_last_checkpoint,
)
from cor_geo.utils import (
    ConfigError,
    load_experiment_config,
    load_yaml,
    resolve_training_topology,
    sha256_json,
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
    assert config.evaluation["during_training"] == {
        "enabled": True,
        "interval_epochs": 8,
        "selection_metric": "macro_r1",
    }
    assert "checkpoint" not in config.train


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


def test_evaluation_defaults_to_the_standard_run_and_best_checkpoint(tmp_path: Path) -> None:
    paths = {"project_root": str(tmp_path), "output_root": "outputs"}
    run_dir = _resolve_run_dir(None, "cvact", paths)
    assert run_dir == tmp_path / "outputs" / "cvact" / "cor_geo_cvact"
    assert _checkpoint_path(run_dir, "best") == run_dir / "checkpoints" / "best.ckpt"
    assert _checkpoint_path(run_dir, "last") == run_dir / "checkpoints" / "last.ckpt"

    explicit = _resolve_run_dir("custom/run", "cvact", paths)
    assert explicit == Path("custom/run").resolve()


def test_evaluation_uses_checkpoint_config_without_run_config(tmp_path: Path) -> None:
    resolved_config = {
        "dataset": {"dataset": "cvact"},
        "model": {"name": "cor_geo"},
    }
    payload = {
        "resolved_config": resolved_config,
        "resolved_config_sha256": sha256_json(resolved_config),
    }
    checkpoint = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload, checkpoint)

    loaded = _load_checkpoint_payload(checkpoint)
    run_config = tmp_path / "config.yaml"
    assert _resolved_config_from_checkpoint(loaded, run_config) == resolved_config

    run_config.write_text(yaml.safe_dump(resolved_config), encoding="utf-8")
    assert _resolved_config_from_checkpoint(loaded, run_config) == resolved_config

    run_config.write_text(yaml.safe_dump({"dataset": {"dataset": "cvusa"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the checkpoint embedded config"):
        _resolved_config_from_checkpoint(loaded, run_config)


def test_evaluation_directories_use_compact_monotonic_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "cvact" / "cor_geo_cvact"
    assert _next_evaluation_root(run_dir) == run_dir / "evaluations" / "eval_001"
    (run_dir / "evaluations" / "eval_001").mkdir(parents=True)
    (run_dir / "evaluations" / "eval_003").mkdir()
    (run_dir / "evaluations" / "notes").mkdir()
    assert _next_evaluation_root(run_dir) == run_dir / "evaluations" / "eval_004"


def test_training_checkpoint_directory_keeps_only_last_and_best(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    last = save_last_checkpoint({"completed_epoch": 1}, checkpoint_dir)
    assert last == checkpoint_dir / "last.ckpt"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["last.ckpt"]

    save_last_checkpoint({"completed_epoch": 2}, checkpoint_dir)
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["last.ckpt"]


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


def test_random_validation_schedule_can_be_replayed_for_all_fovs() -> None:
    manifest = pd.DataFrame({"query_id": ["q0", "q1", "q2"]})
    fovs = (360, 180, 90, 70)
    angles, schedule, schedule_hash = draw_random_crop_schedule(
        manifest,
        "val",
        fovs,
        rng=np.random.default_rng(7),
    )
    replayed_angles, replayed_schedule, replayed_hash = validate_random_crop_schedule(
        schedule,
        manifest,
        "val",
        fovs,
    )
    assert replayed_hash == schedule_hash
    assert replayed_schedule.equals(schedule)
    assert all((replayed_angles[fov] == angles[fov]).all() for fov in fovs)


def test_standalone_evaluation_draws_fresh_schedules_unless_replay_is_requested(tmp_path: Path) -> None:
    manifest = pd.DataFrame({"query_id": [f"q{index}" for index in range(32)]})
    fovs = (360, 180, 90, 70)
    _, first, first_hash, first_source = _resolve_evaluation_crop_schedule(
        manifest,
        "val",
        fovs,
        rng=np.random.default_rng(7),
    )
    _, second, second_hash, second_source = _resolve_evaluation_crop_schedule(
        manifest,
        "val",
        fovs,
        rng=np.random.default_rng(8),
    )
    assert first_source == second_source == "new_random_draw"
    assert first_hash != second_hash
    assert not first.equals(second)

    replay_path = tmp_path / "crop_schedule.parquet"
    write_parquet_atomic(first, replay_path)
    _, replayed, replayed_hash, replayed_source = _resolve_evaluation_crop_schedule(
        manifest,
        "val",
        fovs,
        crop_schedule=replay_path,
    )
    assert replayed_source == "user_supplied"
    assert replayed_hash == first_hash
    assert replayed.equals(first)


def test_best_checkpoint_uses_macro_r1_and_keeps_the_earliest_tie(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    first = run_root / "checkpoints" / "last.ckpt"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"epoch-8")
    summary = {
        "selected_epoch": 8,
        "fovs": {str(fov): {"R@1": value} for fov, value in zip((360, 180, 90, 70), (0.8, 0.7, 0.6, 0.5))},
    }
    summary["selection"] = {
        "metric": "macro_r1",
        "macro_r1": _macro_recall_at_1(summary, (360, 180, 90, 70)),
        "fovs": [360, 180, 90, 70],
    }
    assert _update_best_validation(run_root, first, summary)
    assert (run_root / "checkpoints" / "best.ckpt").read_bytes() == b"epoch-8"
    assert (run_root / "best_metrics.json").is_file()
    assert sorted(path.name for path in first.parent.iterdir()) == ["best.ckpt", "last.ckpt"]

    first.write_bytes(b"epoch-16")
    tied_summary = {**summary, "selected_epoch": 16}
    assert not _update_best_validation(run_root, first, tied_summary)
    assert (run_root / "checkpoints" / "best.ckpt").read_bytes() == b"epoch-8"


def test_validation_metrics_are_appended_to_the_training_log(tmp_path: Path) -> None:
    metrics_path = tmp_path / "run" / "train_metrics.jsonl"
    summaries = []
    for epoch, value in ((8, 0.5), (16, 0.6)):
        summaries.append(
            {
                "dataset": "cvact",
                "selected_epoch": epoch,
                "checkpoint_sha256": f"checkpoint-{epoch}",
                "split": "val",
                "fovs": {str(fov): {"R@1": value} for fov in (360, 180, 90, 70)},
                "selection": {"metric": "macro_r1", "macro_r1": value, "fovs": [360, 180, 90, 70]},
            }
        )

    _append_validation_metrics(metrics_path, summaries[0], new_best=True, trigger="scheduled")
    _append_validation_metrics(metrics_path, summaries[1], new_best=False, trigger="scheduled")

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [record["epoch"] for record in records] == [8, 16]
    assert all(record["record_type"] == "validation" for record in records)
    assert [record["new_best"] for record in records] == [True, False]
    assert records[1]["fovs"]["70"]["R@1"] == pytest.approx(0.6)


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
