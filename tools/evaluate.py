#!/usr/bin/env python
"""Distributed exact evaluation with random or replayed FoV crops."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as distributed
from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_yaml  # noqa: E402
from cor_geo.datasets.cross_view import CrossViewDataset  # noqa: E402
from cor_geo.datasets.manifests import read_manifest, write_parquet_atomic  # noqa: E402
from cor_geo.datasets.resized_cache import cache_request  # noqa: E402
from cor_geo.engine.checkpoint import load_checkpoint  # noqa: E402
from cor_geo.engine.fov_generalization import (  # noqa: E402
    register_evaluation_resamplers,
    resolve_evaluation_fov_geometry,
)
from cor_geo.engine.retrieval_evaluator import (  # noqa: E402
    EncodedSatelliteViews,
    encode_ground_views,
    encode_satellite_views,
    exact_evaluate_shard,
    prepare_satellite_views,
)
from cor_geo.evaluation_protocol import (  # noqa: E402
    draw_random_crop_schedule,
    load_random_crop_schedule,
)
from cor_geo.metrics.retrieval import retrieval_metrics  # noqa: E402
from cor_geo.models.cor_geo_model import CoRGeoModel  # noqa: E402
from cor_geo.reproducibility import configure_deterministic_algorithms  # noqa: E402
from cor_geo.utils.hashing import sha256_file, sha256_json  # noqa: E402
from cor_geo.utils.io import write_json  # noqa: E402

FOVS = (360, 180, 90, 70)


def _initialize(runtime: dict[str, Any]) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    configured = runtime["distributed"].get("world_size", "auto")
    if rank < 0 or local_rank < 0 or world_size < 1:
        raise RuntimeError("Launch evaluation with torchrun")
    if configured != "auto" and world_size != int(configured):
        raise RuntimeError(
            f"Runtime expects world_size={configured}, received {world_size}"
        )
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(
        backend=str(runtime["distributed"]["backend"]),
        timeout=timedelta(minutes=int(runtime["distributed"]["timeout_minutes"])),
    )
    return rank, local_rank, world_size


def _broadcast_error(error: str | None, rank: int, context: str) -> None:
    values = [error if rank == 0 else None]
    distributed.broadcast_object_list(values, src=0)
    if values[0]:
        raise RuntimeError(f"{context}: {values[0]}")


def _bounds(length: int, rank: int, world_size: int) -> tuple[int, int]:
    return length * rank // world_size, length * (rank + 1) // world_size


def _checkpoint_path(run_dir: Path, name: str) -> Path:
    filename = name if name.endswith(".ckpt") else f"{name}.ckpt"
    return run_dir / "checkpoints" / filename


def _save_encoded(root: Path, encoded: EncodedSatelliteViews) -> None:
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / "direction.npy", encoded.direction.astype(np.float32), allow_pickle=False)
    np.save(root / "ids.npy", np.asarray(encoded.ids, dtype=str), allow_pickle=False)


def _merge_satellite_shards(
    staging_root: Path,
    bank_root: Path,
    world_size: int,
    location_count: int,
    angular_bins: int,
    direction_dim: int,
) -> list[str]:
    bank_root.mkdir(parents=True, exist_ok=False)
    direction = np.lib.format.open_memmap(
        bank_root / "direction.fp32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(location_count, angular_bins, direction_dim),
    )
    ids: list[str] = []
    cursor = 0
    for rank in range(world_size):
        root = staging_root / f"rank_{rank:02d}" / "satellite"
        shard_direction = np.load(root / "direction.npy", mmap_mode="r", allow_pickle=False)
        shard_ids = np.load(root / "ids.npy", allow_pickle=False).astype(str).tolist()
        rows = len(shard_direction)
        direction[cursor : cursor + rows] = shard_direction
        ids.extend(shard_ids)
        cursor += rows
    if cursor != location_count:
        raise RuntimeError("Satellite shard merge did not cover the manifest")
    direction.flush()
    np.save(bank_root / "satellite_ids.npy", np.asarray(ids, dtype=str), allow_pickle=False)
    return ids


def _validate_satellite_bank(
    root: Path,
    expected_ids: list[str],
    checkpoint_hash: str,
    manifest_hash: str,
    model_config_hash: str,
    angular_bins: int,
    direction_dim: int,
) -> None:
    """Validate a completed or recoverable satellite bank before reuse."""
    metadata_path = root / "metadata.json"
    required = (
        root / "direction.fp32.npy",
        root / "satellite_ids.npy",
        metadata_path,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Incomplete satellite bank: {root}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "model_config_sha256": model_config_hash,
        "direction_shape": [
            len(expected_ids),
            int(angular_bins),
            int(direction_dim),
        ],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Satellite bank metadata mismatch: {key}")
    direction = np.load(root / "direction.fp32.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(root / "satellite_ids.npy", allow_pickle=False).astype(str).tolist()
    if direction.shape != (
        len(expected_ids),
        int(angular_bins),
        int(direction_dim),
    ) or direction.dtype != np.float32:
        raise ValueError("Satellite direction bank has an invalid shape or dtype")
    if ids != expected_ids:
        raise ValueError("Satellite bank IDs differ from the evaluation manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--fovs", type=int, nargs="+", required=True)
    parser.add_argument(
        "--crop-schedule",
        help=(
            "Optional recorded random_crop_schedule.parquet to replay exactly. "
            "If omitted, a new independent random schedule is drawn."
        ),
    )
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument(
        "--evaluation-dataset-config",
        help=(
            "Optional target dataset YAML. Defaults to the dataset stored in "
            "the checkpoint run."
        ),
    )
    parser.add_argument(
        "--evaluation-train-config",
        help=(
            "Optional target training YAML used only to resolve its resized RGB "
            "evaluation cache. Required for cross-dataset evaluation."
        ),
    )
    parser.add_argument(
        "--allow-cross-dataset-weights",
        action="store_true",
        help=(
            "Explicitly permit strict model-weight loading from the checkpoint "
            "while evaluating a different dataset manifest."
        ),
    )
    parser.add_argument(
        "--allow-unseen-fovs",
        action="store_true",
        help=(
            "Explicitly permit parameter-free FoV geometry not used for model "
            "selection (for example 270 and 120 degrees)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Optional explicit output directory; defaults to a hash-named draw directory",
    )
    parser.add_argument(
        "--satellite-bank-dir",
        help=(
            "Optional shared satellite-bank parent, reusable only when checkpoint, "
            "manifest, and model hashes match."
        ),
    )
    args = parser.parse_args()
    active_fovs = tuple(map(int, args.fovs))
    if active_fovs != FOVS and not args.allow_unseen_fovs:
        raise ValueError(f"FoVs must be supplied in exact order {FOVS}")
    split = str(args.split)
    runtime = load_yaml(args.runtime_config)
    rank, local_rank, world_size = _initialize(runtime)
    device = torch.device("cuda", local_rank)
    succeeded = False
    try:
        run_dir = Path(args.run_dir).expanduser().resolve()
        resolved = load_yaml(run_dir / "config_resolved.yaml")
        checkpoint_dataset_name = str(resolved["dataset"]["dataset"])
        evaluation_dataset = (
            load_yaml(args.evaluation_dataset_config)
            if args.evaluation_dataset_config
            else resolved["dataset"]
        )
        dataset_name = str(evaluation_dataset["dataset"])
        cross_dataset = dataset_name != checkpoint_dataset_name
        if cross_dataset and not args.allow_cross_dataset_weights:
            raise ValueError(
                "Cross-dataset evaluation requires --allow-cross-dataset-weights"
            )
        if not cross_dataset and args.allow_cross_dataset_weights:
            raise ValueError(
                "--allow-cross-dataset-weights was supplied for the checkpoint dataset"
            )
        if cross_dataset and not args.evaluation_train_config:
            raise ValueError(
                "Cross-dataset evaluation requires --evaluation-train-config"
            )
        manifest_splits = tuple(map(str, evaluation_dataset["manifest_splits"]))
        if split not in manifest_splits:
            raise ValueError(
                f"Split {split} is unavailable for {dataset_name}: {manifest_splits}"
            )
        model_config = resolved["model"]
        checkpoint_train_config = resolved["train"]
        evaluation_train_config = (
            load_yaml(args.evaluation_train_config)
            if args.evaluation_train_config
            else checkpoint_train_config
        )
        eval_config = resolved["evaluation"]
        paths = load_yaml(args.paths)
        configure_deterministic_algorithms()
        checkpoint = _checkpoint_path(run_dir, args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        selected_epoch = int(checkpoint.stem.removeprefix("epoch_"))
        if selected_epoch not in set(
            map(int, checkpoint_train_config["checkpoint"]["retained_epochs"])
        ):
            raise ValueError("Evaluation accepts retained checkpoint epochs only")
        checkpoint_hash = sha256_file(checkpoint)
        project_root = Path(paths["project_root"])
        manifest_root = project_root / "data_manifests" / dataset_name
        manifest_path = manifest_root / f"{split}.parquet"
        manifest = read_manifest(manifest_path).reset_index(drop=True)
        manifest_hash = sha256_file(manifest_path)
        schedule_values: list[
            tuple[dict[int, np.ndarray], pd.DataFrame, str] | None
        ] = [None]
        if rank == 0:
            if args.crop_schedule:
                schedule_values[0] = load_random_crop_schedule(
                    args.crop_schedule,
                    manifest,
                    split,
                    active_fovs,
                )
            else:
                schedule_values[0] = draw_random_crop_schedule(
                    manifest,
                    split,
                    active_fovs,
                )
        distributed.broadcast_object_list(schedule_values, src=0)
        if schedule_values[0] is None:
            raise RuntimeError("Random crop schedule broadcast failed")
        random_roll_angles_deg, random_crop_schedule, crop_hash = schedule_values[0]
        model_config_hash = sha256_json(model_config)
        fov_geometries = resolve_evaluation_fov_geometry(active_fovs, model_config)

        model = CoRGeoModel(
            model_config,
            dinov2_root=paths["dinov2_root"],
            checkpoint_path=paths["checkpoints"]["dinov2_vitb14"],
        ).to(device)
        payload = load_checkpoint(
            checkpoint,
            model,
            restore_rng=False,
            expected_resolved_config=resolved,
        )
        if int(payload["completed_epoch"]) != selected_epoch:
            raise ValueError("Checkpoint epoch metadata differs from its filename")
        expected = {"model": (payload["model_config_sha256"], model_config_hash)}
        if not cross_dataset:
            expected["manifest"] = (
                payload["manifest_hashes"][split],
                manifest_hash,
            )
        mismatches = [name for name, values in expected.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(f"Evaluation provenance mismatch: {mismatches}")
        register_evaluation_resamplers(model, fov_geometries, device)
        model.eval()
        model.float()
        score_config = dict(model_config["score"])
        bank_angular_bins = int(model.angular_bins)
        bank_direction_dim = int(model.direction_dim)
        torch.backends.cuda.matmul.allow_tf32 = bool(eval_config["exact"]["allow_tf32"])
        cache_root, require_cache = cache_request(evaluation_train_config, split)
        ground_widths = {
            int(fov): geometry.aligned_input_width
            for fov, geometry in fov_geometries.items()
        }
        dataset = CrossViewDataset(
            manifest,
            global_seed=int(checkpoint_train_config["seed"]),
            ground_height=int(model_config["input"]["ground_height"]),
            panorama_width=int(model_config["input"]["panorama_width"]),
            satellite_size=int(model_config["input"]["satellite_size"][0]),
            ground_widths=ground_widths,
            resized_cache_root=cache_root,
            require_resized_cache=require_cache,
            dataset_name=dataset_name,
        )
        evaluation_root = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else (
                run_dir
                / "evaluations"
                / f"{split}_random"
                / f"epoch_{selected_epoch:03d}"
                / f"draw_{crop_hash[:12]}"
            )
        )
        summary_path = evaluation_root / "summary.json"
        staging_root = evaluation_root / str(runtime["storage"]["staging_directory"])
        bank_parent = (
            Path(args.satellite_bank_dir).expanduser().resolve()
            if args.satellite_bank_dir
            else run_dir / "evaluations" / "shared_satellite_bank"
        )
        bank_root = bank_parent / split / checkpoint_hash
        setup_error: str | None = None
        reuse_satellite_bank: bool | None = None
        if rank == 0:
            try:
                if summary_path.exists():
                    raise FileExistsError("Evaluation summary already exists")
                expected_satellite_ids = manifest["satellite_id"].astype(str).tolist()
                validation_arguments = (
                    expected_satellite_ids,
                    checkpoint_hash,
                    manifest_hash,
                    model_config_hash,
                    bank_angular_bins,
                    bank_direction_dim,
                )
                reuse_satellite_bank = False
                if bank_root.exists():
                    try:
                        _validate_satellite_bank(bank_root, *validation_arguments)
                        reuse_satellite_bank = True
                    except (FileNotFoundError, ValueError, OSError):
                        shutil.rmtree(bank_root)
                recoverable_bank = staging_root / "merged_bank"
                if not reuse_satellite_bank and recoverable_bank.exists():
                    try:
                        _validate_satellite_bank(recoverable_bank, *validation_arguments)
                        bank_root.parent.mkdir(parents=True, exist_ok=True)
                        recoverable_bank.replace(bank_root)
                        reuse_satellite_bank = True
                    except (FileNotFoundError, ValueError, OSError):
                        pass
                if staging_root.exists():
                    shutil.rmtree(staging_root)
                staging_root.mkdir(parents=True, exist_ok=False)
                schedule_path = evaluation_root / "random_crop_schedule.parquet"
                write_parquet_atomic(random_crop_schedule, schedule_path)
            except Exception as error:
                setup_error = f"{type(error).__name__}: {error}"
        _broadcast_error(setup_error, rank, "Evaluation setup")
        reuse_values = [reuse_satellite_bank if rank == 0 else None]
        distributed.broadcast_object_list(reuse_values, src=0)
        reuse_satellite_bank = bool(reuse_values[0])
        distributed.barrier()

        start, stop = _bounds(len(manifest), rank, world_size)
        indices = np.arange(start, stop, dtype=np.int64)
        export = runtime["descriptor_export"]
        rank_root = staging_root / f"rank_{rank:02d}"
        rank_root.mkdir(parents=True, exist_ok=False)
        if not reuse_satellite_bank:
            satellite_shard = encode_satellite_views(
                model,
                dataset,
                indices,
                device,
                int(export["batch_size_per_gpu"]),
                int(export["workers_per_rank"]),
                progress_label=f"rank{rank}/{split}",
            )
            _save_encoded(rank_root / "satellite", satellite_shard)
            distributed.barrier()

            publish_error: str | None = None
            if rank == 0:
                try:
                    bank_partial = staging_root / "merged_bank"
                    satellite_ids = _merge_satellite_shards(
                        staging_root,
                        bank_partial,
                        world_size,
                        len(manifest),
                        bank_angular_bins,
                        bank_direction_dim,
                    )
                    expected_ids = manifest["satellite_id"].astype(str).tolist()
                    if satellite_ids != expected_ids:
                        raise ValueError(
                            "Merged satellite IDs differ from the evaluation manifest"
                        )
                    write_json(
                        bank_partial / "metadata.json",
                        {
                            "checkpoint_sha256": checkpoint_hash,
                            "manifest_sha256": manifest_hash,
                            "model_config_sha256": model_config_hash,
                            "location_score": model_config["score"]["name"],
                            "shift_reduction": str(
                                model_config["score"]["shift_reduction"]
                            ),
                            "direction_shape": [
                                len(manifest),
                                bank_angular_bins,
                                bank_direction_dim,
                            ],
                            "shared_across_fovs": True,
                            "distributed_satellite_shards": world_size,
                        },
                    )
                    _validate_satellite_bank(
                        bank_partial,
                        expected_ids,
                        checkpoint_hash,
                        manifest_hash,
                        model_config_hash,
                        bank_angular_bins,
                        bank_direction_dim,
                    )
                    bank_root.parent.mkdir(parents=True, exist_ok=True)
                    bank_partial.replace(bank_root)
                except Exception as error:
                    publish_error = f"{type(error).__name__}: {error}"
            _broadcast_error(publish_error, rank, "Satellite bank publication")
            distributed.barrier()

        satellite = EncodedSatelliteViews(
            np.load(bank_root / "direction.fp32.npy", mmap_mode="r", allow_pickle=False),
            np.load(bank_root / "satellite_ids.npy", allow_pickle=False).astype(str).tolist(),
        )
        queries = encode_ground_views(
            model,
            dataset,
            indices,
            active_fovs,
            device,
            int(export["batch_size_per_gpu"]),
            int(export["workers_per_rank"]),
            mode="random_eval",
            random_roll_angles_deg={
                fov: random_roll_angles_deg[fov][start:stop]
                for fov in active_fovs
            },
            progress_label=f"rank{rank}/{split}",
        )
        expected_ids = manifest.iloc[start:stop]["query_id"].astype(str).tolist()
        if any(queries[fov].ids != expected_ids for fov in active_fovs):
            raise ValueError("Query shard order differs from evaluation manifest")
        del model
        torch.cuda.empty_cache()
        prepared_satellite = prepare_satellite_views(satellite, device)
        exact = runtime["exact_score"]
        for fov in active_fovs:
            predictions, _ = exact_evaluate_shard(
                queries[fov],
                fov,
                satellite,
                device,
                query_chunk_size=int(exact["query_chunk_size"]),
                location_chunk_size=int(exact["location_chunk_size"]),
                score_config=score_config,
                checkpoint_sha256=checkpoint_hash,
                manifest_sha256=manifest_hash,
                crop_schedule_sha256=crop_hash,
                split=split,
                dataset_name=dataset_name,
                progress_label=f"rank{rank}/fov{fov}",
                prepared_satellite=prepared_satellite,
            )
            write_parquet_atomic(
                predictions,
                rank_root / f"predictions_fov_{fov}.parquet",
            )
        distributed.barrier()

        summary: dict[str, Any] | None = None
        merge_error: str | None = None
        if rank == 0:
            try:
                summary = {
                    "output_directory": str(evaluation_root),
                    "dataset": dataset_name,
                    "checkpoint_dataset": checkpoint_dataset_name,
                    "cross_dataset_weights_only": cross_dataset,
                    "selected_epoch": selected_epoch,
                    "checkpoint_sha256": checkpoint_hash,
                    "split": split,
                    "crop_mode": "random",
                    "manifest_sha256": manifest_hash,
                    "random_crop_schedule_sha256": crop_hash,
                    "location_score": model_config["score"]["name"],
                    "shift_reduction": model_config["score"]["shift_reduction"],
                    "direction_representation": "content_attention_plus_first_cosine_order",
                    "evaluation_fov_geometry": {
                        str(fov): fov_geometries[fov].to_dict()
                        for fov in active_fovs
                    },
                    "execution": {
                        "engine": "distributed_exact_cyclic_v1",
                        "world_size": world_size,
                        "used_for_model_selection": False,
                        "single_satellite_bank_for_all_fovs": True,
                    },
                    "fovs": {},
                }
                summary["random_crop_protocol"] = {
                    "angle_sampler": "independent_uniform_integer_0_359",
                    "operation": "right_roll_then_left_fov_crop",
                    "independent_random_angle_per_query_fov": True,
                    "schedule_source": (
                        "replayed" if args.crop_schedule else "new_random_draw"
                    ),
                    "schedule": str(
                        evaluation_root / "random_crop_schedule.parquet"
                    ),
                    "schedule_sha256": crop_hash,
                }
                prediction_root = evaluation_root / "predictions"
                for fov in active_fovs:
                    parts = [
                        pd.read_parquet(
                            staging_root
                            / f"rank_{shard_rank:02d}"
                            / f"predictions_fov_{fov}.parquet",
                            engine="pyarrow",
                        )
                        for shard_rank in range(world_size)
                    ]
                    predictions = pd.concat(parts, ignore_index=True)
                    if predictions["query_id"].astype(str).tolist() != manifest[
                        "query_id"
                    ].astype(str).tolist():
                        raise ValueError(
                            "Merged predictions differ from the evaluation manifest"
                        )
                    metrics = retrieval_metrics(
                        predictions["predicted_rank_1based"].to_numpy(np.int64),
                        len(manifest),
                        recall_ks=(1, 5, 10),
                        r1_percent_rounding=str(
                            eval_config["metrics"]["r1_percent_rounding"]
                        ),
                    )
                    write_parquet_atomic(
                        predictions,
                        prediction_root / f"{split}_fov{fov}.parquet",
                    )
                    summary["fovs"][str(fov)] = metrics
                write_json(summary_path, summary)
                shutil.rmtree(staging_root)
            except Exception as error:
                merge_error = f"{type(error).__name__}: {error}"
        _broadcast_error(merge_error, rank, "Prediction merge")
        if rank == 0:
            print(json.dumps(summary, indent=2))
        succeeded = True
    finally:
        # Keep the process group alive for PyTorch's distributed exception hook
        # on failures; torchrun tears it down when the process exits.
        if succeeded and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
