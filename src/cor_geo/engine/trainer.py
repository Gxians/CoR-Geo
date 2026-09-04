"""Two-GPU trainer for the registered CoR-Geo protocol."""

from __future__ import annotations

import json
import os
import shutil
from datetime import timedelta
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from cor_geo.config import ExperimentConfig
from cor_geo.datasets.cross_view import CrossViewDataset, SampleRequest, collate_mixed_fov
from cor_geo.datasets.manifests import read_manifest
from cor_geo.datasets.resized_cache import (
    ResizedRGBMemmapCache,
    cache_metadata_path,
    cache_request,
)
from cor_geo.datasets.samplers import (
    PlannedRankBatchSampler,
    build_epoch_plan,
    stage_for_epoch,
)
from cor_geo.engine.checkpoint import checkpoint_payload, load_checkpoint, save_epoch_checkpoint
from cor_geo.engine.stage_scheduler import GroupCosineScheduler, build_optimizer
from cor_geo.losses.info_nce import CoRGeoLoss
from cor_geo.mining.hard_negative_mining import HardNegativeCandidateBank
from cor_geo.mining.hard_pool import load_compact_hard_pool, write_compact_hard_pool
from cor_geo.models.cor_geo_model import CoRGeoModel
from cor_geo.reproducibility import configure_determinism, stable_seed
from cor_geo.utils.environment import environment_snapshot, git_state
from cor_geo.utils.hashing import sha256_file, sha256_json
from cor_geo.utils.io import write_json, write_yaml
from cor_geo.utils.logging import append_jsonl


def initialize_distributed(train_config: dict[str, Any]) -> tuple[int, int, int]:
    """Initialize the fixed two-GPU process group."""
    if not torch.cuda.is_available():
        raise RuntimeError("CoR-Geo training requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    expected = int(train_config["distributed"]["world_size"])
    if rank < 0 or local_rank < 0 or world_size != expected:
        raise RuntimeError(f"Launch with torchrun --nproc_per_node={expected}")
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(
        backend=str(train_config["distributed"]["backend"]),
        timeout=timedelta(
            minutes=int(train_config["distributed"]["timeout_minutes"])
        ),
    )
    return rank, local_rank, world_size


def _broadcast(value: Any, rank: int) -> Any:
    values = [value if rank == 0 else None]
    distributed.broadcast_object_list(values, src=0)
    return values[0]


def _assert_global_unique(local_indices: torch.Tensor, world_size: int) -> None:
    gathered = [torch.empty_like(local_indices) for _ in range(world_size)]
    distributed.all_gather(gathered, local_indices)
    values = torch.cat(gathered)
    if len(torch.unique(values)) != len(values):
        raise ValueError("A global mixed batch contains duplicate locations")


def _logical_modules(model: CoRGeoModel) -> dict[str, nn.Module]:
    output: dict[str, nn.Module] = {
        "shared_content_order_encoder": model.content_order_encoder,
    }
    output.update(
        {
            f"dinov2_block_{index}": model.backbone.blocks[index]
            for index in model.backbone.registered_indices
        }
    )
    output["dinov2_final_norm"] = model.backbone.model.norm
    return output


def _gradient_norms(model: CoRGeoModel) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, module in _logical_modules(model).items():
        squared = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        value = squared**0.5
        if not isfinite(value):
            raise FloatingPointError(f"Non-finite gradient norm for {name}")
        output[name] = value
    return output


class _MiningSatelliteRows(Dataset[dict[str, Any]]):
    """Contiguous satellite shard used by one mining rank."""

    def __init__(
        self,
        dataset: CrossViewDataset,
        indices: list[int],
        source_epoch: int,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.source_epoch = int(source_epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.dataset.load_satellite(
            SampleRequest(
                index=self.indices[int(position)],
                epoch=self.source_epoch,
                fov_deg=360,
                orientation_mode="hard_mining",
                source_epoch=self.source_epoch,
            )
        )


class _MiningGroundRows(Dataset[dict[str, Any]]):
    """Open one panorama once and return the four source-epoch crops."""

    def __init__(
        self,
        dataset: CrossViewDataset,
        indices: list[int],
        source_epoch: int,
        fovs: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.source_epoch = int(source_epoch)
        self.fovs = tuple(map(int, fovs))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.dataset.load_mining_ground_bundle(
            self.indices[int(position)],
            self.source_epoch,
            self.fovs,
        )


def _descriptor_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
        persistent_workers=int(workers) > 0,
        drop_last=False,
    )


def _encode_satellite_directions(
    model: CoRGeoModel,
    dataset: CrossViewDataset,
    indices: list[int],
    source_epoch: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    progress_label: str,
) -> tuple[np.ndarray, list[str]]:
    """Export normalized satellite direction sequences for self-mining."""
    was_training = model.training
    model.eval()
    directions: list[np.ndarray] = []
    ids: list[str] = []
    started = perf_counter()
    cursor = 0
    loader = _descriptor_loader(
        _MiningSatelliteRows(dataset, indices, source_epoch),
        batch_size,
        workers,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["satellite"].to(device, dtype=torch.float32, non_blocking=True)
            directions.append(
                model.encode_satellite(images).direction.float().cpu().numpy()
            )
            ids.extend(map(str, batch["satellite_id"]))
            cursor += len(images)
            if batch_index == 1 or batch_index % 100 == 0:
                rate = cursor / max(perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] satellite {cursor}/{len(indices)} "
                    f"{rate:.1f} rows/s",
                    flush=True,
                )
    if was_training:
        model.train(True)
        model.set_train_epoch(source_epoch)
    if cursor != len(indices):
        raise RuntimeError("Satellite Structure export is incomplete")
    return np.concatenate(directions).astype(np.float32, copy=False), ids


def _encode_ground_directions(
    model: CoRGeoModel,
    dataset: CrossViewDataset,
    indices: list[int],
    source_epoch: int,
    fovs: list[int],
    device: torch.device,
    batch_size: int,
    workers: int,
    progress_label: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    """Export four direction sequences while opening every panorama once."""
    was_training = model.training
    model.eval()
    directions: dict[int, list[np.ndarray]] = {fov: [] for fov in fovs}
    validities: dict[int, list[np.ndarray]] = {fov: [] for fov in fovs}
    ids: list[str] = []
    started = perf_counter()
    cursor = 0
    loader = _descriptor_loader(
        _MiningGroundRows(dataset, indices, source_epoch, fovs),
        batch_size,
        workers,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            rows = len(batch["query_id"])
            for fov in fovs:
                images = batch[f"ground_{fov}"].to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                representation = model.encode_ground(images, fov)
                directions[fov].append(
                    representation.direction.float().cpu().numpy()
                )
                validities[fov].append(representation.valid.cpu().numpy())
            ids.extend(map(str, batch["query_id"]))
            cursor += rows
            if batch_index == 1 or batch_index % 100 == 0:
                rate = cursor / max(perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] ground {cursor}/{len(indices)} "
                    f"{rate:.1f} rows/s",
                    flush=True,
                )
    if was_training:
        model.train(True)
        model.set_train_epoch(source_epoch)
    if cursor != len(indices):
        raise RuntimeError("Ground Structure export is incomplete")
    return (
        {
            fov: np.concatenate(parts).astype(np.float32, copy=False)
            for fov, parts in directions.items()
        },
        {
            fov: np.concatenate(parts).astype(np.bool_, copy=False)
            for fov, parts in validities.items()
        },
        ids,
    )


def _hard_source_epoch(epoch: int, train_config: dict[str, Any]) -> int:
    eligible = [
        int(value)
        for value in train_config["hard_mining"]["refresh_after_epochs"]
        if int(value) < int(epoch)
    ]
    if not eligible:
        raise ValueError(f"Epoch {epoch} has no prior mining refresh")
    return max(eligible)


def _expected_pool_metadata(
    source_epoch: int,
    checkpoint_hash: str,
    provenance: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    hard = train_config["hard_mining"]
    return {
        "source_epoch": int(source_epoch),
        "source_checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": str(provenance["manifest_hashes"]["train"]),
        "model_config_sha256": str(provenance["model_config_sha256"]),
        "candidate_strategy": str(hard["candidate_strategy"]),
        "coarse_frequency_count": int(hard["coarse_frequency_count"]),
        "coarse_candidate_locations": int(hard["coarse_candidate_locations"]),
        "keep_negative_locations": int(hard["keep_negative_locations"]),
        "crop_orientation_contract": str(hard["crop_orientation_contract"]),
        "format": str(hard["storage_format"]),
    }


def _hard_pool_root(
    run_root: Path,
    source_epoch: int,
    checkpoint_hash: str,
) -> Path:
    return (
        run_root
        / "artifacts"
        / "hard_negatives"
        / checkpoint_hash
        / f"epoch_{source_epoch}"
    )


def _load_hard_pools(
    run_root: Path,
    epoch: int,
    fovs: list[int],
    provenance: dict[str, Any],
    train_config: dict[str, Any],
    manifest_size: int,
) -> dict[int, np.ndarray]:
    source_epoch = _hard_source_epoch(epoch, train_config)
    hard = train_config["hard_mining"]
    if hard.get("pool_source") != "current_cor_geo_model":
        raise ValueError("Hard pools must come from the current CoR-Geo run")
    checkpoint = run_root / "checkpoints" / f"epoch_{source_epoch:03d}.ckpt"
    checkpoint_hash = sha256_file(checkpoint)
    root = _hard_pool_root(run_root, source_epoch, checkpoint_hash)
    metadata_path = root / "refresh.metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Current CoR-Geo hard pool is missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        refresh_metadata = json.load(handle)
    expected = _expected_pool_metadata(
        source_epoch,
        checkpoint_hash,
        provenance,
        train_config,
    )
    for key, value in {
        **expected,
        "fovs": list(map(int, fovs)),
        "location_count": int(manifest_size),
    }.items():
        if refresh_metadata.get(key) != value:
            raise ValueError(f"Current CoR-Geo hard-pool metadata mismatch: {key}")
    return {
        fov: load_compact_hard_pool(
            root,
            fov,
            expected,
            manifest_size,
            int(train_config["hard_mining"]["keep_negative_locations"]),
        )
        for fov in fovs
    }


def _shard_bounds(length: int, rank: int, world_size: int) -> tuple[int, int]:
    return length * rank // world_size, length * (rank + 1) // world_size


def _generate_hard_pools(
    model: CoRGeoModel,
    dataset: CrossViewDataset,
    train_manifest: Any,
    config: ExperimentConfig,
    source_epoch: int,
    source_checkpoint: Path,
    run_root: Path,
    device: torch.device,
    provenance: dict[str, Any],
    rank: int,
    world_size: int,
) -> None:
    """Generate self-contained Top-64 hard pools for all four FoVs."""
    checkpoint_hash = _broadcast(
        sha256_file(source_checkpoint) if rank == 0 else None,
        rank,
    )
    hard = config.train["hard_mining"]
    output_root = _hard_pool_root(run_root, source_epoch, checkpoint_hash)
    complete = _broadcast(
        (output_root / "refresh.metadata.json").is_file() if rank == 0 else None,
        rank,
    )
    if complete:
        return
    staging_root = (
        run_root
        / "artifacts"
        / ".hard_negative_staging"
        / checkpoint_hash
        / f"epoch_{source_epoch}"
    )
    setup_error: str | None = None
    if rank == 0:
        try:
            if output_root.exists():
                shutil.rmtree(output_root)
            if staging_root.exists():
                shutil.rmtree(staging_root)
            staging_root.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    setup_error = _broadcast(setup_error, rank)
    if setup_error:
        raise RuntimeError(setup_error)
    distributed.barrier()

    location_count = len(train_manifest)
    shard_start, shard_stop = _shard_bounds(location_count, rank, world_size)
    query_indices = np.arange(shard_start, shard_stop, dtype=np.int32)
    index_list = query_indices.tolist()
    fovs = list(map(int, hard["mining_fovs"]))
    batch_size = int(hard["descriptor_export_batch_size_per_gpu"])
    workers = int(hard["descriptor_export_workers_per_rank"])
    rank_root = staging_root / f"rank_{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=False)

    satellite_direction, satellite_ids = _encode_satellite_directions(
        model,
        dataset,
        index_list,
        source_epoch,
        device,
        batch_size,
        workers,
        f"rank{rank}/mine",
    )
    expected_satellite_ids = (
        train_manifest.iloc[shard_start:shard_stop]["satellite_id"].astype(str).tolist()
    )
    if satellite_ids != expected_satellite_ids:
        raise ValueError("Satellite mining order differs from train manifest")
    np.save(
        rank_root / "satellite_direction.fp32.npy",
        satellite_direction,
        allow_pickle=False,
    )
    del satellite_direction
    distributed.barrier()
    satellite_direction = np.concatenate(
        [
            np.load(
                staging_root
                / f"rank_{shard_rank:02d}"
                / "satellite_direction.fp32.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for shard_rank in range(world_size)
        ]
    )
    if len(satellite_direction) != location_count:
        raise RuntimeError("Satellite mining shards do not cover the manifest")

    query_directions, query_validities, query_ids = _encode_ground_directions(
        model,
        dataset,
        index_list,
        source_epoch,
        fovs,
        device,
        batch_size,
        workers,
        f"rank{rank}/mine",
    )
    expected_query_ids = (
        train_manifest.iloc[shard_start:shard_stop]["query_id"].astype(str).tolist()
    )
    if query_ids != expected_query_ids:
        raise ValueError("Ground mining order differs from train manifest")
    common = _expected_pool_metadata(
        source_epoch,
        checkpoint_hash,
        provenance,
        config.train,
    )
    candidate_bank = HardNegativeCandidateBank(
        satellite_direction,
        device,
        int(hard["coarse_frequency_count"]),
    )
    del satellite_direction
    for fov in fovs:
        result = candidate_bank.mine(
            query_directions[fov],
            query_validities[fov],
            query_indices,
            coarse_keep=int(hard["coarse_candidate_locations"]),
            final_keep=int(hard["keep_negative_locations"]),
            search_chunk_size=int(hard["coarse_query_chunk_size"]),
            rerank_chunk_size=int(hard["rerank_query_chunk_size"]),
        )
        fov_root = rank_root / f"fov_{fov}"
        fov_root.mkdir(parents=True, exist_ok=False)
        np.save(fov_root / "query_indices.npy", query_indices, allow_pickle=False)
        np.save(
            fov_root / "negative_indices.npy",
            result.indices,
            allow_pickle=False,
        )
        np.save(
            fov_root / "negative_scores.npy",
            result.scores,
            allow_pickle=False,
        )
        write_json(
            fov_root / "mining_diagnostics.json",
            {
                "query_count": len(query_indices),
                "positive_coarse_topk_recall": result.positive_coarse_topk_recall,
            },
        )
        print(
            f"[rank{rank}/mine] FoV {fov} "
            f"Top-{int(hard['keep_negative_locations'])} ready; "
            f"positive coarse Top-{int(hard['coarse_candidate_locations'])} "
            f"recall={result.positive_coarse_topk_recall:.4f}",
            flush=True,
        )
    distributed.barrier()

    publish_error: str | None = None
    if rank == 0:
        try:
            mining_diagnostics: dict[str, Any] = {}
            for fov in fovs:
                merged_queries = np.concatenate(
                    [
                        np.load(
                            staging_root
                            / f"rank_{shard_rank:02d}"
                            / f"fov_{fov}"
                            / "query_indices.npy",
                            allow_pickle=False,
                        )
                        for shard_rank in range(world_size)
                    ]
                )
                if not np.array_equal(
                    merged_queries,
                    np.arange(location_count, dtype=np.int32),
                ):
                    raise RuntimeError("Mining query shards do not cover the manifest")
                write_compact_hard_pool(
                    output_root,
                    fov,
                    np.concatenate(
                        [
                            np.load(
                                staging_root
                                / f"rank_{shard_rank:02d}"
                                / f"fov_{fov}"
                                / "negative_indices.npy",
                                allow_pickle=False,
                            )
                            for shard_rank in range(world_size)
                        ]
                    ),
                    np.concatenate(
                        [
                            np.load(
                                staging_root
                                / f"rank_{shard_rank:02d}"
                                / f"fov_{fov}"
                                / "negative_scores.npy",
                                allow_pickle=False,
                            )
                            for shard_rank in range(world_size)
                        ]
                    ),
                    common,
                )
                rank_diagnostics = []
                for shard_rank in range(world_size):
                    path = (
                        staging_root
                        / f"rank_{shard_rank:02d}"
                        / f"fov_{fov}"
                        / "mining_diagnostics.json"
                    )
                    with path.open("r", encoding="utf-8") as handle:
                        rank_diagnostics.append(json.load(handle))
                query_total = sum(int(row["query_count"]) for row in rank_diagnostics)
                positive_total = sum(
                    int(row["query_count"])
                    * float(row["positive_coarse_topk_recall"])
                    for row in rank_diagnostics
                )
                mining_diagnostics[str(fov)] = {
                    "query_count": query_total,
                    "positive_coarse_topk_recall": positive_total
                    / max(query_total, 1),
                }
            write_json(
                output_root / "refresh.metadata.json",
                {
                    **common,
                    "fovs": fovs,
                    "location_count": location_count,
                    "world_size": world_size,
                    "mining_diagnostics": mining_diagnostics,
                },
            )
            shutil.rmtree(staging_root)
        except Exception as error:
            publish_error = f"{type(error).__name__}: {error}"
    publish_error = _broadcast(publish_error, rank)
    if publish_error:
        raise RuntimeError(f"Hard-pool publication failed: {publish_error}")
    distributed.barrier()
    del candidate_bank, query_directions, query_validities
    torch.cuda.empty_cache()


def _loss_from_config(model_config: dict[str, Any]) -> CoRGeoLoss:
    loss = model_config["loss"]
    return CoRGeoLoss(
        info_nce_temperature=float(loss["info_nce_temperature"]),
        label_smoothing=float(loss["label_smoothing"]),
        symmetric=bool(loss["symmetric"]),
        order_retrieval_weight=float(loss["order_retrieval_weight"]),
    )


def train(
    config: ExperimentConfig,
    run_name: str,
    resume: str | Path | None = None,
) -> None:
    """Train one complete CoR-Geo run using the seed stored in the config."""
    seed = int(config.train["seed"])
    rank, local_rank, world_size = initialize_distributed(config.train)
    configure_determinism(seed)
    device = torch.device("cuda", local_rank)
    project_root = Path(config.paths["project_root"])
    dataset_name = str(config.dataset["dataset"])
    run_root = Path(config.paths["output_root"]) / dataset_name / run_name
    manifest_root = project_root / "data_manifests" / dataset_name
    manifest_splits = tuple(map(str, config.dataset["manifest_splits"]))
    manifest_paths = {
        split: manifest_root / f"{split}.parquet"
        for split in manifest_splits
    }
    for path in manifest_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train_manifest = read_manifest(manifest_paths["train"])
    global_batch = int(config.train["distributed"]["global_batch_size"])
    steps_per_epoch = len(train_manifest) // global_batch
    if steps_per_epoch != 555:
        raise ValueError(
            f"Unexpected {dataset_name.upper()} steps per epoch: {steps_per_epoch}"
        )

    cache_root, require_train_cache = cache_request(config.train, "train")
    cache_metadata_hashes: dict[str, str] = {}
    for split in map(
        str,
        config.train["dataset_cache"]["required_splits"],
    ):
        split_manifest = (
            train_manifest
            if split == "train"
            else read_manifest(manifest_paths[split])
        )
        ResizedRGBMemmapCache(
            config.train["dataset_cache"]["root"],
            split_manifest,
            int(config.model["input"]["ground_height"]),
            int(config.model["input"]["panorama_width"]),
            int(config.model["input"]["satellite_size"][0]),
        )
        metadata_path = cache_metadata_path(
            config.train["dataset_cache"]["root"],
            split,
        )
        cache_metadata_hashes[split] = sha256_file(metadata_path)

    dataset = CrossViewDataset(
        train_manifest,
        global_seed=seed,
        ground_height=int(config.model["input"]["ground_height"]),
        panorama_width=int(config.model["input"]["panorama_width"]),
        satellite_size=int(config.model["input"]["satellite_size"][0]),
        ground_widths=config.model["input"]["widths"],
        resized_cache_root=cache_root,
        require_resized_cache=require_train_cache,
        dataset_name=dataset_name,
    )
    model = CoRGeoModel(
        config.model,
        dinov2_root=config.paths["dinov2_root"],
        checkpoint_path=config.paths["checkpoints"]["dinov2_vitb14"],
    ).to(device)
    optimizer = build_optimizer(model, config.train)
    scheduler = GroupCosineScheduler(optimizer, config.train, steps_per_epoch)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=bool(
            config.train["distributed"]["find_unused_parameters"]
        ),
    )
    loss_function = _loss_from_config(config.model).to(device)

    setup_error: str | None = None
    created_new_run = False
    if rank == 0:
        try:
            if resume is None:
                if run_root.exists():
                    raise FileExistsError(f"Run already exists: {run_root}")
                run_root.mkdir(parents=True, exist_ok=False)
                write_yaml(run_root / "config_resolved.yaml", config.as_dict())
                write_json(run_root / "environment.json", environment_snapshot())
                created_new_run = True
            elif not run_root.is_dir():
                raise FileNotFoundError(f"Resume run does not exist: {run_root}")
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    setup_error = _broadcast(setup_error, rank)
    if setup_error:
        raise RuntimeError(setup_error)
    distributed.barrier()

    provenance = {
        "manifest_hashes": {
            split: sha256_file(path) for split, path in manifest_paths.items()
        },
        "dataset_config_sha256": sha256_json(config.dataset),
        "dinov2_checkpoint_sha256": sha256_file(
            config.paths["checkpoints"]["dinov2_vitb14"]
        ),
        "dataset_cache_metadata_hashes": cache_metadata_hashes,
        "model_config_sha256": sha256_json(config.model),
        "score_config_sha256": sha256_json(config.model["score"]),
        "project_git_state": git_state(project_root),
        "dinov2_git_state": git_state(config.paths["dinov2_root"]),
    }
    exclusions_config = config.dataset.get("exclusions_config")
    if exclusions_config is not None:
        provenance["exclusions_sha256"] = sha256_file(
            project_root / str(exclusions_config)
        )
    if rank == 0 and created_new_run:
        write_json(
            run_root / "data_protocol.json",
            {
                "dataset": dataset_name,
                "dataset_protocol_version": str(config.dataset["protocol_version"]),
                "protocol_version": (
                    "cor_geo_bilinear_l16_top64_refresh6_gbs64_"
                    f"{dataset_name}_v1"
                ),
                "manifest_counts": {
                    split: len(read_manifest(path))
                    for split, path in manifest_paths.items()
                },
                "manifest_hashes": provenance["manifest_hashes"],
                "dataset_cache_metadata_hashes": cache_metadata_hashes,
                "hard_pool_source": {
                    "mode": "current_cor_geo_model",
                    "candidate_strategy": config.train["hard_mining"][
                        "candidate_strategy"
                    ],
                },
            },
        )
        write_json(
            run_root / "git_status.json",
            {
                "project": provenance["project_git_state"],
                "dinov2": provenance["dinov2_git_state"],
            },
        )

    start_epoch = 1
    if resume is not None:
        payload = load_checkpoint(
            resume,
            ddp_model,
            optimizer,
            scheduler,
            restore_rng=True,
            expected_resolved_config=config.as_dict(),
        )
        start_epoch = int(payload["completed_epoch"]) + 1
        distributed.barrier()
    completed_epoch = start_epoch - 1
    final_epoch = int(config.train["epochs"])
    if not start_epoch <= final_epoch:
        raise ValueError(
            f"Training is already complete: start={start_epoch}, final={final_epoch}"
        )

    fovs = list(map(int, config.train["mixed_fov_batch"]["fovs"]))
    refresh_epochs = set(
        map(int, config.train["hard_mining"]["refresh_after_epochs"])
    )
    if (
        resume is not None
    ) and completed_epoch in refresh_epochs:
        retained_checkpoint = (
            run_root / "checkpoints" / f"epoch_{completed_epoch:03d}.ckpt"
        )
        if not retained_checkpoint.is_file():
            retained_checkpoint = run_root / "checkpoints" / "last.ckpt"
        _generate_hard_pools(
            model,
            dataset,
            train_manifest,
            config,
            completed_epoch,
            retained_checkpoint,
            run_root,
            device,
            provenance,
            rank,
            world_size,
        )
    sampler = PlannedRankBatchSampler(
        rank,
        world_size,
        int(config.train["distributed"]["per_gpu_batch_size"]),
    )
    loader_config = config.train["dataloader"]
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_mixed_fov,
        num_workers=int(loader_config["num_workers_per_rank"]),
        pin_memory=bool(loader_config["pin_memory"]),
        persistent_workers=bool(loader_config["persistent_workers"]),
        prefetch_factor=int(loader_config["prefetch_factor"]),
    )
    metrics_path = run_root / "train_metrics.jsonl"
    retained_epochs = set(
        map(int, config.train["checkpoint"]["retained_epochs"])
    )
    for epoch in range(start_epoch, final_epoch + 1):
        ddp_model.train(True)
        model.set_train_epoch(epoch)
        stage = stage_for_epoch(config.train["stages"], epoch)
        hard_pools = None
        if rank == 0 and float(stage["hard_batch_fraction"]) > 0:
            hard_pools = _load_hard_pools(
                run_root,
                epoch,
                fovs,
                provenance,
                config.train,
                len(train_manifest),
            )
        plan = (
            build_epoch_plan(
                train_manifest,
                epoch,
                seed,
                config.train,
                hard_pools,
            )
            if rank == 0
            else None
        )
        plan = _broadcast(plan, rank)
        sampler.set_plan(plan, epoch)
        for step_in_epoch, batch in enumerate(loader, start=1):
            torch.cuda.reset_peak_memory_stats(device)
            local_indices = batch["manifest_index"].to(device, non_blocking=True)
            if step_in_epoch == 1:
                _assert_global_unique(local_indices, world_size)
            local_counts = {
                fov: int((batch["fov_deg"] == fov).sum()) for fov in fovs
            }
            expected_per_fov = int(
                config.train["mixed_fov_batch"]["samples_per_fov_per_rank"]
            )
            if local_counts != {fov: expected_per_fov for fov in fovs}:
                raise ValueError(f"Rank-local FoV counts are invalid: {local_counts}")

            learning_rates = scheduler.set_for_next_step()
            optimizer.zero_grad(set_to_none=True)
            ground_by_fov = {
                int(fov): tensor.to(device, non_blocking=True)
                for fov, tensor in batch["ground_by_fov"].items()
            }
            positions_by_fov = {
                int(fov): tensor.to(device, non_blocking=True)
                for fov, tensor in batch["ground_positions_by_fov"].items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = ddp_model(
                    ground_by_fov,
                    positions_by_fov,
                    batch["satellite"].to(device, non_blocking=True),
                )
                losses = loss_function(output)
            losses.total.backward()

            diagnostics_interval = int(
                config.train["diagnostics"][
                    "module_gradient_norm_interval_steps"
                ]
            )
            record_module_norms = rank == 0 and (
                step_in_epoch == 1
                or step_in_epoch == len(loader)
                or step_in_epoch % diagnostics_interval == 0
            )
            gradient_norms = _gradient_norms(model) if record_module_norms else None
            if epoch <= 8 and gradient_norms is not None and any(
                gradient_norms[name] != 0.0
                for name in gradient_norms
                if name.startswith("dinov2")
            ):
                raise RuntimeError("DINOv2 received gradients during frozen epochs 1-8")
            batch_kind = plan[step_in_epoch - 1].kind
            gradient_config = config.train["gradient"]
            clip_norm = float(
                gradient_config["hard_batch_clip_norm"]
                if batch_kind == "hard"
                else gradient_config["normal_batch_clip_norm"]
            )
            global_gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=clip_norm,
            )
            optimizer.step()
            scheduler.step_completed()
            if rank == 0:
                global_norm = float(global_gradient_norm)
                append_jsonl(
                    metrics_path,
                    {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": scheduler.global_step,
                        "batch_kind": batch_kind,
                        "fov_counts": {
                            str(key): value for key, value in local_counts.items()
                        },
                        "loss": float(losses.total.detach()),
                        "joint_retrieval_loss": float(losses.joint_retrieval),
                        "order_retrieval_loss": float(losses.order_retrieval),
                        "joint_in_batch_accuracy": float(
                            losses.joint_in_batch_accuracy
                        ),
                        "order_in_batch_accuracy": float(
                            losses.order_in_batch_accuracy
                        ),
                        "learning_rates": learning_rates,
                        "gradient_norms": gradient_norms,
                        "global_gradient_norm_pre_clip": global_norm,
                        "gradient_clip_norm": clip_norm,
                        "gradient_clip_coefficient": min(
                            1.0, clip_norm / (global_norm + 1.0e-6)
                        ),
                        "peak_gpu_memory": torch.cuda.max_memory_allocated(
                            device
                        ),
                    },
                )
            del output, losses, ground_by_fov, positions_by_fov

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        epoch_checkpoint: Path | None = None
        if rank == 0:
            payload = checkpoint_payload(
                ddp_model,
                optimizer,
                scheduler,
                completed_epoch=epoch,
                resolved_config=config.as_dict(),
                provenance=provenance,
                next_epoch_schedule_seed=stable_seed(
                    seed,
                    dataset_name,
                    "train",
                    epoch + 1,
                    "unique_locations",
                ),
            )
            epoch_checkpoint, _ = save_epoch_checkpoint(
                payload,
                run_root / "checkpoints",
                epoch,
                retain_epoch=epoch in retained_epochs,
            )
        checkpoint_value = _broadcast(
            str(epoch_checkpoint) if rank == 0 else None,
            rank,
        )
        distributed.barrier()
        if epoch in refresh_epochs:
            _generate_hard_pools(
                model,
                dataset,
                train_manifest,
                config,
                epoch,
                Path(checkpoint_value),
                run_root,
                device,
                provenance,
                rank,
                world_size,
            )
    distributed.destroy_process_group()
