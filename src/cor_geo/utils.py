"""Configuration, reproducibility, I/O, and lightweight runtime utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

"""Cryptographic hashing helpers for immutable experiment artifacts."""


def sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it entirely into memory."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_json_mapping_keys(value: Any) -> Any:
    """Recursively apply JSON's string-key semantics before sorting mappings.

    ``yaml.safe_load`` preserves numeric mapping keys as numbers.  Passing a
    mapping containing both numeric and string keys directly to
    ``json.dumps(..., sort_keys=True)`` makes Python try to order incomparable
    objects (for example ``360`` and ``"finite_fov"``).  JSON object keys are
    strings, so normalize them explicitly and reject ambiguous collisions.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        original_keys: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str):
                json_key = key
            elif key is None:
                json_key = "null"
            elif isinstance(key, bool):
                json_key = "true" if key else "false"
            elif isinstance(key, int | float):
                json_key = str(key)
            else:
                raise TypeError(
                    "JSON mapping keys must be strings, numbers, booleans, or null; " f"got {type(key).__name__}"
                )
            if json_key in normalized:
                raise ValueError(
                    "Mapping keys become ambiguous after JSON normalization: "
                    f"{original_keys[json_key]!r} and {key!r}"
                )
            original_keys[json_key] = key
            normalized[json_key] = _normalize_json_mapping_keys(child)
        return normalized
    if isinstance(value, list | tuple):
        return [_normalize_json_mapping_keys(child) for child in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    normalized = _normalize_json_mapping_keys(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value in canonical form."""
    return sha256_bytes(canonical_json_bytes(value))


"""Safe filesystem IO used by tools and training runs."""


def ensure_parent(path: str | Path) -> Path:
    """Create and return the parent directory of a path."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_bytes(path: str | Path, payload: bytes, overwrite: bool = False) -> Path:
    """Atomically write bytes while protecting existing outputs by default."""
    resolved = ensure_parent(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def write_json(path: str | Path, value: Any, overwrite: bool = False) -> Path:
    """Write indented UTF-8 JSON atomically."""
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    return atomic_write_bytes(path, payload, overwrite=overwrite)


def write_yaml(path: str | Path, value: Any, overwrite: bool = False) -> Path:
    """Write deterministic YAML atomically."""
    payload = yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")
    return atomic_write_bytes(path, payload, overwrite=overwrite)


def assert_within_run(target: str | Path, run_root: str | Path) -> Path:
    """Ensure a destructive overwrite target is strictly inside one run root."""
    resolved_target = Path(target).expanduser().resolve()
    resolved_root = Path(run_root).expanduser().resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"Target must be strictly inside run root: {resolved_target}")
    return resolved_target


"""Structured append-only logging without process-global logger state."""


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one deterministic JSON object to a UTF-8 JSONL file."""
    resolved = ensure_parent(path)
    payload = json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    return resolved


"""Deterministic hashing, random state, and framework configuration."""


def stable_seed(*items: object) -> int:
    """Map a tuple of stable string representations to a uint32 seed."""
    payload = "::".join(map(str, items)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def configure_deterministic_algorithms() -> None:
    """Request deterministic PyTorch kernels without fixing any RNG state."""
    workspace_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise ValueError(f"Unsupported deterministic CUBLAS_WORKSPACE_CONFIG: {workspace_config}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def configure_determinism(seed: int) -> None:
    """Seed supported RNGs and request deterministic PyTorch algorithms."""
    configure_deterministic_algorithms()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_state() -> dict[str, Any]:
    """Capture RNG state for exact epoch-boundary resume."""
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    """Restore state produced by :func:`capture_random_state`."""
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_random_state"])
    if torch.cuda.is_available() and state["torch_cuda_random_states"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_states"])


"""Strict validation for reproducible CoR-Geo experiments."""


class ConfigError(ValueError):
    """Raised when a configuration violates the supported protocol."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {resolved}")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved configuration bundle stored in every checkpoint."""

    dataset: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    evaluation: dict[str, Any]
    runtime: dict[str, Any]
    paths: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "train": self.train,
            "evaluation": self.evaluation,
            "runtime": self.runtime,
            "paths": self.paths,
        }


@dataclass(frozen=True)
class TrainingTopology:
    """Per-process geometry derived from the invariant global batch."""

    world_size: int
    per_gpu_batch_size: int
    global_batch_size: int
    samples_per_fov_per_rank: int


def resolve_training_topology(train_config: dict[str, Any], world_size: int) -> TrainingTopology:
    """Resolve an equivalent one- or two-GPU execution of the global protocol."""
    world_size = int(world_size)
    if world_size not in {1, 2}:
        raise ConfigError(f"Training supports one or two GPUs, received world_size={world_size}")
    distributed_config = train_config["distributed"]
    configured_world_size = distributed_config.get("world_size", "auto")
    if configured_world_size != "auto" and int(configured_world_size) != world_size:
        raise ConfigError(
            f"Configured world_size={configured_world_size} differs from launcher world_size={world_size}"
        )
    global_batch_size = int(distributed_config["global_batch_size"])
    if global_batch_size % world_size:
        raise ConfigError("global_batch_size must be divisible by world_size")
    per_gpu_batch_size = global_batch_size // world_size
    configured_per_gpu = distributed_config.get("per_gpu_batch_size", "auto")
    if configured_per_gpu != "auto" and int(configured_per_gpu) != per_gpu_batch_size:
        raise ConfigError(
            f"Configured per_gpu_batch_size={configured_per_gpu} differs from derived value={per_gpu_batch_size}"
        )
    per_fov_global = int(train_config["mixed_fov_batch"]["samples_per_fov_per_global_batch"])
    if per_fov_global % world_size:
        raise ConfigError("samples_per_fov_per_global_batch must be divisible by world_size")
    per_fov_per_rank = per_fov_global // world_size
    configured_per_fov = train_config["mixed_fov_batch"].get("samples_per_fov_per_rank", "auto")
    if configured_per_fov != "auto" and int(configured_per_fov) != per_fov_per_rank:
        raise ConfigError(
            f"Configured samples_per_fov_per_rank={configured_per_fov} differs from derived value={per_fov_per_rank}"
        )
    return TrainingTopology(
        world_size=world_size,
        per_gpu_batch_size=per_gpu_batch_size,
        global_batch_size=global_batch_size,
        samples_per_fov_per_rank=per_fov_per_rank,
    )


def load_experiment_config(
    dataset_config: str | Path,
    default_config: str | Path = "configs/default.yaml",
) -> ExperimentConfig:
    """Load one dataset overlay on top of the shared CoR-Geo configuration."""
    shared = load_yaml(default_config)
    dataset = load_yaml(dataset_config)
    train = deepcopy(shared["train"])
    cache_root = str(train["dataset_cache"]["root"])
    train["dataset_cache"]["root"] = cache_root.format(dataset=str(dataset["dataset"]))
    config = ExperimentConfig(
        dataset=dataset,
        model=deepcopy(shared["model"]),
        train=train,
        evaluation=deepcopy(shared["evaluation"]),
        runtime=deepcopy(shared["runtime"]),
        paths=deepcopy(shared["paths"]),
    )
    validate_experiment_config(config)
    return config


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate implementation contracts and relationships between settings."""
    dataset_name = str(config.dataset.get("dataset"))
    _expect(dataset_name in {"cvact", "cvusa"}, "Only CVACT and CVUSA are supported")
    splits = list(map(str, config.dataset.get("manifest_splits", [])))
    _expect("train" in splits and "val" in splits and len(splits) == len(set(splits)), "Train/Val splits are required")
    _expect(config.model.get("name") == "cor_geo", "Unexpected model architecture")
    _expect(isinstance(config.train.get("seed"), int), "Training seed must be an integer")
    epochs = int(config.train["epochs"])
    _expect(epochs > 0, "epochs must be positive")
    _expect(int(config.train["progress_every_steps"]) > 0, "progress_every_steps must be positive")

    distributed = config.train["distributed"]
    _expect(distributed.get("world_size") == "auto", "Launcher must select world_size")
    _expect(distributed.get("per_gpu_batch_size") == "auto", "Per-GPU batch size must be derived")
    _expect(int(distributed["global_batch_size"]) > 0, "global_batch_size must be positive")
    _expect(bool(str(distributed["backend"]).strip()), "Distributed backend is required")
    _expect(int(distributed["timeout_minutes"]) > 0, "Distributed timeout must be positive")
    for world_size in (1, 2):
        resolve_training_topology(config.train, world_size)

    cache = config.train["dataset_cache"]
    _expect(cache.get("enabled") is True, "The resized input cache must be enabled")
    _expect(cache.get("format") == "cor_geo_paired_resized_rgb_uint8_memmap_v1", "Unsupported cache format")
    required_splits = set(map(str, cache.get("required_splits", [])))
    _expect(required_splits <= set(splits), "Cache-required splits must exist in the dataset configuration")
    _expect(bool(str(cache["root"]).strip()), "Dataset cache root is required")

    backbone = config.model["backbone"]
    patch_size = int(backbone["patch_size"])
    output_dim = int(backbone["output_dim"])
    block_count = int(backbone["expected_transformer_blocks"])
    _expect(patch_size > 0 and output_dim > 0 and block_count > 0, "Backbone dimensions must be positive")
    finetuning = backbone["finetuning"]
    trainable_blocks = tuple(map(int, finetuning["trainable_blocks"]))
    _expect(bool(trainable_blocks), "At least one trainable DINO block is required")
    _expect(
        trainable_blocks == tuple(range(trainable_blocks[0], block_count)),
        "Trainable DINO blocks must form a contiguous suffix",
    )
    starts = {int(index): int(epoch) for index, epoch in finetuning["block_update_start_epochs"].items()}
    _expect(set(starts) == set(trainable_blocks), "Every trainable DINO block needs one start epoch")
    _expect(all(1 <= epoch <= epochs for epoch in starts.values()), "DINO start epochs must lie within training")
    _expect(finetuning.get("train_final_norm") is True, "The current optimizer requires trainable DINO final norm")
    _expect(
        1 <= int(finetuning["final_norm_update_start_epoch"]) <= epochs,
        "Final-norm start epoch must lie within training",
    )

    inputs = config.model["input"]
    ground_height = int(inputs["ground_height"])
    panorama_width = int(inputs["panorama_width"])
    satellite_size = tuple(map(int, inputs["satellite_size"]))
    crop_widths = {int(key): int(value) for key, value in inputs["crop_widths"].items()}
    widths = {int(key): int(value) for key, value in inputs["widths"].items()}
    _expect(ground_height > 0 and panorama_width > 0, "Ground dimensions must be positive")
    _expect(len(satellite_size) == 2 and satellite_size[0] == satellite_size[1], "Satellite input must be square")
    _expect(set(crop_widths) == set(widths) and bool(widths), "Crop/model FoV sets must match")
    _expect(all(0 < width <= panorama_width for width in crop_widths.values()), "Invalid physical crop width")
    _expect(
        ground_height % patch_size == 0
        and all(width > 0 and width % patch_size == 0 for width in widths.values())
        and all(size > 0 and size % patch_size == 0 for size in satellite_size),
        "Model input dimensions must be positive multiples of patch_size",
    )

    architecture = config.model["architecture"]
    _expect(architecture["name"] == "column_ray_content_order", "Unsupported CoR-Geo architecture")
    angular_bins = int(architecture["angular_bins"])
    _expect(angular_bins > 1 and float(architecture["angular_step_deg"]) > 0, "Invalid angular discretization")
    resampling = architecture["ground_angular_resampling"]
    _expect(
        resampling["name"] == "conservative_interval_overlap" and resampling["applied_after_backbone"] is True,
        "Ground sequence construction must use post-backbone interval-overlap resampling",
    )
    source_columns = {int(key): int(value) for key, value in resampling["source_patch_columns"].items()}
    target_bins = {int(key): int(value) for key, value in resampling["target_direction_bins"].items()}
    _expect(set(source_columns) == set(widths) == set(target_bins), "Ground resampling FoV sets must match")
    _expect(
        source_columns == {fov: width // patch_size for fov, width in widths.items()},
        "Ground source columns must match the backbone grid",
    )
    _expect(all(0 < bins <= angular_bins for bins in target_bins.values()), "Invalid target direction count")

    column = architecture["content_order_encoder"]
    hidden_dim = int(column["hidden_dim"])
    content_dim = int(column["content_dim"])
    order_dim = int(column["order_dim"])
    sequence_length = ground_height // patch_size
    _expect(int(column["input_dim"]) == output_dim, "Encoder input_dim must match backbone output_dim")
    _expect(content_dim > 0 and order_dim > 0 and content_dim + order_dim == hidden_dim, "Invalid subspace dimensions")
    _expect(int(architecture["direction_dim"]) == hidden_dim, "direction_dim must equal encoder hidden_dim")
    _expect(
        int(column["ground_sequence_length"]) == sequence_length
        and int(column["satellite_sequence_length"]) == sequence_length,
        "Ground and satellite sequence lengths must equal the ground patch height",
    )
    _expect(0.0 < float(column["joint_order_weight"]) < 1.0, "joint_order_weight must lie in (0, 1)")
    _expect(
        int(column["attention_queries"]) == 1
        and column["attention_scaling"] == "inverse_sqrt_hidden_dim"
        and column["order_pooling"] == "normalized_first_cosine_moment"
        and column["joint_composition"] == "weighted_normalized_subspace_concat"
        and column["shared_across_modalities"] is True,
        "Unsupported Content--Order encoder settings",
    )
    _expect(
        column["ground_sequence_order"] == "bottom_to_top"
        and column["satellite_sampling"] == "dense_center_to_square_boundary_midpoint_bilinear_assignment",
        "Unsupported sequence geometry",
    )
    score = config.model["score"]
    _expect(
        score.get("name") == "fov_masked_cyclic_hard_max" and score.get("shift_reduction") == "hard_max",
        "CoR-Geo requires FoV-masked cyclic Hard-Max scoring",
    )

    loss = config.model["loss"]
    _expect(float(loss["info_nce_temperature"]) > 0, "InfoNCE temperature must be positive")
    _expect(0 <= float(loss["label_smoothing"]) < 1, "label_smoothing must lie in [0, 1)")
    _expect(loss.get("symmetric") is True, "The implemented objective is symmetric")
    _expect(float(loss["order_retrieval_weight"]) >= 0, "order_retrieval_weight cannot be negative")

    mixed = config.train["mixed_fov_batch"]
    fovs = list(map(int, mixed["fovs"]))
    _expect(len(fovs) == len(set(fovs)) and set(fovs) == set(target_bins), "Training FoVs must match model FoVs")
    per_fov = int(mixed["samples_per_fov_per_global_batch"])
    _expect(per_fov > 0, "Per-FoV batch size must be positive")
    _expect(
        per_fov * len(fovs) == int(distributed["global_batch_size"]),
        "Per-FoV quotas must sum to global_batch_size",
    )
    _expect(mixed.get("samples_per_fov_per_rank") == "auto", "Per-rank FoV quota must be derived")

    optimizer = config.train["optimizer"]
    optimizer_groups = optimizer["param_groups"]
    expected_groups = {"content_order_encoder", "dinov2_final_norm"} | {
        f"dinov2_block_{index}" for index in trainable_blocks
    }
    _expect(set(map(str, optimizer_groups)) == expected_groups, "Optimizer groups must match trainable modules")
    for name, values in optimizer_groups.items():
        base_lr = float(values["base_learning_rate"])
        min_lr = float(values["min_learning_rate"])
        active_epoch = int(values["active_from_epoch"])
        _expect(0 < min_lr <= base_lr, f"Invalid learning rates for {name}")
        _expect(1 <= active_epoch <= epochs, f"Invalid activation epoch for {name}")

    stages = sorted(config.train["stages"], key=lambda stage: int(stage["epoch_start"]))
    _expect(bool(stages), "At least one training stage is required")
    next_epoch = 1
    hard_epochs: list[int] = []
    for stage in stages:
        start = int(stage["epoch_start"])
        end = int(stage["epoch_end"])
        fraction = float(stage["hard_batch_fraction"])
        _expect(start == next_epoch and start <= end <= epochs, "Training stages must cover contiguous epochs")
        _expect(int(stage["steps_per_epoch"]) > 0, "Stage steps_per_epoch must be positive")
        _expect(0.0 <= fraction <= 1.0, "hard_batch_fraction must lie in [0, 1]")
        if fraction > 0:
            hard_epochs.append(start)
        next_epoch = end + 1
    _expect(next_epoch == epochs + 1, "Training stages must cover the full training range")

    gradient = config.train["gradient"]
    _expect(
        float(gradient["normal_batch_clip_norm"]) > 0 and float(gradient["hard_batch_clip_norm"]) > 0,
        "Gradient clip norms must be positive",
    )
    _expect(int(gradient["accumulation_steps"]) == 1, "Gradient accumulation is not implemented")

    hard = config.train["hard_mining"]
    if hard_epochs:
        _expect(hard.get("enabled") is True, "Hard mining is required by the configured stages")
    _expect(hard.get("pool_source") == "current_cor_geo_model", "Unsupported hard-pool source")
    refresh_epochs = sorted(set(map(int, hard["refresh_after_epochs"])))
    _expect(all(1 <= epoch < epochs for epoch in refresh_epochs), "Hard-pool refresh epochs are invalid")
    _expect(set(map(int, hard["mining_fovs"])) == set(fovs), "Hard-mining FoVs must match training FoVs")
    _expect(
        0 < int(hard["coarse_frequency_count"]) <= angular_bins // 2 + 1,
        "Invalid coarse Fourier frequency count",
    )
    coarse_count = int(hard["coarse_candidate_locations"])
    keep_count = int(hard["keep_negative_locations"])
    _expect(0 < keep_count <= coarse_count, "Hard-negative retention must fit the coarse candidate set")
    rank_start, rank_end = map(int, hard["sampling_rank_range"])
    _expect(1 <= rank_start <= rank_end <= keep_count, "Hard-negative sampling range is invalid")
    _expect(1 + int(hard["hard_neighbors_per_anchor"]) == per_fov, "Hard subgroup must fill one FoV quota")
    _expect(
        int(hard["descriptor_export_batch_size_per_gpu"]) > 0
        and int(hard["descriptor_export_workers_per_rank"]) >= 0
        and int(hard["coarse_query_chunk_size"]) > 0
        and int(hard["rerank_query_chunk_size"]) > 0,
        "Hard-mining batch, worker, and chunk sizes are invalid",
    )
    _expect(
        all(any(refresh < start for refresh in refresh_epochs) for start in hard_epochs),
        "Every hard-training stage needs an earlier pool refresh",
    )

    evaluation_fovs = list(map(int, config.evaluation["main_fovs"]))
    _expect(bool(evaluation_fovs) and len(evaluation_fovs) == len(set(evaluation_fovs)), "Invalid evaluation FoVs")
    _expect(evaluation_fovs == fovs, "Automatic validation FoVs must match the four training FoVs")
    _expect(str(config.evaluation["report_split"]) == "val", "Automatic model selection must use Val")
    during_training = config.evaluation["during_training"]
    _expect(during_training.get("enabled") is True, "Automatic validation must be enabled")
    interval = int(during_training["interval_epochs"])
    _expect(interval > 0, "Validation interval must be positive")
    _expect(during_training.get("selection_metric") == "macro_r1", "Best-model selection must use Macro R@1")
    metrics = config.evaluation["metrics"]
    _expect(all(int(value) > 0 for value in metrics["recall_ks"]), "Recall cutoffs must be positive")
    _expect(metrics["r1_percent_rounding"] in {"floor", "ceil"}, "Unsupported R@1% rounding")

def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).expanduser() / candidate
    return candidate.resolve()
