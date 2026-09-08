"""Configuration, reproducibility, I/O, and lightweight runtime utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# ---- src/cor_geo/utils/hashing.py ----

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


# ---- src/cor_geo/utils/io.py ----

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


# ---- src/cor_geo/utils/logging.py ----

"""Structured append-only logging without process-global logger state."""


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one deterministic JSON object to a UTF-8 JSONL file."""
    resolved = ensure_parent(path)
    payload = json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    return resolved


# ---- src/cor_geo/utils/environment.py ----

"""Environment and repository provenance collection."""


def git_state(path: str | Path) -> dict[str, Any]:
    """Return commit and dirty state without mutating a repository."""
    root = Path(path).expanduser().resolve()
    if not (root / ".git").exists():
        return {"path": str(root), "is_repository": False, "commit": None, "dirty": None}
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"path": str(root), "is_repository": True, "commit": commit, "dirty": bool(status.strip())}


def environment_snapshot() -> dict[str, Any]:
    """Collect runtime versions relevant to reproducibility."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


# ---- src/cor_geo/reproducibility.py ----

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


# ---- src/cor_geo/config.py ----

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
    paths: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "train": self.train,
            "evaluation": self.evaluation,
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
            f"Configured world_size={configured_world_size} differs from torchrun world_size={world_size}"
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
        paths=deepcopy(shared["paths"]),
    )
    validate_experiment_config(config)
    return config


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Reject silent changes to the supported CVACT/CVUSA protocol."""
    dataset_name = str(config.dataset.get("dataset"))
    _expect(dataset_name in {"cvact", "cvusa"}, "Only CVACT and CVUSA are supported")
    expected_splits = ["train", "val", "test"] if dataset_name == "cvact" else ["train", "val"]
    _expect(
        list(map(str, config.dataset.get("manifest_splits", []))) == expected_splits,
        f"Unexpected {dataset_name.upper()} manifest splits",
    )
    _expect(config.model.get("name") == "cor_geo", "Unexpected model architecture")
    distributed_config = config.train["distributed"]
    _expect(distributed_config.get("world_size") == "auto", "Training world_size must be selected by torchrun")
    _expect(
        distributed_config.get("per_gpu_batch_size") == "auto",
        "Per-GPU batch size must be derived from the fixed global batch",
    )
    _expect(int(distributed_config["global_batch_size"]) == 64, "Global batch size must remain 64")
    _expect(int(config.train["epochs"]) == 64, "Training endpoint must be epoch 64")
    _expect(isinstance(config.train.get("seed"), int), "Training seed must be an integer")
    dataset_cache = config.train["dataset_cache"]
    _expect(dataset_cache.get("enabled") is True, "The SSD input cache must be enabled")
    _expect(
        dataset_cache.get("format") == "cor_geo_paired_resized_rgb_uint8_memmap_v1",
        "Unexpected dataset cache format",
    )
    _expect(
        list(map(str, dataset_cache.get("required_splits", []))) == ["train", "val"],
        "Train and Val caches must be mandatory",
    )
    _expect(bool(str(dataset_cache["root"]).strip()), "Dataset cache root is required")

    input_config = config.model["input"]
    expected_ground_height = 224
    _expect(
        int(input_config["ground_height"]) == expected_ground_height,
        f"{dataset_name.upper()} ground height must be {expected_ground_height}",
    )
    _expect(
        int(input_config["panorama_width"]) == 756,
        "Ground panorama width must be 756",
    )
    _expect(
        list(map(int, input_config["satellite_size"])) == [378, 378],
        "Satellite input must be 378x378",
    )
    crop_widths = {int(key): int(value) for key, value in input_config["crop_widths"].items()}
    _expect(
        crop_widths == {360: 756, 180: 378, 90: 189, 70: 147},
        f"Unexpected physical FoV crop widths: {crop_widths}",
    )
    widths = {int(key): int(value) for key, value in input_config["widths"].items()}
    _expect(
        widths == {360: 756, 180: 378, 90: 196, 70: 154},
        f"Unexpected ground widths: {widths}",
    )
    patch_size = int(config.model["backbone"]["patch_size"])
    _expect(
        int(input_config["ground_height"]) % patch_size == 0
        and all(width % patch_size == 0 for width in widths.values())
        and all(size % patch_size == 0 for size in input_config["satellite_size"]),
        "Every model tensor size must be divisible by the DINO patch size",
    )
    architecture = config.model["architecture"]
    _expect(int(architecture["angular_bins"]) == 36, "Core requires 36 angular bins")
    _expect(
        architecture["name"] == "column_ray_content_order",
        "Unexpected active architecture",
    )
    resampling = architecture["ground_angular_resampling"]
    source_columns = {int(key): int(value) for key, value in resampling["source_patch_columns"].items()}
    target_bins = {int(key): int(value) for key, value in resampling["target_direction_bins"].items()}
    _expect(
        resampling["name"] == "conservative_interval_overlap"
        and resampling["applied_after_backbone"] is True
        and source_columns == {360: 54, 180: 27, 90: 14, 70: 11}
        and target_bins == {360: 36, 180: 18, 90: 9, 70: 7},
        "Ground features must use the fixed 54/27/14/11 to 36/18/9/7 resampling",
    )
    _expect(
        source_columns == {fov: width // patch_size for fov, width in widths.items()},
        "Ground resampling sources must equal the actual DINO patch columns",
    )
    column = architecture["content_order_encoder"]
    expected_sequence_length = expected_ground_height // patch_size
    _expect(
        int(column["input_dim"]) == 768
        and int(column["hidden_dim"]) == 256
        and int(column["content_dim"]) == 128
        and int(column["order_dim"]) == 128
        and float(column["joint_order_weight"]) == 0.2
        and int(column["ground_sequence_length"]) == expected_sequence_length
        and int(column["satellite_sequence_length"]) == expected_sequence_length
        and "variable_length_shared_encoder" not in column,
        "Content--Order encoder must share one 768-to-(128 Content + 128 Order) projection across "
        f"the configured Ground/Satellite length {expected_sequence_length}",
    )
    _expect(
        column["order_pooling"] == "normalized_first_cosine_moment"
        and column["joint_composition"] == "weighted_normalized_subspace_concat",
        "Order must use the signed first cosine moment and explicit normalized concat",
    )
    _expect(
        int(column["attention_queries"]) == 1
        and column["attention_scaling"] == "inverse_sqrt_hidden_dim"
        and column["shared_across_modalities"] is True,
        "Ground and satellite must share exactly one Content query and encoder",
    )
    _expect(
        column["ground_sequence_order"] == "bottom_to_top"
        and column["satellite_sampling"] == "dense_center_to_square_boundary_midpoint_bilinear_assignment",
        "Unexpected ordered sequence geometry",
    )
    _expect(int(architecture["direction_dim"]) == 256, "Direction dimension must be 256")
    finetuning = config.model["backbone"]["finetuning"]
    _expect(
        list(map(int, finetuning["trainable_blocks"])) == [8, 9, 10, 11],
        "Core must register DINO blocks 8 through 11",
    )
    starts = {int(index): int(epoch) for index, epoch in finetuning["block_update_start_epochs"].items()}
    _expect(
        starts == {8: 9, 9: 9, 10: 9, 11: 9},
        f"Unexpected DINO activation: {starts}",
    )
    _expect(
        int(finetuning["final_norm_update_start_epoch"]) == 9,
        "Shared DINO final norm must activate at epoch 9",
    )
    _expect(
        float(config.model["loss"]["label_smoothing"]) == 0.1,
        "Label smoothing must be 0.1",
    )
    _expect(
        float(config.model["loss"]["order_retrieval_weight"]) == 0.15,
        "Order-only cyclic retrieval loss weight must be 0.15",
    )
    score = config.model["score"]
    _expect(
        score["name"] == "fov_masked_cyclic_hard_max" and score.get("shift_reduction") == "hard_max",
        "CoR-Geo training, mining, and evaluation must use cyclic Hard Max",
    )
    _expect(
        set(score) == {"name", "shift_reduction"},
        "CoR-Geo scoring must contain only cyclic Hard-Max settings",
    )
    _expect(
        "ranking" not in config.model["loss"],
        "Top-K ranking loss has been removed from the active protocol",
    )
    expected_groups = {
        "content_order_encoder",
        "dinov2_block_8",
        "dinov2_block_9",
        "dinov2_block_10",
        "dinov2_block_11",
        "dinov2_final_norm",
    }
    _expect(
        set(config.train["optimizer"]["param_groups"]) == expected_groups,
        "Optimizer groups must match the active CoR-Geo modules exactly",
    )

    mixed = config.train["mixed_fov_batch"]
    fovs = list(map(int, mixed["fovs"]))
    _expect(fovs == [360, 180, 90, 70], f"Unexpected mixed FoVs: {fovs}")
    per_fov = int(mixed["samples_per_fov_per_global_batch"])
    _expect(per_fov == 16 and per_fov * len(fovs) == 64, "Mixed batch must be 16x4")
    _expect(
        mixed.get("samples_per_fov_per_rank") == "auto",
        "Per-rank FoV quota must be derived from torchrun world_size",
    )
    for supported_world_size in (1, 2):
        resolve_training_topology(config.train, supported_world_size)
    expected_bounds = [(1, 8), (9, 16), (17, 48), (49, 56), (57, 64)]
    stages = config.train["stages"]
    bounds = [(int(stage["epoch_start"]), int(stage["epoch_end"])) for stage in stages]
    _expect(bounds == expected_bounds, f"Unexpected stage boundaries: {bounds}")
    _expect(
        all(int(stage["steps_per_epoch"]) == 555 for stage in stages),
        f"{dataset_name.upper()} train requires 555 optimizer steps per epoch",
    )
    _expect(
        [float(stage["hard_batch_fraction"]) for stage in stages] == [0.0, 0.0, 0.25, 0.25, 0.25],
        "Clustered hard batches must activate at 25% only after epoch 16",
    )
    _expect(
        float(config.train["gradient"]["normal_batch_clip_norm"]) == 16.0
        and float(config.train["gradient"]["hard_batch_clip_norm"]) == 24.0
        and int(config.train["gradient"]["accumulation_steps"]) == 1,
        "CoR-Geo requires clip 16 for normal batches and 24 for hard batches",
    )
    hard = config.train["hard_mining"]
    _expect(hard.get("enabled") is True, "Minimal hard mining must be enabled")
    _expect(
        hard.get("pool_source") == "current_cor_geo_model" and "common_pool_run_root_template" not in hard,
        "Hard mining must be self-contained",
    )
    _expect(
        list(map(int, hard["refresh_after_epochs"])) == [16, 24, 32, 40, 48, 56],
        "Hard pools must refresh every eight epochs after epoch 16",
    )
    _expect(
        hard["candidate_strategy"] == "signed_dc_low2_fft_magnitude_top512_exact_cyclic_hardmax_top64_v1"
        and int(hard["coarse_frequency_count"]) == 3
        and int(hard["coarse_candidate_locations"]) == 512
        and int(hard["coarse_query_chunk_size"]) == 512
        and int(hard["rerank_query_chunk_size"]) == 32
        and int(hard["keep_negative_locations"]) == 64,
        "CoR-Geo requires low-frequency Top-512 screening and exact Hard-Max Top-64 retention",
    )
    _expect(
        hard["sampling_strategy"] == "anchor_plus_uniform_top64"
        and int(hard["hard_neighbors_per_anchor"]) == 15
        and list(map(int, hard["sampling_rank_range"])) == [1, 64]
        and hard["anchor_selection"] == "globally_unique_shuffle_no_pool_exclusion"
        and "independent_anchor_clusters_per_global_batch" not in hard
        and "clusters_per_fov_hard_batch" not in hard
        and "high_similarity_neighbors_per_cluster" not in hard
        and "random_tail_neighbors_per_cluster" not in hard
        and "high_similarity_rank_range" not in hard
        and "random_tail_rank_range" not in hard,
        "Each FoV hard subgroup must be one anchor plus 15 uniform Top-64 negatives",
    )

    _expect(
        config.evaluation.get("report_split") == "val",
        "The default report split must be validation",
    )
    _expect(
        list(map(int, config.evaluation["main_fovs"])) == fovs,
        "Evaluation must report all four FoVs",
    )
    metrics = config.evaluation["metrics"]
    _expect(metrics["r1_percent_rounding"] == "floor", "R@1% must use a floor threshold")

    scheduler = config.train["scheduler"]
    _expect(
        int(scheduler["content_order_encoder"]["cosine_end_global_step"]) == 48 * 555
        and int(scheduler["dinov2"]["active_optimizer_steps"]) == 40 * 555
        and int(scheduler["dinov2"]["cosine_end_global_step"]) == 48 * 555,
        "The primary cosine must preserve the configured epoch-1--48 schedule",
    )
    expected_overrides = [
        {
            "name": "epoch49_to56_low_lr_restart",
            "epoch_start": 49,
            "epoch_end": 56,
            "groups": {
                "content_order_encoder": {"start_lr": 2.0e-5, "end_lr": 2.0e-6},
                "dinov2_block_11": {"start_lr": 2.0e-6, "end_lr": 1.0e-6},
                "dinov2_block_10": {"start_lr": 1.0e-6, "end_lr": 5.0e-7},
                "dinov2_block_9": {"start_lr": 5.0e-7, "end_lr": 2.5e-7},
                "dinov2_block_8": {"start_lr": 2.5e-7, "end_lr": 1.25e-7},
                "dinov2_final_norm": {"start_lr": 2.0e-6, "end_lr": 1.0e-6},
            },
        },
        {
            "name": "conservative_epoch57_to64_refinement",
            "epoch_start": 57,
            "epoch_end": 64,
            "groups": {
                "content_order_encoder": {"start_lr": 1.0e-5, "end_lr": 1.0e-6},
                "dinov2_block_11": {"start_lr": 1.0e-6, "end_lr": 5.0e-7},
                "dinov2_block_10": {"start_lr": 5.0e-7, "end_lr": 2.5e-7},
                "dinov2_block_9": {"start_lr": 2.5e-7, "end_lr": 1.25e-7},
                "dinov2_block_8": {"start_lr": 1.25e-7, "end_lr": 6.25e-8},
                "dinov2_final_norm": {"start_lr": 1.0e-6, "end_lr": 5.0e-7},
            },
        },
    ]
    _expect(
        scheduler.get("late_cosine_overrides") == expected_overrides,
        "Unexpected epoch-49--64 low-learning-rate schedule",
    )
    _expect(
        list(map(int, config.train["checkpoint"]["retained_epochs"])) == [8, 16, 24, 32, 40, 48, 56, 64],
        "CoR-Geo must retain every configured eight-epoch checkpoint through epoch 64",
    )


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).expanduser() / candidate
    return candidate.resolve()
