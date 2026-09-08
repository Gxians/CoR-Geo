"""Balanced multi-FoV and hard-negative batch planning."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

from cor_geo.datasets import SampleRequest
from cor_geo.utils import stable_seed

# ---- src/cor_geo/datasets/samplers.py ----

"""Deterministic balanced mixed-FoV batches with unique locations per epoch."""


@dataclass(frozen=True)
class PlannedBatch:
    """One globally unique balanced mixed-FoV batch."""

    kind: str
    indices: tuple[int, ...]
    fov_degrees: tuple[int, ...]
    hard_source_epoch: int | None = None


def stage_for_epoch(
    stages: Sequence[dict[str, Any]],
    epoch: int,
) -> dict[str, Any]:
    """Resolve the single configured stage containing an epoch."""
    matches = [stage for stage in stages if int(stage["epoch_start"]) <= int(epoch) <= int(stage["epoch_end"])]
    if len(matches) != 1:
        raise ValueError(f"Epoch {epoch} belongs to {len(matches)} stages")
    return matches[0]


def _shuffle(values: Sequence[Any], *seed_items: object) -> list[Any]:
    output = list(values)
    generator = np.random.default_rng(stable_seed(*seed_items))
    order = generator.permutation(len(output))
    return [output[int(index)] for index in order]


def _interleave_fov_groups(
    groups: dict[int, Sequence[int]],
    fovs: Sequence[int],
    *seed_items: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Interleave FoVs so every contiguous DDP rank shard is balanced."""
    group_sizes = {len(values) for values in groups.values()}
    if len(group_sizes) != 1:
        raise ValueError("Every FoV subgroup must contain the same number of rows")
    fov_cycle = _shuffle(list(map(int, fovs)), *seed_items, "fov_cycle")
    indices: list[int] = []
    assignments: list[int] = []
    for offset in range(next(iter(group_sizes))):
        for fov in fov_cycle:
            indices.append(int(groups[fov][offset]))
            assignments.append(int(fov))
    return tuple(indices), tuple(assignments)


def _hard_source_epoch(epoch: int, train_config: dict[str, Any]) -> int:
    eligible = [int(value) for value in train_config["hard_mining"]["refresh_after_epochs"] if int(value) < int(epoch)]
    if not eligible:
        raise ValueError(f"Epoch {epoch} has no earlier hard-mining refresh")
    return max(eligible)


def _simple_hard_anchors(
    manifest_size: int,
    hard_batch_count: int,
    fovs: Sequence[int],
    seed: int,
    epoch: int,
    dataset_name: str,
) -> tuple[dict[int, list[int]], set[int]]:
    """Choose unique anchors without pairwise candidate-pool exclusion checks."""
    reserved: dict[int, list[int]] = {int(fov): [] for fov in fovs}
    order = _shuffle(
        range(manifest_size),
        seed,
        dataset_name,
        "train",
        epoch,
        "streamlined_hard_anchors",
    )
    required = int(hard_batch_count) * len(fovs)
    if len(order) < required:
        raise RuntimeError("The manifest cannot provide enough unique hard anchors")
    cursor = 0
    for _batch_index in range(hard_batch_count):
        for fov in fovs:
            reserved[int(fov)].append(int(order[cursor]))
            cursor += 1
    used = {int(value) for value in order[:required]}
    return reserved, used


def _take_unique(
    candidates: Sequence[int],
    count: int,
    used: set[int],
) -> list[int]:
    """Take at most ``count`` candidates while preserving global uniqueness."""
    output: list[int] = []
    for value in candidates:
        index = int(value)
        if index in used:
            continue
        output.append(index)
        used.add(index)
        if len(output) == int(count):
            break
    return output


def _uniform_top64_hard_subgroups(
    pool: np.ndarray,
    anchors: Sequence[int],
    hard_batch_count: int,
    neighbors_per_anchor: int,
    rank_range: Sequence[int],
    used: set[int],
    seed: int,
    epoch: int,
    fov: int,
    dataset_name: str,
) -> list[tuple[int, ...]]:
    """Build one anchor plus a uniform sample from its exact Top-64 pool."""
    ranked_pool = np.asarray(pool)
    if ranked_pool.ndim != 2:
        raise ValueError("Hard pool must have shape [location, rank]")
    rank_start, rank_stop = map(int, rank_range)
    if not 1 <= rank_start <= rank_stop <= ranked_pool.shape[1]:
        raise ValueError("Uniform hard rank range must fit the stored pool")
    if len(anchors) != hard_batch_count:
        raise ValueError("Reserved hard-anchor count differs from the schedule")

    all_locations = list(range(len(ranked_pool)))
    output: list[tuple[int, ...]] = []
    for anchor in anchors:
        candidates = _shuffle(
            ranked_pool[int(anchor), rank_start - 1 : rank_stop].tolist(),
            seed,
            dataset_name,
            "train",
            epoch,
            fov,
            int(anchor),
            "uniform_exact_top64",
        )
        selected = _take_unique(candidates, neighbors_per_anchor, used)
        if len(selected) < neighbors_per_anchor:
            fallback = _shuffle(
                all_locations,
                seed,
                dataset_name,
                "train",
                epoch,
                fov,
                int(anchor),
                "manifest_fallback",
            )
            selected.extend(_take_unique(fallback, neighbors_per_anchor - len(selected), used))
        subgroup = [int(anchor), *selected]
        expected_size = 1 + neighbors_per_anchor
        if len(subgroup) != expected_size or len(set(subgroup)) != expected_size:
            raise RuntimeError("Hard FoV subgroup geometry is invalid")
        output.append(tuple(subgroup))
    return output


def build_epoch_plan(
    manifest: pd.DataFrame,
    epoch: int,
    global_seed: int,
    train_config: dict[str, Any],
    hard_pools: dict[int, np.ndarray] | None = None,
) -> list[PlannedBatch]:
    """Build unique-location normal/hard batches with a balanced FoV quota."""
    dataset_names = sorted(set(manifest["dataset"].astype(str))) if "dataset" in manifest.columns else ["cvact"]
    if len(dataset_names) != 1:
        raise ValueError(f"Epoch plan requires one dataset: {dataset_names}")
    dataset_name = dataset_names[0]
    global_batch = int(train_config["distributed"]["global_batch_size"])
    steps = len(manifest) // global_batch
    stage = stage_for_epoch(train_config["stages"], epoch)
    if int(stage["steps_per_epoch"]) != steps:
        raise ValueError("Stage steps do not match floor(manifest/global_batch)")
    mixed = train_config["mixed_fov_batch"]
    fovs = list(map(int, mixed["fovs"]))
    per_fov = int(mixed["samples_per_fov_per_global_batch"])
    if per_fov * len(fovs) != global_batch:
        raise ValueError("Mixed-FoV geometry differs from global batch size")

    hard_count = int(round(steps * float(stage["hard_batch_fraction"])))
    normal_count = steps - hard_count
    hard_source = _hard_source_epoch(epoch, train_config) if hard_count else None
    used: set[int] = set()
    hard_batches: list[PlannedBatch] = []
    if hard_count:
        if hard_pools is None or set(hard_pools) != set(fovs):
            raise FileNotFoundError("Every FoV hard pool is required for hard batches")
        hard = train_config["hard_mining"]
        neighbors_per_anchor = int(hard["hard_neighbors_per_anchor"])
        if 1 + neighbors_per_anchor != per_fov:
            raise ValueError("Streamlined hard-batch geometry differs from the FoV quota")
        anchors, used = _simple_hard_anchors(
            len(manifest),
            hard_count,
            fovs,
            global_seed,
            epoch,
            dataset_name,
        )
        subgroups = {
            fov: _uniform_top64_hard_subgroups(
                hard_pools[fov],
                anchors[fov],
                hard_count,
                neighbors_per_anchor,
                hard["sampling_rank_range"],
                used,
                global_seed,
                epoch,
                fov,
                dataset_name,
            )
            for fov in fovs
        }
        for batch_index in range(hard_count):
            indices, assignments = _interleave_fov_groups(
                {fov: subgroups[fov][batch_index] for fov in fovs},
                fovs,
                global_seed,
                dataset_name,
                "train",
                epoch,
                batch_index,
                "hard",
            )
            hard_batches.append(PlannedBatch("hard", indices, assignments, hard_source))

    available = _shuffle(
        [index for index in range(len(manifest)) if index not in used],
        global_seed,
        dataset_name,
        "train",
        int(epoch),
        "normal_locations",
    )
    needed = normal_count * global_batch
    if len(available) < needed:
        raise RuntimeError("Hard batches consumed too many locations")
    normal_batches: list[PlannedBatch] = []
    cursor = 0
    for batch_index in range(normal_count):
        groups: dict[int, Sequence[int]] = {}
        for fov in fovs:
            groups[fov] = available[cursor : cursor + per_fov]
            cursor += per_fov
        indices, assignments = _interleave_fov_groups(
            groups,
            fovs,
            global_seed,
            dataset_name,
            "train",
            int(epoch),
            batch_index,
            "normal",
        )
        normal_batches.append(PlannedBatch("normal", indices, assignments))

    plan = _shuffle(
        [*normal_batches, *hard_batches],
        global_seed,
        dataset_name,
        "train",
        int(epoch),
        "mixed_batch_schedule",
    )
    flattened = [index for batch in plan for index in batch.indices]
    if len(plan) != steps or len(flattened) != len(set(flattened)):
        raise RuntimeError("Epoch plan reused a location or has the wrong length")
    for batch in plan:
        counts = {fov: batch.fov_degrees.count(fov) for fov in fovs}
        if counts != {fov: per_fov for fov in fovs}:
            raise RuntimeError(f"Mixed batch is not FoV-balanced: {counts}")
    return plan


class PlannedRankBatchSampler(Sampler[list[SampleRequest]]):
    """Yield one balanced contiguous rank shard from the global plan."""

    def __init__(self, rank: int, world_size: int, per_rank_batch_size: int) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.plan: list[PlannedBatch] = []
        self.epoch = 0

    def set_plan(self, plan: Sequence[PlannedBatch], epoch: int) -> None:
        expected = self.world_size * self.per_rank_batch_size
        if any(len(batch.indices) != expected or len(batch.fov_degrees) != expected for batch in plan):
            raise ValueError("Global plan does not match DDP geometry")
        self.plan = list(plan)
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[SampleRequest]]:
        start = self.rank * self.per_rank_batch_size
        stop = start + self.per_rank_batch_size
        for batch in self.plan:
            orientation_mode = "hard_train" if batch.kind == "hard" else "train"
            requests = [
                SampleRequest(
                    index=int(index),
                    epoch=self.epoch,
                    fov_deg=int(fov),
                    orientation_mode=orientation_mode,
                    source_epoch=batch.hard_source_epoch,
                )
                for index, fov in zip(
                    batch.indices[start:stop],
                    batch.fov_degrees[start:stop],
                    strict=True,
                )
            ]
            supported = tuple(sorted(set(batch.fov_degrees), reverse=True))
            expected_per_fov = self.per_rank_batch_size // len(supported)
            counts = {fov: sum(request.fov_deg == fov for request in requests) for fov in supported}
            if counts != {fov: expected_per_fov for fov in supported}:
                raise RuntimeError(f"Rank-local mixed batch is not balanced: {counts}")
            yield requests

    def __len__(self) -> int:
        return len(self.plan)
