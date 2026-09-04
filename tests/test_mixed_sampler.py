from copy import deepcopy

import numpy as np
import pandas as pd

from cor_geo.datasets.samplers import (
    PlannedRankBatchSampler,
    _hard_source_epoch,
    build_epoch_plan,
)


def _manifest() -> pd.DataFrame:
    ids = [f"id_{index:05d}" for index in range(35531)]
    return pd.DataFrame({"query_id": ids, "satellite_id": ids})


def test_every_batch_and_rank_are_four_fov_balanced(train_config: dict) -> None:
    plan = build_epoch_plan(_manifest(), 1, 42, train_config)
    assert len(plan) == 555
    for batch in plan[:10]:
        assert {fov: batch.fov_degrees.count(fov) for fov in (360, 180, 90, 70)} == {
            360: 16,
            180: 16,
            90: 16,
            70: 16,
        }
    sampler = PlannedRankBatchSampler(0, 2, 32)
    sampler.set_plan(plan[:1], 1)
    requests = next(iter(sampler))
    assert {fov: sum(row.fov_deg == fov for row in requests) for fov in (360, 180, 90, 70)} == {
        360: 8,
        180: 8,
        90: 8,
        70: 8,
    }
    used = [index for batch in plan for index in batch.indices]
    assert len(used) == len(set(used)) == 555 * 64


def test_epoch_plan_is_repeatable_but_changes_between_epochs(train_config: dict) -> None:
    manifest = _manifest()
    epoch_one = build_epoch_plan(manifest, 1, 42, train_config)
    repeated = build_epoch_plan(manifest, 1, 42, train_config)
    epoch_two = build_epoch_plan(manifest, 2, 42, train_config)
    assert epoch_one == repeated
    assert epoch_one != epoch_two


def test_top64_refresh_and_sampling_contract(train_config: dict) -> None:
    hard = train_config["hard_mining"]
    assert hard["refresh_after_epochs"] == [16, 24, 32, 40, 48, 56]
    assert hard["keep_negative_locations"] == 64
    assert hard["sampling_strategy"] == "anchor_plus_uniform_top64"
    assert hard["hard_neighbors_per_anchor"] == 15
    assert hard["sampling_rank_range"] == [1, 64]
    assert hard["anchor_selection"] == "globally_unique_shuffle_no_pool_exclusion"
    assert _hard_source_epoch(17, train_config) == 16
    assert _hard_source_epoch(25, train_config) == 24
    assert _hard_source_epoch(33, train_config) == 32
    assert _hard_source_epoch(41, train_config) == 40
    assert _hard_source_epoch(49, train_config) == 48
    assert _hard_source_epoch(57, train_config) == 56


def test_light_hard_batches_are_unique_and_replay_source_crop(train_config: dict) -> None:
    config = deepcopy(train_config)
    config["stages"][2]["steps_per_epoch"] = 10
    manifest_size = 10 * 64
    manifest = pd.DataFrame(
        {
            "query_id": [f"q_{index}" for index in range(manifest_size)],
            "satellite_id": [f"s_{index}" for index in range(manifest_size)],
        }
    )
    offsets = np.arange(1, 65, dtype=np.int32)[None, :]
    anchors = np.arange(manifest_size, dtype=np.int32)[:, None]
    pool = (anchors + offsets) % manifest_size
    plan = build_epoch_plan(
        manifest,
        17,
        42,
        config,
        {fov: pool for fov in (360, 180, 90, 70)},
    )
    assert sum(batch.kind == "hard" for batch in plan) == 2
    assert len({index for batch in plan for index in batch.indices}) == manifest_size
    hard = next(batch for batch in plan if batch.kind == "hard")
    sampler = PlannedRankBatchSampler(0, 2, 32)
    sampler.set_plan([hard], 17)
    requests = next(iter(sampler))
    assert all(request.orientation_mode == "hard_train" for request in requests)
    assert all(request.source_epoch == 16 for request in requests)
    for fov in (360, 180, 90, 70):
        rows = [
            index
            for index, assigned_fov in zip(
                hard.indices,
                hard.fov_degrees,
                strict=True,
            )
            if assigned_fov == fov
        ]
        assert len(rows) == len(set(rows)) == 16
