from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cor_geo.evaluation_protocol import (
    draw_random_crop_schedule,
    load_random_crop_schedule,
    validate_random_crop_schedule,
)

FOVS = (360, 180, 90, 70)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame({"query_id": ["q0", "q1", "q2"]})


def test_seeded_draw_is_reproducible_and_independent() -> None:
    first = draw_random_crop_schedule(
        _manifest(),
        "val",
        FOVS,
        np.random.default_rng(7),
    )
    second = draw_random_crop_schedule(
        _manifest(),
        "val",
        FOVS,
        np.random.default_rng(7),
    )
    assert first[2] == second[2]
    pd.testing.assert_frame_equal(first[1], second[1])
    assert set(first[0]) == set(FOVS)
    assert any(not np.array_equal(first[0][360], first[0][fov]) for fov in FOVS[1:])


def test_recorded_schedule_round_trip(tmp_path) -> None:
    expected = draw_random_crop_schedule(
        _manifest(),
        "val",
        FOVS,
        np.random.default_rng(11),
    )
    path = tmp_path / "schedule.parquet"
    expected[1].to_parquet(path, index=False, engine="pyarrow")
    loaded = load_random_crop_schedule(path, _manifest(), "val", FOVS)
    assert loaded[2] == expected[2]
    pd.testing.assert_frame_equal(loaded[1], expected[1])


@pytest.mark.parametrize("bad_value", [-1, 360, 1.5, np.nan])
def test_invalid_angles_are_rejected(bad_value: float) -> None:
    _, schedule, _ = draw_random_crop_schedule(
        _manifest(),
        "val",
        FOVS,
        np.random.default_rng(5),
    )
    schedule["roll_angle_deg_90"] = schedule["roll_angle_deg_90"].astype(float)
    schedule.loc[0, "roll_angle_deg_90"] = bad_value
    with pytest.raises(ValueError):
        validate_random_crop_schedule(schedule, _manifest(), "val", FOVS)


def test_query_order_is_strict() -> None:
    _, schedule, _ = draw_random_crop_schedule(
        _manifest(),
        "val",
        FOVS,
        np.random.default_rng(3),
    )
    schedule.loc[0, "query_id"] = "wrong"
    with pytest.raises(ValueError, match="query order"):
        validate_random_crop_schedule(schedule, _manifest(), "val", FOVS)
