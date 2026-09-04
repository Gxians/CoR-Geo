import pytest

from cor_geo.datasets.panorama_crop import (
    orientation_to_center_px,
    random_crop_spec,
)
from cor_geo.metrics.retrieval import retrieval_metrics


def test_random_crop_matches_roll_then_left_crop() -> None:
    roll_angle = 137
    panorama_width = 504
    expected_start = (-(roll_angle * panorama_width // 360)) % panorama_width
    crops = {
        fov: random_crop_spec(roll_angle, fov, panorama_width)
        for fov in (360, 180, 90, 70)
    }
    assert [crops[fov].width for fov in (360, 180, 90, 70)] == [504, 252, 126, 98]
    for crop in crops.values():
        start, _ = crop.interval
        assert start % panorama_width == expected_start
        assert (
            orientation_to_center_px(crop.orientation_u32, panorama_width)
            == crop.center_px
        )


def test_zero_roll_full_panorama_is_unchanged() -> None:
    crop = random_crop_spec(0, 360, 504)
    assert crop.width == 504
    assert crop.center_px == 252
    assert crop.interval == (0, 504)


def test_random_crop_supports_unseen_integer_width_fovs() -> None:
    roll_angle = 137
    panorama_width = 756
    expected_start = (-(roll_angle * panorama_width // 360)) % panorama_width
    crops = {
        fov: random_crop_spec(roll_angle, fov, panorama_width)
        for fov in (270, 120)
    }
    assert [crops[fov].width for fov in (270, 120)] == [567, 252]
    assert all(crop.interval[0] % panorama_width == expected_start for crop in crops.values())


def test_random_crop_rejects_fractional_pixel_width() -> None:
    with pytest.raises(ValueError, match="exact integer width"):
        random_crop_spec(0, 125, 756)


def test_r1_percent_keeps_floor_and_ceil_for_audit() -> None:
    metrics = retrieval_metrics(
        [88, 89, 90],
        database_size=8884,
        recall_ks=(1, 5, 10),
        r1_percent_rounding="floor",
    )
    assert metrics["R@1%_threshold_floor"] == 88
    assert metrics["R@1%_threshold_ceil"] == 89
    assert metrics["R@1%"] == pytest.approx(1 / 3)
