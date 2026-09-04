import numpy as np
import torch

from cor_geo.engine.retrieval_evaluator import (
    EncodedGroundViews,
    EncodedSatelliteViews,
    _overwrite_positive_in_score_chunk,
    exact_evaluate_shard,
    score_aligned_candidates,
)
from cor_geo.losses.cyclic_matching import (
    cyclic_direction_logits,
    cyclic_hard_max_score,
)


def _score_config() -> dict[str, object]:
    return {
        "name": "fov_masked_cyclic_hard_max",
        "shift_reduction": "hard_max",
    }


def test_fft_correlation_matches_all_brute_force_shifts() -> None:
    generator = torch.Generator().manual_seed(7)
    ground = torch.randn(2, 36, 5, generator=generator)
    satellite = torch.randn(3, 36, 5, generator=generator)
    valid = torch.ones(2, 36, dtype=torch.bool)
    actual = cyclic_direction_logits(ground, satellite, valid)
    expected = torch.stack(
        [
            torch.einsum("qad,nad->qn", ground, torch.roll(satellite, -shift, dims=1))
            / 36
            for shift in range(36)
        ],
        dim=-1,
    )
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_location_score_is_exact_hard_max() -> None:
    shifts = torch.tensor(
        [
            [[-0.2, 0.3, 0.1], [0.4, 0.2, 0.35]],
            [[0.0, -0.1, -0.2], [0.5, 0.7, 0.6]],
        ]
    )
    actual = cyclic_hard_max_score(shifts)
    assert torch.equal(actual, shifts.max(dim=-1).values)


def test_hard_max_backpropagates_only_through_winning_shift() -> None:
    shifts = torch.tensor(
        [[[0.1, 0.7, 0.2], [0.8, 0.4, 0.3]]],
        requires_grad=True,
    )
    cyclic_hard_max_score(shifts).sum().backward()
    expected = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    assert torch.equal(shifts.grad, expected)


def test_candidate_rerank_uses_the_same_cyclic_score() -> None:
    generator = np.random.default_rng(9)
    query_direction = generator.normal(size=(2, 36, 4)).astype(np.float32)
    satellite_direction = generator.normal(size=(5, 36, 4)).astype(np.float32)
    query = EncodedGroundViews(
        query_direction,
        np.ones((2, 36), dtype=bool),
        ["0", "1"],
    )
    satellite = EncodedSatelliteViews(
        satellite_direction,
        [str(index) for index in range(5)],
    )
    candidates = np.asarray([[0, 3, 4], [1, 2, 4]], dtype=np.int64)
    config = _score_config()
    actual = score_aligned_candidates(
        query,
        satellite,
        candidates,
        torch.device("cpu"),
        query_batch_size=2,
        candidate_chunk_size=2,
        score_config=config,
    )
    for row in range(2):
        shifts = cyclic_direction_logits(
            torch.from_numpy(query_direction[row : row + 1]),
            torch.from_numpy(satellite_direction[candidates[row]]),
            torch.ones(1, 36, dtype=torch.bool),
        )
        expected = cyclic_hard_max_score(shifts)
        assert np.allclose(actual[row], expected.numpy()[0], atol=1.0e-5)


def test_exact_cyclic_evaluator_recovers_identity_top1() -> None:
    direction = np.zeros((4, 36, 4), dtype=np.float32)
    for index in range(4):
        direction[index, :, index] = 1.0
    ids = [f"id_{index}" for index in range(4)]
    valid = np.ones((4, 36), dtype=bool)
    satellite = EncodedSatelliteViews(direction, ids)
    query = EncodedGroundViews(
        direction.copy(),
        valid.copy(),
        ids,
    )
    predictions, metrics = exact_evaluate_shard(
        query,
        360,
        satellite,
        torch.device("cpu"),
        query_chunk_size=2,
        location_chunk_size=2,
        score_config=_score_config(),
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        crop_schedule_sha256="crop",
    )
    assert metrics["R@1"] == 1.0
    assert predictions["predicted_rank_1based"].tolist() == [1, 1, 1, 1]


def test_positive_scan_value_is_overwritten_before_rank_counting() -> None:
    positive_scores = torch.tensor([0.5, 0.7])
    scores = torch.tensor([[0.1, 0.50000006], [0.70000005, 0.2]])
    shifts = torch.zeros(2, 2, 4)
    positive_shifts = torch.ones(2, 4)
    _overwrite_positive_in_score_chunk(
        scores,
        shifts,
        positive_scores,
        positive_shifts,
        np.asarray([1, 0]),
        location_start=0,
        location_stop=2,
    )
    assert scores[0, 1].item() == positive_scores[0].item()
    assert scores[1, 0].item() == positive_scores[1].item()
    assert torch.count_nonzero(scores > positive_scores[:, None]).item() == 0


def test_exact_top1_and_rank_use_lexical_id_for_score_ties() -> None:
    direction = np.ones((2, 36, 3), dtype=np.float32)
    direction /= np.linalg.norm(direction, axis=2, keepdims=True)
    valid = np.ones((2, 36), dtype=bool)
    satellite = EncodedSatelliteViews(direction, ["z", "a"])
    query = EncodedGroundViews(
        direction.copy(),
        valid.copy(),
        ["z", "a"],
    )

    predictions, _ = exact_evaluate_shard(
        query,
        360,
        satellite,
        torch.device("cpu"),
        query_chunk_size=2,
        location_chunk_size=2,
        score_config=_score_config(),
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        crop_schedule_sha256="crop",
    )

    assert predictions["top1_satellite_id"].tolist() == ["a", "a"]
    assert predictions["predicted_rank_1based"].tolist() == [2, 1]
