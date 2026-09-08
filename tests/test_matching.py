"""Cyclic matching and hard-negative screening tests."""

from __future__ import annotations

import numpy as np
import torch

from cor_geo.matching import cyclic_direction_logits, cyclic_hard_max_score
from cor_geo.mining import rotation_invariant_fft_signature


def test_fft_correlation_matches_brute_force_shifts_with_a_mask() -> None:
    generator = torch.Generator().manual_seed(7)
    ground = torch.randn(2, 8, 5, generator=generator)
    satellite = torch.randn(3, 8, 5, generator=generator)
    valid = torch.tensor([[True] * 8, [True, True, True, False, False, False, False, False]])

    actual = cyclic_direction_logits(ground, satellite, valid)
    expected = []
    for shift in range(8):
        aligned = torch.roll(satellite, -shift, dims=1)
        score = torch.einsum("qad,nad->qna", ground, aligned)
        expected.append((score * valid[:, None]).sum(-1) / valid.sum(-1, keepdim=True))
    assert torch.allclose(actual, torch.stack(expected, dim=-1), atol=1e-6)


def test_hard_max_selects_and_differentiates_only_the_best_shift() -> None:
    shifts = torch.tensor([[[0.1, 0.7, 0.2], [0.8, 0.4, 0.3]]], requires_grad=True)
    scores = cyclic_hard_max_score(shifts)
    assert torch.equal(scores, torch.tensor([[0.7, 0.8]]))
    scores.sum().backward()
    assert torch.equal(shifts.grad, torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]))


def test_mining_signature_is_rotation_invariant() -> None:
    rng = np.random.default_rng(3)
    direction = torch.from_numpy(rng.normal(size=(4, 36, 16)).astype(np.float32))
    valid = torch.ones(4, 36, dtype=torch.bool)
    original, _ = rotation_invariant_fft_signature(direction, valid, frequency_count=3)
    shifted, _ = rotation_invariant_fft_signature(
        torch.roll(direction, 11, dims=1), valid, frequency_count=3
    )
    assert torch.allclose(original, shifted, atol=1e-5)
