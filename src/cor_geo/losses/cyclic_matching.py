"""Unified hard-max cyclic location scoring for every FoV."""

from __future__ import annotations

import torch
from torch import Tensor


def cyclic_direction_logits(
    ground_direction: Tensor,
    satellite_direction: Tensor,
    ground_valid: Tensor,
) -> Tensor:
    """Return [Q,N,A] correlations for every circular satellite shift."""
    if ground_direction.ndim != 3 or satellite_direction.ndim != 3:
        raise ValueError("Direction descriptors must have shape [B,A,D]")
    if ground_direction.shape[1:] != satellite_direction.shape[1:]:
        raise ValueError("Ground and satellite direction geometry differs")
    if ground_valid.shape != ground_direction.shape[:2]:
        raise ValueError("Ground validity mask has the wrong shape")
    masked_ground = ground_direction * ground_valid[..., None].to(ground_direction.dtype)
    ground_fft = torch.fft.rfft(masked_ground.float(), dim=1)
    satellite_fft = torch.fft.rfft(satellite_direction.float(), dim=1)
    cross_spectrum = torch.einsum(
        "qfd,nfd->qnf",
        torch.conj(ground_fft),
        satellite_fft,
    )
    correlation = torch.fft.irfft(cross_spectrum, n=ground_direction.shape[1], dim=-1)
    normalizer = ground_valid.sum(dim=1).clamp_min(1).to(correlation.dtype)
    return correlation / normalizer[:, None, None]


def cyclic_hard_max_score(shift_logits: Tensor) -> Tensor:
    """Select the best latent circular alignment for every location pair."""
    if shift_logits.ndim not in {2, 3}:
        raise ValueError("Shift logits must have shape [Q,A] or [Q,N,A]")
    return shift_logits.max(dim=-1).values
