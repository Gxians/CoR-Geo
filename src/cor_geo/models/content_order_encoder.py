"""Content-and-order per-direction encoding without discrete radial slots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


class ConservativeAngularResampler(nn.Module):
    """Resample horizontal patch columns by normalized interval overlap.

    The operator is fixed and parameter-free.  Each target direction is the
    area-weighted average over its source interval, so all source evidence is
    retained while different input resolutions share one canonical azimuth
    grid.
    """

    def __init__(self, source_bins: int, target_bins: int) -> None:
        super().__init__()
        if source_bins <= 0 or target_bins <= 0:
            raise ValueError("Angular bin counts must be positive")
        self.source_bins = int(source_bins)
        self.target_bins = int(target_bins)
        source_left = torch.arange(source_bins, dtype=torch.float64) / source_bins
        source_right = torch.arange(1, source_bins + 1, dtype=torch.float64) / source_bins
        target_left = torch.arange(target_bins, dtype=torch.float64) / target_bins
        target_right = torch.arange(1, target_bins + 1, dtype=torch.float64) / target_bins
        overlap = torch.minimum(target_right[:, None], source_right[None, :]) - torch.maximum(
            target_left[:, None], source_left[None, :]
        )
        assignment = overlap.clamp_min(0.0) * target_bins
        if not torch.allclose(
            assignment.sum(dim=1),
            torch.ones(target_bins, dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError("Angular interval weights do not conserve unit mass")
        self.register_buffer("assignment", assignment.float(), persistent=True)

    def forward(self, features: Tensor) -> Tensor:
        """Return BHWC features on the configured canonical horizontal grid."""
        if features.ndim != 4 or features.shape[2] != self.source_bins:
            raise ValueError(
                f"Expected BH{self.source_bins}C source features, got {tuple(features.shape)}"
            )
        assignment = self.assignment.to(device=features.device, dtype=features.dtype)
        return torch.einsum("bhwc,aw->bhac", features, assignment)


class BilinearSquareRaySampler(nn.Module):
    """Sample a Cartesian feature grid into bilinear center-to-boundary rays.

    Every angular ray uses the same number of midpoint samples.  Its maximum
    radius is the intersection with the square boundary, so no samples fall
    outside the DINO patch grid and no four-bin radial assignment is required.
    """

    def __init__(
        self,
        input_size: int = 24,
        angular_bins: int = 36,
        sequence_length: int = 16,
    ) -> None:
        super().__init__()
        if input_size <= 1 or angular_bins <= 1 or sequence_length <= 1:
            raise ValueError("Dense square-ray sampling requires all dimensions > 1")
        self.input_size = int(input_size)
        self.angular_bins = int(angular_bins)
        self.sequence_length = int(sequence_length)
        theta = torch.arange(angular_bins, dtype=torch.float64) * (
            2.0 * math.pi / angular_bins
        )
        sin_theta = theta.sin()
        cos_theta = theta.cos()
        center = (input_size - 1) / 2.0
        ray_limit = center / torch.maximum(sin_theta.abs(), cos_theta.abs())
        # Midpoints avoid repeating the ambiguous center token on all 36 rays.
        fraction = (
            torch.arange(sequence_length, dtype=torch.float64) + 0.5
        ) / sequence_length
        radius = ray_limit[:, None] * fraction[None, :]
        x = center + sin_theta[:, None] * radius
        y = center - cos_theta[:, None] * radius
        grid = torch.stack(
            (
                2.0 * x / (input_size - 1) - 1.0,
                2.0 * y / (input_size - 1) - 1.0,
            ),
            dim=-1,
        )
        if torch.any(grid < -1.0 - 1.0e-9) or torch.any(grid > 1.0 + 1.0e-9):
            raise RuntimeError("Square-ray grid escaped the normalized image boundary")
        flat_x = x.reshape(-1).clamp(0.0, input_size - 1.0)
        flat_y = y.reshape(-1).clamp(0.0, input_size - 1.0)
        x0 = flat_x.floor().long()
        y0 = flat_y.floor().long()
        x1 = (x0 + 1).clamp_max(input_size - 1)
        y1 = (y0 + 1).clamp_max(input_size - 1)
        wx = flat_x - x0.to(flat_x.dtype)
        wy = flat_y - y0.to(flat_y.dtype)
        rows = torch.arange(angular_bins * sequence_length)
        assignment = torch.zeros(
            angular_bins * sequence_length,
            input_size * input_size,
            dtype=torch.float64,
        )
        for yy, xx, weight in (
            (y0, x0, (1.0 - wy) * (1.0 - wx)),
            (y0, x1, (1.0 - wy) * wx),
            (y1, x0, wy * (1.0 - wx)),
            (y1, x1, wy * wx),
        ):
            assignment.index_put_(
                (rows, yy * input_size + xx),
                weight,
                accumulate=True,
            )
        if not torch.allclose(
            assignment.sum(dim=1),
            torch.ones(angular_bins * sequence_length, dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError("Bilinear ray weights do not conserve unit mass")
        self.register_buffer("sampling_grid", grid.float(), persistent=True)
        self.register_buffer("assignment", assignment.float(), persistent=True)

    def forward(self, features: Tensor) -> Tensor:
        """Return ``[B,A,L,C]`` rays ordered from center to image boundary."""
        if features.ndim != 4:
            raise ValueError(f"Expected BHWC satellite features, got {tuple(features.shape)}")
        batch, height, width, _ = features.shape
        if (height, width) != (self.input_size, self.input_size):
            raise ValueError(
                f"Satellite token grid must be {self.input_size}x{self.input_size}, "
                f"got {height}x{width}"
            )
        assignment = self.assignment.to(device=features.device, dtype=features.dtype)
        sampled = torch.einsum(
            "bpc,qp->bqc",
            features.reshape(batch, height * width, -1),
            assignment,
        )
        return sampled.reshape(
            batch,
            self.angular_bins,
            self.sequence_length,
            features.shape[-1],
        )


@dataclass(frozen=True)
class ContentOrderEncoding:
    """Joint descriptor and its Content and Order subspaces."""

    direction: Tensor
    content_direction: Tensor
    order_direction: Tensor
    content_attention: Tensor | None = None


class ContentOrderEncoder(nn.Module):
    """Encode content and first-order position in separate channel subspaces.

    The shared 768-to-256 projection is split into two 128-D halves.  Content
    is pooled by one permutation-invariant attention query.  Order is the
    signed first cosine moment of the sequence, so reversing a column reverses
    its order descriptor.  The normalized halves are concatenated with a
    fixed similarity contribution; no convolution, gate, radial slot, or
    modality-specific parameter is retained.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        content_dim: int = 128,
        order_dim: int = 128,
        joint_order_weight: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or content_dim <= 0 or order_dim <= 0:
            raise ValueError("Content/Order dimensions must be positive")
        if content_dim + order_dim != hidden_dim:
            raise ValueError("content_dim + order_dim must equal hidden_dim")
        if not 0.0 < joint_order_weight < 1.0:
            raise ValueError("joint_order_weight must be strictly between zero and one")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.content_dim = int(content_dim)
        self.order_dim = int(order_dim)
        self.joint_order_weight = float(joint_order_weight)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=False),
        )
        self.pool_query = nn.Parameter(torch.empty(content_dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    @staticmethod
    def first_cosine_weights(length: int, device: torch.device) -> Tensor:
        """Return a zero-mean, L1-normalized first cosine basis."""
        if length <= 1:
            raise ValueError("Order pooling requires a sequence length greater than one")
        positions = torch.arange(length, device=device, dtype=torch.float32)
        weights = torch.cos(math.pi * (positions + 0.5) / float(length))
        weights = weights - weights.mean()
        return weights / weights.abs().sum().clamp_min(1.0e-12)

    def forward(
        self,
        sequences: Tensor,
        return_attention: bool = False,
    ) -> ContentOrderEncoding:
        """Encode ``[B,A,L,768]`` ordered sequences into ``[B,A,256]``."""
        if sequences.ndim != 4 or sequences.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [B,A,L,{self.input_dim}] sequences, got {tuple(sequences.shape)}"
            )
        projected = self.projection(sequences)
        content_tokens, order_tokens = projected.split(
            (self.content_dim, self.order_dim), dim=-1
        )
        logits = torch.einsum(
            "bald,d->bal",
            content_tokens.float(),
            self.pool_query.float(),
        ) / math.sqrt(self.content_dim)
        attention = torch.softmax(logits, dim=-1)
        content_raw = torch.einsum(
            "bal,bald->bad", attention.to(content_tokens.dtype), content_tokens
        )
        order_weights = self.first_cosine_weights(
            sequences.shape[2], sequences.device
        ).to(order_tokens.dtype)
        order_raw = torch.einsum("l,bald->bad", order_weights, order_tokens)
        content = functional.normalize(content_raw.float(), dim=-1).to(projected.dtype)
        order = functional.normalize(order_raw.float(), dim=-1).to(projected.dtype)
        content_scale = math.sqrt(1.0 - self.joint_order_weight)
        order_scale = math.sqrt(self.joint_order_weight)
        direction = torch.cat(
            (content * content_scale, order * order_scale), dim=-1
        )
        return ContentOrderEncoding(
            direction=direction,
            content_direction=content,
            order_direction=order,
            content_attention=attention if return_attention else None,
        )
