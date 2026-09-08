"""Distributed symmetric retrieval objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as distributed
import torch.distributed.nn.functional as distributed_nn
from torch import Tensor, nn
from torch.nn import functional as functional

from cor_geo.matching import cyclic_direction_logits, cyclic_hard_max_score
from cor_geo.model import CoRGeoOutput

# ---- src/cor_geo/losses/info_nce.py ----

"""Content+Order hard-max retrieval without orientation supervision."""


def differentiable_all_gather(tensor: Tensor) -> Tensor:
    tensor = tensor.contiguous()
    if not distributed.is_available() or not distributed.is_initialized():
        return tensor
    return torch.cat(tuple(distributed_nn.all_gather(tensor)), dim=0)


def metadata_all_gather(tensor: Tensor) -> Tensor:
    tensor = tensor.contiguous()
    if not distributed.is_available() or not distributed.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(distributed.get_world_size())]
    distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def symmetric_cross_entropy(
    scores: Tensor,
    temperature: float,
    label_smoothing: float = 0.0,
) -> Tensor:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Symmetric InfoNCE needs a square matrix, got {tuple(scores.shape)}")
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive")
    targets = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (
        functional.cross_entropy(scores / float(temperature), targets, label_smoothing=float(label_smoothing))
        + functional.cross_entropy(scores.T / float(temperature), targets, label_smoothing=float(label_smoothing))
    )


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    joint_retrieval: Tensor
    order_retrieval: Tensor
    joint_in_batch_accuracy: Tensor
    order_in_batch_accuracy: Tensor


class CoRGeoLoss(nn.Module):
    """Train joint and Order-only cyclic retrieval."""

    def __init__(
        self,
        info_nce_temperature: float = 0.07,
        label_smoothing: float = 0.1,
        symmetric: bool = True,
        order_retrieval_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.info_nce_temperature = float(info_nce_temperature)
        self.label_smoothing = float(label_smoothing)
        self.symmetric = bool(symmetric)
        self.order_retrieval_weight = float(order_retrieval_weight)
        if not self.symmetric:
            raise ValueError("The configured Content+Order loss must remain symmetric")
        if self.order_retrieval_weight <= 0:
            raise ValueError("Order-retrieval weight must be positive")

    def forward(self, output: CoRGeoOutput) -> LossOutput:
        ground_direction = differentiable_all_gather(output.ground.direction)
        ground_order_direction = differentiable_all_gather(output.ground.order_direction)
        ground_valid = metadata_all_gather(output.ground.valid)
        satellite_direction = differentiable_all_gather(output.satellite.direction)
        satellite_order_direction = differentiable_all_gather(output.satellite.order_direction)
        shift_logits = cyclic_direction_logits(ground_direction, satellite_direction, ground_valid)
        scores = cyclic_hard_max_score(shift_logits)
        joint_retrieval = symmetric_cross_entropy(scores, self.info_nce_temperature, self.label_smoothing)
        order_shift_logits = cyclic_direction_logits(ground_order_direction, satellite_order_direction, ground_valid)
        order_scores = cyclic_hard_max_score(order_shift_logits)
        order_retrieval = symmetric_cross_entropy(order_scores, self.info_nce_temperature, self.label_smoothing)
        total = joint_retrieval + self.order_retrieval_weight * order_retrieval
        targets = torch.arange(scores.shape[0], device=scores.device)
        accuracy = (scores.argmax(dim=1) == targets).float().mean()
        order_accuracy = (order_scores.argmax(dim=1) == targets).float().mean()
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite CoR-Geo hard-max objective")
        return LossOutput(
            total=total,
            joint_retrieval=joint_retrieval.detach(),
            order_retrieval=order_retrieval.detach(),
            joint_in_batch_accuracy=accuracy.detach(),
            order_in_batch_accuracy=order_accuracy.detach(),
        )
