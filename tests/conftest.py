"""Lightweight DINO-compatible fixtures for CoR-Geo tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from cor_geo.config import load_yaml


class ScaleBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * (1.0 + self.scale) + self.bias


class TinyDINO(nn.Module):
    def __init__(self, dimension: int = 768, patch_size: int = 14) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3,
            dimension,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.blocks = nn.ModuleList([ScaleBlock(dimension) for _ in range(12)])
        self.norm = nn.LayerNorm(dimension)
        self.num_register_tokens = 0

    def prepare_tokens_with_masks(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del masks
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls, patches), dim=1)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.prepare_tokens_with_masks(images)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return {
            "x_norm_clstoken": tokens[:, 0],
            "x_norm_patchtokens": tokens[:, 1:],
        }


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def model_config(project_root: Path) -> dict:
    return load_yaml(project_root / "configs" / "model.yaml")


@pytest.fixture
def train_config(project_root: Path) -> dict:
    return load_yaml(project_root / "configs" / "train_cvact.yaml")
