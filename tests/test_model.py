"""Core model tests that do not require the external DINOv2 checkout."""

from __future__ import annotations

import torch
from torch import nn

from cor_geo.content_order import (
    BilinearSquareRaySampler,
    ConservativeAngularResampler,
    ContentOrderEncoder,
)
from cor_geo.losses import CoRGeoLoss
from cor_geo.model import CoRGeoModel


class _ScaleBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dimension))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * (1.0 + self.scale)


class _TinyDINO(nn.Module):
    def __init__(self, dimension: int = 16, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dimension, patch_size, patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.blocks = nn.ModuleList([_ScaleBlock(dimension) for _ in range(4)])
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
        return {"x_norm_patchtokens": tokens[:, 1:]}


def _tiny_config() -> dict:
    return {
        "backbone": {
            "name": "tiny_dino",
            "patch_size": 2,
            "output_dim": 16,
            "strict_checkpoint": True,
            "expected_transformer_blocks": 4,
            "finetuning": {
                "update_start_epoch": 2,
                "trainable_blocks": [2, 3],
                "block_update_start_epochs": {2: 2, 3: 2},
                "train_final_norm": True,
                "final_norm_update_start_epoch": 2,
            },
        },
        "input": {
            "ground_height": 8,
            "widths": {360: 16, 180: 8, 90: 4, 70: 2},
            "satellite_size": [6, 6],
        },
        "architecture": {
            "name": "column_ray_content_order",
            "angular_bins": 8,
            "direction_dim": 8,
            "ground_angular_resampling": {
                "source_patch_columns": {360: 8, 180: 4, 90: 2, 70: 1},
                "target_direction_bins": {360: 8, 180: 4, 90: 2, 70: 1},
            },
            "content_order_encoder": {
                "hidden_dim": 8,
                "content_dim": 4,
                "order_dim": 4,
                "joint_order_weight": 0.2,
                "ground_sequence_length": 4,
                "satellite_sequence_length": 4,
                "shared_across_modalities": True,
            },
        },
        "score": {"name": "fov_masked_cyclic_hard_max", "shift_reduction": "hard_max"},
    }


def test_sequence_construction_preserves_geometry() -> None:
    resampler = ConservativeAngularResampler(8, 4)
    constant = torch.ones(2, 4, 8, 3, requires_grad=True)
    output = resampler(constant)
    assert output.shape == (2, 4, 4, 3)
    assert torch.allclose(output, torch.ones_like(output))
    output.mean().backward()
    assert constant.grad is not None

    rays = BilinearSquareRaySampler(input_size=3, angular_bins=8, sequence_length=4)
    sampled = rays(torch.ones(2, 3, 3, 5))
    assert sampled.shape == (2, 8, 4, 5)
    assert torch.allclose(sampled, torch.ones_like(sampled))
    assert bool((rays.sampling_grid.abs() <= 1).all())


def test_content_order_encoding_is_normalized_and_order_sensitive() -> None:
    encoder = ContentOrderEncoder(16, 8, content_dim=4, order_dim=4, joint_order_weight=0.2)
    sequences = torch.randn(2, 5, 4, 16)
    forward = encoder(sequences, return_attention=True)
    reverse = encoder(sequences.flip(2))

    assert forward.direction.shape == (2, 5, 8)
    assert torch.allclose(forward.direction.norm(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.allclose(forward.content_attention.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.allclose(forward.content_direction, reverse.content_direction, atol=1e-5)
    assert torch.allclose(forward.order_direction, -reverse.order_direction, atol=1e-5)


def test_model_and_loss_follow_the_mixed_fov_contract() -> None:
    config = _tiny_config()
    model = CoRGeoModel(config, backbone_model=_TinyDINO())
    ground = {fov: torch.randn(1, 3, 8, width) for fov, width in config["input"]["widths"].items()}
    positions = {fov: torch.tensor([index]) for index, fov in enumerate(ground)}
    satellite = torch.randn(4, 3, 6, 6)

    model.eval()
    output = model(ground, positions, satellite)
    assert output.ground.direction.shape == (4, 8, 8)
    assert output.ground.valid.sum(dim=1).tolist() == [8, 4, 2, 1]
    assert output.satellite.direction.shape == (4, 8, 8)

    result = CoRGeoLoss(order_retrieval_weight=0.15)(output)
    assert torch.allclose(
        result.total,
        result.joint_retrieval + 0.15 * result.order_retrieval,
    )


def test_dino_suffix_activates_only_at_its_scheduled_epoch() -> None:
    model = CoRGeoModel(_tiny_config(), backbone_model=_TinyDINO())
    model.set_train_epoch(1)
    assert model.backbone.active_indices == ()
    model.set_train_epoch(2)
    assert model.backbone.active_indices == (2, 3)
    assert all(not parameter.requires_grad for block in model.backbone.blocks[:2] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in model.backbone.blocks[2:] for parameter in block.parameters())
