#!/usr/bin/env python
"""Verify CoR-Geo shapes, gradients, losses, and GPU memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_yaml  # noqa: E402
from cor_geo.datasets.manifests import parquet_row_count  # noqa: E402
from cor_geo.engine.stage_scheduler import GroupCosineScheduler, build_optimizer  # noqa: E402
from cor_geo.losses.info_nce import CoRGeoLoss  # noqa: E402
from cor_geo.models.cor_geo_model import CoRGeoModel  # noqa: E402
from cor_geo.reproducibility import configure_determinism  # noqa: E402


def _gradient_state(module: torch.nn.Module) -> str:
    gradients = [parameter.grad for parameter in module.parameters()]
    if not gradients or all(value is None for value in gradients):
        return "all_none"
    tensors = [value for value in gradients if value is not None]
    if not all(torch.isfinite(value).all() for value in tensors):
        return "non_finite"
    if sum(float(value.detach().float().abs().sum()) for value in tensors) == 0.0:
        return "finite_zero"
    return "finite_nonzero"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--per-gpu-batch-size", type=int, required=True)
    parser.add_argument("--full-train-step", action="store_true", required=True)
    parser.add_argument(
        "--simulate-epoch",
        type=int,
        choices=[1, 9, 16, 17, 32, 48, 49, 56, 57, 64],
        required=True,
    )
    args = parser.parse_args()
    if args.per_gpu_batch_size % 4:
        raise ValueError("Per-GPU batch size must be divisible by four FoVs")
    model_config = load_yaml(args.model_config)
    train_config = load_yaml(args.train_config)
    paths = load_yaml(args.paths)
    configure_determinism(int(train_config["seed"]))
    for path in (
        Path(paths["dinov2_root"]),
        Path(paths["checkpoints"]["dinov2_vitb14"]),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = Path(paths["project_root"]) / "data_manifests" / "cvact" / "train.parquet"
    steps_per_epoch = parquet_row_count(manifest) // int(
        train_config["distributed"]["global_batch_size"]
    )
    if steps_per_epoch != 555:
        raise ValueError(f"Unexpected steps per epoch: {steps_per_epoch}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    model = CoRGeoModel(
        model_config,
        dinov2_root=paths["dinov2_root"],
        checkpoint_path=paths["checkpoints"]["dinov2_vitb14"],
    ).to(device)
    model.train(True)
    model.set_train_epoch(args.simulate_epoch)
    optimizer = build_optimizer(model, train_config)
    scheduler = GroupCosineScheduler(
        optimizer,
        train_config,
        steps_per_epoch,
        global_step=(args.simulate_epoch - 1) * steps_per_epoch,
    )
    scheduler.set_for_next_step()
    fovs = (360, 180, 90, 70)
    rows_per_fov = args.per_gpu_batch_size // 4
    widths = {int(key): int(value) for key, value in model_config["input"]["widths"].items()}
    ground_by_fov = {
        fov: torch.randn(
            rows_per_fov,
            3,
            int(model_config["input"]["ground_height"]),
            widths[fov],
            device=device,
        )
        for fov in fovs
    }
    positions_by_fov = {
        fov: torch.arange(index, args.per_gpu_batch_size, 4, device=device)
        for index, fov in enumerate(fovs)
    }
    satellite_size = int(model_config["input"]["satellite_size"][0])
    satellite = torch.randn(
        args.per_gpu_batch_size,
        3,
        satellite_size,
        satellite_size,
        device=device,
    )
    fov_tensor = torch.tensor(
        [fov for _ in range(rows_per_fov) for fov in fovs],
        dtype=torch.int64,
        device=device,
    )
    loss_config = model_config["loss"]
    loss_function = CoRGeoLoss(
        info_nce_temperature=float(loss_config["info_nce_temperature"]),
        label_smoothing=float(loss_config["label_smoothing"]),
        symmetric=bool(loss_config["symmetric"]),
        order_retrieval_weight=float(loss_config["order_retrieval_weight"]),
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            ground_by_fov,
            positions_by_fov,
            satellite,
        )
        losses = loss_function(output)
    losses.total.backward()
    active_suffix_expected = (
        "all_none" if args.simulate_epoch < 9 else "finite_nonzero"
    )
    result = {
        "mixed_fov_rows_per_gpu": {str(fov): rows_per_fov for fov in fovs},
        "ground_source_patch_columns": {
            str(fov): int(widths[fov] // int(model_config["backbone"]["patch_size"]))
            for fov in fovs
        },
        "ground_valid_direction_counts": {
            str(fov): int(
                output.ground.valid[fov_tensor == fov].sum(dim=1).unique().item()
            )
            for fov in fovs
        },
        "ground_direction_shape": list(output.ground.direction.shape),
        "satellite_direction_shape": list(output.satellite.direction.shape),
        "ground_order_direction_shape": list(output.ground.order_direction.shape),
        "satellite_order_direction_shape": list(output.satellite.order_direction.shape),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "shared_content_order_encoder_gradient": _gradient_state(model.content_order_encoder),
        "content_query_gradient": _gradient_state(
            torch.nn.ParameterList([model.content_order_encoder.pool_query])
        ),
        "frozen_prefix_gradients": (
            "all_none"
            if all(
                parameter.grad is None
                for block in model.backbone.blocks[:8]
                for parameter in block.parameters()
            )
            else "unexpected_gradient"
        ),
        "block_8_gradient": _gradient_state(model.backbone.blocks[8]),
        "block_9_gradient": _gradient_state(model.backbone.blocks[9]),
        "block_10_gradient": _gradient_state(model.backbone.blocks[10]),
        "block_11_gradient": _gradient_state(model.backbone.blocks[11]),
        "final_norm_gradient": _gradient_state(model.backbone.model.norm),
        "loss": float(losses.total.detach()),
        "joint_retrieval_loss": float(losses.joint_retrieval),
        "order_retrieval_loss": float(losses.order_retrieval),
    }
    if (
        result["shared_content_order_encoder_gradient"] != "finite_nonzero"
        or result["content_query_gradient"] != "finite_nonzero"
        or result["frozen_prefix_gradients"] != "all_none"
        or result["block_8_gradient"] != active_suffix_expected
        or result["block_9_gradient"] != active_suffix_expected
        or result["block_10_gradient"] != active_suffix_expected
        or result["block_11_gradient"] != active_suffix_expected
        or result["final_norm_gradient"] != active_suffix_expected
    ):
        raise RuntimeError(f"CoR-Geo gradient contract failed: {result}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
