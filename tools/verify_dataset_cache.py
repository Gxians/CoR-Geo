#!/usr/bin/env python
"""Prove cached and direct preprocessing are tensor-identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_yaml  # noqa: E402
from cor_geo.datasets.cross_view import CrossViewDataset, SampleRequest  # noqa: E402
from cor_geo.datasets.manifests import read_manifest  # noqa: E402
from cor_geo.datasets.resized_cache import cache_request  # noqa: E402

FOVS = (360, 180, 90, 70)


def _assert_equal(name: str, cached: torch.Tensor, direct: torch.Tensor) -> None:
    if not torch.equal(cached, direct):
        difference = float((cached.float() - direct.float()).abs().max())
        raise ValueError(f"{name} differs; maximum absolute error={difference}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--samples-per-split", type=int, default=8)
    args = parser.parse_args()
    if args.samples_per_split <= 0:
        raise ValueError("--samples-per-split must be positive")
    dataset_config = load_yaml(args.dataset_config)
    train_config = load_yaml(args.train_config)
    model_config = load_yaml(args.model_config)
    paths = load_yaml(args.paths)
    dataset_name = str(dataset_config["dataset"])
    project_root = Path(paths["project_root"]).expanduser().resolve()
    input_config = model_config["input"]
    results = {}
    for split in args.splits:
        manifest = read_manifest(
            project_root / "data_manifests" / dataset_name / f"{split}.parquet"
        )
        cache_root, _ = cache_request(train_config, split)
        if cache_root is None:
            raise ValueError("Dataset cache is disabled")
        common = {
            "manifest": manifest,
            "global_seed": int(train_config["seed"]),
            "ground_height": int(input_config["ground_height"]),
            "panorama_width": int(input_config["panorama_width"]),
            "satellite_size": int(input_config["satellite_size"][0]),
            "ground_widths": input_config["widths"],
            "dataset_name": dataset_name,
        }
        direct = CrossViewDataset(**common)
        cached = CrossViewDataset(
            **common,
            resized_cache_root=cache_root,
            require_resized_cache=True,
        )
        count = min(int(args.samples_per_split), len(manifest))
        indices = np.linspace(0, len(manifest) - 1, count, dtype=np.int64)
        comparisons = 0
        for offset, value in enumerate(indices):
            index = int(value)
            fov = FOVS[offset % len(FOVS)]
            if split == "train":
                request = SampleRequest(index, 1 + offset % 17, fov, "train")
                direct_row = direct[request]
                cached_row = cached[request]
                _assert_equal("ground", cached_row["ground"], direct_row["ground"])
                _assert_equal(
                    "satellite", cached_row["satellite"], direct_row["satellite"]
                )
                comparisons += 2
            else:
                roll_angles = {
                    active_fov: (offset * 73 + active_fov) % 360
                    for active_fov in FOVS
                }
                direct_bundle = direct.load_random_evaluation_ground_bundle(
                    index,
                    FOVS,
                    roll_angles,
                )
                cached_bundle = cached.load_random_evaluation_ground_bundle(
                    index,
                    FOVS,
                    roll_angles,
                )
                for active_fov in FOVS:
                    key = f"ground_{active_fov}"
                    _assert_equal(key, cached_bundle[key], direct_bundle[key])
                    comparisons += 1
                request = SampleRequest(index, 1, 360, "eval")
                _assert_equal(
                    "satellite",
                    cached.load_satellite(request)["satellite"],
                    direct.load_satellite(request)["satellite"],
                )
                comparisons += 1
        results[split] = {
            "samples": count,
            "tensor_comparisons": comparisons,
            "status": "exact_equal",
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
