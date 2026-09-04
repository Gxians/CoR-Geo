#!/usr/bin/env python
"""Build exact resized uint8 caches for CVACT or CVUSA."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_yaml  # noqa: E402
from cor_geo.datasets.manifests import read_manifest  # noqa: E402
from cor_geo.datasets.resized_cache import build_resized_cache, cache_request  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=2048)
    args = parser.parse_args()

    dataset_config = load_yaml(args.dataset_config)
    train_config = load_yaml(args.train_config)
    model_config = load_yaml(args.model_config)
    paths = load_yaml(args.paths)
    dataset_name = str(dataset_config["dataset"])
    allowed = set(map(str, dataset_config["manifest_splits"]))
    unexpected = set(map(str, args.splits)) - allowed
    if unexpected:
        raise ValueError(f"Unavailable {dataset_name} cache splits: {sorted(unexpected)}")
    project_root = Path(paths["project_root"]).expanduser().resolve()
    input_config = model_config["input"]
    manifests = {
        split: read_manifest(
            project_root / "data_manifests" / dataset_name / f"{split}.parquet"
        )
        for split in args.splits
    }
    cache_root, _ = cache_request(train_config, str(args.splits[0]))
    if cache_root is None:
        raise ValueError("Dataset cache is disabled")
    expected_bytes = sum(
        len(manifest)
        * (
            int(input_config["ground_height"])
            * int(input_config["panorama_width"])
            * 3
            + int(input_config["satellite_size"][0]) ** 2 * 3
        )
        for split, manifest in manifests.items()
        if not (cache_root / str(split)).exists()
    )
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_root.parent).free
    required_bytes = int(expected_bytes * 1.15)
    if free_bytes < required_bytes:
        raise OSError(
            "Insufficient cache space: "
            f"cache data need about {expected_bytes / 2**30:.1f} GiB, "
            f"15% safety margin requires {required_bytes / 2**30:.1f} GiB, "
            f"but only {free_bytes / 2**30:.1f} GiB are available"
        )
    for split in args.splits:
        build_resized_cache(
            manifests[split],
            cache_root,
            int(input_config["ground_height"]),
            int(input_config["panorama_width"]),
            int(input_config["satellite_size"][0]),
            workers=int(args.workers),
            batch_size=int(args.batch_size),
            progress_every=int(args.progress_every),
        )


if __name__ == "__main__":
    main()
