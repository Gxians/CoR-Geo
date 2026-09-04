#!/usr/bin/env python
"""Verify CVUSA manifests, source files, counts, order, and hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_yaml  # noqa: E402
from cor_geo.datasets.manifests import (  # noqa: E402
    load_manifest_metadata,
    read_manifest,
    validate_manifest_frames,
)
from cor_geo.utils.hashing import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/cvusa.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()
    dataset = load_yaml(args.dataset_config)
    paths = load_yaml(args.paths)
    root = Path(paths["project_root"]) / "data_manifests" / "cvusa"
    frames = {
        split: read_manifest(root / f"{split}.parquet")
        for split in ("train", "val")
    }
    expected = {
        split: int(dataset["splits"][split]["expected_count"])
        for split in frames
    }
    validate_manifest_frames(frames, expected)
    metadata = load_manifest_metadata(root / "manifest_metadata.json")
    hashes = {
        split: sha256_file(root / f"{split}.parquet") for split in frames
    }
    if metadata["manifest_sha256"] != hashes:
        raise ValueError("CVUSA manifest hashes differ from manifest_metadata.json")
    print("CVUSA manifest verification: PASS")


if __name__ == "__main__":
    main()
