#!/usr/bin/env python
"""Verify generated manifests without modifying them."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_yaml  # noqa: E402
from cor_geo.datasets.manifests import load_manifest_metadata, read_manifest, validate_manifest_frames  # noqa: E402
from cor_geo.utils.hashing import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()
    dataset = load_yaml(args.dataset_config)
    paths = load_yaml(args.paths)
    root = Path(paths["project_root"]) / "data_manifests" / "cvact"
    frames = {split: read_manifest(root / f"{split}.parquet") for split in ("train", "val", "test")}
    expected = {split: int(dataset["splits"][split]["expected_final_count"]) for split in frames}
    validate_manifest_frames(frames, expected)
    metadata = load_manifest_metadata(root / "manifest_metadata.json")
    actual_hashes = {split: sha256_file(root / f"{split}.parquet") for split in frames}
    if metadata["manifest_sha256"] != actual_hashes:
        raise ValueError("Manifest hashes do not match manifest_metadata.json")
    print("CVACT manifest verification: PASS")
    print({split: len(frame) for split, frame in frames.items()})


if __name__ == "__main__":
    main()
