#!/usr/bin/env python
"""Create immutable audited CVACT pair manifests."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cor_geo.datasets.manifests import build_cvact_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-image-decode", action="store_true")
    args = parser.parse_args()
    outputs = build_cvact_manifests(
        args.dataset_config,
        args.paths,
        overwrite=args.overwrite,
        verify_images=not args.skip_image_decode,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
