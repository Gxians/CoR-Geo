#!/usr/bin/env python
"""Create audited CVUSA train/val manifests from the 19zl split CSVs."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from cor_geo.datasets.manifests import build_cvusa_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/cvusa.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    outputs = build_cvusa_manifests(
        args.dataset_config,
        args.paths,
        overwrite=bool(args.overwrite),
        verify_images=not bool(args.skip_image_verification),
        progress_every=int(args.progress_every),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
