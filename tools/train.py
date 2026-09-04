#!/usr/bin/env python
"""Launch the configured two-GPU CoR-Geo training protocol."""

from __future__ import annotations

import argparse
import os

# Must be configured before the first CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from _bootstrap import bootstrap

bootstrap()

from cor_geo.config import load_experiment_config  # noqa: E402
from cor_geo.engine.trainer import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_experiment_config(
        args.dataset_config,
        args.model_config,
        args.train_config,
        args.eval_config,
        args.paths,
    )
    train(
        config,
        args.run_name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
