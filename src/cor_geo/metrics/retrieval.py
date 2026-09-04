"""Location-level Recall definitions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def retrieval_metrics(
    ranks_1based: Sequence[int],
    database_size: int,
    recall_ks: Sequence[int],
    r1_percent_rounding: str = "ceil",
) -> dict[str, float | int]:
    """Compute recall metrics and retain both one-percent boundary conventions."""
    ranks = np.asarray(ranks_1based, dtype=np.int64)
    if ranks.ndim != 1 or len(ranks) == 0:
        raise ValueError("ranks_1based must be a non-empty vector")
    if np.any(ranks < 1) or np.any(ranks > database_size):
        raise ValueError("Ranks lie outside the database")
    output = {f"R@{int(k)}": float(np.mean(ranks <= int(k))) for k in recall_ks}
    if r1_percent_rounding not in {"floor", "ceil"}:
        raise ValueError("r1_percent_rounding must be 'floor' or 'ceil'")
    one_percent_floor = max(1, math.floor(0.01 * database_size))
    one_percent_ceil = max(1, math.ceil(0.01 * database_size))
    output["R@1%_floor"] = float(np.mean(ranks <= one_percent_floor))
    output["R@1%_ceil"] = float(np.mean(ranks <= one_percent_ceil))
    output["R@1%"] = output[f"R@1%_{r1_percent_rounding}"]
    output["R@1%_threshold_floor"] = int(one_percent_floor)
    output["R@1%_threshold_ceil"] = int(one_percent_ceil)
    return output
