"""Deterministic hashing, random state, and framework configuration."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch


def stable_seed(*items: object) -> int:
    """Map a tuple of stable string representations to a uint32 seed."""
    payload = "::".join(map(str, items)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def configure_deterministic_algorithms() -> None:
    """Request deterministic PyTorch kernels without fixing any RNG state."""
    workspace_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise ValueError(f"Unsupported deterministic CUBLAS_WORKSPACE_CONFIG: {workspace_config}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def configure_determinism(seed: int) -> None:
    """Seed supported RNGs and request deterministic PyTorch algorithms."""
    configure_deterministic_algorithms()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_state() -> dict[str, Any]:
    """Capture RNG state for exact epoch-boundary resume."""
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    """Restore state produced by :func:`capture_random_state`."""
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_random_state"])
    if torch.cuda.is_available() and state["torch_cuda_random_states"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_states"])
