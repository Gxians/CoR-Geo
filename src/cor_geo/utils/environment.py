"""Environment and repository provenance collection."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def git_state(path: str | Path) -> dict[str, Any]:
    """Return commit and dirty state without mutating a repository."""
    root = Path(path).expanduser().resolve()
    if not (root / ".git").exists():
        return {"path": str(root), "is_repository": False, "commit": None, "dirty": None}
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"path": str(root), "is_repository": True, "commit": commit, "dirty": bool(status.strip())}


def environment_snapshot() -> dict[str, Any]:
    """Collect runtime versions relevant to reproducibility."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
