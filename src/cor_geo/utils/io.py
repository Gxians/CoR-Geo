"""Safe filesystem IO used by tools and training runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: str | Path) -> Path:
    """Create and return the parent directory of a path."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_bytes(path: str | Path, payload: bytes, overwrite: bool = False) -> Path:
    """Atomically write bytes while protecting existing outputs by default."""
    resolved = ensure_parent(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def write_json(path: str | Path, value: Any, overwrite: bool = False) -> Path:
    """Write indented UTF-8 JSON atomically."""
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    return atomic_write_bytes(path, payload, overwrite=overwrite)


def write_yaml(path: str | Path, value: Any, overwrite: bool = False) -> Path:
    """Write deterministic YAML atomically."""
    payload = yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")
    return atomic_write_bytes(path, payload, overwrite=overwrite)


def assert_within_run(target: str | Path, run_root: str | Path) -> Path:
    """Ensure a destructive overwrite target is strictly inside one run root."""
    resolved_target = Path(target).expanduser().resolve()
    resolved_root = Path(run_root).expanduser().resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"Target must be strictly inside run root: {resolved_target}")
    return resolved_target
