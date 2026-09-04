"""Structured append-only logging without process-global logger state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cor_geo.utils.io import ensure_parent


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one deterministic JSON object to a UTF-8 JSONL file."""
    resolved = ensure_parent(path)
    payload = json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    return resolved
