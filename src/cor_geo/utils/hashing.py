"""Cryptographic hashing helpers for immutable experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it entirely into memory."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_json_mapping_keys(value: Any) -> Any:
    """Recursively apply JSON's string-key semantics before sorting mappings.

    ``yaml.safe_load`` preserves numeric mapping keys as numbers.  Passing a
    mapping containing both numeric and string keys directly to
    ``json.dumps(..., sort_keys=True)`` makes Python try to order incomparable
    objects (for example ``360`` and ``"finite_fov"``).  JSON object keys are
    strings, so normalize them explicitly and reject ambiguous collisions.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        original_keys: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str):
                json_key = key
            elif key is None:
                json_key = "null"
            elif isinstance(key, bool):
                json_key = "true" if key else "false"
            elif isinstance(key, int | float):
                json_key = str(key)
            else:
                raise TypeError(
                    "JSON mapping keys must be strings, numbers, booleans, or null; "
                    f"got {type(key).__name__}"
                )
            if json_key in normalized:
                raise ValueError(
                    "Mapping keys become ambiguous after JSON normalization: "
                    f"{original_keys[json_key]!r} and {key!r}"
                )
            original_keys[json_key] = key
            normalized[json_key] = _normalize_json_mapping_keys(child)
        return normalized
    if isinstance(value, list | tuple):
        return [_normalize_json_mapping_keys(child) for child in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    normalized = _normalize_json_mapping_keys(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value in canonical form."""
    return sha256_bytes(canonical_json_bytes(value))
