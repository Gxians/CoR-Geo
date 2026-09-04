from __future__ import annotations

import pytest

from cor_geo.utils.hashing import canonical_json_bytes, sha256_json


def test_canonical_json_accepts_mixed_yaml_mapping_key_types() -> None:
    value = {
        "branch": {
            360: "circular",
            "finite_fov": "replicate_no_wrap",
        }
    }
    equivalent = {
        "branch": {
            "finite_fov": "replicate_no_wrap",
            "360": "circular",
        }
    }

    assert canonical_json_bytes(value) == canonical_json_bytes(equivalent)
    assert sha256_json(value) == sha256_json(equivalent)


def test_canonical_json_rejects_key_collisions_after_normalization() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        canonical_json_bytes({1: "numeric", "1": "string"})
