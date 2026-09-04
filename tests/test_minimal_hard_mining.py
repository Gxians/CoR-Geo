from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional

from cor_geo.engine.trainer import (
    _expected_pool_metadata,
    _hard_pool_root,
    _load_hard_pools,
)
from cor_geo.losses.cyclic_matching import (
    cyclic_direction_logits,
    cyclic_hard_max_score,
)
from cor_geo.mining.hard_negative_mining import (
    HardNegativeCandidateBank,
    rotation_invariant_fft_signature,
)
from cor_geo.mining.hard_pool import load_compact_hard_pool, write_compact_hard_pool
from cor_geo.utils.hashing import sha256_file
from cor_geo.utils.io import write_json


def test_fft_signature_is_invariant_to_circular_direction_roll() -> None:
    generator = torch.Generator().manual_seed(7)
    direction = functional.normalize(torch.randn(3, 12, 8, generator=generator), dim=-1)
    valid = torch.ones(3, 12, dtype=torch.bool)
    first, _ = rotation_invariant_fft_signature(direction, valid, 3)
    second, _ = rotation_invariant_fft_signature(direction.roll(4, dims=1), valid, 3)
    assert torch.allclose(first, second, atol=1.0e-5, rtol=1.0e-5)


def test_hard_negative_miner_excludes_positive_and_exact_scores_are_sorted() -> None:
    generator = torch.Generator().manual_seed(11)
    satellite = functional.normalize(
        torch.randn(8, 12, 16, generator=generator),
        dim=-1,
    )
    positive_indices = np.asarray([1, 5, 7], dtype=np.int32)
    query = satellite[positive_indices].roll(3, dims=1).numpy()
    valid = np.ones(query.shape[:2], dtype=np.bool_)
    bank = HardNegativeCandidateBank(satellite.numpy(), torch.device("cpu"), 3)
    result = bank.mine(
        query,
        valid,
        positive_indices,
        coarse_keep=6,
        final_keep=4,
        search_chunk_size=2,
        rerank_chunk_size=1,
    )
    assert result.indices.shape == (3, 4)
    assert not np.any(result.indices == positive_indices[:, None])
    assert np.all(result.scores[:, :-1] >= result.scores[:, 1:])
    assert result.positive_coarse_topk_recall == 1.0
    for row in range(len(query)):
        expected = cyclic_hard_max_score(
            cyclic_direction_logits(
                torch.from_numpy(query[row : row + 1]),
                satellite[result.indices[row]],
                torch.from_numpy(valid[row : row + 1]),
            )
        )[0]
        assert np.allclose(
            result.scores[row],
            expected.numpy(),
            atol=1.0e-5,
            rtol=1.0e-5,
        )


def test_compact_pool_round_trip_validates_metadata(tmp_path) -> None:
    size = 8
    keep = 3
    positives = np.arange(size, dtype=np.int32)[:, None]
    negatives = (positives + np.arange(1, keep + 1, dtype=np.int32)) % size
    scores = np.tile(np.asarray([0.9, 0.8, 0.7], dtype=np.float32), (size, 1))
    metadata = {
        "source_epoch": 16,
        "source_checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "model_config_sha256": "model",
        "candidate_strategy": "signed_dc_low2_fft_magnitude_top512_exact_cyclic_hardmax_top64_v1",
        "coarse_frequency_count": 3,
        "coarse_candidate_locations": 512,
        "keep_negative_locations": keep,
        "crop_orientation_contract": "source_epoch_pool_equals_hard_batch",
        "format": "dense_ranked_location_indices_v1",
    }
    write_compact_hard_pool(tmp_path, 90, negatives, scores, metadata)
    loaded = load_compact_hard_pool(tmp_path, 90, metadata, size, keep)
    assert np.array_equal(loaded, negatives)


def test_pool_load_uses_current_cor_geo_run(tmp_path, train_config) -> None:
    size = 160
    keep = 64
    train_config = deepcopy(train_config)
    run_root = tmp_path / "cor_geo_run"
    checkpoint = run_root / "checkpoints" / "epoch_016.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"CoR-Geo checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    provenance = {
        "manifest_hashes": {"train": "manifest"},
        "model_config_sha256": "b" * 64,
    }
    root = _hard_pool_root(run_root, 16, checkpoint_hash)
    metadata = _expected_pool_metadata(16, checkpoint_hash, provenance, train_config)
    positives = np.arange(size, dtype=np.int32)[:, None]
    negatives = (positives + np.arange(1, keep + 1, dtype=np.int32)) % size
    scores = np.broadcast_to(
        np.linspace(1.0, 0.0, keep, dtype=np.float32),
        (size, keep),
    ).copy()
    for fov in (360, 180, 90, 70):
        write_compact_hard_pool(root, fov, negatives, scores, metadata)
    write_json(
        root / "refresh.metadata.json",
        {**metadata, "fovs": [360, 180, 90, 70], "location_count": size},
    )

    loaded = _load_hard_pools(
        run_root,
        17,
        [360, 180, 90, 70],
        provenance,
        train_config,
        size,
    )
    assert all(pool.shape == (size, keep) for pool in loaded.values())
