"""Efficient descriptor export and exact cyclic Hard-Max evaluation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from cor_geo.datasets.cross_view import CrossViewDataset, SampleRequest
from cor_geo.losses.cyclic_matching import cyclic_hard_max_score
from cor_geo.metrics.retrieval import retrieval_metrics


def reduce_cyclic_shifts(
    shifts: Tensor,
    score_config: Mapping[str, Any],
) -> Tensor:
    """Apply the configured Hard-Max reduction over cyclic shifts."""
    if score_config != {
        "name": "fov_masked_cyclic_hard_max",
        "shift_reduction": "hard_max",
    }:
        raise ValueError("CoR-Geo evaluation supports cyclic Hard Max only")
    return cyclic_hard_max_score(shifts)


@dataclass(frozen=True)
class EncodedGroundViews:
    """Ground descriptors and their valid-FoV masks in manifest order."""

    direction: np.ndarray
    valid: np.ndarray
    ids: list[str]


@dataclass(frozen=True)
class EncodedSatelliteViews:
    """Satellite descriptors in immutable manifest order."""

    direction: np.ndarray
    ids: list[str]


@dataclass(frozen=True)
class PreparedSatelliteViews:
    """One GPU-resident satellite bank with reusable direction FFTs."""

    direction_fft: Tensor
    ids: list[str]


def prepare_satellite_views(
    satellite: EncodedSatelliteViews,
    device: torch.device,
    transfer_chunk_size: int = 4096,
) -> PreparedSatelliteViews:
    """Transfer a satellite bank once and precompute its cyclic FFT once."""
    if transfer_chunk_size <= 0:
        raise ValueError("transfer_chunk_size must be positive")
    if satellite.direction.ndim != 3:
        raise ValueError("Satellite directions must have shape [N,A,D]")
    count, angular_bins, direction_dim = satellite.direction.shape
    direction_fft = torch.empty(
        (count, angular_bins // 2 + 1, direction_dim),
        dtype=torch.complex64,
        device=device,
    )
    for start in range(0, count, int(transfer_chunk_size)):
        stop = min(start + int(transfer_chunk_size), count)
        direction = torch.as_tensor(
            np.ascontiguousarray(satellite.direction[start:stop]),
            dtype=torch.float32,
            device=device,
        )
        direction_fft[start:stop] = torch.fft.rfft(direction, dim=1)
    return PreparedSatelliteViews(direction_fft, list(satellite.ids))


class _SatelliteRows(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: CrossViewDataset,
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(map(int, indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return self.dataset.load_satellite(
            SampleRequest(
                index=self.indices[int(position)],
                epoch=1,
                fov_deg=360,
                orientation_mode="eval",
            )
        )


class _GroundBundleRows(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: CrossViewDataset,
        indices: Sequence[int],
        fovs: Sequence[int],
        mode: Literal["random_eval"],
        random_roll_angles_deg: Mapping[int, Sequence[int]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = list(map(int, indices))
        self.fovs = tuple(map(int, fovs))
        self.mode = mode
        self.random_roll_angles_deg = (
            None
            if random_roll_angles_deg is None
            else {
                int(fov): list(map(int, angles))
                for fov, angles in random_roll_angles_deg.items()
            }
        )
        if self.random_roll_angles_deg is None or set(
            self.random_roll_angles_deg
        ) != set(self.fovs):
            raise ValueError("Random evaluation requires an angle sequence per FoV")
        if any(
            len(angles) != len(self.indices)
            for angles in self.random_roll_angles_deg.values()
        ):
            raise ValueError("Every random angle sequence must match the indices")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.indices[int(position)]
        if self.mode == "random_eval":
            assert self.random_roll_angles_deg is not None
            return self.dataset.load_random_evaluation_ground_bundle(
                index,
                self.fovs,
                {
                    fov: self.random_roll_angles_deg[fov][int(position)]
                    for fov in self.fovs
                },
            )
        raise ValueError(f"Unsupported ground export mode: {self.mode}")


def _loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    if batch_size <= 0 or workers < 0:
        raise ValueError("Invalid descriptor export loader configuration")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _underlying(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def encode_satellite_views(
    model: nn.Module,
    dataset: CrossViewDataset,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int = 0,
    progress_label: str | None = None,
) -> EncodedSatelliteViews:
    """Encode each satellite exactly once."""
    resolved = _underlying(model)
    was_training = resolved.training
    resolved.eval()
    index_list = list(map(int, indices))
    direction: np.ndarray | None = None
    ids: list[str] = []
    cursor = 0
    started = time.perf_counter()
    loader = _loader(
        _SatelliteRows(dataset, index_list),
        batch_size,
        workers,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["satellite"].to(device=device, dtype=torch.float32, non_blocking=True)
            representation = resolved.encode_satellite(images)
            rows = len(images)
            if direction is None:
                direction = np.empty(
                    (len(index_list), *representation.direction.shape[1:]),
                    np.float32,
                )
            direction[cursor : cursor + rows] = representation.direction.float().cpu().numpy()
            cursor += rows
            ids.extend(map(str, batch["satellite_id"]))
            if progress_label and (batch_index == 1 or batch_index % 100 == 0):
                rate = cursor / max(time.perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] satellite {cursor}/{len(index_list)} "
                    f"{rate:.1f} rows/s",
                    flush=True,
                )
    if direction is None or cursor != len(index_list):
        raise RuntimeError("Satellite descriptor export is incomplete")
    if was_training:
        resolved.train(True)
        resolved.set_train_epoch(1)
    return EncodedSatelliteViews(direction, ids)


def encode_ground_views(
    model: nn.Module,
    dataset: CrossViewDataset,
    indices: Sequence[int],
    fovs: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int = 0,
    mode: Literal["random_eval"] = "random_eval",
    random_roll_angles_deg: Mapping[int, Sequence[int]] | None = None,
    progress_label: str | None = None,
) -> dict[int, EncodedGroundViews]:
    """Open each panorama once and encode all requested FoV crops."""
    resolved = _underlying(model)
    was_training = resolved.training
    resolved.eval()
    index_list = list(map(int, indices))
    normalized_fovs = tuple(map(int, fovs))
    storage: dict[int, dict[str, np.ndarray]] = {}
    ids: list[str] = []
    cursor = 0
    started = time.perf_counter()
    loader = _loader(
        _GroundBundleRows(
            dataset,
            index_list,
            normalized_fovs,
            mode,
            random_roll_angles_deg,
        ),
        batch_size,
        workers,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            rows = len(batch["query_id"])
            for fov in normalized_fovs:
                images = batch[f"ground_{fov}"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                representation = resolved.encode_ground(images, fov)
                if fov not in storage:
                    storage[fov] = {
                        "direction": np.empty(
                            (len(index_list), *representation.direction.shape[1:]),
                            np.float32,
                        ),
                        "valid": np.empty(
                            (len(index_list), representation.valid.shape[1]),
                            np.bool_,
                        ),
                    }
                storage[fov]["direction"][cursor : cursor + rows] = (
                    representation.direction.float().cpu().numpy()
                )
                storage[fov]["valid"][cursor : cursor + rows] = representation.valid.cpu().numpy()
            ids.extend(map(str, batch["query_id"]))
            cursor += rows
            if progress_label and (batch_index == 1 or batch_index % 100 == 0):
                rate = cursor / max(time.perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] ground {cursor}/{len(index_list)} "
                    f"{rate:.1f} rows/s",
                    flush=True,
                )
    if cursor != len(index_list):
        raise RuntimeError("Ground descriptor export is incomplete")
    if was_training:
        resolved.train(True)
        resolved.set_train_epoch(1)
    return {
        fov: EncodedGroundViews(
            storage[fov]["direction"],
            storage[fov]["valid"],
            list(ids),
        )
        for fov in normalized_fovs
    }


def score_aligned_candidates(
    query: EncodedGroundViews,
    satellite: EncodedSatelliteViews,
    candidate_indices: np.ndarray,
    device: torch.device,
    query_batch_size: int,
    candidate_chunk_size: int,
    score_config: dict[str, Any],
    prepared_satellite: PreparedSatelliteViews | None = None,
) -> np.ndarray:
    """Score a different fixed candidate list for every query on GPU."""
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 2 or len(candidates) != len(query.direction):
        raise ValueError("candidate_indices must have shape [query, candidate]")
    output = np.empty(candidates.shape, dtype=np.float32)
    angular_bins = query.direction.shape[1]
    prepared = (
        prepare_satellite_views(satellite, device)
        if prepared_satellite is None
        else prepared_satellite
    )
    if prepared.ids != satellite.ids:
        raise ValueError("Prepared satellite IDs differ from the encoded bank")
    for query_start in range(0, len(query.direction), query_batch_size):
        query_stop = min(query_start + query_batch_size, len(query.direction))
        ground_direction = torch.as_tensor(
            query.direction[query_start:query_stop],
            dtype=torch.float32,
            device=device,
        )
        ground_valid = torch.as_tensor(
            query.valid[query_start:query_stop],
            dtype=torch.bool,
            device=device,
        )
        ground_fft = torch.fft.rfft(
            ground_direction * ground_valid[..., None],
            dim=1,
        )
        normalizer = ground_valid.sum(dim=1).clamp_min(1).float()
        rows = candidates[query_start:query_stop]
        for candidate_start in range(0, candidates.shape[1], candidate_chunk_size):
            candidate_stop = min(candidate_start + candidate_chunk_size, candidates.shape[1])
            candidate_chunk = rows[:, candidate_start:candidate_stop]
            candidate_tensor = torch.as_tensor(
                candidate_chunk,
                dtype=torch.long,
                device=device,
            )
            satellite_fft = prepared.direction_fft[candidate_tensor]
            spectrum = torch.einsum(
                "bfd,bcfd->bcf",
                torch.conj(ground_fft),
                satellite_fft,
            )
            shifts = torch.fft.irfft(spectrum, n=angular_bins, dim=-1)
            shifts = shifts / normalizer[:, None, None]
            scores = reduce_cyclic_shifts(shifts, score_config)
            output[
                query_start:query_stop,
                candidate_start:candidate_stop,
            ] = scores.cpu().numpy()
    return output


def _overwrite_positive_in_score_chunk(
    scores: Tensor,
    shifts: Tensor,
    positive_scores: Tensor,
    positive_shift_logits: Tensor,
    positive_indices: np.ndarray,
    location_start: int,
    location_stop: int,
) -> None:
    """Make a positive's scanned value bit-identical to its reference value.

    The positive is scored once before database scanning. Recomputing it in a
    different GEMM shape can differ by a few floating-point ulps; without this
    replacement a true rank-1 match can count itself as a higher-scoring item.
    """
    positives = np.asarray(positive_indices, dtype=np.int64)
    contained = np.flatnonzero(
        (positives >= int(location_start)) & (positives < int(location_stop))
    )
    if not len(contained):
        return
    rows = torch.as_tensor(contained, dtype=torch.long, device=scores.device)
    columns = torch.as_tensor(
        positives[contained] - int(location_start),
        dtype=torch.long,
        device=scores.device,
    )
    scores[rows, columns] = positive_scores.index_select(0, rows)
    shifts[rows, columns] = positive_shift_logits.index_select(0, rows)


def exact_evaluate_shard(
    query: EncodedGroundViews,
    fov_deg: int,
    satellite: EncodedSatelliteViews,
    device: torch.device,
    query_chunk_size: int,
    location_chunk_size: int,
    score_config: dict[str, Any],
    checkpoint_sha256: str,
    manifest_sha256: str,
    crop_schedule_sha256: str,
    split: str = "val",
    dataset_name: str = "cvact",
    progress_label: str | None = None,
    prepared_satellite: PreparedSatelliteViews | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute exact ranks with one resident bank and no full score matrix."""
    satellite_ids = np.asarray(satellite.ids, dtype=str)
    id_to_index = {value: index for index, value in enumerate(satellite_ids.tolist())}
    if len(id_to_index) != len(satellite_ids):
        raise ValueError("Satellite IDs must be unique")
    positive_indices = np.asarray([id_to_index[value] for value in query.ids], dtype=np.int64)
    lexical_order = np.argsort(satellite_ids, kind="stable")
    lexical_rank = np.empty(len(satellite_ids), dtype=np.int64)
    lexical_rank[lexical_order] = np.arange(len(satellite_ids))
    lexical_rank_tensor = torch.as_tensor(
        lexical_rank,
        dtype=torch.int64,
        device=device,
    )
    prepared = (
        prepare_satellite_views(satellite, device)
        if prepared_satellite is None
        else prepared_satellite
    )
    if prepared.ids != satellite.ids:
        raise ValueError("Prepared satellite IDs differ from the encoded bank")
    angular_bins = int(query.direction.shape[1])
    rows: list[dict[str, Any]] = []
    all_ranks: list[int] = []
    started = time.perf_counter()
    for query_start in range(0, len(query.direction), query_chunk_size):
        query_stop = min(query_start + query_chunk_size, len(query.direction))
        local_count = query_stop - query_start
        positives = positive_indices[query_start:query_stop]
        ground_direction = torch.as_tensor(
            query.direction[query_start:query_stop], dtype=torch.float32, device=device
        )
        ground_valid = torch.as_tensor(
            query.valid[query_start:query_stop], dtype=torch.bool, device=device
        )
        ground_fft = torch.fft.rfft(
            ground_direction * ground_valid[..., None],
            dim=1,
        )
        normalizer = ground_valid.sum(dim=1).clamp_min(1).float()
        positive_tensor = torch.as_tensor(
            positives,
            dtype=torch.long,
            device=device,
        )
        local_rows = torch.arange(local_count, device=device)
        positive_spectrum = torch.einsum(
            "bfd,bfd->bf",
            torch.conj(ground_fft),
            prepared.direction_fft[positive_tensor],
        )
        positive_shifts = torch.fft.irfft(
            positive_spectrum,
            n=angular_bins,
            dim=-1,
        )
        positive_shifts = positive_shifts / normalizer[:, None]
        positive_scores = reduce_cyclic_shifts(
            positive_shifts,
            score_config,
        )
        positive_shift_ids = positive_shifts.argmax(dim=-1)
        ranks = torch.ones(local_count, dtype=torch.int64, device=device)
        best_scores = torch.full((local_count,), float("-inf"), device=device)
        best_indices = torch.full(
            (local_count,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        best_shift_ids = torch.full_like(best_indices, -1)
        best_lexical = torch.full_like(best_indices, len(satellite_ids))
        positive_lexical = lexical_rank_tensor[positive_tensor]
        for location_start in range(0, len(satellite.ids), location_chunk_size):
            location_stop = min(location_start + location_chunk_size, len(satellite.ids))
            satellite_fft = prepared.direction_fft[location_start:location_stop]
            spectrum = torch.einsum(
                "qfd,nfd->qnf",
                torch.conj(ground_fft),
                satellite_fft,
            )
            shifts = torch.fft.irfft(spectrum, n=angular_bins, dim=-1)
            shifts = shifts / normalizer[:, None, None]
            scores = reduce_cyclic_shifts(shifts, score_config)
            _overwrite_positive_in_score_chunk(
                scores,
                shifts,
                positive_scores,
                positive_shifts,
                positives,
                location_start,
                location_stop,
            )
            chunk_lexical = lexical_rank_tensor[location_start:location_stop]
            ranks += torch.count_nonzero(scores > positive_scores[:, None], dim=1)
            ranks += torch.count_nonzero(
                (scores == positive_scores[:, None])
                & (chunk_lexical[None, :] < positive_lexical[:, None]),
                dim=1,
            )
            chunk_max = scores.max(dim=1).values
            is_chunk_max = scores == chunk_max[:, None]
            chunk_lexical_ranks, chunk_local_index = torch.where(
                is_chunk_max,
                chunk_lexical[None, :],
                len(satellite_ids),
            ).min(dim=1)
            chunk_indices = chunk_local_index + int(location_start)
            chunk_shift_ids = shifts[local_rows, chunk_local_index].argmax(dim=-1)
            replace = (chunk_max > best_scores) | (
                (chunk_max == best_scores) & (chunk_lexical_ranks < best_lexical)
            )
            best_scores = torch.where(replace, chunk_max, best_scores)
            best_indices = torch.where(replace, chunk_indices, best_indices)
            best_shift_ids = torch.where(replace, chunk_shift_ids, best_shift_ids)
            best_lexical = torch.where(replace, chunk_lexical_ranks, best_lexical)
        rank_values = ranks.cpu().numpy()
        positive_score_values = positive_scores.cpu().numpy()
        positive_shift_values = positive_shift_ids.cpu().numpy()
        best_score_values = best_scores.cpu().numpy()
        best_index_values = best_indices.cpu().numpy()
        best_shift_values = best_shift_ids.cpu().numpy()
        for local_index in range(local_count):
            rank = int(rank_values[local_index])
            all_ranks.append(rank)
            positive = int(positives[local_index])
            top1 = int(best_index_values[local_index])
            rows.append(
                {
                    "dataset": str(dataset_name),
                    "split": split,
                    "query_id": str(query.ids[query_start + local_index]),
                    "fov_deg": int(fov_deg),
                    "positive_satellite_id": str(satellite_ids[positive]),
                    "predicted_rank_1based": rank,
                    "top1_satellite_id": str(satellite_ids[top1]),
                    "top1_score": float(best_score_values[local_index]),
                    "positive_score": float(positive_score_values[local_index]),
                    "best_shift_id": int(best_shift_values[local_index]),
                    "positive_best_shift_id": int(positive_shift_values[local_index]),
                    "checkpoint_sha256": checkpoint_sha256,
                    "manifest_sha256": manifest_sha256,
                    "random_crop_schedule_sha256": crop_schedule_sha256,
                }
            )
        if progress_label:
            completed = query_stop
            rate = completed / max(time.perf_counter() - started, 1.0e-9)
            print(
                f"[{progress_label}] exact {completed}/{len(query.direction)} "
                f"{rate:.2f} queries/s",
                flush=True,
            )
    metrics = retrieval_metrics(
        all_ranks,
        len(satellite.ids),
        recall_ks=(1, 5, 10),
        r1_percent_rounding="floor",
    )
    return pd.DataFrame(rows), metrics
