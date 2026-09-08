"""Random-FoV evaluation, shared-gallery encoding, and exact retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.distributed as distributed
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from cor_geo.content_order import ConservativeAngularResampler
from cor_geo.datasets import CrossViewDataset, SampleRequest, cache_request, read_manifest, write_parquet_atomic
from cor_geo.matching import cyclic_hard_max_score
from cor_geo.model import CoRGeoModel
from cor_geo.train import load_checkpoint
from cor_geo.utils import configure_deterministic_algorithms, load_yaml, sha256_file, sha256_json, write_json

"""Location-level Recall definitions."""


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


"""Parameter-free geometry for evaluation at unseen fields of view."""


@dataclass(frozen=True)
class EvaluationFovGeometry:
    """Resolved crop, patch-grid, and canonical direction sizes for one FoV."""

    fov_deg: int
    physical_crop_width: int
    aligned_input_width: int
    source_patch_columns: int
    target_direction_bins: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-serializable provenance."""
        return {key: int(value) for key, value in asdict(self).items()}


def resolve_evaluation_fov_geometry(
    fovs: Sequence[int],
    model_config: dict[str, Any],
) -> dict[int, EvaluationFovGeometry]:
    """Resolve unseen FoVs without changing any learned model parameter.

    The panorama is first cropped at its exact physical angular width.  The
    crop is then rounded *up* to a ViT patch multiple, matching the registered
    90-degree (189->196) and 70-degree (147->154) preprocessing rule.  The
    canonical direction count remains one 10-degree bin per direction.
    """
    input_config = model_config["input"]
    architecture = model_config["architecture"]
    patch_size = int(model_config["backbone"]["patch_size"])
    panorama_width = int(input_config["panorama_width"])
    angular_bins = int(architecture["angular_bins"])
    registered_widths = {int(key): int(value) for key, value in input_config["widths"].items()}
    resampling = architecture["ground_angular_resampling"]
    registered_sources = {int(key): int(value) for key, value in resampling["source_patch_columns"].items()}
    registered_targets = {int(key): int(value) for key, value in resampling["target_direction_bins"].items()}
    resolved: dict[int, EvaluationFovGeometry] = {}
    for raw_fov in fovs:
        fov = int(raw_fov)
        if not 0 < fov <= 360:
            raise ValueError(f"FoV must be in (0, 360], got {fov}")
        physical_numerator = panorama_width * fov
        if physical_numerator % 360:
            raise ValueError(f"FoV {fov} does not map to an integer crop at panorama width " f"{panorama_width}")
        direction_numerator = angular_bins * fov
        if direction_numerator % 360:
            raise ValueError(
                f"FoV {fov} does not map to an integer number of canonical "
                f"direction bins for angular_bins={angular_bins}"
            )
        physical_width = physical_numerator // 360
        aligned_width = (
            registered_widths[fov]
            if fov in registered_widths
            else ((physical_width + patch_size - 1) // patch_size) * patch_size
        )
        if aligned_width % patch_size:
            raise ValueError(f"Aligned width for FoV {fov} is not patch-compatible")
        source_columns = aligned_width // patch_size
        target_bins = direction_numerator // 360
        if fov in registered_sources and registered_sources[fov] != source_columns:
            raise ValueError(f"Registered source-column geometry differs for FoV {fov}")
        if fov in registered_targets and registered_targets[fov] != target_bins:
            raise ValueError(f"Registered target-bin geometry differs for FoV {fov}")
        resolved[fov] = EvaluationFovGeometry(
            fov_deg=fov,
            physical_crop_width=physical_width,
            aligned_input_width=aligned_width,
            source_patch_columns=source_columns,
            target_direction_bins=target_bins,
        )
    if len(resolved) != len(tuple(fovs)):
        raise ValueError("Evaluation FoVs must be unique")
    return resolved


def register_evaluation_resamplers(
    model: nn.Module,
    geometries: dict[int, EvaluationFovGeometry],
    device: torch.device,
) -> None:
    """Attach fixed resamplers for unseen FoVs after strict checkpoint load."""
    for fov, geometry in geometries.items():
        key = str(int(fov))
        if key in model.ground_angular_resamplers:
            existing = model.ground_angular_resamplers[key]
            if (
                int(existing.source_bins) != geometry.source_patch_columns
                or int(existing.target_bins) != geometry.target_direction_bins
            ):
                raise ValueError(f"Existing resampler geometry differs for FoV {fov}")
            continue
        resampler = ConservativeAngularResampler(
            geometry.source_patch_columns,
            geometry.target_direction_bins,
        ).to(device)
        resampler.eval()
        model.ground_angular_resamplers[key] = resampler


"""Efficient descriptor export and exact cyclic Hard-Max evaluation."""


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
            else {int(fov): list(map(int, angles)) for fov, angles in random_roll_angles_deg.items()}
        )
        if self.random_roll_angles_deg is None or set(self.random_roll_angles_deg) != set(self.fovs):
            raise ValueError("Random evaluation requires an angle sequence per FoV")
        if any(len(angles) != len(self.indices) for angles in self.random_roll_angles_deg.values()):
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
                {fov: self.random_roll_angles_deg[fov][int(position)] for fov in self.fovs},
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
                    f"[{progress_label}] satellite {cursor}/{len(index_list)} " f"{rate:.1f} rows/s",
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
                storage[fov]["direction"][cursor : cursor + rows] = representation.direction.float().cpu().numpy()
                storage[fov]["valid"][cursor : cursor + rows] = representation.valid.cpu().numpy()
            ids.extend(map(str, batch["query_id"]))
            cursor += rows
            if progress_label and (batch_index == 1 or batch_index % 100 == 0):
                rate = cursor / max(time.perf_counter() - started, 1.0e-9)
                print(
                    f"[{progress_label}] ground {cursor}/{len(index_list)} " f"{rate:.1f} rows/s",
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
    prepared = prepare_satellite_views(satellite, device) if prepared_satellite is None else prepared_satellite
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
    contained = np.flatnonzero((positives >= int(location_start)) & (positives < int(location_stop)))
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
    prepared = prepare_satellite_views(satellite, device) if prepared_satellite is None else prepared_satellite
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
        ground_direction = torch.as_tensor(query.direction[query_start:query_stop], dtype=torch.float32, device=device)
        ground_valid = torch.as_tensor(query.valid[query_start:query_stop], dtype=torch.bool, device=device)
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
                (scores == positive_scores[:, None]) & (chunk_lexical[None, :] < positive_lexical[:, None]),
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
            replace = (chunk_max > best_scores) | ((chunk_max == best_scores) & (chunk_lexical_ranks < best_lexical))
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
                }
            )
        if progress_label:
            completed = query_stop
            rate = completed / max(time.perf_counter() - started, 1.0e-9)
            print(
                f"[{progress_label}] exact {completed}/{len(query.direction)} " f"{rate:.2f} queries/s",
                flush=True,
            )
    metrics = retrieval_metrics(
        all_ranks,
        len(satellite.ids),
        recall_ks=(1, 5, 10),
        r1_percent_rounding="floor",
    )
    return pd.DataFrame(rows), metrics


"""Auditable random-crop schedules for public evaluation."""


def _schedule_hash(
    schedule: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
) -> str:
    return sha256_json(
        {
            "protocol": "independent_random_roll_then_left_crop_v1",
            "split": split,
            "query_ids": schedule["query_id"].astype(str).tolist(),
            "roll_angles_deg": {str(fov): schedule[f"roll_angle_deg_{fov}"].astype(int).tolist() for fov in fovs},
        }
    )


def validate_random_crop_schedule(
    schedule: pd.DataFrame,
    manifest: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], pd.DataFrame, str]:
    """Validate, canonicalize, and hash a query--FoV heading schedule."""
    expected_columns = [
        "split",
        "query_id",
        *(f"roll_angle_deg_{fov}" for fov in fovs),
    ]
    if list(schedule.columns) != expected_columns:
        raise ValueError(f"Random crop schedule columns are {list(schedule.columns)}, " f"expected {expected_columns}")
    if len(schedule) != len(manifest):
        raise ValueError("Random crop schedule length differs from the manifest")
    if schedule["split"].astype(str).tolist() != [split] * len(manifest):
        raise ValueError("Random crop schedule split differs from the request")
    query_ids = manifest["query_id"].astype(str).tolist()
    if schedule["query_id"].astype(str).tolist() != query_ids:
        raise ValueError("Random crop schedule query order differs from the manifest")

    canonical = pd.DataFrame({"split": split, "query_id": query_ids})
    angles: dict[int, np.ndarray] = {}
    for fov in fovs:
        column = f"roll_angle_deg_{fov}"
        numeric = pd.to_numeric(schedule[column], errors="raise").to_numpy()
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain finite integer degrees")
        if ((numeric < 0) | (numeric > 359)).any():
            raise ValueError(f"{column} must lie in [0, 359]")
        values = numeric.astype(np.int16)
        canonical[column] = values
        angles[int(fov)] = values
    return angles, canonical, _schedule_hash(canonical, split, fovs)


def draw_random_crop_schedule(
    manifest: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
    rng: np.random.Generator | None = None,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, str]:
    """Draw one independent integer heading for every query--FoV pair."""
    generator = np.random.default_rng() if rng is None else rng
    schedule = pd.DataFrame(
        {
            "split": split,
            "query_id": manifest["query_id"].astype(str),
            **{
                f"roll_angle_deg_{fov}": generator.integers(
                    0,
                    360,
                    size=len(manifest),
                    dtype=np.int16,
                )
                for fov in fovs
            },
        }
    )
    return validate_random_crop_schedule(schedule, manifest, split, fovs)


def load_random_crop_schedule(
    path: str | Path,
    manifest: pd.DataFrame,
    split: str,
    fovs: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], pd.DataFrame, str]:
    """Load and validate a previously recorded Parquet crop schedule."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return validate_random_crop_schedule(
        pd.read_parquet(resolved, engine="pyarrow"),
        manifest,
        split,
        fovs,
    )


# ---- command-line interface ----

"""Exact evaluation with random or replayed FoV crops."""


FOVS = (360, 180, 90, 70)


def _parse_device_ids(value: str | None) -> tuple[int, ...]:
    """Resolve the public device list while keeping single-GPU as the default."""
    if value is None:
        return (0,)
    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise ValueError("--devices must be a comma-separated list such as 0 or 0,1")
    try:
        devices = tuple(int(field) for field in fields)
    except ValueError as error:
        raise ValueError("--devices must contain non-negative integer GPU IDs") from error
    if any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative integer GPU IDs")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate GPU IDs")
    return devices


def _launch_workers(devices_arg: str | None) -> None:
    """Hide the distributed launcher behind the public ``--devices`` option."""
    if "RANK" in os.environ:
        return
    devices = _parse_device_ids(devices_arg)
    if devices_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={len(devices)}",
        "-m",
        "cor_geo.evaluate",
        *sys.argv[1:],
    ]
    os.execvpe(sys.executable, command, os.environ.copy())


def _initialize(runtime: dict[str, Any]) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    configured = runtime["distributed"].get("world_size", "auto")
    if rank < 0 or local_rank < 0 or world_size < 1:
        raise RuntimeError("Evaluation launcher did not initialize the distributed runtime")
    if configured != "auto" and world_size != int(configured):
        raise RuntimeError(f"Runtime expects world_size={configured}, received {world_size}")
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(
        backend=str(runtime["distributed"]["backend"]),
        timeout=timedelta(minutes=int(runtime["distributed"]["timeout_minutes"])),
    )
    return rank, local_rank, world_size


def _broadcast_error(error: str | None, rank: int, context: str) -> None:
    values = [error if rank == 0 else None]
    distributed.broadcast_object_list(values, src=0)
    if values[0]:
        raise RuntimeError(f"{context}: {values[0]}")


def _bounds(length: int, rank: int, world_size: int) -> tuple[int, int]:
    return length * rank // world_size, length * (rank + 1) // world_size


def _checkpoint_path(run_dir: Path, name: str) -> Path:
    filename = name if name.endswith(".ckpt") else f"{name}.ckpt"
    return run_dir / "checkpoints" / filename


def _save_encoded(root: Path, encoded: EncodedSatelliteViews) -> None:
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / "direction.npy", encoded.direction.astype(np.float32), allow_pickle=False)
    np.save(root / "ids.npy", np.asarray(encoded.ids, dtype=str), allow_pickle=False)


def _merge_satellite_shards(
    staging_root: Path,
    bank_root: Path,
    world_size: int,
    location_count: int,
    angular_bins: int,
    direction_dim: int,
) -> list[str]:
    bank_root.mkdir(parents=True, exist_ok=False)
    direction = np.lib.format.open_memmap(
        bank_root / "direction.fp32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(location_count, angular_bins, direction_dim),
    )
    ids: list[str] = []
    cursor = 0
    for rank in range(world_size):
        root = staging_root / f"rank_{rank:02d}" / "satellite"
        shard_direction = np.load(root / "direction.npy", mmap_mode="r", allow_pickle=False)
        shard_ids = np.load(root / "ids.npy", allow_pickle=False).astype(str).tolist()
        rows = len(shard_direction)
        direction[cursor : cursor + rows] = shard_direction
        ids.extend(shard_ids)
        cursor += rows
    if cursor != location_count:
        raise RuntimeError("Satellite shard merge did not cover the manifest")
    direction.flush()
    np.save(bank_root / "satellite_ids.npy", np.asarray(ids, dtype=str), allow_pickle=False)
    return ids


def _validate_satellite_bank(
    root: Path,
    expected_ids: list[str],
    checkpoint_hash: str,
    manifest_hash: str,
    model_config_hash: str,
    angular_bins: int,
    direction_dim: int,
) -> None:
    """Validate a completed or recoverable satellite bank before reuse."""
    metadata_path = root / "metadata.json"
    required = (
        root / "direction.fp32.npy",
        root / "satellite_ids.npy",
        metadata_path,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Incomplete satellite bank: {root}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "model_config_sha256": model_config_hash,
        "direction_shape": [
            len(expected_ids),
            int(angular_bins),
            int(direction_dim),
        ],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Satellite bank metadata mismatch: {key}")
    direction = np.load(root / "direction.fp32.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(root / "satellite_ids.npy", allow_pickle=False).astype(str).tolist()
    if (
        direction.shape
        != (
            len(expected_ids),
            int(angular_bins),
            int(direction_dim),
        )
        or direction.dtype != np.float32
    ):
        raise ValueError("Satellite direction bank has an invalid shape or dtype")
    if ids != expected_ids:
        raise ValueError("Satellite bank IDs differ from the evaluation manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        help="Comma-separated GPU IDs; defaults to one GPU (for example: 0,1)",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--fovs", type=int, nargs="+", default=list(FOVS))
    parser.add_argument(
        "--crop-schedule",
        help=(
            "Optional recorded random_crop_schedule.parquet to replay exactly. "
            "If omitted, a new independent random schedule is drawn."
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--evaluation-dataset-config",
        help=("Optional target dataset YAML. Defaults to the dataset stored in " "the checkpoint run."),
    )
    parser.add_argument(
        "--allow-cross-dataset-weights",
        action="store_true",
        help=(
            "Explicitly permit strict model-weight loading from the checkpoint "
            "while evaluating a different dataset manifest."
        ),
    )
    parser.add_argument(
        "--allow-unseen-fovs",
        action="store_true",
        help=(
            "Explicitly permit parameter-free FoV geometry not used for model "
            "selection (for example 270 and 120 degrees)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Optional explicit output directory; defaults to a hash-named draw directory",
    )
    parser.add_argument(
        "--satellite-bank-dir",
        help=(
            "Optional shared satellite-bank parent, reusable only when checkpoint, " "manifest, and model hashes match."
        ),
    )
    args = parser.parse_args()
    _launch_workers(args.devices)
    active_fovs = tuple(map(int, args.fovs))
    if active_fovs != FOVS and not args.allow_unseen_fovs:
        raise ValueError(f"FoVs must be supplied in exact order {FOVS}")
    split = str(args.split)
    shared = load_yaml(args.config)
    runtime = shared["runtime"]
    rank, local_rank, world_size = _initialize(runtime)
    device = torch.device("cuda", local_rank)
    succeeded = False
    try:
        run_dir = Path(args.run_dir).expanduser().resolve()
        resolved = load_yaml(run_dir / "config_resolved.yaml")
        checkpoint_dataset_name = str(resolved["dataset"]["dataset"])
        evaluation_dataset = (
            load_yaml(args.evaluation_dataset_config) if args.evaluation_dataset_config else resolved["dataset"]
        )
        dataset_name = str(evaluation_dataset["dataset"])
        cross_dataset = dataset_name != checkpoint_dataset_name
        if cross_dataset and not args.allow_cross_dataset_weights:
            raise ValueError("Cross-dataset evaluation requires --allow-cross-dataset-weights")
        if not cross_dataset and args.allow_cross_dataset_weights:
            raise ValueError("--allow-cross-dataset-weights was supplied for the checkpoint dataset")
        manifest_splits = tuple(map(str, evaluation_dataset["manifest_splits"]))
        if split not in manifest_splits:
            raise ValueError(f"Split {split} is unavailable for {dataset_name}: {manifest_splits}")
        model_config = resolved["model"]
        checkpoint_train_config = resolved["train"]
        evaluation_train_config = dict(shared["train"]) if cross_dataset else checkpoint_train_config
        evaluation_train_config["dataset_cache"] = dict(evaluation_train_config["dataset_cache"])
        evaluation_train_config["dataset_cache"]["root"] = str(evaluation_train_config["dataset_cache"]["root"]).format(
            dataset=dataset_name
        )
        eval_config = resolved["evaluation"]
        paths = shared["paths"]
        configure_deterministic_algorithms()
        checkpoint = _checkpoint_path(run_dir, args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        selected_epoch = int(checkpoint.stem.removeprefix("epoch_"))
        if selected_epoch not in set(map(int, checkpoint_train_config["checkpoint"]["retained_epochs"])):
            raise ValueError("Evaluation accepts retained checkpoint epochs only")
        checkpoint_hash = sha256_file(checkpoint)
        project_root = Path(paths["project_root"])
        manifest_root = project_root / "data_manifests" / dataset_name
        manifest_path = manifest_root / f"{split}.parquet"
        manifest = read_manifest(manifest_path).reset_index(drop=True)
        manifest_hash = sha256_file(manifest_path)
        schedule_values: list[tuple[dict[int, np.ndarray], pd.DataFrame, str] | None] = [None]
        if rank == 0:
            if args.crop_schedule:
                schedule_values[0] = load_random_crop_schedule(
                    args.crop_schedule,
                    manifest,
                    split,
                    active_fovs,
                )
            else:
                schedule_values[0] = draw_random_crop_schedule(
                    manifest,
                    split,
                    active_fovs,
                )
        distributed.broadcast_object_list(schedule_values, src=0)
        if schedule_values[0] is None:
            raise RuntimeError("Random crop schedule broadcast failed")
        random_roll_angles_deg, random_crop_schedule, crop_hash = schedule_values[0]
        model_config_hash = sha256_json(model_config)
        fov_geometries = resolve_evaluation_fov_geometry(active_fovs, model_config)

        model = CoRGeoModel(
            model_config,
            dinov2_root=paths["dinov2_root"],
            checkpoint_path=paths["checkpoints"]["dinov2_vitb14"],
        ).to(device)
        payload = load_checkpoint(
            checkpoint,
            model,
            restore_rng=False,
            expected_resolved_config=resolved,
        )
        if int(payload["completed_epoch"]) != selected_epoch:
            raise ValueError("Checkpoint epoch metadata differs from its filename")
        expected = {"model": (payload["model_config_sha256"], model_config_hash)}
        if not cross_dataset:
            expected["manifest"] = (
                payload["manifest_hashes"][split],
                manifest_hash,
            )
        mismatches = [name for name, values in expected.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(f"Evaluation provenance mismatch: {mismatches}")
        register_evaluation_resamplers(model, fov_geometries, device)
        model.eval()
        model.float()
        score_config = dict(model_config["score"])
        bank_angular_bins = int(model.angular_bins)
        bank_direction_dim = int(model.direction_dim)
        torch.backends.cuda.matmul.allow_tf32 = bool(eval_config["exact"]["allow_tf32"])
        cache_root, require_cache = cache_request(evaluation_train_config, split)
        ground_widths = {int(fov): geometry.aligned_input_width for fov, geometry in fov_geometries.items()}
        dataset = CrossViewDataset(
            manifest,
            global_seed=int(checkpoint_train_config["seed"]),
            ground_height=int(model_config["input"]["ground_height"]),
            panorama_width=int(model_config["input"]["panorama_width"]),
            satellite_size=int(model_config["input"]["satellite_size"][0]),
            ground_widths=ground_widths,
            resized_cache_root=cache_root,
            require_resized_cache=require_cache,
            dataset_name=dataset_name,
        )
        evaluation_root = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else (
                run_dir / "evaluations" / f"{split}_random" / f"epoch_{selected_epoch:03d}" / f"draw_{crop_hash[:12]}"
            )
        )
        summary_path = evaluation_root / "summary.json"
        staging_root = evaluation_root / str(runtime["storage"]["staging_directory"])
        bank_parent = (
            Path(args.satellite_bank_dir).expanduser().resolve()
            if args.satellite_bank_dir
            else run_dir / "evaluations" / "shared_satellite_bank"
        )
        bank_root = bank_parent / split / checkpoint_hash
        setup_error: str | None = None
        reuse_satellite_bank: bool | None = None
        if rank == 0:
            try:
                if summary_path.exists():
                    raise FileExistsError("Evaluation summary already exists")
                expected_satellite_ids = manifest["satellite_id"].astype(str).tolist()
                validation_arguments = (
                    expected_satellite_ids,
                    checkpoint_hash,
                    manifest_hash,
                    model_config_hash,
                    bank_angular_bins,
                    bank_direction_dim,
                )
                reuse_satellite_bank = False
                if bank_root.exists():
                    try:
                        _validate_satellite_bank(bank_root, *validation_arguments)
                        reuse_satellite_bank = True
                    except (FileNotFoundError, ValueError, OSError):
                        shutil.rmtree(bank_root)
                recoverable_bank = staging_root / "merged_bank"
                if not reuse_satellite_bank and recoverable_bank.exists():
                    try:
                        _validate_satellite_bank(recoverable_bank, *validation_arguments)
                        bank_root.parent.mkdir(parents=True, exist_ok=True)
                        recoverable_bank.replace(bank_root)
                        reuse_satellite_bank = True
                    except (FileNotFoundError, ValueError, OSError):
                        pass
                if staging_root.exists():
                    shutil.rmtree(staging_root)
                staging_root.mkdir(parents=True, exist_ok=False)
                schedule_path = evaluation_root / "random_crop_schedule.parquet"
                write_parquet_atomic(random_crop_schedule, schedule_path)
            except Exception as error:
                setup_error = f"{type(error).__name__}: {error}"
        _broadcast_error(setup_error, rank, "Evaluation setup")
        reuse_values = [reuse_satellite_bank if rank == 0 else None]
        distributed.broadcast_object_list(reuse_values, src=0)
        reuse_satellite_bank = bool(reuse_values[0])
        distributed.barrier()

        start, stop = _bounds(len(manifest), rank, world_size)
        indices = np.arange(start, stop, dtype=np.int64)
        export = runtime["descriptor_export"]
        rank_root = staging_root / f"rank_{rank:02d}"
        rank_root.mkdir(parents=True, exist_ok=False)
        if not reuse_satellite_bank:
            satellite_shard = encode_satellite_views(
                model,
                dataset,
                indices,
                device,
                int(export["batch_size_per_gpu"]),
                int(export["workers_per_rank"]),
                progress_label=f"rank{rank}/{split}",
            )
            _save_encoded(rank_root / "satellite", satellite_shard)
            distributed.barrier()

            publish_error: str | None = None
            if rank == 0:
                try:
                    bank_partial = staging_root / "merged_bank"
                    satellite_ids = _merge_satellite_shards(
                        staging_root,
                        bank_partial,
                        world_size,
                        len(manifest),
                        bank_angular_bins,
                        bank_direction_dim,
                    )
                    expected_ids = manifest["satellite_id"].astype(str).tolist()
                    if satellite_ids != expected_ids:
                        raise ValueError("Merged satellite IDs differ from the evaluation manifest")
                    write_json(
                        bank_partial / "metadata.json",
                        {
                            "checkpoint_sha256": checkpoint_hash,
                            "manifest_sha256": manifest_hash,
                            "model_config_sha256": model_config_hash,
                            "location_score": model_config["score"]["name"],
                            "shift_reduction": str(model_config["score"]["shift_reduction"]),
                            "direction_shape": [
                                len(manifest),
                                bank_angular_bins,
                                bank_direction_dim,
                            ],
                            "shared_across_fovs": True,
                            "distributed_satellite_shards": world_size,
                        },
                    )
                    _validate_satellite_bank(
                        bank_partial,
                        expected_ids,
                        checkpoint_hash,
                        manifest_hash,
                        model_config_hash,
                        bank_angular_bins,
                        bank_direction_dim,
                    )
                    bank_root.parent.mkdir(parents=True, exist_ok=True)
                    bank_partial.replace(bank_root)
                except Exception as error:
                    publish_error = f"{type(error).__name__}: {error}"
            _broadcast_error(publish_error, rank, "Satellite bank publication")
            distributed.barrier()

        satellite = EncodedSatelliteViews(
            np.load(bank_root / "direction.fp32.npy", mmap_mode="r", allow_pickle=False),
            np.load(bank_root / "satellite_ids.npy", allow_pickle=False).astype(str).tolist(),
        )
        queries = encode_ground_views(
            model,
            dataset,
            indices,
            active_fovs,
            device,
            int(export["batch_size_per_gpu"]),
            int(export["workers_per_rank"]),
            mode="random_eval",
            random_roll_angles_deg={fov: random_roll_angles_deg[fov][start:stop] for fov in active_fovs},
            progress_label=f"rank{rank}/{split}",
        )
        expected_ids = manifest.iloc[start:stop]["query_id"].astype(str).tolist()
        if any(queries[fov].ids != expected_ids for fov in active_fovs):
            raise ValueError("Query shard order differs from evaluation manifest")
        del model
        torch.cuda.empty_cache()
        prepared_satellite = prepare_satellite_views(satellite, device)
        exact = runtime["exact_score"]
        for fov in active_fovs:
            predictions, _ = exact_evaluate_shard(
                queries[fov],
                fov,
                satellite,
                device,
                query_chunk_size=int(exact["query_chunk_size"]),
                location_chunk_size=int(exact["location_chunk_size"]),
                score_config=score_config,
                split=split,
                dataset_name=dataset_name,
                progress_label=f"rank{rank}/fov{fov}",
                prepared_satellite=prepared_satellite,
            )
            write_parquet_atomic(
                predictions,
                rank_root / f"predictions_fov_{fov}.parquet",
            )
        distributed.barrier()

        summary: dict[str, Any] | None = None
        merge_error: str | None = None
        if rank == 0:
            try:
                summary = {
                    "output_directory": str(evaluation_root),
                    "dataset": dataset_name,
                    "checkpoint_dataset": checkpoint_dataset_name,
                    "cross_dataset_weights_only": cross_dataset,
                    "selected_epoch": selected_epoch,
                    "checkpoint_sha256": checkpoint_hash,
                    "split": split,
                    "crop_mode": "random",
                    "manifest_sha256": manifest_hash,
                    "location_score": model_config["score"]["name"],
                    "shift_reduction": model_config["score"]["shift_reduction"],
                    "direction_representation": "content_attention_plus_first_cosine_order",
                    "evaluation_fov_geometry": {str(fov): fov_geometries[fov].to_dict() for fov in active_fovs},
                    "execution": {
                        "engine": "distributed_exact_cyclic_v1",
                        "world_size": world_size,
                        "used_for_model_selection": False,
                        "single_satellite_bank_for_all_fovs": True,
                    },
                    "fovs": {},
                }
                summary["random_crop_protocol"] = {
                    "angle_sampler": "independent_uniform_integer_0_359",
                    "operation": "right_roll_then_left_fov_crop",
                    "independent_random_angle_per_query_fov": True,
                    "schedule_source": ("replayed" if args.crop_schedule else "new_random_draw"),
                    "schedule": str(evaluation_root / "random_crop_schedule.parquet"),
                    "schedule_sha256": crop_hash,
                }
                prediction_root = evaluation_root / "predictions"
                for fov in active_fovs:
                    parts = [
                        pd.read_parquet(
                            staging_root / f"rank_{shard_rank:02d}" / f"predictions_fov_{fov}.parquet",
                            engine="pyarrow",
                        )
                        for shard_rank in range(world_size)
                    ]
                    predictions = pd.concat(parts, ignore_index=True)
                    if predictions["query_id"].astype(str).tolist() != manifest["query_id"].astype(str).tolist():
                        raise ValueError("Merged predictions differ from the evaluation manifest")
                    metrics = retrieval_metrics(
                        predictions["predicted_rank_1based"].to_numpy(np.int64),
                        len(manifest),
                        recall_ks=(1, 5, 10),
                        r1_percent_rounding=str(eval_config["metrics"]["r1_percent_rounding"]),
                    )
                    write_parquet_atomic(
                        predictions,
                        prediction_root / f"{split}_fov{fov}.parquet",
                    )
                    summary["fovs"][str(fov)] = metrics
                write_json(summary_path, summary)
                shutil.rmtree(staging_root)
            except Exception as error:
                merge_error = f"{type(error).__name__}: {error}"
        _broadcast_error(merge_error, rank, "Prediction merge")
        if rank == 0:
            print(json.dumps(summary, indent=2))
        succeeded = True
    finally:
        # Keep the process group alive for PyTorch's distributed exception hook
        # on failures; torchrun tears it down when the process exits.
        if succeeded and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
