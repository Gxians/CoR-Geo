"""Self-contained coarse-to-exact hard-negative mining for CoR-Geo.

The coarse signature keeps the signed DC component and magnitudes of the
lowest non-zero angular FFT frequencies. It is therefore invariant to circular
azimuth shifts. Candidate ranking is finalized with the exact cyclic Hard-Max
score used by training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from cor_geo.losses.cyclic_matching import cyclic_hard_max_score


def rotation_invariant_fft_signature(
    direction: Tensor,
    valid: Tensor,
    frequency_count: int,
) -> tuple[Tensor, Tensor]:
    """Return a normalized low-frequency signature and the reusable FFT.

    ``direction`` is ``[B,A,D]`` and ``valid`` is ``[B,A]``.  Taking FFT
    magnitudes removes circular phase while retaining substantially more
    directional structure than a simple mean (the DC component alone).  DC
    remains signed because it is already phase invariant and its sign carries
    useful descriptor semantics.
    """
    if direction.ndim != 3 or valid.shape != direction.shape[:2]:
        raise ValueError("Invalid direction or validity geometry")
    available = direction.shape[1] // 2 + 1
    if not 1 <= int(frequency_count) <= available:
        raise ValueError("frequency_count is outside the angular FFT range")
    masked = direction.float() * valid[..., None].to(direction.dtype)
    spectrum = torch.fft.rfft(masked, dim=1)
    components = [spectrum[:, :1].real]
    if int(frequency_count) > 1:
        components.append(spectrum[:, 1 : int(frequency_count)].abs())
    signature = torch.cat(components, dim=1).flatten(1)
    signature = functional.normalize(signature, dim=-1)
    return signature, spectrum


@dataclass(frozen=True)
class MiningResult:
    indices: np.ndarray
    scores: np.ndarray
    positive_coarse_topk_recall: float


class HardNegativeCandidateBank:
    """GPU-resident satellite bank shared by all four FoV mining passes."""

    def __init__(
        self,
        satellite_direction: np.ndarray,
        device: torch.device,
        frequency_count: int,
    ) -> None:
        value = np.asarray(satellite_direction)
        if value.ndim != 3:
            raise ValueError("Satellite directions must have shape [N,A,D]")
        satellite = torch.as_tensor(
            np.ascontiguousarray(value),
            dtype=torch.float32,
            device=device,
        )
        valid = torch.ones(
            satellite.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        self.signature, self.spectrum = rotation_invariant_fft_signature(
            satellite,
            valid,
            int(frequency_count),
        )
        self.location_count = int(satellite.shape[0])
        self.angular_bins = int(satellite.shape[1])
        self.direction_dim = int(satellite.shape[2])
        self.device = device
        self.frequency_count = int(frequency_count)
        del satellite, valid

    def mine(
        self,
        query_direction: np.ndarray,
        query_valid: np.ndarray,
        query_indices: np.ndarray,
        coarse_keep: int,
        final_keep: int,
        search_chunk_size: int,
        rerank_chunk_size: int,
    ) -> MiningResult:
        """Coarse-screen all locations and rerank by exact cyclic Hard Max."""
        directions = np.asarray(query_direction)
        validity = np.asarray(query_valid)
        positives = np.asarray(query_indices, dtype=np.int64)
        if directions.ndim != 3 or validity.shape != directions.shape[:2]:
            raise ValueError("Query direction or validity geometry is invalid")
        if directions.shape[1:] != (self.angular_bins, self.direction_dim):
            raise ValueError("Query and satellite direction geometry differs")
        if len(positives) != len(directions):
            raise ValueError("Query indices and descriptors differ in length")
        if np.any(positives < 0) or np.any(positives >= self.location_count):
            raise ValueError("Positive location index is outside the satellite bank")
        if not 0 < int(final_keep) <= int(coarse_keep) < self.location_count:
            raise ValueError("Invalid coarse/final hard-negative counts")
        if min(int(search_chunk_size), int(rerank_chunk_size)) <= 0:
            raise ValueError("Mining chunk sizes must be positive")

        output_indices = np.empty((len(directions), int(final_keep)), dtype=np.int32)
        output_scores = np.empty((len(directions), int(final_keep)), dtype=np.float32)
        positive_covered = 0

        for search_start in range(0, len(directions), int(search_chunk_size)):
            search_stop = min(search_start + int(search_chunk_size), len(directions))
            query = torch.as_tensor(
                np.ascontiguousarray(directions[search_start:search_stop]),
                dtype=torch.float32,
                device=self.device,
            )
            valid = torch.as_tensor(
                np.ascontiguousarray(validity[search_start:search_stop]),
                dtype=torch.bool,
                device=self.device,
            )
            query_signature, query_spectrum = rotation_invariant_fft_signature(
                query,
                valid,
                self.frequency_count,
            )
            coarse_scores = query_signature @ self.signature.T
            positive_columns = torch.as_tensor(
                positives[search_start:search_stop],
                dtype=torch.long,
                device=self.device,
            )
            rows = torch.arange(search_stop - search_start, device=self.device)
            positive_scores = coarse_scores[rows, positive_columns].clone()
            coarse_scores[rows, positive_columns] = float("-inf")
            coarse_values, coarse_indices = torch.topk(
                coarse_scores,
                k=int(coarse_keep),
                dim=1,
                sorted=True,
            )
            # Free cross-view compatibility diagnostic: whether the true pair
            # would survive the same coarse Top-K screen before exclusion.
            positive_covered += int(
                (positive_scores >= coarse_values[:, -1]).sum().item()
            )
            del coarse_scores, query_signature, positive_scores, coarse_values

            for local_start in range(0, len(query), int(rerank_chunk_size)):
                local_stop = min(local_start + int(rerank_chunk_size), len(query))
                candidates = coarse_indices[local_start:local_stop]
                candidate_spectrum = self.spectrum[candidates]
                cross_spectrum = torch.einsum(
                    "qfd,qkfd->qkf",
                    torch.conj(query_spectrum[local_start:local_stop]),
                    candidate_spectrum,
                )
                correlations = torch.fft.irfft(
                    cross_spectrum,
                    n=self.angular_bins,
                    dim=-1,
                )
                normalizer = valid[local_start:local_stop].sum(dim=1).clamp_min(1)
                correlations = correlations / normalizer[:, None, None]
                exact_scores = cyclic_hard_max_score(correlations)
                values, order = torch.topk(
                    exact_scores,
                    k=int(final_keep),
                    dim=1,
                    sorted=True,
                )
                selected = torch.gather(candidates, 1, order)
                absolute_start = search_start + local_start
                absolute_stop = search_start + local_stop
                output_indices[absolute_start:absolute_stop] = (
                    selected.cpu().numpy().astype(np.int32, copy=False)
                )
                output_scores[absolute_start:absolute_stop] = (
                    values.cpu().numpy().astype(np.float32, copy=False)
                )
                del candidate_spectrum, cross_spectrum, correlations, exact_scores
                del values, order, selected
            del query, valid, query_spectrum, coarse_indices

        if np.any(output_indices == positives.astype(np.int32)[:, None]):
            raise RuntimeError("Hard-negative miner retained a positive location")
        return MiningResult(
            indices=output_indices,
            scores=output_scores,
            positive_coarse_topk_recall=(
                float(positive_covered) / max(len(directions), 1)
            ),
        )
