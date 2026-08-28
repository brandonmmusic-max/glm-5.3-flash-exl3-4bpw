"""Deterministic multi-draw vectors and exact 128-block G-scale folding."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .constants import HAD_K, MAX_DRAWS, MIN_DRAWS
from .determinism import derive_seed, sha256_bytes


def _rademacher_value(seed: int, index: int) -> float:
    payload = struct.pack(">QQ", seed & ((1 << 64) - 1), index)
    return 1.0 if hashlib.sha256(payload).digest()[0] & 1 else -1.0


def rademacher_vector(length: int, *seed_parts: object) -> tuple[float, ...]:
    if length <= 0:
        raise ValueError("rotation length must be positive")
    seed = derive_seed("rotation", *seed_parts, bits=64)
    return tuple(_rademacher_value(seed, index) for index in range(length))


def vector_sha256(vector: Sequence[float]) -> str:
    return sha256_bytes(b"".join(struct.pack(">d", float(value)) for value in vector))


@dataclass(frozen=True)
class RotationDraw:
    draw: int
    suh: tuple[float, ...]
    svh: tuple[float, ...]
    score: float

    @property
    def digest(self) -> str:
        return sha256_bytes(
            bytes.fromhex(vector_sha256(self.suh))
            + bytes.fromhex(vector_sha256(self.svh))
        )


def select_pair_multidraw(
    *,
    k: int,
    n: int,
    draws: int,
    seed_parts: Sequence[object],
    score: Callable[[Sequence[float], Sequence[float], int], float],
) -> RotationDraw:
    if not MIN_DRAWS <= draws <= MAX_DRAWS:
        raise ValueError(f"draws must be in [{MIN_DRAWS},{MAX_DRAWS}]")
    candidates: list[RotationDraw] = []
    for draw in range(draws):
        suh = rademacher_vector(k, *seed_parts, "suh", draw)
        svh = rademacher_vector(n, *seed_parts, "svh", draw)
        value = float(score(suh, svh, draw))
        if not math.isfinite(value):
            raise ValueError(f"non-finite rotation proxy at draw {draw}")
        candidates.append(RotationDraw(draw, suh, svh, value))
    return min(candidates, key=lambda candidate: (candidate.score, candidate.draw))


def expand_block_scales(
    scales: Iterable[float], size: int, block: int = HAD_K
) -> tuple[float, ...]:
    values = tuple(float(value) for value in scales)
    if size % block or len(values) != size // block:
        raise ValueError(f"need {size // block} scales for size={size}, block={block}")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("G-scales must be finite and positive")
    return tuple(value for value in values for _ in range(block))


@dataclass(frozen=True)
class FoldedBlockScale:
    suh: tuple[float, ...]
    svh: tuple[float, ...]
    row_multiplier: tuple[float, ...]
    column_multiplier: tuple[float, ...]


def fold_block_g_scale(
    suh: Sequence[float],
    svh: Sequence[float],
    k_block_scales: Iterable[float],
    n_block_scales: Iterable[float],
    block: int = HAD_K,
) -> FoldedBlockScale:
    """Fold separable regularized-weight scales into stored vectors.

    If the regularized matrix is multiplied by `gk[:,None]*gn[None,:]`,
    division of stored `suh` and `svh` by the same repeated block factors
    leaves the decoded matrix unchanged. Block-constant factors commute with
    the corresponding block Hadamard transform.
    """

    k_expanded = expand_block_scales(k_block_scales, len(suh), block)
    n_expanded = expand_block_scales(n_block_scales, len(svh), block)
    return FoldedBlockScale(
        suh=tuple(float(value) / scale for value, scale in zip(suh, k_expanded)),
        svh=tuple(float(value) / scale for value, scale in zip(svh, n_expanded)),
        row_multiplier=k_expanded,
        column_multiplier=n_expanded,
    )


def coordinate_search_block_scales(
    *,
    k_blocks: int,
    n_blocks: int,
    score: Callable[[tuple[float, ...], tuple[float, ...]], float],
    grid: Sequence[float] = (0.85, 0.925, 1.0, 1.075, 1.15),
    sweeps: int = 2,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Deterministic per-128-block GSS with a caller-supplied RT proxy."""

    if k_blocks <= 0 or n_blocks <= 0 or sweeps <= 0:
        raise ValueError("block counts and sweeps must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("scale grid must be finite and positive")
    ks = [1.0] * k_blocks
    ns = [1.0] * n_blocks
    best = float(score(tuple(ks), tuple(ns)))
    if not math.isfinite(best):
        raise ValueError("initial G-scale score is non-finite")
    for _ in range(sweeps):
        for side, values in (("k", ks), ("n", ns)):
            for index in range(len(values)):
                incumbent_value = values[index]
                candidates: list[tuple[float, float]] = []
                for value in grid:
                    values[index] = float(value)
                    candidate = float(score(tuple(ks), tuple(ns)))
                    if not math.isfinite(candidate):
                        raise ValueError("non-finite G-scale proxy")
                    candidates.append((candidate, float(value)))
                best_score, best_value = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
                values[index] = best_value if best_score <= best else incumbent_value
                best = min(best, best_score)
    return tuple(ks), tuple(ns), best
