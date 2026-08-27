"""Streaming, identity-bound calibration statistics for routed projections.

The fitter deliberately keeps natural routed observations separate from
supplemental observations.  Supplemental observations are additionally kept
in both raw and inverse-inclusion-probability-corrected forms.  The combined
estimator is therefore natural + corrected supplemental; no implicit blending
or layer-global pooling is performed.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..core.artifacts import canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json


SCHEMA = "quant-pipeline.calibration-fit.v2"
VECTOR_SEARCH_SCHEMA = "quant-pipeline.transform-vector-search.v1"
SEARCH_SCORE_RECEIPT_SCHEMA = "quant-pipeline.search-score-receipt.v1"
ROUTE_WEIGHT_POWERS = (0, 1, 2)
ACCOUNTING_KINDS = ("natural", "supplemental_raw", "supplemental_corrected", "combined")
STORED_ARRAY_FIELDS = ("mean", "second_moment")
_COVARIANCE_MODES = ("full", "block_diagonal", "diagonal")
_VECTOR_OBJECTIVE_ARMS = {
    "absolute_gate_squared_sse",
    "relative_sse",
    "energy_normalized_sse",
    "fisher_projection",
}
_HASH_RE = re.compile(r"[0-9a-f]{64}")
PRODUCTION_DEFAULTS = {
    "retained_accounting": ("combined",),
    "retained_powers": ROUTE_WEIGHT_POWERS,
    "covariance_mode": "full",
    "block_size": 128,
    "artifact_dtype": "float32",
}


def _as_numpy(value: Any) -> np.ndarray:
    """Convert CPU/accelerator array-likes without making torch a dependency."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite numeric values")


def _validate_source_identities(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_identities must be a non-empty mapping")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError("source identity keys and values must be strings")
        key = raw_key
        item = raw_value
        if not key or not item or key.strip() != key or item.strip() != item:
            raise ValueError("source identity keys and values must be non-empty canonical strings")
        if key in result:
            raise ValueError(f"duplicate source identity {key!r}")
        result[key] = item
    return dict(sorted(result.items()))


def _normalise_expert_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        if int(value) < 0:
            raise ValueError("expert IDs must be non-negative integers or non-empty strings")
        return str(int(value))
    if isinstance(value, str) and value and value.strip() == value:
        # Numeric IDs have one canonical spelling.  This prevents separate
        # artifacts for expert 1 and expert 01 and gives numeric ordering below.
        if value.isdecimal():
            return str(int(value))
        return value
    raise ValueError("expert IDs must be non-negative integers or non-empty strings")


def _expert_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdecimal() else (1, value)


def _covariance_shape(dimension: int, mode: str, block_size: int) -> tuple[int, ...]:
    if mode == "full":
        return (dimension, dimension)
    if mode == "block_diagonal":
        return (dimension // block_size, block_size, block_size)
    if mode == "diagonal":
        return (dimension,)
    raise ValueError(f"unknown covariance mode {mode!r}")


def _outer(value: np.ndarray, mode: str, block_size: int) -> np.ndarray:
    if mode == "full":
        return np.outer(value, value)
    if mode == "block_diagonal":
        blocks = value.reshape(-1, block_size)
        return np.einsum("bi,bj->bij", blocks, blocks, optimize=True)
    if mode == "diagonal":
        return np.square(value)
    raise ValueError(f"unknown covariance mode {mode!r}")


def _weighted_outer(values: np.ndarray, weights: np.ndarray, mode: str, block_size: int) -> np.ndarray:
    if mode == "full":
        return np.einsum("n,ni,nj->ij", weights, values, values, optimize=True)
    if mode == "block_diagonal":
        blocks = values.reshape(values.shape[0], -1, block_size)
        return np.einsum("n,nbi,nbj->bij", weights, blocks, blocks, optimize=True)
    if mode == "diagonal":
        return np.einsum("n,ni,ni->i", weights, values, values, optimize=True)
    raise ValueError(f"unknown covariance mode {mode!r}")


def _materialize_dense(value: np.ndarray, mode: str, dimension: int, block_size: int) -> np.ndarray:
    if mode == "full":
        return value
    result = np.zeros((dimension, dimension), dtype=value.dtype)
    if mode == "block_diagonal":
        for index, block in enumerate(value):
            start = index * block_size
            result[start : start + block_size, start : start + block_size] = block
        return result
    if mode == "diagonal":
        np.fill_diagonal(result, value)
        return result
    raise ValueError(f"unknown covariance mode {mode!r}")


@dataclass(frozen=True)
class CalibrationBatch:
    """One routed projection-input batch.

    ``origins`` may be a scalar ``"natural"``/``"supplemental"`` or one value
    per row.  Supplemental rows require their population inclusion probability;
    they are corrected by ``1 / inclusion_probability``.  Natural rows either
    omit probabilities or provide exactly one.
    """

    expert_inputs: Any
    expert_ids: Any
    route_weights: Any
    document_ids: Sequence[str]
    token_offsets: Any
    layer_id: int
    predecessor_checkpoint_hash: str
    projection: str
    origins: str | Sequence[str] = "natural"
    inclusion_probabilities: Any | None = None


@dataclass(frozen=True)
class SearchScore:
    """One finite score plus a hash-bound external evaluation receipt."""

    score: float
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class TransformVectorCandidate:
    draw_index: int
    candidate_id: str
    baseline: bool
    vectors: dict[str, np.ndarray]
    block_scale_sha256: dict[str, str]

    def vectors_for_expert(self, expert_id: int | str, expert_ids: Sequence[str]) -> dict[str, np.ndarray]:
        normalized = _normalise_expert_id(expert_id)
        try:
            index = tuple(expert_ids).index(normalized)
        except ValueError as error:
            raise KeyError(f"unknown expert {normalized!r}") from error
        return {
            "gate.suh": self.vectors["gate_up_input_shared"],
            "gate.svh": self.vectors["gate_output"][index],
            "up.suh": self.vectors["gate_up_input_shared"],
            "up.svh": self.vectors["up_output"][index],
            "down.suh": self.vectors["down_input"][index],
            "down.svh": self.vectors["down_output_shared"],
        }


@dataclass(frozen=True)
class SearchedTransformVectors:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def vectors_for_expert(self, expert_id: int | str) -> dict[str, np.ndarray]:
        # Loaded arrays may be mmap-backed and can change after load.  Recheck
        # their generated content identity at the point of consumption.
        verify_searched_transform_vectors(self)
        candidate = TransformVectorCandidate(
            draw_index=int(self.metadata["winner"]["draw_index"]),
            candidate_id=self.metadata["winner"]["candidate_id"],
            baseline=bool(self.metadata["winner"]["baseline"]),
            vectors=self.arrays,
            block_scale_sha256=self.metadata["winner"]["block_scale_sha256"],
        )
        return candidate.vectors_for_expert(
            expert_id, self.metadata["identity"]["generator"]["expert_ids"]
        )


class _WeightedMoments:
    """Mergeable weighted Chan/Welford sufficient statistics in float64."""

    def __init__(
        self,
        dimension: int,
        *,
        covariance_mode: str,
        block_size: int,
        retain_covariance: bool,
    ) -> None:
        self.dimension = dimension
        self.covariance_mode = covariance_mode
        self.block_size = block_size
        self.retain_covariance = retain_covariance
        self.count = 0
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self.mean = np.zeros(dimension, dtype=np.float64)
        # Deliberately independent from Chan/Welford's centered M2.  The
        # codec consumes E[w xx^T]/E[w] directly; reconstructing it from a
        # rounded covariance plus a rounded mean loses observable ULPs.
        self.raw_second_numerator = (
            np.zeros(_covariance_shape(dimension, covariance_mode, block_size), dtype=np.float64)
            if retain_covariance
            else None
        )
        self.document_ids: set[str] = set()
        self.sample_keys: set[tuple[str, int]] = set()

    def update(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        document_ids: Sequence[str],
        token_offsets: np.ndarray,
        *,
        record_samples: bool,
    ) -> None:
        if values.shape[0] == 0:
            return
        positive = weights > 0.0
        batch_weight = float(np.sum(weights, dtype=np.float64))
        batch_weight_square = float(np.dot(weights, weights))
        if batch_weight > 0.0:
            active_values = values[positive]
            active_weights = weights[positive]
            batch_mean = np.einsum("n,nd->d", active_weights, active_values) / batch_weight
            if self.retain_covariance:
                batch_raw_second = _weighted_outer(
                    active_values, active_weights, self.covariance_mode, self.block_size
                )
            else:
                batch_raw_second = None
            self._merge_numeric(
                values.shape[0], batch_weight, batch_weight_square, batch_mean,
                batch_raw_second,
            )
        else:
            self.count += values.shape[0]
        self.document_ids.update(document_ids)
        if record_samples:
            self.sample_keys.update(zip(document_ids, map(int, token_offsets), strict=True))

    def _merge_numeric(
        self,
        count: int,
        weight_sum: float,
        weight_square_sum: float,
        mean: np.ndarray,
        raw_second_numerator: np.ndarray | None,
    ) -> None:
        if weight_sum <= 0.0:
            self.count += count
            return
        if self.weight_sum == 0.0:
            self.count += count
            self.weight_sum = weight_sum
            self.weight_square_sum = weight_square_sum
            self.mean[...] = mean
            if self.retain_covariance:
                assert self.raw_second_numerator is not None and raw_second_numerator is not None
                self.raw_second_numerator[...] = raw_second_numerator
            return
        total = self.weight_sum + weight_sum
        delta = mean - self.mean
        if self.retain_covariance:
            assert self.raw_second_numerator is not None and raw_second_numerator is not None
            self.raw_second_numerator += raw_second_numerator
        self.mean += delta * (weight_sum / total)
        self.count += count
        self.weight_sum = total
        self.weight_square_sum += weight_square_sum

    def merge(self, other: _WeightedMoments, *, record_samples: bool) -> None:
        if (
            self.dimension != other.dimension
            or self.covariance_mode != other.covariance_mode
            or self.block_size != other.block_size
            or self.retain_covariance != other.retain_covariance
        ):
            raise ValueError("cannot merge moment accumulators with different dimensions or geometry")
        overlap = self.sample_keys & other.sample_keys if record_samples else set()
        if overlap:
            example = min(overlap)
            raise ValueError(f"duplicate calibration sample during merge: {example!r}")
        self._merge_numeric(
            other.count, other.weight_sum, other.weight_square_sum, other.mean,
            other.raw_second_numerator,
        )
        self.document_ids.update(other.document_ids)
        if record_samples:
            self.sample_keys.update(other.sample_keys)


@dataclass(frozen=True)
class FittedExpertStatistics:
    """Finalized metadata and numeric arrays for one layer/expert/projection."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def array(self, accounting: str, power: int, field: str) -> np.ndarray:
        prefix = f"{accounting}.p{power}"
        if accounting not in self.metadata["estimator"]["retained_accounting"] or power not in self.metadata["estimator"]["retained_powers"]:
            raise KeyError(f"covariance geometry was not retained for {prefix}")
        if field in STORED_ARRAY_FIELDS:
            return self.arrays[f"{prefix}.{field}"]
        mean = np.asarray(self.arrays[f"{prefix}.mean"], dtype=np.float64)
        second_moment = np.asarray(self.arrays[f"{prefix}.second_moment"], dtype=np.float64)
        mode = self.metadata["estimator"]["covariance_mode"]
        block_size = self.metadata["estimator"]["block_size"]
        covariance = _symmetrize(second_moment - _outer(mean, mode, block_size), mode)
        if field == "covariance":
            return covariance
        if field == "regularized_covariance":
            record = self.metadata["accounting"][accounting]["powers"][str(power)]
            return _oas_style_identity_shrinkage(
                covariance,
                float(record["effective_sample_size"]),
                float(self.metadata["estimator"]["regularization_floor"]),
                dimension=int(self.metadata["identity"]["hidden_size"]),
                covariance_mode=mode,
            )[2]
        if field == "regularized_second_moment":
            record = self.metadata["accounting"][accounting]["powers"][str(power)]
            return _oas_style_identity_shrinkage(
                second_moment,
                float(record["effective_sample_size"]),
                float(self.metadata["estimator"]["regularization_floor"]),
                dimension=int(self.metadata["identity"]["hidden_size"]),
                covariance_mode=mode,
            )[2]
        raise KeyError(f"unknown fitted-statistics field {field!r}")

    def dense_covariance(self, accounting: str, power: int, *, regularized: bool = True) -> np.ndarray:
        """Materialize centered covariance for diagnostics, not codec fitting."""
        field = "regularized_covariance" if regularized else "covariance"
        value = self.array(accounting, power, field)
        mode = self.metadata["estimator"]["covariance_mode"]
        return _materialize_dense(value, mode, int(self.metadata["identity"]["hidden_size"]), int(self.metadata["estimator"]["block_size"]))

    def dense_second_moment(
        self, accounting: str, power: int, *, regularized: bool = False
    ) -> np.ndarray:
        """Return ``sum(w*x*x.T)/sum(w)`` in dense geometry.

        This is the uncentered routed second moment used as the EXL3/MCG
        Hessian proxy.  The default is deliberately raw: codec-level damping
        remains a separately identified operation.  ``regularized=True`` is
        an explicitly labelled OAS-style heuristic, not the exact OAS
        estimator for weighted routed samples.
        """
        field = "regularized_second_moment" if regularized else "second_moment"
        value = self.array(accounting, power, field)
        mode = self.metadata["estimator"]["covariance_mode"]
        return _materialize_dense(
            value,
            mode,
            int(self.metadata["identity"]["hidden_size"]),
            int(self.metadata["estimator"]["block_size"]),
        )

    def dense_hessian(
        self, accounting: str, power: int, *, regularized: bool = False
    ) -> np.ndarray:
        """Explicit codec-facing alias for :meth:`dense_second_moment`."""
        return self.dense_second_moment(accounting, power, regularized=regularized)


class CalibrationFitter:
    """Stream route-aware covariance/vector statistics for one projection.

    The production defaults retain only the combined covariance for all three
    route-weight powers.  Natural and supplemental ledgers still retain exact
    scalar/sample accounting and float64 means, but their redundant covariance
    matrices are not persisted.  Only ``covariance_mode="full"`` is suitable
    for the competitive exact EXL3/MCG codec.  ``block_diagonal`` and
    ``diagonal`` are explicitly disclosed diagnostic approximations.

    Gate and up share an input-statistics fitter because their input vectors
    are identical (2,048 dimensions for Qwen3-30B-A3B).  Down uses a separate
    fitter over post-SwiGLU inputs (768 dimensions).  The dimensions remain
    constructor inputs so a mismatched capture fails rather than being guessed.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        projection: str,
        hidden_size: int,
        predecessor_checkpoint_hash: str,
        source_identities: Mapping[str, str],
        regularization_floor: float = 1e-12,
        retained_accounting: Sequence[str] = ("combined",),
        retained_powers: Sequence[int] = ROUTE_WEIGHT_POWERS,
        covariance_mode: str = "full",
        block_size: int = 128,
        artifact_dtype: str = "float32",
    ) -> None:
        if isinstance(layer_id, bool) or not isinstance(layer_id, (int, np.integer)) or int(layer_id) < 0:
            raise ValueError("layer_id must be a non-negative integer")
        if not isinstance(projection, str) or not projection or projection.strip() != projection:
            raise ValueError("projection must be a non-empty canonical string")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, (int, np.integer)) or int(hidden_size) < 1:
            raise ValueError("hidden_size must be a positive integer")
        if not isinstance(predecessor_checkpoint_hash, str) or not _HASH_RE.fullmatch(predecessor_checkpoint_hash):
            raise ValueError("predecessor_checkpoint_hash must be a lowercase 64-hex SHA256")
        if not math.isfinite(regularization_floor) or regularization_floor <= 0.0:
            raise ValueError("regularization_floor must be finite and positive")
        retained_accounting_tuple = tuple(retained_accounting)
        if (
            not retained_accounting_tuple
            or len(set(retained_accounting_tuple)) != len(retained_accounting_tuple)
            or any(item not in ACCOUNTING_KINDS for item in retained_accounting_tuple)
        ):
            raise ValueError(f"retained_accounting must be unique members of {ACCOUNTING_KINDS}")
        retained_powers_raw = tuple(retained_powers)
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in retained_powers_raw):
            raise ValueError(f"retained_powers must be unique members of {ROUTE_WEIGHT_POWERS}")
        retained_powers_tuple = tuple(int(item) for item in retained_powers_raw)
        if (
            not retained_powers_tuple
            or len(set(retained_powers_tuple)) != len(retained_powers_tuple)
            or any(item not in ROUTE_WEIGHT_POWERS for item in retained_powers_tuple)
        ):
            raise ValueError(f"retained_powers must be unique members of {ROUTE_WEIGHT_POWERS}")
        if covariance_mode not in _COVARIANCE_MODES:
            raise ValueError(f"covariance_mode must be one of {_COVARIANCE_MODES}")
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
            raise ValueError("block_size must be a positive integer")
        if covariance_mode == "block_diagonal" and int(hidden_size) % int(block_size):
            raise ValueError("block-diagonal covariance requires hidden_size divisible by block_size")
        if artifact_dtype not in {"float32", "float64"}:
            raise ValueError("artifact_dtype must be float32 or float64")
        self.layer_id = int(layer_id)
        self.projection = projection
        self.hidden_size = int(hidden_size)
        self.predecessor_checkpoint_hash = predecessor_checkpoint_hash
        self.source_identities = _validate_source_identities(source_identities)
        self.regularization_floor = float(regularization_floor)
        self.retained_accounting = retained_accounting_tuple
        self.retained_powers = retained_powers_tuple
        self.covariance_mode = covariance_mode
        self.block_size = int(block_size)
        self.artifact_dtype = artifact_dtype
        self._experts: dict[str, dict[str, dict[int, _WeightedMoments]]] = {}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "projection": self.projection,
            "hidden_size": self.hidden_size,
            "predecessor_checkpoint_hash": self.predecessor_checkpoint_hash,
            "source_identities": self.source_identities,
            "regularization_floor": self.regularization_floor,
            "retained_accounting": list(self.retained_accounting),
            "retained_powers": list(self.retained_powers),
            "covariance_mode": self.covariance_mode,
            "block_size": self.block_size,
            "artifact_dtype": self.artifact_dtype,
        }

    def _new_expert(self) -> dict[str, dict[int, _WeightedMoments]]:
        return {
            kind: {
                power: _WeightedMoments(
                    self.hidden_size,
                    covariance_mode=self.covariance_mode,
                    block_size=self.block_size,
                    retain_covariance=kind in self.retained_accounting and power in self.retained_powers,
                )
                for power in ROUTE_WEIGHT_POWERS
            }
            for kind in ACCOUNTING_KINDS
        }

    def storage_estimate(self, *, expert_count: int = 1, layer_count: int = 1) -> dict[str, int | str]:
        """Exact numeric-array estimate, excluding small JSON/set overhead."""
        if (
            isinstance(expert_count, bool)
            or not isinstance(expert_count, int)
            or isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or expert_count < 1
            or layer_count < 1
        ):
            raise ValueError("expert_count and layer_count must be positive")
        covariance_elements = math.prod(_covariance_shape(self.hidden_size, self.covariance_mode, self.block_size))
        retained_arms = len(self.retained_accounting) * len(self.retained_powers)
        accumulator_per_expert = (
            len(ACCOUNTING_KINDS) * len(ROUTE_WEIGHT_POWERS) * self.hidden_size * 8
            + retained_arms * covariance_elements * 8
        )
        artifact_itemsize = np.dtype(self.artifact_dtype).itemsize
        artifact_per_expert = retained_arms * (
            self.hidden_size + covariance_elements
        ) * artifact_itemsize
        return {
            "covariance_mode": self.covariance_mode,
            "artifact_dtype": self.artifact_dtype,
            "accumulator_bytes_per_expert": accumulator_per_expert,
            "accumulator_bytes_one_layer": accumulator_per_expert * expert_count,
            "artifact_bytes_per_expert": artifact_per_expert,
            "artifact_bytes_total": artifact_per_expert * expert_count * layer_count,
        }

    def update(self, batch: CalibrationBatch | None = None, **kwargs: Any) -> None:
        """Validate and add a batch transactionally.

        Keyword arguments are accepted as a convenience and are used to build
        :class:`CalibrationBatch`; callers may not mix both forms.
        """
        if batch is None:
            batch = CalibrationBatch(**kwargs)
        elif kwargs:
            raise TypeError("pass either CalibrationBatch or keyword fields, not both")
        if not isinstance(batch, CalibrationBatch):
            raise TypeError("batch must be CalibrationBatch")
        if batch.layer_id != self.layer_id or batch.projection != self.projection:
            raise ValueError("batch layer/projection identity drift")
        if batch.predecessor_checkpoint_hash != self.predecessor_checkpoint_hash:
            raise ValueError("batch predecessor checkpoint identity drift")

        values_raw = _as_numpy(batch.expert_inputs)
        if values_raw.ndim != 2 or values_raw.shape[1] != self.hidden_size:
            raise ValueError(f"expert_inputs must have shape [rows, {self.hidden_size}]")
        _require_finite("expert_inputs", values_raw)
        values = np.asarray(values_raw, dtype=np.float64)
        rows = values.shape[0]

        expert_ids_raw = _as_numpy(batch.expert_ids)
        if expert_ids_raw.ndim != 1 or expert_ids_raw.shape[0] != rows:
            raise ValueError("expert_ids must be one-dimensional with one value per row")
        expert_ids = [_normalise_expert_id(item) for item in expert_ids_raw.tolist()]

        route_weights_raw = _as_numpy(batch.route_weights)
        if route_weights_raw.dtype != np.float32:
            raise TypeError("route_weights must be FP32")
        if route_weights_raw.ndim != 1 or route_weights_raw.shape[0] != rows:
            raise ValueError("route_weights must be one-dimensional with one value per row")
        _require_finite("route_weights", route_weights_raw)
        route_weights = route_weights_raw.astype(np.float64)
        if np.any(route_weights < 0.0) or np.any(route_weights > 1.0):
            raise ValueError("route_weights must lie in [0, 1]")

        document_ids = list(batch.document_ids)
        if len(document_ids) != rows or any(
            not isinstance(item, str) or not item or item.strip() != item for item in document_ids
        ):
            raise ValueError("document_ids must contain one non-empty canonical string per row")
        offsets_raw = _as_numpy(batch.token_offsets)
        if offsets_raw.ndim != 1 or offsets_raw.shape[0] != rows or not np.issubdtype(offsets_raw.dtype, np.integer):
            raise ValueError("token_offsets must be a one-dimensional integer array with one value per row")
        token_offsets = offsets_raw.astype(np.int64)
        if np.any(token_offsets < 0):
            raise ValueError("token_offsets must be non-negative")

        if isinstance(batch.origins, str):
            origins = np.full(rows, batch.origins, dtype=object)
        else:
            origins = np.asarray(list(batch.origins), dtype=object)
        if origins.ndim != 1 or origins.shape[0] != rows:
            raise ValueError("origins must be scalar or contain one value per row")
        if any(item not in {"natural", "supplemental"} for item in origins.tolist()):
            raise ValueError("origins must contain only 'natural' or 'supplemental'")

        if batch.inclusion_probabilities is None:
            if np.any(origins == "supplemental"):
                raise ValueError("supplemental rows require inclusion_probabilities")
            inclusion = np.ones(rows, dtype=np.float64)
        else:
            inclusion_raw = _as_numpy(batch.inclusion_probabilities)
            if inclusion_raw.ndim != 1 or inclusion_raw.shape[0] != rows:
                raise ValueError("inclusion_probabilities must contain one value per row")
            _require_finite("inclusion_probabilities", inclusion_raw)
            inclusion = inclusion_raw.astype(np.float64)
            if np.any(inclusion <= 0.0) or np.any(inclusion > 1.0):
                raise ValueError("inclusion_probabilities must lie in (0, 1]")
            if np.any(inclusion[origins == "natural"] != 1.0):
                raise ValueError("natural rows must have inclusion probability exactly one")

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            importance_correction = np.where(origins == "supplemental", 1.0 / inclusion, 1.0)
            for power in ROUTE_WEIGHT_POWERS:
                corrected = np.power(route_weights, power, dtype=np.float64) * importance_correction
                if not np.isfinite(corrected).all() or not np.isfinite(np.square(corrected)).all():
                    raise ValueError("importance-corrected route weights overflow float64 sufficient statistics")

        # Validate uniqueness against existing state before any numeric mutation.
        incoming: dict[str, set[tuple[str, int]]] = {}
        for expert_id, document_id, offset in zip(expert_ids, document_ids, token_offsets, strict=True):
            key = (document_id, int(offset))
            keys = incoming.setdefault(expert_id, set())
            if key in keys:
                raise ValueError(f"duplicate calibration sample in batch: expert={expert_id!r}, sample={key!r}")
            keys.add(key)
        for expert_id, keys in incoming.items():
            existing = self._experts.get(expert_id)
            if existing is not None:
                overlap = existing["combined"][0].sample_keys & keys
                if overlap:
                    raise ValueError(f"duplicate calibration sample across updates: expert={expert_id!r}, sample={min(overlap)!r}")

        expert_ids_array = np.asarray(expert_ids, dtype=object)
        if np.any(route_weights <= 0.0):
            raise ValueError(
                "routed calibration rows must have strictly positive route_weights; "
                "filter unrouted/padded rows with the capture routed mask"
            )

        for expert_id in sorted(incoming, key=_expert_sort_key):
            state = self._experts.setdefault(expert_id, self._new_expert())
            expert_mask = expert_ids_array == expert_id
            for origin in ("natural", "supplemental"):
                mask = expert_mask & (origins == origin)
                if not np.any(mask):
                    continue
                selected_values = values[mask]
                selected_route = route_weights[mask]
                selected_documents = [document_ids[index] for index in np.flatnonzero(mask)]
                selected_offsets = token_offsets[mask]
                selected_inclusion = inclusion[mask]
                for power in ROUTE_WEIGHT_POWERS:
                    raw_weights = np.power(selected_route, power, dtype=np.float64)
                    if origin == "natural":
                        state["natural"][power].update(
                            selected_values, raw_weights, selected_documents, selected_offsets, record_samples=True
                        )
                        state["combined"][power].update(
                            selected_values, raw_weights, selected_documents, selected_offsets, record_samples=True
                        )
                    else:
                        corrected_weights = raw_weights / selected_inclusion
                        state["supplemental_raw"][power].update(
                            selected_values, raw_weights, selected_documents, selected_offsets, record_samples=True
                        )
                        state["supplemental_corrected"][power].update(
                            selected_values, corrected_weights, selected_documents, selected_offsets, record_samples=True
                        )
                        state["combined"][power].update(
                            selected_values, corrected_weights, selected_documents, selected_offsets, record_samples=True
                        )

    def merge(self, other: CalibrationFitter) -> CalibrationFitter:
        """Merge another fitter into this one, rejecting every identity drift."""
        if not isinstance(other, CalibrationFitter):
            raise TypeError("other must be CalibrationFitter")
        if other is self:
            raise ValueError("cannot merge a fitter with itself")
        if self.identity != other.identity:
            raise ValueError("cannot merge fitters with different identities")
        # Check all overlaps first, keeping merge failure transactional.
        for expert_id in self._experts.keys() & other._experts.keys():
            overlap = self._experts[expert_id]["combined"][0].sample_keys & other._experts[expert_id]["combined"][0].sample_keys
            if overlap:
                raise ValueError(f"duplicate calibration sample during merge: expert={expert_id!r}, sample={min(overlap)!r}")
        for expert_id in sorted(other._experts, key=_expert_sort_key):
            if expert_id not in self._experts:
                self._experts[expert_id] = self._new_expert()
            for kind in ACCOUNTING_KINDS:
                for power in ROUTE_WEIGHT_POWERS:
                    self._experts[expert_id][kind][power].merge(
                        other._experts[expert_id][kind][power], record_samples=True
                    )
        return self

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._experts, key=_expert_sort_key))

    def finalize(self, expert_id: int | str) -> FittedExpertStatistics:
        """Finalize one expert without changing the mergeable accumulator."""
        normalized = _normalise_expert_id(expert_id)
        if normalized not in self._experts:
            raise KeyError(f"expert {normalized!r} has no observations")
        state = self._experts[normalized]
        arrays: dict[str, np.ndarray] = {}
        accounting: dict[str, Any] = {}
        for kind in ACCOUNTING_KINDS:
            power_metadata: dict[str, Any] = {}
            for power in ROUTE_WEIGHT_POWERS:
                moments = state[kind][power]
                effective_n = (
                    moments.weight_sum * moments.weight_sum / moments.weight_square_sum
                    if moments.weight_square_sum > 0.0
                    else 0.0
                )
                retained = moments.retain_covariance
                prefix = f"{kind}.p{power}"
                if retained:
                    assert moments.raw_second_numerator is not None
                    second_moment = (
                        moments.raw_second_numerator / moments.weight_sum
                        if moments.weight_sum > 0.0
                        else np.zeros_like(moments.raw_second_numerator)
                    )
                    second_moment = _symmetrize(second_moment, self.covariance_mode)
                    output_dtype = np.dtype(self.artifact_dtype)
                    arrays[f"{prefix}.mean"] = moments.mean.astype(output_dtype)
                    arrays[f"{prefix}.second_moment"] = second_moment.astype(output_dtype)
                    # Diagnostic shrinkage is defined on the persisted
                    # diagnostic covariance, while the codec consumes the
                    # independently persisted raw second moment above.
                    stored_covariance = _symmetrize(
                        arrays[f"{prefix}.second_moment"].astype(np.float64)
                        - _outer(
                            arrays[f"{prefix}.mean"].astype(np.float64),
                            self.covariance_mode,
                            self.block_size,
                        ),
                        self.covariance_mode,
                    )
                    alpha, target_scale, _ = _oas_style_identity_shrinkage(
                        stored_covariance,
                        effective_n,
                        self.regularization_floor,
                        dimension=self.hidden_size,
                        covariance_mode=self.covariance_mode,
                    )
                else:
                    alpha = None
                    target_scale = None
                sample_keys = sorted(f"{document_id}\0{offset}" for document_id, offset in moments.sample_keys)
                power_metadata[str(power)] = {
                    "matrix_retained": retained,
                    "count": moments.count,
                    "document_count": len(moments.document_ids),
                    "sample_count": len(moments.sample_keys),
                    "sample_keys_sha256": sha256_bytes(canonical_json(sample_keys)),
                    "weight_sum": moments.weight_sum,
                    "weight_square_sum": moments.weight_square_sum,
                    "effective_sample_size": effective_n,
                    "shrinkage_coefficient": alpha,
                    "shrinkage_target": "scaled_identity",
                    "shrinkage_target_scale": target_scale,
                }
            accounting[kind] = {"powers": power_metadata}
        metadata = {
            "schema": SCHEMA,
            "identity": {
                "layer_id": self.layer_id,
                "expert_id": normalized,
                "projection": self.projection,
                "hidden_size": self.hidden_size,
                "predecessor_checkpoint_hash": self.predecessor_checkpoint_hash,
                "source_identities": self.source_identities,
            },
            "estimator": {
                "accumulator_dtype": "float64",
                "artifact_array_dtype": self.artifact_dtype,
                "merge_comparison_tolerance": {"rtol": 1e-12, "atol": 1e-12},
                "covariance": "centered_diagnostic_derived_from_persisted_raw_second_moment",
                "route_weight_powers": list(ROUTE_WEIGHT_POWERS),
                "retained_accounting": list(self.retained_accounting),
                "retained_powers": list(self.retained_powers),
                "covariance_mode": self.covariance_mode,
                "block_size": self.block_size,
                "stored_array_fields": list(STORED_ARRAY_FIELDS),
                "derived_array_fields": ["regularized_covariance", "regularized_second_moment"],
                "supplemental_correction": "inverse_inclusion_probability",
                "combined_accounting": "natural_plus_supplemental_corrected",
                "regularization": "oas_style_heuristic_scaled_identity_for_weighted_routed_moments",
                "regularization_floor": self.regularization_floor,
            },
            "accounting": accounting,
        }
        result = FittedExpertStatistics(metadata=metadata, arrays=arrays)
        verify_fitted_statistics(result)
        return result

    def finalize_all(self) -> dict[str, FittedExpertStatistics]:
        return {expert_id: self.finalize(expert_id) for expert_id in self.expert_ids}

    def save(self, path: str | Path, expert_id: int | str) -> FittedExpertStatistics:
        result = self.finalize(expert_id)
        save_fitted_statistics(path, result)
        return result

    @staticmethod
    def load(path: str | Path) -> FittedExpertStatistics:
        return load_fitted_statistics(path)

    @staticmethod
    def verify(value: FittedExpertStatistics) -> None:
        verify_fitted_statistics(value)


class TransformVectorSearch:
    """Deterministic multi-draw H128 sign/block-scale search.

    This class owns proposal identity and selection discipline, while callers
    supply the exact-codec proxy and held-out full-expert roundtrip evaluators.
    Competitive artifacts cannot be built without both evaluator stages.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        expert_ids: Sequence[int | str],
        hidden_size: int,
        intermediate_size: int,
        predecessor_checkpoint_hash: str,
        source_identities: Mapping[str, str],
        seed: int,
        draw_count: int = 16,
        block_size: int = 128,
        block_log2_scale_grid: Sequence[float] = (-0.25, 0.0, 0.25),
        codebook_scale: float,
        objective_arm: str,
        heldout_artifact_sha256: str,
        reference_baseline_sha256: str,
        selection_role: str = "selection",
    ) -> None:
        if isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("layer_id must be a non-negative integer")
        normalized_experts = tuple(_normalise_expert_id(item) for item in expert_ids)
        if not normalized_experts or len(set(normalized_experts)) != len(normalized_experts):
            raise ValueError("expert_ids must be non-empty and unique")
        for name, size in (("hidden_size", hidden_size), ("intermediate_size", intermediate_size)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if hidden_size % block_size or intermediate_size % block_size:
            raise ValueError("transform dimensions must be divisible by the H128/block size")
        if not isinstance(predecessor_checkpoint_hash, str) or not _HASH_RE.fullmatch(predecessor_checkpoint_hash):
            raise ValueError("predecessor_checkpoint_hash must be a lowercase 64-hex SHA256")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 2:
            raise ValueError("draw_count must be at least two for a searched artifact")
        grid = tuple(float(item) for item in block_log2_scale_grid)
        if len(grid) < 2 or len(set(grid)) != len(grid) or 0.0 not in grid or any(not math.isfinite(item) or abs(item) > 4.0 for item in grid):
            raise ValueError("block_log2_scale_grid must be finite, unique, include zero, and stay within [-4, 4]")
        if isinstance(codebook_scale, bool) or not isinstance(codebook_scale, (int, float)) or not math.isfinite(float(codebook_scale)) or float(codebook_scale) == 0.0:
            raise ValueError("codebook_scale must be finite and nonzero")
        if objective_arm not in _VECTOR_OBJECTIVE_ARMS:
            raise ValueError(f"objective_arm must be one of {sorted(_VECTOR_OBJECTIVE_ARMS)}")
        if not isinstance(heldout_artifact_sha256, str) or not _HASH_RE.fullmatch(heldout_artifact_sha256):
            raise ValueError("heldout_artifact_sha256 must be a lowercase 64-hex SHA256")
        if not isinstance(reference_baseline_sha256, str) or not _HASH_RE.fullmatch(reference_baseline_sha256):
            raise ValueError("reference_baseline_sha256 must be a lowercase 64-hex SHA256")
        if selection_role != "selection":
            raise ValueError("transform search must use the held-out selection role")
        self.layer_id = layer_id
        self.expert_ids = normalized_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.predecessor_checkpoint_hash = predecessor_checkpoint_hash
        self.source_identities = _validate_source_identities(source_identities)
        self.seed = seed
        self.draw_count = draw_count
        self.block_size = block_size
        self.block_log2_scale_grid = grid
        self.codebook_scale = float(codebook_scale)
        self.objective_arm = objective_arm
        self.heldout_artifact_sha256 = heldout_artifact_sha256
        self.reference_baseline_sha256 = reference_baseline_sha256
        self.selection_role = selection_role
        self._root_digest = sha256_bytes(canonical_json(self.generator_identity))

    @property
    def generator_identity(self) -> dict[str, Any]:
        """Inputs that may change proposals and candidate IDs.

        Held-out data and objective selection are deliberately absent.  They
        belong to evaluation identity and therefore cannot perturb proposals.
        """
        return {
            "generator": "sha256_counter_rademacher_h128_block_log2_v1",
            "layer_id": self.layer_id,
            "expert_ids": list(self.expert_ids),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "source_identities": self.source_identities,
            "seed": self.seed,
            "draw_count": self.draw_count,
            "block_size": self.block_size,
            "block_log2_scale_grid": list(self.block_log2_scale_grid),
            "codebook_scale": self.codebook_scale,
        }

    @property
    def evaluation_identity(self) -> dict[str, Any]:
        return {
            "predecessor_checkpoint_hash": self.predecessor_checkpoint_hash,
            "objective_arm": self.objective_arm,
            "heldout_artifact_sha256": self.heldout_artifact_sha256,
            "reference_baseline_sha256": self.reference_baseline_sha256,
            "selection_role": self.selection_role,
        }

    @property
    def identity(self) -> dict[str, Any]:
        return {"generator": self.generator_identity, "evaluation": self.evaluation_identity}

    def storage_estimate(self) -> dict[str, int | str]:
        winner_elements = 2 * self.hidden_size + 3 * len(self.expert_ids) * self.intermediate_size
        block_scale_elements = 2 * self.hidden_size // self.block_size + 3 * len(self.expert_ids) * self.intermediate_size // self.block_size
        # run() regenerates shortlisted candidates instead of retaining all
        # draws. This is a conservative upper bound for one FP16 candidate,
        # float64 sign/scale temporaries, and one persisted winner copy.
        candidate_working = winner_elements * (2 + 4 * 8) + block_scale_elements * (2 * 8)
        return {
            "generated_vector_bytes_per_draw": winner_elements * 2,
            "generated_block_scale_working_bytes_per_draw": block_scale_elements * 8,
            "peak_generator_bytes_upper_bound": candidate_working,
            "peak_search_vector_bytes_upper_bound": candidate_working + winner_elements * 2,
            "persisted_winner_vector_bytes": winner_elements * 2,
            "candidate_retention_policy": 0,
            "draw_count": self.draw_count,
        }

    def score_receipt(
        self,
        candidate: TransformVectorCandidate,
        *,
        score: float,
        method: str,
        evaluator_code_sha256: str,
        codec_identity_sha256: str,
        artifact_sha256: str,
        rows: int,
        coverage: Mapping[str, Any],
    ) -> SearchScore:
        """Build the only accepted canonical evaluator receipt."""
        if method not in {"exact_codec_proxy", "heldout_full_expert_roundtrip"}:
            raise ValueError("unknown transform-search evaluator method")
        vector_sha256 = {
            name: sha256_bytes(np.asarray(array).tobytes(order="C"))
            for name, array in sorted(candidate.vectors.items())
        }
        body = {
            "schema": SEARCH_SCORE_RECEIPT_SCHEMA,
            "method": method,
            "candidate_id": candidate.candidate_id,
            "vector_sha256": vector_sha256,
            "score": float(score),
            "evaluator_code_sha256": evaluator_code_sha256,
            "codec_identity_sha256": codec_identity_sha256,
            "objective_arm": self.objective_arm,
            "selection_role": "proxy" if method == "exact_codec_proxy" else self.selection_role,
            "artifact_sha256": artifact_sha256,
            "reference_baseline_sha256": self.reference_baseline_sha256,
            "predecessor_checkpoint_hash": self.predecessor_checkpoint_hash,
            "rows": rows,
            "coverage": dict(coverage),
        }
        body["receipt_sha256"] = sha256_bytes(canonical_json(body))
        return _validate_search_score(SearchScore(float(score), body), method, candidate, self)

    def candidate(self, draw_index: int) -> TransformVectorCandidate:
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or not 0 <= draw_index < self.draw_count:
            raise ValueError("draw_index lies outside the declared search")
        vectors: dict[str, np.ndarray] = {}
        scale_hashes: dict[str, str] = {}
        roles = {
            "gate_up_input_shared": (self.hidden_size,),
            "down_output_shared": (self.hidden_size,),
            "gate_output": (len(self.expert_ids), self.intermediate_size),
            "up_output": (len(self.expert_ids), self.intermediate_size),
            "down_input": (len(self.expert_ids), self.intermediate_size),
        }
        for role, shape in roles.items():
            count = math.prod(shape)
            signs = _deterministic_signs(self._root_digest, draw_index, role, count).reshape(shape)
            block_count = count // self.block_size
            if draw_index == 0:
                scales = np.ones(block_count, dtype=np.float64)
            else:
                offsets = _deterministic_scale_offsets(
                    self._root_digest, draw_index, role, block_count, self.block_log2_scale_grid
                )
                scales = np.exp2(offsets - offsets.mean())
            scale_hashes[role] = sha256_bytes(np.asarray(scales, dtype="<f8").tobytes(order="C"))
            expanded = np.repeat(scales, self.block_size).reshape(shape)
            vector_float64 = signs * expanded
            if role in {"gate_up_input_shared", "down_input"}:
                vector_float64 = vector_float64 / (-self.codebook_scale)
            vector = vector_float64.astype(np.float16)
            if not np.isfinite(vector).all() or np.any(vector == 0):
                raise ValueError("generated transform vector is non-finite or singular after FP16 rounding")
            vector.setflags(write=False)
            vectors[role] = vector
        vector_hashes = {name: sha256_bytes(value.tobytes(order="C")) for name, value in sorted(vectors.items())}
        candidate_id = sha256_bytes(
            canonical_json(
                {
                    "root_digest": self._root_digest,
                    "draw_index": draw_index,
                    "vector_sha256": vector_hashes,
                    "block_scale_sha256": scale_hashes,
                }
            )
        )
        return TransformVectorCandidate(
            draw_index=draw_index,
            candidate_id=candidate_id,
            baseline=draw_index == 0,
            vectors=vectors,
            block_scale_sha256=scale_hashes,
        )

    def run(
        self,
        proxy_evaluator: Callable[[TransformVectorCandidate], SearchScore],
        roundtrip_evaluator: Callable[[TransformVectorCandidate], SearchScore],
        *,
        shortlist_count: int = 4,
    ) -> SearchedTransformVectors:
        if not callable(proxy_evaluator) or not callable(roundtrip_evaluator):
            raise TypeError("both exact-codec proxy and held-out roundtrip evaluators are required")
        if isinstance(shortlist_count, bool) or not isinstance(shortlist_count, int) or not 2 <= shortlist_count <= self.draw_count:
            raise ValueError("shortlist_count must be in [2, draw_count]")
        evaluations: list[dict[str, Any]] = []
        for draw_index in range(self.draw_count):
            candidate = self.candidate(draw_index)
            proxy = _validate_search_score(proxy_evaluator(candidate), "exact_codec_proxy", candidate, self)
            evaluations.append(
                {
                    "draw_index": draw_index,
                    "candidate_id": candidate.candidate_id,
                    "baseline": candidate.baseline,
                    "block_scale_sha256": candidate.block_scale_sha256,
                    "proxy": {"score": proxy.score, "evidence": dict(proxy.evidence)},
                    "shortlisted": False,
                    "roundtrip": None,
                }
            )
        ranked = sorted(evaluations, key=lambda item: (item["proxy"]["score"], item["candidate_id"]))
        baseline = next(item for item in evaluations if item["baseline"])
        shortlist_ids = {baseline["candidate_id"]}
        for item in ranked:
            shortlist_ids.add(item["candidate_id"])
            if len(shortlist_ids) == shortlist_count:
                break
        for item in evaluations:
            if item["candidate_id"] in shortlist_ids:
                item["shortlisted"] = True
                candidate = self.candidate(item["draw_index"])
                if candidate.candidate_id != item["candidate_id"]:
                    raise RuntimeError("deterministic transform candidate regeneration drifted")
                score = _validate_search_score(
                    roundtrip_evaluator(candidate), "heldout_full_expert_roundtrip", candidate, self
                )
                item["roundtrip"] = {"score": score.score, "evidence": dict(score.evidence)}
        finalists = [item for item in evaluations if item["roundtrip"] is not None]
        winner_record = min(
            finalists,
            key=lambda item: (item["roundtrip"]["score"], item["proxy"]["score"], item["candidate_id"]),
        )
        winner = self.candidate(winner_record["draw_index"])
        baseline_record = next(item for item in finalists if item["baseline"])
        metadata = {
            "schema": VECTOR_SEARCH_SCHEMA,
            "identity": self.identity,
            "policy": {
                "generator": "sha256_counter_rademacher_h128_block_log2_v1",
                "vector_dtype": "float16",
                "baseline": "rademacher_with_identity_block_g_scales",
                "baseline_scope": "proposal-control-only",
                "production_reference_chain": {
                    "hessian": "raw_route_weighted_uncentered_second_moment",
                    "normalization": "source_derived_absolute_v31",
                    "gss": "selected_bit_pinned",
                    "artifact_sha256": self.reference_baseline_sha256,
                    "relationship": "transform_search_is_additive_and_requires_ablation_against_reference_chain",
                },
                "selection_method": "multidraw_h128_block_gscale_exact_codec_proxy_then_heldout_full_expert_v1",
                "selection": "min_heldout_full_expert_roundtrip_then_exact_codec_proxy_then_candidate_id",
                "codebook_normalization": {
                    "input_vector_roles": ["gate_up_input_shared", "down_input"],
                    "divisor": -self.codebook_scale,
                    "applications": 1,
                },
                "shortlist_count": shortlist_count,
                "baseline_always_confirmed": True,
                "competitive_search_complete": True,
            },
            "evaluations": evaluations,
            "winner": {
                "draw_index": winner.draw_index,
                "candidate_id": winner.candidate_id,
                "baseline": winner.baseline,
                "block_scale_sha256": winner.block_scale_sha256,
                "improves_over_baseline": winner_record["roundtrip"]["score"] < baseline_record["roundtrip"]["score"],
                "proxy_score": winner_record["proxy"]["score"],
                "roundtrip_score": winner_record["roundtrip"]["score"],
            },
            "storage_estimate": self.storage_estimate()
            | {
                "proxy_receipt_count": self.draw_count,
                "roundtrip_receipt_count": shortlist_count,
                "score_evidence_canonical_bytes": len(canonical_json(evaluations)),
            },
        }
        result = SearchedTransformVectors(metadata=metadata, arrays={name: value.copy() for name, value in winner.vectors.items()})
        verify_searched_transform_vectors(result)
        return result


def _hash_stream(root_digest: str, draw_index: int, role: str, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(
            bytes.fromhex(
                sha256_bytes(
                    canonical_json(
                        {"root_digest": root_digest, "draw_index": draw_index, "role": role, "counter": counter}
                    )
                )
            )
        )
        counter += 1
    return bytes(output[:length])


def _deterministic_signs(root_digest: str, draw_index: int, role: str, count: int) -> np.ndarray:
    packed = _hash_stream(root_digest, draw_index, f"{role}:sign", (count + 7) // 8)
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:count]
    return np.where(bits == 0, -1.0, 1.0)


def _deterministic_scale_offsets(
    root_digest: str,
    draw_index: int,
    role: str,
    count: int,
    grid: Sequence[float],
) -> np.ndarray:
    raw = _hash_stream(root_digest, draw_index, f"{role}:block-scale", count * 4)
    values = np.frombuffer(raw, dtype="<u4", count=count)
    choices = np.asarray(grid, dtype=np.float64)
    return choices[values % len(choices)]


def _validate_search_score(
    value: SearchScore,
    required_method: str,
    candidate: TransformVectorCandidate,
    search: TransformVectorSearch,
) -> SearchScore:
    if not isinstance(value, SearchScore):
        raise TypeError("search evaluators must return SearchScore")
    if isinstance(value.score, bool) or not isinstance(value.score, (int, float)) or not math.isfinite(float(value.score)) or float(value.score) < 0.0:
        raise ValueError("search score must be finite and non-negative")
    evidence = dict(value.evidence)
    required_fields = {
        "schema", "method", "candidate_id", "vector_sha256", "score",
        "evaluator_code_sha256", "codec_identity_sha256", "objective_arm",
        "selection_role", "artifact_sha256", "reference_baseline_sha256", "predecessor_checkpoint_hash",
        "rows", "coverage", "receipt_sha256",
    }
    if set(evidence) != required_fields:
        raise ValueError("search evidence does not match canonical receipt schema")
    if evidence["schema"] != SEARCH_SCORE_RECEIPT_SCHEMA or evidence["method"] != required_method:
        raise ValueError(f"search evidence must attest method={required_method!r}")
    vector_sha256 = {
        name: sha256_bytes(np.asarray(array).tobytes(order="C"))
        for name, array in sorted(candidate.vectors.items())
    }
    if evidence["candidate_id"] != candidate.candidate_id or evidence["vector_sha256"] != vector_sha256:
        raise ValueError("search receipt candidate/vector identity mismatch")
    if evidence["score"] != float(value.score):
        raise ValueError("search receipt score mismatch")
    for field in (
        "evaluator_code_sha256", "codec_identity_sha256", "artifact_sha256",
        "reference_baseline_sha256", "predecessor_checkpoint_hash",
    ):
        if not isinstance(evidence[field], str) or not _HASH_RE.fullmatch(evidence[field]):
            raise ValueError(f"search receipt has invalid {field}")
    if evidence["objective_arm"] != search.objective_arm:
        raise ValueError("search receipt objective identity mismatch")
    expected_role = "proxy" if required_method == "exact_codec_proxy" else search.selection_role
    if evidence["selection_role"] != expected_role:
        raise ValueError("search receipt selection-role mismatch")
    if evidence["predecessor_checkpoint_hash"] != search.predecessor_checkpoint_hash:
        raise ValueError("search receipt predecessor identity mismatch")
    if evidence["reference_baseline_sha256"] != search.reference_baseline_sha256:
        raise ValueError("search receipt v31 reference-baseline identity mismatch")
    if required_method == "heldout_full_expert_roundtrip" and evidence["artifact_sha256"] != search.heldout_artifact_sha256:
        raise ValueError("held-out receipt artifact identity mismatch")
    rows = evidence["rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError("search receipt rows must be a positive integer")
    coverage = evidence["coverage"]
    coverage_fields = {"layer_id", "expert_ids", "projection_roles", "row_identity_sha256"}
    if not isinstance(coverage, dict) or set(coverage) != coverage_fields:
        raise ValueError("search receipt coverage is incomplete or unknown")
    if (
        coverage["layer_id"] != search.layer_id
        or coverage["expert_ids"] != list(search.expert_ids)
        or coverage["projection_roles"] != ["gate", "up", "down"]
        or not isinstance(coverage["row_identity_sha256"], str)
        or not _HASH_RE.fullmatch(coverage["row_identity_sha256"])
    ):
        raise ValueError("search receipt coverage identity mismatch")
    receipt = evidence["receipt_sha256"]
    if not isinstance(receipt, str) or not _HASH_RE.fullmatch(receipt):
        raise ValueError("search evidence must bind a receipt_sha256")
    receipt_body = dict(evidence)
    del receipt_body["receipt_sha256"]
    if sha256_bytes(canonical_json(receipt_body)) != receipt:
        raise ValueError("search evidence receipt content hash mismatch")
    canonical_json(evidence)
    return SearchScore(float(value.score), evidence)


def _symmetrize(covariance: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full":
        return (covariance + covariance.T) * 0.5
    if mode == "block_diagonal":
        return (covariance + np.swapaxes(covariance, -1, -2)) * 0.5
    if mode == "diagonal":
        return covariance
    raise ValueError(f"unknown covariance mode {mode!r}")


def _oas_style_identity_shrinkage(
    covariance: np.ndarray,
    effective_n: float,
    floor: float,
    *,
    dimension: int,
    covariance_mode: str,
) -> tuple[float, float, np.ndarray]:
    """Return an OAS-style heuristic toward an expert-local scaled identity.

    Classical OAS assumes unweighted independent samples. Routed and
    inverse-probability-corrected observations violate that assumption, so the
    effective-sample-size substitution is explicitly heuristic.
    """
    if covariance_mode == "full":
        trace = float(np.trace(covariance))
    elif covariance_mode == "block_diagonal":
        trace = float(np.trace(covariance, axis1=-2, axis2=-1).sum())
    elif covariance_mode == "diagonal":
        trace = float(covariance.sum())
    else:
        raise ValueError(f"unknown covariance mode {covariance_mode!r}")
    trace_square = float(np.sum(np.square(covariance), dtype=np.float64))
    target_scale = max(trace / dimension, floor)
    centered_energy = trace_square - trace * trace / dimension
    denominator = (effective_n + 1.0 - 2.0 / dimension) * centered_energy
    numerator = (1.0 - 2.0 / dimension) * trace_square + trace * trace
    if effective_n <= 1.0 or denominator <= 0.0 or not math.isfinite(numerator / denominator):
        alpha = 1.0
    else:
        alpha = min(1.0, max(0.0, numerator / denominator))
    regularized = (1.0 - alpha) * covariance
    if covariance_mode == "full":
        regularized.flat[:: dimension + 1] += alpha * target_scale
    elif covariance_mode == "block_diagonal":
        regularized[:, np.arange(regularized.shape[1]), np.arange(regularized.shape[2])] += alpha * target_scale
    else:
        regularized += alpha * target_scale
    regularized = _symmetrize(regularized, covariance_mode)
    return float(alpha), float(target_scale), regularized


def verify_fitted_statistics(value: FittedExpertStatistics) -> None:
    if not isinstance(value, FittedExpertStatistics):
        raise TypeError("value must be FittedExpertStatistics")
    metadata = value.metadata
    if not isinstance(metadata, dict) or set(metadata) != {"schema", "identity", "estimator", "accounting"}:
        raise ValueError("fitted-statistics metadata fields are incomplete or unknown")
    if metadata["schema"] != SCHEMA:
        raise ValueError("unsupported fitted-statistics schema")
    identity = metadata["identity"]
    identity_fields = {"layer_id", "expert_id", "projection", "hidden_size", "predecessor_checkpoint_hash", "source_identities"}
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError("fitted statistics are missing identity")
    hidden_size = identity["hidden_size"]
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("invalid fitted-statistics hidden size")
    if isinstance(identity["layer_id"], bool) or not isinstance(identity["layer_id"], int) or identity["layer_id"] < 0:
        raise ValueError("invalid fitted-statistics layer identity")
    if not isinstance(identity["expert_id"], str) or _normalise_expert_id(identity["expert_id"]) != identity["expert_id"]:
        raise ValueError("invalid fitted-statistics expert identity")
    if not isinstance(identity["projection"], str) or not identity["projection"] or identity["projection"].strip() != identity["projection"]:
        raise ValueError("invalid fitted-statistics projection identity")
    if not isinstance(identity["predecessor_checkpoint_hash"], str) or not _HASH_RE.fullmatch(identity["predecessor_checkpoint_hash"]):
        raise ValueError("invalid fitted-statistics predecessor identity")
    if _validate_source_identities(identity["source_identities"]) != identity["source_identities"]:
        raise ValueError("fitted-statistics source identities are not canonical")

    estimator = metadata["estimator"]
    estimator_fields = {
        "accumulator_dtype", "artifact_array_dtype", "merge_comparison_tolerance", "covariance",
        "route_weight_powers", "retained_accounting", "retained_powers", "covariance_mode", "block_size",
        "stored_array_fields", "derived_array_fields", "supplemental_correction", "combined_accounting",
        "regularization", "regularization_floor",
    }
    if not isinstance(estimator, dict) or set(estimator) != estimator_fields:
        raise ValueError("fitted-statistics estimator metadata is incomplete or unknown")
    fixed = {
        "accumulator_dtype": "float64",
        "merge_comparison_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "covariance": "centered_diagnostic_derived_from_persisted_raw_second_moment",
        "route_weight_powers": list(ROUTE_WEIGHT_POWERS),
        "stored_array_fields": list(STORED_ARRAY_FIELDS),
        "derived_array_fields": ["regularized_covariance", "regularized_second_moment"],
        "supplemental_correction": "inverse_inclusion_probability",
        "combined_accounting": "natural_plus_supplemental_corrected",
        "regularization": "oas_style_heuristic_scaled_identity_for_weighted_routed_moments",
    }
    if any(estimator.get(key) != expected for key, expected in fixed.items()):
        raise ValueError("invalid fitted-statistics estimator policy")
    artifact_dtype = estimator["artifact_array_dtype"]
    if artifact_dtype not in {"float32", "float64"}:
        raise ValueError("invalid fitted-statistics artifact dtype")
    retained_accounting = estimator["retained_accounting"]
    retained_powers = estimator["retained_powers"]
    if not isinstance(retained_accounting, list) or not retained_accounting or len(set(retained_accounting)) != len(retained_accounting) or any(item not in ACCOUNTING_KINDS for item in retained_accounting):
        raise ValueError("invalid retained accounting")
    if not isinstance(retained_powers, list) or not retained_powers or len(set(retained_powers)) != len(retained_powers) or any(item not in ROUTE_WEIGHT_POWERS for item in retained_powers):
        raise ValueError("invalid retained route powers")
    mode = estimator["covariance_mode"]
    block_size = estimator["block_size"]
    if mode not in _COVARIANCE_MODES or isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("invalid covariance geometry")
    if mode == "block_diagonal" and hidden_size % block_size:
        raise ValueError("invalid block-diagonal covariance geometry")
    floor = estimator["regularization_floor"]
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not math.isfinite(float(floor)) or float(floor) <= 0.0:
        raise ValueError("invalid fitted-statistics regularization floor")

    expected_keys = {
        f"{kind}.p{power}.{field}"
        for kind in retained_accounting
        for power in retained_powers
        for field in STORED_ARRAY_FIELDS
    }
    if set(value.arrays) != expected_keys:
        raise ValueError("fitted-statistics array set is incomplete or contains unknown arrays")
    covariance_shape = _covariance_shape(hidden_size, mode, block_size)
    accounting = metadata["accounting"]
    if not isinstance(accounting, dict) or set(accounting) != set(ACCOUNTING_KINDS):
        raise ValueError("fitted-statistics accounting is incomplete")
    for kind in ACCOUNTING_KINDS:
        powers = accounting[kind].get("powers") if isinstance(accounting[kind], dict) else None
        if not isinstance(powers, dict) or set(powers) != {str(power) for power in ROUTE_WEIGHT_POWERS}:
            raise ValueError("fitted-statistics route powers are incomplete")
        for power in ROUTE_WEIGHT_POWERS:
            record = powers[str(power)]
            record_fields = {
                "matrix_retained", "count", "document_count", "sample_count", "sample_keys_sha256",
                "weight_sum", "weight_square_sum", "effective_sample_size", "shrinkage_coefficient",
                "shrinkage_target", "shrinkage_target_scale",
            }
            if not isinstance(record, dict) or set(record) != record_fields:
                raise ValueError(f"malformed fitted-statistics record {kind}.p{power}")
            retained = kind in retained_accounting and power in retained_powers
            if record["matrix_retained"] is not retained:
                raise ValueError("matrix-retention metadata mismatch")
            for name in ("weight_sum", "weight_square_sum", "effective_sample_size"):
                number = record[name]
                if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(float(number)) or float(number) < 0.0:
                    raise ValueError(f"non-finite or invalid fitted-statistics scalar {kind}.p{power}.{name}")
            for name in ("count", "document_count", "sample_count"):
                if not isinstance(record[name], int) or isinstance(record[name], bool) or record[name] < 0:
                    raise ValueError(f"invalid fitted-statistics count {kind}.p{power}.{name}")
            if record["document_count"] > record["sample_count"] or record["sample_count"] != record["count"]:
                raise ValueError(f"inconsistent fitted-statistics counts in {kind}.p{power}")
            if not isinstance(record["sample_keys_sha256"], str) or not _HASH_RE.fullmatch(record["sample_keys_sha256"]):
                raise ValueError("invalid fitted-statistics sample identity hash")
            if record["shrinkage_target"] != "scaled_identity":
                raise ValueError("invalid fitted-statistics shrinkage target")
            if retained:
                for name in ("shrinkage_coefficient", "shrinkage_target_scale"):
                    number = record[name]
                    if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(float(number)):
                        raise ValueError(f"invalid retained shrinkage scalar {kind}.p{power}.{name}")
                if not 0.0 <= float(record["shrinkage_coefficient"]) <= 1.0:
                    raise ValueError("shrinkage coefficient lies outside [0, 1]")
                mean = np.asarray(value.arrays[f"{kind}.p{power}.mean"])
                second_moment = np.asarray(value.arrays[f"{kind}.p{power}.second_moment"])
                if mean.dtype != np.dtype(artifact_dtype) or mean.shape != (hidden_size,):
                    raise ValueError(f"invalid {kind}.p{power}.mean dtype or dimensions")
                if second_moment.dtype != np.dtype(artifact_dtype) or second_moment.shape != covariance_shape:
                    raise ValueError(f"invalid {kind}.p{power}.second_moment dtype or dimensions")
                if not np.isfinite(mean).all() or not np.isfinite(second_moment).all():
                    raise ValueError(f"non-finite retained arrays in {kind}.p{power}")
                if not np.allclose(second_moment, _symmetrize(second_moment, mode), rtol=0.0, atol=1e-6 if artifact_dtype == "float32" else 1e-12):
                    raise ValueError(f"non-symmetric second moment in {kind}.p{power}")
                covariance = _symmetrize(
                    second_moment.astype(np.float64)
                    - _outer(mean.astype(np.float64), mode, block_size),
                    mode,
                )
                alpha, scale, _ = _oas_style_identity_shrinkage(
                    covariance.astype(np.float64), float(record["effective_sample_size"]), float(floor),
                    dimension=hidden_size, covariance_mode=mode,
                )
                tolerance = 1e-6 if artifact_dtype == "float32" else 1e-12
                if not math.isclose(float(record["shrinkage_coefficient"]), alpha, rel_tol=tolerance, abs_tol=tolerance):
                    raise ValueError(f"shrinkage coefficient mismatch in {kind}.p{power}")
                if not math.isclose(float(record["shrinkage_target_scale"]), scale, rel_tol=tolerance, abs_tol=tolerance):
                    raise ValueError(f"shrinkage scale mismatch in {kind}.p{power}")
            elif record["shrinkage_coefficient"] is not None or record["shrinkage_target_scale"] is not None:
                raise ValueError("unretained matrix arm must not claim shrinkage results")

    for power in ROUTE_WEIGHT_POWERS:
        natural = accounting["natural"]["powers"][str(power)]
        supplemental_raw = accounting["supplemental_raw"]["powers"][str(power)]
        supplemental_corrected = accounting["supplemental_corrected"]["powers"][str(power)]
        combined = accounting["combined"]["powers"][str(power)]
        if any(supplemental_raw[name] != supplemental_corrected[name] for name in ("count", "document_count", "sample_count", "sample_keys_sha256")):
            raise ValueError("supplemental raw/corrected sample accounting mismatch")
        if combined["count"] != natural["count"] + supplemental_corrected["count"]:
            raise ValueError("combined natural/supplemental count mismatch")
        for name in ("weight_sum", "weight_square_sum"):
            if not math.isclose(float(combined[name]), float(natural[name]) + float(supplemental_corrected[name]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"combined natural/supplemental {name} mismatch")
        expected_effective = float(combined["weight_sum"]) ** 2 / float(combined["weight_square_sum"]) if float(combined["weight_square_sum"]) > 0.0 else 0.0
        if not math.isclose(float(combined["effective_sample_size"]), expected_effective, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("combined effective sample size mismatch")


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def save_fitted_statistics(path: str | Path, value: FittedExpertStatistics) -> None:
    verify_fitted_statistics(value)
    destination = prepare_empty_destination(path)
    arrays_manifest: dict[str, Any] = {}
    for index, name in enumerate(sorted(value.arrays)):
        filename = f"array-{index:02d}.npy"
        target = destination / filename
        _atomic_save_npy(target, value.arrays[name])
        arrays_manifest[name] = {
            "file": filename,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "dtype": str(value.arrays[name].dtype),
            "shape": list(value.arrays[name].shape),
        }
    manifest = dict(value.metadata)
    manifest["arrays"] = arrays_manifest
    manifest["seal_sha256"] = sha256_bytes(canonical_json(manifest))
    write_json(destination / "manifest.json", manifest)


def load_fitted_statistics(path: str | Path) -> FittedExpertStatistics:
    import json

    source = Path(path)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fitted-statistics manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = manifest.pop("seal_sha256", None)
    if not isinstance(seal, str) or not _HASH_RE.fullmatch(seal) or sha256_bytes(canonical_json(manifest)) != seal:
        raise ValueError("fitted-statistics manifest seal mismatch")
    arrays_manifest = manifest.pop("arrays", None)
    if not isinstance(arrays_manifest, dict):
        raise ValueError("fitted-statistics manifest has no array inventory")
    arrays: dict[str, np.ndarray] = {}
    filenames: set[str] = set()
    for name, record in arrays_manifest.items():
        if not isinstance(record, dict) or set(record) != {"file", "bytes", "sha256", "dtype", "shape"}:
            raise ValueError(f"malformed fitted-statistics array record {name!r}")
        filename = record["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("fitted-statistics array paths must be local filenames")
        if filename in filenames:
            raise ValueError("fitted-statistics array files must be unique")
        filenames.add(filename)
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] < 1
            or not isinstance(record["sha256"], str)
            or not _HASH_RE.fullmatch(record["sha256"])
            or record["dtype"] not in {"float32", "float64"}
            or not isinstance(record["shape"], list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in record["shape"])
        ):
            raise ValueError(f"malformed fitted-statistics array metadata for {name!r}")
        array_path = source / filename
        if not array_path.is_file() or array_path.stat().st_size != record["bytes"] or sha256_file(array_path) != record["sha256"]:
            raise ValueError(f"fitted-statistics array identity mismatch for {name!r}")
        array = np.load(array_path, allow_pickle=False, mmap_mode="r")
        if str(array.dtype) != record["dtype"] or list(array.shape) != record["shape"]:
            raise ValueError(f"fitted-statistics array metadata mismatch for {name!r}")
        arrays[name] = array
    expected_files = {"manifest.json", *filenames}
    if {item.name for item in source.iterdir()} != expected_files:
        raise ValueError("fitted-statistics directory contains missing or unbound files")
    result = FittedExpertStatistics(metadata=manifest, arrays=arrays)
    verify_fitted_statistics(result)
    return result


def verify_searched_transform_vectors(value: SearchedTransformVectors) -> None:
    if not isinstance(value, SearchedTransformVectors):
        raise TypeError("value must be SearchedTransformVectors")
    metadata = value.metadata
    if not isinstance(metadata, dict) or set(metadata) != {"schema", "identity", "policy", "evaluations", "winner", "storage_estimate"}:
        raise ValueError("transform-search metadata fields are incomplete or unknown")
    if metadata["schema"] != VECTOR_SEARCH_SCHEMA:
        raise ValueError("unsupported transform-vector search schema")
    identity = metadata["identity"]
    if not isinstance(identity, dict) or set(identity) != {"generator", "evaluation"}:
        raise ValueError("transform search identity is missing")
    generator_identity = identity["generator"]
    evaluation_identity = identity["evaluation"]
    if not isinstance(generator_identity, dict) or not isinstance(evaluation_identity, dict):
        raise ValueError("transform search generator/evaluation identity is malformed")
    try:
        search = TransformVectorSearch(
            layer_id=generator_identity["layer_id"],
            expert_ids=generator_identity["expert_ids"],
            hidden_size=generator_identity["hidden_size"],
            intermediate_size=generator_identity["intermediate_size"],
            predecessor_checkpoint_hash=evaluation_identity["predecessor_checkpoint_hash"],
            source_identities=generator_identity["source_identities"],
            seed=generator_identity["seed"],
            draw_count=generator_identity["draw_count"],
            block_size=generator_identity["block_size"],
            block_log2_scale_grid=generator_identity["block_log2_scale_grid"],
            codebook_scale=generator_identity["codebook_scale"],
            objective_arm=evaluation_identity["objective_arm"],
            heldout_artifact_sha256=evaluation_identity["heldout_artifact_sha256"],
            reference_baseline_sha256=evaluation_identity["reference_baseline_sha256"],
            selection_role=evaluation_identity["selection_role"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid transform search identity") from error
    if search.identity != identity:
        raise ValueError("transform search identity mismatch")
    expected_policy = {
        "generator": "sha256_counter_rademacher_h128_block_log2_v1",
        "vector_dtype": "float16",
        "baseline": "rademacher_with_identity_block_g_scales",
        "baseline_scope": "proposal-control-only",
            "production_reference_chain": {
                "hessian": "raw_route_weighted_uncentered_second_moment",
                "normalization": "source_derived_absolute_v31",
                "gss": "selected_bit_pinned",
                "artifact_sha256": search.reference_baseline_sha256,
                "relationship": "transform_search_is_additive_and_requires_ablation_against_reference_chain",
            },
        "selection_method": "multidraw_h128_block_gscale_exact_codec_proxy_then_heldout_full_expert_v1",
        "selection": "min_heldout_full_expert_roundtrip_then_exact_codec_proxy_then_candidate_id",
        "codebook_normalization": {
            "input_vector_roles": ["gate_up_input_shared", "down_input"],
            "divisor": -search.codebook_scale,
            "applications": 1,
        },
        "shortlist_count": metadata.get("policy", {}).get("shortlist_count"),
        "baseline_always_confirmed": True,
        "competitive_search_complete": True,
    }
    policy = metadata["policy"]
    if not isinstance(policy, dict) or policy != expected_policy:
        raise ValueError("invalid transform search policy")
    shortlist_count = policy["shortlist_count"]
    if isinstance(shortlist_count, bool) or not isinstance(shortlist_count, int) or not 2 <= shortlist_count <= search.draw_count:
        raise ValueError("invalid transform search shortlist count")
    evaluations = metadata["evaluations"]
    if not isinstance(evaluations, list) or len(evaluations) != search.draw_count:
        raise ValueError("transform search evaluation count mismatch")
    evaluation_fields = {"draw_index", "candidate_id", "baseline", "block_scale_sha256", "proxy", "shortlisted", "roundtrip"}
    seen_draws: set[int] = set()
    seen_ids: set[str] = set()
    baseline_record: dict[str, Any] | None = None
    finalists: list[dict[str, Any]] = []
    for record in evaluations:
        if not isinstance(record, dict) or set(record) != evaluation_fields:
            raise ValueError("malformed transform search evaluation")
        draw_index = record["draw_index"]
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index in seen_draws:
            raise ValueError("duplicate or invalid transform search draw")
        candidate = search.candidate(draw_index)
        if (
            record["candidate_id"] != candidate.candidate_id
            or record["candidate_id"] in seen_ids
            or record["baseline"] is not candidate.baseline
            or record["block_scale_sha256"] != candidate.block_scale_sha256
        ):
            raise ValueError("transform search candidate identity mismatch")
        seen_draws.add(draw_index)
        seen_ids.add(record["candidate_id"])
        proxy = record["proxy"]
        if not isinstance(proxy, dict) or set(proxy) != {"score", "evidence"}:
            raise ValueError("malformed transform search proxy score")
        _validate_search_score(
            SearchScore(proxy["score"], proxy["evidence"]), "exact_codec_proxy", candidate, search
        )
        if not isinstance(record["shortlisted"], bool):
            raise ValueError("invalid transform search shortlist flag")
        if record["shortlisted"]:
            roundtrip = record["roundtrip"]
            if not isinstance(roundtrip, dict) or set(roundtrip) != {"score", "evidence"}:
                raise ValueError("shortlisted transform lacks roundtrip evidence")
            _validate_search_score(
                SearchScore(roundtrip["score"], roundtrip["evidence"]),
                "heldout_full_expert_roundtrip",
                candidate,
                search,
            )
            finalists.append(record)
        elif record["roundtrip"] is not None:
            raise ValueError("non-shortlisted transform claims roundtrip evidence")
        if candidate.baseline:
            baseline_record = record
    if seen_draws != set(range(search.draw_count)) or len(finalists) != shortlist_count:
        raise ValueError("transform search draw or shortlist coverage mismatch")
    ranked = sorted(evaluations, key=lambda item: (item["proxy"]["score"], item["candidate_id"]))
    expected_shortlist_ids = {baseline_record["candidate_id"]} if baseline_record is not None else set()
    for record in ranked:
        expected_shortlist_ids.add(record["candidate_id"])
        if len(expected_shortlist_ids) == shortlist_count:
            break
    actual_shortlist_ids = {record["candidate_id"] for record in evaluations if record["shortlisted"]}
    if actual_shortlist_ids != expected_shortlist_ids:
        raise ValueError("transform search shortlist does not follow canonical proxy-score tie ordering")
    expected_storage = search.storage_estimate() | {
        "proxy_receipt_count": search.draw_count,
        "roundtrip_receipt_count": shortlist_count,
        "score_evidence_canonical_bytes": len(canonical_json(evaluations)),
    }
    if metadata["storage_estimate"] != expected_storage:
        raise ValueError("transform search storage/evidence estimate mismatch")
    if baseline_record is None or not baseline_record["shortlisted"]:
        raise ValueError("identity-G Rademacher baseline was not confirmed")
    expected_winner_record = min(
        finalists,
        key=lambda item: (item["roundtrip"]["score"], item["proxy"]["score"], item["candidate_id"]),
    )
    winner = metadata["winner"]
    winner_fields = {
        "draw_index", "candidate_id", "baseline", "block_scale_sha256", "improves_over_baseline",
        "proxy_score", "roundtrip_score",
    }
    if not isinstance(winner, dict) or set(winner) != winner_fields:
        raise ValueError("malformed transform search winner")
    expected_winner = search.candidate(expected_winner_record["draw_index"])
    expected_improvement = expected_winner_record["roundtrip"]["score"] < baseline_record["roundtrip"]["score"]
    if (
        winner["draw_index"] != expected_winner.draw_index
        or winner["candidate_id"] != expected_winner.candidate_id
        or winner["baseline"] is not expected_winner.baseline
        or winner["block_scale_sha256"] != expected_winner.block_scale_sha256
        or winner["improves_over_baseline"] is not expected_improvement
        or winner["proxy_score"] != expected_winner_record["proxy"]["score"]
        or winner["roundtrip_score"] != expected_winner_record["roundtrip"]["score"]
    ):
        raise ValueError("transform search winner does not follow sealed selection policy")
    expected_arrays = {
        "gate_up_input_shared": (search.hidden_size,),
        "down_output_shared": (search.hidden_size,),
        "gate_output": (len(search.expert_ids), search.intermediate_size),
        "up_output": (len(search.expert_ids), search.intermediate_size),
        "down_input": (len(search.expert_ids), search.intermediate_size),
    }
    if set(value.arrays) != set(expected_arrays):
        raise ValueError("transform search winner arrays are incomplete or unknown")
    for name, shape in expected_arrays.items():
        array = np.asarray(value.arrays[name])
        if array.dtype != np.float16 or array.shape != shape or not np.isfinite(array).all() or np.any(array == 0):
            raise ValueError(f"invalid transform winner vector {name!r}")
        if not np.array_equal(array, expected_winner.vectors[name]):
            raise ValueError(f"transform winner vector identity mismatch for {name!r}")


def searched_transform_content_sha256(value: SearchedTransformVectors) -> str:
    """Stable content identity independent of an artifact's filesystem path."""
    verify_searched_transform_vectors(value)
    array_records = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": sha256_bytes(np.asarray(array).tobytes(order="C")),
        }
        for name, array in sorted(value.arrays.items())
    }
    return sha256_bytes(canonical_json({"metadata": value.metadata, "arrays": array_records}))


def save_searched_transform_vectors(path: str | Path, value: SearchedTransformVectors) -> None:
    verify_searched_transform_vectors(value)
    destination = prepare_empty_destination(path)
    arrays_manifest: dict[str, Any] = {}
    for index, name in enumerate(sorted(value.arrays)):
        filename = f"vector-{index:02d}.npy"
        target = destination / filename
        _atomic_save_npy(target, value.arrays[name])
        arrays_manifest[name] = {
            "file": filename,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "dtype": "float16",
            "shape": list(value.arrays[name].shape),
        }
    manifest = dict(value.metadata)
    manifest["arrays"] = arrays_manifest
    manifest["seal_sha256"] = sha256_bytes(canonical_json(manifest))
    write_json(destination / "manifest.json", manifest)


def load_searched_transform_vectors(path: str | Path) -> SearchedTransformVectors:
    import json

    source = Path(path)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"transform-search manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = manifest.pop("seal_sha256", None)
    if not isinstance(seal, str) or not _HASH_RE.fullmatch(seal) or sha256_bytes(canonical_json(manifest)) != seal:
        raise ValueError("transform-search manifest seal mismatch")
    inventory = manifest.pop("arrays", None)
    if not isinstance(inventory, dict):
        raise ValueError("transform-search manifest has no array inventory")
    arrays: dict[str, np.ndarray] = {}
    filenames: set[str] = set()
    for name, record in inventory.items():
        if not isinstance(record, dict) or set(record) != {"file", "bytes", "sha256", "dtype", "shape"}:
            raise ValueError("malformed transform-search array record")
        filename = record["file"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in filenames
            or not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] < 1
            or not isinstance(record["sha256"], str)
            or not _HASH_RE.fullmatch(record["sha256"])
            or record["dtype"] != "float16"
            or not isinstance(record["shape"], list)
        ):
            raise ValueError("invalid transform-search array identity")
        filenames.add(filename)
        array_path = source / filename
        if not array_path.is_file() or array_path.stat().st_size != record["bytes"] or sha256_file(array_path) != record["sha256"]:
            raise ValueError(f"transform-search array identity mismatch for {name!r}")
        array = np.load(array_path, allow_pickle=False, mmap_mode="r")
        if str(array.dtype) != record["dtype"] or list(array.shape) != record["shape"]:
            raise ValueError(f"transform-search array metadata mismatch for {name!r}")
        arrays[name] = array
    if {item.name for item in source.iterdir()} != {"manifest.json", *filenames}:
        raise ValueError("transform-search directory contains missing or unbound files")
    result = SearchedTransformVectors(metadata=manifest, arrays=arrays)
    verify_searched_transform_vectors(result)
    return result


__all__ = [
    "ACCOUNTING_KINDS",
    "PRODUCTION_DEFAULTS",
    "ROUTE_WEIGHT_POWERS",
    "SCHEMA",
    "SEARCH_SCORE_RECEIPT_SCHEMA",
    "VECTOR_SEARCH_SCHEMA",
    "CalibrationBatch",
    "CalibrationFitter",
    "FittedExpertStatistics",
    "SearchScore",
    "SearchedTransformVectors",
    "TransformVectorCandidate",
    "TransformVectorSearch",
    "load_fitted_statistics",
    "load_searched_transform_vectors",
    "save_fitted_statistics",
    "save_searched_transform_vectors",
    "verify_searched_transform_vectors",
    "verify_fitted_statistics",
    "searched_transform_content_sha256",
]
