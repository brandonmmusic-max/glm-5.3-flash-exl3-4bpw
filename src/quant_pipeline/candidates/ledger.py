"""Exact-codec, full-expert candidate ledgers.

This module deliberately does not contain a fake quantizer.  A competitive
ledger is admitted only when a corrected EXL3/MCG backend and its executable
closure are hash-attested.  Tests may opt into a deterministic test backend,
but that permission is explicit in both the generator and the attestation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

from ..allocation.global_dp import AccountedAllocation, Candidate, allocate_with_fixed_layer_cost, pareto_frontier
from ..codecs.protocols import CodecAdapter, CodecCandidate
from ..core.artifacts import canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json
from .payload_store import ExactPayloadStore, SCHEMA_PAYLOAD_MANIFEST, SCHEMA_PAYLOAD_REF


SCHEMA_ATTESTATION = "quant-pipeline.codec-backend-attestation.v2"
SCHEMA_CANDIDATE = "quant-pipeline.exact-codec-candidate.v3"
SCHEMA_JOURNAL = "quant-pipeline.candidate-journal.v2"
SCHEMA_JOURNAL_RECORD = "quant-pipeline.candidate-journal-record.v2"
SCHEMA_LEDGER = "quant-pipeline.candidate-ledger.v3"
SCHEMA_EXPECTED_INVENTORY = "quant-pipeline.expected-expert-inventory.v1"
SCHEMA_COST_BREAKDOWN = "quant-pipeline.codec-cost-breakdown.v1"
SCHEMA_K5_ADMISSION = "quant-pipeline.k5-admission-evidence.v1"
K5_RULE_VERSION = "k5-confirmation-tail-rescue-v1"
CORRECTED_BACKEND = "corrected-exl3-mcg-r10"
CORRECTED_CODEC_NAME = "exl3-mcg-corrected"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
OBJECTIVE_METRICS = {
    "absolute_gate_squared_sse": "absolute_gate_squared_output_sse",
    "relative_sse": "relative_output_sse",
    "energy_normalized_sse": "energy_normalized_output_sse",
    "fisher_projection": "fisher_projection_damage",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BackendAttestation:
    schema: str
    backend: str
    codec_name: str
    codec_identity: dict[str, Any]
    codec_identity_sha256: str
    test_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend": self.backend,
            "codec_name": self.codec_name,
            "codec_identity": _json_value(self.codec_identity),
            "codec_identity_sha256": self.codec_identity_sha256,
            "test_only": self.test_only,
        }

    @property
    def sha256(self) -> str:
        return _hash_json(self.as_dict())


@dataclass(frozen=True)
class ProjectionTensors:
    gate_proj: Any
    up_proj: Any
    down_proj: Any

    def as_mapping(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in PROJECTIONS}


@dataclass(frozen=True)
class FittedProjection:
    covariance: Any
    input_vector: Any
    output_vector: Any
    fit_identity: Mapping[str, Any]
    transform_identity: Mapping[str, Any]
    bit_vectors: Mapping[int, tuple[Any, Any]] | None = None


@dataclass(frozen=True)
class RoutedExpertBatch:
    """Held-out observations routed to one expert.

    ``route_weights`` are the router gates associated with each row.  Full
    source/candidate top-k arrays are optional only for the fixed-router
    counterfactual; if either candidate route array is supplied, all four route
    arrays are required and agreement is measured rather than assumed.
    """

    batch_id: str
    hidden_states: Any
    route_weights: Any
    source_route_indices: Any | None = None
    source_route_weights: Any | None = None
    candidate_route_indices: Any | None = None
    candidate_route_weights: Any | None = None
    fisher_gradients: Any | None = None
    identity: Mapping[str, Any] | None = None
    row_keys: Sequence[str] | None = None


@dataclass(frozen=True)
class ReconciledLedgerAllocation:
    """A validated allocation whose selected-record cost closes exactly."""

    allocation: AccountedAllocation
    selected_records: tuple[Mapping[str, Any], ...]
    selected_cost: Mapping[str, Any]


@dataclass(frozen=True)
class ConditionalDownFitBatch:
    """Calibration-only rows for decoded gate/up -> down Hessian rebuilds."""

    batch_id: str
    hidden_states: Any
    route_weights: Any
    sampling_weights: Any
    source_route_indices: Any
    source_route_weights: Any
    identity: Mapping[str, Any]
    row_keys: Sequence[str]


@dataclass(frozen=True)
class K5Decision:
    admitted: bool
    reason: str
    screening: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "screening": _json_value(dict(self.screening or {})),
        }


@dataclass(frozen=True)
class MCGTransformArtifact:
    """Legacy raw-search carrier retained only for explicit rejection tests.

    Production conversion is forbidden. A search proposal must first become a
    canonical ``normalization.artifact_v31.AbsoluteV31Artifact`` through the
    source-derived layer fit and pinned per-bit GSS producer.
    """

    layer: int
    expert: int
    gate_up_suh: Any
    gate_svh: Any
    up_svh: Any
    down_suh: Any
    down_svh: Any
    codebook_scale: float
    selection_method: str
    objective_arm: str
    candidates_evaluated: int
    selected_score: float
    selection_role: str
    heldout_artifact_sha256: str
    evidence_sha256: str
    provenance: Mapping[str, Any]
    absolute_v31_baseline: Mapping[str, Any] | None = None
    bit_private_vectors: Mapping[str, Mapping[int, Any]] | None = None
    schema: str = "quant-pipeline.mcg-searched-transform.v1"


class TransformSearchAdapter(Protocol):
    """Integration boundary returning a canonical AbsoluteV31Artifact."""

    def search_transform(
        self,
        *,
        layer: int,
        expert: int,
        source: ProjectionTensors,
        gate_up_statistics: Any,
        down_statistics: Any,
        heldout_batches: Callable[[], Iterable[RoutedExpertBatch]],
        codebook_scale: float,
        objective_arm: str,
    ) -> Any: ...


@dataclass(frozen=True)
class ExpertCandidateInput:
    layer: int
    expert: int
    source: ProjectionTensors
    fitted: Mapping[str, FittedProjection]
    heldout_batches: Sequence[RoutedExpertBatch] | Callable[[], Iterable[RoutedExpertBatch]]
    k5_screen: Mapping[tuple[int, int, int], K5Decision]
    conditional_down_fit_batches: Sequence[ConditionalDownFitBatch] | Callable[[], Iterable[ConditionalDownFitBatch]] | None = None
    proposal_score_candidate_identity: Mapping[str, Mapping[str, Any]] | None = None

    @property
    def unit_id(self) -> str:
        return f"L{self.layer}.E{self.expert}"


def expected_expert_inventory(
    units: Iterable[tuple[int, int]],
    *,
    profile: str,
    test_fixture: bool = False,
) -> dict[str, Any]:
    """Seal the exact expert/projection inventory for one ledger run.

    Production Qwen uses the named 48x128x3 profile.  Any smaller inventory is
    admitted only when explicitly labelled as a test fixture; it cannot be
    mistaken for production coverage.
    """
    raw_identities = list(units)
    identities = sorted(set(raw_identities))
    if not identities or len(identities) != len(raw_identities):
        raise ValueError("expected inventory must contain unique expert units")
    if any(
        isinstance(layer, bool)
        or isinstance(expert, bool)
        or not isinstance(layer, int)
        or not isinstance(expert, int)
        or layer < 0
        or expert < 0
        for layer, expert in identities
    ):
        raise ValueError("expected inventory identities must be non-negative integers")
    if not isinstance(profile, str) or not profile:
        raise ValueError("expected inventory profile must be non-empty")
    production = profile == "qwen3-30b-a3b-48x128x3"
    expected_qwen = [(layer, expert) for layer in range(48) for expert in range(128)]
    if production and identities != expected_qwen:
        raise ValueError("production Qwen inventory must be exactly 48 layers x 128 experts x 3 projections")
    if not production and not test_fixture:
        raise ValueError("non-production inventory must be explicitly labelled test_fixture")
    body = {
        "schema": SCHEMA_EXPECTED_INVENTORY,
        "profile": profile,
        "test_fixture": bool(test_fixture),
        "projection_names": list(PROJECTIONS),
        "units": [
            {"unit_id": f"L{layer}.E{expert}", "layer": layer, "expert": expert, "projection_count": 3}
            for layer, expert in identities
        ],
    }
    body["inventory_sha256"] = _hash_json(body)
    return body


def all_k3_k4_triplets() -> tuple[tuple[int, int, int], ...]:
    """The required complete K3/K4 joint gate/up/down design (2^3)."""
    return tuple(product((3, 4), repeat=3))


def all_k5_triplets() -> tuple[tuple[int, int, int], ...]:
    """All additional K3/K4/K5 tuples containing at least one K5 (27 - 8)."""
    baseline = set(all_k3_k4_triplets())
    return tuple(value for value in product((3, 4, 5), repeat=3) if value not in baseline)


def reject_all_k5(reason: str) -> dict[tuple[int, int, int], K5Decision]:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("K5 rejection reason must be non-empty")
    return {triplet: K5Decision(False, reason) for triplet in all_k5_triplets()}


def admit_k5(
    reason: str,
    *,
    selection_artifact_sha256: str,
    confirmation_artifact_sha256: str,
    p99_relative_output_sse_delta: float,
    mean_relative_output_sse_delta: float,
    max_p99_relative_output_sse_delta: float,
    max_mean_relative_output_sse_delta: float,
) -> K5Decision:
    """Create a versioned K5 admission with independently sealed evidence."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("K5 admission reason must be non-empty")
    screening = {
        "schema": SCHEMA_K5_ADMISSION,
        "rule_version": K5_RULE_VERSION,
        "selection_role": "selection",
        "selection_artifact_sha256": _require_hash(selection_artifact_sha256, "K5 selection artifact"),
        "confirmation_role": "disjoint-confirmation",
        "confirmation_artifact_sha256": _require_hash(confirmation_artifact_sha256, "K5 confirmation artifact"),
        "metrics": {
            "p99_relative_output_sse_delta": float(p99_relative_output_sse_delta),
            "mean_relative_output_sse_delta": float(mean_relative_output_sse_delta),
        },
        "thresholds": {
            "max_p99_relative_output_sse_delta": float(max_p99_relative_output_sse_delta),
            "max_mean_relative_output_sse_delta": float(max_mean_relative_output_sse_delta),
        },
    }
    screening["evidence_sha256"] = _hash_json(screening)
    decision = K5Decision(True, reason, screening)
    _validate_k5_decision(decision)
    return decision


def _validate_k5_decision(decision: K5Decision, *, expected_selection_artifact_sha256: str | None = None) -> None:
    if not isinstance(decision, K5Decision) or not isinstance(decision.admitted, bool):
        raise ValueError("K5 decision is malformed")
    if not isinstance(decision.reason, str) or not decision.reason.strip():
        raise ValueError("K5 decision reason must be non-empty")
    if not decision.admitted:
        if decision.screening not in (None, {}):
            raise ValueError("rejected K5 decision must not carry admission evidence")
        return
    evidence = dict(decision.screening or {})
    required = {
        "schema",
        "rule_version",
        "selection_role",
        "selection_artifact_sha256",
        "confirmation_role",
        "confirmation_artifact_sha256",
        "metrics",
        "thresholds",
        "evidence_sha256",
    }
    if set(evidence) != required or evidence.get("schema") != SCHEMA_K5_ADMISSION or evidence.get("rule_version") != K5_RULE_VERSION:
        raise ValueError("K5 admission lacks the exact versioned rule evidence")
    if evidence.get("selection_role") != "selection" or evidence.get("confirmation_role") != "disjoint-confirmation":
        raise ValueError("K5 admission roles are not selection plus disjoint confirmation")
    selection = _require_hash(evidence.get("selection_artifact_sha256"), "K5 selection artifact")
    confirmation = _require_hash(evidence.get("confirmation_artifact_sha256"), "K5 confirmation artifact")
    if selection == confirmation:
        raise ValueError("K5 confirmation artifact must be disjoint from selection")
    if expected_selection_artifact_sha256 is not None and selection != expected_selection_artifact_sha256:
        raise ValueError("K5 selection evidence differs from the sealed held-out artifact")
    metrics = evidence.get("metrics")
    thresholds = evidence.get("thresholds")
    if not isinstance(metrics, dict) or set(metrics) != {
        "p99_relative_output_sse_delta",
        "mean_relative_output_sse_delta",
    }:
        raise ValueError("K5 admission metrics are incomplete")
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "max_p99_relative_output_sse_delta",
        "max_mean_relative_output_sse_delta",
    }:
        raise ValueError("K5 admission thresholds are incomplete")
    pairs = (
        (metrics["p99_relative_output_sse_delta"], thresholds["max_p99_relative_output_sse_delta"]),
        (metrics["mean_relative_output_sse_delta"], thresholds["max_mean_relative_output_sse_delta"]),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for pair in pairs
        for value in pair
    ):
        raise ValueError("K5 admission metrics and thresholds must be finite numbers")
    if any(float(metric) > float(limit) for metric, limit in pairs):
        raise ValueError("K5 admission does not satisfy its declared thresholds")
    sealed = _require_hash(evidence.get("evidence_sha256"), "K5 admission evidence")
    if _hash_json({key: value for key, value in evidence.items() if key != "evidence_sha256"}) != sealed:
        raise ValueError("K5 admission evidence hash mismatch")


def _iter_batches(item: ExpertCandidateInput) -> Iterator[RoutedExpertBatch]:
    source = item.heldout_batches
    values = source() if callable(source) else source
    return iter(values)


def _iter_down_fit_batches(item: ExpertCandidateInput) -> Iterator[ConditionalDownFitBatch]:
    source = item.conditional_down_fit_batches
    if source is None:
        return iter(())
    values = source() if callable(source) else source
    return iter(values)


def _rademacher(length: int, *identity: Any) -> Any:
    """Deterministic domain-separated H128 sign vector.

    The definition is intentionally local and hash-stable rather than relying
    on a process RNG.  It is a reproducible baseline transform, not a claim
    that a multi-draw transform search has already been performed.
    """
    if length < 1 or length % 128:
        raise ValueError("MCG transform vector length must be positive and divisible by 128")
    import numpy as np

    prefix = canonical_json(["quant-pipeline.mcg-rademacher-h128.v1", *_json_value(identity)])
    result = np.empty(length, dtype=np.float32)
    for index in range(length):
        digest = sha256_bytes(prefix + index.to_bytes(8, "big"))
        result[index] = 1.0 if int(digest[:2], 16) & 1 else -1.0
    return result


def _statistics_identity(value: Any) -> dict[str, Any]:
    metadata = _json_value(value.metadata)
    arrays = {name: _tensor_record(array) for name, array in sorted(value.arrays.items())}
    body = {"metadata": metadata, "arrays": arrays}
    return body | {"statistics_sha256": _hash_json(body)}


def _searched_transform_vectors(
    artifact: Any,
    *,
    layer: int,
    expert: int,
    hidden: int,
    intermediate: int,
    codebook_scale: float,
) -> tuple[dict[str, tuple[Any, Any, str]], dict[str, Any], dict[str, dict[int, tuple[Any, Any]]] | None]:
    from ..normalization.artifact_v31 import AbsoluteV31Artifact, verify_absolute_v31_artifact

    if isinstance(artifact, MCGTransformArtifact):
        raise RuntimeError(
            "raw MCGTransformArtifact conversion is forbidden; pass the canonical verified AbsoluteV31Artifact"
        )
    if not isinstance(artifact, AbsoluteV31Artifact):
        raise TypeError("searched transform must be the canonical AbsoluteV31Artifact")
    verify_absolute_v31_artifact(artifact)
    identity = artifact.metadata["identity"]
    if identity["layer_id"] != layer:
        raise ValueError("absolute-v31 artifact layer identity mismatch")
    records = {row["key"]: row for row in artifact.metadata["matrices"]}
    selected = {}
    bit_vectors: dict[str, dict[int, tuple[Any, Any]]] = {}
    bit_records: dict[str, Any] = {}
    for projection in PROJECTIONS:
        key = f"E{expert}.{projection}"
        if key not in records:
            raise ValueError(f"absolute-v31 artifact does not contain {key}")
        record = records[key]
        shared = artifact.arrays[record["shared_vector"]["array"]]
        expected_shared = hidden
        if _shape(shared) != (expected_shared,):
            raise ValueError(f"{key} canonical shared vector shape mismatch")
        bit_vectors[projection] = {}
        bit_records[projection] = {}
        for bit in (3, 4, 5):
            candidate = record["candidates"][str(bit)]
            private = artifact.arrays[candidate["private_vector"]["array"]]
            if _shape(private) != (intermediate,):
                raise ValueError(f"{key} K{bit} canonical private vector shape mismatch")
            pair = (shared, private) if projection in {"gate_proj", "up_proj"} else (private, shared)
            bit_vectors[projection][bit] = pair
            bit_records[projection][str(bit)] = {
                "suh": _tensor_record(pair[0]),
                "svh": _tensor_record(pair[1]),
                "gss_receipt_sha256": candidate["gss_receipt"]["receipt_sha256"],
                "selection_decision_sha256": record["selection"]["decision_sha256"],
            }
        selected[projection] = bit_vectors[projection][record["selection"]["bits"]]
    vectors = {
        "gate_proj": (*selected["gate_proj"], "gate-up-layer-shared-suh"),
        "up_proj": (*selected["up_proj"], "gate-up-layer-shared-suh"),
        "down_proj": (*selected["down_proj"], "down-layer-shared-svh"),
    }
    selection = {
        "selection_status": "canonical-absolute-v31-gss",
        "canonical_artifact_content_sha256": artifact.content_sha256,
        "canonical_policy": _json_value(artifact.metadata["policy"]),
        "canonical_identity": _json_value(identity),
        "expert": expert,
        "bit_gss_vectors": bit_records,
    }
    selection["artifact_sha256"] = artifact.content_sha256
    return vectors, selection, bit_vectors


def build_expert_candidate_input(
    *,
    layer: int,
    expert: int,
    source: ProjectionTensors,
    gate_up_statistics: Any,
    down_statistics: Any,
    heldout_batches: Sequence[RoutedExpertBatch] | Callable[[], Iterable[RoutedExpertBatch]],
    k5_screen: Mapping[tuple[int, int, int], K5Decision],
    route_power: int,
    accounting: str,
    transform_seed_sha256: str,
    codebook_scale: float,
    searched_transform: Any | None = None,
    allow_fixed_transform_baseline: bool = False,
    conditional_down_fit_batches: Sequence[ConditionalDownFitBatch] | Callable[[], Iterable[ConditionalDownFitBatch]] | None = None,
    proposal_score_candidate_identity: Mapping[str, Mapping[str, Any]] | None = None,
) -> ExpertCandidateInput:
    """Create exact codec inputs directly from verified fitter outputs.

    Gate/up share the same fitted input covariance and layer-shared ``suh``;
    down uses its separately fitted SwiGLU-input covariance and a layer-shared
    ``svh``.  The remaining sides are expert-specific.  The deterministic
    Rademacher H128 vectors use identity block G-scales and apply the corrected
    R10 codebook normalization to ``suh`` exactly once.
    """
    from ..calibration.fitter import FittedExpertStatistics, verify_fitted_statistics

    if not isinstance(gate_up_statistics, FittedExpertStatistics) or not isinstance(down_statistics, FittedExpertStatistics):
        raise TypeError("fitter bridge requires FittedExpertStatistics for gate/up and down inputs")
    verify_fitted_statistics(gate_up_statistics)
    verify_fitted_statistics(down_statistics)
    if route_power not in (0, 1, 2):
        raise ValueError("route_power must be one of the fitted powers 0, 1, or 2")
    if not isinstance(accounting, str) or not accounting:
        raise ValueError("accounting policy must be non-empty")
    _require_hash(transform_seed_sha256, "transform seed")
    if not isinstance(codebook_scale, (int, float)) or isinstance(codebook_scale, bool) or not math.isfinite(float(codebook_scale)) or float(codebook_scale) == 0.0:
        raise ValueError("MCG codebook scale must be finite and nonzero")
    source_map = source.as_mapping()
    gate_shape = _shape(source_map["gate_proj"])
    down_shape = _shape(source_map["down_proj"])
    if len(gate_shape) != 2 or _shape(source_map["up_proj"]) != gate_shape or down_shape != (gate_shape[1], gate_shape[0]):
        raise ValueError("source projection geometry is not a gate/up/down expert")
    hidden, intermediate = gate_shape[1], gate_shape[0]
    if hidden % 128 or intermediate % 128:
        raise ValueError("corrected MCG bridge requires hidden and intermediate dimensions divisible by 128")
    identities = []
    for role, statistics, dimension in (
        ("gate_up_input", gate_up_statistics, hidden),
        ("down_input", down_statistics, intermediate),
    ):
        identity = statistics.metadata.get("identity", {})
        estimator = statistics.metadata.get("estimator", {})
        if identity.get("layer_id") != layer or identity.get("expert_id") != str(expert):
            raise ValueError(f"{role} fitted layer/expert identity mismatch")
        if identity.get("hidden_size") != dimension:
            raise ValueError(f"{role} fitted dimension mismatch")
        if estimator.get("covariance_mode") != "full":
            raise ValueError(f"competitive exact-codec {role} requires full covariance geometry")
        if accounting not in estimator.get("retained_accounting", ()) or route_power not in estimator.get("retained_powers", ()):
            raise ValueError(f"{role} does not retain requested {accounting} p{route_power} covariance")
        identities.append(identity)
    if identities[0].get("predecessor_checkpoint_hash") != identities[1].get("predecessor_checkpoint_hash"):
        raise ValueError("gate/up and down fitter predecessor identities differ")
    if identities[0].get("source_identities") != identities[1].get("source_identities"):
        raise ValueError("gate/up and down fitter source identities differ")

    import numpy as np

    # Exact R10 solves consume the uncentered weighted Gram / second moment,
    # not a mean-subtracted covariance.  Failing closed here prevents a
    # statistically valid covariance artifact from being silently used as the
    # wrong numeric object by the codec.
    if not hasattr(gate_up_statistics, "dense_hessian") or not hasattr(down_statistics, "dense_hessian"):
        raise TypeError("fitter artifact lacks dense_hessian required by the exact codec")
    gate_up_covariance = np.asarray(
        gate_up_statistics.dense_hessian(accounting, route_power, regularized=False),
        dtype=np.float32,
    )
    down_covariance = np.asarray(
        down_statistics.dense_hessian(accounting, route_power, regularized=False),
        dtype=np.float32,
    )
    factor = -float(codebook_scale)
    statistics_identities = {
        "gate_up_input": _statistics_identity(gate_up_statistics),
        "down_input": _statistics_identity(down_statistics),
    }
    if searched_transform is None:
        searched_bit_vectors = None
        if not allow_fixed_transform_baseline:
            raise ValueError(
                "competitive preparation requires a searched MCG transform; "
                "set allow_fixed_transform_baseline only for an explicitly labelled baseline"
            )
        shared_gate_up_suh = _rademacher(
            hidden, transform_seed_sha256, layer, "layer-shared", "gate-up-suh"
        ) / factor
        shared_down_svh = _rademacher(
            hidden, transform_seed_sha256, layer, "layer-shared", "down-svh"
        )
        gate_svh = _rademacher(
            intermediate, transform_seed_sha256, layer, expert, "gate-svh"
        )
        up_svh = _rademacher(
            intermediate, transform_seed_sha256, layer, expert, "up-svh"
        )
        down_suh = _rademacher(
            intermediate, transform_seed_sha256, layer, expert, "down-suh"
        ) / factor
        vectors = {
            "gate_proj": (shared_gate_up_suh, gate_svh, "gate-up-layer-shared-suh"),
            "up_proj": (shared_gate_up_suh.copy(), up_svh, "gate-up-layer-shared-suh"),
            "down_proj": (down_suh, shared_down_svh, "down-layer-shared-svh"),
        }
        common_transform = {
            "schema": "quant-pipeline.mcg-transform.v1",
            "policy": "deterministic-domain-separated-rademacher-h128-identity-gscale",
            "selection_status": "fixed-reproducible-baseline-not-multidraw-searched",
            "transform_seed_sha256": transform_seed_sha256,
            "search_artifact_sha256": transform_seed_sha256,
            "codebook_normalization": {"side": "suh", "divisor": factor},
            "block_size": 128,
            "block_g_scales": "identity",
            "layer": layer,
            "expert": expert,
        }
    else:
        vectors, search_identity, searched_bit_vectors = _searched_transform_vectors(
            searched_transform,
            layer=layer,
            expert=expert,
            hidden=hidden,
            intermediate=intermediate,
            codebook_scale=float(codebook_scale),
        )
        common_transform = {
            "schema": "quant-pipeline.mcg-transform.v1",
            "policy": "canonical-source-derived-absolute-v31-with-pinned-per-bit-gss",
            "selection_status": "canonical-absolute-v31-gss",
            "transform_seed_sha256": transform_seed_sha256,
            "search_artifact": search_identity,
            "search_artifact_sha256": search_identity["artifact_sha256"],
            "block_size": 128,
            "layer": layer,
            "expert": expert,
        }
    fit_common = {
        "accounting": accounting,
        "route_weight_power": route_power,
        "regularized": False,
        "hessian_regularization": "codec-level-sigma-reg-only",
        "predecessor_checkpoint_hash": identities[0]["predecessor_checkpoint_hash"],
        "source_identities": identities[0]["source_identities"],
        "statistics": statistics_identities,
        "fit_artifact_sha256": _hash_json(statistics_identities),
        "model_revision": identities[0]["source_identities"]["model_revision"],
        "dataset_revision": identities[0]["source_identities"]["dataset_revision"],
        "predecessor_state_sha256": identities[0]["predecessor_checkpoint_hash"],
    }
    covariance = {
        "gate_proj": gate_up_covariance,
        "up_proj": gate_up_covariance.copy(),
        "down_proj": down_covariance,
    }
    result: dict[str, FittedProjection] = {}
    for projection in PROJECTIONS:
        suh, svh, shared_side = vectors[projection]
        transform = common_transform | {
            "projection": projection,
            "shared_side": shared_side,
            "shared_gate_up_suh_sha256": _tensor_record(vectors["gate_proj"][0])["sha256"],
            "shared_down_svh_sha256": _tensor_record(vectors["down_proj"][1])["sha256"],
            "suh": _tensor_record(suh),
            "svh": _tensor_record(svh),
        }
        transform["transform_sha256"] = _hash_json(transform)
        result[projection] = FittedProjection(
            covariance=covariance[projection],
            input_vector=suh,
            output_vector=svh,
            fit_identity=fit_common | {
                "projection": projection,
                "codec_matrix_kind": "uncentered-weighted-second-moment-hessian",
                "hessian_source": "gate_up_input" if projection != "down_proj" else "down_input",
            },
            transform_identity=transform,
            bit_vectors=None if searched_bit_vectors is None else searched_bit_vectors[projection],
        )
    return ExpertCandidateInput(
        layer=layer,
        expert=expert,
        source=source,
        fitted=result,
        heldout_batches=heldout_batches,
        k5_screen=k5_screen,
        conditional_down_fit_batches=conditional_down_fit_batches,
        proposal_score_candidate_identity=proposal_score_candidate_identity,
    )


def _json_value(value: Any) -> Any:
    """Convert provenance to strict canonical-JSON values or fail closed."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON value")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    try:
        scalar = value.item()
    except (AttributeError, ValueError, TypeError):
        raise TypeError(f"value is not strict JSON provenance: {type(value).__name__}") from None
    return _json_value(scalar)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(_json_value(value)))


def _tensor_bytes(value: Any) -> bytes:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            return tensor.view(torch.uint8).numpy().tobytes()
    except ImportError:  # pragma: no cover - torch is an HF optional dependency
        pass
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be hash-bound")
    return array.view(np.uint8).tobytes()


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(map(int, value.shape))


def _dtype(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "shape": list(_shape(value)),
        "dtype": _dtype(value),
        "bytes": len(_tensor_bytes(value)),
        "sha256": sha256_bytes(_tensor_bytes(value)),
    }


def routed_batch_sha256(batch: RoutedExpertBatch) -> str:
    """Hash every byte-affecting/scoring tensor in a routed batch."""
    body = {
        "batch_id": batch.batch_id,
        "row_keys": list(batch.row_keys or ()),
        "identity": {
            key: _json_value(value)
            for key, value in sorted(dict(batch.identity or {}).items())
            if key != "batch_payload_sha256"
        },
    }
    for name in (
        "hidden_states",
        "route_weights",
        "source_route_indices",
        "source_route_weights",
        "candidate_route_indices",
        "candidate_route_weights",
        "fisher_gradients",
    ):
        value = getattr(batch, name)
        body[name] = None if value is None else _tensor_record(value)
    return _hash_json(body)


def conditional_down_fit_batch_sha256(batch: ConditionalDownFitBatch) -> str:
    return _hash_json(
        {
            "batch_id": batch.batch_id,
            "hidden_states": _tensor_record(batch.hidden_states),
            "route_weights": _tensor_record(batch.route_weights),
            "sampling_weights": _tensor_record(batch.sampling_weights),
            "source_route_indices": _tensor_record(batch.source_route_indices),
            "source_route_weights": _tensor_record(batch.source_route_weights),
            "identity": {
                key: _json_value(value)
                for key, value in sorted(dict(batch.identity).items())
                if key != "batch_payload_sha256"
            },
            "row_keys": list(batch.row_keys),
        }
    )


def _require_finite_tensor(value: Any, label: str) -> None:
    try:
        import torch

        tensor = torch.as_tensor(value)
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{label} contains non-finite values")
    except ImportError:  # pragma: no cover
        import numpy as np

        if not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite values")


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be an immutable 40-hex revision")
    return value


def attest_corrected_exl3_mcg(codec: CodecAdapter) -> BackendAttestation:
    """Build a live attestation for the production adapter's sealed closure."""
    from ..codecs.exl3_mcg import Exl3MCGCodec

    if not isinstance(codec, Exl3MCGCodec) or codec.name != CORRECTED_CODEC_NAME:
        raise TypeError("only Exl3MCGCodec can receive a production attestation")
    identity = _json_value(codec.identity)
    if identity.get("identity_schema") != "quant-pipeline.exl3-mcg-numeric-identity.v1":
        raise ValueError("codec identity schema is missing or unsupported")
    if identity.get("backend_class") != "r7_encoder.r10_codec.R10TrellisCodec":
        raise ValueError("codec identity does not name the corrected backend implementation")
    sigma_reg = identity.get("sigma_reg")
    if isinstance(sigma_reg, bool) or not isinstance(sigma_reg, (int, float)) or not math.isfinite(sigma_reg) or sigma_reg <= 0:
        raise ValueError("codec identity lacks a valid sigma_reg")
    environment = identity.get("environment")
    if not isinstance(environment, dict) or set(environment) != {"python", "machine", "torch", "torch_cuda", "compute_capability"}:
        raise ValueError("codec identity lacks its stable numeric environment")
    for field in ("numeric_core_sha256", "extension_sha256"):
        _require_hash(identity.get(field), f"codec identity {field}")
    closure = identity.get("python_closure_sha256")
    if not isinstance(closure, dict) or not closure:
        raise ValueError("codec identity lacks its Python closure")
    for filename, digest in closure.items():
        _require_hash(digest, f"codec closure {filename}")
    # Re-read executable files so this attestation cannot be issued after drift.
    if sha256_file(codec.numeric_core) != identity["numeric_core_sha256"]:
        raise RuntimeError("numeric core drifted before attestation")
    if sha256_file(codec.extension) != identity["extension_sha256"]:
        raise RuntimeError("extension drifted before attestation")
    for filename, digest in closure.items():
        if sha256_file(codec.source_root / "r7_encoder" / filename) != digest:
            raise RuntimeError(f"codec Python closure drifted before attestation: {filename}")
    return BackendAttestation(
        schema=SCHEMA_ATTESTATION,
        backend=CORRECTED_BACKEND,
        codec_name=codec.name,
        codec_identity=identity,
        codec_identity_sha256=_hash_json(identity),
        test_only=False,
    )


def _validate_attestation(
    codec: CodecAdapter,
    attestation: BackendAttestation,
    *,
    competitive: bool,
    allow_test_backend: bool,
) -> dict[str, Any]:
    raw = attestation.as_dict()
    if raw["schema"] != SCHEMA_ATTESTATION:
        raise ValueError("unsupported codec attestation schema")
    if raw["codec_name"] != getattr(codec, "name", None):
        raise ValueError("codec name differs from backend attestation")
    if _hash_json(raw["codec_identity"]) != raw["codec_identity_sha256"]:
        raise ValueError("codec identity differs from attestation hash")
    if not isinstance(raw["test_only"], bool):
        raise ValueError("codec attestation test_only must be Boolean")
    if competitive:
        if raw["backend"] != CORRECTED_BACKEND or raw["codec_name"] != CORRECTED_CODEC_NAME:
            raise ValueError("competitive ledgers require the corrected EXL3/MCG backend")
        if raw["test_only"]:
            if not allow_test_backend:
                raise ValueError("test-only codec attestation is forbidden in a competitive ledger")
        else:
            from ..codecs.exl3_mcg import Exl3MCGCodec

            if not isinstance(codec, Exl3MCGCodec):
                raise TypeError("production competitive attestation requires Exl3MCGCodec")
            current = attest_corrected_exl3_mcg(codec).as_dict()
            if current != raw:
                raise RuntimeError("live corrected backend identity differs from attestation")
    return raw


class CandidateJournal:
    """Atomic, resumable, identity-bound per-candidate journal.

    Every screening decision and candidate is a separate atomically replaced
    JSON file.  Existing records are immutable: an identical replay is a no-op,
    while any drift fails rather than overwriting evidence.
    """

    def __init__(self, root: str | Path, run_identity: Mapping[str, Any], *, resume: bool = False) -> None:
        self.root = Path(root)
        self.records_dir = self.root / "records"
        identity = _json_value(dict(run_identity))
        self._validate_run_identity(identity)
        self.run_identity = identity
        self.run_identity_sha256 = _hash_json(identity)
        header_body = {
            "schema": SCHEMA_JOURNAL,
            "run_identity": identity,
            "run_identity_sha256": self.run_identity_sha256,
        }
        header = header_body | {"header_sha256": _hash_json(header_body)}
        if resume:
            existing = json.loads((self.root / "journal.json").read_text())
            if existing != header:
                raise ValueError("journal resume identity drift")
            if not self.records_dir.is_dir():
                raise ValueError("journal records directory is missing")
        else:
            prepare_empty_destination(self.root)
            self.records_dir.mkdir()
            write_json(self.root / "journal.json", header)

    @staticmethod
    def _validate_run_identity(identity: Mapping[str, Any]) -> None:
        required = (
            "model_revision",
            "dataset_revision",
            "fit_artifact_sha256",
            "heldout_artifact_sha256",
            "predecessor_state_sha256",
            "codec_attestation_sha256",
            "search_artifact_sha256",
            "capture_artifact_sha256",
            "conditional_down_fit_artifact_sha256",
            "fisher_probe_sha256",
            "fisher_window_sha256",
            "expected_inventory",
            "expected_inventory_sha256",
        )
        if set(required) - set(identity):
            raise ValueError(f"journal run identity is incomplete: {sorted(set(required) - set(identity))}")
        _require_revision(identity["model_revision"], "model_revision")
        _require_revision(identity["dataset_revision"], "dataset_revision")
        for key in required[2:11]:
            _require_hash(identity[key], key)
        inventory = identity["expected_inventory"]
        if not isinstance(inventory, dict) or inventory.get("schema") != SCHEMA_EXPECTED_INVENTORY:
            raise ValueError("journal expected inventory is missing or malformed")
        _require_hash(inventory.get("inventory_sha256"), "embedded expected inventory")
        if _hash_json({key: value for key, value in inventory.items() if key != "inventory_sha256"}) != inventory["inventory_sha256"]:
            raise ValueError("embedded expected inventory hash mismatch")
        if identity["expected_inventory_sha256"] != inventory["inventory_sha256"]:
            raise ValueError("journal expected inventory identity mismatch")

    @staticmethod
    def _record_name(kind: str, key: str) -> str:
        if re.fullmatch(r"[a-z][a-z0-9-]*", kind) is None:
            raise ValueError("journal record kind is invalid")
        if not isinstance(key, str) or not key:
            raise ValueError("journal record key must be non-empty")
        return f"{kind}-{sha256_bytes(key.encode())}.json"

    def load(self, kind: str, key: str, *, input_sha256: str | None = None) -> dict[str, Any] | None:
        path = self.records_dir / self._record_name(kind, key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        self._validate_wrapper(raw, kind, key)
        if input_sha256 is not None and raw["input_sha256"] != input_sha256:
            raise ValueError(f"journal input drift for {kind}:{key}")
        return raw["payload"]

    def record(self, kind: str, key: str, input_sha256: str, payload: Mapping[str, Any]) -> str:
        _require_hash(input_sha256, "journal input_sha256")
        body = {
            "schema": SCHEMA_JOURNAL_RECORD,
            "kind": kind,
            "key": key,
            "run_identity_sha256": self.run_identity_sha256,
            "input_sha256": input_sha256,
            "payload": _json_value(dict(payload)),
        }
        wrapper = body | {"record_sha256": _hash_json(body)}
        path = self.records_dir / self._record_name(kind, key)
        if path.exists():
            existing = json.loads(path.read_text())
            self._validate_wrapper(existing, kind, key)
            if existing != wrapper:
                raise ValueError(f"refusing to overwrite drifted journal record {kind}:{key}")
            return existing["record_sha256"]
        return write_json(path, wrapper)

    def _validate_wrapper(self, raw: Mapping[str, Any], kind: str, key: str) -> None:
        if raw.get("schema") != SCHEMA_JOURNAL_RECORD or raw.get("kind") != kind or raw.get("key") != key:
            raise ValueError(f"malformed journal record {kind}:{key}")
        if raw.get("run_identity_sha256") != self.run_identity_sha256:
            raise ValueError(f"journal record identity drift for {kind}:{key}")
        record_hash = raw.get("record_sha256")
        _require_hash(record_hash, "journal record hash")
        if _hash_json({name: value for name, value in raw.items() if name != "record_sha256"}) != record_hash:
            raise ValueError(f"journal record hash mismatch for {kind}:{key}")

    def inventory(self) -> dict[str, str]:
        records: dict[str, str] = {}
        for path in sorted(self.records_dir.glob("*.json")):
            raw = json.loads(path.read_text())
            self._validate_wrapper(raw, str(raw.get("kind")), str(raw.get("key")))
            records[path.name] = sha256_file(path)
        return records

    def record_index(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.records_dir.glob("*.json")):
            raw = json.loads(path.read_text())
            self._validate_wrapper(raw, str(raw.get("kind")), str(raw.get("key")))
            result.append(
                {
                    "filename": path.name,
                    "kind": raw["kind"],
                    "key": raw["key"],
                    "input_sha256": raw["input_sha256"],
                    "record_sha256": raw["record_sha256"],
                    "file_sha256": sha256_file(path),
                }
            )
        return result


def _validate_expert_input(item: ExpertCandidateInput, competitive: bool) -> None:
    if isinstance(item.layer, bool) or not isinstance(item.layer, int) or item.layer < 0:
        raise ValueError("layer must be a non-negative integer")
    if isinstance(item.expert, bool) or not isinstance(item.expert, int) or item.expert < 0:
        raise ValueError("expert must be a non-negative integer")
    if set(item.fitted) != set(PROJECTIONS):
        raise ValueError(f"fitted projections must be exactly {PROJECTIONS}")
    if not callable(item.heldout_batches) and not item.heldout_batches:
        raise ValueError("at least one routed held-out batch is required")
    if item.conditional_down_fit_batches is None:
        raise ValueError("decoded gate/up conditional down fitting requires a disjoint calibration stream")
    if set(item.k5_screen) != set(all_k5_triplets()):
        raise ValueError("K5 screening must explicitly decide all 19 K5-containing triplets")
    if item.proposal_score_candidate_identity is not None:
        expected = {"gate_proj", "up_proj", "down_proj.g4u4"}
        if set(item.proposal_score_candidate_identity) != expected:
            raise ValueError("proposal-score candidate identity must cover exact conditional uniform-K4 bytes")
        for projection, row in item.proposal_score_candidate_identity.items():
            if not isinstance(row, Mapping) or set(row) != {
                "bits", "packed_sha256", "reconstruction_deployed_fp16_sha256",
                "reconstruction_hf_sha256",
            }:
                raise ValueError(f"proposal-score candidate identity is malformed for {projection}")
            if row["bits"] != 4:
                raise ValueError("proposal-score candidate identity must bind K4")
            for key in ("packed_sha256", "reconstruction_deployed_fp16_sha256", "reconstruction_hf_sha256"):
                _require_hash(row[key], f"proposal-score {projection} {key}")
    source = item.source.as_mapping()
    gate_shape, up_shape, down_shape = (_shape(source[name]) for name in PROJECTIONS)
    if len(gate_shape) != 2 or up_shape != gate_shape:
        raise ValueError("gate/up source tensors must share [intermediate, hidden] shape")
    if down_shape != (gate_shape[1], gate_shape[0]):
        raise ValueError("down source tensor must be [hidden, intermediate]")
    for name in PROJECTIONS:
        value = source[name]
        _require_finite_tensor(value, f"source {name}")
        if competitive and _dtype(value) != "bfloat16":
            raise ValueError(f"competitive source {name} must be exact BF16")
        fit = item.fitted[name]
        n, k = _shape(value)
        if _shape(fit.covariance) != (k, k):
            raise ValueError(f"{name} covariance shape mismatch")
        if _shape(fit.input_vector) != (k,) or _shape(fit.output_vector) != (n,):
            raise ValueError(f"{name} codec-vector shape mismatch")
        for label, tensor in (("covariance", fit.covariance), ("input_vector", fit.input_vector), ("output_vector", fit.output_vector)):
            _require_finite_tensor(tensor, f"{name} {label}")
        if _dtype(fit.input_vector) != "float16" or _dtype(fit.output_vector) != "float16":
            raise ValueError(f"{name} stored codec vectors must be exact FP16 bytes")
        if not fit.fit_identity or not fit.transform_identity:
            raise ValueError(f"{name} lacks fit/transform identity")
        if fit.bit_vectors is not None:
            if set(fit.bit_vectors) != {3, 4, 5}:
                raise ValueError(f"{name} bit-specific GSS vectors must cover K3/K4/K5")
            for bit, pair in fit.bit_vectors.items():
                if not isinstance(pair, tuple) or len(pair) != 2 or _shape(pair[0]) != (k,) or _shape(pair[1]) != (n,):
                    raise ValueError(f"{name} K{bit} bit-specific codec-vector shape mismatch")
                _require_finite_tensor(pair[0], f"{name} K{bit} suh")
                _require_finite_tensor(pair[1], f"{name} K{bit} svh")
                if _dtype(pair[0]) != "float16" or _dtype(pair[1]) != "float16":
                    raise ValueError(f"{name} K{bit} stored GSS vectors must be exact FP16 bytes")
    hidden = gate_shape[1]
    fisher_rank: int | None = None
    fisher_presence: set[bool] = set()
    fisher_identities: set[tuple[str, str]] = set()
    heldout_row_keys: set[str] = set()
    heldout_documents: set[str] = set()
    batch_count = 0
    for batch in _iter_batches(item):
        batch_count += 1
        if not isinstance(batch, RoutedExpertBatch):
            raise TypeError("held-out batch source must yield RoutedExpertBatch values")
        if not batch.batch_id:
            raise ValueError("held-out batch_id must be non-empty")
        batch_identity = dict(batch.identity or {})
        if batch_identity.get("role") != "selection":
            raise ValueError(
                f"held-out batch {batch.batch_id} must come from the selection role"
            )
        declared_layer = batch_identity.get("layer")
        declared_expert = batch_identity.get("expert")
        if (
            isinstance(declared_layer, bool)
            or not isinstance(declared_layer, int)
            or declared_layer != item.layer
            or isinstance(declared_expert, bool)
            or not isinstance(declared_expert, int)
            or declared_expert != item.expert
        ):
            raise ValueError(
                f"held-out batch {batch.batch_id} identity must bind layer {item.layer} and expert {item.expert}"
            )
        sealed_batch = _require_hash(batch_identity.get("batch_payload_sha256"), "routed batch payload")
        heldout_document = _require_hash(batch_identity.get("document_sha256"), "held-out document identity")
        heldout_documents.add(heldout_document)
        if routed_batch_sha256(batch) != sealed_batch:
            raise ValueError(f"held-out batch {batch.batch_id} differs from its sealed payload identity")
        if len(_shape(batch.hidden_states)) != 2 or _shape(batch.hidden_states)[1] != hidden:
            raise ValueError(f"held-out batch {batch.batch_id} hidden shape mismatch")
        rows = _shape(batch.hidden_states)[0]
        if batch.row_keys is None or len(batch.row_keys) != rows or any(not isinstance(key, str) or not key for key in batch.row_keys):
            raise ValueError(f"held-out batch {batch.batch_id} row keys are missing or malformed")
        if len(set(batch.row_keys)) != rows or heldout_row_keys.intersection(batch.row_keys):
            raise ValueError(f"held-out batch {batch.batch_id} row keys are duplicated")
        heldout_row_keys.update(batch.row_keys)
        if _shape(batch.route_weights) != (rows,):
            raise ValueError(f"held-out batch {batch.batch_id} route weight shape mismatch")
        _require_finite_tensor(batch.hidden_states, f"batch {batch.batch_id} hidden")
        _require_finite_tensor(batch.route_weights, f"batch {batch.batch_id} route weights")
        import torch

        if bool((torch.as_tensor(batch.route_weights) < 0).any().item()):
            raise ValueError(f"held-out batch {batch.batch_id} has negative route weights")
        if batch.source_route_indices is None or batch.source_route_weights is None:
            raise ValueError(f"held-out batch {batch.batch_id} cannot prove membership of expert {item.expert}")
        if (batch.candidate_route_indices is None) != (batch.candidate_route_weights is None):
            raise ValueError(f"held-out batch {batch.batch_id} has an incomplete candidate route input")
        candidate_indices_value = (
            batch.source_route_indices if batch.candidate_route_indices is None else batch.candidate_route_indices
        )
        candidate_weights_value = (
            batch.source_route_weights if batch.candidate_route_weights is None else batch.candidate_route_weights
        )
        if _shape(batch.source_route_indices) != _shape(candidate_indices_value):
                raise ValueError(f"held-out batch {batch.batch_id} route index shape mismatch")
        if _shape(batch.source_route_weights) != _shape(batch.source_route_indices) or _shape(candidate_weights_value) != _shape(candidate_indices_value):
                raise ValueError(f"held-out batch {batch.batch_id} route weight shape mismatch")
        if _shape(batch.source_route_indices)[0] != rows:
                raise ValueError(f"held-out batch {batch.batch_id} route row mismatch")
        _require_finite_tensor(batch.source_route_weights, f"batch {batch.batch_id} source route weights")
        _require_finite_tensor(candidate_weights_value, f"batch {batch.batch_id} candidate route weights")
        source_indices = torch.as_tensor(batch.source_route_indices)
        candidate_indices = torch.as_tensor(candidate_indices_value)
        if source_indices.dtype.is_floating_point or candidate_indices.dtype.is_floating_point:
                raise ValueError(f"held-out batch {batch.batch_id} route indices must be integral")
        if bool((source_indices < 0).any().item()) or bool((candidate_indices < 0).any().item()):
                raise ValueError(f"held-out batch {batch.batch_id} has negative route indices")
        if bool((torch.as_tensor(batch.source_route_weights) < 0).any().item()) or bool((torch.as_tensor(candidate_weights_value) < 0).any().item()):
                raise ValueError(f"held-out batch {batch.batch_id} has negative full-route weights")
        for indices in (source_indices, candidate_indices):
            if any(len(set(row)) != len(row) for row in indices.cpu().tolist()):
                raise ValueError(f"held-out batch {batch.batch_id} has duplicate experts in a route set")
        matches = source_indices.long() == item.expert
        if not bool((matches.sum(dim=1) == 1).all().item()):
            raise ValueError(f"held-out batch {batch.batch_id} must contain declared expert {item.expert} exactly once per row")
        selected = torch.as_tensor(batch.source_route_weights)[matches]
        declared = torch.as_tensor(batch.route_weights)
        if not torch.equal(selected.to(dtype=declared.dtype, device=declared.device), declared):
            raise ValueError(f"held-out batch {batch.batch_id} declared route weights do not match expert {item.expert}")
        fisher_presence.add(batch.fisher_gradients is not None)
        if batch.fisher_gradients is not None:
            batch_identity = dict(batch.identity or {})
            fisher_identities.add(
                (
                    _require_hash(batch_identity.get("fisher_probe_sha256"), "Fisher probe identity"),
                    _require_hash(batch_identity.get("fisher_window_sha256"), "Fisher window identity"),
                )
            )
            shape = _shape(batch.fisher_gradients)
            if len(shape) != 3 or shape[1:] != (rows, hidden):
                raise ValueError(f"held-out batch {batch.batch_id} Fisher gradient shape mismatch")
            if fisher_rank is None:
                fisher_rank = shape[0]
            elif fisher_rank != shape[0]:
                raise ValueError("Fisher rank differs across held-out batches")
            _require_finite_tensor(batch.fisher_gradients, f"batch {batch.batch_id} Fisher gradients")
    if len(fisher_presence) > 1:
        raise ValueError("Fisher/Jacobian projections must be present for all held-out batches or none")
    if len(fisher_identities) > 1:
        raise ValueError("Fisher/Jacobian batches do not share one probe and window identity")
    if batch_count == 0:
        raise ValueError("at least one routed held-out batch is required")
    fit_count = 0
    down_fit_row_keys: set[str] = set()
    down_fit_documents: set[str] = set()
    for batch in _iter_down_fit_batches(item):
        fit_count += 1
        if not isinstance(batch, ConditionalDownFitBatch) or not batch.batch_id:
            raise ValueError("conditional down fit source yielded a malformed batch")
        if len(_shape(batch.hidden_states)) != 2 or _shape(batch.hidden_states)[1] != hidden:
            raise ValueError(f"conditional down fit batch {batch.batch_id} hidden shape mismatch")
        rows = _shape(batch.hidden_states)[0]
        if len(batch.row_keys) != rows or any(not isinstance(key, str) or not key for key in batch.row_keys):
            raise ValueError(f"conditional down fit batch {batch.batch_id} row keys are missing or malformed")
        if len(set(batch.row_keys)) != rows or down_fit_row_keys.intersection(batch.row_keys):
            raise ValueError(f"conditional down fit batch {batch.batch_id} row keys are duplicated")
        down_fit_row_keys.update(batch.row_keys)
        if _shape(batch.route_weights) != (rows,):
            raise ValueError(f"conditional down fit batch {batch.batch_id} route weight shape mismatch")
        if _shape(batch.sampling_weights) != (rows,):
            raise ValueError(f"conditional down fit batch {batch.batch_id} sampling weight shape mismatch")
        if len(_shape(batch.source_route_indices)) != 2 or _shape(batch.source_route_indices)[0] != rows:
            raise ValueError(f"conditional down fit batch {batch.batch_id} source route index shape mismatch")
        if _shape(batch.source_route_weights) != _shape(batch.source_route_indices):
            raise ValueError(f"conditional down fit batch {batch.batch_id} source route weight shape mismatch")
        _require_finite_tensor(batch.hidden_states, f"conditional down fit batch {batch.batch_id} hidden")
        _require_finite_tensor(batch.route_weights, f"conditional down fit batch {batch.batch_id} route weights")
        _require_finite_tensor(batch.sampling_weights, f"conditional down fit batch {batch.batch_id} sampling weights")
        _require_finite_tensor(batch.source_route_weights, f"conditional down fit batch {batch.batch_id} source route weights")
        import torch

        if bool((torch.as_tensor(batch.route_weights) <= 0).any().item()) or bool((torch.as_tensor(batch.sampling_weights) <= 0).any().item()):
            raise ValueError(f"conditional down fit batch {batch.batch_id} has invalid route/sampling weights")
        route_indices = torch.as_tensor(batch.source_route_indices)
        full_route_weights = torch.as_tensor(batch.source_route_weights)
        if route_indices.dtype.is_floating_point or bool((route_indices < 0).any().item()):
            raise ValueError(f"conditional down fit batch {batch.batch_id} source route indices must be non-negative integers")
        if bool((full_route_weights < 0).any().item()):
            raise ValueError(f"conditional down fit batch {batch.batch_id} has negative source route weights")
        if any(len(set(row)) != len(row) for row in route_indices.cpu().tolist()):
            raise ValueError(f"conditional down fit batch {batch.batch_id} has duplicate experts in a route set")
        matches = route_indices.long() == item.expert
        if not bool((matches.sum(dim=1) == 1).all().item()):
            raise ValueError(
                f"conditional down fit batch {batch.batch_id} must contain declared expert {item.expert} exactly once per row"
            )
        selected_route_weights = full_route_weights[matches]
        declared_route_weights = torch.as_tensor(batch.route_weights)
        if not torch.equal(
            selected_route_weights.to(dtype=declared_route_weights.dtype, device=declared_route_weights.device),
            declared_route_weights,
        ):
            raise ValueError(
                f"conditional down fit batch {batch.batch_id} declared route weights do not match expert {item.expert}"
            )
        identity = dict(batch.identity or {})
        if identity.get("role") != "conditional_fit":
            raise ValueError(
                f"conditional down fit batch {batch.batch_id} must come from conditional_fit, not confirmation"
            )
        down_fit_documents.add(
            _require_hash(identity.get("document_sha256"), "conditional down fit document identity")
        )
        declared_layer = identity.get("layer")
        declared_expert = identity.get("expert")
        if (
            isinstance(declared_layer, bool)
            or not isinstance(declared_layer, int)
            or declared_layer != item.layer
            or isinstance(declared_expert, bool)
            or not isinstance(declared_expert, int)
            or declared_expert != item.expert
        ):
            raise ValueError(f"conditional down fit batch {batch.batch_id} declared layer/expert identity mismatch")
        sealed = _require_hash(identity.get("batch_payload_sha256"), "conditional down fit batch payload")
        if conditional_down_fit_batch_sha256(batch) != sealed:
            raise ValueError(f"conditional down fit batch {batch.batch_id} differs from its sealed payload identity")
        _require_hash(identity.get("conditional_down_fit_artifact_sha256"), "conditional down fit artifact")
        _require_hash(identity.get("row_identity_sha256"), "conditional down fit row identity")
    if fit_count == 0:
        raise ValueError("conditional down fitting requires at least one calibration batch")
    overlap = heldout_row_keys & down_fit_row_keys
    if overlap:
        raise ValueError(f"conditional down fit rows overlap held-out selection rows: {min(overlap)}")
    document_overlap = heldout_documents & down_fit_documents
    if document_overlap:
        raise ValueError("conditional down fit documents overlap held-out selection documents")


def _input_fingerprint(
    item: ExpertCandidateInput,
    objective_arm: str,
    attestation: Mapping[str, Any],
    expert_compute_dtype: str,
) -> dict[str, Any]:
    sources = {name: _tensor_record(value) for name, value in item.source.as_mapping().items()}
    fitted = {}
    for name, fit in sorted(item.fitted.items()):
        fitted[name] = {
            "covariance": _tensor_record(fit.covariance),
            "input_vector": _tensor_record(fit.input_vector),
            "output_vector": _tensor_record(fit.output_vector),
            "fit_identity": _json_value(dict(fit.fit_identity)),
            "transform_identity": _json_value(dict(fit.transform_identity)),
            "bit_vectors": None
            if fit.bit_vectors is None
            else {
                str(bit): {"suh": _tensor_record(pair[0]), "svh": _tensor_record(pair[1])}
                for bit, pair in sorted(fit.bit_vectors.items())
            },
        }
    batches = []
    for batch in _iter_batches(item):
        sealed = _require_hash(dict(batch.identity or {}).get("batch_payload_sha256"), "routed batch payload")
        if routed_batch_sha256(batch) != sealed:
            raise ValueError(f"held-out batch {batch.batch_id} differs from its sealed payload identity")
        row = {
            "batch_id": batch.batch_id,
            "row_keys": list(batch.row_keys or ()),
            "hidden_states": _tensor_record(batch.hidden_states),
            "route_weights": _tensor_record(batch.route_weights),
            "identity": _json_value(dict(batch.identity or {})),
        }
        for name in ("source_route_indices", "source_route_weights", "candidate_route_indices", "candidate_route_weights", "fisher_gradients"):
            value = getattr(batch, name)
            row[name] = None if value is None else _tensor_record(value)
        batches.append(row)
    down_fit_batches = []
    for batch in _iter_down_fit_batches(item):
        sealed = _require_hash(dict(batch.identity or {}).get("batch_payload_sha256"), "conditional down fit batch payload")
        if conditional_down_fit_batch_sha256(batch) != sealed:
            raise ValueError(f"conditional down fit batch {batch.batch_id} differs from its sealed payload identity")
        down_fit_batches.append(
            {
                "batch_id": batch.batch_id,
                "row_keys": list(batch.row_keys),
                "hidden_states": _tensor_record(batch.hidden_states),
                "route_weights": _tensor_record(batch.route_weights),
                "sampling_weights": _tensor_record(batch.sampling_weights),
                "source_route_indices": _tensor_record(batch.source_route_indices),
                "source_route_weights": _tensor_record(batch.source_route_weights),
                "identity": _json_value(dict(batch.identity)),
            }
        )
    return {
        "unit_id": item.unit_id,
        "source": sources,
        "fitted": fitted,
        "heldout_batches": batches,
        "conditional_down_fit_batches": down_fit_batches,
        "proposal_score_candidate_identity": (
            None
            if item.proposal_score_candidate_identity is None
            else _json_value(dict(item.proposal_score_candidate_identity))
        ),
        "objective_arm": objective_arm,
        "expert_function_compute_dtype": expert_compute_dtype,
        "backend_attestation_sha256": _hash_json(attestation),
    }


def _single_bound_value(values: Iterable[Any], *, label: str) -> Any:
    unique = {_json_value(value) for value in values}
    if len(unique) != 1:
        raise ValueError(f"{label} is missing or differs across the actual inputs")
    return next(iter(unique))


def _validate_item_run_identity(item: ExpertCandidateInput, run_identity: Mapping[str, Any]) -> None:
    fits = [dict(item.fitted[name].fit_identity) for name in PROJECTIONS]
    transforms = [dict(item.fitted[name].transform_identity) for name in PROJECTIONS]
    fit_hash = _single_bound_value((row.get("fit_artifact_sha256") for row in fits), label="fit artifact identity")
    model_revision = _single_bound_value((row.get("model_revision") for row in fits), label="fit model revision")
    dataset_revision = _single_bound_value((row.get("dataset_revision") for row in fits), label="fit dataset revision")
    predecessor = _single_bound_value((row.get("predecessor_state_sha256") for row in fits), label="fit predecessor identity")
    search_hash = _single_bound_value((row.get("search_artifact_sha256") for row in transforms), label="transform search identity")
    for value, key, label in (
        (fit_hash, "fit_artifact_sha256", "fit artifact"),
        (model_revision, "model_revision", "model revision"),
        (dataset_revision, "dataset_revision", "dataset revision"),
        (predecessor, "predecessor_state_sha256", "predecessor state"),
        (search_hash, "search_artifact_sha256", "transform search"),
    ):
        if value != run_identity[key]:
            raise ValueError(f"actual {label} differs from journal run identity")
    batch_identities = [dict(batch.identity or {}) for batch in _iter_batches(item)]
    for key, label in (
        ("heldout_artifact_sha256", "held-out artifact"),
        ("capture_artifact_sha256", "capture artifact"),
    ):
        actual = _single_bound_value((row.get(key) for row in batch_identities), label=label)
        if actual != run_identity[key]:
            raise ValueError(f"actual {label} differs from journal run identity")
    down_fit_identities = [dict(batch.identity or {}) for batch in _iter_down_fit_batches(item)]
    actual_down_fit = _single_bound_value(
        (row.get("conditional_down_fit_artifact_sha256") for row in down_fit_identities),
        label="conditional down fit artifact",
    )
    if actual_down_fit != run_identity["conditional_down_fit_artifact_sha256"]:
        raise ValueError("actual conditional down fit artifact differs from journal run identity")
    if actual_down_fit == run_identity["heldout_artifact_sha256"]:
        raise ValueError("conditional down fit and held-out selection artifacts must be disjoint")
    for triplet, decision in item.k5_screen.items():
        _validate_k5_decision(
            decision,
            expected_selection_artifact_sha256=run_identity["heldout_artifact_sha256"] if decision.admitted else None,
        )
    for key, label in (
        ("fisher_probe_sha256", "Fisher probe"),
        ("fisher_window_sha256", "Fisher window"),
    ):
        actual = _single_bound_value((row.get(key) for row in batch_identities), label=label)
        if actual != run_identity[key]:
            raise ValueError(f"actual {label} differs from journal run identity")


def _codec_candidates(
    codec: CodecAdapter,
    item: ExpertCandidateInput,
    triplets: Sequence[tuple[int, int, int]],
    provenance_base: Mapping[str, Any],
) -> dict[str, Any]:
    bits = {bit for triplet in triplets for bit in triplet}
    encoded: dict[str, Any] = {}
    source = item.source.as_mapping()
    for name in ("gate_proj", "up_proj"):
        fit = item.fitted[name]
        encoded[name] = {}
        bit_groups = [tuple(sorted(bits))] if fit.bit_vectors is None else [(bit,) for bit in sorted(bits)]
        for bit_group in bit_groups:
            suh, svh = (
                (fit.input_vector, fit.output_vector)
                if fit.bit_vectors is None
                else fit.bit_vectors[bit_group[0]]
            )
            values = codec.encode_candidates(
                unit_id=f"{item.unit_id}.{name}",
                weight_hf=source[name],
                covariance=fit.covariance,
                bits=bit_group,
                input_vector=suh,
                output_vector=svh,
                provenance=dict(provenance_base)
                | {
                    "projection": name,
                    "bit_specific_gss": fit.bit_vectors is not None,
                    "fit_identity": _json_value(dict(fit.fit_identity)),
                    "transform_identity": _json_value(dict(fit.transform_identity)),
                },
            )
            encoded[name].update(values)
        if set(encoded[name]) != bits:
            raise ValueError(f"codec omitted or added bit candidates for {name}")
        for bit, candidate in encoded[name].items():
            _validate_codec_candidate(candidate, source[name], name, bit)
    # The down-projection input is the decoded gate/up candidate function, so
    # its exact-codec Hessian is conditional on (gate_bit, up_bit).  Reusing a
    # single source-derived down encoding across all triplets would reproduce
    # neither the prior proven GLM pipeline nor the deployed expert function.
    encoded["down_proj"] = {}
    down_fit = item.fitted["down_proj"]
    for gate_bit, up_bit in sorted({(g, u) for g, u, _ in triplets}):
        conditional = _conditional_down_hessian(
            item,
            encoded["gate_proj"][gate_bit].reconstructed,
            encoded["up_proj"][up_bit].reconstructed,
        )
        down_bits = {d for g, u, d in triplets if (g, u) == (gate_bit, up_bit)}
        values = {}
        bit_groups = [tuple(sorted(down_bits))] if down_fit.bit_vectors is None else [(bit,) for bit in sorted(down_bits)]
        for bit_group in bit_groups:
            bit = bit_group[0]
            if down_fit.bit_vectors is not None:
                suh, svh = down_fit.bit_vectors[bit]
            else:
                suh, svh = down_fit.input_vector, down_fit.output_vector
            group_values = codec.encode_candidates(
                unit_id=f"{item.unit_id}.down_proj.g{gate_bit}u{up_bit}",
                weight_hf=source["down_proj"],
                covariance=conditional,
                bits=bit_group,
                input_vector=suh,
                output_vector=svh,
                provenance=dict(provenance_base)
                | {
                "projection": "down_proj",
                "bit_specific_gss": down_fit.bit_vectors is not None,
                "conditional_gate_bit": gate_bit,
                "conditional_up_bit": up_bit,
                "conditional_hessian": _tensor_record(conditional),
                "conditional_hessian_kind": "decoded-gate-up-weighted-raw-second-moment",
                "fit_identity": _json_value(dict(down_fit.fit_identity)),
                "transform_identity": _json_value(dict(down_fit.transform_identity)),
                },
            )
            values.update(group_values)
        if set(values) != down_bits:
            raise ValueError("codec omitted or added conditional down bit candidates")
        for bit, candidate in values.items():
            _validate_codec_candidate(candidate, source["down_proj"], "down_proj", bit)
        encoded["down_proj"][(gate_bit, up_bit)] = values
    return encoded


def _conditional_down_hessian(item: ExpertCandidateInput, gate: Any, up: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    route_power = item.fitted["down_proj"].fit_identity.get("route_weight_power")
    if route_power not in (0, 1, 2):
        raise ValueError("conditional down Hessian lacks a valid route-weight power")
    dimension = _shape(item.source.down_proj)[1]
    gram = torch.zeros((dimension, dimension), dtype=torch.float64)
    mass = torch.zeros((), dtype=torch.float64)
    for batch in _iter_down_fit_batches(item):
        hidden = torch.as_tensor(batch.hidden_states).to(dtype=torch.bfloat16)
        gate_value = functional.linear(hidden, torch.as_tensor(gate, device=hidden.device).to(torch.bfloat16))
        up_value = functional.linear(hidden, torch.as_tensor(up, device=hidden.device).to(torch.bfloat16))
        middle = (functional.silu(gate_value) * up_value).double().cpu()
        weights = (
            torch.as_tensor(batch.route_weights).double().cpu().pow(route_power)
            * torch.as_tensor(batch.sampling_weights).double().cpu()
        )
        gram += middle.T @ (middle * weights[:, None])
        mass += weights.sum()
    if not math.isfinite(float(mass)) or mass <= 0:
        raise ValueError("conditional down Hessian has zero or non-finite route mass")
    result = (gram / mass).float()
    _require_finite_tensor(result, "conditional down Hessian")
    return result


def _validate_codec_candidate(candidate: CodecCandidate, source: Any, projection: str, bit: int) -> None:
    if candidate.bits != bit:
        raise ValueError(f"codec bit identity mismatch for {projection}")
    if isinstance(candidate.stored_bytes, bool) or not isinstance(candidate.stored_bytes, int) or candidate.stored_bytes <= 0:
        raise ValueError(f"codec payload byte count is invalid for {projection} K{bit}")
    _require_hash(candidate.packed_sha256, f"{projection} K{bit} packed hash")
    _require_hash(candidate.reconstruction_sha256, f"{projection} K{bit} reconstruction hash")
    packed_record = _tensor_record(candidate.packed)
    if packed_record["sha256"] != candidate.packed_sha256:
        raise ValueError(f"codec packed bytes differ from its hash for {projection} K{bit}")
    if _shape(candidate.reconstructed) != _shape(source):
        raise ValueError(f"codec reconstruction shape mismatch for {projection} K{bit}")
    _require_finite_tensor(candidate.reconstructed, f"{projection} K{bit} reconstruction")
    # R10 hashes the deployed [K,N] FP16 reconstruction; the adapter exposes
    # the Hugging Face [N,K] orientation.  Verify the exact oracle, not a JSON
    # or float32 surrogate.
    try:
        import torch

        deployed = torch.as_tensor(candidate.reconstructed).T.contiguous().half()
    except ImportError:  # pragma: no cover
        import numpy as np

        deployed = np.ascontiguousarray(np.asarray(candidate.reconstructed).T.astype(np.float16))
    if sha256_bytes(_tensor_bytes(deployed)) != candidate.reconstruction_sha256:
        raise ValueError(f"codec reconstruction differs from its deployed hash for {projection} K{bit}")
    if packed_record["bytes"] > candidate.stored_bytes:
        raise ValueError(f"packed bytes exceed total codec payload for {projection} K{bit}")
    _json_value(candidate.metadata)


def _full_expert(hidden: Any, gate: Any, up: Any, down: Any, *, compute_dtype: str):
    import torch
    import torch.nn.functional as functional

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[compute_dtype]
    x = torch.as_tensor(hidden).to(dtype=dtype)
    gate_value = functional.linear(x, torch.as_tensor(gate, device=x.device).to(dtype=dtype))
    up_value = functional.linear(x, torch.as_tensor(up, device=x.device).to(dtype=dtype))
    middle = functional.silu(gate_value) * up_value
    return functional.linear(middle, torch.as_tensor(down, device=x.device).to(dtype=dtype))


def _route_agreement(batch: RoutedExpertBatch) -> dict[str, Any]:
    import torch

    if batch.candidate_route_indices is None:
        masses = [float(value) for value in torch.as_tensor(batch.source_route_weights).double().sum(dim=1).tolist()]
        return {
            "basis": "fixed-router-expert-weight-counterfactual",
            "route_set_scores": [1.0] * _shape(batch.hidden_states)[0],
            "route_mass_numerators": masses,
            "route_mass_denominators": masses,
        }
    source_index = torch.as_tensor(batch.source_route_indices).long().cpu()
    candidate_index = torch.as_tensor(batch.candidate_route_indices).long().cpu()
    source_weight = torch.as_tensor(batch.source_route_weights).double().cpu()
    candidate_weight = torch.as_tensor(batch.candidate_route_weights).double().cpu()
    set_scores: list[float] = []
    mass_numerators: list[float] = []
    mass_denominators: list[float] = []
    for row in range(source_index.shape[0]):
        source = {int(index): float(weight) for index, weight in zip(source_index[row].tolist(), source_weight[row].tolist(), strict=True)}
        candidate = {int(index): float(weight) for index, weight in zip(candidate_index[row].tolist(), candidate_weight[row].tolist(), strict=True)}
        union = set(source) | set(candidate)
        intersection = set(source) & set(candidate)
        set_scores.append(len(intersection) / len(union) if union else 1.0)
        mass_numerators.append(sum(min(source.get(key, 0.0), candidate.get(key, 0.0)) for key in union))
        mass_denominators.append(sum(max(source.get(key, 0.0), candidate.get(key, 0.0)) for key in union))
    return {
        "basis": "measured-source-versus-candidate-routes",
        "route_set_scores": set_scores,
        "route_mass_numerators": mass_numerators,
        "route_mass_denominators": mass_denominators,
    }


def _score_triplet(
    item: ExpertCandidateInput,
    triplet: tuple[int, int, int],
    encoded: Mapping[str, Any],
    expected_batches: Sequence[Mapping[str, Any]],
    expert_compute_dtype: str,
) -> dict[str, Any]:
    import torch

    source = item.source.as_mapping()
    gate_bit, up_bit, down_bit = triplet
    candidate = {
        "gate_proj": encoded["gate_proj"][gate_bit].reconstructed,
        "up_proj": encoded["up_proj"][up_bit].reconstructed,
        "down_proj": encoded["down_proj"][(gate_bit, up_bit)][down_bit].reconstructed,
    }
    weighted_sse = torch.zeros((), dtype=torch.float64)
    weighted_source_energy = torch.zeros((), dtype=torch.float64)
    unweighted_sse = torch.zeros((), dtype=torch.float64)
    unweighted_source_energy = torch.zeros((), dtype=torch.float64)
    signed_sum = torch.zeros((), dtype=torch.float64)
    aggregate_residual = torch.zeros(_shape(source["down_proj"])[0], dtype=torch.float64)
    individual_weighted_sse = {name: torch.zeros((), dtype=torch.float64) for name in PROJECTIONS}
    fisher_projection = None
    route_set_scores: list[float] = []
    route_mass_numerators: list[float] = []
    route_mass_denominators: list[float] = []
    route_bases: set[str] = set()
    observed_batches: list[dict[str, Any]] = []
    for batch in _iter_batches(item):
        row = {
            "batch_id": batch.batch_id,
            "row_keys": list(batch.row_keys or ()),
            "hidden_states": _tensor_record(batch.hidden_states),
            "route_weights": _tensor_record(batch.route_weights),
            "identity": _json_value(dict(batch.identity or {})),
        }
        for name in ("source_route_indices", "source_route_weights", "candidate_route_indices", "candidate_route_weights", "fisher_gradients"):
            value = getattr(batch, name)
            row[name] = None if value is None else _tensor_record(value)
        observed_batches.append(row)
        reference = _full_expert(
            batch.hidden_states,
            source["gate_proj"],
            source["up_proj"],
            source["down_proj"],
            compute_dtype=expert_compute_dtype,
        )
        output = _full_expert(
            batch.hidden_states,
            candidate["gate_proj"],
            candidate["up_proj"],
            candidate["down_proj"],
            compute_dtype=expert_compute_dtype,
        )
        delta = (output - reference).double()
        route_weights = torch.as_tensor(batch.route_weights, device=delta.device).double()
        weighted_delta = delta * route_weights[:, None]
        weighted_sse += weighted_delta.square().sum().cpu()
        weighted_source_energy += (reference.double() * route_weights[:, None]).square().sum().cpu()
        unweighted_sse += delta.square().sum().cpu()
        unweighted_source_energy += reference.double().square().sum().cpu()
        signed_sum += weighted_delta.sum().cpu()
        aggregate_residual += weighted_delta.sum(dim=0).cpu()
        for projection in PROJECTIONS:
            single = dict(source)
            single[projection] = candidate[projection]
            single_delta = (
                _full_expert(
                    batch.hidden_states,
                    single["gate_proj"],
                    single["up_proj"],
                    single["down_proj"],
                    compute_dtype=expert_compute_dtype,
                )
                - reference
            ).double()
            individual_weighted_sse[projection] += (single_delta * route_weights[:, None]).square().sum().cpu()
        if batch.fisher_gradients is not None:
            gradients = torch.as_tensor(batch.fisher_gradients, device=delta.device).double()
            projected = torch.einsum("rth,th->r", gradients, weighted_delta)
            fisher_projection = projected.cpu() if fisher_projection is None else fisher_projection + projected.cpu()
        agreement = _route_agreement(batch)
        route_set_scores.extend(agreement["route_set_scores"])
        route_mass_numerators.extend(agreement["route_mass_numerators"])
        route_mass_denominators.extend(agreement["route_mass_denominators"])
        route_bases.add(agreement["basis"])
    if observed_batches != list(expected_batches):
        raise ValueError("held-out batch source drifted between identity capture and scoring")
    absolute = float(weighted_sse.item())
    source_energy = float(weighted_source_energy.item())
    plain_source_energy = float(unweighted_source_energy.item())
    if source_energy <= 0.0 or plain_source_energy <= 0.0:
        raise ValueError("source expert output energy must be positive for relative scoring")
    interaction = absolute - sum(float(value.item()) for value in individual_weighted_sse.values())
    fisher_values = None if fisher_projection is None else fisher_projection.tolist()
    fisher_damage = None if fisher_projection is None else 0.5 * float(fisher_projection.square().mean().item())
    metrics = {
        "absolute_gate_squared_output_sse": absolute,
        "relative_output_sse": float(unweighted_sse.item()) / plain_source_energy,
        "energy_normalized_output_sse": absolute / source_energy,
        "signed_aggregate_error": float(signed_sum.item()),
        "signed_aggregate_l2": float(aggregate_residual.norm().item()),
        "interaction_term": interaction,
        "interaction_share": abs(interaction) / max(absolute, 1e-300),
        "individual_gate_squared_output_sse": {
            name: float(value.item()) for name, value in individual_weighted_sse.items()
        },
        "route_agreement": {
            "basis": sorted(route_bases),
            "route_set_agreement": math.fsum(route_set_scores) / len(route_set_scores),
            "route_mass_agreement": math.fsum(route_mass_numerators) / math.fsum(route_mass_denominators) if math.fsum(route_mass_denominators) else 1.0,
        },
        "fisher_projection": None
        if fisher_projection is None
        else {
            "values": fisher_values,
            "sha256": _hash_json(fisher_values),
            "damage": fisher_damage,
            "rank": len(fisher_values),
        },
    }
    _assert_finite_tree(metrics)
    return metrics


def _assert_finite_tree(value: Any, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is non-finite")


def _projection_payload_record(
    candidate: CodecCandidate,
    fit: FittedProjection,
    *,
    projection: str,
    bit: int,
    store: ExactPayloadStore,
) -> dict[str, Any]:
    packed = _tensor_record(candidate.packed)
    reconstructed_hf = _tensor_record(candidate.reconstructed)
    metadata = _json_value(candidate.metadata)
    suh, svh = (
        (fit.input_vector, fit.output_vector)
        if fit.bit_vectors is None
        else fit.bit_vectors[bit]
    )
    refs = {
        "packed_trellis": store.put_tensor(candidate.packed, role=f"{projection}.packed_trellis"),
        "suh": store.put_tensor(suh, role=f"{projection}.suh"),
        "svh": store.put_tensor(svh, role=f"{projection}.svh"),
        "reconstruction_hf": store.put_tensor(candidate.reconstructed, role=f"{projection}.reconstruction_hf"),
    }
    return {
        "bits": candidate.bits,
        "codec_reported_payload_bytes": candidate.stored_bytes,
        "packed": packed | {"codec_sha256": candidate.packed_sha256},
        "reconstruction_hf": reconstructed_hf,
        "reconstruction_deployed_fp16_sha256": candidate.reconstruction_sha256,
        "exact_payload_refs": refs,
        "codec_metadata": metadata,
        "codec_metadata_sha256": _hash_json(metadata),
        "fit_identity": _json_value(dict(fit.fit_identity)),
        "transform_identity": _json_value(dict(fit.transform_identity)),
    }


def _candidate_cost_breakdown(projections: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Reconcile codec, semantic allocation, and physical object accounting.

    Semantic expert-private slots are deliberately *not* content deduplicated:
    identical private values still occupy distinct runtime slots.  The two
    declared layer-shared vector families are charged once per layer.  Physical
    content-addressed bytes remain a separate observation.
    """
    projection_rows: dict[str, Any] = {}
    private_slots: list[dict[str, Any]] = []
    shared_slots: list[dict[str, Any]] = []
    physical: dict[str, Mapping[str, Any]] = {}
    for projection in PROJECTIONS:
        row = projections[projection]
        refs = row["exact_payload_refs"]
        role_refs = {
            "packed_trellis": refs["packed_trellis"],
            "suh": refs["suh"],
            "svh": refs["svh"],
        }
        logical = sum(int(ref["bytes"]) for ref in role_refs.values())
        reported = row.get("codec_reported_payload_bytes")
        if isinstance(reported, bool) or not isinstance(reported, int) or reported != logical:
            raise ValueError(
                f"{projection} codec-reported payload bytes do not equal exact packed-plus-vector bytes"
            )
        projection_rows[projection] = {
            "codec_reported_logical_bytes": reported,
            "derived_logical_bytes": logical,
            "packed_trellis_bytes": int(refs["packed_trellis"]["bytes"]),
            "suh_bytes": int(refs["suh"]["bytes"]),
            "svh_bytes": int(refs["svh"]["bytes"]),
        }
        for role, ref in role_refs.items():
            physical[ref["sha256"]] = ref
            slot = {
                "projection": projection,
                "role": role,
                "sha256": ref["sha256"],
                "bytes": int(ref["bytes"]),
            }
            is_shared = (projection in {"gate_proj", "up_proj"} and role == "suh") or (
                projection == "down_proj" and role == "svh"
            )
            if is_shared:
                slot["shared_group"] = "gate_up.suh" if role == "suh" else "down.svh"
            (shared_slots if is_shared else private_slots).append(slot)
    gate_shared = projections["gate_proj"]["exact_payload_refs"]["suh"]
    up_shared = projections["up_proj"]["exact_payload_refs"]["suh"]
    if gate_shared["sha256"] != up_shared["sha256"] or gate_shared["bytes"] != up_shared["bytes"]:
        raise ValueError("gate/up declared layer-shared suh bytes differ")
    shared_objects = {
        slot["shared_group"]: {
            "shared_group": slot["shared_group"],
            "sha256": slot["sha256"],
            "bytes": slot["bytes"],
        }
        for slot in shared_slots
    }
    if any(
        shared_objects[slot["shared_group"]]["bytes"] != slot["bytes"]
        or shared_objects[slot["shared_group"]]["sha256"] != slot["sha256"]
        for slot in shared_slots
    ):
        raise ValueError("one semantic shared group contains different bytes")
    body = {
        "schema": SCHEMA_COST_BREAKDOWN,
        "byte_semantics": "codec-payload-including-codec-vectors-excluding-container",
        "projections": projection_rows,
        "codec_reported_logical_bytes": sum(row["codec_reported_logical_bytes"] for row in projection_rows.values()),
        "semantic_expert_private_slots": private_slots,
        "semantic_expert_private_bytes": sum(slot["bytes"] for slot in private_slots),
        "semantic_layer_shared_objects": sorted(shared_objects.values(), key=lambda row: row["shared_group"]),
        "semantic_layer_shared_bytes": sum(row["bytes"] for row in shared_objects.values()),
        "semantic_allocated_bytes_including_layer_fixed": sum(slot["bytes"] for slot in private_slots)
        + sum(row["bytes"] for row in shared_objects.values()),
        "physical_deployment_object_sha256": sorted(physical),
        "physical_deployment_bytes": sum(int(ref["bytes"]) for ref in physical.values()),
    }
    body["cost_breakdown_sha256"] = _hash_json(body)
    return body


def _candidate_record(
    item: ExpertCandidateInput,
    triplet: tuple[int, int, int],
    encoded: Mapping[str, Any],
    metrics: Mapping[str, Any],
    objective_arm: str,
    attestation: Mapping[str, Any],
    input_sha256: str,
    scoring_inputs: Mapping[str, Any],
    *,
    payload_store: ExactPayloadStore,
    competitive: bool,
    allow_test_backend: bool,
) -> dict[str, Any]:
    gate_bit, up_bit, down_bit = triplet
    selected = {
        "gate_proj": encoded["gate_proj"][gate_bit],
        "up_proj": encoded["up_proj"][up_bit],
        "down_proj": encoded["down_proj"][(gate_bit, up_bit)][down_bit],
    }
    projections = {}
    for index, name in enumerate(PROJECTIONS):
        projections[name] = _projection_payload_record(
            selected[name],
            item.fitted[name],
            projection=name,
            bit=triplet[index],
            store=payload_store,
        )
    deployment_refs = [
        row["exact_payload_refs"][role]
        for row in projections.values()
        for role in ("packed_trellis", "suh", "svh")
    ]
    unique_deployment = {row["sha256"]: row for row in deployment_refs}
    cost_breakdown = _candidate_cost_breakdown(projections)
    shared_hashes = {row["sha256"] for row in cost_breakdown["semantic_layer_shared_objects"]}
    shared_payload_bytes = cost_breakdown["semantic_layer_shared_bytes"]
    payload_bytes = cost_breakdown["semantic_expert_private_bytes"]
    artifact_refs = [
        ref
        for row in projections.values()
        for ref in row["exact_payload_refs"].values()
    ]
    unique_artifacts = {row["sha256"]: row for row in artifact_refs}
    payload_identity = {
        name: {
            "bits": row["bits"],
            "codec_reported_payload_bytes": row["codec_reported_payload_bytes"],
            "packed_sha256": row["packed"]["codec_sha256"],
            "reconstruction_sha256": row["reconstruction_deployed_fp16_sha256"],
            "transform_identity": row["transform_identity"],
        }
        for name, row in projections.items()
    }
    predicted = metrics[OBJECTIVE_METRICS[objective_arm]] if objective_arm != "fisher_projection" else (
        None if metrics["fisher_projection"] is None else metrics["fisher_projection"]["damage"]
    )
    if predicted is None:
        raise ValueError("fisher_projection objective requires Fisher/Jacobian scoring inputs")
    record = {
        "schema": SCHEMA_CANDIDATE,
        "candidate_id": f"{item.unit_id}.g{triplet[0]}u{triplet[1]}d{triplet[2]}",
        "unit_id": item.unit_id,
        "layer": item.layer,
        "expert": item.expert,
        "bit_triplet": list(triplet),
        "rate_class": "K3/K4-complete" if 5 not in triplet else "K5-screened-admission",
        "k5_admission": None if 5 not in triplet else item.k5_screen[triplet].as_dict(),
        "objective_arm": objective_arm,
        "predicted_damage": float(predicted),
        "payload_bytes": payload_bytes,
        "shared_layer_payload_bytes": shared_payload_bytes,
        "cost_breakdown": cost_breakdown,
        "physical_payload_accounting": {
            "policy": "content-addressed-physical-observation-not-allocator-cost",
            "unit_incremental_object_sha256": sorted(set(unique_deployment) - shared_hashes),
            "layer_shared_object_sha256": sorted(shared_hashes),
            "deployment_physical_bytes": sum(row["bytes"] for row in unique_deployment.values()),
            "artifact_object_sha256": sorted(unique_artifacts),
            "artifact_physical_bytes": sum(row["bytes"] for row in unique_artifacts.values()),
        },
        "payload_sha256": _hash_json(payload_identity),
        "projections": projections,
        "metrics": _json_value(metrics),
        "backend_attestation": _json_value(attestation),
        "scoring_inputs": _json_value(scoring_inputs),
        "scoring_inputs_sha256": _hash_json(scoring_inputs),
        "input_sha256": input_sha256,
        "finite_validation": True,
    }
    record["record_sha256"] = _hash_json(record)
    validate_candidate_record(
        record,
        competitive=competitive,
        allow_test_backend=allow_test_backend,
    )
    return record


def validate_candidate_record(
    record: Mapping[str, Any],
    *,
    competitive: bool = True,
    allow_test_backend: bool = False,
) -> None:
    if record.get("schema") != SCHEMA_CANDIDATE:
        raise ValueError("unsupported candidate schema")
    if not isinstance(record.get("candidate_id"), str) or not record["candidate_id"]:
        raise ValueError("candidate_id must be non-empty")
    if (
        isinstance(record.get("layer"), bool)
        or not isinstance(record.get("layer"), int)
        or record["layer"] < 0
        or isinstance(record.get("expert"), bool)
        or not isinstance(record.get("expert"), int)
        or record["expert"] < 0
    ):
        raise ValueError("candidate layer/expert must be non-negative integers")
    if record.get("unit_id") != f"L{record.get('layer')}.E{record.get('expert')}":
        raise ValueError("candidate layer/expert/unit identity mismatch")
    triplet = tuple(record.get("bit_triplet", ()))
    if len(triplet) != 3 or any(isinstance(bit, bool) or bit not in (3, 4, 5) for bit in triplet):
        raise ValueError("invalid candidate bit triplet")
    expected_candidate_id = f"{record['unit_id']}.g{triplet[0]}u{triplet[1]}d{triplet[2]}"
    if record["candidate_id"] != expected_candidate_id:
        raise ValueError("candidate_id is not derived from unit and bit triplet")
    if 5 in triplet:
        admission = record.get("k5_admission")
        if not isinstance(admission, dict) or admission.get("admitted") is not True or not admission.get("reason"):
            raise ValueError("K5 candidate lacks an explicit admitted screening decision")
        _validate_k5_decision(
            K5Decision(
                admitted=admission["admitted"],
                reason=admission["reason"],
                screening=admission.get("screening"),
            )
        )
    elif record.get("k5_admission") is not None:
        raise ValueError("K3/K4 candidate unexpectedly contains K5 admission")
    objective = record.get("objective_arm")
    if objective not in OBJECTIVE_METRICS:
        raise ValueError("unknown candidate objective arm")
    projections = record.get("projections")
    if not isinstance(projections, dict) or set(projections) != set(PROJECTIONS):
        raise ValueError("candidate projection payloads are incomplete")
    deployment_refs: list[Mapping[str, Any]] = []
    identity = {}
    for name, bit in zip(PROJECTIONS, triplet, strict=True):
        row = projections[name]
        if row.get("bits") != bit or not isinstance(row.get("codec_reported_payload_bytes"), int) or isinstance(row.get("codec_reported_payload_bytes"), bool) or row["codec_reported_payload_bytes"] <= 0:
            raise ValueError(f"invalid projection payload for {name}")
        _require_hash(row.get("packed", {}).get("codec_sha256"), f"{name} packed hash")
        _require_hash(row.get("packed", {}).get("sha256"), f"{name} observed packed hash")
        if row["packed"]["codec_sha256"] != row["packed"]["sha256"]:
            raise ValueError(f"{name} codec/observed packed hash mismatch")
        _require_hash(row.get("reconstruction_deployed_fp16_sha256"), f"{name} reconstruction hash")
        _require_hash(row.get("reconstruction_hf", {}).get("sha256"), f"{name} observed HF reconstruction hash")
        if not row.get("fit_identity") or not row.get("transform_identity"):
            raise ValueError(f"{name} lacks fit/transform identity")
        _require_hash(row.get("codec_metadata_sha256"), f"{name} codec metadata hash")
        if _hash_json(row.get("codec_metadata")) != row["codec_metadata_sha256"]:
            raise ValueError(f"{name} codec metadata hash mismatch")
        refs = row.get("exact_payload_refs")
        if not isinstance(refs, dict) or set(refs) != {"packed_trellis", "suh", "svh", "reconstruction_hf"}:
            raise ValueError(f"{name} exact payload references are incomplete")
        for role, ref in refs.items():
            if not isinstance(ref, dict) or ref.get("schema") != SCHEMA_PAYLOAD_REF or ref.get("role") != f"{name}.{role}":
                raise ValueError(f"{name} {role} exact payload reference is malformed")
            _require_hash(ref.get("sha256"), f"{name} {role} payload hash")
            if not isinstance(ref.get("bytes"), int) or isinstance(ref.get("bytes"), bool) or ref["bytes"] <= 0:
                raise ValueError(f"{name} {role} exact payload byte count is invalid")
            if ref.get("path") != f"objects/{ref['sha256'][:2]}/{ref['sha256']}.bin":
                raise ValueError(f"{name} {role} payload path is not content-addressed")
            if not isinstance(ref.get("dtype"), str) or not isinstance(ref.get("shape"), list):
                raise ValueError(f"{name} {role} payload tensor descriptor is malformed")
        if refs["packed_trellis"]["sha256"] != row["packed"]["sha256"]:
            raise ValueError(f"{name} packed payload reference hash mismatch")
        if refs["reconstruction_hf"]["sha256"] != row["reconstruction_hf"]["sha256"]:
            raise ValueError(f"{name} reconstruction payload reference hash mismatch")
        deployment_refs.extend(refs[role] for role in ("packed_trellis", "suh", "svh"))
        identity[name] = {
            "bits": row["bits"],
            "codec_reported_payload_bytes": row["codec_reported_payload_bytes"],
            "packed_sha256": row["packed"]["codec_sha256"],
            "reconstruction_sha256": row["reconstruction_deployed_fp16_sha256"],
            "transform_identity": row["transform_identity"],
        }
    unique = {ref["sha256"]: ref for ref in deployment_refs}
    shared = {
        projections["gate_proj"]["exact_payload_refs"]["suh"]["sha256"],
        projections["down_proj"]["exact_payload_refs"]["svh"]["sha256"],
    }
    derived_cost = _candidate_cost_breakdown(projections)
    if record.get("cost_breakdown") != derived_cost:
        raise ValueError("candidate cost breakdown differs from exact payload descriptors")
    payload = derived_cost["semantic_expert_private_bytes"]
    shared_bytes = derived_cost["semantic_layer_shared_bytes"]
    accounting = record.get("physical_payload_accounting")
    if not isinstance(accounting, dict) or accounting.get("unit_incremental_object_sha256") != sorted(set(unique) - shared):
        raise ValueError("candidate physical payload accounting is malformed")
    if accounting.get("layer_shared_object_sha256") != sorted(shared):
        raise ValueError("candidate shared-vector accounting is malformed")
    if accounting.get("deployment_physical_bytes") != sum(ref["bytes"] for ref in unique.values()):
        raise ValueError("candidate physical deployment byte accounting is stale")
    if record.get("payload_bytes") != payload or record.get("shared_layer_payload_bytes") != shared_bytes or record.get("payload_sha256") != _hash_json(identity):
        raise ValueError("candidate payload aggregate mismatch")
    proposal_identity = record.get("scoring_inputs", {}).get("proposal_score_candidate_identity")
    if triplet == (4, 4, 4) and proposal_identity is not None:
        expected_names = {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj.g4u4": "down_proj",
        }
        if set(proposal_identity) != set(expected_names):
            raise ValueError("uniform-K4 proposal-score candidate identity is incomplete")
        for score_name, projection in expected_names.items():
            expected = proposal_identity[score_name]
            observed = projections[projection]
            if (
                expected.get("bits") != 4
                or expected.get("packed_sha256") != observed["packed"]["codec_sha256"]
                or expected.get("reconstruction_deployed_fp16_sha256")
                != observed["reconstruction_deployed_fp16_sha256"]
                or expected.get("reconstruction_hf_sha256")
                != observed["reconstruction_hf"]["sha256"]
            ):
                raise ValueError(
                    f"uniform-K4 canonical ledger bytes differ from proposal scoring for {score_name}"
                )
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("candidate metrics are missing")
    required_metrics = {
        "absolute_gate_squared_output_sse",
        "relative_output_sse",
        "energy_normalized_output_sse",
        "signed_aggregate_error",
        "signed_aggregate_l2",
        "interaction_term",
        "interaction_share",
        "individual_gate_squared_output_sse",
        "route_agreement",
        "fisher_projection",
    }
    if set(metrics) != required_metrics:
        raise ValueError("candidate metric set is incomplete")
    _assert_finite_tree(metrics)
    for key in ("absolute_gate_squared_output_sse", "relative_output_sse", "energy_normalized_output_sse", "signed_aggregate_l2", "interaction_share"):
        if not isinstance(metrics[key], (int, float)) or isinstance(metrics[key], bool) or metrics[key] < 0:
            raise ValueError(f"candidate metric {key} must be non-negative")
    for key in ("signed_aggregate_error", "interaction_term"):
        if not isinstance(metrics[key], (int, float)) or isinstance(metrics[key], bool):
            raise ValueError(f"candidate metric {key} must be numeric")
    individual = metrics["individual_gate_squared_output_sse"]
    if not isinstance(individual, dict) or set(individual) != set(PROJECTIONS):
        raise ValueError("individual projection SSE metrics are incomplete")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in individual.values()):
        raise ValueError("individual projection SSE metrics must be non-negative")
    route = metrics["route_agreement"]
    if not isinstance(route, dict) or set(route) != {"basis", "route_set_agreement", "route_mass_agreement"}:
        raise ValueError("route agreement metrics are incomplete")
    if not isinstance(route["basis"], list) or not route["basis"] or any(not isinstance(value, str) or not value for value in route["basis"]):
        raise ValueError("route agreement basis is invalid")
    for key in ("route_set_agreement", "route_mass_agreement"):
        if not isinstance(route.get(key), (int, float)) or not 0.0 <= route[key] <= 1.0:
            raise ValueError(f"route agreement {key} is invalid")
    fisher = metrics["fisher_projection"]
    if fisher is not None:
        if not isinstance(fisher, dict) or set(fisher) != {"values", "sha256", "damage", "rank"}:
            raise ValueError("Fisher projection metrics are malformed")
        if not isinstance(fisher["values"], list) or isinstance(fisher["rank"], bool) or fisher["rank"] != len(fisher["values"]) or fisher["rank"] < 1:
            raise ValueError("Fisher projection rank is invalid")
        if fisher["sha256"] != _hash_json(fisher["values"]):
            raise ValueError("Fisher projection hash mismatch")
        if isinstance(fisher["damage"], bool) or not isinstance(fisher["damage"], (int, float)) or fisher["damage"] < 0:
            raise ValueError("Fisher projection damage must be non-negative")
    expected = metrics[OBJECTIVE_METRICS[objective]] if objective != "fisher_projection" else (
        None if metrics["fisher_projection"] is None else metrics["fisher_projection"]["damage"]
    )
    if expected is None or record.get("predicted_damage") != expected:
        raise ValueError("predicted damage does not match the declared objective arm")
    if isinstance(record["predicted_damage"], bool) or not isinstance(record["predicted_damage"], (int, float)) or record["predicted_damage"] < 0:
        raise ValueError("predicted damage must be non-negative")
    if record.get("finite_validation") is not True:
        raise ValueError("candidate lacks finite validation")
    attestation = record.get("backend_attestation")
    if competitive:
        if not isinstance(attestation, dict) or attestation.get("schema") != SCHEMA_ATTESTATION:
            raise ValueError("competitive candidate lacks backend attestation")
        if attestation.get("backend") != CORRECTED_BACKEND or attestation.get("codec_name") != CORRECTED_CODEC_NAME:
            raise ValueError("competitive candidate was not produced by corrected EXL3/MCG")
        if attestation.get("test_only") not in (True, False):
            raise ValueError("backend attestation test-only status is invalid")
        if attestation.get("test_only") and not allow_test_backend:
            raise ValueError("test-only backend cannot validate as a production competitive candidate")
        if _hash_json(attestation.get("codec_identity")) != attestation.get("codec_identity_sha256"):
            raise ValueError("candidate backend attestation hash mismatch")
        for name, row in projections.items():
            metadata = row["codec_metadata"]
            if metadata.get("codec_identity") != attestation.get("codec_identity") or metadata.get("codec_identity_sha256") != attestation.get("codec_identity_sha256"):
                raise ValueError(f"{name} codec metadata is not bound to the backend attestation")
            transform_status = row["transform_identity"].get("selection_status")
            if not attestation.get("test_only") and transform_status == "fixed-reproducible-baseline-not-multidraw-searched":
                raise ValueError("fixed-transform baseline cannot validate as production")
    scoring_inputs = record.get("scoring_inputs")
    if not isinstance(scoring_inputs, dict):
        raise ValueError("candidate scoring-input identity is missing")
    _require_hash(record.get("scoring_inputs_sha256"), "candidate scoring-input hash")
    if _hash_json(scoring_inputs) != record["scoring_inputs_sha256"]:
        raise ValueError("candidate scoring-input identity hash mismatch")
    if scoring_inputs.get("unit_id") != record["unit_id"] or scoring_inputs.get("objective_arm") != objective:
        raise ValueError("candidate scoring-input identity is inconsistent")
    if scoring_inputs.get("expert_function_compute_dtype") not in {"bfloat16", "float32"}:
        raise ValueError("candidate expert-function compute dtype is invalid")
    if competitive and scoring_inputs["expert_function_compute_dtype"] != "bfloat16":
        raise ValueError("competitive candidate expert-function scoring is not BF16")
    expected_input = _hash_json(
        {
            "base_sha256": record["scoring_inputs_sha256"],
            "triplet": triplet,
            "k5_admission": record["k5_admission"],
        }
    )
    if record.get("input_sha256") != expected_input:
        raise ValueError("candidate input hash is inconsistent with its scoring inputs")
    _require_hash(record.get("input_sha256"), "candidate input hash")
    _require_hash(record.get("record_sha256"), "candidate record hash")
    if _hash_json({key: value for key, value in record.items() if key != "record_sha256"}) != record["record_sha256"]:
        raise ValueError("candidate record hash mismatch")


def allocator_handoff(
    records: Iterable[Mapping[str, Any]],
    *,
    competitive: bool = True,
    allow_test_backend: bool = False,
) -> list[Candidate]:
    result = []
    for raw in records:
        record = dict(raw)
        validate_candidate_record(
            record,
            competitive=competitive,
            allow_test_backend=allow_test_backend,
        )
        result.append(
            Candidate(
                unit_id=record["unit_id"],
                choice_id=record["candidate_id"],
                stored_bytes=record["payload_bytes"],
                predicted_damage=record["predicted_damage"],
                metadata=record,
            )
        )
    return result


def layer_shared_costs(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return layer-fixed semantic costs without cross-layer content dedup."""
    by_layer: dict[int, tuple[tuple[str, str, int], ...]] = {}
    for record in records:
        layer = int(record["layer"])
        cost = record.get("cost_breakdown")
        if not isinstance(cost, dict):
            raise ValueError("candidate lacks a verified cost breakdown")
        objects = tuple(
            sorted(
                (str(row["shared_group"]), str(row["sha256"]), int(row["bytes"]))
                for row in cost["semantic_layer_shared_objects"]
            )
        )
        incumbent = by_layer.setdefault(layer, objects)
        if incumbent != objects:
            raise ValueError(f"layer {layer} candidates disagree on layer-shared payload bytes")
    return [
        {
            "layer": layer,
            "objects": [
                {"shared_group": group, "sha256": digest, "bytes": size}
                for group, digest, size in objects
            ],
            "semantic_layer_shared_bytes": sum(size for _, _, size in objects),
        }
        for layer, objects in sorted(by_layer.items())
    ]


def selected_allocation_cost(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the exact semantic budget for one selected candidate per unit."""
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("selected allocation must contain candidates")
    units = [row.get("unit_id") for row in rows]
    if len(units) != len(set(units)):
        raise ValueError("selected allocation must contain exactly one candidate per unit")
    for row in rows:
        validate_candidate_record(row, competitive=False, allow_test_backend=True)
    fixed = layer_shared_costs(rows)
    private = sum(int(row["cost_breakdown"]["semantic_expert_private_bytes"]) for row in rows)
    shared = sum(int(row["semantic_layer_shared_bytes"]) for row in fixed)
    fixed_by_layer = {int(row["layer"]): row for row in fixed}
    selected_layer_costs = []
    for layer in sorted(fixed_by_layer):
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        layer_private = sum(
            int(row["cost_breakdown"]["semantic_expert_private_bytes"])
            for row in layer_rows
        )
        layer_fixed = fixed_by_layer[layer]
        layer_shared = int(layer_fixed["semantic_layer_shared_bytes"])
        selected_layer_costs.append(
            {
                "layer": layer,
                "selected_candidate_record_sha256": sorted(
                    row["record_sha256"] for row in layer_rows
                ),
                "selected_candidate_identities": sorted(
                    (
                        {
                            "record_sha256": row["record_sha256"],
                            "candidate_id": row["candidate_id"],
                            "unit_id": row["unit_id"],
                        }
                        for row in layer_rows
                    ),
                    key=lambda row: row["record_sha256"],
                ),
                "semantic_expert_private_bytes": layer_private,
                "semantic_layer_shared_objects": layer_fixed["objects"],
                "semantic_layer_shared_bytes": layer_shared,
                "allocated_payload_bytes": layer_private + layer_shared,
            }
        )
    body = {
        "schema": "quant-pipeline.selected-allocation-cost.v1",
        "selected_candidate_record_sha256": sorted(row["record_sha256"] for row in rows),
        "semantic_expert_private_bytes": private,
        "layer_shared_costs": fixed,
        "selected_layer_costs": selected_layer_costs,
        "semantic_layer_shared_bytes": shared,
        "allocated_payload_bytes": private + shared,
    }
    body["allocation_cost_sha256"] = _hash_json(body)
    return body


def allocate_validated_records(
    records: Iterable[Mapping[str, Any]],
    *,
    byte_budget: int,
    quantum: int = 1,
    competitive: bool = True,
    allow_test_backend: bool = False,
    damage_overrides: Mapping[str, float] | None = None,
) -> ReconciledLedgerAllocation:
    """Allocate validated ledger records and prove allocator/cost closure.

    Candidate ``stored_bytes`` carry only expert-private semantic cost. The
    layer-shared term is independently derived over the complete candidate
    set, charged once per represented layer, and then reconciled against the
    exact records selected by the allocator.
    """

    rows = tuple(dict(row) for row in records)
    if not rows:
        raise ValueError("validated allocation requires candidate records")
    candidates = allocator_handoff(
        rows,
        competitive=competitive,
        allow_test_backend=allow_test_backend,
    )
    if damage_overrides is not None:
        expected = {candidate.choice_id for candidate in candidates}
        if set(damage_overrides) != expected:
            raise ValueError("damage override inventory differs from validated candidate records")
        overridden: list[Candidate] = []
        for candidate in candidates:
            damage = damage_overrides[candidate.choice_id]
            if isinstance(damage, bool) or not isinstance(damage, (int, float)) or not math.isfinite(float(damage)):
                raise ValueError("candidate damage override must be finite")
            overridden.append(
                Candidate(
                    unit_id=candidate.unit_id,
                    choice_id=candidate.choice_id,
                    stored_bytes=candidate.stored_bytes,
                    predicted_damage=float(damage),
                    metadata=candidate.metadata,
                )
            )
        candidates = overridden
    fixed_by_layer = layer_shared_costs(rows)
    fixed_bytes = sum(int(row["semantic_layer_shared_bytes"]) for row in fixed_by_layer)
    allocation = allocate_with_fixed_layer_cost(
        candidates,
        byte_budget=byte_budget,
        fixed_layer_shared_bytes=fixed_bytes,
        quantum=quantum,
    )
    selected_records = tuple(dict(choice.metadata or {}) for choice in allocation.choices)
    if any(not row for row in selected_records):
        raise RuntimeError("validated allocator lost its candidate-record metadata")
    selected_cost = selected_allocation_cost(selected_records)
    if selected_cost["semantic_expert_private_bytes"] != allocation.variable_payload_bytes:
        raise RuntimeError("allocator private bytes differ from selected candidate records")
    if selected_cost["semantic_layer_shared_bytes"] != allocation.fixed_layer_shared_bytes:
        raise RuntimeError("allocator shared bytes differ from selected candidate records")
    if selected_cost["allocated_payload_bytes"] != allocation.stored_bytes:
        raise RuntimeError("allocator total differs from selected candidate records")
    return ReconciledLedgerAllocation(allocation, selected_records, selected_cost)


def build_pareto_frontiers(
    records: Iterable[Mapping[str, Any]],
    *,
    competitive: bool = True,
    allow_test_backend: bool = False,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    candidates = allocator_handoff(
        records,
        competitive=competitive,
        allow_test_backend=allow_test_backend,
    )
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.unit_id, []).append(candidate)
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for unit_id, values in sorted(grouped.items()):
        match = re.fullmatch(r"L(\d+)\.E(\d+)", unit_id)
        if match is None:
            raise ValueError(f"invalid expert unit ID: {unit_id}")
        layer, expert = match.groups()
        result.setdefault(layer, {})[expert] = [dict(candidate.metadata or {}) for candidate in pareto_frontier(values)]
    return result


def validate_ledger(
    ledger: Mapping[str, Any],
    *,
    competitive: bool = True,
    allow_test_backend: bool = False,
) -> None:
    if ledger.get("schema") != SCHEMA_LEDGER:
        raise ValueError("unsupported candidate ledger schema")
    if ledger.get("competitive") is not competitive:
        raise ValueError("candidate ledger competitive-mode declaration mismatch")
    if not isinstance(ledger.get("fixed_transform_baseline_allowed"), bool):
        raise ValueError("candidate ledger fixed-transform baseline declaration is invalid")
    attestation = ledger.get("backend_attestation")
    if competitive and isinstance(attestation, dict) and not attestation.get("test_only") and ledger.get("fixed_transform_baseline_allowed"):
        raise ValueError("fixed-transform baseline cannot validate as a production ledger")
    objective_arm = ledger.get("objective_arm")
    if objective_arm not in OBJECTIVE_METRICS:
        raise ValueError("candidate ledger objective arm is invalid")
    if ledger.get("expert_function_compute_dtype") not in {"bfloat16", "float32"}:
        raise ValueError("candidate ledger expert-function compute dtype is invalid")
    if competitive and ledger["expert_function_compute_dtype"] != "bfloat16":
        raise ValueError("competitive candidate ledger expert-function scoring is not BF16")
    records = ledger.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError("candidate ledger is empty")
    run_identity = ledger.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("candidate ledger lacks its journal run identity")
    CandidateJournal._validate_run_identity(run_identity)
    if _hash_json(run_identity) != ledger.get("journal_identity_sha256"):
        raise ValueError("candidate ledger journal identity hash mismatch")
    inventory = ledger.get("expected_inventory")
    if inventory != run_identity.get("expected_inventory"):
        raise ValueError("candidate ledger expected inventory differs from journal identity")
    expected_units = [row["unit_id"] for row in inventory["units"]]
    seen: set[str] = set()
    by_unit: dict[str, set[tuple[int, int, int]]] = {}
    for record in records:
        validate_candidate_record(
            record,
            competitive=competitive,
            allow_test_backend=allow_test_backend,
        )
        candidate_id = record["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)
        if (
            record["objective_arm"] != objective_arm
            or record["backend_attestation"] != ledger.get("backend_attestation")
            or record["scoring_inputs"]["expert_function_compute_dtype"] != ledger["expert_function_compute_dtype"]
        ):
            raise ValueError("candidate record differs from ledger objective/backend identity")
        by_unit.setdefault(record["unit_id"], set()).add(tuple(record["bit_triplet"]))
    required = set(all_k3_k4_triplets())
    for unit_id, triplets in by_unit.items():
        if not required.issubset(triplets):
            raise ValueError(f"{unit_id} omits one or more required K3/K4 triplets")
    if sorted(by_unit, key=lambda value: tuple(map(int, re.fullmatch(r"L(\d+)\.E(\d+)", value).groups()))) != expected_units:
        raise ValueError("candidate ledger unit coverage differs from sealed expected inventory")
    k5_policy = ledger.get("k5_policy")
    if not isinstance(k5_policy, dict) or set(k5_policy) != set(expected_units):
        raise ValueError("candidate ledger K5 policy unit coverage is incomplete")
    expected_k5_labels = {f"g{g}u{u}d{d}" for g, u, d in all_k5_triplets()}
    for unit, policy in k5_policy.items():
        if not isinstance(policy, dict) or set(policy) != expected_k5_labels:
            raise ValueError(f"{unit} K5 policy does not contain all 19 decisions")
        for label, decision in policy.items():
            if not isinstance(decision, dict) or not isinstance(decision.get("admitted"), bool) or not decision.get("reason"):
                raise ValueError(f"{unit} K5 decision {label} is malformed")
            _validate_k5_decision(
                K5Decision(decision["admitted"], decision["reason"], decision.get("screening")),
                expected_selection_artifact_sha256=run_identity["heldout_artifact_sha256"] if decision["admitted"] else None,
            )
        admitted = {
            tuple(record["bit_triplet"])
            for record in records
            if record["unit_id"] == unit and 5 in record["bit_triplet"]
        }
        policy_admitted = {
            triplet
            for triplet in all_k5_triplets()
            if policy[f"g{triplet[0]}u{triplet[1]}d{triplet[2]}"]["admitted"]
        }
        if admitted != policy_admitted:
            raise ValueError(f"{unit} K5 candidates differ from embedded policy")
    if ledger.get("k5_policy_sha256") != _hash_json(k5_policy):
        raise ValueError("candidate ledger K5 policy hash mismatch")
    expected_frontiers = build_pareto_frontiers(
        records,
        competitive=competitive,
        allow_test_backend=allow_test_backend,
    )
    if ledger.get("pareto_frontiers") != expected_frontiers:
        raise ValueError("candidate ledger Pareto frontiers are stale or malformed")
    expected_handoff = [
        {
            "unit_id": value.unit_id,
            "choice_id": value.choice_id,
            "stored_bytes": value.stored_bytes,
            "predicted_damage": value.predicted_damage,
            "candidate_record_sha256": value.metadata["record_sha256"],
        }
        for value in allocator_handoff(
            records,
            competitive=competitive,
            allow_test_backend=allow_test_backend,
        )
    ]
    if ledger.get("allocator_handoff") != expected_handoff:
        raise ValueError("candidate ledger allocator handoff is stale or malformed")
    _require_hash(ledger.get("journal_identity_sha256"), "journal identity hash")
    inventory = ledger.get("journal_records")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("candidate ledger lacks journal inventory")
    for name, digest in inventory.items():
        if not isinstance(name, str):
            raise ValueError("journal inventory filename is invalid")
        _require_hash(digest, f"journal record {name}")
    record_index = ledger.get("journal_record_index")
    if not isinstance(record_index, list) or not record_index:
        raise ValueError("candidate ledger lacks journal record index")
    expected_journal_keys = {
        ("k5-screen", f"{unit}.g{g}u{u}d{d}")
        for unit in expected_units
        for g, u, d in all_k5_triplets()
    } | {("candidate", record["candidate_id"]) for record in records}
    if {(row.get("kind"), row.get("key")) for row in record_index} != expected_journal_keys:
        raise ValueError("journal record index differs from derived candidate/K5 inventory")
    if len(record_index) != len(expected_journal_keys):
        raise ValueError("journal record index contains duplicate keys")
    for row in record_index:
        expected_name = CandidateJournal._record_name(row["kind"], row["key"])
        if row.get("filename") != expected_name or inventory.get(expected_name) != row.get("file_sha256"):
            raise ValueError("journal record filename/hash inventory mismatch")
        _require_hash(row.get("input_sha256"), "journal indexed input")
        _require_hash(row.get("record_sha256"), "journal indexed record")
    candidate_inputs = {row["key"]: row["input_sha256"] for row in record_index if row["kind"] == "candidate"}
    if any(candidate_inputs.get(record["candidate_id"]) != record["input_sha256"] for record in records):
        raise ValueError("journal candidate input identity differs from candidate record")
    payload_manifest = ledger.get("exact_payload_store")
    if not isinstance(payload_manifest, dict) or payload_manifest.get("schema") != SCHEMA_PAYLOAD_MANIFEST:
        raise ValueError("candidate ledger lacks exact payload-store manifest")
    manifest_body = {key: value for key, value in payload_manifest.items() if key != "manifest_sha256"}
    if payload_manifest.get("manifest_sha256") != _hash_json(manifest_body):
        raise ValueError("exact payload-store manifest hash mismatch")
    objects = payload_manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("exact payload-store manifest is empty")
    object_map = {row.get("sha256"): row for row in objects}
    if len(object_map) != len(objects) or any(not isinstance(key, str) for key in object_map):
        raise ValueError("exact payload-store manifest has duplicate or invalid objects")
    all_refs = [
        ref
        for record in records
        for projection in record["projections"].values()
        for ref in projection["exact_payload_refs"].values()
    ]
    if set(object_map) != {ref["sha256"] for ref in all_refs}:
        raise ValueError("exact payload-store manifest does not exactly cover candidate payloads")
    for ref in all_refs:
        stable = {key: ref[key] for key in ("sha256", "bytes", "path")}
        if object_map[ref["sha256"]] != stable:
            raise ValueError("exact payload-store manifest descriptor differs from candidate reference")
    if payload_manifest.get("physical_bytes") != sum(row["bytes"] for row in objects):
        raise ValueError("exact payload-store physical byte count is stale")
    shared = {
        digest: object_map[digest]
        for record in records
        for digest in record["physical_payload_accounting"]["layer_shared_object_sha256"]
    }
    fixed_costs = layer_shared_costs(records)
    if (
        ledger.get("fixed_layer_shared_object_sha256") != sorted(shared)
        or ledger.get("fixed_layer_shared_physical_bytes") != sum(row["bytes"] for row in shared.values())
        or ledger.get("fixed_layer_shared_costs") != fixed_costs
        or ledger.get("fixed_layer_shared_payload_bytes")
        != sum(row["semantic_layer_shared_bytes"] for row in fixed_costs)
    ):
        raise ValueError("ledger fixed layer-shared payload accounting is stale")
    _require_hash(ledger.get("ledger_sha256"), "ledger hash")
    if _hash_json({key: value for key, value in ledger.items() if key != "ledger_sha256"}) != ledger["ledger_sha256"]:
        raise ValueError("candidate ledger hash mismatch")


class CandidateLedgerGenerator:
    def __init__(
        self,
        codec: CodecAdapter,
        attestation: BackendAttestation,
        *,
        objective_arm: str = "energy_normalized_sse",
        competitive: bool = True,
        allow_test_backend: bool = False,
        allow_fixed_transform_baseline: bool = False,
        expert_compute_dtype: str = "bfloat16",
    ) -> None:
        if objective_arm not in OBJECTIVE_METRICS:
            raise ValueError(f"unknown objective arm: {objective_arm}")
        if expert_compute_dtype not in {"bfloat16", "float32"}:
            raise ValueError("expert_compute_dtype must be bfloat16 or float32")
        if competitive and expert_compute_dtype != "bfloat16":
            raise ValueError("competitive expert-function scoring must use BF16 arithmetic")
        self.codec = codec
        self.objective_arm = objective_arm
        self.competitive = competitive
        self.allow_test_backend = allow_test_backend
        self.allow_fixed_transform_baseline = allow_fixed_transform_baseline
        self.expert_compute_dtype = expert_compute_dtype
        self._layer_shared_transform_hashes: dict[int, tuple[str, str]] = {}
        self.attestation = _validate_attestation(
            codec,
            attestation,
            competitive=competitive,
            allow_test_backend=allow_test_backend,
        )
        if competitive and not self.attestation["test_only"] and allow_fixed_transform_baseline:
            raise ValueError("fixed-transform baseline cannot be enabled for a production ledger")

    def prepare_expert_input(
        self,
        *,
        layer: int,
        expert: int,
        source: ProjectionTensors,
        gate_up_statistics: Any,
        down_statistics: Any,
        heldout_batches: Sequence[RoutedExpertBatch] | Callable[[], Iterable[RoutedExpertBatch]],
        k5_screen: Mapping[tuple[int, int, int], K5Decision],
        route_power: int,
        accounting: str,
        transform_seed_sha256: str,
        searched_transform: Any | None = None,
        conditional_down_fit_batches: Sequence[ConditionalDownFitBatch] | Callable[[], Iterable[ConditionalDownFitBatch]] | None = None,
    ) -> ExpertCandidateInput:
        """Bridge verified fitter artifacts to one streamable codec work unit.

        On the production adapter, resolving ``codebook_scale`` loads the
        already-attested corrected codec closure.  Callers must therefore use
        this operation only in an approved execution stage, never while merely
        planning a campaign.
        """
        if self.attestation["test_only"]:
            scale = getattr(self.codec, "codebook_scale", None)
            if scale is None:
                raise ValueError("test codec must expose its exact codebook_scale")
        else:
            from ..codecs.exl3_mcg import Exl3MCGCodec

            if not isinstance(self.codec, Exl3MCGCodec):  # defended again at this GPU-loading boundary
                raise TypeError("production fitter bridge requires Exl3MCGCodec")
            scale = self.codec._codec().codebook_scale
        return build_expert_candidate_input(
            layer=layer,
            expert=expert,
            source=source,
            gate_up_statistics=gate_up_statistics,
            down_statistics=down_statistics,
            heldout_batches=heldout_batches,
            k5_screen=k5_screen,
            route_power=route_power,
            accounting=accounting,
            transform_seed_sha256=transform_seed_sha256,
            codebook_scale=float(scale),
            searched_transform=searched_transform,
            conditional_down_fit_batches=conditional_down_fit_batches,
            allow_fixed_transform_baseline=self.allow_fixed_transform_baseline,
        )

    def _validate_transform_policy(self, item: ExpertCandidateInput) -> None:
        # Test-only codec fixtures may use compact transform identities; those
        # records are independently barred from production validation.
        if self.attestation["test_only"]:
            return
        identities = [item.fitted[name].transform_identity for name in PROJECTIONS]
        if any(identity.get("schema") != "quant-pipeline.mcg-transform.v1" for identity in identities):
            raise ValueError("competitive production candidates require sealed MCG transform identities")
        statuses = {identity.get("selection_status") for identity in identities}
        if len(statuses) != 1:
            raise ValueError("projection transform selection status differs within an expert")
        status = next(iter(statuses))
        if status == "fixed-reproducible-baseline-not-multidraw-searched":
            if not self.allow_fixed_transform_baseline:
                raise ValueError("fixed Rademacher transform is an explicit baseline, not a production searched transform")
        elif status == "canonical-absolute-v31-gss":
            for projection, identity in zip(PROJECTIONS, identities, strict=True):
                search = identity.get("search_artifact")
                if not isinstance(search, dict) or search.get("selection_status") != status:
                    raise ValueError("transform identity lacks canonical absolute-v31/GSS evidence")
                canonical_hash = _require_hash(
                    search.get("canonical_artifact_content_sha256"),
                    "canonical absolute-v31 artifact",
                )
                if search.get("artifact_sha256") != canonical_hash:
                    raise ValueError("canonical absolute-v31 artifact identity mismatch")
                if set(search.get("bit_gss_vectors", {}).get(projection, {})) != {"3", "4", "5"}:
                    raise ValueError(f"{projection} lacks per-bit v31 GSS vector evidence")
                if item.fitted[projection].bit_vectors is None:
                    raise ValueError(f"{projection} lacks byte-bearing per-bit v31 GSS vectors")
        else:
            raise ValueError("unknown MCG transform selection status")
        shared_pairs = {
            (
                identity.get("shared_gate_up_suh_sha256"),
                identity.get("shared_down_svh_sha256"),
            )
            for identity in identities
        }
        if len(shared_pairs) != 1:
            raise ValueError("shared MCG transform identities differ across projections")
        shared = next(iter(shared_pairs))
        _require_hash(shared[0], "shared gate/up suh")
        _require_hash(shared[1], "shared down svh")
        incumbent = self._layer_shared_transform_hashes.setdefault(item.layer, shared)
        if incumbent != shared:
            raise ValueError("layer-shared MCG vectors differ across experts")

    def generate(
        self,
        experts: Iterable[ExpertCandidateInput],
        *,
        journal: CandidateJournal,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if journal.run_identity["codec_attestation_sha256"] != _hash_json(self.attestation):
            raise ValueError("journal codec attestation identity mismatch")
        inventory = journal.run_identity["expected_inventory"]
        expected_units = [row["unit_id"] for row in inventory["units"]]
        self._payload_store = ExactPayloadStore(journal.root / "payloads")
        records: list[dict[str, Any]] = []
        seen_units: set[str] = set()
        k5_policy: dict[str, dict[str, Any]] = {}
        last_identity: tuple[int, int] | None = None
        expert_count = 0
        for item in experts:
            expert_count += 1
            identity = (item.layer, item.expert)
            if last_identity is not None and identity <= last_identity:
                raise ValueError("streamed experts must be unique and strictly ordered by layer, expert")
            last_identity = identity
            if item.unit_id in seen_units:
                raise ValueError(f"duplicate expert input: {item.unit_id}")
            if expert_count > len(expected_units) or item.unit_id != expected_units[expert_count - 1]:
                raise ValueError("streamed expert coverage differs from sealed expected inventory")
            seen_units.add(item.unit_id)
            _validate_item_run_identity(item, journal.run_identity)
            k5_policy[item.unit_id] = {
                f"g{triplet[0]}u{triplet[1]}d{triplet[2]}": item.k5_screen[triplet].as_dict()
                for triplet in all_k5_triplets()
            }
            records.extend(self._generate_expert(item, journal))
        if expert_count == 0:
            raise ValueError("candidate generation requires at least one expert")
        if expert_count != len(expected_units):
            raise ValueError("streamed expert coverage is incomplete for sealed expected inventory")
        records.sort(key=lambda row: (row["layer"], row["expert"], row["bit_triplet"]))
        journal_index = journal.record_index()
        expected_journal_keys = {
            ("k5-screen", f"{unit}.g{triplet[0]}u{triplet[1]}d{triplet[2]}")
            for unit in expected_units
            for triplet in all_k5_triplets()
        } | {("candidate", row["candidate_id"]) for row in records}
        if {(row["kind"], row["key"]) for row in journal_index} != expected_journal_keys:
            raise ValueError("journal inventory differs from the derived candidate/K5 inventory")
        payload_refs = [
            ref
            for record in records
            for projection in record["projections"].values()
            for ref in projection["exact_payload_refs"].values()
        ]
        payload_manifest = self._payload_store.manifest(payload_refs)
        shared_refs = {
            digest: ref
            for record in records
            for digest in record["physical_payload_accounting"]["layer_shared_object_sha256"]
            for ref in payload_refs
            if ref["sha256"] == digest
        }
        fixed_costs = layer_shared_costs(records)
        ledger = {
            "schema": SCHEMA_LEDGER,
            "competitive": self.competitive,
            "fixed_transform_baseline_allowed": self.allow_fixed_transform_baseline,
            "objective_arm": self.objective_arm,
            "expert_function_compute_dtype": self.expert_compute_dtype,
            "backend_attestation": self.attestation,
            "run_identity": journal.run_identity,
            "journal_identity_sha256": journal.run_identity_sha256,
            "journal_records": journal.inventory(),
            "journal_record_index": journal_index,
            "expected_inventory": inventory,
            "k5_policy": k5_policy,
            "k5_policy_sha256": _hash_json(k5_policy),
            "exact_payload_store": payload_manifest,
            "fixed_layer_shared_payload_bytes": sum(row["semantic_layer_shared_bytes"] for row in fixed_costs),
            "fixed_layer_shared_costs": fixed_costs,
            "fixed_layer_shared_physical_bytes": sum(ref["bytes"] for ref in shared_refs.values()),
            "fixed_layer_shared_object_sha256": sorted(shared_refs),
            "candidates": records,
            "pareto_frontiers": build_pareto_frontiers(
                records,
                competitive=self.competitive,
                allow_test_backend=self.allow_test_backend,
            ),
            "allocator_handoff": [
                {
                    "unit_id": value.unit_id,
                    "choice_id": value.choice_id,
                    "stored_bytes": value.stored_bytes,
                    "predicted_damage": value.predicted_damage,
                    "candidate_record_sha256": value.metadata["record_sha256"],
                }
                for value in allocator_handoff(
                    records,
                    competitive=self.competitive,
                    allow_test_backend=self.allow_test_backend,
                )
            ],
        }
        ledger["ledger_sha256"] = _hash_json(ledger)
        validate_ledger(
            ledger,
            competitive=self.competitive,
            allow_test_backend=self.allow_test_backend,
        )
        if output_path is not None:
            write_json(output_path, ledger)
        return ledger

    def _generate_expert(self, item: ExpertCandidateInput, journal: CandidateJournal) -> list[dict[str, Any]]:
        self._validate_transform_policy(item)
        _validate_expert_input(item, self.competitive)
        base = _input_fingerprint(
            item,
            self.objective_arm,
            self.attestation,
            self.expert_compute_dtype,
        )
        base_sha = _hash_json(base)
        for triplet in all_k5_triplets():
            decision = item.k5_screen[triplet]
            _validate_k5_decision(
                decision,
                expected_selection_artifact_sha256=journal.run_identity["heldout_artifact_sha256"] if decision.admitted else None,
            )
            screen_key = f"{item.unit_id}.g{triplet[0]}u{triplet[1]}d{triplet[2]}"
            screen_input = _hash_json({"base_sha256": base_sha, "triplet": triplet})
            payload = {"unit_id": item.unit_id, "bit_triplet": list(triplet), "decision": decision.as_dict()}
            journal.record("k5-screen", screen_key, screen_input, payload)
        triplets = list(all_k3_k4_triplets()) + [
            triplet for triplet in all_k5_triplets() if item.k5_screen[triplet].admitted
        ]
        records: list[dict[str, Any]] = []
        pending: list[tuple[tuple[int, int, int], str]] = []
        for triplet in triplets:
            key = f"{item.unit_id}.g{triplet[0]}u{triplet[1]}d{triplet[2]}"
            input_sha = _hash_json(
                {
                    "base_sha256": base_sha,
                    "triplet": triplet,
                    "k5_admission": None if 5 not in triplet else item.k5_screen[triplet].as_dict(),
                }
            )
            cached = journal.load("candidate", key, input_sha256=input_sha)
            if cached is not None:
                validate_candidate_record(
                    cached,
                    competitive=self.competitive,
                    allow_test_backend=self.allow_test_backend,
                )
                records.append(cached)
            else:
                pending.append((triplet, input_sha))
        if not pending:
            return records
        encoded = _codec_candidates(
            self.codec,
            item,
            [triplet for triplet, _ in pending],
            {
                "candidate_input_base_sha256": base_sha,
                "backend_attestation_sha256": _hash_json(self.attestation),
                "objective_arm": self.objective_arm,
                "layer": item.layer,
                "expert": item.expert,
                "codec_identity": self.attestation["codec_identity"],
                "codec_identity_sha256": self.attestation["codec_identity_sha256"],
            },
        )
        for triplet, input_sha in pending:
            metrics = _score_triplet(
                item,
                triplet,
                encoded,
                base["heldout_batches"],
                self.expert_compute_dtype,
            )
            record = _candidate_record(
                item,
                triplet,
                encoded,
                metrics,
                self.objective_arm,
                self.attestation,
                input_sha,
                base,
                payload_store=self._payload_store,
                competitive=self.competitive,
                allow_test_backend=self.allow_test_backend,
            )
            journal.record("candidate", record["candidate_id"], input_sha, record)
            records.append(record)
        return records
