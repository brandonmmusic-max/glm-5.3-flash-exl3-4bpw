"""Model-agnostic union of independently attested candidate factories.

Factories may propose candidates, but they never select or self-score them.
Every proposal is evaluated by one caller-supplied common instrument and then
handed to the existing exact-byte allocator.  A native factory can be marked
required; optional upstream factories may be absent or support only a subset
of units without making the pipeline model-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ..allocation.global_dp import Candidate
from ..core.artifacts import canonical_json, sha256_bytes


SCHEMA_FACTORY_IDENTITY = "quant-pipeline.candidate-factory-identity.v1"
SCHEMA_FACTORY_UNIT = "quant-pipeline.candidate-factory-unit.v2"
SCHEMA_FACTORY_PROPOSAL = "quant-pipeline.candidate-factory-proposal.v2"
SCHEMA_COMMON_SCORE = "quant-pipeline.candidate-factory-common-score.v2"
SCHEMA_FACTORY_UNION = "quant-pipeline.candidate-factory-union-ledger.v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class FactoryIdentity:
    name: str
    version: str
    implementation_sha256: str
    encoding_family: str
    runtime_format_sha256: str
    provenance: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        if not self.name or not self.version or not self.encoding_family:
            raise ValueError("factory name, version, and encoding family must be non-empty")
        body = {
            "schema": SCHEMA_FACTORY_IDENTITY,
            "name": self.name,
            "version": self.version,
            "implementation_sha256": _hash(
                self.implementation_sha256, "factory implementation identity"
            ),
            "encoding_family": self.encoding_family,
            "runtime_format_sha256": _hash(
                self.runtime_format_sha256, "factory runtime format identity"
            ),
            "provenance": dict(self.provenance),
        }
        body["identity_sha256"] = _hash_json(body)
        return body


@dataclass(frozen=True)
class CandidateUnit:
    """One independently allocatable weight unit, dense or routed."""

    unit_id: str
    coupling_group_id: str
    source_sha256: str
    source_shape: tuple[int, ...]
    requested_rates: tuple[int, ...]
    calibration_identity_sha256: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        if not self.unit_id or not self.coupling_group_id:
            raise ValueError("candidate unit_id and coupling_group_id must be non-empty")
        if not self.source_shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.source_shape
        ):
            raise ValueError("source shape must contain positive integers")
        if not self.requested_rates or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.requested_rates
        ):
            raise ValueError("requested rates must contain positive integers")
        if len(set(self.requested_rates)) != len(self.requested_rates):
            raise ValueError("requested rates must be unique")
        body = {
            "schema": SCHEMA_FACTORY_UNIT,
            "unit_id": self.unit_id,
            "coupling_group_id": self.coupling_group_id,
            "source_sha256": _hash(self.source_sha256, "source identity"),
            "source_shape": list(self.source_shape),
            "requested_rates": sorted(self.requested_rates),
            "calibration_identity_sha256": _hash(
                self.calibration_identity_sha256, "calibration identity"
            ),
            "metadata": dict(self.metadata),
        }
        body["unit_sha256"] = _hash_json(body)
        return body


@dataclass(frozen=True)
class FactoryProposal:
    """One private payload proposal plus its shared compatibility contract.

    ``exact_stored_bytes`` counts only the unit-private payload.
    ``fixed_shared_bytes`` is the complete shared payload cost for the named
    coupling group/domain and is charged once by the coupled allocator.
    """

    unit_id: str
    rate: int
    factory_name: str
    factory_identity_sha256: str
    packed_sha256: str
    reconstruction_sha256: str
    exact_stored_bytes: int
    payload_ref: str
    reconstruction_ref: str
    compatibility_domain_sha256: str
    fixed_shared_bytes: int
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        if not self.unit_id or not self.factory_name:
            raise ValueError("proposal unit and factory names must be non-empty")
        if isinstance(self.rate, bool) or not isinstance(self.rate, int) or self.rate <= 0:
            raise ValueError("proposal rate must be a positive integer")
        if (
            isinstance(self.exact_stored_bytes, bool)
            or not isinstance(self.exact_stored_bytes, int)
            or self.exact_stored_bytes <= 0
        ):
            raise ValueError("proposal exact byte cost must be positive")
        if not self.payload_ref or not self.reconstruction_ref:
            raise ValueError("proposal payload references must be non-empty")
        if (
            isinstance(self.fixed_shared_bytes, bool)
            or not isinstance(self.fixed_shared_bytes, int)
            or self.fixed_shared_bytes < 0
        ):
            raise ValueError("proposal fixed shared byte cost must be nonnegative")
        body = {
            "schema": SCHEMA_FACTORY_PROPOSAL,
            "unit_id": self.unit_id,
            "rate": self.rate,
            "factory_name": self.factory_name,
            "factory_identity_sha256": _hash(
                self.factory_identity_sha256, "proposal factory identity"
            ),
            "packed_sha256": _hash(self.packed_sha256, "packed payload identity"),
            "reconstruction_sha256": _hash(
                self.reconstruction_sha256, "reconstruction identity"
            ),
            "exact_stored_bytes": self.exact_stored_bytes,
            "payload_ref": self.payload_ref,
            "reconstruction_ref": self.reconstruction_ref,
            "compatibility_domain_sha256": _hash(
                self.compatibility_domain_sha256,
                "proposal compatibility domain",
            ),
            "fixed_shared_bytes": self.fixed_shared_bytes,
            # Factory-local losses are provenance only.  The allocator never
            # reads metadata when choosing a candidate.
            "metadata": dict(self.metadata),
        }
        body["proposal_sha256"] = _hash_json(body)
        return body


@dataclass(frozen=True)
class CommonCandidateScore:
    raw_damage: float
    calibrated_damage: float
    uncertainty: float
    instrument_sha256: str
    calibration_fit_sha256: str
    metadata: Mapping[str, Any]
    domain_fixed_damage: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA_COMMON_SCORE,
            # Causal/Shapley contributions and factory-swap deltas may be
            # signed.  The exact allocator only needs finite comparable
            # objectives; forbidding negative values would bias away from
            # genuinely beneficial substitutions.
            "raw_damage": float(self.raw_damage),
            "calibrated_damage": float(self.calibrated_damage),
            "uncertainty": _finite_nonnegative(self.uncertainty, "score uncertainty"),
            "domain_fixed_damage": float(self.domain_fixed_damage),
            "instrument_sha256": _hash(self.instrument_sha256, "common instrument"),
            "calibration_fit_sha256": _hash(
                self.calibration_fit_sha256, "score calibration fit"
            ),
            "metadata": dict(self.metadata),
        }
        for key in ("raw_damage", "calibrated_damage", "domain_fixed_damage"):
            if not math.isfinite(body[key]):
                raise ValueError(f"{key.replace('_', ' ')} must be finite")
        body["score_sha256"] = _hash_json(body)
        return body


class CandidateFactory(Protocol):
    """Optional proposer.  It may decline unsupported units or rates."""

    @property
    def identity(self) -> FactoryIdentity: ...

    def propose(self, unit: CandidateUnit) -> Iterable[FactoryProposal]: ...


class CommonCandidateScorer(Protocol):
    """Factory-blind scorer owned by the ShapleyMCG pipeline."""

    def score(
        self, unit: CandidateUnit, proposal: FactoryProposal
    ) -> CommonCandidateScore: ...


@dataclass(frozen=True)
class FactoryUnion:
    ledger: Mapping[str, Any]
    allocator_candidates: tuple[Candidate, ...]


def build_factory_union(
    units: Sequence[CandidateUnit],
    factories: Sequence[CandidateFactory],
    scorer: CommonCandidateScorer,
    *,
    required_factory_names: Sequence[str],
) -> FactoryUnion:
    """Generate, common-score, seal, and hand off a factory candidate union.

    Required factories must cover every requested unit/rate pair.  Optional
    factories are additive and may be absent.  Consequently a native required
    factory keeps new-model quantization fully functional without any upstream
    checkpoint or model-specific adapter.
    """

    if not units or not factories:
        raise ValueError("factory union requires units and factories")
    unit_rows = [unit.as_dict() for unit in units]
    if len({row["unit_id"] for row in unit_rows}) != len(unit_rows):
        raise ValueError("candidate factory units must be unique")
    identities = [factory.identity.as_dict() for factory in factories]
    by_name = {row["name"]: row for row in identities}
    if len(by_name) != len(identities):
        raise ValueError("candidate factory names must be unique")
    required = sorted(set(required_factory_names))
    if not required or any(name not in by_name for name in required):
        raise ValueError("every required factory must be registered")
    required_formats = {by_name[name]["runtime_format_sha256"] for name in required}
    if len(required_formats) != 1:
        raise ValueError("required factories disagree on runtime payload format")
    runtime_format = next(iter(required_formats))
    incompatible = [
        row["name"]
        for row in identities
        if row["runtime_format_sha256"] != runtime_format
    ]
    if incompatible:
        raise ValueError(
            "candidate factories are not directly co-emittable in the required runtime format: "
            + ", ".join(sorted(incompatible))
        )

    records: list[dict[str, Any]] = []
    handoff: list[Candidate] = []
    coverage: dict[str, set[tuple[str, int]]] = {name: set() for name in by_name}
    seen: set[tuple[str, int, str, str]] = set()
    for unit, unit_row in zip(units, unit_rows, strict=True):
        requested = set(unit.requested_rates)
        for factory, identity in zip(factories, identities, strict=True):
            for proposal in factory.propose(unit):
                proposal_row = proposal.as_dict()
                if proposal.unit_id != unit.unit_id:
                    raise ValueError("factory proposal belongs to the wrong unit")
                if proposal.factory_name != identity["name"]:
                    raise ValueError("factory proposal name differs from its adapter")
                if proposal.factory_identity_sha256 != identity["identity_sha256"]:
                    raise ValueError("factory proposal identity differs from its adapter")
                if proposal.rate not in requested:
                    raise ValueError("factory proposed an unrequested rate")
                key = (
                    proposal.unit_id,
                    proposal.rate,
                    proposal.factory_name,
                    proposal.reconstruction_sha256,
                )
                if key in seen:
                    raise ValueError("duplicate factory proposal")
                seen.add(key)
                score = scorer.score(unit, proposal).as_dict()
                choice_id = (
                    f"{unit.unit_id}.{proposal.factory_name}.K{proposal.rate}."
                    f"{proposal.reconstruction_sha256[:16]}"
                )
                record = {
                    "unit": unit_row,
                    "factory": identity,
                    "proposal": proposal_row,
                    "common_score": score,
                    "choice_id": choice_id,
                }
                record["record_sha256"] = _hash_json(record)
                records.append(record)
                coverage[proposal.factory_name].add((proposal.unit_id, proposal.rate))
                handoff.append(
                    Candidate(
                        unit_id=unit.unit_id,
                        choice_id=choice_id,
                        stored_bytes=proposal.exact_stored_bytes,
                        predicted_damage=float(score["calibrated_damage"]),
                        metadata={
                            "factory_name": proposal.factory_name,
                            "rate": proposal.rate,
                            "coupling_group_id": unit.coupling_group_id,
                            "compatibility_domain_sha256": proposal.compatibility_domain_sha256,
                            "fixed_shared_bytes": proposal.fixed_shared_bytes,
                            "domain_fixed_damage": score["domain_fixed_damage"],
                            "proposal_sha256": proposal_row["proposal_sha256"],
                            "common_score_sha256": score["score_sha256"],
                            "record_sha256": record["record_sha256"],
                        },
                    )
                )

    expected = {
        (unit.unit_id, rate)
        for unit in units
        for rate in unit.requested_rates
    }
    for name in required:
        missing = expected - coverage[name]
        if missing:
            raise ValueError(f"required factory {name} lacks {len(missing)} unit/rate proposals")
    if {candidate.unit_id for candidate in handoff} != {unit.unit_id for unit in units}:
        raise ValueError("factory union produced incomplete allocator coverage")

    records.sort(
        key=lambda row: (
            row["proposal"]["unit_id"],
            row["proposal"]["rate"],
            row["proposal"]["factory_name"],
            row["proposal"]["reconstruction_sha256"],
        )
    )
    handoff.sort(key=lambda row: (row.unit_id, row.stored_bytes, row.choice_id))
    domains_by_group: dict[str, set[str]] = {}
    for candidate in handoff:
        metadata = candidate.metadata or {}
        domains_by_group.setdefault(
            str(metadata["coupling_group_id"]), set()
        ).add(str(metadata["compatibility_domain_sha256"]))
    coupled_allocation_required = any(
        len(domains) > 1 for domains in domains_by_group.values()
    )
    ledger = {
        "schema": SCHEMA_FACTORY_UNION,
        "required_factory_names": required,
        "runtime_format_sha256": runtime_format,
        "factory_identities": sorted(identities, key=lambda row: row["name"]),
        "units": sorted(unit_rows, key=lambda row: row["unit_id"]),
        "coverage": {
            name: [
                {"unit_id": unit_id, "rate": rate}
                for unit_id, rate in sorted(values)
            ]
            for name, values in sorted(coverage.items())
        },
        "records": records,
        "selection_authority": "common ShapleyMCG scorer plus exact-byte allocator",
        "factory_metadata_is_nonobjective": True,
        "allocation_contract": "one complete shared-transform domain per coupling group",
        "coupled_allocation_required": coupled_allocation_required,
        "compatibility_domains_by_group": {
            group: sorted(domains) for group, domains in sorted(domains_by_group.items())
        },
    }
    ledger["ledger_sha256"] = _hash_json(ledger)
    return FactoryUnion(ledger=ledger, allocator_candidates=tuple(handoff))
