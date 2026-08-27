"""Cross-fitted calibration of factory-local proxy deltas to causal KLD deltas."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.artifacts import canonical_json, sha256_bytes


SCHEMA_FACTORY_DELTA_CALIBRATION = "quant-pipeline.candidate-factory-delta-calibration.v1"
SCHEMA_FACTORY_DOMAIN_REANCHOR = "quant-pipeline.candidate-factory-domain-reanchor.v1"


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class FactoryDeltaAnchor:
    """One selection-only direct swap and its factory-blind local proxy."""

    group_id: str
    reference_factory: str
    challenger_factory: str
    proxy_delta: float
    measured_delta_kld: float
    weight: float = 1.0
    evidence_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        if not self.group_id or not self.reference_factory or not self.challenger_factory:
            raise ValueError("factory calibration identities must be non-empty")
        if self.reference_factory == self.challenger_factory:
            raise ValueError("factory calibration requires distinct factories")
        weight = _finite(self.weight, "anchor weight")
        if weight <= 0.0:
            raise ValueError("anchor weight must be positive")
        if self.evidence_sha256 and (
            len(self.evidence_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.evidence_sha256)
        ):
            raise ValueError("anchor evidence identity must be a SHA-256")
        return {
            "group_id": self.group_id,
            "reference_factory": self.reference_factory,
            "challenger_factory": self.challenger_factory,
            "proxy_delta": _finite(self.proxy_delta, "proxy delta"),
            "measured_delta_kld": _finite(self.measured_delta_kld, "measured KLD delta"),
            "weight": weight,
            "evidence_sha256": self.evidence_sha256 or None,
        }


@dataclass(frozen=True)
class FactoryRateProxyEvidence:
    """Source-fixed common-instrument evidence for one reconstructed arm.

    Scores are comparable across factories only when they bind the same source
    tensor, corpus role, scoring instrument, and calibration artifact.  Those
    identities are explicit because factory-local receipt scores are not a
    valid common objective.
    """

    unit_id: str
    coupling_group_id: str
    factory_name: str
    compatibility_domain_sha256: str
    rate: int
    reconstruction_sha256: str
    source_tensor_sha256: str
    source_shape: tuple[int, ...]
    scoring_instrument_sha256: str
    calibration_artifact_sha256: str
    corpus_role: str
    proxy_damage: float
    causal_scale: float
    evidence_sha256: str

    def as_dict(self) -> dict[str, Any]:
        if not self.unit_id or not self.coupling_group_id or not self.factory_name:
            raise ValueError("factory-rate evidence identities must be non-empty")
        if isinstance(self.rate, bool) or not isinstance(self.rate, int) or self.rate <= 0:
            raise ValueError("factory-rate evidence rate must be a positive integer")
        for value, label in (
            (self.compatibility_domain_sha256, "compatibility domain"),
            (self.reconstruction_sha256, "reconstruction identity"),
            (self.source_tensor_sha256, "source tensor identity"),
            (self.scoring_instrument_sha256, "scoring instrument"),
            (self.calibration_artifact_sha256, "calibration artifact"),
            (self.evidence_sha256, "factory-rate evidence"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{label} must be a SHA-256")
        proxy = _finite(self.proxy_damage, "proxy damage")
        scale = _finite(self.causal_scale, "causal scale")
        if not self.source_shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.source_shape
        ):
            raise ValueError("source shape must contain positive integers")
        if not self.corpus_role or self.corpus_role == "final":
            raise ValueError("factory-rate evidence must name a non-final corpus role")
        return {
            "unit_id": self.unit_id,
            "coupling_group_id": self.coupling_group_id,
            "factory_name": self.factory_name,
            "compatibility_domain_sha256": self.compatibility_domain_sha256,
            "rate": self.rate,
            "reconstruction_sha256": self.reconstruction_sha256,
            "source_tensor_sha256": self.source_tensor_sha256,
            "source_shape": list(self.source_shape),
            "scoring_instrument_sha256": self.scoring_instrument_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "corpus_role": self.corpus_role,
            "proxy_damage": proxy,
            "causal_scale": scale,
            "scaled_proxy_damage": proxy * scale,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class FactoryDomainAnchor:
    """Observed selection-fold damage for one frozen layer/factory profile."""

    coupling_group_id: str
    factory_name: str
    compatibility_domain_sha256: str
    anchor_rates: Mapping[str, int]
    observed_profile_delta_kld: float
    confidence_low: float
    confidence_high: float
    evidence_sha256: str

    def as_dict(self) -> dict[str, Any]:
        if not self.coupling_group_id or not self.factory_name or not self.anchor_rates:
            raise ValueError("factory-domain anchor identities and rates must be non-empty")
        for value, label in (
            (self.compatibility_domain_sha256, "compatibility domain"),
            (self.evidence_sha256, "factory-domain anchor evidence"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{label} must be a SHA-256")
        rates = {}
        for unit_id, rate in sorted(self.anchor_rates.items()):
            if not unit_id or isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
                raise ValueError("factory-domain anchor rates must map units to positive integers")
            rates[str(unit_id)] = rate
        observed = _finite(self.observed_profile_delta_kld, "observed profile KLD delta")
        low = _finite(self.confidence_low, "profile KLD confidence low")
        high = _finite(self.confidence_high, "profile KLD confidence high")
        if low > observed or observed > high:
            raise ValueError("factory-domain anchor confidence interval does not contain its estimate")
        return {
            "coupling_group_id": self.coupling_group_id,
            "factory_name": self.factory_name,
            "compatibility_domain_sha256": self.compatibility_domain_sha256,
            "anchor_rates": rates,
            "observed_profile_delta_kld": observed,
            "confidence_low": low,
            "confidence_high": high,
            "evidence_sha256": self.evidence_sha256,
        }


def build_factory_domain_reanchor(
    rate_evidence: Iterable[FactoryRateProxyEvidence],
    anchors: Iterable[FactoryDomainAnchor],
    *,
    required_rates: Sequence[int] = (3, 4),
) -> dict[str, Any]:
    """Re-anchor every factory/rate arm to an observed frozen layer swap.

    The per-unit term is the same factory-blind causal proxy multiplied by the
    native Aumann--Shapley/Fisher scale.  A single group/domain correction then
    makes the *frozen profile* sum exactly to its measured selection-fold KLD
    delta.  The coupled allocator adds that correction once and is therefore
    free to choose both factory domain and K3/K4 profile under one byte budget.
    """

    rates = tuple(sorted(set(required_rates)))
    if not rates or any(isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0 for rate in rates):
        raise ValueError("required factory rates must be positive integers")
    evidence_rows = [row.as_dict() for row in rate_evidence]
    anchor_rows = [row.as_dict() for row in anchors]
    if not evidence_rows or not anchor_rows:
        raise ValueError("factory-domain re-anchoring requires rate evidence and anchors")

    evidence_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    units_by_group: dict[str, set[str]] = {}
    scale_by_unit: dict[str, float] = {}
    source_by_unit: dict[str, tuple[str, tuple[int, ...]]] = {}
    domain_by_factory_group: dict[tuple[str, str], str] = {}
    scoring_instruments: set[str] = set()
    calibration_artifacts: set[str] = set()
    corpus_roles: set[str] = set()
    for row in evidence_rows:
        key = (
            row["unit_id"], row["factory_name"],
            row["compatibility_domain_sha256"], int(row["rate"]),
        )
        if key in evidence_by_key:
            raise ValueError("duplicate factory/unit/rate proxy evidence")
        evidence_by_key[key] = row
        units_by_group.setdefault(row["coupling_group_id"], set()).add(row["unit_id"])
        incumbent_scale = scale_by_unit.setdefault(row["unit_id"], float(row["causal_scale"]))
        if not math.isclose(incumbent_scale, float(row["causal_scale"]), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("causal scale differs across factory/rate candidates for one unit")
        source_identity = (
            str(row["source_tensor_sha256"]),
            tuple(int(value) for value in row["source_shape"]),
        )
        incumbent_source = source_by_unit.setdefault(row["unit_id"], source_identity)
        if incumbent_source != source_identity:
            raise ValueError("source tensor identity differs across factory/rate candidates")
        scoring_instruments.add(str(row["scoring_instrument_sha256"]))
        calibration_artifacts.add(str(row["calibration_artifact_sha256"]))
        corpus_roles.add(str(row["corpus_role"]))
        factory_group = (row["coupling_group_id"], row["factory_name"])
        incumbent_domain = domain_by_factory_group.setdefault(
            factory_group, row["compatibility_domain_sha256"]
        )
        if incumbent_domain != row["compatibility_domain_sha256"]:
            raise ValueError("one factory exposes multiple compatibility domains in a coupling group")
    if len(scoring_instruments) != 1:
        raise ValueError("factory candidates were not scored by one common instrument")
    if len(calibration_artifacts) != 1:
        raise ValueError("factory candidates bind different calibration artifacts")
    if len(corpus_roles) != 1:
        raise ValueError("factory candidates bind different corpus roles")

    anchors_by_domain: dict[tuple[str, str], dict[str, Any]] = {}
    domain_rows = []
    for anchor in anchor_rows:
        group = anchor["coupling_group_id"]
        domain = anchor["compatibility_domain_sha256"]
        key = (group, domain)
        if key in anchors_by_domain:
            raise ValueError("duplicate factory-domain anchor")
        if domain_by_factory_group.get((group, anchor["factory_name"])) != domain:
            raise ValueError("factory-domain anchor has no matching rate evidence")
        expected_units = units_by_group.get(group, set())
        if set(anchor["anchor_rates"]) != expected_units:
            raise ValueError("factory-domain anchor profile does not cover its complete coupling group")
        anchor_terms = []
        for unit_id, rate in sorted(anchor["anchor_rates"].items()):
            row = evidence_by_key.get((unit_id, anchor["factory_name"], domain, int(rate)))
            if row is None:
                raise ValueError("factory-domain anchor selects an absent unit/rate candidate")
            anchor_terms.append(float(row["scaled_proxy_damage"]))
        proxy_total = math.fsum(anchor_terms)
        target = float(anchor["observed_profile_delta_kld"])
        correction = target - proxy_total
        reconstructed = math.fsum((proxy_total, correction))
        if not math.isclose(reconstructed, target, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("factory-domain re-anchor failed exact numerical closure")
        domain_row = dict(anchor) | {
            "anchor_scaled_proxy_damage": proxy_total,
            "domain_fixed_damage": correction,
            "reconstructed_anchor_damage": reconstructed,
            "closure_error": reconstructed - target,
            "uncertainty_half_width": max(
                target - float(anchor["confidence_low"]),
                float(anchor["confidence_high"]) - target,
            ),
        }
        anchors_by_domain[key] = domain_row
        domain_rows.append(domain_row)

    candidate_rows = []
    coverage: dict[tuple[str, str, str], set[int]] = {}
    for row in sorted(
        evidence_rows,
        key=lambda item: (
            item["coupling_group_id"], item["factory_name"],
            item["unit_id"], int(item["rate"]),
        ),
    ):
        domain_key = (row["coupling_group_id"], row["compatibility_domain_sha256"])
        domain = anchors_by_domain.get(domain_key)
        if domain is None:
            raise ValueError("factory/rate evidence has no observed domain anchor")
        coverage.setdefault(
            (row["unit_id"], row["factory_name"], row["compatibility_domain_sha256"]),
            set(),
        ).add(int(row["rate"]))
        candidate_rows.append(dict(row) | {
            "calibrated_unit_damage": float(row["scaled_proxy_damage"]),
            "domain_fixed_damage": float(domain["domain_fixed_damage"]),
            "domain_anchor_evidence_sha256": domain["evidence_sha256"],
            "domain_uncertainty": float(domain["uncertainty_half_width"]),
        })
    incomplete = sorted(key for key, observed in coverage.items() if observed != set(rates))
    if incomplete:
        raise ValueError(
            "factory-domain re-anchor lacks complete requested-rate evidence for "
            f"{len(incomplete)} unit/factory domains"
        )

    body = {
        "schema": SCHEMA_FACTORY_DOMAIN_REANCHOR,
        "method": "selection-swap-anchor-plus-native-causal-proxy-delta-v1",
        "scoring_instrument_sha256": next(iter(scoring_instruments)),
        "calibration_artifact_sha256": next(iter(calibration_artifacts)),
        "scoring_corpus_role": next(iter(corpus_roles)),
        "selection_only": True,
        "final_evaluation_rows_consumed": False,
        "required_rates": list(rates),
        "candidate_scores": candidate_rows,
        "domain_anchors": sorted(
            domain_rows,
            key=lambda row: (row["coupling_group_id"], row["factory_name"]),
        ),
        "closure": {
            "domain_count": len(domain_rows),
            "max_abs_anchor_closure_error": max(abs(float(row["closure_error"])) for row in domain_rows),
        },
    }
    body["calibration_sha256"] = _hash_json(body)
    return body


class ReanchoredFactoryScorer:
    """Factory-blind scorer backed by a sealed domain-reanchor artifact."""

    def __init__(self, calibration: Mapping[str, Any]) -> None:
        body = dict(calibration)
        seal = body.pop("calibration_sha256", None)
        if body.get("schema") != SCHEMA_FACTORY_DOMAIN_REANCHOR or seal != _hash_json(body):
            raise ValueError("factory-domain re-anchor seal mismatch")
        self.calibration = dict(calibration)
        self.calibration_sha256 = str(seal)
        self.instrument_sha256 = str(body["scoring_instrument_sha256"])
        self._rows: dict[tuple[str, str, str, int, str], Mapping[str, Any]] = {}
        for row in body["candidate_scores"]:
            key = (
                str(row["unit_id"]),
                str(row["factory_name"]),
                str(row["compatibility_domain_sha256"]),
                int(row["rate"]),
                str(row["reconstruction_sha256"]),
            )
            if key in self._rows:
                raise ValueError("factory-domain re-anchor contains duplicate score identities")
            self._rows[key] = row

    def score(self, unit: Any, proposal: Any) -> Any:
        # Local import avoids coupling the calibration dataclasses to the
        # candidate-factory protocol at module import time.
        from .factory_union import CommonCandidateScore

        if proposal.unit_id != unit.unit_id:
            raise ValueError("factory-domain scorer received a proposal for the wrong unit")
        key = (
            str(proposal.unit_id),
            str(proposal.factory_name),
            str(proposal.compatibility_domain_sha256),
            int(proposal.rate),
            str(proposal.reconstruction_sha256),
        )
        row = self._rows.get(key)
        if row is None:
            raise ValueError("factory proposal lacks sealed comparable rate evidence")
        if str(row["coupling_group_id"]) != str(unit.coupling_group_id):
            raise ValueError("factory proposal coupling group differs from re-anchor evidence")
        if str(row["source_tensor_sha256"]) != str(unit.source_sha256):
            raise ValueError("factory proposal source tensor differs from re-anchor evidence")
        if tuple(int(value) for value in row["source_shape"]) != tuple(unit.source_shape):
            raise ValueError("factory proposal source shape differs from re-anchor evidence")
        if str(row["calibration_artifact_sha256"]) != str(
            unit.calibration_identity_sha256
        ):
            raise ValueError("factory proposal calibration differs from candidate unit")
        return CommonCandidateScore(
            raw_damage=float(row["proxy_damage"]),
            calibrated_damage=float(row["calibrated_unit_damage"]),
            uncertainty=0.0,
            instrument_sha256=self.instrument_sha256,
            calibration_fit_sha256=self.calibration_sha256,
            metadata={
                "basis": "selection-swap-anchor-plus-native-causal-proxy-delta-v1",
                "rate_evidence_sha256": row["evidence_sha256"],
                "domain_anchor_evidence_sha256": row["domain_anchor_evidence_sha256"],
                "domain_uncertainty": row["domain_uncertainty"],
                "causal_scale": row["causal_scale"],
            },
            domain_fixed_damage=float(row["domain_fixed_damage"]),
        )


def _fit_identity_residual(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    ridge: float,
) -> tuple[float, float]:
    """Fit y = x + intercept + slope_adjustment*x, shrunk to y=x."""

    design = np.stack((np.ones_like(x), x), axis=1)
    gram = design.T @ (weight[:, None] * design)
    rhs = design.T @ (weight * (y - x))
    penalty = np.eye(2, dtype=np.float64) * ridge
    # The intercept is expressed in the observed KLD scale. Penalize it after
    # normalizing by the proxy RMS so ridge strength is unit-stable.
    scale = max(float(np.sqrt(np.average(x * x, weights=weight))), 1e-12)
    penalty[0, 0] /= scale * scale
    theta = np.linalg.solve(gram + penalty, rhs)
    return float(theta[0]), float(1.0 + theta[1])


def _predict(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    return intercept + slope * x


def fit_factory_delta_calibration(
    anchors: Iterable[FactoryDeltaAnchor],
    *,
    ridge_grid: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> dict[str, Any]:
    """Fit one cross-validated shrinkage map for each factory comparison.

    The target is a direct selection-only end-to-end KLD swap.  The predictor
    is the difference between factory-blind common local scores.  Ridge shrinks
    toward the identity map rather than toward either factory.  Leave-one-group
    out predictions select ridge strength and are retained as evidence; final
    evaluation rows must remain outside this function.
    """

    rows = [anchor.as_dict() for anchor in anchors]
    if not rows:
        raise ValueError("factory calibration requires anchors")
    ridge_values = sorted({_finite(value, "ridge value") for value in ridge_grid})
    if not ridge_values or ridge_values[0] < 0.0:
        raise ValueError("ridge grid must contain nonnegative values")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["reference_factory"], row["challenger_factory"]), []
        ).append(row)

    fits = []
    for (reference, challenger), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["group_id"])
        if len(values) < 4 or len({row["group_id"] for row in values}) != len(values):
            raise ValueError("each factory comparison needs at least four unique groups")
        x = np.asarray([row["proxy_delta"] for row in values], dtype=np.float64)
        y = np.asarray([row["measured_delta_kld"] for row in values], dtype=np.float64)
        weight = np.asarray([row["weight"] for row in values], dtype=np.float64)
        candidates = []
        for ridge in ridge_values:
            predictions = np.empty_like(y)
            for index in range(len(values)):
                keep = np.arange(len(values)) != index
                intercept, slope = _fit_identity_residual(
                    x[keep], y[keep], weight[keep], ridge
                )
                predictions[index] = intercept + slope * x[index]
            mse = float(np.average((predictions - y) ** 2, weights=weight))
            candidates.append({
                "ridge": ridge,
                "weighted_oof_mse": mse,
                "oof_predictions": predictions.tolist(),
            })
        # Prefer stronger shrinkage on an exact CV tie.
        chosen = min(candidates, key=lambda row: (row["weighted_oof_mse"], -row["ridge"]))
        intercept, slope = _fit_identity_residual(
            x, y, weight, float(chosen["ridge"])
        )
        oof = np.asarray(chosen["oof_predictions"], dtype=np.float64)
        effective = max(float(weight.sum()) - 2.0, 1.0)
        residual_sigma = float(np.sqrt(np.sum(weight * (oof - y) ** 2) / effective))
        fits.append({
            "reference_factory": reference,
            "challenger_factory": challenger,
            "anchor_count": len(values),
            "anchor_group_ids": [row["group_id"] for row in values],
            "selected_ridge": float(chosen["ridge"]),
            "intercept": intercept,
            "slope": slope,
            "oof_weighted_mse": float(chosen["weighted_oof_mse"]),
            "oof_residual_sigma": residual_sigma,
            "identity_oof_weighted_mse": float(np.average((x - y) ** 2, weights=weight)),
            "ridge_search": candidates,
            "anchors": values,
        })
    body = {
        "schema": SCHEMA_FACTORY_DELTA_CALIBRATION,
        "method": "leave-one-group-out-ridge-to-identity-v1",
        "selection_only": True,
        "final_evaluation_rows_consumed": False,
        "fits": fits,
    }
    body["calibration_sha256"] = _hash_json(body)
    return body


def apply_factory_delta_calibration(
    calibration: Mapping[str, Any],
    *,
    reference_factory: str,
    challenger_factory: str,
    proxy_delta: float,
) -> tuple[float, float]:
    """Return calibrated challenger-minus-reference damage and uncertainty."""

    body = dict(calibration)
    seal = body.pop("calibration_sha256", None)
    if seal != _hash_json(body):
        raise ValueError("factory delta calibration seal mismatch")
    matches = [
        row
        for row in calibration["fits"]
        if row["reference_factory"] == reference_factory
        and row["challenger_factory"] == challenger_factory
    ]
    if len(matches) != 1:
        raise ValueError("factory comparison calibration is absent or ambiguous")
    row = matches[0]
    value = float(row["intercept"] + row["slope"] * _finite(proxy_delta, "proxy delta"))
    return value, float(row["oof_residual_sigma"])
