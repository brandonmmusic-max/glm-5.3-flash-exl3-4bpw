"""Exact per-layer mass x sensitivity mixed-bit allocation."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Iterable, Mapping

from .constants import (
    ALLOWED_BITS,
    FLOOR_BITS,
    NUM_EXPERTS,
    PROJECTIONS,
    TARGET_BIT_UNITS_PER_LAYER,
    TENSORS_PER_LAYER,
    UPGRADE_UNITS_PER_LAYER,
    TensorId,
    all_layer_tensor_ids,
)
from .determinism import atomic_write_json, canonical_json_bytes, sha256_bytes
from .types import CandidateLoss, LayerAllocation

SCORE_SCALE = 10**15
_NEG_INF: int | None = None


@dataclass(frozen=True)
class SensitivityCurve:
    tensor_id: TensorId
    mass: Decimal
    loss_by_bits: Mapping[int, Decimal]
    evidence_hash_by_bits: Mapping[int, str]

    def gain(self, bits: int) -> Decimal:
        return self.mass * (self.loss_by_bits[FLOOR_BITS] - self.loss_by_bits[bits])


def _decimal(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal {label}={value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"non-finite decimal {label}={value!r}")
    return parsed


def build_curves(
    layer: int,
    candidates: Iterable[CandidateLoss],
    mass_by_expert: Iterable[float | str | Decimal],
) -> tuple[SensitivityCurve, ...]:
    masses = tuple(
        _decimal(
            format(value, ".17g") if isinstance(value, float) else str(value), "mass"
        )
        for value in mass_by_expert
    )
    if len(masses) != NUM_EXPERTS or any(value < 0 for value in masses):
        raise ValueError(f"expected {NUM_EXPERTS} nonnegative expert masses")

    grouped: dict[str, dict[int, CandidateLoss]] = {}
    for candidate in candidates:
        tensor_id = candidate.tensor_id
        if tensor_id.layer != layer:
            raise ValueError(f"foreign layer candidate {tensor_id.key}")
        if candidate.bits not in ALLOWED_BITS:
            raise ValueError(f"unsupported candidate bits {candidate.bits}")
        by_bits = grouped.setdefault(tensor_id.key, {})
        if candidate.bits in by_bits:
            raise ValueError(f"duplicate candidate {tensor_id.key}@{candidate.bits}")
        by_bits[candidate.bits] = candidate

    curves: list[SensitivityCurve] = []
    for tensor_id in all_layer_tensor_ids(layer):
        by_bits = grouped.get(tensor_id.key)
        if by_bits is None or set(by_bits) != set(ALLOWED_BITS):
            present = sorted(by_bits or {})
            raise ValueError(
                f"incomplete probe for {tensor_id.key}: {present}, need {ALLOWED_BITS}"
            )
        losses = {bits: _decimal(by_bits[bits].loss, "loss") for bits in ALLOWED_BITS}
        if any(loss < 0 for loss in losses.values()):
            raise ValueError(f"negative probe loss for {tensor_id.key}")
        declared = {
            _decimal(by_bits[bits].mass, "candidate.mass") for bits in ALLOWED_BITS
        }
        if len(declared) != 1 or declared.pop() != masses[tensor_id.expert]:
            raise ValueError(f"probe mass drift for {tensor_id.key}")
        if any(not by_bits[bits].roundtrip_sha256 for bits in ALLOWED_BITS):
            raise ValueError(f"missing round-trip hash for {tensor_id.key}")
        if tensor_id.projection == "down_proj" and any(
            not by_bits[bits].gate_up_roundtrip_sha256 for bits in ALLOWED_BITS
        ):
            raise ValueError(f"down probe lacks gate/up provenance for {tensor_id.key}")
        curves.append(
            SensitivityCurve(
                tensor_id=tensor_id,
                mass=masses[tensor_id.expert],
                loss_by_bits=losses,
                evidence_hash_by_bits={
                    bits: by_bits[bits].roundtrip_sha256 for bits in ALLOWED_BITS
                },
            )
        )
    if len(grouped) != TENSORS_PER_LAYER:
        extras = sorted(set(grouped) - {curve.tensor_id.key for curve in curves})
        raise ValueError(f"probe contains unexpected tensor keys: {extras[:3]}")
    return tuple(curves)


def _gain_integer(curve: SensitivityCurve, bits: int) -> int:
    with localcontext() as context:
        context.prec = 50
        scaled = curve.gain(bits) * Decimal(SCORE_SCALE)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def solve_exact_allocation(
    layer: int,
    curves: Iterable[SensitivityCurve],
    *,
    fixed_point_iteration: int,
    probe_sha256: str,
) -> LayerAllocation:
    expected = all_layer_tensor_ids(layer)
    by_id = {curve.tensor_id: curve for curve in curves}
    if len(by_id) != len(expected) or set(by_id) != set(expected):
        raise ValueError("allocation curves are not exactly the layer tensor set")
    ordered = tuple(by_id[tensor_id] for tensor_id in expected)

    budget = UPGRADE_UNITS_PER_LAYER
    scores: list[int | None] = [_NEG_INF] * (budget + 1)
    scores[0] = 0
    parent_costs: list[array] = []
    parent_bits: list[bytearray] = []

    for curve in ordered:
        next_scores: list[int | None] = [_NEG_INF] * (budget + 1)
        costs = array("h", [-1]) * (budget + 1)
        choices = bytearray(budget + 1)
        # Low bits first plus replace-only-on-strict-gain is the tie rule.
        for prior_cost, prior_score in enumerate(scores):
            if prior_score is None:
                continue
            for bits in ALLOWED_BITS:
                extra = bits - FLOOR_BITS
                new_cost = prior_cost + extra
                if new_cost > budget:
                    continue
                candidate_score = prior_score + _gain_integer(curve, bits)
                incumbent = next_scores[new_cost]
                if incumbent is None or candidate_score > incumbent:
                    next_scores[new_cost] = candidate_score
                    costs[new_cost] = prior_cost
                    choices[new_cost] = bits
        if next_scores[budget] is None and len(parent_costs) == len(ordered) - 1:
            raise AssertionError("exact layer budget is unreachable")
        scores = next_scores
        parent_costs.append(costs)
        parent_bits.append(choices)

    final_score = scores[budget]
    if final_score is None:
        raise AssertionError("exact layer budget is unreachable")
    selected_reversed: list[int] = []
    cost = budget
    for item_index in range(len(ordered) - 1, -1, -1):
        bits = int(parent_bits[item_index][cost])
        prior = int(parent_costs[item_index][cost])
        if bits not in ALLOWED_BITS or prior < 0:
            raise AssertionError("allocation backpointer corruption")
        selected_reversed.append(bits)
        cost = prior
    if cost != 0:
        raise AssertionError("allocation did not return to zero budget")
    selected = tuple(reversed(selected_reversed))
    result = {curve.tensor_id.key: bits for curve, bits in zip(ordered, selected)}
    audit_allocation(result)
    return LayerAllocation(
        layer=layer,
        bits=result,
        masses=tuple(format(curve.mass, "f") for curve in ordered[:: len(PROJECTIONS)]),
        score_integer=int(final_score),
        score_scale=SCORE_SCALE,
        upgrade_units=budget,
        fixed_point_iteration=fixed_point_iteration,
        probe_sha256=probe_sha256,
    )


def audit_allocation(bits_by_tensor: Mapping[str, int]) -> None:
    if len(bits_by_tensor) != TENSORS_PER_LAYER:
        raise ValueError(
            f"allocation has {len(bits_by_tensor)} tensors, need {TENSORS_PER_LAYER}"
        )
    values = tuple(bits_by_tensor.values())
    if any(type(bits) is not int or bits not in ALLOWED_BITS for bits in values):
        raise ValueError(f"allocation bits must be integers in {ALLOWED_BITS}")
    if min(values) < FLOOR_BITS:
        raise AssertionError("sub-floor tensor escaped validation")
    if sum(values) != TARGET_BIT_UNITS_PER_LAYER:
        raise ValueError(
            f"allocation sums to {sum(values)}, need {TARGET_BIT_UNITS_PER_LAYER}"
        )
    upgrades = sum(bits - FLOOR_BITS for bits in values)
    if upgrades != UPGRADE_UNITS_PER_LAYER:
        raise AssertionError("upgrade arithmetic drift")


def allocation_to_json(allocation: LayerAllocation) -> dict[str, object]:
    audit_allocation(allocation.bits)
    histogram = {str(bits): 0 for bits in ALLOWED_BITS}
    for bits in allocation.bits.values():
        histogram[str(bits)] += 1
    return {
        "layer": allocation.layer,
        "bits": dict(sorted(allocation.bits.items())),
        "histogram": histogram,
        "tensor_count": len(allocation.bits),
        "bit_units": sum(allocation.bits.values()),
        "target_bpw": "3.5",
        "upgrade_units": allocation.upgrade_units,
        "score_integer": allocation.score_integer,
        "score_scale": allocation.score_scale,
        "fixed_point_iteration": allocation.fixed_point_iteration,
        "probe_sha256": allocation.probe_sha256,
        "mass_by_expert": list(allocation.masses),
    }


def write_allocation(path: str | Path, allocation: LayerAllocation) -> str:
    payload = allocation_to_json(allocation)
    atomic_write_json(path, payload)
    return sha256_bytes(canonical_json_bytes(payload))
