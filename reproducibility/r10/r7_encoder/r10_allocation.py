"""Round-10 one-pass allocator with an explicit 3 -> 4 -> 5 dependency.

The legacy DP compared the cumulative 3-bit and 5-bit losses directly.  It
could therefore award two units to a tensor whose 4-bit measurement was worse
than its 3-bit measurement.  R10 admits 5 bits only when both measured marginal
steps are strictly beneficial, while retaining the exact 384-unit layer budget.
"""

from __future__ import annotations

from array import array
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from typing import Iterable

from .allocation import SCORE_SCALE, SensitivityCurve, audit_allocation
from .constants import FLOOR_BITS, UPGRADE_UNITS_PER_LAYER, all_layer_tensor_ids
from .types import LayerAllocation


def _score(curve: SensitivityCurve, bits: int) -> int:
    with localcontext() as context:
        context.prec = 50
        value = curve.mass * (
            curve.loss_by_bits[FLOOR_BITS] - curve.loss_by_bits[bits]
        )
        return int(
            (value * Decimal(SCORE_SCALE)).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )


def _choices(curve: SensitivityCurve) -> tuple[int, ...]:
    result = [3]
    four_helps = curve.loss_by_bits[4] < curve.loss_by_bits[3]
    five_helps_after_four = curve.loss_by_bits[5] < curve.loss_by_bits[4]
    if four_helps:
        result.append(4)
        if five_helps_after_four:
            result.append(5)
    return tuple(result)


def solve_sequential_allocation(
    layer: int,
    curves: Iterable[SensitivityCurve],
    *,
    probe_sha256: str,
) -> LayerAllocation:
    """Spend exactly 384 units; a 5-bit state contains two justified steps."""

    expected = all_layer_tensor_ids(layer)
    by_id = {curve.tensor_id: curve for curve in curves}
    if set(by_id) != set(expected):
        raise ValueError("allocation curves are not exactly the layer tensor set")
    ordered = tuple(by_id[tensor_id] for tensor_id in expected)
    budget = UPGRADE_UNITS_PER_LAYER
    scores: list[int | None] = [None] * (budget + 1)
    scores[0] = 0
    parent_costs: list[array] = []
    parent_bits: list[bytearray] = []

    for curve in ordered:
        next_scores: list[int | None] = [None] * (budget + 1)
        costs = array("h", [-1]) * (budget + 1)
        bits_at = bytearray(budget + 1)
        for prior_cost, prior_score in enumerate(scores):
            if prior_score is None:
                continue
            for bits in _choices(curve):
                cost = prior_cost + bits - FLOOR_BITS
                if cost > budget:
                    continue
                candidate = prior_score + _score(curve, bits)
                incumbent = next_scores[cost]
                # Canonical tie rule: lower bit choice wins.
                if incumbent is None or candidate > incumbent:
                    next_scores[cost] = candidate
                    costs[cost] = prior_cost
                    bits_at[cost] = bits
        scores = next_scores
        parent_costs.append(costs)
        parent_bits.append(bits_at)

    if scores[budget] is None:
        helpful_four = sum(4 in _choices(curve) for curve in ordered)
        helpful_five = sum(5 in _choices(curve) for curve in ordered)
        raise RuntimeError(
            "exact 384-upgrade budget is incompatible with measured sequential "
            f"improvements: four_steps={helpful_four}, five_steps={helpful_five}"
        )

    selected_reversed: list[int] = []
    cost = budget
    for index in range(len(ordered) - 1, -1, -1):
        bits = int(parent_bits[index][cost])
        prior = int(parent_costs[index][cost])
        if bits not in (3, 4, 5) or prior < 0:
            raise AssertionError("allocation backpointer corruption")
        selected_reversed.append(bits)
        cost = prior
    if cost:
        raise AssertionError("allocation did not return to the floor")
    selected = tuple(reversed(selected_reversed))
    bits = {curve.tensor_id.key: value for curve, value in zip(ordered, selected)}
    audit_allocation(bits)
    return LayerAllocation(
        layer=layer,
        bits=bits,
        masses=tuple(format(curve.mass, "f") for curve in ordered[::3]),
        score_integer=int(scores[budget]),
        score_scale=SCORE_SCALE,
        upgrade_units=budget,
        fixed_point_iteration=0,
        probe_sha256=probe_sha256,
    )
