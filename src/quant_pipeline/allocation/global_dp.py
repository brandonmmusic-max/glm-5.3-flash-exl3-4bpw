from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    unit_id: str
    choice_id: str
    stored_bytes: int
    predicted_damage: float
    metadata: dict | None = None


@dataclass(frozen=True)
class Allocation:
    choices: tuple[Candidate, ...]
    stored_bytes: int
    predicted_damage: float


@dataclass(frozen=True)
class AccountedAllocation:
    choices: tuple[Candidate, ...]
    variable_payload_bytes: int
    fixed_layer_shared_bytes: int
    stored_bytes: int
    predicted_damage: float


def pareto_frontier(candidates: Iterable[Candidate]) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda c: (c.stored_bytes, c.predicted_damage, c.choice_id))
    frontier: list[Candidate] = []
    best_damage = float("inf")
    for candidate in ordered:
        if candidate.predicted_damage < best_damage:
            frontier.append(candidate)
            best_damage = candidate.predicted_damage
    return frontier


def allocate(candidates: Iterable[Candidate], byte_budget: int, quantum: int = 1) -> Allocation:
    """Exact multiple-choice knapsack over a codec-payload-byte budget.

    A quantum greater than one is accepted only when the budget and every
    candidate cost are exactly divisible, so discretization cannot hide or
    reject a feasible allocation.
    """
    if byte_budget < 0 or quantum < 1:
        raise ValueError("invalid budget or quantum")
    all_candidates = list(candidates)
    grouped: dict[str, list[Candidate]] = {}
    for candidate in all_candidates:
        if not isinstance(candidate.unit_id, str) or not candidate.unit_id:
            raise ValueError("candidate unit_id must be a non-empty string")
        if not isinstance(candidate.choice_id, str) or not candidate.choice_id:
            raise ValueError("candidate choice_id must be a non-empty string")
        if isinstance(candidate.stored_bytes, bool) or not isinstance(candidate.stored_bytes, int) or candidate.stored_bytes < 0:
            raise ValueError("candidate stored_bytes must be a non-negative integer")
        if not isinstance(candidate.predicted_damage, (int, float)) or not math.isfinite(float(candidate.predicted_damage)):
            raise ValueError("candidate predicted_damage must be finite")
        grouped.setdefault(candidate.unit_id, []).append(candidate)
    if not grouped:
        return Allocation((), 0, 0.0)
    if byte_budget % quantum or any(candidate.stored_bytes % quantum for candidate in all_candidates):
        raise ValueError("budget and every candidate cost must be exactly divisible by quantum")
    budget = byte_budget // quantum
    states: dict[int, tuple[float, tuple[Candidate, ...]]] = {0: (0.0, ())}
    for unit_id in sorted(grouped):
        choices = pareto_frontier(grouped[unit_id])
        new_states: dict[int, tuple[float, tuple[Candidate, ...]]] = {}
        for used, (damage, selected) in states.items():
            for choice in choices:
                cost = choice.stored_bytes // quantum
                new_used = used + cost
                if new_used > budget:
                    continue
                new_damage = damage + choice.predicted_damage
                previous = new_states.get(new_used)
                if previous is None or new_damage < previous[0]:
                    new_states[new_used] = (new_damage, selected + (choice,))
        if not new_states:
            raise ValueError(f"budget infeasible at unit {unit_id}")
        pruned: dict[int, tuple[float, tuple[Candidate, ...]]] = {}
        best = float("inf")
        for used in sorted(new_states):
            damage, selected = new_states[used]
            if damage < best:
                pruned[used] = (damage, selected)
                best = damage
        states = pruned
    used, (damage, selected) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    actual_bytes = sum(choice.stored_bytes for choice in selected)
    return Allocation(selected, actual_bytes, damage)


def allocate_with_fixed_layer_cost(
    candidates: Iterable[Candidate],
    byte_budget: int,
    fixed_layer_shared_bytes: int,
    quantum: int = 1,
) -> AccountedAllocation:
    """Allocate expert-private choices after charging sealed layer-fixed bytes."""
    if (
        isinstance(fixed_layer_shared_bytes, bool)
        or not isinstance(fixed_layer_shared_bytes, int)
        or fixed_layer_shared_bytes < 0
    ):
        raise ValueError("fixed layer-shared cost must be a non-negative integer")
    if fixed_layer_shared_bytes > byte_budget:
        raise ValueError("fixed layer-shared payload alone exceeds the byte budget")
    variable_budget = byte_budget - fixed_layer_shared_bytes
    result = allocate(candidates, variable_budget, quantum)
    total = result.stored_bytes + fixed_layer_shared_bytes
    if total > byte_budget:
        raise RuntimeError("accounted allocator exceeded the total byte budget")
    return AccountedAllocation(
        choices=result.choices,
        variable_payload_bytes=result.stored_bytes,
        fixed_layer_shared_bytes=fixed_layer_shared_bytes,
        stored_bytes=total,
        predicted_damage=result.predicted_damage,
    )
