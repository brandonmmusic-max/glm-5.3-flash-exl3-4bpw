"""Compatibility-safe exact allocation across candidate factories.

An MCG layer can own shared transform payloads.  Flat per-matrix knapsack
selection is therefore invalid when proposals belong to different shared
transform domains.  This module first constructs an exact Pareto frontier for
every ``(coupling_group, compatibility_domain)`` pair, then runs the global
multiple-choice knapsack across coupling groups.  The selected checkpoint has
exactly one compatible shared domain per group while still allowing different
factories to compete inside a domain when they attest identical shared state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

from ..allocation.global_dp import Allocation, Candidate, allocate, pareto_frontier
from ..core.artifacts import canonical_json, sha256_bytes
from .factory_union import FactoryUnion


SCHEMA_COUPLED_FACTORY_ALLOCATION = (
    "quant-pipeline.coupled-candidate-factory-allocation.v1"
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


@dataclass(frozen=True)
class CoupledFactoryAllocation:
    allocation: Allocation
    selected_candidates: tuple[Candidate, ...]
    manifest: Mapping[str, Any]


def _candidate_metadata(candidate: Candidate) -> Mapping[str, Any]:
    metadata = candidate.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("factory candidate lacks coupling metadata")
    required = {
        "factory_name",
        "rate",
        "coupling_group_id",
        "compatibility_domain_sha256",
        "fixed_shared_bytes",
        "domain_fixed_damage",
        "proposal_sha256",
        "common_score_sha256",
        "record_sha256",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"factory candidate coupling metadata lacks {sorted(missing)}")
    if not isinstance(metadata["coupling_group_id"], str) or not metadata["coupling_group_id"]:
        raise ValueError("factory candidate coupling group must be non-empty")
    domain = metadata["compatibility_domain_sha256"]
    if not isinstance(domain, str) or re.fullmatch(r"[0-9a-f]{64}", domain) is None:
        raise ValueError("factory candidate compatibility domain must be a SHA-256")
    fixed = metadata["fixed_shared_bytes"]
    if isinstance(fixed, bool) or not isinstance(fixed, int) or fixed < 0:
        raise ValueError("factory candidate fixed shared bytes must be nonnegative")
    fixed_damage = metadata["domain_fixed_damage"]
    if isinstance(fixed_damage, bool) or not isinstance(fixed_damage, (int, float)) or not math.isfinite(float(fixed_damage)):
        raise ValueError("factory candidate domain-fixed damage must be finite")
    return metadata


def _internal_frontier(
    candidates: Iterable[Candidate],
    *,
    expected_units: set[str],
    quantum: int,
    preserve_every_cost: bool = False,
) -> list[Candidate]:
    """Return all non-dominated exact private-byte profiles for one domain."""

    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.unit_id, []).append(candidate)
    if set(grouped) != expected_units:
        return []
    if quantum < 1 or any(candidate.stored_bytes % quantum for rows in grouped.values() for candidate in rows):
        raise ValueError("every private candidate cost must be divisible by quantum")

    states: dict[int, tuple[float, tuple[Candidate, ...]]] = {0: (0.0, ())}
    for unit_id in sorted(grouped):
        if preserve_every_cost:
            best_by_cost: dict[int, Candidate] = {}
            for candidate in grouped[unit_id]:
                previous = best_by_cost.get(candidate.stored_bytes)
                if previous is None or (
                    candidate.predicted_damage,
                    candidate.choice_id,
                ) < (
                    previous.predicted_damage,
                    previous.choice_id,
                ):
                    best_by_cost[candidate.stored_bytes] = candidate
            choices = list(best_by_cost.values())
        else:
            choices = pareto_frontier(grouped[unit_id])
        next_states: dict[int, tuple[float, tuple[Candidate, ...]]] = {}
        for used, (damage, selected) in states.items():
            for choice in choices:
                new_used = used + choice.stored_bytes
                new_damage = damage + float(choice.predicted_damage)
                previous = next_states.get(new_used)
                if previous is None or new_damage < previous[0]:
                    next_states[new_used] = (new_damage, selected + (choice,))
        if preserve_every_cost:
            # Exact-fill allocation cannot discard a higher-cost state merely
            # because a cheaper state has lower damage: that exact cost can be
            # the only way to close the global byte budget.
            states = next_states
        else:
            states = {}
            best = math.inf
            for used in sorted(next_states):
                damage, selected = next_states[used]
                if damage < best:
                    states[used] = (damage, selected)
                    best = damage
    return [
        Candidate(
            unit_id="internal-profile",
            choice_id=f"private-{used}",
            stored_bytes=used,
            predicted_damage=damage,
            metadata={"selected_candidates": selected},
        )
        for used, (damage, selected) in sorted(states.items())
    ]


def allocate_factory_union(
    union: FactoryUnion,
    *,
    byte_budget: int,
    quantum: int = 1,
    require_exact_budget: bool = False,
) -> CoupledFactoryAllocation:
    """Select exact-rate and factory choices without mixing shared domains.

    ``fixed_shared_bytes`` is charged exactly once for a selected compatibility
    domain, regardless of how many private candidates use it.  A domain that
    does not cover every unit in its coupling group is ineligible.  Native MCG
    coverage remains guaranteed by :func:`build_factory_union`.
    """

    if byte_budget < 0 or quantum < 1 or byte_budget % quantum:
        raise ValueError("invalid coupled factory allocation budget or quantum")
    candidates = list(union.allocator_candidates)
    units = union.ledger.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("factory union lacks candidate units")
    unit_to_group: dict[str, str] = {}
    units_by_group: dict[str, set[str]] = {}
    for row in units:
        unit_id = str(row["unit_id"])
        group = str(row.get("coupling_group_id", ""))
        if not group or unit_id in unit_to_group:
            raise ValueError("factory union has invalid coupling-group inventory")
        unit_to_group[unit_id] = group
        units_by_group.setdefault(group, set()).add(unit_id)

    by_group_domain: dict[tuple[str, str], list[Candidate]] = {}
    fixed_costs: dict[tuple[str, str], int] = {}
    fixed_damages: dict[tuple[str, str], float] = {}
    for candidate in candidates:
        metadata = _candidate_metadata(candidate)
        group = str(metadata["coupling_group_id"])
        domain = str(metadata["compatibility_domain_sha256"])
        if unit_to_group.get(candidate.unit_id) != group:
            raise ValueError("factory candidate coupling group differs from its unit")
        key = (group, domain)
        fixed = int(metadata["fixed_shared_bytes"])
        incumbent = fixed_costs.setdefault(key, fixed)
        if incumbent != fixed:
            raise ValueError("fixed shared byte cost differs inside a compatibility domain")
        fixed_damage = float(metadata["domain_fixed_damage"])
        incumbent_damage = fixed_damages.setdefault(key, fixed_damage)
        if not math.isclose(incumbent_damage, fixed_damage, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("domain-fixed damage differs inside a compatibility domain")
        by_group_domain.setdefault(key, []).append(candidate)

    outer_candidates: list[Candidate] = []
    domain_rows: list[dict[str, Any]] = []
    for (group, domain), values in sorted(by_group_domain.items()):
        fixed = fixed_costs[(group, domain)]
        if fixed % quantum:
            raise ValueError("fixed shared candidate cost must be divisible by quantum")
        profiles = _internal_frontier(
            values,
            expected_units=units_by_group[group],
            quantum=quantum,
            preserve_every_cost=require_exact_budget,
        )
        if not profiles:
            continue
        for profile in profiles:
            selected = tuple(profile.metadata["selected_candidates"])
            fixed_damage = fixed_damages[(group, domain)]
            choice_body = {
                "coupling_group_id": group,
                "compatibility_domain_sha256": domain,
                "fixed_shared_bytes": fixed,
                "private_payload_bytes": profile.stored_bytes,
                "selected_choice_ids": [row.choice_id for row in selected],
                "domain_fixed_damage": fixed_damage,
            }
            choice_sha = _hash_json(choice_body)
            outer_candidates.append(
                Candidate(
                    unit_id=group,
                    choice_id=f"{group}.{domain[:16]}.{choice_sha[:16]}",
                    stored_bytes=fixed + profile.stored_bytes,
                    predicted_damage=profile.predicted_damage + fixed_damage,
                    metadata=choice_body
                    | {
                        "choice_sha256": choice_sha,
                        "selected_candidates": selected,
                    },
                )
            )
        domain_rows.append(
            {
                "coupling_group_id": group,
                "compatibility_domain_sha256": domain,
                "fixed_shared_bytes": fixed,
                "domain_fixed_damage": fixed_damages[(group, domain)],
                "profile_count": len(profiles),
            }
        )
    missing_groups = sorted(set(units_by_group) - {row.unit_id for row in outer_candidates})
    if missing_groups:
        raise ValueError(
            "no complete compatibility domain covers coupling groups: "
            + ", ".join(missing_groups)
        )

    if require_exact_budget:
        grouped: dict[str, list[Candidate]] = {}
        for candidate in outer_candidates:
            grouped.setdefault(candidate.unit_id, []).append(candidate)
        states: dict[int, tuple[float, tuple[Candidate, ...]]] = {0: (0.0, ())}
        for group in sorted(grouped):
            next_states: dict[int, tuple[float, tuple[Candidate, ...]]] = {}
            for used, (damage, selected) in states.items():
                for choice in grouped[group]:
                    new_used = used + choice.stored_bytes
                    if new_used > byte_budget:
                        continue
                    new_damage = damage + choice.predicted_damage
                    previous = next_states.get(new_used)
                    new_ids = tuple(row.choice_id for row in selected + (choice,))
                    previous_ids = (
                        tuple(row.choice_id for row in previous[1])
                        if previous is not None
                        else ()
                    )
                    if previous is None or (new_damage, new_ids) < (
                        previous[0], previous_ids
                    ):
                        next_states[new_used] = (new_damage, selected + (choice,))
            if not next_states:
                raise ValueError(f"exact coupled budget is infeasible at group {group}")
            states = next_states
        exact = states.get(byte_budget)
        if exact is None:
            raise ValueError(
                f"no compatibility-safe candidate-factory allocation exactly fills {byte_budget} bytes"
            )
        result = Allocation(
            choices=exact[1],
            stored_bytes=byte_budget,
            predicted_damage=exact[0],
        )
    else:
        result = allocate(outer_candidates, byte_budget=byte_budget, quantum=quantum)
    selected_private = tuple(
        private
        for profile in result.choices
        for private in profile.metadata["selected_candidates"]
    )
    if {row.unit_id for row in selected_private} != set(unit_to_group):
        raise RuntimeError("coupled factory allocation selected incomplete unit coverage")
    selected_groups = []
    for profile in result.choices:
        metadata = profile.metadata
        selected_groups.append(
            {
                "coupling_group_id": profile.unit_id,
                "compatibility_domain_sha256": metadata["compatibility_domain_sha256"],
                "fixed_shared_bytes": metadata["fixed_shared_bytes"],
                "private_payload_bytes": metadata["private_payload_bytes"],
                "stored_bytes": profile.stored_bytes,
                "predicted_damage": profile.predicted_damage,
                "domain_fixed_damage": metadata["domain_fixed_damage"],
                "private_predicted_damage": profile.predicted_damage
                - metadata["domain_fixed_damage"],
                "choice_sha256": metadata["choice_sha256"],
                "selected_choice_ids": metadata["selected_choice_ids"],
            }
        )
    selected_rows = [
        {
            "unit_id": row.unit_id,
            "choice_id": row.choice_id,
            "stored_bytes": row.stored_bytes,
            "predicted_damage": row.predicted_damage,
            "factory_name": row.metadata["factory_name"],
            "rate": row.metadata["rate"],
            "coupling_group_id": row.metadata["coupling_group_id"],
            "compatibility_domain_sha256": row.metadata["compatibility_domain_sha256"],
            "domain_fixed_damage": row.metadata["domain_fixed_damage"],
            "record_sha256": row.metadata["record_sha256"],
        }
        for row in selected_private
    ]
    manifest = {
        "schema": SCHEMA_COUPLED_FACTORY_ALLOCATION,
        "factory_union_ledger_sha256": union.ledger["ledger_sha256"],
        "byte_budget": byte_budget,
        "quantum": quantum,
        "exact_budget_required": bool(require_exact_budget),
        "stored_bytes": result.stored_bytes,
        "predicted_damage": result.predicted_damage,
        "compatibility_policy": "one complete shared-transform domain per coupling group",
        "available_domains": domain_rows,
        "selected_groups": selected_groups,
        "selected_candidates": selected_rows,
    }
    manifest["allocation_sha256"] = _hash_json(manifest)
    return CoupledFactoryAllocation(
        allocation=result,
        selected_candidates=selected_private,
        manifest=manifest,
    )
