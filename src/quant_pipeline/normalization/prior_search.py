"""Source-derived GLM-3.5 search controls, generalized only in geometry.

The five permutation policies and three per-128 scale families are direct
ports of ``reproducibility/r10/r7_encoder/{permutation,search}.py`` from the
sealed GLM-5.2 3.5-bpw control.  Shapley/MCG candidates are evaluated in
addition to these controls; they do not silently replace them.

One deliberate fail-closed deviation is applied before evaluating the family:
a covariance/Hessian diagonal containing a negative entry is rejected.  The
historical permutation helpers would rank such values, but negative diagonal
energy is outside the valid estimator domain and must not select a policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as functional


PERMUTATION_POLICIES = (
    "identity",
    "ldlq_visit_descending_diag",
    "stored_descending_diag",
    "energy_balanced",
    "energy_balanced_contiguous",
)
SCALE_GRID = (0.5, 0.625, 0.8, 0.9, 1.0, 1.1, 1.25, 1.6, 2.0)
SCALE_FAMILIES = ("identity", "per128-grid", "inverse-per128-grid")


def validate_permutation(permutation: Sequence[int], size: int | None = None) -> tuple[int, ...]:
    result = tuple(int(index) for index in permutation)
    expected = len(result) if size is None else int(size)
    if expected < 1 or len(result) != expected or sorted(result) != list(range(expected)):
        raise ValueError(f"permutation must be a bijection over [0,{expected})")
    return result


def identity_permutation(size: int) -> tuple[int, ...]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    return tuple(range(size))


def ldlq_visit_old_indices(permutation: Sequence[int]) -> tuple[int, ...]:
    return tuple(reversed(validate_permutation(permutation)))


def assert_ldlq_visit_descending(permutation: Sequence[int], diagonal: Sequence[float]) -> None:
    values = tuple(float(value) for value in diagonal)
    visit = ldlq_visit_old_indices(permutation)
    if len(visit) != len(values):
        raise ValueError("permutation and diagonal size differ")
    energies = tuple(values[index] for index in visit)
    if any(left < right for left, right in zip(energies, energies[1:])):
        raise ValueError("LDLQ visit order is not descending by covariance diagonal")


def descending_diag_permutation(diagonal: Iterable[float]) -> tuple[int, ...]:
    values = tuple(float(value) for value in diagonal)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("diagonal must be nonempty and finite")
    ranked = tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))
    stored = tuple(reversed(ranked))
    assert_ldlq_visit_descending(stored, values)
    return stored


def stored_descending_diag_permutation(diagonal: Iterable[float]) -> tuple[int, ...]:
    values = tuple(float(value) for value in diagonal)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("diagonal must be nonempty and finite")
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))


def energy_balanced_permutation(
    diagonal: Iterable[float], *, block: int = 128, serpentine: bool = True
) -> tuple[int, ...]:
    values = tuple(float(value) for value in diagonal)
    if (
        not values or isinstance(block, bool) or not isinstance(block, int)
        or block < 1 or len(values) % block
        or not all(math.isfinite(value) and value >= 0.0 for value in values)
    ):
        raise ValueError("diagonal must be finite, nonnegative, and a positive multiple of block")
    nblocks = len(values) // block
    ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
    buckets: list[list[int]] = [[] for _ in range(nblocks)]
    totals = [0.0] * nblocks
    for index in ranked:
        eligible = [bucket for bucket in range(nblocks) if len(buckets[bucket]) < block]
        destination = min(eligible, key=lambda bucket: (totals[bucket], bucket))
        buckets[destination].append(index)
        totals[destination] += values[index]
    for bucket, indices in enumerate(buckets):
        indices.sort(key=lambda index: (-values[index], index))
        if serpentine and bucket % 2:
            indices.reverse()
    return validate_permutation(tuple(index for bucket in buckets for index in bucket), len(values))


def policy_permutations(diagonal: Sequence[float], *, block: int = 128) -> dict[str, tuple[int, ...]]:
    values = tuple(float(value) for value in diagonal)
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("policy diagonal must be nonempty, finite, and nonnegative")
    result = {
        "identity": identity_permutation(len(values)),
        "ldlq_visit_descending_diag": descending_diag_permutation(values),
        "stored_descending_diag": stored_descending_diag_permutation(values),
        "energy_balanced": energy_balanced_permutation(values, block=block),
        "energy_balanced_contiguous": energy_balanced_permutation(values, block=block, serpentine=False),
    }
    if tuple(result) != PERMUTATION_POLICIES:
        raise AssertionError("control must evaluate exactly five permutation policies")
    return result


def inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    perm = validate_permutation(permutation)
    inverse = [0] * len(perm)
    for new, old in enumerate(perm):
        inverse[old] = new
    return tuple(inverse)


def permute_expert_hf(gate_weight: Any, up_weight: Any, down_weight: Any, permutation: Sequence[int]):
    """Apply ``P Wg``, ``P Wu``, ``Wd P.T`` to HF ``[out,in]`` tensors."""
    perm = validate_permutation(permutation, int(gate_weight.shape[0]))
    if tuple(gate_weight.shape) != tuple(up_weight.shape):
        raise ValueError("gate and up shapes differ")
    if down_weight.shape[1] != gate_weight.shape[0]:
        raise ValueError("down input does not match intermediate size")
    index = torch.tensor(perm, device=gate_weight.device, dtype=torch.long)
    return (
        gate_weight.index_select(0, index).contiguous(),
        up_weight.index_select(0, index.to(up_weight.device)).contiguous(),
        down_weight.index_select(1, index.to(down_weight.device)).contiguous(),
    )


def permute_down_inputs(inputs: Any, permutation: Sequence[int]):
    perm = validate_permutation(permutation, int(inputs.shape[-1]))
    return inputs.index_select(-1, torch.tensor(perm, device=inputs.device, dtype=torch.long)).contiguous()


def permute_down_hessian(hessian: Any, permutation: Sequence[int]):
    perm = validate_permutation(permutation, int(hessian.shape[0]))
    if hessian.ndim != 2 or hessian.shape[1] != hessian.shape[0]:
        raise ValueError("down Hessian must be square")
    index = torch.tensor(perm, device=hessian.device, dtype=torch.long)
    return hessian.index_select(0, index).index_select(1, index).contiguous()


def normalized_quarter_scales(values: Sequence[float]) -> tuple[float, ...]:
    cleaned = [max(float(value), 1e-20) for value in values]
    if not cleaned or not all(math.isfinite(value) for value in cleaned):
        raise ValueError("scale statistics must be nonempty and finite")
    geometric = math.exp(sum(math.log(value) for value in cleaned) / len(cleaned))
    scales = [max(0.5, min(2.0, (geometric / value) ** 0.25)) for value in cleaned]
    normalization = math.exp(sum(math.log(value) for value in scales) / len(scales))
    return tuple(value / normalization for value in scales)


def coordinate_grid_scales(values: Sequence[float]) -> tuple[float, ...]:
    targets = normalized_quarter_scales(values)
    selected = [
        min(SCALE_GRID, key=lambda candidate: (
            abs(math.log(float(candidate) / float(target))), float(candidate)
        ))
        for target in targets
    ]
    normalization = math.exp(sum(math.log(float(value)) for value in selected) / len(selected))
    # Python-f64 division is part of the authoritative checkpoint boundary.
    return tuple(float(value) / normalization for value in selected)


def scale_family_candidates(values: Sequence[float]) -> dict[str, tuple[float, ...]]:
    grid = coordinate_grid_scales(values)
    result = {
        "identity": tuple(1.0 for _ in values),
        "per128-grid": grid,
        "inverse-per128-grid": tuple(1.0 / float(value) for value in grid),
    }
    if tuple(result) != SCALE_FAMILIES:
        raise AssertionError("control must evaluate exactly three scale families")
    return result


@dataclass(frozen=True)
class PermutationAudit:
    size: int
    max_abs_function_error: float
    relative_function_error: float
    exact_inverse: bool
    exact_weight_roundtrip: bool


def functional_oracle(x: Any, gate_weight: Any, up_weight: Any, down_weight: Any, permutation: Sequence[int]) -> PermutationAudit:
    perm = validate_permutation(permutation, int(gate_weight.shape[0]))
    pg, pu, pd = permute_expert_hf(gate_weight, up_weight, down_weight, perm)
    reference = (functional.silu(x @ gate_weight.T) * (x @ up_weight.T)) @ down_weight.T
    candidate = (functional.silu(x @ pg.T) * (x @ pu.T)) @ pd.T
    delta = reference - candidate
    inverse = inverse_permutation(perm)
    index = torch.tensor(inverse, dtype=torch.long)
    exact_weights = (
        torch.equal(pg.index_select(0, index.to(pg.device)), gate_weight)
        and torch.equal(pu.index_select(0, index.to(pu.device)), up_weight)
        and torch.equal(pd.index_select(1, index.to(pd.device)), down_weight)
    )
    return PermutationAudit(
        len(perm), float(delta.abs().max().item()),
        float((delta.double().norm() / reference.double().norm().clamp_min(1e-30)).item()),
        all(inverse[old] == new for new, old in enumerate(perm)), exact_weights,
    )
