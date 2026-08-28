"""Baked intermediate-channel permutation policies and exactness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .constants import HAD_K, INTERMEDIATE_SIZE


def validate_permutation(
    permutation: Sequence[int], size: int = INTERMEDIATE_SIZE
) -> tuple[int, ...]:
    result = tuple(int(index) for index in permutation)
    if len(result) != size or sorted(result) != list(range(size)):
        raise ValueError(f"permutation must be a bijection over [0,{size})")
    return result


def identity_permutation(size: int = INTERMEDIATE_SIZE) -> tuple[int, ...]:
    return tuple(range(size))


def descending_diag_permutation(diagonal: Iterable[float]) -> tuple[int, ...]:
    """Stored order whose bottom-up LDLQ visit sees descending energy."""

    values = tuple(float(value) for value in diagonal)
    ranked = tuple(
        sorted(range(len(values)), key=lambda index: (-values[index], index))
    )
    stored = tuple(reversed(ranked))
    assert_ldlq_visit_descending(stored, values)
    return stored


def stored_descending_diag_permutation(diagonal: Iterable[float]) -> tuple[int, ...]:
    """High-to-low stored coordinates; retained as a separately named pilot."""

    values = tuple(float(value) for value in diagonal)
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))


def ldlq_visit_old_indices(permutation: Sequence[int]) -> tuple[int, ...]:
    """Map the inherited bottom-up K walk into original channel indices."""

    return tuple(reversed(validate_permutation(permutation, len(permutation))))


def assert_ldlq_visit_descending(
    permutation: Sequence[int], diagonal: Sequence[float]
) -> None:
    visit = ldlq_visit_old_indices(permutation)
    energies = tuple(float(diagonal[index]) for index in visit)
    if any(left < right for left, right in zip(energies, energies[1:])):
        raise ValueError("LDLQ visit order is not descending by covariance diagonal")


def energy_balanced_permutation(
    diagonal: Iterable[float], *, block: int = HAD_K, serpentine: bool = True
) -> tuple[int, ...]:
    """Balance high-energy channels across future Hadamard blocks.

    The returned tuple maps new coordinate -> old coordinate. Within each
    future block channels are ordered high-to-low except that alternate blocks
    may be reversed to avoid aligning the same ranks at every boundary.
    """

    values = tuple(float(value) for value in diagonal)
    if not values or len(values) % block:
        raise ValueError("diagonal length must be a positive multiple of block")
    nblocks = len(values) // block
    ranked = sorted(range(len(values)), key=lambda index: (-values[index], index))
    buckets: list[list[int]] = [[] for _ in range(nblocks)]
    # Greedy least-energy assignment keeps both counts and total diagonal mass
    # balanced. Ties are resolved by block index.
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
    return validate_permutation(
        tuple(index for bucket in buckets for index in bucket), len(values)
    )


def inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    perm = validate_permutation(permutation, len(permutation))
    inverse = [0] * len(perm)
    for new, old in enumerate(perm):
        inverse[old] = new
    return tuple(inverse)


def permute_expert_hf(gate_weight, up_weight, down_weight, permutation: Sequence[int]):
    """Apply `P Wg`, `P Wu`, `Wd P^T` to HF `[out,in]` tensors."""

    import torch

    perm = validate_permutation(permutation, int(gate_weight.shape[0]))
    index = torch.tensor(perm, device=gate_weight.device, dtype=torch.long)
    if tuple(gate_weight.shape) != tuple(up_weight.shape):
        raise ValueError("gate and up shapes differ")
    if down_weight.shape[1] != gate_weight.shape[0]:
        raise ValueError("down input does not match intermediate size")
    return (
        gate_weight.index_select(0, index).contiguous(),
        up_weight.index_select(0, index).contiguous(),
        down_weight.index_select(1, index.to(down_weight.device)).contiguous(),
    )


def permute_covariance(covariance, permutation: Sequence[int]):
    import torch

    perm = validate_permutation(permutation, int(covariance.shape[0]))
    if covariance.ndim != 2 or covariance.shape[1] != covariance.shape[0]:
        raise ValueError("covariance must be square")
    index = torch.tensor(perm, device=covariance.device, dtype=torch.long)
    return covariance.index_select(0, index).index_select(1, index).contiguous()


def choose_policy(losses: Mapping[str, float], *, required: Iterable[str]) -> str:
    names = tuple(required)
    missing = [name for name in names if name not in losses]
    if missing:
        raise ValueError(f"permutation pilot missing policies: {missing}")
    return min(names, key=lambda name: (float(losses[name]), name))


@dataclass(frozen=True)
class PermutationAudit:
    size: int
    max_abs_function_error: float
    relative_function_error: float
    exact_inverse: bool
    exact_weight_roundtrip: bool


def functional_oracle(
    x,
    gate_weight,
    up_weight,
    down_weight,
    permutation: Sequence[int],
) -> PermutationAudit:
    import torch.nn.functional as functional

    perm = validate_permutation(permutation, int(gate_weight.shape[0]))
    pg, pu, pd = permute_expert_hf(gate_weight, up_weight, down_weight, perm)
    reference = (functional.silu(x @ gate_weight.T) * (x @ up_weight.T)) @ down_weight.T
    candidate = (functional.silu(x @ pg.T) * (x @ pu.T)) @ pd.T
    maximum = float((reference - candidate).abs().max().item())
    relative = float(
        (
            (reference - candidate).double().norm()
            / reference.double().norm().clamp_min(1e-30)
        ).item()
    )
    inverse = inverse_permutation(perm)
    exact_inverse = all(inverse[old] == new for new, old in enumerate(perm))
    index = __import__("torch").tensor(inverse, dtype=__import__("torch").long)
    exact_weights = (
        __import__("torch").equal(pg.index_select(0, index.to(pg.device)), gate_weight)
        and __import__("torch").equal(
            pu.index_select(0, index.to(pu.device)), up_weight
        )
        and __import__("torch").equal(
            pd.index_select(1, index.to(pd.device)), down_weight
        )
    )
    return PermutationAudit(len(perm), maximum, relative, exact_inverse, exact_weights)
