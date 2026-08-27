from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def gauss_legendre_nodes(count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("path node count must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (nodes + 1.0) / 2.0, weights / 2.0


def aumann_shapley(
    deltas: Sequence[np.ndarray],
    gradient_at: Callable[[float], Sequence[np.ndarray]],
    path_nodes: int = 5,
) -> np.ndarray:
    """Integrate <gradient_i(t), delta_i> along a common quantization path."""
    nodes, weights = gauss_legendre_nodes(path_nodes)
    result = np.zeros(len(deltas), dtype=np.float64)
    for node, quadrature_weight in zip(nodes, weights, strict=True):
        gradients = gradient_at(float(node))
        if len(gradients) != len(deltas):
            raise ValueError("gradient_at returned the wrong number of units")
        for index, (gradient, delta) in enumerate(zip(gradients, deltas, strict=True)):
            result[index] += quadrature_weight * float(np.vdot(gradient, delta).real)
    return result


def quadratic_expert_attribution(
    projected_residuals: np.ndarray,
    observation_weights: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Signed expert shares that close exactly to 0.5*||sum(z_e)||^2.

    Each z_e may already include the downstream Fisher/Jacobian square-root
    projection. Cross-expert terms are shared symmetrically by the identity
    psi_e = 0.5 <z_e, sum_j z_j>.
    """
    z = np.asarray(projected_residuals, dtype=np.float64)
    if z.ndim < 2:
        raise ValueError("expected [experts, observations...] residuals")
    total = np.sum(z, axis=0)
    observations = z.shape[1:]
    if observation_weights is None:
        weights = np.ones(observations, dtype=np.float64)
    else:
        supplied = np.asarray(observation_weights, dtype=np.float64)
        if supplied.ndim == 1 and len(observations) > 1 and supplied.shape[0] == observations[0]:
            supplied = supplied.reshape((observations[0],) + (1,) * (len(observations) - 1))
        try:
            weights = np.broadcast_to(supplied, observations).astype(np.float64, copy=False)
        except ValueError as error:
            raise ValueError("observation weights do not broadcast to projected residuals") from error
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("observation weights must be finite and nonnegative")
    mass = float(np.sum(weights))
    if mass <= 0.0:
        raise ValueError("observation weights must have positive mass")
    normalized = weights / mass
    axes = tuple(range(1, z.ndim))
    return 0.5 * np.sum(z * total * normalized, axis=axes)


def conditional_quadratic_damage(current: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Forecast 0.5||R+d||^2 - 0.5||R||^2 for projected candidates."""
    residual = np.asarray(current, dtype=np.float64)
    deltas = np.asarray(candidates, dtype=np.float64)
    if deltas.ndim < residual.ndim + 1 or deltas.shape[1:] != residual.shape:
        raise ValueError("candidates must have shape [candidate, *current.shape]")
    axes = tuple(range(1, deltas.ndim))
    return np.mean(deltas * residual, axis=axes) + 0.5 * np.mean(deltas * deltas, axis=axes)


@dataclass(frozen=True)
class Reconciliation:
    raw: np.ndarray
    reconciled: np.ndarray
    measured_total: float
    raw_total: float
    closure_residual: float
    method: str


def reconcile_signed_completeness(
    raw: Sequence[float],
    measured_total: float,
    *,
    minimum_relative_total: float = 1e-12,
) -> Reconciliation:
    """Rescale signed directional shares to an independently measured total.

    The scale is explicit and preserves every signed ratio.  A nearly
    cancelling raw total fails closed because proportional reconciliation
    would otherwise amplify numerical noise into arbitrary attribution.
    """
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("signed completeness reconciliation requires finite 1D shares")
    measured = float(measured_total)
    if not np.isfinite(measured):
        raise ValueError("signed completeness reconciliation requires a finite measured total")
    raw_total = float(np.sum(values))
    magnitude = float(np.sum(np.abs(values)))
    if magnitude == 0.0 or abs(raw_total) <= minimum_relative_total * magnitude:
        raise ValueError("signed completeness reconciliation raw total is numerically singular")
    scale = measured / raw_total
    reconciled = values * scale
    # Put the final floating-point ulp on the largest share so serialized
    # values close exactly under the same summation order.
    index = int(np.argmax(np.abs(reconciled)))
    reconciled[index] += measured - float(np.sum(reconciled))
    return Reconciliation(
        raw=values,
        reconciled=reconciled,
        measured_total=measured,
        raw_total=raw_total,
        closure_residual=measured - raw_total,
        method="signed-proportional-completeness",
    )


def reconcile_explicit_remainder(raw: Sequence[float], measured_total: float) -> Reconciliation:
    """Keep proxy values untouched and expose non-closure as its own component."""
    values = np.asarray(raw, dtype=np.float64)
    residual = float(measured_total - np.sum(values))
    reconciled = np.concatenate([values, np.asarray([residual])])
    return Reconciliation(
        raw=values,
        reconciled=reconciled,
        measured_total=float(measured_total),
        raw_total=float(np.sum(values)),
        closure_residual=residual,
        method="explicit-unresolved-remainder",
    )


def reconcile_layer_components_for_allocation(
    *,
    expert_direct: Sequence[float],
    routing_state_shift: float,
    explicit_residual: float,
    raw_layer_damage: float,
    reconciled_layer_damage: float,
) -> dict:
    """Preserve component closure and derive a separately labelled expert score.

    Direct expert evidence, routing/backend shift, and the explicit residual
    are all scaled by the same independently measured layer-completeness
    factor.  The non-expert components remain visible.  Only the allocation
    score redistributes them over experts, by absolute direct magnitude (or
    uniformly when every direct term is zero), so that an expert-only
    optimizer has a closing objective without disguising the redistribution
    as measured expert evidence.
    """

    direct_raw = np.asarray(expert_direct, dtype=np.float64)
    if direct_raw.ndim != 1 or not len(direct_raw) or not np.isfinite(direct_raw).all():
        raise ValueError("layer component reconciliation requires finite expert-direct values")
    raw_layer = float(raw_layer_damage)
    reconciled_layer = float(reconciled_layer_damage)
    route_raw = float(routing_state_shift)
    residual_raw = float(explicit_residual)
    if not all(np.isfinite(value) for value in (raw_layer, reconciled_layer, route_raw, residual_raw)):
        raise ValueError("layer component reconciliation requires finite scalar components")
    tolerance = max(1e-15, abs(raw_layer) * 1e-12)
    if abs(float(np.sum(direct_raw)) + route_raw + residual_raw - raw_layer) > tolerance:
        raise ValueError("raw expert, route, and residual components do not close to raw layer damage")
    if raw_layer == 0.0:
        if reconciled_layer != 0.0:
            raise ValueError("zero raw layer damage cannot reconcile to a nonzero layer share")
        scale = 0.0
    else:
        scale = reconciled_layer / raw_layer
    direct = direct_raw * scale
    route = route_raw * scale
    residual = residual_raw * scale
    nonexpert = route + residual
    magnitude = np.abs(direct)
    if float(np.sum(magnitude)) > 0.0:
        redistribution = magnitude / float(np.sum(magnitude))
        policy = "absolute-direct-magnitude-proportional-v1"
    else:
        redistribution = np.full(len(direct), 1.0 / len(direct))
        policy = "uniform-when-direct-magnitude-is-zero-v1"
    allocation = direct + nonexpert * redistribution
    allocation[int(np.argmax(np.abs(allocation)))] += reconciled_layer - float(np.sum(allocation))
    return {
        "layer_completeness_scale": scale,
        "expert_direct_reconciled": direct.tolist(),
        "reconciled_expert_direct_total": float(np.sum(direct)),
        "routing_state_shift_reconciled": route,
        "within_layer_unresolved_remainder_reconciled": residual,
        "reconciled_component_total": float(np.sum(direct) + route + residual),
        "expert_allocation_score_reconciled": allocation.tolist(),
        "expert_redistribution_policy": policy,
        "redistributed_nonexpert_total": nonexpert,
        "reconciled_expert_total": float(np.sum(allocation)),
    }


def split_layer_damage(
    measured_layer_damage: float,
    projected_expert_residuals: np.ndarray,
    routing_state_shift: float = 0.0,
    projected_routing_residual: np.ndarray | None = None,
    observation_weights: np.ndarray | Sequence[float] | None = None,
) -> dict:
    expert = np.asarray(projected_expert_residuals, dtype=np.float64)
    if projected_routing_residual is not None:
        routing = np.asarray(projected_routing_residual, dtype=np.float64)
        if routing.shape != expert.shape[1:]:
            raise ValueError("projected routing residual must match one expert residual observation shape")
        joint = np.concatenate([expert, routing[None]], axis=0)
        shares = quadratic_expert_attribution(joint, observation_weights)
        direct = shares[:-1]
        routing_state_shift = float(shares[-1])
    else:
        direct = quadratic_expert_attribution(expert, observation_weights)
    raw = np.concatenate([direct, np.asarray([routing_state_shift], dtype=np.float64)])
    accounting = reconcile_explicit_remainder(raw, measured_layer_damage)
    return {
        "expert_direct": direct.tolist(),
        "routing_state_shift": float(routing_state_shift),
        "unresolved_nonlinear_remainder": accounting.closure_residual,
        "component_policy": "direct-plus-route-plus-explicit-residual-v1",
        "raw_total": accounting.raw_total,
        "measured_layer_damage": accounting.measured_total,
        "closed_total": float(np.sum(accounting.reconciled)),
    }
