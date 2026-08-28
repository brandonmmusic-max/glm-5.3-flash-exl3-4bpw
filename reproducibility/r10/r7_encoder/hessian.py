"""Streaming full covariance construction, including down via gate/up RT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FinalizedCovariance:
    matrix: Any
    rows: int
    weight_sum: float
    diag_mean: float
    sigma_reg: float
    guided: bool


class FullCovarianceAccumulator:
    """Accumulate a full, unsliced covariance on an explicit device."""

    def __init__(self, dimension: int, *, device: str, guided: bool = False) -> None:
        import torch

        if dimension <= 0 or dimension % 16:
            raise ValueError("covariance dimension must be positive and 16-aligned")
        self.dimension = dimension
        self.device = device
        self.guided = guided
        self.matrix = torch.zeros(
            (dimension, dimension), dtype=torch.float64, device=device
        )
        self.rows = 0
        self.weight_sum = 0.0

    def add(self, activations, row_weights=None) -> None:
        import torch

        x = torch.as_tensor(activations, device=self.device, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(
                f"activations must be [rows,{self.dimension}], got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("non-finite covariance activations")
        if row_weights is None:
            if self.guided:
                raise ValueError("guided covariance requires row weights")
            scaled = x
            weight_sum = float(x.shape[0])
        else:
            weights = torch.as_tensor(
                row_weights, device=self.device, dtype=torch.float64
            ).flatten()
            if weights.shape[0] != x.shape[0]:
                raise ValueError("row-weight length mismatch")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("row weights must be finite and nonnegative")
            scaled = x * weights.sqrt().unsqueeze(1)
            weight_sum = float(weights.sum().item())
        self.matrix.addmm_(scaled.T, scaled)
        self.rows += int(x.shape[0])
        self.weight_sum += weight_sum

    def merge_shrinkage(self, fallback, alpha: float) -> None:
        import torch

        if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
            raise ValueError("shrinkage alpha must be finite in [0,1]")
        other = torch.as_tensor(fallback, device=self.device, dtype=torch.float64)
        if tuple(other.shape) != (self.dimension, self.dimension):
            raise ValueError("fallback covariance shape mismatch")
        self.matrix.mul_(1.0 - alpha).add_(other, alpha=alpha)

    def finalize(
        self, sigma_reg: float, *, add_damping: bool = False
    ) -> FinalizedCovariance:
        import torch

        if self.rows <= 0 or self.weight_sum <= 0:
            raise ValueError("cannot finalize empty covariance")
        if not math.isfinite(sigma_reg) or sigma_reg < 0:
            raise ValueError("sigma_reg must be finite and nonnegative")
        covariance = self.matrix / self.weight_sum
        covariance = (covariance + covariance.T) * 0.5
        diagonal = covariance.diagonal()
        diag_mean = float(diagonal.mean().item())
        if not math.isfinite(diag_mean) or diag_mean <= 1e-20:
            raise ValueError("degenerate covariance; identity fallback is forbidden")
        if add_damping:
            diagonal.add_(sigma_reg * diag_mean)
        if not torch.isfinite(covariance).all():
            raise ValueError("non-finite finalized covariance")
        return FinalizedCovariance(
            matrix=covariance.to(dtype=torch.float32),
            rows=self.rows,
            weight_sum=self.weight_sum,
            diag_mean=diag_mean,
            sigma_reg=sigma_reg,
            guided=self.guided,
        )


def down_inputs_from_roundtrip(x, gate_kn_rt, up_kn_rt):
    """Form full 2048-coordinate SwiGLU inputs from decoded gate/up matrices."""

    import torch
    import torch.nn.functional as functional

    x = torch.as_tensor(x)
    gate = torch.as_tensor(gate_kn_rt, device=x.device, dtype=x.dtype)
    up = torch.as_tensor(up_kn_rt, device=x.device, dtype=x.dtype)
    if gate.ndim != 2 or gate.shape != up.shape or gate.shape[0] != x.shape[1]:
        raise ValueError(
            f"round-trip matrices must share [K,N] with K={x.shape[1]}; "
            f"gate={tuple(gate.shape)} up={tuple(up.shape)}"
        )
    result = functional.silu(x @ gate) * (x @ up)
    if not torch.isfinite(result).all():
        raise ValueError("non-finite reconstructed SwiGLU activations")
    return result


def least_squares_output_scale(reference, reconstructed, eps: float = 1e-20):
    """Fit per-output scale suitable for folding into `svh`."""

    import torch

    y = torch.as_tensor(reference, dtype=torch.float64)
    yhat = torch.as_tensor(reconstructed, device=y.device, dtype=torch.float64)
    if y.shape != yhat.shape or y.ndim != 2:
        raise ValueError("least-squares inputs must have identical [rows,N] shape")
    numerator = (yhat * y).sum(dim=0)
    denominator = yhat.square().sum(dim=0).clamp_min(eps)
    scale = numerator / denominator
    if not torch.isfinite(scale).all():
        raise ValueError("least-squares scale is non-finite")
    return scale
