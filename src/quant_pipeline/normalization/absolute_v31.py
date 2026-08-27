"""Topology-preserving absolute normalization for the R10 recovery.

This module restores the amplitude-setting part of the proven v31
regularizer without changing the R7 serving topology:

* gate/up share one residual-input ``suh`` and keep private ``svh``;
* down keeps private ``suh`` and shares one residual-output ``svh``;
* the existing separable per-block factors remain folded into those vectors;
* every matrix/bit-specific GSS scalar is folded into a *private* vector.

The numerical primitives are supplied by the caller and are required to be
the pinned v31 primitives (``block_rms``, block Hadamards and their inverse
forms).  Keeping that dependency explicit lets the recovery driver seal the
same numeric-core hash used by TRELLIS instead of silently growing another
Hadamard/RMS implementation here.

The frozen control has staged FP16 materializations.  In particular, gate/up
relative ``svh`` is rounded before row-RMS fitting, its beta-adjusted private
vector is rounded again, and the private vector is rounded after GSS folding.
Down shared ``svh`` is rounded before row-RMS fitting, while its private
``suh`` is rounded before and after GSS folding.  Regularization and decode
consume the final stored FP16 values.  No quantizer, model, checkpoint writer,
network, or GPU orchestration is invoked by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


SCHEMA = "r10-topology-absolute-normalization-v1"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GATE_UP = frozenset(("gate_proj", "up_proj"))
ALLOWED_BITS = frozenset((3, 4, 5))
DEFAULT_BLOCK = 128
_EPSILON = 1e-30


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash checkpoint-visible tensor bytes, including dtype and order."""

    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def tensor_identity_sha256(value: torch.Tensor) -> str:
    """Hash bytes with dtype and shape; historical raw-byte hashes remain stable."""

    tensor = value.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + _tensor_bytes(tensor)).hexdigest()


def _positive_scalar(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive, not boolean")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _positive_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError(f"{label} must be finite and positive")
    return value


def _validate_signs(value: Any, length: int, label: str, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if tuple(result.shape) != (length,):
        raise ValueError(f"{label} has shape {tuple(result.shape)}, expected {(length,)}")
    if not torch.all((result == 1.0) | (result == -1.0)):
        raise ValueError(f"{label} must contain only exact Rademacher signs")
    return result


def _expand_block_scales(
    scales: Sequence[float], length: int, block: int, label: str, device: torch.device
) -> torch.Tensor:
    if block <= 0 or length % block:
        raise ValueError(f"{label}: length {length} is not divisible by block {block}")
    # The deployed search artifact materialized every factor through Python
    # float division and only then crossed into FP32.  Keep the supplied
    # factors in float64 until that division so a recovered frozen decision is
    # not perturbed by an earlier FP32 factor rounding (observably one ULP for
    # some inverse/per128 families).
    values = torch.as_tensor(tuple(scales), dtype=torch.float64, device=device).flatten()
    if tuple(values.shape) != (length // block,):
        raise ValueError(
            f"{label} has {values.numel()} blocks, expected {length // block}"
        )
    _positive_tensor(values, label)
    return values.repeat_interleave(block)


def _base_from_sign_and_scale(
    signs: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Reproduce deployed Python-f64 division, then materialize FP32."""

    sign_values = signs.detach().cpu().tolist()
    scale_values = scales.detach().cpu().tolist()
    values = [
        float(sign) / float(scale)
        for sign, scale in zip(sign_values, scale_values, strict=True)
    ]
    return torch.tensor(values, dtype=torch.float32, device=signs.device).contiguous()


def _store_fp16(value: torch.Tensor, label: str) -> torch.Tensor:
    """Apply and validate the exact checkpoint vector boundary."""

    candidate = torch.as_tensor(value, dtype=torch.float32, device=value.device).flatten()
    if not torch.isfinite(candidate).all() or (candidate == 0).any():
        raise ValueError(f"{label} is non-finite or non-invertible before FP16 storage")
    stored = candidate.to(torch.float16)
    if not torch.isfinite(stored).all():
        raise ValueError(f"{label} overflows FP16 storage")
    if (stored == 0).any():
        raise ValueError(f"{label} underflows FP16 storage")
    return stored.contiguous()


def _weighted_log_mean(values: Sequence[torch.Tensor], masses: Sequence[float]) -> torch.Tensor:
    if not values or len(values) != len(masses):
        raise ValueError("weighted log mean requires matching nonempty inputs")
    denominator = sum(_positive_scalar(mass, "matrix mass") for mass in masses)
    result = torch.zeros_like(values[0], dtype=torch.float64)
    for value, mass in zip(values, masses):
        if tuple(value.shape) != tuple(values[0].shape):
            raise ValueError("weighted log mean shape mismatch")
        _positive_tensor(value, "absolute scale")
        result.add_(value.double().log(), alpha=float(mass) / denominator)
    return result.exp().float()


def _relative_output_rms(core: Any, weight_kn: torch.Tensor) -> torch.Tensor:
    """Proven v31 output-channel RMS, normalized to arithmetic mean one."""

    scale = core.block_rms(weight_kn, dim=0, keepdim=True).flatten().float()
    _positive_tensor(scale, "output-channel RMS")
    mean = float(scale.mean().item())
    if not math.isfinite(mean) or mean <= _EPSILON:
        raise ValueError("output-channel RMS has a degenerate mean")
    return (scale / mean).clamp_min(_EPSILON)


def _right_transform(core: Any, weight_kn: torch.Tensor, stored_svh: torch.Tensor, block: int) -> torch.Tensor:
    value = weight_kn.clone().float()
    value.div_(stored_svh.float().unsqueeze(0))
    core.blockwise_preapply_had_r_(value, block)
    return value


def regularize_from_stored(
    core: Any,
    weight_kn: torch.Tensor,
    stored_suh: torch.Tensor,
    stored_svh: torch.Tensor,
    *,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Regularize with exactly the FP16 vectors visible to the loader."""

    if stored_suh.dtype != torch.float16 or stored_svh.dtype != torch.float16:
        raise TypeError("regularize_from_stored requires checkpoint FP16 vectors")
    if tuple(weight_kn.shape) != (stored_suh.numel(), stored_svh.numel()):
        raise ValueError("weight/vector geometry mismatch")
    value = _right_transform(core, weight_kn, stored_svh, block)
    value.div_(stored_suh.float().unsqueeze(1))
    core.blockwise_preapply_had_l_(value, block)
    return value


def decode_with_stored(
    core: Any,
    regularized: torch.Tensor,
    stored_suh: torch.Tensor,
    stored_svh: torch.Tensor,
    *,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Apply the serving-side inverse using the same FP16 vector bytes."""

    if stored_suh.dtype != torch.float16 or stored_svh.dtype != torch.float16:
        raise TypeError("decode_with_stored requires checkpoint FP16 vectors")
    if tuple(regularized.shape) != (stored_suh.numel(), stored_svh.numel()):
        raise ValueError("regularized/vector geometry mismatch")
    value = core.preapply_had_l(regularized.float(), block).float()
    value.mul_(stored_suh.float().unsqueeze(1))
    value = core.preapply_had_r(value, block).float()
    value.mul_(stored_svh.float().unsqueeze(0))
    return value


@dataclass(frozen=True)
class MatrixInput:
    """One source matrix plus the selected R10 signs/block factors.

    ``weight_kn`` uses EXL3's internal ``[in_features, out_features]`` order.
    Block factors are the existing R10 regularized-weight multipliers.  They
    are folded by dividing the matching stored vector, exactly as in
    ``r7_encoder.rotations.fold_block_g_scale``.
    """

    key: str
    projection: str
    bits: int
    weight_kn: torch.Tensor
    suh_sign: Any
    svh_sign: Any
    k_block_scales: Sequence[float]
    n_block_scales: Sequence[float]
    mass: float = 1.0


@dataclass(frozen=True)
class _PreparedMatrix:
    source: MatrixInput
    weight_kn: torch.Tensor
    base_suh: torch.Tensor
    base_svh: torch.Tensor
    output_scale: torch.Tensor


@dataclass(frozen=True)
class FinalizedMatrix:
    key: str
    projection: str
    bits: int
    g_scale: float
    gss_fold_side: str
    private_amplitude: float
    source_weight_identity_sha256: str
    stored_suh: torch.Tensor
    stored_svh: torch.Tensor
    regularized: torch.Tensor | None

    @property
    def suh_sha256(self) -> str:
        return tensor_sha256(self.stored_suh)

    @property
    def svh_sha256(self) -> str:
        return tensor_sha256(self.stored_svh)


@dataclass
class AbsoluteNormalizationFit:
    """Source-derived layer fit before matrix-specific GSS is folded."""

    core: Any
    block: int
    codebook_scale: float
    matrices: dict[str, _PreparedMatrix]
    shared_gate_up_suh: torch.Tensor
    shared_down_svh: torch.Tensor
    pre_gss_suh: dict[str, torch.Tensor]
    pre_gss_svh: dict[str, torch.Tensor]
    private_amplitudes: dict[str, float]

    def gss_targets(self) -> dict[str, torch.Tensor]:
        """Return exact-boundary targets on which v31 GSS must run.

        The caller runs the pinned v31 scalar search independently for each
        key at that key's selected bit width, then passes every result to
        :meth:`finalize`.  Pooling kernel launches is valid; sharing the scalar
        result between matrices is not.
        """

        return {
            key: regularize_from_stored(
                self.core,
                prepared.weight_kn,
                self.pre_gss_suh[key],
                self.pre_gss_svh[key],
                block=self.block,
            )
            for key, prepared in self.matrices.items()
        }

    def finalize(self, g_scales: Mapping[str, float]) -> "FinalizedLayerNormalization":
        """Fold every mixed-bit GSS scalar into its matrix-private side."""

        expected = set(self.matrices)
        supplied = set(g_scales)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"GSS map must cover every matrix; missing={missing}, extra={extra}")

        finalized: dict[str, FinalizedMatrix] = {}
        for key, prepared in self.matrices.items():
            g_scale = _positive_scalar(g_scales[key], f"{key} GSS scale")
            if prepared.source.projection in GATE_UP:
                # SU is layer-shared.  A scalar commutes with both H128 blocks,
                # so the v31 `su /= g` fold moves exactly to private SV.
                stored_suh = self.shared_gate_up_suh
                stored_svh = _store_fp16(
                    self.pre_gss_svh[key].float() / g_scale,
                    f"{key} private svh after GSS",
                )
                fold_side = "svh"
            else:
                # SV is layer-shared; keep the scalar on private down SU.
                stored_suh = _store_fp16(
                    self.pre_gss_suh[key].float() / g_scale,
                    f"{key} private suh after GSS",
                )
                stored_svh = self.shared_down_svh
                fold_side = "suh"
            regularized = regularize_from_stored(
                self.core,
                prepared.weight_kn,
                stored_suh,
                stored_svh,
                block=self.block,
            )
            finalized[key] = FinalizedMatrix(
                key=key,
                projection=prepared.source.projection,
                bits=prepared.source.bits,
                g_scale=g_scale,
                gss_fold_side=fold_side,
                private_amplitude=self.private_amplitudes[key],
                source_weight_identity_sha256=tensor_identity_sha256(
                    prepared.weight_kn
                ),
                stored_suh=stored_suh,
                stored_svh=stored_svh,
                regularized=regularized,
            )

        result = FinalizedLayerNormalization(
            schema=SCHEMA,
            matrices=finalized,
            shared_gate_up_suh_sha256=tensor_sha256(self.shared_gate_up_suh),
            shared_down_svh_sha256=tensor_sha256(self.shared_down_svh),
        )
        result.assert_topology()
        return result


@dataclass(frozen=True)
class FinalizedLayerNormalization:
    schema: str
    matrices: Mapping[str, FinalizedMatrix]
    shared_gate_up_suh_sha256: str
    shared_down_svh_sha256: str

    def assert_topology(self) -> None:
        gate_hashes = {
            value.suh_sha256
            for value in self.matrices.values()
            if value.projection in GATE_UP
        }
        down_hashes = {
            value.svh_sha256
            for value in self.matrices.values()
            if value.projection == "down_proj"
        }
        if gate_hashes != {self.shared_gate_up_suh_sha256}:
            raise AssertionError("gate/up shared SU bytes drifted")
        if down_hashes != {self.shared_down_svh_sha256}:
            raise AssertionError("down shared SV bytes drifted")
        if any(
            value.gss_fold_side != ("svh" if value.projection in GATE_UP else "suh")
            for value in self.matrices.values()
        ):
            raise AssertionError("a GSS scalar was folded onto a shared side")


def fit_layer_absolute_normalization(
    core: Any,
    inputs: Sequence[MatrixInput],
    *,
    codebook_scale: float,
    block: int = DEFAULT_BLOCK,
) -> AbsoluteNormalizationFit:
    """Fit one layer's absolute scales without relaxing shared topology.

    Gate/up use a mass-weighted geometric mean of the post-right-Hadamard row
    RMS vectors.  The per-matrix geometric residual is stored as one private
    ``svh`` scalar.  This two-way log decomposition makes the regularized
    targets invariant to independently rescaling any source matrix (apart
    from unavoidable FP16 rounding).

    Down uses a mass-weighted geometric mean of v31's relative output RMS for
    the shared ``svh``.  Its exact absolute post-right-Hadamard row RMS remains
    private in each matrix's ``suh``.
    """

    codebook = _positive_scalar(codebook_scale, "codebook scale")
    if block <= 0:
        raise ValueError("block must be positive")
    required = (
        "block_rms",
        "blockwise_preapply_had_l_",
        "blockwise_preapply_had_r_",
        "preapply_had_l",
        "preapply_had_r",
    )
    missing = [name for name in required if not callable(getattr(core, name, None))]
    if missing:
        raise TypeError(f"numeric core lacks pinned v31 primitives: {missing}")

    # Prepare inline so output RMS is computed by the bound v31 core.
    prepared: dict[str, _PreparedMatrix] = {}
    for item in inputs:
        if not item.key or item.key in prepared:
            raise ValueError(f"matrix keys must be nonempty and unique: {item.key!r}")
        if item.projection not in PROJECTIONS:
            raise ValueError(f"{item.key}: unsupported projection {item.projection!r}")
        if item.bits not in ALLOWED_BITS:
            raise ValueError(f"{item.key}: bits must be one of {sorted(ALLOWED_BITS)}")
        _positive_scalar(item.mass, f"{item.key} mass")
        weight = torch.as_tensor(item.weight_kn, dtype=torch.float32).contiguous()
        if weight.ndim != 2 or min(weight.shape) <= 0:
            raise ValueError(f"{item.key}: weight must be a nonempty matrix")
        if not torch.isfinite(weight).all() or float(weight.square().sum().item()) <= _EPSILON:
            raise ValueError(f"{item.key}: weight is non-finite or degenerate")
        k, n = weight.shape
        su_sign = _validate_signs(item.suh_sign, k, f"{item.key} suh signs", weight.device)
        sv_sign = _validate_signs(item.svh_sign, n, f"{item.key} svh signs", weight.device)
        k_scale = _expand_block_scales(
            item.k_block_scales, k, block, f"{item.key} K block factors", weight.device
        )
        n_scale = _expand_block_scales(
            item.n_block_scales, n, block, f"{item.key} N block factors", weight.device
        )
        prepared[item.key] = _PreparedMatrix(
            source=item,
            weight_kn=weight,
            base_suh=_base_from_sign_and_scale(su_sign, k_scale),
            base_svh=_base_from_sign_and_scale(sv_sign, n_scale),
            output_scale=_relative_output_rms(core, weight),
        )

    gate_up = [value for value in prepared.values() if value.source.projection in GATE_UP]
    down = [value for value in prepared.values() if value.source.projection == "down_proj"]
    if not gate_up or not down:
        raise ValueError("a layer fit requires both gate/up and down matrices")
    gate_k = gate_up[0].weight_kn.shape[0]
    down_n = down[0].weight_kn.shape[1]
    if any(value.weight_kn.shape[0] != gate_k for value in gate_up):
        raise ValueError("gate/up shared SU geometry differs")
    if any(value.weight_kn.shape[1] != down_n for value in down):
        raise ValueError("down shared SV geometry differs")
    if any(not torch.equal(value.base_suh, gate_up[0].base_suh) for value in gate_up[1:]):
        raise ValueError("gate/up input signs or K-block factors are not layer-shared")
    if any(not torch.equal(value.base_svh, down[0].base_svh) for value in down[1:]):
        raise ValueError("down output signs or N-block factors are not layer-shared")

    # Gate/up: private relative output scales first, at their real FP16
    # boundary, then fit the one shared absolute input profile.
    gate_relative_svh: dict[str, torch.Tensor] = {}
    gate_row_rms: dict[str, torch.Tensor] = {}
    for value in gate_up:
        stored = _store_fp16(
            value.base_svh * value.output_scale,
            f"{value.source.key} relative private svh",
        )
        gate_relative_svh[value.source.key] = stored
        right = _right_transform(core, value.weight_kn, stored, block)
        gate_row_rms[value.source.key] = _positive_tensor(
            core.block_rms(right, dim=1, keepdim=True).flatten().float(),
            f"{value.source.key} post-right-Hadamard row RMS",
        )
    shared_gate_magnitude = _weighted_log_mean(
        [gate_row_rms[value.source.key] for value in gate_up],
        [value.source.mass for value in gate_up],
    )
    shared_gate_suh = _store_fp16(
        gate_up[0].base_suh * shared_gate_magnitude / (-codebook),
        "layer-shared gate/up absolute suh",
    )

    # Recover the effective magnitude after FP16 storage.  Beta is the
    # geometric scalar residual per matrix, placed on private SV.
    effective_shared_gate_magnitude = (
        shared_gate_suh.float() * (-codebook) / gate_up[0].base_suh
    ).abs()
    _positive_tensor(effective_shared_gate_magnitude, "stored shared gate/up magnitude")
    pre_su: dict[str, torch.Tensor] = {}
    pre_sv: dict[str, torch.Tensor] = {}
    private_amplitudes: dict[str, float] = {}
    for value in gate_up:
        log_ratio = (
            gate_row_rms[value.source.key].double().log()
            - effective_shared_gate_magnitude.double().log()
        )
        beta = _positive_scalar(float(log_ratio.mean().exp().item()), "gate/up beta")
        pre_su[value.source.key] = shared_gate_suh
        pre_sv[value.source.key] = _store_fp16(
            gate_relative_svh[value.source.key].float() * beta,
            f"{value.source.key} private svh amplitude",
        )
        private_amplitudes[value.source.key] = beta

    # Down: shared relative output profile, normalized back to mean one so it
    # carries shape rather than an arbitrary common unit.  Absolute row RMS is
    # then exact and private for every down matrix.
    shared_down_output = _weighted_log_mean(
        [value.output_scale for value in down], [value.source.mass for value in down]
    )
    shared_down_output.div_(shared_down_output.mean())
    shared_down_svh = _store_fp16(
        down[0].base_svh * shared_down_output,
        "layer-shared down relative svh",
    )
    for value in down:
        right = _right_transform(core, value.weight_kn, shared_down_svh, block)
        row_rms = _positive_tensor(
            core.block_rms(right, dim=1, keepdim=True).flatten().float(),
            f"{value.source.key} absolute down row RMS",
        )
        pre_su[value.source.key] = _store_fp16(
            value.base_suh * row_rms / (-codebook),
            f"{value.source.key} private absolute suh",
        )
        pre_sv[value.source.key] = shared_down_svh
        private_amplitudes[value.source.key] = 1.0

    return AbsoluteNormalizationFit(
        core=core,
        block=block,
        codebook_scale=codebook,
        matrices=prepared,
        shared_gate_up_suh=shared_gate_suh,
        shared_down_svh=shared_down_svh,
        pre_gss_suh=pre_su,
        pre_gss_svh=pre_sv,
        private_amplitudes=private_amplitudes,
    )
