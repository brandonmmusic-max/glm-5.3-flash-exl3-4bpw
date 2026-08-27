"""Memory-bounded counterpart to :mod:`absolute_normalization_v1`.

The authoritative batch implementation is intentionally kept as the numeric
specification.  This module reproduces that specification in two bounded
stages:

1. a chosen mass-weighted fit sample is streamed once to seal the layer-shared
   gate/up ``suh`` and down ``svh`` vectors;
2. source matrices are prepared, GSS-scored, finalized, and released one at a
   time.

The fit retains two running float64 log accumulators and the final shared
vectors, never a source matrix or a per-matrix RMS vector.  The caller supplies
lightweight sample metadata up front so the mass denominator is known before
the streaming pass.  That permits the exact same ordered ``add_(alpha=...)``
operations as the batch implementation, including its staged FP16 storage
boundaries.

Every GSS result is bound to both the matrix key and selected K3/K4/K5 width.
Raw scalars and results for another key or bit width fail closed.  GSS is folded
only into private ``svh`` for gate/up and private ``suh`` for down; a shared
vector is never rewritten after the fit is sealed.

No checkpoint writer, model loader, quantizer, network, or GPU orchestration is
invoked here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

import torch

from .absolute_v31 import (
    ALLOWED_BITS,
    DEFAULT_BLOCK,
    GATE_UP,
    PROJECTIONS,
    SCHEMA as BATCH_SCHEMA,
    FinalizedMatrix,
    MatrixInput,
    _PreparedMatrix,
    _base_from_sign_and_scale,
    _expand_block_scales,
    _positive_scalar,
    _positive_tensor,
    _relative_output_rms,
    _right_transform,
    _store_fp16,
    _validate_signs,
    regularize_from_stored,
    tensor_identity_sha256,
    tensor_sha256,
)


SCHEMA = "r10-topology-absolute-normalization-streaming-v1"
_EPSILON = 1e-30


def _sealed_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_core(core: Any) -> None:
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


def _shape_of(weight: Any, label: str) -> tuple[int, int]:
    shape = tuple(getattr(weight, "shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{label}: weight must be a nonempty matrix")
    return int(shape[0]), int(shape[1])


def _base_vectors(
    item: MatrixInput,
    *,
    block: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    k, n = _shape_of(item.weight_kn, item.key)
    su_sign = _validate_signs(item.suh_sign, k, f"{item.key} suh signs", device)
    sv_sign = _validate_signs(item.svh_sign, n, f"{item.key} svh signs", device)
    k_scale = _expand_block_scales(
        item.k_block_scales, k, block, f"{item.key} K block factors", device
    )
    n_scale = _expand_block_scales(
        item.n_block_scales, n, block, f"{item.key} N block factors", device
    )
    return (
        _base_from_sign_and_scale(su_sign, k_scale),
        _base_from_sign_and_scale(sv_sign, n_scale),
    )


def _prepare_one(core: Any, item: MatrixInput, block: int) -> _PreparedMatrix:
    if not item.key:
        raise ValueError("matrix key must be nonempty")
    if item.projection not in PROJECTIONS:
        raise ValueError(f"{item.key}: unsupported projection {item.projection!r}")
    if item.bits not in ALLOWED_BITS:
        raise ValueError(f"{item.key}: bits must be one of {sorted(ALLOWED_BITS)}")
    _positive_scalar(item.mass, f"{item.key} mass")
    weight = torch.as_tensor(item.weight_kn, dtype=torch.float32).contiguous()
    _shape_of(weight, item.key)
    if not torch.isfinite(weight).all() or float(weight.square().sum().item()) <= _EPSILON:
        raise ValueError(f"{item.key}: weight is non-finite or degenerate")
    base_suh, base_svh = _base_vectors(
        item,
        block=block,
        device=weight.device,
    )
    return _PreparedMatrix(
        source=item,
        weight_kn=weight,
        base_suh=base_suh,
        base_svh=base_svh,
        output_scale=_relative_output_rms(core, weight),
    )


@dataclass(frozen=True)
class FitSampleSpec:
    """Weight-free identity for one matrix in the ordered fit sample."""

    key: str
    projection: str
    bits: int
    shape: tuple[int, int]
    mass: float
    weight_identity_sha256: str
    base_suh_sha256: str
    base_svh_sha256: str

    @classmethod
    def from_input(
        cls,
        item: MatrixInput,
        *,
        block: int = DEFAULT_BLOCK,
    ) -> "FitSampleSpec":
        if not item.key:
            raise ValueError("matrix key must be nonempty")
        if item.projection not in PROJECTIONS:
            raise ValueError(f"{item.key}: unsupported projection {item.projection!r}")
        if item.bits not in ALLOWED_BITS:
            raise ValueError(f"{item.key}: bits must be one of {sorted(ALLOWED_BITS)}")
        mass = _positive_scalar(item.mass, f"{item.key} mass")
        shape = _shape_of(item.weight_kn, item.key)
        base_suh, base_svh = _base_vectors(
            item,
            block=block,
            device=torch.device("cpu"),
        )
        return cls(
            key=item.key,
            projection=item.projection,
            bits=item.bits,
            shape=shape,
            mass=mass,
            weight_identity_sha256=tensor_identity_sha256(
                torch.as_tensor(item.weight_kn).contiguous()
            ),
            base_suh_sha256=tensor_sha256(base_suh),
            base_svh_sha256=tensor_sha256(base_svh),
        )


@dataclass(frozen=True)
class FitSamplePlan:
    """Ordered, matrix-free manifest for the chosen fit sample."""

    specs: tuple[FitSampleSpec, ...]
    block: int
    gate_up_mass: float
    down_mass: float
    shared_gate_up_base_sha256: str
    shared_down_base_sha256: str

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": SCHEMA,
                "block": self.block,
                "gate_up_mass": self.gate_up_mass,
                "down_mass": self.down_mass,
                "shared_gate_up_base_sha256": self.shared_gate_up_base_sha256,
                "shared_down_base_sha256": self.shared_down_base_sha256,
                "specs": [
                    {
                        "key": spec.key,
                        "projection": spec.projection,
                        "bits": spec.bits,
                        "shape": list(spec.shape),
                        "mass": spec.mass,
                        "weight_identity_sha256": spec.weight_identity_sha256,
                        "base_suh_sha256": spec.base_suh_sha256,
                        "base_svh_sha256": spec.base_svh_sha256,
                    }
                    for spec in self.specs
                ],
            }
        )

    @classmethod
    def from_specs(
        cls,
        specs: Iterable[FitSampleSpec],
        *,
        block: int = DEFAULT_BLOCK,
    ) -> "FitSamplePlan":
        if block <= 0:
            raise ValueError("block must be positive")
        ordered = tuple(specs)
        if not ordered:
            raise ValueError("fit sample must be nonempty")
        keys = [spec.key for spec in ordered]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("fit sample keys must be nonempty and unique")
        gate = [spec for spec in ordered if spec.projection in GATE_UP]
        down = [spec for spec in ordered if spec.projection == "down_proj"]
        if not gate or not down:
            raise ValueError("a fit sample requires both gate/up and down matrices")
        if any(spec.projection not in PROJECTIONS for spec in ordered):
            raise ValueError("fit sample contains an unsupported projection")
        if any(spec.bits not in ALLOWED_BITS for spec in ordered):
            raise ValueError("fit sample contains an unsupported bit width")
        for spec in ordered:
            _sealed_sha256(spec.weight_identity_sha256, f"{spec.key} weight identity")
            _sealed_sha256(spec.base_suh_sha256, f"{spec.key} base suh identity")
            _sealed_sha256(spec.base_svh_sha256, f"{spec.key} base svh identity")
        if any(
            len(spec.shape) != 2
            or min(spec.shape) <= 0
            or spec.shape[0] % block
            or spec.shape[1] % block
            for spec in ordered
        ):
            raise ValueError(f"every fit-sample dimension must be positive and divisible by {block}")
        if any(spec.shape[0] != gate[0].shape[0] for spec in gate):
            raise ValueError("gate/up shared SU geometry differs")
        if any(spec.shape[1] != down[0].shape[1] for spec in down):
            raise ValueError("down shared SV geometry differs")
        if any(spec.base_suh_sha256 != gate[0].base_suh_sha256 for spec in gate[1:]):
            raise ValueError("gate/up input signs or K-block factors are not layer-shared")
        if any(spec.base_svh_sha256 != down[0].base_svh_sha256 for spec in down[1:]):
            raise ValueError("down output signs or N-block factors are not layer-shared")
        gate_mass = sum(_positive_scalar(spec.mass, f"{spec.key} mass") for spec in gate)
        down_mass = sum(_positive_scalar(spec.mass, f"{spec.key} mass") for spec in down)
        return cls(
            specs=ordered,
            block=block,
            gate_up_mass=gate_mass,
            down_mass=down_mass,
            shared_gate_up_base_sha256=gate[0].base_suh_sha256,
            shared_down_base_sha256=down[0].base_svh_sha256,
        )

    @classmethod
    def from_inputs(
        cls,
        inputs: Iterable[MatrixInput],
        *,
        block: int = DEFAULT_BLOCK,
    ) -> "FitSamplePlan":
        """Build a plan while retaining metadata only, never ``weight_kn``."""

        return cls.from_specs(
            (FitSampleSpec.from_input(item, block=block) for item in inputs),
            block=block,
        )


def _assert_spec_matches(
    spec: FitSampleSpec,
    prepared: _PreparedMatrix,
    *,
    block: int,
) -> None:
    # Recompute the metadata fingerprints on CPU, just as the planning pass
    # did.  That avoids treating a CPU-vs-CUDA division ULP as a metadata
    # change while the actual fit continues on the matrix's native device.
    actual = FitSampleSpec.from_input(prepared.source, block=block)
    if actual != spec:
        raise ValueError(
            f"fit sample replay differs at {spec.key!r}: expected={spec}, actual={actual}"
        )


class StreamingLayerFitter:
    """Consume the planned fit matrices once using O(K + N) fit memory."""

    def __init__(
        self,
        core: Any,
        plan: FitSamplePlan,
        *,
        codebook_scale: float,
        numeric_core_sha256: str,
    ) -> None:
        _required_core(core)
        self.core = core
        self.plan = plan
        self.codebook_scale = _positive_scalar(codebook_scale, "codebook scale")
        self.numeric_core_sha256 = _sealed_sha256(
            numeric_core_sha256, "numeric core identity"
        )
        self._index = 0
        self._gate_log_sum: torch.Tensor | None = None
        self._down_log_sum: torch.Tensor | None = None
        self._gate_base: torch.Tensor | None = None
        self._down_base: torch.Tensor | None = None

    def add_fit_matrix(self, item: MatrixInput) -> None:
        if self._index >= len(self.plan.specs):
            raise ValueError(f"unexpected extra fit matrix {item.key!r}")
        expected = self.plan.specs[self._index]
        prepared = _prepare_one(self.core, item, self.plan.block)
        _assert_spec_matches(expected, prepared, block=self.plan.block)

        if item.projection in GATE_UP:
            relative_svh = _store_fp16(
                prepared.base_svh * prepared.output_scale,
                f"{item.key} relative private svh",
            )
            right = _right_transform(
                self.core,
                prepared.weight_kn,
                relative_svh,
                self.plan.block,
            )
            row_rms = _positive_tensor(
                self.core.block_rms(right, dim=1, keepdim=True).flatten().float(),
                f"{item.key} post-right-Hadamard row RMS",
            )
            if self._gate_log_sum is None:
                self._gate_log_sum = torch.zeros_like(row_rms, dtype=torch.float64)
                self._gate_base = prepared.base_suh.detach().clone()
            elif not torch.equal(prepared.base_suh, self._gate_base):
                raise ValueError("gate/up shared SU values differ during fit replay")
            assert self._gate_log_sum is not None
            self._gate_log_sum.add_(
                row_rms.double().log(),
                alpha=float(item.mass) / self.plan.gate_up_mass,
            )
        else:
            output_scale = prepared.output_scale
            if self._down_log_sum is None:
                self._down_log_sum = torch.zeros_like(output_scale, dtype=torch.float64)
                self._down_base = prepared.base_svh.detach().clone()
            elif not torch.equal(prepared.base_svh, self._down_base):
                raise ValueError("down shared SV values differ during fit replay")
            assert self._down_log_sum is not None
            self._down_log_sum.add_(
                output_scale.double().log(),
                alpha=float(item.mass) / self.plan.down_mass,
            )
        self._index += 1

    def finish(self) -> "StreamingAbsoluteNormalizationFit":
        if self._index != len(self.plan.specs):
            missing = [spec.key for spec in self.plan.specs[self._index :]]
            raise ValueError(f"fit sample replay incomplete; missing={missing}")
        if (
            self._gate_log_sum is None
            or self._down_log_sum is None
            or self._gate_base is None
            or self._down_base is None
        ):
            raise AssertionError("fit sample did not initialize both shared sides")

        shared_gate_magnitude = self._gate_log_sum.exp().float()
        shared_gate_suh = _store_fp16(
            self._gate_base * shared_gate_magnitude / (-self.codebook_scale),
            "layer-shared gate/up absolute suh",
        )
        effective_gate_magnitude = (
            shared_gate_suh.float() * (-self.codebook_scale) / self._gate_base
        ).abs()
        _positive_tensor(effective_gate_magnitude, "stored shared gate/up magnitude")

        shared_down_output = self._down_log_sum.exp().float()
        shared_down_output.div_(shared_down_output.mean())
        shared_down_svh = _store_fp16(
            self._down_base * shared_down_output,
            "layer-shared down relative svh",
        )

        return StreamingAbsoluteNormalizationFit(
            core=self.core,
            block=self.plan.block,
            codebook_scale=self.codebook_scale,
            fit_sample_keys=tuple(spec.key for spec in self.plan.specs),
            shared_gate_up_suh=shared_gate_suh,
            shared_down_svh=shared_down_svh,
            shared_gate_up_base=self._gate_base,
            shared_down_base=self._down_base,
            effective_shared_gate_magnitude=effective_gate_magnitude,
            numeric_core_sha256=self.numeric_core_sha256,
            fit_plan_sha256=self.plan.content_sha256,
        )


@dataclass(frozen=True)
class BitSpecificGSS:
    """A scalar result bound to one exact matrix key and selected bit width."""

    key: str
    bits: int
    scale: float


@dataclass(frozen=True)
class StreamingPreparedMatrix:
    """One live source matrix at its pre-GSS FP16 boundary."""

    fit: "StreamingAbsoluteNormalizationFit" = field(repr=False)
    prepared: _PreparedMatrix = field(repr=False)
    pre_gss_suh: torch.Tensor
    pre_gss_svh: torch.Tensor
    private_amplitude: float

    @property
    def key(self) -> str:
        return self.prepared.source.key

    @property
    def projection(self) -> str:
        return self.prepared.source.projection

    @property
    def bits(self) -> int:
        return self.prepared.source.bits

    def gss_target(self) -> torch.Tensor:
        """Return the exact FP16-boundary target for pinned-v31 GSS."""

        return regularize_from_stored(
            self.fit.core,
            self.prepared.weight_kn,
            self.pre_gss_suh,
            self.pre_gss_svh,
            block=self.fit.block,
        )

    def bind_gss(self, scale: float) -> BitSpecificGSS:
        """Bind a completed scalar search to this matrix and bit width."""

        return BitSpecificGSS(
            key=self.key,
            bits=self.bits,
            scale=_positive_scalar(scale, f"{self.key} GSS scale"),
        )

    def finalize(
        self,
        gss: BitSpecificGSS,
        *,
        materialize_regularized: bool = True,
    ) -> FinalizedMatrix:
        """Fold GSS privately and reproduce batch ``FinalizedMatrix`` bytes.

        The selected-bit pilot sets ``materialize_regularized=False`` because
        its codec immediately regularizes the same source matrix from these
        exact FP16 vectors.  The default retains the authoritative batch and
        unit-test behavior.
        """

        if not isinstance(gss, BitSpecificGSS):
            raise TypeError("finalize requires a BitSpecificGSS result, not a raw scalar")
        if gss.key != self.key:
            raise ValueError(f"GSS key mismatch: expected {self.key!r}, got {gss.key!r}")
        if gss.bits != self.bits:
            raise ValueError(
                f"GSS bit-width mismatch for {self.key}: expected K{self.bits}, got K{gss.bits}"
            )
        scale = _positive_scalar(gss.scale, f"{self.key} GSS scale")
        if self.projection in GATE_UP:
            stored_suh = self.fit.shared_gate_up_suh
            stored_svh = _store_fp16(
                self.pre_gss_svh.float() / scale,
                f"{self.key} private svh after GSS",
            )
            fold_side = "svh"
        else:
            stored_suh = _store_fp16(
                self.pre_gss_suh.float() / scale,
                f"{self.key} private suh after GSS",
            )
            stored_svh = self.fit.shared_down_svh
            fold_side = "suh"
        regularized = (
            regularize_from_stored(
                self.fit.core,
                self.prepared.weight_kn,
                stored_suh,
                stored_svh,
                block=self.fit.block,
            )
            if materialize_regularized
            else None
        )
        return FinalizedMatrix(
            key=self.key,
            projection=self.projection,
            bits=self.bits,
            g_scale=scale,
            gss_fold_side=fold_side,
            private_amplitude=self.private_amplitude,
            source_weight_identity_sha256=tensor_identity_sha256(
                self.prepared.weight_kn
            ),
            stored_suh=stored_suh,
            stored_svh=stored_svh,
            regularized=regularized,
        )


@dataclass(frozen=True)
class StreamingAbsoluteNormalizationFit:
    """Sealed shared fit; it contains no source or prepared matrices."""

    core: Any = field(repr=False)
    block: int
    codebook_scale: float
    fit_sample_keys: tuple[str, ...]
    shared_gate_up_suh: torch.Tensor
    shared_down_svh: torch.Tensor
    shared_gate_up_base: torch.Tensor = field(repr=False)
    shared_down_base: torch.Tensor = field(repr=False)
    effective_shared_gate_magnitude: torch.Tensor = field(repr=False)
    numeric_core_sha256: str
    fit_plan_sha256: str

    @property
    def shared_gate_up_suh_sha256(self) -> str:
        return tensor_sha256(self.shared_gate_up_suh)

    @property
    def shared_down_svh_sha256(self) -> str:
        return tensor_sha256(self.shared_down_svh)

    def prepare_matrix(self, item: MatrixInput) -> StreamingPreparedMatrix:
        """Prepare one matrix, retaining no state in the layer fit."""

        prepared = _prepare_one(self.core, item, self.block)
        if item.projection in GATE_UP:
            if not torch.equal(prepared.base_suh, self.shared_gate_up_base):
                raise ValueError(f"{item.key}: gate/up shared SU signs/factors drifted")
            relative_svh = _store_fp16(
                prepared.base_svh * prepared.output_scale,
                f"{item.key} relative private svh",
            )
            right = _right_transform(
                self.core,
                prepared.weight_kn,
                relative_svh,
                self.block,
            )
            row_rms = _positive_tensor(
                self.core.block_rms(right, dim=1, keepdim=True).flatten().float(),
                f"{item.key} post-right-Hadamard row RMS",
            )
            log_ratio = (
                row_rms.double().log()
                - self.effective_shared_gate_magnitude.double().log()
            )
            beta = _positive_scalar(
                float(log_ratio.mean().exp().item()),
                f"{item.key} private gate/up amplitude",
            )
            pre_suh = self.shared_gate_up_suh
            pre_svh = _store_fp16(
                relative_svh.float() * beta,
                f"{item.key} private svh amplitude",
            )
        else:
            if not torch.equal(prepared.base_svh, self.shared_down_base):
                raise ValueError(f"{item.key}: down shared SV signs/factors drifted")
            right = _right_transform(
                self.core,
                prepared.weight_kn,
                self.shared_down_svh,
                self.block,
            )
            row_rms = _positive_tensor(
                self.core.block_rms(right, dim=1, keepdim=True).flatten().float(),
                f"{item.key} absolute down row RMS",
            )
            pre_suh = _store_fp16(
                prepared.base_suh * row_rms / (-self.codebook_scale),
                f"{item.key} private absolute suh",
            )
            pre_svh = self.shared_down_svh
            beta = 1.0
        return StreamingPreparedMatrix(
            fit=self,
            prepared=prepared,
            pre_gss_suh=pre_suh,
            pre_gss_svh=pre_svh,
            private_amplitude=beta,
        )


@dataclass(frozen=True)
class MatrixVectorHashes:
    key: str
    projection: str
    bits: int
    suh_sha256: str
    svh_sha256: str
    gss_fold_side: str
    source_weight_identity_sha256: str


@dataclass(frozen=True)
class StreamingNormalizationManifest:
    schema: str
    batch_schema: str
    shared_gate_up_suh_sha256: str
    shared_down_svh_sha256: str
    numeric_core_sha256: str
    fit_plan_sha256: str
    matrices: Mapping[str, MatrixVectorHashes]


class StreamingTopologyLedger:
    """Hash-only topology gate that never retains regularized matrices."""

    def __init__(
        self,
        fit: StreamingAbsoluteNormalizationFit,
        expected_keys: Iterable[str],
    ) -> None:
        self.fit = fit
        self.expected_keys = tuple(expected_keys)
        if len(set(self.expected_keys)) != len(self.expected_keys):
            raise ValueError("expected matrix keys must be unique")
        self._records: dict[str, MatrixVectorHashes] = {}

    def add(self, value: FinalizedMatrix) -> None:
        if value.key not in self.expected_keys:
            raise ValueError(f"unexpected finalized matrix {value.key!r}")
        if value.key in self._records:
            raise ValueError(f"duplicate finalized matrix {value.key!r}")
        if value.projection in GATE_UP:
            if value.suh_sha256 != self.fit.shared_gate_up_suh_sha256:
                raise AssertionError(f"{value.key}: shared gate/up SU bytes drifted")
            if value.gss_fold_side != "svh":
                raise AssertionError(f"{value.key}: gate/up GSS was not private")
        elif value.projection == "down_proj":
            if value.svh_sha256 != self.fit.shared_down_svh_sha256:
                raise AssertionError(f"{value.key}: shared down SV bytes drifted")
            if value.gss_fold_side != "suh":
                raise AssertionError(f"{value.key}: down GSS was not private")
        else:
            raise ValueError(f"{value.key}: unsupported projection {value.projection!r}")
        self._records[value.key] = MatrixVectorHashes(
            key=value.key,
            projection=value.projection,
            bits=value.bits,
            suh_sha256=value.suh_sha256,
            svh_sha256=value.svh_sha256,
            gss_fold_side=value.gss_fold_side,
            source_weight_identity_sha256=value.source_weight_identity_sha256,
        )

    def finish(self) -> StreamingNormalizationManifest:
        missing = [key for key in self.expected_keys if key not in self._records]
        if missing:
            raise ValueError(f"finalized matrix set incomplete; missing={missing}")
        return StreamingNormalizationManifest(
            schema=SCHEMA,
            batch_schema=BATCH_SCHEMA,
            shared_gate_up_suh_sha256=self.fit.shared_gate_up_suh_sha256,
            shared_down_svh_sha256=self.fit.shared_down_svh_sha256,
            numeric_core_sha256=self.fit.numeric_core_sha256,
            fit_plan_sha256=self.fit.fit_plan_sha256,
            matrices=dict(self._records),
        )
