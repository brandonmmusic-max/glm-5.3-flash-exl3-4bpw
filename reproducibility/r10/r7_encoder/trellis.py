"""Pinned MCG TRELLIS codec with forced rotations and full-matrix LDLQ.

This module reuses the previously audited v31 numerical primitives for the
tile codec, block LDL, LDLQ walk, Hadamard transforms, and pack/reconstruct
oracles. It does not reuse v31's sliced orchestration, source model, schema, or
allocation logic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from .constants import (
    DEFAULT_SIGMA_REG,
    HAD_K,
    HAD_N,
    LDL_FACTORIZATION_POLICY,
    MCG_MULT,
    TensorId,
)
from .determinism import sha256_bytes, sha256_file
from .types import EncodedTensor

DEFAULT_NUMERIC_CORE = Path(
    "/home/brandonmusic/klc-linux/glm52_hybrid_opt/encode_tr3_v31.py"
)


def _tensor_bytes(tensor) -> bytes:
    import torch

    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor) -> str:
    return sha256_bytes(_tensor_bytes(tensor))


def load_numeric_core(
    path: str | Path = DEFAULT_NUMERIC_CORE,
    *,
    expected_sha256: str | None = None,
    verify_file: bool = True,
) -> ModuleType:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"numeric core not found: {source}")
    if not expected_sha256:
        raise ValueError("numeric core requires a sealed expected SHA-256")
    if verify_file and sha256_file(source) != expected_sha256:
        raise ValueError("numeric core bytes differ from the sealed environment")
    name = f"_r7_numeric_core_{expected_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import numeric core {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "block_ldl",
        "ldlq",
        "pack_trellis",
        "blockwise_preapply_had_l_",
        "blockwise_preapply_had_r_",
        "preapply_had_l",
        "preapply_had_r",
        "_lazy_torch",
        "CODEBOOK_SCALE",
    )
    missing = [attribute for attribute in required if not hasattr(module, attribute)]
    if missing:
        raise ImportError(f"numeric core missing required functions: {missing}")
    return module


def load_sealed_extension(
    path: str | Path, *, expected_sha256: str, verify_file: bool = True
) -> ModuleType:
    """Load exactly the inventoried ``exllamav3_ext`` module or fail closed."""

    import torch  # noqa: F401 -- the extension requires Torch first

    source = Path(path).resolve()
    if not source.is_file() or (
        verify_file and sha256_file(source) != expected_sha256
    ):
        raise ValueError("TRELLIS extension differs from the sealed binary")
    incumbent = sys.modules.get("exllamav3_ext")
    if incumbent is not None:
        loaded_path = Path(str(getattr(incumbent, "__file__", ""))).resolve()
        if (
            not loaded_path.is_file()
            or loaded_path != source
            or (verify_file and sha256_file(loaded_path) != expected_sha256)
        ):
            raise RuntimeError("ambient exllamav3_ext differs from the sealed binary")
        return incumbent
    spec = importlib.util.spec_from_file_location("exllamav3_ext", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sealed TRELLIS extension {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exllamav3_ext"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("exllamav3_ext", None)
        raise
    loaded_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_path != source or (
        verify_file and sha256_file(loaded_path) != expected_sha256
    ):
        sys.modules.pop("exllamav3_ext", None)
        raise RuntimeError("loaded TRELLIS extension path/hash drift")
    return module


def cuda_only_block_ldl(h, block: int, quant_args: Mapping[str, Any]):
    """Pinned v31 LDL math with the resource-dependent CPU fallback removed."""

    import torch

    if h.device.type != "cuda":
        raise ValueError("production LDL factorization is pinned to CUDA")
    size = int(h.shape[0])
    if size % block:
        raise ValueError("LDL dimension is not divisible by its block")
    blocks = size // block
    retries = 0
    while True:
        try:
            factor = torch.linalg.cholesky(h)
            proxy_h = h.cpu()
            h.copy_(factor)
            factor = h
            h = proxy_h
            break
        except torch._C._LinAlgError:
            retries += 1
            if retries > 10:
                raise
            h.diagonal().add_(
                2.0
                * float(quant_args.get("sigma_reg", DEFAULT_SIGMA_REG))
                * h.diagonal().mean()
            )
        except Exception as exc:
            if (
                exc.__class__.__name__ == "OutOfMemoryError"
                or "out of memory" in str(exc).lower()
            ):
                raise RuntimeError(
                    "CUDA LDL out of memory; CPU fallback is forbidden by the sealed recipe"
                ) from exc
            raise
    diagonal_blocks = torch.diagonal(
        factor.reshape(blocks, block, blocks, block), dim1=0, dim2=2
    ).permute(2, 0, 1)
    inverse = torch.linalg.inv(diagonal_blocks)
    factor = factor.view(size, blocks, block)
    for index in range(blocks):
        factor[:, index, :] = factor[:, index, :] @ inverse[index, :, :]
    factor = factor.reshape(size, size).contiguous()
    blocked = factor.view(blocks, block, blocks, block).permute(0, 2, 1, 3)
    indices = torch.arange(blocks)
    blocked[indices, indices] = torch.stack(
        [torch.eye(block, device=factor.device, dtype=h.dtype)] * blocks
    )
    return factor, h


@dataclass(frozen=True)
class CodecConfig:
    device: str = "cuda:0"
    sigma_reg: float = DEFAULT_SIGMA_REG
    numeric_core: Path = DEFAULT_NUMERIC_CORE
    numeric_core_sha256: str | None = None
    extension: Path | None = None
    extension_sha256: str | None = None
    factorization_policy: str = LDL_FACTORIZATION_POLICY
    verify_files: bool = True


class Exl3TrellisCodec:
    """Encode one complete tensor with caller-selected final `suh`/`svh`."""

    def __init__(self, config: CodecConfig = CodecConfig()) -> None:
        self.config = config
        self._core: ModuleType | None = None

    @property
    def core(self) -> ModuleType:
        if self._core is None:
            if self.config.extension is None or not self.config.extension_sha256:
                raise ValueError("codec requires a sealed TRELLIS extension path/hash")
            load_sealed_extension(
                self.config.extension,
                expected_sha256=self.config.extension_sha256,
                verify_file=self.config.verify_files,
            )
            self._core = load_numeric_core(
                self.config.numeric_core,
                expected_sha256=self.config.numeric_core_sha256,
                verify_file=self.config.verify_files,
            )
        return self._core

    @property
    def codebook_scale(self) -> float:
        value = float(self.core.CODEBOOK_SCALE)
        if not math.isfinite(value) or value == 0:
            raise ValueError("numeric core exposes an invalid MCG codebook scale")
        return value

    def _validate_vectors(
        self, suh: Sequence[float] | Any, svh: Sequence[float] | Any, k: int, n: int
    ):
        import torch

        su = torch.as_tensor(
            suh, dtype=torch.float32, device=self.config.device
        ).flatten()
        sv = torch.as_tensor(
            svh, dtype=torch.float32, device=self.config.device
        ).flatten()
        if tuple(su.shape) != (k,) or tuple(sv.shape) != (n,):
            raise ValueError(
                f"rotation shape mismatch: suh={tuple(su.shape)} vs {k}, "
                f"svh={tuple(sv.shape)} vs {n}"
            )
        if not torch.isfinite(su).all() or not torch.isfinite(sv).all():
            raise ValueError("rotation vectors must be finite")
        if (su == 0).any() or (sv == 0).any():
            raise ValueError("zero rotation/scale entry is not invertible")
        # Store exactly what the loader sees; validate *after* FP16 rounding
        # and use those rounded values for the encode transformation so encode
        # and serve cannot disagree.  A finite FP32 candidate can otherwise
        # become zero or infinity at the actual checkpoint boundary.
        su_stored = su.half()
        sv_stored = sv.half()
        if not torch.isfinite(su_stored).all() or not torch.isfinite(sv_stored).all():
            raise ValueError("rotation vector overflows its stored FP16 representation")
        if (su_stored == 0).any() or (sv_stored == 0).any():
            raise ValueError(
                "rotation vector underflows its stored FP16 representation"
            )
        return su_stored.float(), sv_stored.float()

    def _regularize_weight(self, weight_kn, su, sv):
        weight = weight_kn.clone().to(dtype=weight_kn.dtype)
        weight.div_(sv.unsqueeze(0))
        self.core.blockwise_preapply_had_r_(weight, HAD_N)
        weight.div_(su.unsqueeze(1))
        self.core.blockwise_preapply_had_l_(weight, HAD_K)
        return weight

    def _factor_covariance(self, covariance, su, bits: int, sigma_reg: float):
        import torch

        h = torch.as_tensor(
            covariance, dtype=torch.float32, device=self.config.device
        ).clone()
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] != su.numel():
            raise ValueError("full covariance shape does not match tensor K")
        h = (h + h.T) * 0.5
        diagonal = h.diagonal()
        mean = diagonal.mean()
        if not torch.isfinite(h).all() or not math.isfinite(float(mean.item())):
            raise ValueError("non-finite covariance")
        if float(mean.item()) <= 1e-20:
            raise ValueError("degenerate covariance; identity-H fallback is forbidden")
        diagonal.add_(float(sigma_reg) * mean)

        # Transform the metric by the exact stored input-side vector. This is
        # the covariance corresponding to the inverse weight transform below.
        h.mul_(su.unsqueeze(0))
        self.core.blockwise_preapply_had_r_(h, HAD_K)
        h.mul_(su.unsqueeze(1))
        self.core.blockwise_preapply_had_l_(h, HAD_K)
        quant_args = {
            "K": int(bits),
            "devices": [int(str(self.config.device).split(":")[-1])],
            "mcg": True,
            "sigma_reg": float(sigma_reg),
            "buf_size_k": 128,
        }
        if self.config.factorization_policy != LDL_FACTORIZATION_POLICY:
            raise ValueError("LDL factorization policy differs from the sealed recipe")
        factor, proxy_h = cuda_only_block_ldl(h, 16, quant_args)
        indices = torch.arange(factor.shape[0], device=factor.device)
        factor[indices, indices] = 0
        return factor, proxy_h, quant_args

    def _decode_regularized(self, packed, k: int, n: int, bits: int):
        torch, extension = self.core._lazy_torch()
        decoded = torch.empty((k, n), dtype=torch.float16, device=packed.device)
        extension.reconstruct(decoded, packed, bits, True, False)
        return decoded

    def decode_to_original(self, packed, suh, svh, bits: int):
        import torch

        k = int(torch.as_tensor(suh).numel())
        n = int(torch.as_tensor(svh).numel())
        regularized = self._decode_regularized(packed, k, n, bits).float()
        su = torch.as_tensor(
            suh, device=regularized.device, dtype=torch.float32
        ).flatten()
        sv = torch.as_tensor(
            svh, device=regularized.device, dtype=torch.float32
        ).flatten()
        decoded = self.core.preapply_had_l(regularized, HAD_K)
        decoded.mul_(su.unsqueeze(1))
        decoded = self.core.preapply_had_r(decoded, HAD_N)
        decoded.mul_(sv.unsqueeze(0))
        return decoded

    def encode(
        self,
        *,
        tensor_id: TensorId,
        weight_hf,
        covariance,
        bits: int,
        suh,
        svh,
        sigma_reg: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> EncodedTensor:
        import torch

        if bits not in (3, 4, 5):
            raise ValueError("Round 7 production codec accepts only 3, 4, or 5 bits")
        weight = torch.as_tensor(
            weight_hf, device=self.config.device, dtype=torch.float32
        )
        if tuple(weight.shape) != (tensor_id.n, tensor_id.k):
            raise ValueError(
                f"{tensor_id.key}: expected HF [N,K]={(tensor_id.n, tensor_id.k)}, "
                f"got {tuple(weight.shape)}"
            )
        weight_kn = weight.T.contiguous()
        su, sv = self._validate_vectors(suh, svh, tensor_id.k, tensor_id.n)
        regularized = self._regularize_weight(weight_kn, su, sv)
        factor, proxy_h, quant_args = self._factor_covariance(
            covariance,
            su,
            bits,
            self.config.sigma_reg if sigma_reg is None else sigma_reg,
        )
        quantized_regularized, encoded = self.core.ldlq(regularized, factor, quant_args)
        packed = self.core.pack_trellis(encoded, quant_args)

        torch_module, extension = self.core._lazy_torch()
        unpacked = torch_module.zeros_like(encoded)
        extension.unpack_trellis(unpacked, packed, bits)
        if not torch_module.equal(unpacked, encoded):
            raise AssertionError("TRELLIS pack/unpack index mismatch")
        extension_decoded = self._decode_regularized(
            packed, tensor_id.k, tensor_id.n, bits
        )
        if not torch_module.equal(extension_decoded, quantized_regularized.half()):
            raise AssertionError("extension reconstruction differs from LDLQ values")
        repeated_decoded = self._decode_regularized(
            packed, tensor_id.k, tensor_id.n, bits
        )
        if not torch_module.equal(repeated_decoded, extension_decoded):
            raise AssertionError(
                "TRELLIS extension reconstruction is not byte deterministic"
            )
        expected_bytes = tensor_id.k * tensor_id.n * bits // 8
        if packed.numel() * packed.element_size() != expected_bytes:
            raise AssertionError(
                "packed TRELLIS byte count disagrees with integer bits"
            )

        reconstructed = self.decode_to_original(packed, su.half(), sv.half(), bits)
        error = reconstructed.double() - weight_kn.double()
        numerator = torch.einsum(
            "kn,kl,ln->",
            error,
            torch.as_tensor(covariance, device=error.device, dtype=torch.float64),
            error,
        )
        denominator = torch.einsum(
            "kn,kl,ln->",
            weight_kn.double(),
            torch.as_tensor(covariance, device=error.device, dtype=torch.float64),
            weight_kn.double(),
        ).clamp_min(1e-30)
        proxy_loss = float((numerator / denominator).item())
        packed_hash = _tensor_sha256(packed)
        reconstruction_hash = _tensor_sha256(reconstructed.half())
        metadata = dict(provenance or {})
        metadata.update(
            {
                "numeric_core": str(self.config.numeric_core),
                "full_k": tensor_id.k,
                "full_n": tensor_id.n,
                "mcg": f"0x{MCG_MULT:08X}",
                "codebook_scale": self.codebook_scale,
                "sigma_reg": self.config.sigma_reg if sigma_reg is None else sigma_reg,
                "factorization_policy": self.config.factorization_policy,
                "extension_sha256": self.config.extension_sha256,
                "extension_repeat_oracle": True,
                "covariance_sha256": _tensor_sha256(
                    torch.as_tensor(covariance, dtype=torch.float32)
                ),
            }
        )
        return EncodedTensor(
            tensor_id=tensor_id,
            bits=bits,
            trellis=packed.detach().cpu(),
            suh=su.half().detach().cpu(),
            svh=sv.half().detach().cpu(),
            reconstructed_kn=reconstructed.detach().cpu(),
            proxy_loss=proxy_loss,
            packed_sha256=packed_hash,
            reconstruction_sha256=reconstruction_hash,
            provenance=metadata,
        )


def heldout_output_loss(x, weight_hf, reconstructed_kn) -> float:
    import torch

    inputs = torch.as_tensor(x, dtype=torch.float64)
    reference_weight = torch.as_tensor(weight_hf, dtype=torch.float64)
    candidate = torch.as_tensor(reconstructed_kn, dtype=torch.float64)
    if tuple(reference_weight.T.shape) != tuple(candidate.shape):
        raise ValueError("reconstructed matrix shape mismatch")
    reference = inputs @ reference_weight.T
    decoded = inputs @ candidate
    denominator = reference.square().sum().clamp_min(1e-30)
    return float(((reference - decoded).square().sum() / denominator).item())
