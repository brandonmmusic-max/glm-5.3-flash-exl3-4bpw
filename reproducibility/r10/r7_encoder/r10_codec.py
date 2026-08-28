"""Round 10 wall-clock optimized TRELLIS codec.

The emitted representation is deliberately the same one produced by
``Exl3TrellisCodec``: the sealed v31 numerical core still performs the weight
regularization, LDLQ walk, and TRELLIS packing, and the sealed extension still
performs the one reconstruction consumed by the loss/provenance path.

Round 10 removes work which cannot affect those bytes:

* a transformed covariance/Cholesky factor is cached across projections and
  bit probes when covariance content, stored (FP16-rounded) ``suh``, and sigma
  are identical;
* the factorization no longer copies the otherwise-unused transformed
  covariance back to the host; and
* pack/unpack, LDLQ-vs-extension, repeat-decode, and covariance-proxy-loss
  encode-time audits are not run.  The packed byte count is still checked and
  one packed reconstruction is retained for the real held-out loss.

The factor cache is byte-bounded because a 6144-square FP32 factor occupies
144 MiB.  Covariance digests are memoized by live tensor identity and PyTorch
version, while the factor itself is keyed by content digest.  Thus the common
same-object path avoids both another device-to-host covariance copy and another
Cholesky, and an equal-content replacement tensor can still reuse the factor.
"""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from typing import Any, Mapping, Sequence

from .constants import (
    DEFAULT_SIGMA_REG,
    HAD_K,
    HAD_N,
    LDL_FACTORIZATION_POLICY,
    MCG_MULT,
    TensorId,
)
from .trellis import CodecConfig, Exl3TrellisCodec, _tensor_sha256
from .types import EncodedTensor


DEFAULT_FACTOR_CACHE_BYTES = 512 * 1024 * 1024
DEFAULT_DIGEST_CACHE_ENTRIES = 64


def _cuda_only_block_ldl_without_proxy(
    h, block: int, quant_args: Mapping[str, Any]
):
    """The pinned block-LDL arithmetic without v31's unused ``H.cpu()``.

    The original memory trick copies the Cholesky result into ``h`` and returns
    the old covariance as a CPU proxy.  Round 10 keeps the first half exactly --
    including reuse of ``h`` for the factor -- but drops the host copy because
    no codec caller consumes it.
    """

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
            cholesky = torch.linalg.cholesky(h)
            # Preserve the numeric core's storage/layout path.  Only the
            # preceding `h.cpu()` synchronization/copy is intentionally gone.
            h.copy_(cholesky)
            factor = h
            del cholesky
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
                    "CUDA LDL out of memory; CPU fallback is forbidden by the "
                    "sealed recipe"
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
        [torch.eye(block, device=factor.device, dtype=factor.dtype)] * blocks
    )
    return factor


class R10TrellisCodec(Exl3TrellisCodec):
    """Fast, byte-compatible production codec for the Round 10 scripts."""

    def __init__(
        self,
        config: CodecConfig = CodecConfig(),
        *,
        factor_cache_bytes: int = DEFAULT_FACTOR_CACHE_BYTES,
        digest_cache_entries: int = DEFAULT_DIGEST_CACHE_ENTRIES,
    ) -> None:
        super().__init__(config)
        if factor_cache_bytes < 0:
            raise ValueError("factor_cache_bytes must be nonnegative")
        if digest_cache_entries < 0:
            raise ValueError("digest_cache_entries must be nonnegative")
        self.factor_cache_bytes = int(factor_cache_bytes)
        self.digest_cache_entries = int(digest_cache_entries)
        self._factor_cache: OrderedDict[tuple[object, ...], Any] = OrderedDict()
        self._factor_cache_size = 0
        # id(tensor) -> (weak reference, PyTorch version, float32 content digest)
        self._covariance_digests: OrderedDict[
            int, tuple[weakref.ReferenceType[Any], int | None, str]
        ] = OrderedDict()
        self._factor_cache_hits = 0
        self._factor_cache_misses = 0
        self._factorizations = 0

    @property
    def cache_stats(self) -> Mapping[str, int]:
        """Small diagnostic surface; it performs no synchronization."""

        return {
            "factor_hits": self._factor_cache_hits,
            "factor_misses": self._factor_cache_misses,
            "factorizations": self._factorizations,
            "factor_entries": len(self._factor_cache),
            "factor_bytes": self._factor_cache_size,
            "covariance_digest_entries": len(self._covariance_digests),
        }

    def clear_caches(self) -> None:
        """Release codec-owned cache references without forcing a CUDA flush."""

        self._factor_cache.clear()
        self._factor_cache_size = 0
        self._covariance_digests.clear()

    @staticmethod
    def _tensor_version(value: Any) -> int | None:
        try:
            return int(value._version)
        except (AttributeError, RuntimeError, TypeError):
            return None

    def _covariance_sha256(self, covariance: Any) -> str:
        """Hash the exact float32 covariance, memoizing safe tensor identities."""

        import torch

        if isinstance(covariance, torch.Tensor) and self.digest_cache_entries:
            identity = id(covariance)
            version = self._tensor_version(covariance)
            cached = self._covariance_digests.get(identity)
            if (
                cached is not None
                and cached[0]() is covariance
                and cached[1] == version
            ):
                self._covariance_digests.move_to_end(identity)
                return cached[2]

            digest = _tensor_sha256(
                torch.as_tensor(covariance, dtype=torch.float32)
            )
            try:
                reference = weakref.ref(covariance)
            except TypeError:
                return digest
            self._covariance_digests[identity] = (reference, version, digest)
            self._covariance_digests.move_to_end(identity)
            while len(self._covariance_digests) > self.digest_cache_entries:
                self._covariance_digests.popitem(last=False)
            return digest

        return _tensor_sha256(torch.as_tensor(covariance, dtype=torch.float32))

    def _factor_cache_key(
        self,
        *,
        covariance_sha256: str,
        su,
        sigma_reg: float,
        size: int,
    ) -> tuple[object, ...]:
        # `su` consists exactly of FP16-storable values promoted to FP32 by
        # `_validate_vectors`; hash its checkpoint representation explicitly.
        stored_su_sha256 = _tensor_sha256(su.half())
        return (
            covariance_sha256,
            stored_su_sha256,
            float(sigma_reg).hex(),
            int(size),
            str(self.config.device),
            self.config.factorization_policy,
        )

    def _build_factor(self, transformed_covariance, quant_args):
        """Injection point used by CPU-only tests; production remains CUDA-only."""

        return _cuda_only_block_ldl_without_proxy(
            transformed_covariance, 16, quant_args
        )

    @staticmethod
    def _tensor_nbytes(value) -> int:
        return int(value.numel()) * int(value.element_size())

    def _remember_factor(self, key: tuple[object, ...], factor) -> None:
        if self.factor_cache_bytes == 0:
            return
        factor_bytes = self._tensor_nbytes(factor)
        if factor_bytes > self.factor_cache_bytes:
            return
        incumbent = self._factor_cache.pop(key, None)
        if incumbent is not None:
            self._factor_cache_size -= self._tensor_nbytes(incumbent)
        self._factor_cache[key] = factor
        self._factor_cache_size += factor_bytes
        while self._factor_cache_size > self.factor_cache_bytes:
            _, evicted = self._factor_cache.popitem(last=False)
            self._factor_cache_size -= self._tensor_nbytes(evicted)

    def _quant_args(self, bits: int, sigma_reg: float) -> dict[str, object]:
        device_text = str(self.config.device)
        device_index = int(device_text.rsplit(":", 1)[1]) if ":" in device_text else 0
        return {
            "K": int(bits),
            "devices": [device_index],
            "mcg": True,
            "sigma_reg": float(sigma_reg),
            "buf_size_k": 128,
        }

    def _factor_covariance_cached(
        self,
        covariance,
        su,
        bits: int,
        sigma_reg: float,
        covariance_sha256: str,
    ):
        import torch

        quant_args = self._quant_args(bits, sigma_reg)
        if self.config.factorization_policy != LDL_FACTORIZATION_POLICY:
            raise ValueError("LDL factorization policy differs from the sealed recipe")

        source = torch.as_tensor(
            covariance, dtype=torch.float32, device=self.config.device
        )
        if (
            source.ndim != 2
            or source.shape[0] != source.shape[1]
            or source.shape[0] != su.numel()
        ):
            raise ValueError("full covariance shape does not match tensor K")
        key = self._factor_cache_key(
            covariance_sha256=covariance_sha256,
            su=su,
            sigma_reg=sigma_reg,
            size=int(source.shape[0]),
        )
        factor = self._factor_cache.get(key)
        if factor is not None:
            self._factor_cache_hits += 1
            self._factor_cache.move_to_end(key)
            return factor, quant_args

        self._factor_cache_misses += 1
        h = source.clone()
        h = (h + h.T) * 0.5
        diagonal = h.diagonal()
        mean = diagonal.mean()
        if not torch.isfinite(h).all() or not math.isfinite(float(mean.item())):
            raise ValueError("non-finite covariance")
        if float(mean.item()) <= 1e-20:
            raise ValueError("degenerate covariance; identity-H fallback is forbidden")
        diagonal.add_(float(sigma_reg) * mean)

        # Same transform and ordering as Exl3TrellisCodec._factor_covariance.
        h.mul_(su.unsqueeze(0))
        self.core.blockwise_preapply_had_r_(h, HAD_K)
        h.mul_(su.unsqueeze(1))
        self.core.blockwise_preapply_had_l_(h, HAD_K)
        factor = self._build_factor(h, quant_args)
        indices = torch.arange(factor.shape[0], device=factor.device)
        factor[indices, indices] = 0
        self._factorizations += 1
        self._remember_factor(key, factor)
        return factor, quant_args

    def encode(
        self,
        *,
        tensor_id: TensorId,
        weight_hf,
        covariance,
        bits: int,
        suh: Sequence[float] | Any,
        svh: Sequence[float] | Any,
        sigma_reg: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> EncodedTensor:
        """Encode one bit width through the shared Round 10 implementation."""

        return self.encode_bits(
            tensor_id=tensor_id,
            weight_hf=weight_hf,
            covariance=covariance,
            bits=(bits,),
            suh=suh,
            svh=svh,
            sigma_reg=sigma_reg,
            provenance=provenance,
        )[bits]

    def encode_bits(
        self,
        *,
        tensor_id: TensorId,
        weight_hf,
        covariance,
        bits: Sequence[int] = (3, 4, 5),
        suh: Sequence[float] | Any,
        svh: Sequence[float] | Any,
        sigma_reg: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[int, EncodedTensor]:
        """Encode one tensor's candidates with one preparation/factorization."""

        return self.encode_group(
            (
                {
                    "tensor_id": tensor_id,
                    "weight_hf": weight_hf,
                    "covariance": covariance,
                    "bits": bits,
                    "suh": suh,
                    "svh": svh,
                    "sigma_reg": sigma_reg,
                    "provenance": provenance,
                },
            )
        )[0]

    def encode_group(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[int, EncodedTensor], ...]:
        """Prepare tensors once and co-step equal-K LDLQ walks.

        Brandon's v31 numeric core already contains the production
        ``LDLQWalk``/``lockstep_ldlq`` path used by his fast B300 encoder.  A
        mixed 3/4/5 pool cannot share one quantizer call, so this method groups
        requests by K and locksteps every same-K walk.  It does not run the
        v31 self-check/oracle path during the encode.
        """

        import torch

        if not requests:
            raise ValueError("encode_group requires at least one request")
        prepared: list[dict[str, Any]] = []
        work_by_bits: dict[int, list[dict[str, Any]]] = {}
        for request_index, request in enumerate(requests):
            tensor_id = request["tensor_id"]
            bit_widths = tuple(int(value) for value in request.get("bits", (3, 4, 5)))
            if not bit_widths or len(set(bit_widths)) != len(bit_widths):
                raise ValueError("candidate bit widths must be nonempty and unique")
            if any(value not in (3, 4, 5) for value in bit_widths):
                raise ValueError("Round 10 accepts only 3, 4, or 5 bits")
            weight = torch.as_tensor(
                request["weight_hf"], device=self.config.device, dtype=torch.float32
            )
            if tuple(weight.shape) != (tensor_id.n, tensor_id.k):
                raise ValueError(
                    f"{tensor_id.key}: expected HF [N,K]="
                    f"{(tensor_id.n, tensor_id.k)}, got {tuple(weight.shape)}"
                )
            su, sv = self._validate_vectors(
                request["suh"], request["svh"], tensor_id.k, tensor_id.n
            )
            regularized = self._regularize_weight(weight.T.contiguous(), su, sv)
            raw_sigma = request.get("sigma_reg")
            resolved_sigma = (
                self.config.sigma_reg if raw_sigma is None else float(raw_sigma)
            )
            covariance = request["covariance"]
            covariance_sha256 = self._covariance_sha256(covariance)
            factor, first_quant_args = self._factor_covariance_cached(
                covariance,
                su,
                bit_widths[0],
                resolved_sigma,
                covariance_sha256,
            )
            metadata = dict(request.get("provenance") or {})
            metadata.update(
                {
                    "numeric_core": str(self.config.numeric_core),
                    "full_k": tensor_id.k,
                    "full_n": tensor_id.n,
                    "mcg": f"0x{MCG_MULT:08X}",
                    "codebook_scale": self.codebook_scale,
                    "sigma_reg": resolved_sigma,
                    "factorization_policy": self.config.factorization_policy,
                    "extension_sha256": self.config.extension_sha256,
                    "extension_repeat_oracle": False,
                    "covariance_proxy_loss_evaluated": False,
                    "covariance_sha256": covariance_sha256,
                }
            )
            record = {
                "request_index": request_index,
                "tensor_id": tensor_id,
                "regularized": regularized,
                "factor": factor,
                "su": su,
                "sv": sv,
                "stored_su": su.half().detach().cpu(),
                "stored_sv": sv.half().detach().cpu(),
                "metadata": metadata,
                "results": {},
            }
            prepared.append(record)
            for bit_index, bit_width in enumerate(bit_widths):
                quant_args = (
                    first_quant_args
                    if bit_index == 0
                    else self._quant_args(bit_width, resolved_sigma)
                )
                work_by_bits.setdefault(bit_width, []).append(
                    {"record": record, "bits": bit_width, "quant_args": quant_args}
                )

        walk_type = getattr(self.core, "LDLQWalk", None)
        lockstep = getattr(self.core, "lockstep_ldlq", None)
        for bit_width in sorted(work_by_bits):
            jobs = work_by_bits[bit_width]
            if len(jobs) > 1 and callable(walk_type) and callable(lockstep):
                walks = []
                for job in jobs:
                    record = job["record"]
                    context = {
                        "weight_r": record["regularized"],
                        "L": record["factor"],
                        "quant_args": job["quant_args"],
                    }
                    job["context"] = context
                    walks.append(walk_type(context))
                lockstep(walks, len(walks))
                encoded_values = [job["context"]["encoded"] for job in jobs]
            else:
                encoded_values = [
                    self.core.ldlq(
                        job["record"]["regularized"],
                        job["record"]["factor"],
                        job["quant_args"],
                    )[1]
                    for job in jobs
                ]

            for job, encoded in zip(jobs, encoded_values):
                record = job["record"]
                tensor_id = record["tensor_id"]
                quant_args = job["quant_args"]
                packed = self.core.pack_trellis(encoded, quant_args)
                expected_bytes = tensor_id.k * tensor_id.n * bit_width // 8
                if packed.numel() * packed.element_size() != expected_bytes:
                    raise AssertionError(
                        "packed TRELLIS byte count disagrees with integer bits"
                    )
                regularized_decoded = self._decode_regularized(
                    packed, tensor_id.k, tensor_id.n, bit_width
                ).float()
                reconstructed = self.core.preapply_had_l(
                    regularized_decoded, HAD_K
                )
                reconstructed.mul_(record["su"].unsqueeze(1))
                reconstructed = self.core.preapply_had_r(reconstructed, HAD_N)
                reconstructed.mul_(record["sv"].unsqueeze(0))
                packed_cpu = packed.detach().cpu()
                record["results"][bit_width] = EncodedTensor(
                    tensor_id=tensor_id,
                    bits=bit_width,
                    trellis=packed_cpu,
                    suh=record["stored_su"],
                    svh=record["stored_sv"],
                    reconstructed_kn=reconstructed,
                    proxy_loss=0.0,
                    packed_sha256=_tensor_sha256(packed_cpu),
                    reconstruction_sha256=_tensor_sha256(reconstructed.half()),
                    provenance=dict(record["metadata"]),
                )
        return tuple(record["results"] for record in prepared)


# Short alias for scripts which prefer the round number over the full name.
R10Codec = R10TrellisCodec


__all__ = [
    "DEFAULT_FACTOR_CACHE_BYTES",
    "R10Codec",
    "R10TrellisCodec",
]
