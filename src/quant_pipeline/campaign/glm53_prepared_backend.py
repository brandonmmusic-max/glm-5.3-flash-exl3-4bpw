"""Direct public-ShapleyMCG numerical backend for GLM-5.3 uniform K4/K6.

The producer is deliberately a thin GLM geometry/capture adapter around the
reviewed public path: streaming v31 normalization, pinned per-matrix GSS and
``Exl3MCGCodec`` backed by ``r7_encoder.r10_codec.R10TrellisCodec``.  It does
not import an alternate quantizer or run a bit allocator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..codecs.exl3_mcg import Exl3MCGCodec
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file
from .glm53_direct_k4 import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    RECIPE_ID,
    EncodeRequest,
)


PREPARATION_SCHEMA = "quant-pipeline.glm53-public-shapleymcg-layer-preparation.v1"
BACKEND_SCHEMA = "quant-pipeline.glm53-public-shapleymcg-backend.v1"
MCG_MARKER_SIGNED_INT32 = -877912083
SUPPORTED_BITS = (4, 6)
HADAMARD_BLOCK = 128
_HASH = __import__("re").compile(r"[0-9a-f]{64}")
_FORBIDDEN = ("glm52_fresh_" + "sqg", "score_" + "sqg", "encode_uniform_" + "sqg")
_PROCESS_STRUCTURE = {
    "driver": "scripts/run_qwen_fast_encode.py",
    "normalization": "src/quant_pipeline/normalization/streaming_v31.py",
    "codec_adapter": "src/quant_pipeline/codecs/exl3_mcg.py",
    "operation_order": "reproducibility/local-corrected-v1",
    "numeric_closure": "r7_encoder",
}


def _tensor_sha256(value: Any) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _raw_payload_sha256(value: Any) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _source_closure(root: Path) -> list[dict[str, str]]:
    required = tuple(_PROCESS_STRUCTURE[key] for key in ("driver", "normalization", "codec_adapter"))
    paths = [root / relative for relative in required]
    paths.extend(sorted((root / "r7_encoder").rglob("*.py")))
    if not paths or any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"public ShapleyMCG producer closure is incomplete: {missing}")
    records: list[dict[str, str]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if any(token in lowered or token in text for token in _FORBIDDEN):
            raise ValueError(f"producer closure contains a forbidden alternate path: {relative}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    return records


class Glm53PreparedMCGBackend:
    """One-GPU public ShapleyMCG encoder with grouped equal-K LDLQ walks."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        inventory: Mapping[str, Any],
        prepared_root: str | Path,
        preparation_root: str | Path,
        hessian_root: str | Path,
        reader_abi_sha256: str,
        device: str = "cuda:0",
        chunk_rows: int = 1024,
        numeric_core_path: str | Path | None = None,
        extension_path: str | Path | None = None,
    ) -> None:
        self.contract = copy.deepcopy(dict(contract))
        self.inventory = copy.deepcopy(dict(inventory))
        self.source_root = Path(prepared_root).resolve()
        self.preparation_root = Path(preparation_root).resolve()
        self.hessian_root = Path(hessian_root).resolve()
        self.hessian_root.mkdir(parents=True, exist_ok=True)
        self.reader_abi_sha256 = str(reader_abi_sha256)
        self.device = str(device)
        self.chunk_rows = int(chunk_rows)
        if _HASH.fullmatch(self.reader_abi_sha256) is None:
            raise ValueError("reader ABI must be SHA-256")
        if not self.device.startswith("cuda:") or self.chunk_rows <= 0:
            raise ValueError("production backend requires one explicit CUDA device")
        rate = self.contract.get("rate", {})
        self.bits = int(rate.get("bits", -1))
        if (
            self.bits not in SUPPORTED_BITS
            or rate.get("allowed_bits") != [self.bits]
            or rate.get("global_allocator") is not False
            or rate.get("candidate_rate_grid") is not False
        ):
            raise ValueError("public ShapleyMCG backend requires one uniform K4/K6 rate without allocator")

        recipe = self.contract.get("recipe", {})
        exllama = self.contract.get("exllama", {})
        core_path = Path(numeric_core_path or str(recipe.get("numeric_core_path", ""))).resolve()
        extension = Path(extension_path or str(exllama.get("extension_path", ""))).resolve()
        core_sha = str(recipe.get("numeric_core_sha256", ""))
        extension_sha = str(exllama.get("extension_sha256", ""))
        if (
            not core_path.is_file()
            or _HASH.fullmatch(core_sha) is None
            or sha256_file(core_path) != core_sha
            or not extension.is_file()
            or _HASH.fullmatch(extension_sha) is None
            or sha256_file(extension) != extension_sha
        ):
            raise ValueError("sealed public MCG numeric core/extension differs")
        self.adapter = Exl3MCGCodec(
            source_root=self.source_root,
            numeric_core=core_path,
            extension=extension,
            device=self.device,
            sigma_reg=0.025,
        )
        self.codec = self.adapter._codec()
        self._producer_source_closure = _source_closure(self.source_root)
        self._layer_cache: dict[int, Mapping[str, Any]] = {}

    def identity(self) -> Mapping[str, Any]:
        recipe = self.contract.get("recipe", {})
        result = {
            "schema": BACKEND_SCHEMA,
            "recipe_id": self.contract.get("recipe", {}).get("recipe_id", RECIPE_ID),
            "codec_family": "exl3-mcg",
            "codec_class": "r7_encoder.r10_codec.R10TrellisCodec",
            "public_codec_adapter": "Exl3MCGCodec",
            "mcg_multiplier_hex": "0xCBAC1FED",
            "mcg_marker_signed_int32": MCG_MARKER_SIGNED_INT32,
            "bits": self.bits,
            "candidate_rate_grid": False,
            "global_allocator": False,
            "prepared_source_tree_sha256": recipe.get("tree_sha256"),
            "exllama_extension_sha256": self.adapter.identity["extension_sha256"],
            "numeric_core_sha256": self.adapter.identity["numeric_core_sha256"],
            "reader_abi_sha256": self.reader_abi_sha256,
            "reviewed_glm53_adapter": True,
            "direct_mcg_entrypoint_reviewed": True,
            "source_closure_sqg_free": True,
            "sqg_orchestration_imported": False,
            "shapleymcg_process_structure": dict(_PROCESS_STRUCTURE),
            "producer_source_closure": copy.deepcopy(self._producer_source_closure),
            "codec_identity": copy.deepcopy(self.adapter.identity),
        }
        pure_receipt = self.contract.get("pure_mcg_backend_receipt_sha256")
        if pure_receipt is not None:
            result["pure_mcg_backend_receipt_sha256"] = pure_receipt
        return result

    def _load_preparation(self, layer: int) -> Mapping[str, Any]:
        import torch
        from safetensors import safe_open

        if layer in self._layer_cache:
            return self._layer_cache[layer]
        directory = self.preparation_root / f"layer-{layer:03d}"
        manifest = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
        shard = directory / str(manifest.get("shard", ""))
        required = {"permutations", "gate_suh", "gate_svh", "up_suh", "up_svh", "down_suh", "down_svh"}
        if (
            manifest.get("schema") != PREPARATION_SCHEMA
            or manifest.get("complete") is not True
            or manifest.get("layer") != layer
            or manifest.get("bits") != self.bits
            or manifest.get("codec_family") != "exl3-mcg"
            or manifest.get("policy") != "energy_balanced"
            or manifest.get("scale_family") != "per128-grid"
            or manifest.get("profile_source")
            != "public-run-qwen-fast-encode-defaults"
            or manifest.get("profile_fixed_before_encoding") is not True
            or manifest.get("selection_rows_used") is not False
            or manifest.get("selection_used_for_profile_choice") is not False
            or manifest.get("selection_used_for_final_encoding") is not False
            or manifest.get("confirmation_used_for_choice") is not False
            or manifest.get("confirmation_report_only") is not True
            or manifest.get("global_allocator_invoked") is not False
            or manifest.get("candidate_rate_grid_invoked") is not False
            or manifest.get("source_closure_sqg_free") is not True
            or manifest.get("shapleymcg_process_structure") != _PROCESS_STRUCTURE
            or sha256_file(shard) != manifest.get("shard_sha256")
        ):
            raise ValueError(f"layer {layer}: public ShapleyMCG preparation binding differs")
        manifest_body = copy.deepcopy(manifest)
        preparation_sha = manifest_body.pop("preparation_sha256", None)
        if (
            not isinstance(preparation_sha, str)
            or preparation_sha != sha256_bytes(canonical_json(manifest_body))
        ):
            raise ValueError(f"layer {layer}: public ShapleyMCG preparation seal differs")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != required:
                raise ValueError("public ShapleyMCG preparation tensor census differs")
            tensors = {name: handle.get_tensor(name).contiguous() for name in required}
        shapes = {
            "permutations": (NUM_EXPERTS, INTERMEDIATE_SIZE),
            "gate_suh": (NUM_EXPERTS, HIDDEN_SIZE),
            "gate_svh": (NUM_EXPERTS, INTERMEDIATE_SIZE),
            "up_suh": (NUM_EXPERTS, HIDDEN_SIZE),
            "up_svh": (NUM_EXPERTS, INTERMEDIATE_SIZE),
            "down_suh": (NUM_EXPERTS, INTERMEDIATE_SIZE),
            "down_svh": (NUM_EXPERTS, HIDDEN_SIZE),
        }
        if any(tuple(tensors[name].shape) != shape for name, shape in shapes.items()):
            raise ValueError("public ShapleyMCG preparation geometry differs")
        if tensors["permutations"].dtype != torch.int64 or any(
            tensors[name].dtype != torch.float16 for name in required - {"permutations"}
        ):
            raise ValueError("public ShapleyMCG preparation dtypes differ")
        result = {"manifest": manifest, "tensors": tensors}
        self._layer_cache[layer] = result
        return result

    def _hidden_chunks(self, capture: Any, row_indices: np.ndarray):
        import torch

        for begin in range(0, row_indices.size, self.chunk_rows):
            end = min(row_indices.size, begin + self.chunk_rows)
            words = np.array(capture.hidden_u16[row_indices[begin:end]], dtype=np.uint16, copy=True)
            yield torch.from_numpy(words).view(torch.bfloat16).to(self.device, dtype=torch.float32).contiguous()

    def _gate_covariance(self, capture: Any, expert: int) -> tuple[Any, Mapping[str, Any]]:
        routed = capture.routed_rows(expert, "fit")
        if routed.rows <= 0:
            raise ValueError(f"L{capture.layer} E{expert}: empty fit rows")
        accumulator = __import__("r7_encoder.hessian", fromlist=["FullCovarianceAccumulator"]).FullCovarianceAccumulator(
            HIDDEN_SIZE, device=self.device, guided=True
        )
        cursor = 0
        for hidden in self._hidden_chunks(capture, routed.row_indices):
            stop = cursor + int(hidden.shape[0])
            weights = np.square(routed.applied_weights[cursor:stop], dtype=np.float32)
            accumulator.add(hidden, weights)
            cursor = stop
        value = accumulator.finalize(self.adapter.sigma_reg, add_damping=False)
        evidence = {
            "construction": "routed-p2-uncentered-second-moment-v1",
            "rows": value.rows,
            "documents": int(np.unique(routed.document_epochs).size),
            "weight_sum": float(value.weight_sum),
            "row_indices_sha256": hashlib.sha256(routed.row_indices.tobytes()).hexdigest(),
            "route_weights_sha256": hashlib.sha256(np.asarray(routed.applied_weights, dtype="<f4").tobytes()).hexdigest(),
            "matrix_sha256": _tensor_sha256(value.matrix),
        }
        return value.matrix, evidence

    def _down_covariance(self, capture: Any, expert: int, gate: Any, up: Any) -> tuple[Any, Mapping[str, Any]]:
        routed = capture.routed_rows(expert, "conditional-fit")
        if routed.rows <= 0:
            raise ValueError(f"L{capture.layer} E{expert}: empty conditional-fit rows")
        hessian = __import__("r7_encoder.hessian", fromlist=["FullCovarianceAccumulator", "down_inputs_from_roundtrip"])
        accumulator = hessian.FullCovarianceAccumulator(INTERMEDIATE_SIZE, device=self.device, guided=True)
        gate_rt = gate.reconstructed_kn.to(self.device)
        up_rt = up.reconstructed_kn.to(self.device)
        cursor = 0
        for hidden in self._hidden_chunks(capture, routed.row_indices):
            stop = cursor + int(hidden.shape[0])
            middle = hessian.down_inputs_from_roundtrip(hidden, gate_rt, up_rt)
            weights = np.square(routed.applied_weights[cursor:stop], dtype=np.float32)
            accumulator.add(middle, weights)
            cursor = stop
        value = accumulator.finalize(self.adapter.sigma_reg, add_damping=False)
        evidence = {
            "construction": f"decoded-k{self.bits}-candidate-conditioned-routed-p2-uncentered-second-moment-v1",
            "rows": value.rows,
            "documents": int(np.unique(routed.document_epochs).size),
            "weight_sum": float(value.weight_sum),
            "row_indices_sha256": hashlib.sha256(routed.row_indices.tobytes()).hexdigest(),
            "route_weights_sha256": hashlib.sha256(np.asarray(routed.applied_weights, dtype="<f4").tobytes()).hexdigest(),
            "gate_reconstruction_sha256": gate.reconstruction_sha256,
            "up_reconstruction_sha256": up.reconstruction_sha256,
            "matrix_sha256": _tensor_sha256(value.matrix),
        }
        return value.matrix, evidence

    def _codec_request(
        self,
        request: EncodeRequest,
        projection: str,
        weight: Any,
        covariance: Any,
        suh: Any,
        svh: Any,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        provenance = {
            "source_payload_sha256": _raw_payload_sha256(request.source_weights[projection]),
            "public_shapleymcg_uniform_rate": self.bits,
            "global_allocator": False,
        }
        provenance.update(dict(extra or {}))
        return {
            "tensor_id": self.adapter._parse_unit(
                f"L{request.layer}.E{request.expert}.{projection}", tuple(weight.shape)
            ),
            "weight_hf": weight,
            "covariance": covariance,
            "bits": (self.bits,),
            "suh": suh,
            "svh": svh,
            "sigma_reg": self.adapter.sigma_reg,
            "provenance": provenance,
        }

    def _save_hessians(
        self,
        request: EncodeRequest,
        gate_up_hessian: Any,
        down_hessian: Any,
        gate_up_evidence: Mapping[str, Any],
        down_evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Persist compact full matrices plus exact FP32 recomputation hashes."""

        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        directory = self.hessian_root / f"layer-{request.layer:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"expert-{request.expert:03d}.safetensors"
        metadata = {
            "schema": "quant-pipeline.glm53-routed-p2-hessian-pair.v1",
            "layer": str(request.layer),
            "expert": str(request.expert),
            "stored_dtype": "float16",
            "gate_up_exact_fp32_sha256": str(gate_up_evidence["matrix_sha256"]),
            "down_exact_fp32_sha256": str(down_evidence["matrix_sha256"]),
            "exact_recomputation": "sealed_capture_routes_plus_decoded_gate_up",
        }
        if path.exists():
            if path.is_symlink():
                raise ValueError(f"Hessian artifact is a symlink: {path}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                if (
                    set(handle.keys()) != {"gate_up_hessian", "down_hessian"}
                    or (handle.metadata() or {}) != metadata
                    or tuple(handle.get_slice("gate_up_hessian").get_shape())
                    != (HIDDEN_SIZE, HIDDEN_SIZE)
                    or tuple(handle.get_slice("down_hessian").get_shape())
                    != (INTERMEDIATE_SIZE, INTERMEDIATE_SIZE)
                ):
                    raise ValueError(f"existing Hessian artifact differs: {path}")
        else:
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            save_file(
                {
                    "gate_up_hessian": torch.as_tensor(gate_up_hessian)
                    .detach()
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous(),
                    "down_hessian": torch.as_tensor(down_hessian)
                    .detach()
                    .to(device="cpu", dtype=torch.float16)
                    .contiguous(),
                },
                str(temporary),
                metadata=metadata,
            )
            os.replace(temporary, path)
        return {
            "schema": metadata["schema"],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "stored_dtype": "float16",
            "gate_up_shape": [HIDDEN_SIZE, HIDDEN_SIZE],
            "down_shape": [INTERMEDIATE_SIZE, INTERMEDIATE_SIZE],
            "gate_up_exact_fp32_sha256": gate_up_evidence["matrix_sha256"],
            "down_exact_fp32_sha256": down_evidence["matrix_sha256"],
            "exact_recomputation_inputs": (
                "sealed_raw_capture_routes_plus_decoded_gate_up_plus_numeric_core"
            ),
        }

    @staticmethod
    def _projection(value: Any, projection: str) -> Mapping[str, Any]:
        import torch

        # Keep the selected expert-internal basis consistently across the
        # packed gate/up/down triplet.  Unpermuting only this dense closure
        # would make it disagree with decode of the persisted packed bytes.
        reconstructed = value.reconstructed_kn.T.half().contiguous()
        return {
            "trellis": value.trellis,
            "suh": value.suh,
            "svh": value.svh,
            "mcg": torch.tensor([MCG_MARKER_SIGNED_INT32], dtype=torch.int32),
            "reconstruction": reconstructed,
            "vector_topology": (
                {"suh": "layer_shared", "svh": "expert_private"}
                if projection in {"gate_proj", "up_proj"}
                else {"suh": "expert_private", "svh": "layer_shared"}
            ),
        }

    def _resolve_batch_size(self, requested: int, cap: int | None) -> int:
        import torch

        upper = requested if cap is None else min(requested, int(cap))
        if upper <= 0:
            raise ValueError("max inflight experts must be positive")
        free, _total = torch.cuda.mem_get_info(torch.device(self.device))
        reserve = 12 * 1024**3
        # Conservative full-path budget: live source triplet, two H4/factors,
        # decoded gate/up and one H2/factor plus lockstep workspaces.
        estimated = 1_250 * 1024**2
        dynamic = max(1, int(max(0, free - reserve) // estimated))
        return max(1, min(upper, dynamic, 32))

    def _encode_batch(self, requests: Sequence[EncodeRequest]) -> list[Mapping[str, Any]]:
        import torch
        from ..normalization.prior_search import permute_expert_hf

        if not requests:
            return []
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        prepared_rows: list[dict[str, Any]] = []
        gate_up_requests: list[Mapping[str, Any]] = []
        for request in requests:
            if request.bits != self.bits or request.expert not in range(NUM_EXPERTS):
                raise ValueError("public ShapleyMCG backend accepts only the contract's fixed rate")
            preparation = self._load_preparation(request.layer)
            tensors = preparation["tensors"]
            expert = request.expert
            permutation = tensors["permutations"][expert].tolist()
            gate_weight, up_weight, down_weight = permute_expert_hf(
                request.source_weights["gate_proj"],
                request.source_weights["up_proj"],
                request.source_weights["down_proj"],
                permutation,
            )
            gate_up_hessian, gate_up_hessian_evidence = self._gate_covariance(
                request.capture, expert
            )
            row = {
                "request": request,
                "preparation": preparation,
                "permutation": tensors["permutations"][expert],
                "weights": {"gate_proj": gate_weight, "up_proj": up_weight, "down_proj": down_weight},
                "gate_up_hessian": gate_up_hessian,
                "gate_up_hessian_evidence": gate_up_hessian_evidence,
            }
            prepared_rows.append(row)
            gate_up_requests.extend(
                (
                    self._codec_request(request, "gate_proj", gate_weight, gate_up_hessian, tensors["gate_suh"][expert], tensors["gate_svh"][expert]),
                    self._codec_request(request, "up_proj", up_weight, gate_up_hessian, tensors["up_suh"][expert], tensors["up_svh"][expert]),
                )
            )
        gate_up_encoded = self.codec.encode_group(gate_up_requests)
        for index, row in enumerate(prepared_rows):
            row["gate"] = gate_up_encoded[2 * index][self.bits]
            row["up"] = gate_up_encoded[2 * index + 1][self.bits]

        # The corrected operation order requires a fresh factor domain for
        # candidate-conditioned H2 after exact gate/up decode.
        cache_before_clear = dict(self.codec.cache_stats)
        self.codec.clear_caches()
        down_requests: list[Mapping[str, Any]] = []
        for row in prepared_rows:
            request = row["request"]
            expert = request.expert
            tensors = row["preparation"]["tensors"]
            h2, evidence = self._down_covariance(request.capture, expert, row["gate"], row["up"])
            row["h2"] = h2
            row["h2_evidence"] = evidence
            row["hessian_artifact"] = self._save_hessians(
                request,
                row["gate_up_hessian"],
                h2,
                row["gate_up_hessian_evidence"],
                evidence,
            )
            down_requests.append(
                self._codec_request(
                    request,
                    "down_proj",
                    row["weights"]["down_proj"],
                    h2,
                    tensors["down_suh"][expert],
                    tensors["down_svh"][expert],
                    {"gate_up_roundtrip_sha256": sha256_bytes(canonical_json({"gate": row["gate"].reconstruction_sha256, "up": row["up"].reconstruction_sha256}))},
                )
            )
        down_encoded = self.codec.encode_group(down_requests)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(torch.device(self.device)))
        results: list[Mapping[str, Any]] = []
        for row, encoded in zip(prepared_rows, down_encoded, strict=True):
            request = row["request"]
            manifest = row["preparation"]["manifest"]
            down = encoded[self.bits]
            evidence = {
                "recipe_id": self.contract["recipe"]["recipe_id"],
                "bits": self.bits,
                "candidate_rate_grid": False,
                "global_allocator": False,
                "codec_family": "exl3-mcg",
                "mcg_multiplier_hex": "0xCBAC1FED",
                "gate_up_hessian": "routed_p2_uncentered_full_hessian",
                "down_hessian": f"decoded_k{self.bits}_candidate_conditioned_routed_p2_uncentered_full_hessian",
                "down_candidate_conditioned": True,
                "profile_source": "public-run-qwen-fast-encode-defaults",
                "profile_policy": "energy_balanced",
                "scale_family": "per128-grid",
                "profile_fixed_before_encoding": True,
                "selection_used_for_profile_choice": False,
                "selection_rows_used_for_encoding": False,
                "confirmation_rows_used_for_choice": False,
                "sqg_orchestration_imported": False,
                "profile_selection_sha256": manifest["profile_selection_sha256"],
                "permutation_sha256": _tensor_sha256(row["permutation"]),
                "gate_up_hessian_sha256": row["gate_up_hessian_evidence"]["matrix_sha256"],
                "down_hessian_sha256": row["h2_evidence"]["matrix_sha256"],
                "decoded_gate_reconstruction_sha256": row["gate"].reconstruction_sha256,
                "decoded_up_reconstruction_sha256": row["up"].reconstruction_sha256,
                "hessian_recomputation": {
                    "gate_up": row["gate_up_hessian_evidence"],
                    "down": row["h2_evidence"],
                    "lossless_inputs": "sealed_raw_capture_plus_packed_gate_up_plus_bf16_source_plus_numeric_core",
                },
                "hessian_artifact": row["hessian_artifact"],
                "corrected_operation_order": [
                    "streaming_absolute_v31_fit",
                    "pinned_per_matrix_gss",
                    "fp16_vector_boundary",
                    f"grouped_gate_up_k{self.bits}_encode",
                    "exact_gate_up_decode",
                    "candidate_conditioned_down_p2_hessian",
                    "clear_gate_up_factor_cache",
                    f"fresh_grouped_down_k{self.bits}_encode",
                ],
                "telemetry": {
                    "elapsed_seconds": elapsed,
                    "batch_experts": len(requests),
                    "experts_per_second": len(requests) / max(elapsed, 1e-30),
                    "matrices_per_second": 3 * len(requests) / max(elapsed, 1e-30),
                    "peak_cuda_memory_bytes": peak,
                    "execution_mode": "r10_equal_k_lockstep",
                    "factor_cache_before_down_clear": cache_before_clear,
                },
            }
            results.append(
                {
                    "projections": {
                        "gate_proj": self._projection(row["gate"], "gate_proj"),
                        "up_proj": self._projection(row["up"], "up_proj"),
                        "down_proj": self._projection(down, "down_proj"),
                    },
                    "recipe_evidence": evidence,
                }
            )
        return results

    def encode_expert(self, request: EncodeRequest) -> Mapping[str, Any]:
        return self._encode_batch((request,))[0]

    def encode_experts(
        self,
        requests: Sequence[EncodeRequest],
        *,
        max_inflight_experts: int | None = None,
    ) -> list[Mapping[str, Any]]:
        if not requests:
            return []
        batch = self._resolve_batch_size(len(requests), max_inflight_experts)
        results: list[Mapping[str, Any]] = []
        for start in range(0, len(requests), batch):
            results.extend(self._encode_batch(requests[start : start + batch]))
        return results
