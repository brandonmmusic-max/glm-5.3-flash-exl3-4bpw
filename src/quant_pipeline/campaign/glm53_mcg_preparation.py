"""Pure public-ShapleyMCG preparation for one GLM-5.3 routed layer.

This module performs only the fixed public-fast transform replay, streaming
v31 absolute normalization and pinned per-matrix GSS used by the public fast
encoder. The transform profile is preregistered before encoding; this module
does not inspect selection rows and never runs a global bit allocator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..codecs.exl3_mcg import Exl3MCGCodec
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from ..normalization.absolute_v31 import MatrixInput
from ..normalization.artifact_v31 import PinnedGSSRequest, tensor_identity_sha256, tensor_sha256
from ..normalization.prior_search import (
    permute_expert_hf,
    policy_permutations,
    scale_family_candidates,
)
from ..normalization.streaming_v31 import FitSamplePlan, FitSampleSpec, StreamingLayerFitter
from .glm53_direct_k4 import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS, Glm53BF16Source
from .glm53_prepared_backend import PREPARATION_SCHEMA
from .qwen_services import CorrectedPinnedGSSProducer


PREPARATION_RECEIPT_SCHEMA = "quant-pipeline.glm53-public-shapleymcg-preparation-receipt.v1"
PROFILE_SELECTION_SCHEMA = "quant-pipeline.glm53-shapleymcg-profile-selection.v1"
HADAMARD_BLOCK = 128
SUPPORTED_BITS = (4, 6)
ZERO_HASH = "0" * 64
_HASH = __import__("re").compile(r"[0-9a-f]{64}")
_PROCESS_STRUCTURE = {
    "driver": "scripts/run_qwen_fast_encode.py",
    "normalization": "src/quant_pipeline/normalization/streaming_v31.py",
    "codec_adapter": "src/quant_pipeline/codecs/exl3_mcg.py",
    "operation_order": "reproducibility/local-corrected-v1",
    "numeric_closure": "r7_encoder",
}


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_selection(value: Mapping[str, Any], *, bits: int) -> tuple[str, str, str]:
    body = copy.deepcopy(dict(value))
    digest = body.pop("selection_sha256", None)
    policy = value.get("policy")
    family = value.get("scale_family")
    if (
        value.get("schema") != PROFILE_SELECTION_SCHEMA
        or not isinstance(digest, str)
        or digest != sha256_bytes(canonical_json(body))
        or policy != "energy_balanced"
        or family != "per128-grid"
        or value.get("bits") != bits
        or value.get("global_allocator_invoked") is not False
        or value.get("profile_source") != "public-run-qwen-fast-encode-defaults"
        or value.get("profile_fixed_before_encoding") is not True
        or value.get("selection_rows_used") is not False
        or value.get("selection_used_for_profile_choice") is not False
        or value.get("selection_used_for_final_encoding") is not False
        or value.get("confirmation_used_for_choice") is not False
        or value.get("candidate_rate_grid_invoked") is not False
        or value.get("proposal_search_invoked") is not False
        or value.get("public_driver") != "scripts/run_qwen_fast_encode.py"
        or value.get("public_shapleymcg_revision")
        != "9d83e7d0baea86604d604502f0d5456c2906486b"
        or value.get("run_qwen_fast_encode_sha256")
        != "ceea8c64d63ffb60cdf95adee3ba7b488c54303d3a85502798b2c3fd0fcbb492"
    ):
        raise ValueError("profile selection receipt is absent, unsealed, or semantically foreign")
    return str(policy), str(family), str(digest)


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


def _sign(length: int, seed: str, *domain: Any):
    import torch

    value = int(sha256_bytes(canonical_json([seed, *domain]))[:16], 16)
    generator = torch.Generator(device="cpu").manual_seed(value)
    return (torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8).float() * 2.0 - 1.0).contiguous()


def _block_values(value: Sequence[float], block: int = HADAMARD_BLOCK) -> tuple[float, ...]:
    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if raw.size % block:
        raise ValueError("profile statistic is not block aligned")
    return tuple(float(max(raw[index : index + block].mean(), 1e-30)) for index in range(0, raw.size, block))


def _hidden_chunks(capture: Any, rows: np.ndarray, *, device: str, chunk_rows: int):
    import torch

    for begin in range(0, rows.size, chunk_rows):
        stop = min(rows.size, begin + chunk_rows)
        words = np.array(capture.hidden_u16[rows[begin:stop]], dtype=np.uint16, copy=True)
        yield begin, stop, torch.from_numpy(words).view(torch.bfloat16).to(device=device, dtype=torch.float32).contiguous()


def _p2_profile_statistics(
    *, capture: Any, source: Glm53BF16Source, layer: int, device: str, chunk_rows: int
) -> dict[str, Any]:
    """Build only exact decision statistics; production full Hessians remain replayable."""

    import torch
    import torch.nn.functional as functional

    gate_diagonal = torch.empty((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32)
    down_diagonal = torch.empty((NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.float32)
    masses = torch.empty(NUM_EXPERTS, dtype=torch.float64)
    down_output_energy = torch.empty((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float32)
    row_hashes: list[dict[str, Any]] = []
    for expert in range(NUM_EXPERTS):
        routed = capture.routed_rows(expert, "fit")
        if routed.rows <= 0:
            raise ValueError(f"L{layer} E{expert}: profile fit has no routed rows")
        weights = np.asarray(routed.applied_weights, dtype=np.float64)
        p2_mass = float(np.square(weights).sum())
        if not math.isfinite(p2_mass) or p2_mass <= 0:
            raise ValueError(f"L{layer} E{expert}: invalid routed p2 mass")
        triplet = source.load_triplet(layer, expert, device=device)
        gate_weight = triplet["gate_proj"].float()
        up_weight = triplet["up_proj"].float()
        gate_sum = torch.zeros(HIDDEN_SIZE, dtype=torch.float64, device=device)
        down_sum = torch.zeros(INTERMEDIATE_SIZE, dtype=torch.float64, device=device)
        for begin, stop, hidden in _hidden_chunks(capture, routed.row_indices, device=device, chunk_rows=chunk_rows):
            p2 = torch.from_numpy(np.square(routed.applied_weights[begin:stop], dtype=np.float32)).to(device=device, dtype=torch.float64)
            gate_sum.add_((hidden.double().square() * p2.unsqueeze(1)).sum(dim=0))
            middle = functional.silu(hidden @ gate_weight.T) * (hidden @ up_weight.T)
            down_sum.add_((middle.double().square() * p2.unsqueeze(1)).sum(dim=0))
        gate_diagonal[expert].copy_((gate_sum / p2_mass).float().cpu())
        down_diagonal[expert].copy_((down_sum / p2_mass).float().cpu())
        masses[expert] = p2_mass
        down_output_energy[expert].copy_(triplet["down_proj"].float().pow(2).mean(dim=1).cpu())
        row_hashes.append(
            {
                "expert": expert,
                "rows": routed.rows,
                "documents": int(np.unique(routed.document_epochs).size),
                "weight_sum": p2_mass,
                "row_indices_sha256": hashlib.sha256(routed.row_indices.tobytes()).hexdigest(),
                "route_weights_sha256": hashlib.sha256(np.asarray(routed.applied_weights, dtype="<f4").tobytes()).hexdigest(),
            }
        )
        del triplet, gate_weight, up_weight, gate_sum, down_sum
    shared_gate = (gate_diagonal.double() * masses.unsqueeze(1)).sum(dim=0) / masses.sum()
    shared_down_output = (down_output_energy.double() * masses.unsqueeze(1)).sum(dim=0) / masses.sum()
    return {
        "gate_diagonal": gate_diagonal,
        "down_diagonal": down_diagonal,
        "masses": masses,
        "down_output_energy": down_output_energy,
        "shared_gate_diagonal": shared_gate.float(),
        "shared_down_output_energy": shared_down_output.float(),
        "row_evidence": row_hashes,
    }


def _matrix_inputs_for_expert(
    *,
    source: Glm53BF16Source,
    layer: int,
    expert: int,
    device: str,
    policy: str,
    family: str,
    seed: str,
    statistics: Mapping[str, Any],
    shared_gate_scales: Sequence[float],
    shared_down_scales: Sequence[float],
    bits: int,
):
    triplet = source.load_triplet(layer, expert, device=device)
    diagonal = statistics["down_diagonal"][expert].tolist()
    permutation = policy_permutations(diagonal, block=HADAMARD_BLOCK)[policy]
    gate, up, down = permute_expert_hf(
        triplet["gate_proj"], triplet["up_proj"], triplet["down_proj"], permutation
    )
    shared_gate_sign = _sign(HIDDEN_SIZE, seed, layer, policy, family, "gate-up-suh").to(device)
    shared_down_sign = _sign(HIDDEN_SIZE, seed, layer, policy, family, "down-svh").to(device)
    mass = float(statistics["masses"][expert].item())
    permuted_down_diag = statistics["down_diagonal"][expert][list(permutation)].tolist()
    rows = []
    for projection, weight, hdiag, suh, svh in (
        ("gate_proj", gate, statistics["gate_diagonal"][expert].tolist(), shared_gate_sign, _sign(INTERMEDIATE_SIZE, seed, layer, expert, policy, family, "gate-svh").to(device)),
        ("up_proj", up, statistics["gate_diagonal"][expert].tolist(), shared_gate_sign, _sign(INTERMEDIATE_SIZE, seed, layer, expert, policy, family, "up-svh").to(device)),
        ("down_proj", down, permuted_down_diag, _sign(INTERMEDIATE_SIZE, seed, layer, expert, policy, family, "down-suh").to(device), shared_down_sign),
    ):
        k_scales = shared_gate_scales if projection != "down_proj" else scale_family_candidates(_block_values(hdiag))[family]
        n_scales = shared_down_scales if projection == "down_proj" else scale_family_candidates(
            _block_values(weight.float().pow(2).mean(dim=1).detach().cpu().numpy())
        )[family]
        rows.append(
            MatrixInput(
                key=f"E{expert}.{projection}",
                projection=projection,
                bits=bits,
                weight_kn=weight.T.contiguous(),
                suh_sign=suh,
                svh_sign=svh,
                k_block_scales=k_scales,
                n_block_scales=n_scales,
                mass=mass,
            )
        )
    return tuple(rows), tuple(permutation)


def _producer_closure(root: Path) -> list[dict[str, str]]:
    campaign_root = Path(__file__).resolve().parents[3]
    paths = [
        root / "scripts/run_qwen_fast_encode.py",
        root / "src/quant_pipeline/normalization/streaming_v31.py",
        root / "src/quant_pipeline/codecs/exl3_mcg.py",
    ]
    paths.extend(sorted((root / "r7_encoder").rglob("*.py")))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("pure preparation source closure is incomplete")
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in paths
    ]
    adapter = campaign_root / "src/quant_pipeline/campaign/glm53_mcg_preparation.py"
    records.append(
        {
            "path": "campaign-adapter/src/quant_pipeline/campaign/glm53_mcg_preparation.py",
            "sha256": sha256_file(adapter),
        }
    )
    return records


def build_layer_preparation(
    *,
    layer: int,
    capture: Any,
    source: Glm53BF16Source,
    source_root: str | Path,
    numeric_core: str | Path,
    extension: str | Path,
    output_root: str | Path,
    transform_seed_sha256: str,
    profile_selection: Mapping[str, Any],
    device: str = "cuda:0",
    chunk_rows: int = 1024,
    bits: int = 4,
) -> dict[str, Any]:
    """Build one immutable layer preparation from a sealed selected profile."""

    import torch
    from safetensors.torch import save_file

    if bits not in SUPPORTED_BITS:
        raise ValueError("GLM-5.3 preparation supports only uniform K4 or K6")
    if layer != int(capture.layer) or _HASH.fullmatch(transform_seed_sha256) is None:
        raise ValueError("layer/capture or transform seed differs")
    policy, family, selection_sha = _verify_selection(profile_selection, bits=bits)
    output = Path(output_root).resolve()
    final_destination = output / f"layer-{layer:03d}"
    manifest_path = final_destination / "preparation.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("profile_selection_sha256") != selection_sha:
            raise ValueError("existing preparation targets another selected profile")
        return existing
    if final_destination.exists():
        raise ValueError(f"incomplete final preparation directory exists: {final_destination}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f".layer-{layer:03d}.staging-{os.getpid()}"
    destination.mkdir(exist_ok=False)
    root = Path(source_root).resolve()
    codec = Exl3MCGCodec(
        source_root=root,
        numeric_core=numeric_core,
        extension=extension,
        device=device,
        sigma_reg=0.025,
    )
    backend = codec._codec()
    producer = CorrectedPinnedGSSProducer(codec)
    statistics = _p2_profile_statistics(
        capture=capture, source=source, layer=layer, device=device, chunk_rows=chunk_rows
    )
    shared_gate_scales = scale_family_candidates(_block_values(statistics["shared_gate_diagonal"].numpy()))[family]
    shared_down_scales = scale_family_candidates(_block_values(statistics["shared_down_output_energy"].numpy()))[family]

    specs: list[FitSampleSpec] = []
    permutations = torch.empty((NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.int64)
    for expert in range(NUM_EXPERTS):
        matrices, permutation = _matrix_inputs_for_expert(
            source=source, layer=layer, expert=expert, device=device, policy=policy,
            family=family, seed=transform_seed_sha256, statistics=statistics,
            shared_gate_scales=shared_gate_scales, shared_down_scales=shared_down_scales,
            bits=bits,
        )
        permutations[expert].copy_(torch.tensor(permutation, dtype=torch.int64))
        specs.extend(FitSampleSpec.from_input(matrix.to("cpu") if False else matrix) for matrix in matrices)
        del matrices
    plan = FitSamplePlan.from_specs(specs, block=HADAMARD_BLOCK)
    fitter = StreamingLayerFitter(
        backend.core,
        plan,
        codebook_scale=float(backend.codebook_scale),
        numeric_core_sha256=codec.identity["numeric_core_sha256"],
    )
    for expert in range(NUM_EXPERTS):
        matrices, _ = _matrix_inputs_for_expert(
            source=source, layer=layer, expert=expert, device=device, policy=policy,
            family=family, seed=transform_seed_sha256, statistics=statistics,
            shared_gate_scales=shared_gate_scales, shared_down_scales=shared_down_scales,
            bits=bits,
        )
        for matrix in matrices:
            fitter.add_fit_matrix(matrix)
        del matrices
    fit = fitter.finish()

    vectors = {
        "gate_suh": torch.empty((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float16),
        "gate_svh": torch.empty((NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.float16),
        "up_suh": torch.empty((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float16),
        "up_svh": torch.empty((NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.float16),
        "down_suh": torch.empty((NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.float16),
        "down_svh": torch.empty((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float16),
    }
    gss_receipts: list[dict[str, Any]] = []
    for expert in range(NUM_EXPERTS):
        matrices, _ = _matrix_inputs_for_expert(
            source=source, layer=layer, expert=expert, device=device, policy=policy,
            family=family, seed=transform_seed_sha256, statistics=statistics,
            shared_gate_scales=shared_gate_scales, shared_down_scales=shared_down_scales,
            bits=bits,
        )
        for matrix in matrices:
            prepared = fit.prepare_matrix(matrix)
            target = prepared.gss_target()
            result = producer.search(
                PinnedGSSRequest(
                    matrix_key=matrix.key,
                    bits=bits,
                    target=target,
                    target_sha256=tensor_sha256(target),
                    source_weight_identity_sha256=tensor_identity_sha256(matrix.weight_kn),
                    predecessor_checkpoint_hash=ZERO_HASH,
                )
            )
            finalized = prepared.finalize(prepared.bind_gss(result.scale), materialize_regularized=False)
            prefix = matrix.projection.removesuffix("_proj")
            vectors[f"{prefix}_suh"][expert].copy_(finalized.stored_suh.detach().cpu())
            vectors[f"{prefix}_svh"][expert].copy_(finalized.stored_svh.detach().cpu())
            gss_receipts.append(
                {
                    "expert": expert,
                    "projection": matrix.projection,
                    "scale": float(result.scale),
                    "receipt_sha256": result.receipt["receipt_sha256"],
                    "suh_sha256": finalized.suh_sha256,
                    "svh_sha256": finalized.svh_sha256,
                }
            )
        del matrices

    shard = destination / "preparation.safetensors"
    save_file(
        {"permutations": permutations, **vectors},
        str(shard),
        metadata={"schema": PREPARATION_SCHEMA, "layer": str(layer), "bits": str(bits)},
    )
    decision_stats = destination / "profile-decision-statistics.safetensors"
    save_file(
        {
            "gate_p2_diagonal": statistics["gate_diagonal"],
            "source_down_p2_diagonal": statistics["down_diagonal"],
            "p2_mass": statistics["masses"],
            "source_down_output_energy": statistics["down_output_energy"],
        },
        str(decision_stats),
        metadata={"schema": PREPARATION_SCHEMA, "purpose": "lossless-selected-transform-decision-statistics"},
    )
    closure = _producer_closure(root)
    body = {
        "schema": PREPARATION_SCHEMA,
        "complete": True,
        "layer": layer,
        "bits": bits,
        "codec_family": "exl3-mcg",
        "policy": policy,
        "scale_family": family,
        "transform_seed_sha256": transform_seed_sha256,
        "profile_selection_sha256": selection_sha,
        "profile_source": "public-run-qwen-fast-encode-defaults",
        "profile_fixed_before_encoding": True,
        "selection_rows_used": False,
        "selection_used_for_profile_choice": False,
        "selection_used_for_final_encoding": False,
        "confirmation_used_for_choice": False,
        "confirmation_report_only": True,
        "global_allocator_invoked": False,
        "candidate_rate_grid_invoked": False,
        "source_closure_sqg_free": True,
        "direct_mcg_entrypoint_reviewed": True,
        "shapleymcg_process_structure": dict(_PROCESS_STRUCTURE),
        "producer_source_closure": closure,
        "producer_source_closure_sha256": sha256_bytes(canonical_json(closure)),
        "codec_identity": codec.identity,
        "streaming_fit_plan_sha256": plan.content_sha256,
        "shared_gate_up_suh_sha256": fit.shared_gate_up_suh_sha256,
        "shared_down_svh_sha256": fit.shared_down_svh_sha256,
        "permutation_set_sha256": _tensor_sha256(permutations),
        "gss_receipts_sha256": sha256_bytes(canonical_json(gss_receipts)),
        "gss_receipt_count": len(gss_receipts),
        "shard": shard.name,
        "shard_sha256": sha256_file(shard),
        "decision_statistics": decision_stats.name,
        "decision_statistics_sha256": sha256_file(decision_stats),
        "profile_fit_row_evidence_sha256": sha256_bytes(canonical_json(statistics["row_evidence"])),
        "exact_production_hessians": "recomputed_from_sealed_raw_capture_and_sealed_packed_gate_up",
    }
    result = _seal(body, "preparation_sha256")
    write_json(destination / "preparation.json", result)
    os.replace(destination, final_destination)
    return result


def seal_campaign_preparation(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not manifests:
        raise ValueError("preparation campaign is empty")
    layers = [int(row["layer"]) for row in manifests]
    if layers != list(range(3, 45)):
        raise ValueError("main preparation must close routed layers 3..44 in order")
    bits = int(manifests[0].get("bits", -1))
    if bits not in SUPPORTED_BITS:
        raise ValueError("preparation campaign must be uniformly K4 or K6")
    for row in manifests:
        body = copy.deepcopy(dict(row))
        digest = body.pop("preparation_sha256", None)
        if (
            row.get("schema") != PREPARATION_SCHEMA
            or not isinstance(digest, str)
            or digest != sha256_bytes(canonical_json(body))
            or row.get("complete") is not True
            or row.get("bits") != bits
            or row.get("codec_family") != "exl3-mcg"
            or row.get("policy") != "energy_balanced"
            or row.get("scale_family") != "per128-grid"
            or row.get("profile_source")
            != "public-run-qwen-fast-encode-defaults"
            or row.get("profile_fixed_before_encoding") is not True
            or row.get("selection_rows_used") is not False
            or row.get("selection_used_for_profile_choice") is not False
            or row.get("selection_used_for_final_encoding") is not False
            or row.get("confirmation_used_for_choice") is not False
            or row.get("global_allocator_invoked") is not False
            or row.get("candidate_rate_grid_invoked") is not False
            or row.get("source_closure_sqg_free") is not True
            or row.get("gss_receipt_count") != NUM_EXPERTS * 3
        ):
            raise ValueError(f"layer {row.get('layer')}: preparation seal differs")
    fixed_profile_digests = {str(row["profile_selection_sha256"]) for row in manifests}
    if len(fixed_profile_digests) != 1:
        raise ValueError("main preparation layers do not share one fixed profile")
    transform_seeds = {str(row["transform_seed_sha256"]) for row in manifests}
    if len(transform_seeds) != 1 or _HASH.fullmatch(next(iter(transform_seeds))) is None:
        raise ValueError("main preparation layers do not share one valid transform seed")
    body = {
        "schema": PREPARATION_RECEIPT_SCHEMA,
        "complete": True,
        "layers": layers,
        "bits": bits,
        "codec_family": "exl3-mcg",
        "global_allocator_invoked": False,
        "transform_seed_sha256": next(iter(transform_seeds)),
        "layer_preparation_sha256": [row["preparation_sha256"] for row in manifests],
        "fixed_profile_receipt_sha256": sha256_bytes(canonical_json([row["profile_selection_sha256"] for row in manifests])),
        "permutation_set_sha256": sha256_bytes(canonical_json([row["permutation_set_sha256"] for row in manifests])),
        "gate_up_p2_set_sha256": sha256_bytes(canonical_json([row["profile_fit_row_evidence_sha256"] for row in manifests])),
        "gss_receipts_sha256": sha256_bytes(canonical_json([row["gss_receipts_sha256"] for row in manifests])),
        "confirmation_report_receipt_sha256": sha256_bytes(canonical_json([row.get("confirmation_report_sha256", ZERO_HASH) for row in manifests])),
        "profile_source": "public-run-qwen-fast-encode-defaults",
        "profile_fixed_before_encoding": True,
        "selection_rows_used": False,
        "selection_used_for_profile_choice": False,
        "selection_used_for_final_encoding": False,
        "confirmation_used_for_choice": False,
        "confirmation_report_only": True,
        "source_closure_sqg_free": True,
        "candidate_rate_grid_invoked": False,
    }
    return _seal(body, "receipt_sha256")
