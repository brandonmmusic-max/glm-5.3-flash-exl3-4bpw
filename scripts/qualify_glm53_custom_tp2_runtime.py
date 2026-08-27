#!/usr/bin/env python3
"""Qualify the custom packed GLM-5.3 K4/TP2 or K6/TP4 text runtime.

This script never upgrades the materializer's storage receipt.  It writes a
separate runtime receipt and qualifies only when the complete packed load,
multi-token generation, and explicit full-logit reference tolerance all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from quant_pipeline.core.artifacts import sha256_file, write_json
from quant_pipeline.runtime.glm53_tp2_exl3 import (
    build_runtime_receipt,
    packed_runtime_census,
    patch_transformers,
    target_tp_size_for_bits,
)
from quant_pipeline.publication.glm53_k4_postmtp import validate_reference_tolerances


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 6))
    parser.add_argument("--exllamav3-source", type=Path, required=True)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--max-abs-tolerance", type=float, required=True)
    parser.add_argument("--mean-abs-tolerance", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--observed-logits-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reference_panel(path: Path):
    import torch

    if path.suffix != ".safetensors":
        raise ValueError("reference panel must be safetensors")
    from safetensors import safe_open
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    required = {"input_ids", "logits"}
    if not required <= set(tensors) or set(tensors) - required - {"attention_mask", "prediction_indices"}:
        raise ValueError("reference panel has an unsupported tensor set")
    if tensors["input_ids"].dtype not in (torch.int64, torch.int32):
        raise ValueError("reference input_ids must be an integer tensor")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    return tensors, metadata


def _reference_error(value, observed):
    value = value.float()
    observed = observed.detach().float().cpu()
    if tuple(value.shape) != tuple(observed.shape):
        raise ValueError(f"reference shape differs: {tuple(value.shape)} != {tuple(observed.shape)}")
    error = (observed - value).abs()
    return float(error.max()), float(error.mean()), observed


def main() -> int:
    args = _args()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    declared_bits = config.get("quantization_config", {}).get("bits")
    if declared_bits not in (4, 6):
        raise ValueError("model config does not declare uniform routed K4 or K6")
    bits = declared_bits if args.bits is None else args.bits
    if bits != declared_bits:
        raise ValueError(f"requested K{bits} differs from checkpoint K{declared_bits}")
    tp_size = target_tp_size_for_bits(bits)
    if args.max_new_tokens < 2:
        raise ValueError("qualification requires at least two decode steps")
    if args.max_abs_tolerance < 0 or args.mean_abs_tolerance < 0:
        raise ValueError("parity tolerances must be nonnegative")

    patch_transformers(exllamav3_source=args.exllamav3_source)

    import torch
    import torch.distributed as dist
    from transformers import Glm5NextForConditionalGeneration
    from transformers.distributed import DistributedConfig

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)

    panel, panel_metadata = _reference_panel(args.reference_panel)
    validate_reference_tolerances(
        panel_metadata,
        max_abs=args.max_abs_tolerance,
        mean_abs=args.mean_abs_tolerance,
    )
    attention_backend = panel_metadata.get("attention_backend")
    if attention_backend not in {"eager", "sdpa"}:
        raise ValueError("reference panel lacks its measured attention backend")

    model = Glm5NextForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(
            tp_size=tp_size, tp_plan="auto", enable_expert_parallel=False
        ),
        attn_implementation=attention_backend,
        local_files_only=True,
    )
    device = torch.device("cuda", rank)
    input_ids = panel["input_ids"].to(device)
    attention_mask = panel.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    observed_logits = output.logits
    prediction_indices = panel.get("prediction_indices")
    if prediction_indices is not None:
        if prediction_indices.dtype not in (torch.int64, torch.int32) or prediction_indices.ndim != 1:
            raise ValueError("reference prediction_indices must be one integer vector")
        observed_logits = observed_logits[:, :-1, :][0].index_select(
            0, prediction_indices.to(device=device, dtype=torch.int64)
        )
    max_abs, mean_abs, observed_cpu = _reference_error(panel["logits"], observed_logits)
    observed_sha256 = hashlib.sha256(memoryview(observed_cpu.numpy())).hexdigest()
    rank_output_sha256 = [None] * world_size
    dist.all_gather_object(rank_output_sha256, observed_sha256)
    rank_output_identical = len(set(rank_output_sha256)) == 1
    observed_artifact = None
    if rank == 0 and args.observed_logits_output is not None:
        from safetensors.torch import save_file

        args.observed_logits_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"logits": observed_cpu.contiguous()},
            args.observed_logits_output,
            metadata={
                "capture_role": "packed_tp_runtime",
                "bits": str(bits),
                "tp_size": str(tp_size),
                "reference_panel_sha256": sha256_file(args.reference_panel),
                "raw_tensor_sha256": observed_sha256,
            },
        )
        observed_artifact = {
            "path": str(args.observed_logits_output.resolve()),
            "sha256": sha256_file(args.observed_logits_output),
            "raw_tensor_sha256": observed_sha256,
            "bytes": args.observed_logits_output.stat().st_size,
        }
    reference = {
        "path": str(args.reference_panel.resolve()),
        "sha256": sha256_file(args.reference_panel),
        "input_ids_shape": list(input_ids.shape),
        "shape": list(observed_logits.shape),
        "reference_schema": panel_metadata.get("schema"),
        "reference_checkpoint_identity_sha256": panel_metadata.get("checkpoint_identity_sha256"),
        "reference_runtime_reader_sha256": panel_metadata.get("runtime_reader_sha256"),
        "attention_backend": attention_backend,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_abs_tolerance": args.max_abs_tolerance,
        "mean_abs_tolerance": args.mean_abs_tolerance,
        "rank_output_sha256": rank_output_sha256,
        "rank_output_identical": rank_output_identical,
        "observed_logits_artifact": observed_artifact,
        "passed": max_abs <= args.max_abs_tolerance and mean_abs <= args.mean_abs_tolerance,
    }
    census = packed_runtime_census(model)
    local = {
        "rank": rank,
        "world_size": world_size,
        "bits": bits,
        "device": torch.cuda.get_device_name(rank),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(rank)),
        "steady_cuda_bytes": int(torch.cuda.memory_allocated(rank)),
        "packed_matrix_count": census["packed_matrix_count"],
        "bf16_routed_weight_parameter_count": census["bf16_routed_weight_parameter_count"],
        "generated_token_count": int(generated.shape[-1] - input_ids.shape[-1]),
    }
    reports = [None] * world_size
    dist.all_gather_object(reports, local)
    qualified = False
    if rank == 0:
        receipt = build_runtime_receipt(
            rank_reports=reports,
            runtime_module=Path(__file__).parents[1] / "src/quant_pipeline/runtime/glm53_tp2_exl3.py",
            exllamav3_source=args.exllamav3_source,
            reference=reference,
            generation_verified=all(row["generated_token_count"] == args.max_new_tokens for row in reports),
            bits=bits,
            tp_size=tp_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        qualified = bool(receipt["qualified"])
    qualified_rows = [None] * world_size
    dist.all_gather_object(qualified_rows, qualified if rank == 0 else None)
    qualified = next(value for value in qualified_rows if value is not None)
    dist.barrier()
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
