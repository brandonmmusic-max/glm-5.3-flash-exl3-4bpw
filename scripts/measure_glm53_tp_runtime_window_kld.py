#!/usr/bin/env python3
"""Measure sealed teacher-to-custom-TP-runtime KLD on one qualification window."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.core.artifacts import (
    atomic_write,
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.evaluation.glm53_logits import load_capture_receipt, summarize
from quant_pipeline.runtime.glm53_tp2_exl3 import GLM53_TP2_RUNTIME_SCHEMA


def _sealed_runtime(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(value)
    digest = body.pop("receipt_sha256", None)
    if (
        value.get("schema") != GLM53_TP2_RUNTIME_SCHEMA
        or digest != sha256_bytes(canonical_json(body))
    ):
        raise ValueError("custom TP2 runtime receipt seal differs")
    reference = value.get("reference_logit_parity", {})
    if reference.get("rank_output_identical") is not True:
        raise ValueError("custom TP2 runtime logits are not identical across ranks")
    artifact = reference.get("observed_logits_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("custom TP2 runtime receipt lacks retained logits")
    artifact_path = Path(artifact["path"])
    if sha256_file(artifact_path) != artifact.get("sha256"):
        raise ValueError("retained custom TP2 logits hash differs")
    return value


def _load_slice(path: Path, start: int, stop: int):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_slice("logits")[start:stop].clone()


def _gpu_token_kld(teacher, student, device):
    import torch

    teacher64 = teacher.to(device=device, dtype=torch.float64, non_blocking=True)
    student64 = student.to(device=device, dtype=torch.float64, non_blocking=True)
    if teacher64.shape != student64.shape or teacher64.ndim != 2:
        raise ValueError("teacher/runtime logit geometry differs")
    if not torch.isfinite(teacher64).all() or not torch.isfinite(student64).all():
        raise ValueError("teacher/runtime logits must be finite")
    teacher_logp = torch.log_softmax(teacher64, dim=-1)
    student_logp = torch.log_softmax(student64, dim=-1)
    values = torch.sum(torch.exp(teacher_logp) * (teacher_logp - student_logp), dim=-1)
    matches = int(
        torch.count_nonzero(torch.argmax(teacher64, dim=-1) == torch.argmax(student64, dim=-1))
    )
    return values.cpu().numpy(), matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--chunk-positions", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-mean-kld", type=float, default=0.06)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_positions < 1 or args.max_mean_kld <= 0:
        raise ValueError("chunk positions and KLD threshold must be positive")

    teacher = load_capture_receipt(args.teacher_receipt.resolve(), expected_role="bf16_teacher")
    runtime = _sealed_runtime(args.runtime_receipt.resolve())
    matches = [row for row in teacher["logit_files"] if row["window_id"] == args.window_id]
    if len(matches) != 1:
        raise ValueError(f"teacher window census differs for {args.window_id}")
    teacher_row = matches[0]
    runtime_artifact = runtime["reference_logit_parity"]["observed_logits_artifact"]
    teacher_path = Path(teacher_row["path"])
    runtime_path = Path(runtime_artifact["path"])
    count = int(teacher_row["prediction_positions"])
    vocab_size = int(teacher["vocab_size"])

    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("runtime KLD requires an available CUDA device")
    torch.cuda.set_device(device)
    started = time.monotonic()
    values = np.empty(count, dtype=np.float64)
    top1_matches = 0
    for start in range(0, count, args.chunk_positions):
        stop = min(start + args.chunk_positions, count)
        teacher_logits = _load_slice(teacher_path, start, stop)
        runtime_logits = _load_slice(runtime_path, start, stop)
        expected = (stop - start, vocab_size)
        if teacher_logits.shape != expected or runtime_logits.shape != expected:
            raise ValueError("teacher/runtime qualification-window geometry differs")
        chunk_values, chunk_matches = _gpu_token_kld(teacher_logits, runtime_logits, device)
        values[start:stop] = chunk_values
        top1_matches += chunk_matches

    output = prepare_empty_destination(args.output.resolve())
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    token_path = output / "tokenwise-kld.npy"
    atomic_write(token_path, buffer.getvalue())
    summary = summarize(values)
    report = {
        "schema": "quant-pipeline.glm53-custom-tp2-runtime-window-kld.v1",
        "teacher_receipt_sha256": teacher["receipt_sha256"],
        "runtime_receipt_sha256": runtime["receipt_sha256"],
        "runtime_raw_decoded_parity_passed": runtime["reference_logit_parity"]["passed"],
        "runtime_rank_output_identical": True,
        "window_id": args.window_id,
        "token_ids_sha256": teacher_row["token_ids_sha256"],
        "prediction_positions": count,
        "vocab_size": vocab_size,
        "teacher_logits_sha256": sha256_file(teacher_path),
        "runtime_logits_sha256": runtime_artifact["sha256"],
        "kld_direction": "bf16_teacher_to_custom_tp2_runtime",
        "metric": "tokenwise KL over the sealed qualification window",
        "compute_device": str(device),
        "compute_dtype": "float64",
        "summary": summary,
        "top1_agreement": top1_matches / count,
        "max_mean_kld": args.max_mean_kld,
        "mean_kld_gate_passed": bool(summary["mean"] < args.max_mean_kld),
        "tokenwise_kld_path": str(token_path),
        "tokenwise_kld_bytes": token_path.stat().st_size,
        "tokenwise_kld_sha256": sha256_file(token_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    write_json(output / "kld-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["mean_kld_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
