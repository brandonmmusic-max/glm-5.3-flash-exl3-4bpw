#!/usr/bin/env python3
"""Measure exact tokenwise KL(BF16 teacher || packed student) from sealed captures."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time

import numpy as np

from quant_pipeline.core.artifacts import atomic_write, canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json
from quant_pipeline.evaluation.glm53_logits import load_capture_receipt, summarize


FINAL_WINDOW_IDS = tuple(f"final-{index:04d}" for index in range(25))
FINAL_PREDICTION_POSITIONS = 25 * 2047


def _record_map(receipt):
    return {row["window_id"]: row for row in receipt["logit_files"]}


def _load_slice(path: Path, start: int, stop: int):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_slice("logits")[start:stop].clone()


def _gpu_token_kld(teacher, student, device):
    import torch

    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("teacher/student logit geometry mismatch")
    teacher64 = teacher.to(device=device, dtype=torch.float64, non_blocking=True)
    student64 = student.to(device=device, dtype=torch.float64, non_blocking=True)
    if not torch.isfinite(teacher64).all() or not torch.isfinite(student64).all():
        raise ValueError("teacher/student logits must be finite")
    teacher_logp = torch.log_softmax(teacher64, dim=-1)
    student_logp = torch.log_softmax(student64, dim=-1)
    values = torch.sum(torch.exp(teacher_logp) * (teacher_logp - student_logp), dim=-1)
    matches = int(torch.count_nonzero(torch.argmax(teacher64, dim=-1) == torch.argmax(student64, dim=-1)))
    return values.cpu().numpy(), matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--student-receipt", type=Path, required=True)
    parser.add_argument("--student-label", choices=("uniform-k4", "uniform-k6"), required=True)
    parser.add_argument("--chunk-positions", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.chunk_positions < 1:
        raise ValueError("chunk positions must be positive")
    teacher = load_capture_receipt(args.teacher_receipt.resolve(), expected_role="bf16_teacher")
    student = load_capture_receipt(args.student_receipt.resolve(), expected_role="packed_student")
    if teacher["token_panel_receipt_sha256"] != student["token_panel_receipt_sha256"]:
        raise ValueError("teacher and student captures use different sealed token panels")
    if teacher["vocab_size"] != student["vocab_size"]:
        raise ValueError("teacher and student vocabularies differ")
    if not student.get("runtime_reader_sha256"):
        raise ValueError("packed-student capture is not bound to an exact runtime reader")
    teacher_rows = _record_map(teacher)
    student_rows = _record_map(student)
    if set(teacher_rows) != set(student_rows):
        raise ValueError("teacher and student window sets differ")
    if (
        tuple(sorted(teacher_rows)) != FINAL_WINDOW_IDS
        or any(row.get("role") != "final" for row in teacher_rows.values())
        or any(row.get("role") != "final" for row in student_rows.values())
        or sum(int(row.get("prediction_positions", -1)) for row in teacher_rows.values())
        != FINAL_PREDICTION_POSITIONS
    ):
        raise ValueError("KLD qualification requires the sealed 25-window final panel only")
    for window_id, left in teacher_rows.items():
        right = student_rows[window_id]
        fields = ("document_id", "domain", "role", "token_ids_sha256", "attention_mask_sha256", "prediction_positions")
        if any(left[field] != right[field] for field in fields):
            raise ValueError(f"student capture relabels sealed window {window_id}")
    plan = {
        "schema": "quant-pipeline.glm53-packed-student-kld-plan.v1",
        "teacher_receipt_sha256": teacher["receipt_sha256"],
        "student_receipt_sha256": student["receipt_sha256"],
        "student_label": args.student_label,
        "kld_direction": "teacher_to_student",
        "chunk_positions": args.chunk_positions,
        "compute_device": args.device,
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    started = time.monotonic()
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("packed-student KLD is pinned to an available CUDA device")
    torch.cuda.set_device(device)
    output = prepare_empty_destination(args.output.resolve())
    token_values = []
    top1_matches = 0
    per_window = []
    for window_id in sorted(teacher_rows):
        teacher_row = teacher_rows[window_id]
        student_row = student_rows[window_id]
        count = int(teacher_row["prediction_positions"])
        values = np.empty(count, dtype=np.float64)
        for start in range(0, count, args.chunk_positions):
            stop = min(start + args.chunk_positions, count)
            teacher_logits = _load_slice(Path(teacher_row["path"]), start, stop)
            student_logits = _load_slice(Path(student_row["path"]), start, stop)
            if teacher_logits.shape != (stop - start, int(teacher["vocab_size"])) or student_logits.shape != teacher_logits.shape:
                raise ValueError(f"logit geometry mismatch in {window_id}")
            chunk_values, chunk_matches = _gpu_token_kld(teacher_logits, student_logits, device)
            values[start:stop] = chunk_values
            top1_matches += chunk_matches
        token_values.append(values)
        per_window.append(
            {
                "window_id": window_id,
                "document_id": teacher_row["document_id"],
                "domain": teacher_row["domain"],
                "role": teacher_row["role"],
                "summary": summarize(values),
            }
        )
    all_values = np.concatenate(token_values)
    buffer = io.BytesIO()
    np.save(buffer, all_values, allow_pickle=False)
    token_path = output / "tokenwise-kld.npy"
    atomic_write(token_path, buffer.getvalue())
    domains = {}
    for domain in sorted({row["domain"] for row in per_window}):
        indices = [index for index, row in enumerate(per_window) if row["domain"] == domain]
        domains[domain] = summarize(np.concatenate([token_values[index] for index in indices]))
    overall = summarize(all_values)
    report = {
        "schema": "quant-pipeline.glm53-packed-student-kld.v1",
        "teacher_receipt_sha256": teacher["receipt_sha256"],
        "student_receipt_sha256": student["receipt_sha256"],
        "student_label": args.student_label,
        "student_checkpoint_identity_sha256": student["checkpoint_identity_sha256"],
        "runtime_reader_sha256": student["runtime_reader_sha256"],
        "token_panel_receipt_sha256": teacher["token_panel_receipt_sha256"],
        "teacher_backend_identity_sha256": teacher["backend_identity_sha256"],
        "student_backend_identity_sha256": student["backend_identity_sha256"],
        "qualification_panel_final_only": True,
        "qualification_window_count": len(per_window),
        "kld_direction": "teacher_to_student",
        "metric": "tokenwise KL over exact jointly-valid causal prediction positions",
        "compute_device": str(device),
        "compute_dtype": "float64",
        "summary": overall,
        "per_domain": domains,
        "per_window": per_window,
        "top1_agreement": top1_matches / int(all_values.size),
        "mean_kld_lt_0_06": bool(overall["mean"] < 0.06),
        "tokenwise_kld_path": str(token_path.resolve()),
        "tokenwise_kld_bytes": token_path.stat().st_size,
        "tokenwise_kld_sha256": sha256_file(token_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    write_json(output / "kld-report.json", report)
    print(json.dumps({"ok": True, **report}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
