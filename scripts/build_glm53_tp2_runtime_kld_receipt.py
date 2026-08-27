#!/usr/bin/env python3
"""Seal fail-closed K4/TP2 qualification from actual-runtime teacher KLD."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, write_json
from quant_pipeline.publication.glm53_k4_postmtp import REFERENCE_PANEL_SCHEMA, read_sealed
from quant_pipeline.runtime.glm53_tp2_exl3 import GLM53_TP2_RUNTIME_SCHEMA


SCHEMA = "quant-pipeline.glm53-custom-transformers-exl3-mcg-tp2-runtime-kld-qualified.v1"
KLD_SCHEMA = "quant-pipeline.glm53-custom-tp2-runtime-window-kld.v1"


def _read_sealed(path: Path, schema: str, field: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(value)
    digest = body.pop(field, None)
    if value.get("schema") != schema or digest != sha256_bytes(canonical_json(body)):
        raise ValueError(f"sealed {schema} receipt differs: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--runtime-kld-report", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime = _read_sealed(args.runtime_receipt.resolve(), GLM53_TP2_RUNTIME_SCHEMA, "receipt_sha256")
    kld = _read_sealed(args.runtime_kld_report.resolve(), KLD_SCHEMA, "report_sha256")
    reference = read_sealed(
        args.reference_receipt.resolve(), REFERENCE_PANEL_SCHEMA, "receipt_sha256"
    )
    materialization = _read_sealed(
        args.materialization_receipt.resolve(),
        "quant-pipeline.glm53-k4-materialization-receipt.v1",
        "receipt_sha256",
    )
    parity = runtime.get("reference_logit_parity", {})
    ranks = runtime.get("rank_reports", [])
    reasons = []
    if (
        runtime.get("bits") != 4
        or runtime.get("tp_size") != 2
        or runtime.get("generation_verified") is not True
        or runtime.get("main_packed_matrix_count") != 36288
        or len(ranks) != 2
        or [row.get("rank") for row in ranks] != [0, 1]
        or any(row.get("packed_matrix_count") != 36288 for row in ranks)
        or any(row.get("bf16_routed_weight_parameter_count") != 0 for row in ranks)
        or any(row.get("generated_token_count", 0) < 2 for row in ranks)
    ):
        reasons.append("packed TP2 load, generation, or rank census is incomplete")
    if (
        parity.get("rank_output_identical") is not True
        or len(set(parity.get("rank_output_sha256", []))) != 1
        or not isinstance(parity.get("observed_logits_artifact"), dict)
    ):
        reasons.append("full runtime logits are not retained and identical across TP ranks")
    if (
        parity.get("sha256") != reference.get("sha256")
        or parity.get("input_ids_shape") != reference.get("input_shape")
        or parity.get("shape") != reference.get("logit_shape")
    ):
        reasons.append("runtime evaluation is not bound to the sealed decoded reference window")
    if (
        kld.get("runtime_receipt_sha256") != runtime["receipt_sha256"]
        or kld.get("runtime_logits_sha256")
        != parity.get("observed_logits_artifact", {}).get("sha256")
        or kld.get("runtime_rank_output_identical") is not True
        or kld.get("window_id") != reference.get("window_id")
        or kld.get("token_ids_sha256") != reference.get("token_ids_sha256")
        or kld.get("prediction_positions") != reference.get("prediction_positions")
        or kld.get("max_mean_kld") != 0.06
        or kld.get("mean_kld_gate_passed") is not True
        or float(kld.get("summary", {}).get("mean", float("inf"))) >= 0.06
    ):
        reasons.append("actual-runtime BF16-teacher KLD gate did not pass below 0.06")
    if (
        materialization.get("bits") != 4
        or materialization.get("complete") is not True
        or materialization.get("main_and_mtp_complete") is not True
        or materialization.get("nonrouted_native_exact") is not True
    ):
        reasons.append("materialized K4 main/MTP checkpoint is incomplete")

    receipt = {
        "schema": SCHEMA,
        "bits": 4,
        "tp_size": 2,
        "qualification_basis": "actual_runtime_teacher_kld_plus_cross_rank_identity_and_generation",
        "materialization_receipt_sha256": materialization["receipt_sha256"],
        "raw_runtime_receipt_sha256": runtime["receipt_sha256"],
        "runtime_module_sha256": runtime["runtime_module_sha256"],
        "exllamav3_commit": runtime["exllamav3_commit"],
        "generation_verified": runtime.get("generation_verified"),
        "rank_reports": ranks,
        "rank_output_identical": parity.get("rank_output_identical"),
        "rank_output_sha256": parity.get("rank_output_sha256"),
        "runtime_logits_artifact": parity.get("observed_logits_artifact"),
        "decoded_reference": {
            "receipt_sha256": reference["receipt_sha256"],
            "sha256": reference["sha256"],
            "raw_parity_passed": parity.get("passed"),
            "max_abs_error": parity.get("max_abs_error"),
            "mean_abs_error": parity.get("mean_abs_error"),
            "max_abs_tolerance": parity.get("max_abs_tolerance"),
            "mean_abs_tolerance": parity.get("mean_abs_tolerance"),
        },
        "teacher_runtime_kld": {
            "report_sha256": kld["report_sha256"],
            "teacher_receipt_sha256": kld["teacher_receipt_sha256"],
            "window_id": kld["window_id"],
            "prediction_positions": kld["prediction_positions"],
            "mean": kld["summary"]["mean"],
            "p95": kld["summary"]["p95"],
            "max": kld["summary"]["max"],
            "top1_agreement": kld["top1_agreement"],
            "threshold": 0.06,
            "passed": kld["mean_kld_gate_passed"],
        },
        "raw_decoded_parity_is_not_the_qualification_basis": True,
        "stock_vllm_compatible": False,
        "stock_exllamav3_model_compatible": False,
        "qualified": not reasons,
        "failure_reasons": reasons,
        "example_only": False,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
