"""Fail-closed receipts for the uniform-K4 post-MTP release path."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..campaign.glm53_direct_k4 import (
    CONTRACT_SCHEMA,
    MATERIALIZATION_PLAN_SCHEMA,
    MATERIALIZATION_RECEIPT_SCHEMA,
)
from ..campaign.glm53_uniform_k4 import PACKED_KLD_SCHEMA
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file
from ..evaluation.glm53_logits import load_capture_receipt


NATIVE_COPY_SCHEMA = "quant-pipeline.glm53-k4-native-copy-materialization-bridge.v1"
READER_ABI_SCHEMA = "quant-pipeline.glm53-exl3-reader-abi-receipt.v1"
REFERENCE_PANEL_SCHEMA = "quant-pipeline.glm53-decoded-k4-tp2-reference-panel.v1"
FIVE_RUN_KLD_SCHEMA = "quant-pipeline.glm53-packed-student-kld-five-cold-run.v1"
SOURCE_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
_HASH = re.compile(r"[0-9a-f]{64}")


def seal(value: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def verify_seal(value: Mapping[str, Any], schema: str, field: str) -> str:
    body = copy.deepcopy(dict(value))
    digest = body.pop(field, None)
    if (
        value.get("schema") != schema
        or not isinstance(digest, str)
        or _HASH.fullmatch(digest) is None
        or digest != sha256_bytes(canonical_json(body))
    ):
        raise ValueError(f"invalid sealed {schema}")
    return digest


def read_sealed(path: str | Path, schema: str, field: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("example_only") is True:
        raise ValueError(f"not an executable receipt: {path}")
    verify_seal(value, schema, field)
    return value


def build_native_copy_bridge(
    *,
    materialization: Mapping[str, Any],
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    materialization_file_sha256: str,
    full_shard_hash_verification: bool,
) -> dict[str, Any]:
    """Bind the materializer's exact native-copy census to ``seal-k4``."""

    materialization_sha = verify_seal(
        materialization, MATERIALIZATION_RECEIPT_SCHEMA, "receipt_sha256"
    )
    plan_sha = verify_seal(plan, MATERIALIZATION_PLAN_SCHEMA, "plan_sha256")
    contract_sha = verify_seal(contract, CONTRACT_SCHEMA, "contract_sha256")
    inventory_sha = verify_seal(
        inventory, "quant-pipeline.glm-release-inventory.v1", "inventory_sha256"
    )
    if (
        not full_shard_hash_verification
        or materialization.get("plan_sha256") != plan_sha
        or materialization.get("source_inventory_sha256") != inventory_sha
        or materialization.get("source_model_revision") != SOURCE_REVISION
        or materialization.get("bits") != 4
        or materialization.get("nonrouted_native_exact") is not True
        or materialization.get("main_and_mtp_complete") is not True
        or materialization.get("complete") is not True
        or plan.get("contract_sha256") != contract_sha
        or plan.get("inventory_sha256") != inventory_sha
        or not isinstance(materialization.get("native_tensor_count"), int)
        or materialization.get("native_tensor_count") != plan.get("native_tensor_count")
        or _HASH.fullmatch(materialization_file_sha256) is None
    ):
        raise ValueError("K4 materialization does not prove an exact native-copy closure")
    return seal(
        {
            "schema": NATIVE_COPY_SCHEMA,
            "profile": "k4-tp2",
            "target_bits": 4,
            "codec_family": "exl3-mcg",
            "global_allocator_invoked": False,
            "candidate_rate_grid_invoked": False,
            "source_revision": SOURCE_REVISION,
            "inventory_sha256": inventory_sha,
            "contract_sha256": contract_sha,
            "materialization_plan_sha256": plan_sha,
            "materialization_receipt_sha256": materialization_sha,
            "materialization_receipt_file_sha256": materialization_file_sha256,
            "native_tensor_count": materialization["native_tensor_count"],
            "native_tensor_names_sha256": plan["native_tensor_names_sha256"],
            "nonrouted_native_exact": True,
            "full_shard_hash_verification": True,
            "qualified": True,
        }
    )


def _capture_set_sha256(receipt: Mapping[str, Any]) -> str:
    rows = [
        {
            "window_id": row["window_id"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in receipt["logit_files"]
    ]
    return sha256_bytes(canonical_json(rows))


def build_packed_k4_kld_receipt(
    *,
    materialization: Mapping[str, Any],
    native_copy: Mapping[str, Any],
    reader_abi: Mapping[str, Any],
    student_backend: Mapping[str, Any],
    teacher_receipt_path: str | Path,
    student_receipt_path: str | Path,
    kld_report: Mapping[str, Any],
    kld_report_file_sha256: str,
) -> dict[str, Any]:
    """Upgrade measured KLD into the exact campaign qualification receipt."""

    materialization_sha = verify_seal(
        materialization, MATERIALIZATION_RECEIPT_SCHEMA, "receipt_sha256"
    )
    native_sha = verify_seal(native_copy, NATIVE_COPY_SCHEMA, "receipt_sha256")
    reader_sha = verify_seal(reader_abi, READER_ABI_SCHEMA, "receipt_sha256")
    teacher = load_capture_receipt(teacher_receipt_path, expected_role="bf16_teacher")
    student = load_capture_receipt(student_receipt_path, expected_role="packed_student")
    backend_body = copy.deepcopy(dict(student_backend))
    backend_sha = backend_body.pop("backend_identity_sha256", None)
    report_sha = verify_seal(
        kld_report, "quant-pipeline.glm53-packed-student-kld.v1", "report_sha256"
    )
    token_path = Path(str(kld_report.get("tokenwise_kld_path", "")))
    if (
        materialization.get("bits") != 4
        or materialization.get("nonrouted_native_exact") is not True
        or native_copy.get("materialization_receipt_sha256") != materialization_sha
        or native_copy.get("qualified") is not True
        or reader_abi.get("qualified") is not True
        or reader_abi.get("bits") != 4
        or reader_abi.get("tp_sizes") != [2, 4]
        or reader_abi.get("exact_reconstruction_checked") is not True
        or student_backend.get("schema")
        != "quant-pipeline.glm53-packed-k4-offline-reader-backend.v1"
        or backend_sha != sha256_bytes(canonical_json(backend_body))
        or backend_sha != student.get("backend_identity_sha256")
        or student_backend.get("model_revision") != SOURCE_REVISION
        or student_backend.get("checkpoint_identity_sha256")
        != student.get("checkpoint_identity_sha256")
        or student_backend.get("runtime_reader_sha256")
        != student.get("runtime_reader_sha256")
        or student_backend.get("packed_reader_abi_sha256")
        != reader_abi.get("reader_sha256")
        or student_backend.get("attention_backend") not in {"eager", "sdpa"}
        or teacher.get("token_panel_receipt_sha256")
        != student.get("token_panel_receipt_sha256")
        or kld_report.get("teacher_receipt_sha256") != teacher["receipt_sha256"]
        or kld_report.get("student_receipt_sha256") != student["receipt_sha256"]
        or kld_report.get("token_panel_receipt_sha256")
        != teacher["token_panel_receipt_sha256"]
        or kld_report.get("teacher_backend_identity_sha256")
        != teacher.get("backend_identity_sha256")
        or kld_report.get("student_backend_identity_sha256") != backend_sha
        or kld_report.get("student_label") != "uniform-k4"
        or "uniform-k4" not in str(student.get("weight_dtype", ""))
        or teacher.get("model_revision") != SOURCE_REVISION
        or student.get("model_revision") != SOURCE_REVISION
        or kld_report.get("student_checkpoint_identity_sha256")
        != student.get("checkpoint_identity_sha256")
        or kld_report.get("runtime_reader_sha256") != student.get("runtime_reader_sha256")
        or kld_report.get("kld_direction") != "teacher_to_student"
        or kld_report.get("compute_dtype") != "float64"
        or not isinstance(kld_report.get("summary"), Mapping)
        or not isinstance(kld_report["summary"].get("mean"), (int, float))
        or bool(kld_report.get("mean_kld_lt_0_06"))
        != (float(kld_report["summary"]["mean"]) < 0.06)
        or not token_path.is_file()
        or token_path.is_symlink()
        or token_path.stat().st_size != kld_report.get("tokenwise_kld_bytes")
        or sha256_file(token_path) != kld_report.get("tokenwise_kld_sha256")
    ):
        raise ValueError("measured packed-K4 KLD closure differs")
    if _HASH.fullmatch(kld_report_file_sha256) is None:
        raise ValueError("KLD report file hash is invalid")
    passed = float(kld_report["summary"]["mean"]) < 0.06
    return seal(
        {
            "schema": PACKED_KLD_SCHEMA,
            "profile": "k4-tp2",
            "target_bits": 4,
            "codec_family": "exl3-mcg",
            "global_allocator_invoked": False,
            "candidate_rate_grid_invoked": False,
            "qualified": passed,
            "source_revision": SOURCE_REVISION,
            "kld_direction": "teacher_to_student",
            "same_token_panel": True,
            "reader_audit_qualified": True,
            "quality_gate_passed": passed,
            "quality_gate": {"metric": "mean_tokenwise_kld", "threshold_lt": 0.06},
            "measured_mean_kld": float(kld_report["summary"]["mean"]),
            "checkpoint_receipt_sha256": materialization_sha,
            "native_copy_receipt_sha256": native_sha,
            "token_panel_receipt_sha256": teacher["token_panel_receipt_sha256"],
            "reader_audit_receipt_sha256": reader_sha,
            "packed_reader_abi_sha256": reader_abi["reader_sha256"],
            "student_backend_identity_sha256": backend_sha,
            "runtime_reader_sha256": student["runtime_reader_sha256"],
            "student_checkpoint_identity_sha256": student["checkpoint_identity_sha256"],
            "teacher_capture_receipt_sha256": teacher["receipt_sha256"],
            "student_capture_receipt_sha256": student["receipt_sha256"],
            "kld_report_sha256": report_sha,
            "kld_report_file_sha256": kld_report_file_sha256,
            "evidence_artifacts": {
                "teacher_logits": _capture_set_sha256(teacher),
                "final_student_logits": _capture_set_sha256(student),
                "tokenwise_kl": kld_report["tokenwise_kld_sha256"],
            },
        }
    )


def build_five_run_kld_receipt(
    reports: Sequence[tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    """Seal the requested five-cold-execution mean without collapsing evidence."""

    if len(reports) != 5:
        raise ValueError("five-run KLD qualification requires exactly five reports")
    rows: list[dict[str, Any]] = []
    common_fields = (
        "teacher_receipt_sha256",
        "token_panel_receipt_sha256",
        "student_checkpoint_identity_sha256",
        "runtime_reader_sha256",
        "teacher_backend_identity_sha256",
        "student_label",
        "kld_direction",
        "compute_dtype",
    )
    common: dict[str, Any] | None = None
    for index, (report, file_sha256) in enumerate(reports, start=1):
        report_sha = verify_seal(
            report, "quant-pipeline.glm53-packed-student-kld.v1", "report_sha256"
        )
        summary = report.get("summary")
        if (
            not isinstance(summary, Mapping)
            or not isinstance(summary.get("mean"), (int, float))
            or not isinstance(summary.get("count"), int)
            or summary["count"] <= 0
            or _HASH.fullmatch(file_sha256) is None
            or _HASH.fullmatch(str(report.get("student_backend_identity_sha256"))) is None
            or report.get("student_label") != "uniform-k4"
            or report.get("kld_direction") != "teacher_to_student"
            or report.get("compute_dtype") != "float64"
        ):
            raise ValueError(f"cold KLD run {index} is not a complete uniform-K4 report")
        observed_common = {field: report.get(field) for field in common_fields}
        if common is None:
            common = observed_common
        elif observed_common != common:
            raise ValueError("cold KLD reports do not bind one checkpoint, panel, and reader")
        mean = float(summary["mean"])
        passed = mean < 0.06 and report.get("mean_kld_lt_0_06") is True
        rows.append(
            {
                "run": index,
                "report_sha256": report_sha,
                "report_file_sha256": file_sha256,
                "student_capture_receipt_sha256": report.get("student_receipt_sha256"),
                "student_backend_identity_sha256": report.get(
                    "student_backend_identity_sha256"
                ),
                "tokenwise_kld_sha256": report.get("tokenwise_kld_sha256"),
                "prediction_positions": summary["count"],
                "mean_kld": mean,
                "quality_gate_passed": passed,
            }
        )
    assert common is not None
    student_receipts = {row["student_capture_receipt_sha256"] for row in rows}
    if len(student_receipts) != 5 or any(
        _HASH.fullmatch(str(value)) is None for value in student_receipts
    ):
        raise ValueError("five-run KLD evidence does not contain five distinct cold captures")
    if len({row["prediction_positions"] for row in rows}) != 1:
        raise ValueError("five-run KLD reports have different panel sizes")
    means = [row["mean_kld"] for row in rows]
    mean_of_means = sum(means) / len(means)
    variance = sum((value - mean_of_means) ** 2 for value in means) / len(means)
    passed = all(row["quality_gate_passed"] for row in rows) and mean_of_means < 0.06
    return seal(
        {
            "schema": FIVE_RUN_KLD_SCHEMA,
            "profile": "k4-tp2",
            "target_bits": 4,
            "run_count": 5,
            "cold_execution_count": 5,
            **common,
            "quality_gate": {
                "metric": "mean_of_five_run_mean_tokenwise_kld",
                "threshold_lt": 0.06,
            },
            "mean_of_run_means": mean_of_means,
            "population_stddev_of_run_means": variance**0.5,
            "minimum_run_mean": min(means),
            "maximum_run_mean": max(means),
            "all_individual_quality_gates_passed": all(
                row["quality_gate_passed"] for row in rows
            ),
            "quality_gate_passed": passed,
            "qualified": passed,
            "runs": rows,
        }
    )


def validate_reference_tolerances(
    metadata: Mapping[str, str], *, max_abs: float, mean_abs: float
) -> None:
    if metadata.get("schema") != REFERENCE_PANEL_SCHEMA:
        raise ValueError("foreign decoded-K4 TP2 reference panel")
    try:
        expected_max = float(metadata["max_abs_tolerance"])
        expected_mean = float(metadata["mean_abs_tolerance"])
    except (KeyError, ValueError) as error:
        raise ValueError("reference panel lacks predeclared tolerances") from error
    if max_abs != expected_max or mean_abs != expected_mean:
        raise ValueError("TP2 qualification tolerances differ from the predeclared panel")


def capture_set_sha256(receipt: Mapping[str, Any]) -> str:
    """Public helper for publication manifests."""

    return _capture_set_sha256(receipt)
