"""Sealed uniform-K6 follow-on plan for GLM-5.3 ShapleyMCG.

K6 is a separate fixed-rate encode over every routed expert, including MTP45.
It reuses the sealed BF16 inventory, raw calibration routes and numerical
process, but requires K6-specific GSS preparation and is gated on the packed
K4 KL qualification.  This module plans work only; it has no CUDA path.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from ..core.artifacts import canonical_json, sha256_bytes
from . import glm53_uniform_k4 as k4


LAUNCH_PLAN_SCHEMA = "quant-pipeline.glm53-uniform-k6-four-b200-launch-plan.v1"
BITS = 6


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_seal(value: Mapping[str, Any], field: str) -> str:
    if value.get("schema") != LAUNCH_PLAN_SCHEMA:
        raise ValueError("uniform K6 launch-plan schema differs")
    body = copy.deepcopy(dict(value))
    digest = body.pop(field, None)
    if not isinstance(digest, str) or digest != sha256_bytes(canonical_json(body)):
        raise ValueError("uniform K6 launch-plan seal differs")
    return digest


def build_launch_plan(
    inventory: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    k4_plan: Mapping[str, Any],
    k4_authorized_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable K6/TP4 plan after the K4 KL gate."""

    k4.verify_launch_plan(k4_plan)
    k4.verify_state(k4_plan, k4_authorized_state)
    if (
        k4_authorized_state.get("phase") != "k6_authorized"
        or k4_authorized_state.get("k6_authorized") is not True
        or not isinstance(
            k4_authorized_state.get("evidence", {}).get(
                "k4_packed_kld_receipt_sha256"
            ),
            str,
        )
    ):
        raise ValueError("K6 planning requires the sealed packed-K4 KL gate")
    main_rows, mtp_rows, native_rows = k4._inventory_surfaces(inventory)
    inventory_sha = str(inventory["inventory_sha256"])
    workers = k4._b200_workers(preflight, inventory_sha)
    by_layer: dict[int, list[dict[str, Any]]] = {
        layer: [] for layer in k4.MAIN_ROUTED_LAYERS
    }
    tensor_contract: list[dict[str, Any]] = []
    for row in main_rows:
        match = k4._ROUTED.fullmatch(str(row["tensor_name"]))
        assert match is not None
        layer = int(match.group(1))
        by_layer[layer].append(row)
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": BITS,
                "disposition": "uniform_k6_direct_encode",
                "execution_track": "main_dynamic_layer_scheduler",
            }
        )
    for row in mtp_rows:
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": BITS,
                "disposition": "uniform_k6_separately_qualified_mtp_adapter",
                "execution_track": "mtp_adapter_after_main",
            }
        )
    units = []
    for layer, rows in by_layer.items():
        names = sorted(str(row["tensor_name"]) for row in rows)
        units.append(
            {
                "layer": layer,
                "expert_count": k4.ROUTED_EXPERTS,
                "matrix_count": len(rows),
                "source_bytes": sum(int(row["source_bytes"]) for row in rows),
                "source_elements": sum(math.prod(row["shape"]) for row in rows),
                "tensor_names_sha256": sha256_bytes(canonical_json(names)),
                "bits": BITS,
                "allowed_bits": [BITS],
                "global_allocator": False,
                "candidate_rate_grid": False,
            }
        )
    mtp_names = sorted(str(row["tensor_name"]) for row in mtp_rows)
    native_names = sorted(str(row["tensor_name"]) for row in native_rows)
    queue = [
        row["layer"]
        for row in sorted(
            units, key=lambda row: (-int(row["source_bytes"]), int(row["layer"]))
        )
    ]
    body = {
        "schema": LAUNCH_PLAN_SCHEMA,
        "model_revision": inventory.get("model_revision"),
        "inventory_sha256": inventory_sha,
        "preflight_sha256": preflight["preflight_sha256"],
        "k4_launch_plan_sha256": k4_plan["launch_plan_sha256"],
        "k4_authorized_state_sha256": k4_authorized_state["state_sha256"],
        "k4_packed_kld_receipt_sha256": k4_authorized_state["evidence"][
            "k4_packed_kld_receipt_sha256"
        ],
        "launch_authorized": False,
        "profile": "k6-tp4",
        "runtime_target": {"tensor_parallel_size": 4, "physical_b200s": 4},
        "geometry": {
            "main_layers": list(k4.MAIN_ROUTED_LAYERS),
            "mtp_layers": list(k4.MTP_LAYERS),
            "routed_experts": k4.ROUTED_EXPERTS,
            "projections": list(k4.PROJECTIONS),
        },
        "rate_contract": {
            "allocation": "none_uniform_fixed_rate",
            "bits": BITS,
            "allowed_bits": [BITS],
            "global_allocator_invoked": False,
            "candidate_rate_grid_invoked": False,
            "K6": k4.ALL_ROUTED_MATRIX_COUNT,
            "main_routed_matrix_count": k4.MAIN_ROUTED_MATRIX_COUNT,
            "mtp_routed_k6_matrix_count": k4.MTP_ROUTED_MATRIX_COUNT,
            "all_main_plus_mtp_routed_matrix_count": k4.ALL_ROUTED_MATRIX_COUNT,
            "main_must_complete_before_mtp": True,
            "mtp_may_not_remain_native_in_final_k6": True,
        },
        "preparation_contract": {
            "reuse_raw_calibration_and_routes": True,
            "reuse_fixed_policy_and_permutations": True,
            "k6_specific_gss_required": True,
            "reuse_k4_gss_forbidden": True,
            "candidate_conditioned_down_uses_decoded_k6_gate_up": True,
        },
        "native_copy_contract": {
            "policy": "byte_exact_source_copy",
            "includes_all_nonrouted": True,
            "includes_routed_mtp": False,
            "tensor_count": len(native_rows),
            "source_bytes": sum(int(row["source_bytes"]) for row in native_rows),
            "tensor_names_sha256": sha256_bytes(canonical_json(native_names)),
        },
        "routed_tensor_contract": tensor_contract,
        "work_units": units,
        "mtp_work_unit": {
            "layer": k4.MTP_LAYERS[0],
            "expert_count": k4.ROUTED_EXPERTS,
            "matrix_count": len(mtp_rows),
            "source_bytes": sum(int(row["source_bytes"]) for row in mtp_rows),
            "source_elements": sum(math.prod(row["shape"]) for row in mtp_rows),
            "tensor_names_sha256": sha256_bytes(canonical_json(mtp_names)),
            "bits": BITS,
            "allowed_bits": [BITS],
            "adapter_qualification_required": True,
            "scheduled_with_main_layers": False,
        },
        "scheduler": {
            "policy": "dynamic_next_unclaimed_expert_ranges",
            "workers": workers,
            "initial_queue": queue,
        },
        "evaluation": {
            "teacher": "same_sealed_bf16_logits_and_token_panel_as_k4",
            "student": "packed_uniform_k6_full_vocab_logits",
            "direction": "KL(BF16||K6)",
            "tokenwise_kl_required": True,
        },
    }
    if len(tensor_contract) != k4.ALL_ROUTED_MATRIX_COUNT:
        raise AssertionError("uniform K6 routed tensor census drift")
    return _seal(body, "launch_plan_sha256")


def verify_launch_plan(plan: Mapping[str, Any]) -> str:
    digest = _verify_seal(plan, "launch_plan_sha256")
    rate = plan.get("rate_contract", {})
    prep = plan.get("preparation_contract", {})
    if (
        plan.get("profile") != "k6-tp4"
        or plan.get("runtime_target", {}).get("tensor_parallel_size") != 4
        or rate.get("allowed_bits") != [6]
        or rate.get("K6") != k4.ALL_ROUTED_MATRIX_COUNT
        or rate.get("global_allocator_invoked") is not False
        or rate.get("candidate_rate_grid_invoked") is not False
        or prep.get("k6_specific_gss_required") is not True
        or prep.get("reuse_k4_gss_forbidden") is not True
        or prep.get("candidate_conditioned_down_uses_decoded_k6_gate_up") is not True
        or plan.get("launch_authorized") is not False
    ):
        raise ValueError("uniform K6 plan invariant differs")
    return digest
