"""Sealed GLM-5.3 uniform-K4 execution plan and receipt state machine.

This module is deliberately orchestration-only.  It never starts a process,
imports CUDA, or invokes an allocator.  The released full-shard inventory is
the source of truth for the exact routed tensor surface.  Main-model routed
experts are fixed at K4 first; the MTP routed layer is also K4, but only through
its separately qualified adapter.  Every non-routed tensor remains source-native.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

from ..core.artifacts import canonical_json, sha256_bytes


INVENTORY_SCHEMA = "quant-pipeline.glm-release-inventory.v1"
PREFLIGHT_SCHEMA = "quant-pipeline.glm53-sm100-preflight.v1"
LAUNCH_PLAN_SCHEMA = "quant-pipeline.glm53-uniform-k4-four-b200-launch-plan.v1"
STATE_RECEIPT_SCHEMA = "quant-pipeline.glm53-uniform-k4-state-receipt.v1"
CLAIM_RECEIPT_SCHEMA = "quant-pipeline.glm53-uniform-k4-layer-claim.v1"
PACKED_KLD_SCHEMA = "quant-pipeline.glm53-packed-kld-receipt.v1"
MTP_ADAPTER_RECEIPT_SCHEMA = "quant-pipeline.glm53-uniform-k4-mtp-adapter-receipt.v1"

MAIN_LAYER_COUNT = 45
FIRST_MOE_LAYER = 3
MAIN_ROUTED_LAYERS = tuple(range(FIRST_MOE_LAYER, MAIN_LAYER_COUNT))
MTP_LAYERS = (MAIN_LAYER_COUNT,)
ROUTED_EXPERTS = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MAIN_ROUTED_MATRIX_COUNT = len(MAIN_ROUTED_LAYERS) * ROUTED_EXPERTS * len(PROJECTIONS)
MTP_ROUTED_MATRIX_COUNT = len(MTP_LAYERS) * ROUTED_EXPERTS * len(PROJECTIONS)
ALL_ROUTED_MATRIX_COUNT = MAIN_ROUTED_MATRIX_COUNT + MTP_ROUTED_MATRIX_COUNT
WORKERS = 4

_ROUTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_HASH = re.compile(r"[0-9a-f]{64}")


def _seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-hex hash")
    return value


def _verify_seal(document: Mapping[str, Any], *, schema: str, field: str, label: str) -> str:
    if document.get("schema") != schema:
        raise ValueError(f"{label} schema differs")
    seal = _require_hash(document.get(field), f"{label}.{field}")
    body = copy.deepcopy(dict(document))
    del body[field]
    if sha256_bytes(canonical_json(body)) != seal:
        raise ValueError(f"{label} seal differs")
    return seal


def _inventory_surfaces(
    inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _verify_seal(
        inventory,
        schema=INVENTORY_SCHEMA,
        field="inventory_sha256",
        label="full source inventory",
    )
    if inventory.get("seal_mode") != "full-shard-sha256":
        raise ValueError("uniform K4 execution requires a full-shard SHA256 inventory")
    revision = inventory.get("model_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("uniform K4 execution requires an immutable model revision")
    geometry = inventory.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("inventory geometry is absent")
    required_geometry = {
        "model_type": "glm5_next",
        "main_layers": MAIN_LAYER_COUNT,
        "mtp_layers": len(MTP_LAYERS),
        "first_moe_layer": FIRST_MOE_LAYER,
        "routed_experts": ROUTED_EXPERTS,
    }
    for key, expected in required_geometry.items():
        if geometry.get(key) != expected:
            raise ValueError(f"released GLM-5.3 geometry {key} differs: {geometry.get(key)!r} != {expected!r}")
    if geometry.get("discovered_layers") != list(range(MAIN_LAYER_COUNT + len(MTP_LAYERS))):
        raise ValueError("inventory does not discover the complete main plus MTP layer surface")

    tensors = inventory.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("inventory tensor surface is absent")
    names: set[str] = set()
    main: dict[tuple[int, int, str], dict[str, Any]] = {}
    mtp: dict[tuple[int, int, str], dict[str, Any]] = {}
    native: list[dict[str, Any]] = []
    for raw in tensors:
        if not isinstance(raw, Mapping):
            raise ValueError("inventory contains a malformed tensor row")
        row = dict(raw)
        name = row.get("tensor_name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("inventory tensor names are absent or duplicated")
        names.add(name)
        source_bytes = row.get("source_bytes")
        shape = row.get("shape")
        if (
            isinstance(source_bytes, bool)
            or not isinstance(source_bytes, int)
            or source_bytes < 0
            or not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
        ):
            raise ValueError(f"inventory tensor geometry is malformed: {name}")
        match = _ROUTED.fullmatch(name)
        scope = row.get("scope")
        if scope in {"routed_expert", "mtp_routed_expert"} and match is None:
            raise ValueError(f"routed scope has a noncanonical tensor name: {name}")
        if match is None:
            native.append(row)
            continue
        key = (int(match.group(1)), int(match.group(2)), match.group(3))
        if scope == "routed_expert":
            if key in main:
                raise ValueError(f"duplicate main routed tensor: {name}")
            if row.get("dtype") != "BF16" or len(shape) != 2 or any(value % 128 for value in shape):
                raise ValueError(f"main routed tensor is not aligned BF16: {name}")
            main[key] = row
        elif scope == "mtp_routed_expert":
            if key in mtp:
                raise ValueError(f"duplicate MTP routed tensor: {name}")
            mtp[key] = row
        else:
            raise ValueError(f"canonical expert tensor has a non-routed scope: {name}")

    expected_main = {
        (layer, expert, projection)
        for layer in MAIN_ROUTED_LAYERS
        for expert in range(ROUTED_EXPERTS)
        for projection in PROJECTIONS
    }
    expected_mtp = {
        (layer, expert, projection)
        for layer in MTP_LAYERS
        for expert in range(ROUTED_EXPERTS)
        for projection in PROJECTIONS
    }
    if set(main) != expected_main:
        missing = sorted(expected_main - set(main))
        extra = sorted(set(main) - expected_main)
        raise ValueError(f"main routed K4 surface differs: missing={missing[:3]}, extra={extra[:3]}")
    if set(mtp) != expected_mtp:
        missing = sorted(expected_mtp - set(mtp))
        extra = sorted(set(mtp) - expected_mtp)
        raise ValueError(f"native MTP routed surface differs: missing={missing[:3]}, extra={extra[:3]}")
    return (
        [main[key] for key in sorted(main)],
        [mtp[key] for key in sorted(mtp)],
        sorted(native, key=lambda row: row["tensor_name"]),
    )


def _b200_workers(preflight: Mapping[str, Any], inventory_sha256: str) -> list[dict[str, Any]]:
    preflight_sha = _verify_seal(
        preflight,
        schema=PREFLIGHT_SCHEMA,
        field="preflight_sha256",
        label="four-B200 preflight",
    )
    if (
        preflight.get("ready") is not True
        or preflight.get("mode") != "layer-streaming"
        or preflight.get("checkpoint_seal_mode") != "full-shard-sha256"
        or preflight.get("checkpoint_inventory_sha256") != inventory_sha256
        or preflight.get("workers") != WORKERS
    ):
        raise ValueError("four-B200 layer-streaming preflight is not execution-ready")
    gpus = preflight.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != WORKERS:
        raise ValueError("launch plan requires exactly four preflight GPUs")
    workers: list[dict[str, Any]] = []
    indices: set[int] = set()
    for slot, raw in enumerate(gpus):
        if not isinstance(raw, Mapping):
            raise ValueError("preflight GPU row is malformed")
        index = raw.get("index")
        name = raw.get("name")
        capability = raw.get("compute_capability")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in indices
            or not isinstance(name, str)
            or "B200" not in name
            or not isinstance(capability, str)
            or not capability.startswith("10.")
        ):
            raise ValueError("uniform K4 workers must be four distinct attested B200 devices")
        indices.add(index)
        workers.append(
            {
                "worker_id": f"b200-{slot}",
                "physical_gpu": index,
                "cuda_visible_devices": str(index),
                "codec_device": "cuda:0",
                "name": name,
                "compute_capability": capability,
                "preflight_sha256": preflight_sha,
            }
        )
    return workers


def build_launch_plan(inventory: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Build the immutable, no-launch K4 work contract from sealed evidence."""

    main_rows, mtp_rows, native_rows = _inventory_surfaces(inventory)
    inventory_sha = str(inventory["inventory_sha256"])
    workers = _b200_workers(preflight, inventory_sha)
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in MAIN_ROUTED_LAYERS}
    tensor_contract: list[dict[str, Any]] = []
    for row in main_rows:
        match = _ROUTED.fullmatch(row["tensor_name"])
        assert match is not None
        layer = int(match.group(1))
        by_layer[layer].append(row)
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": 4,
                "disposition": "uniform_k4_direct_encode",
                "execution_track": "main_dynamic_layer_scheduler",
            }
        )
    for row in mtp_rows:
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": 4,
                "disposition": "uniform_k4_separately_qualified_mtp_adapter",
                "execution_track": "mtp_adapter_after_main",
            }
        )

    units: list[dict[str, Any]] = []
    for layer, rows in by_layer.items():
        names = sorted(row["tensor_name"] for row in rows)
        if len(rows) != ROUTED_EXPERTS * len(PROJECTIONS):
            raise AssertionError("validated routed layer lost matrices")
        units.append(
            {
                "layer": layer,
                "expert_count": ROUTED_EXPERTS,
                "matrix_count": len(rows),
                "source_bytes": sum(int(row["source_bytes"]) for row in rows),
                "source_elements": sum(math.prod(row["shape"]) for row in rows),
                "tensor_names_sha256": sha256_bytes(canonical_json(names)),
                "bits": 4,
                "allowed_bits": [4],
                "global_allocator": False,
                "candidate_rate_grid": False,
            }
        )
    queue = [
        row["layer"]
        for row in sorted(units, key=lambda row: (-int(row["source_bytes"]), int(row["layer"])))
    ]
    native_names = [row["tensor_name"] for row in native_rows]
    mtp_names = [row["tensor_name"] for row in mtp_rows]
    body = {
        "schema": LAUNCH_PLAN_SCHEMA,
        "model_revision": inventory.get("model_revision"),
        "inventory_sha256": inventory_sha,
        "preflight_sha256": preflight["preflight_sha256"],
        "launch_authorized": False,
        "boundary": "sealed planning and receipt transitions only; this document starts no process",
        "profile": "k4-tp2",
        "runtime_target": {"tensor_parallel_size": 2, "physical_b200s": WORKERS},
        "geometry": {
            "main_layers": list(MAIN_ROUTED_LAYERS),
            "mtp_layers": list(MTP_LAYERS),
            "routed_experts": ROUTED_EXPERTS,
            "projections": list(PROJECTIONS),
        },
        "rate_contract": {
            "allocation": "none_uniform_fixed_rate",
            "global_allocator_invoked": False,
            "candidate_rate_grid_invoked": False,
            "K3": 0,
            "K4": ALL_ROUTED_MATRIX_COUNT,
            "main_routed_matrix_count": MAIN_ROUTED_MATRIX_COUNT,
            "mtp_routed_k4_matrix_count": MTP_ROUTED_MATRIX_COUNT,
            "all_main_plus_mtp_routed_matrix_count": ALL_ROUTED_MATRIX_COUNT,
            "main_must_complete_before_mtp": True,
            "mtp_may_not_remain_native_in_final_k4": True,
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
            "layer": MTP_LAYERS[0],
            "expert_count": ROUTED_EXPERTS,
            "matrix_count": len(mtp_rows),
            "source_bytes": sum(int(row["source_bytes"]) for row in mtp_rows),
            "source_elements": sum(math.prod(row["shape"]) for row in mtp_rows),
            "tensor_names_sha256": sha256_bytes(canonical_json(mtp_names)),
            "bits": 4,
            "allowed_bits": [4],
            "adapter_qualification_required": True,
            "scheduled_with_main_layers": False,
        },
        "scheduler": {
            "policy": "dynamic_next_unclaimed_whole_layer",
            "static_layer_partition_forbidden": True,
            "one_active_layer_per_worker": True,
            "workers": workers,
            "initial_queue": queue,
        },
        "sequential_states": [
            "planned",
            "k4_main_encoding",
            "k4_main_encoded",
            "k4_mtp_qualified",
            "k4_packed",
            "k4_kld_qualified",
            "k6_authorized",
        ],
        "k6_gate": {
            "initially_authorized": False,
            "required_predecessor_state": "k4_kld_qualified",
            "required_receipt_schema": PACKED_KLD_SCHEMA,
            "requires_same_packed_k4_checkpoint": True,
            "requires_same_token_panel": True,
            "requires_reader_audit": True,
        },
    }
    if len(tensor_contract) != ALL_ROUTED_MATRIX_COUNT or len(mtp_rows) != MTP_ROUTED_MATRIX_COUNT:
        raise AssertionError("uniform K4 main/MTP census drift")
    return _seal(body, "launch_plan_sha256")


def verify_launch_plan(plan: Mapping[str, Any]) -> str:
    seal = _verify_seal(
        plan,
        schema=LAUNCH_PLAN_SCHEMA,
        field="launch_plan_sha256",
        label="uniform K4 launch plan",
    )
    rate = plan.get("rate_contract")
    if not isinstance(rate, Mapping) or (
        rate.get("K3"), rate.get("K4"), rate.get("mtp_routed_k4_matrix_count")
    ) != (0, ALL_ROUTED_MATRIX_COUNT, MTP_ROUTED_MATRIX_COUNT):
        raise ValueError("uniform K4 launch-plan rate census differs")
    if plan.get("launch_authorized") is not False:
        raise ValueError("a planning receipt may not authorize process launch")
    return seal


def initial_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan_sha = verify_launch_plan(plan)
    body = {
        "schema": STATE_RECEIPT_SCHEMA,
        "launch_plan_sha256": plan_sha,
        "sequence": 0,
        "previous_state_receipt_sha256": None,
        "phase": "planned",
        "pending_layers": list(plan["scheduler"]["initial_queue"]),
        "active_claims": {},
        "completed_layers": {},
        "evidence": {},
        "k6_authorized": False,
    }
    return _seal(body, "state_receipt_sha256")


def verify_state(plan: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    plan_sha = verify_launch_plan(plan)
    state_sha = _verify_seal(
        state,
        schema=STATE_RECEIPT_SCHEMA,
        field="state_receipt_sha256",
        label="uniform K4 state receipt",
    )
    if state.get("launch_plan_sha256") != plan_sha:
        raise ValueError("state receipt targets a different launch plan")
    pending = state.get("pending_layers")
    active = state.get("active_claims")
    completed = state.get("completed_layers")
    if not isinstance(pending, list) or not isinstance(active, Mapping) or not isinstance(completed, Mapping):
        raise ValueError("state scheduler domains are malformed")
    sequence = state.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("state sequence is malformed")
    predecessor = state.get("previous_state_receipt_sha256")
    if sequence == 0 and predecessor is not None:
        raise ValueError("initial state may not name a predecessor")
    if sequence > 0:
        _require_hash(predecessor, "state predecessor receipt")
    domain = set(MAIN_ROUTED_LAYERS)
    pending_set = set(pending)
    active_layers = {row.get("layer") for row in active.values() if isinstance(row, Mapping)}
    try:
        complete_set = {int(layer) for layer in completed}
    except (TypeError, ValueError) as exc:
        raise ValueError("completed layer keys are malformed") from exc
    if (
        len(pending) != len(pending_set)
        or len(active_layers) != len(active)
        or pending_set & active_layers
        or pending_set & complete_set
        or active_layers & complete_set
        or pending_set | active_layers | complete_set != domain
    ):
        raise ValueError("state layer partition does not close exactly")
    worker_ids = {row["worker_id"] for row in plan["scheduler"]["workers"]}
    if set(active) - worker_ids:
        raise ValueError("state has a claim for an unknown B200 worker")
    for worker_id, claim in active.items():
        if not isinstance(claim, Mapping):
            raise ValueError("state contains a malformed active claim")
        _verify_seal(
            claim,
            schema=CLAIM_RECEIPT_SCHEMA,
            field="claim_receipt_sha256",
            label="active layer claim",
        )
        if (
            claim.get("launch_plan_sha256") != plan_sha
            or claim.get("worker_id") != worker_id
            or claim.get("bits") != 4
        ):
            raise ValueError("active layer claim binding differs")
    for layer, completion in completed.items():
        if not isinstance(completion, Mapping) or completion.get("worker_id") not in worker_ids:
            raise ValueError(f"completed layer {layer} receipt is malformed")
        _require_hash(completion.get("claim_receipt_sha256"), f"completed layer {layer} claim")
        _require_hash(completion.get("layer_receipt_sha256"), f"completed layer {layer} artifact")
    phase = state.get("phase")
    if phase not in plan["sequential_states"]:
        raise ValueError("state phase is outside the launch plan")
    if phase != "k6_authorized" and state.get("k6_authorized") is not False:
        raise ValueError("K6 is authorized before the K4 KLD gate")
    if phase == "k6_authorized" and state.get("k6_authorized") is not True:
        raise ValueError("K6 authorization state is internally inconsistent")
    evidence = state.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("state evidence is malformed")
    if phase == "planned" and (
        list(pending) != list(plan["scheduler"]["initial_queue"])
        or active
        or completed
        or evidence
        or sequence != 0
    ):
        raise ValueError("planned state contains premature execution evidence")
    if phase != "planned":
        _require_hash(evidence.get("k4_readiness_receipt_sha256"), "K4 readiness receipt")
    closed_main_phases = {
        "k4_main_encoded",
        "k4_mtp_qualified",
        "k4_packed",
        "k4_kld_qualified",
        "k6_authorized",
    }
    if phase in closed_main_phases and (
        pending or active or complete_set != set(MAIN_ROUTED_LAYERS)
    ):
        raise ValueError("post-main state does not close all main K4 layers")
    if phase in closed_main_phases:
        _require_hash(evidence.get("main_routed_receipt_sha256"), "main routed K4 receipt")
    if phase in {"k4_mtp_qualified", "k4_packed", "k4_kld_qualified", "k6_authorized"}:
        _require_hash(evidence.get("mtp_k4_adapter_receipt_sha256"), "MTP K4 adapter receipt")
    if phase in {"k4_packed", "k4_kld_qualified", "k6_authorized"}:
        _require_hash(evidence.get("packed_checkpoint_receipt_sha256"), "packed K4 checkpoint receipt")
        _require_hash(evidence.get("native_copy_receipt_sha256"), "native non-routed copy receipt")
    if phase in {"k4_kld_qualified", "k6_authorized"}:
        _require_hash(evidence.get("k4_packed_kld_receipt_sha256"), "packed K4 KLD receipt")
    return state_sha


def _successor(plan: Mapping[str, Any], state: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    previous = verify_state(plan, state)
    body = copy.deepcopy(dict(state))
    del body["state_receipt_sha256"]
    body.update(copy.deepcopy(updates))
    body["sequence"] = int(state["sequence"]) + 1
    body["previous_state_receipt_sha256"] = previous
    return _seal(body, "state_receipt_sha256")


def enter_k4_encoding(
    plan: Mapping[str, Any], state: Mapping[str, Any], *, readiness_receipt_sha256: str
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "planned":
        raise ValueError("K4 encoding may start only from planned")
    evidence = dict(state.get("evidence", {}))
    evidence["k4_readiness_receipt_sha256"] = _require_hash(
        readiness_receipt_sha256, "K4 readiness receipt"
    )
    return _successor(plan, state, phase="k4_main_encoding", evidence=evidence)


def claim_next_layer(
    plan: Mapping[str, Any], state: Mapping[str, Any], *, worker_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_state(plan, state)
    if state.get("phase") != "k4_main_encoding":
        raise ValueError("layer claims are allowed only during K4 encoding")
    workers = {row["worker_id"] for row in plan["scheduler"]["workers"]}
    if worker_id not in workers:
        raise ValueError("unknown B200 worker")
    active = copy.deepcopy(dict(state["active_claims"]))
    if worker_id in active:
        raise ValueError("worker already owns an active layer")
    pending = list(state["pending_layers"])
    if not pending:
        raise ValueError("no unclaimed K4 layers remain")
    layer = int(pending.pop(0))
    unit = next(row for row in plan["work_units"] if row["layer"] == layer)
    claim = _seal(
        {
            "schema": CLAIM_RECEIPT_SCHEMA,
            "launch_plan_sha256": plan["launch_plan_sha256"],
            "parent_state_receipt_sha256": state["state_receipt_sha256"],
            "worker_id": worker_id,
            "layer": layer,
            "tensor_names_sha256": unit["tensor_names_sha256"],
            "bits": 4,
        },
        "claim_receipt_sha256",
    )
    active[worker_id] = claim
    successor = _successor(plan, state, pending_layers=pending, active_claims=active)
    return successor, claim


def complete_layer(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    worker_id: str,
    layer: int,
    layer_receipt_sha256: str,
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k4_main_encoding":
        raise ValueError("layers may complete only during K4 encoding")
    active = copy.deepcopy(dict(state["active_claims"]))
    claim = active.get(worker_id)
    if not isinstance(claim, Mapping) or claim.get("layer") != layer:
        raise ValueError("layer completion does not match the worker claim")
    _verify_seal(
        claim,
        schema=CLAIM_RECEIPT_SCHEMA,
        field="claim_receipt_sha256",
        label="layer claim",
    )
    completed = copy.deepcopy(dict(state["completed_layers"]))
    completed[str(layer)] = {
        "worker_id": worker_id,
        "claim_receipt_sha256": claim["claim_receipt_sha256"],
        "layer_receipt_sha256": _require_hash(layer_receipt_sha256, "K4 layer receipt"),
    }
    del active[worker_id]
    return _successor(plan, state, active_claims=active, completed_layers=completed)


def seal_main_k4(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    main_routed_receipt_sha256: str,
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k4_main_encoding":
        raise ValueError("main K4 may seal only after main K4 encoding")
    if state["pending_layers"] or state["active_claims"] or len(state["completed_layers"]) != len(MAIN_ROUTED_LAYERS):
        raise ValueError("main K4 sealing is blocked until every main routed layer completes")
    evidence = dict(state.get("evidence", {}))
    evidence["main_routed_receipt_sha256"] = _require_hash(
        main_routed_receipt_sha256, "main routed K4 receipt"
    )
    return _successor(plan, state, phase="k4_main_encoded", evidence=evidence)


def verify_mtp_adapter_receipt(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> str:
    seal = _verify_seal(
        receipt,
        schema=MTP_ADAPTER_RECEIPT_SCHEMA,
        field="receipt_sha256",
        label="uniform K4 MTP adapter receipt",
    )
    required = {
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
        "layer": MTP_LAYERS[0],
        "expert_count": ROUTED_EXPERTS,
        "matrix_count": MTP_ROUTED_MATRIX_COUNT,
        "bits": 4,
        "qualified": True,
        "tensor_names_sha256": plan["mtp_work_unit"]["tensor_names_sha256"],
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"uniform K4 MTP adapter {key} differs")
    _require_hash(receipt.get("codec_adapter_sha256"), "MTP codec adapter")
    _require_hash(receipt.get("packed_payload_receipt_sha256"), "MTP packed payload receipt")
    return seal


def qualify_mtp_k4(
    plan: Mapping[str, Any], state: Mapping[str, Any], *, mtp_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k4_main_encoded":
        raise ValueError("MTP K4 adapter may qualify only after all main K4 layers")
    mtp_sha = verify_mtp_adapter_receipt(plan, mtp_receipt)
    evidence = dict(state.get("evidence", {}))
    evidence["mtp_k4_adapter_receipt_sha256"] = mtp_sha
    return _successor(plan, state, phase="k4_mtp_qualified", evidence=evidence)


def seal_k4_packed(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    packed_checkpoint_receipt_sha256: str,
    native_copy_receipt_sha256: str,
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k4_mtp_qualified":
        raise ValueError("final K4 packing requires separately qualified MTP K4")
    evidence = dict(state.get("evidence", {}))
    _require_hash(evidence.get("main_routed_receipt_sha256"), "main routed K4 receipt")
    _require_hash(evidence.get("mtp_k4_adapter_receipt_sha256"), "MTP K4 adapter receipt")
    evidence.update(
        {
            "packed_checkpoint_receipt_sha256": _require_hash(
                packed_checkpoint_receipt_sha256, "packed K4 checkpoint receipt"
            ),
            "native_copy_receipt_sha256": _require_hash(
                native_copy_receipt_sha256, "native non-routed copy receipt"
            ),
        }
    )
    return _successor(plan, state, phase="k4_packed", evidence=evidence)


def verify_k4_kld_receipt(receipt: Mapping[str, Any], *, packed_checkpoint_receipt_sha256: str) -> str:
    seal = _verify_seal(
        receipt,
        schema=PACKED_KLD_SCHEMA,
        field="receipt_sha256",
        label="packed K4 KLD receipt",
    )
    required = {
        "profile": "k4-tp2",
        "target_bits": 4,
        "qualified": True,
        "kld_direction": "teacher_to_student",
        "same_token_panel": True,
        "reader_audit_qualified": True,
        "quality_gate_passed": True,
        "checkpoint_receipt_sha256": packed_checkpoint_receipt_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"packed K4 KLD qualification {key} differs")
    _require_hash(receipt.get("token_panel_receipt_sha256"), "K4 KLD token-panel receipt")
    _require_hash(receipt.get("reader_audit_receipt_sha256"), "K4 KLD reader-audit receipt")
    evidence = receipt.get("evidence_artifacts")
    mandatory_roles = {"teacher_logits", "final_student_logits", "tokenwise_kl"}
    if not isinstance(evidence, Mapping) or not mandatory_roles <= set(evidence):
        raise ValueError("packed K4 KLD receipt lacks preserved logits/tokenwise-KL evidence")
    for role in mandatory_roles:
        _require_hash(evidence[role], f"K4 KLD evidence {role}")
    return seal


def qualify_k4_kld(
    plan: Mapping[str, Any], state: Mapping[str, Any], *, kld_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k4_packed":
        raise ValueError("K4 KLD may qualify only the packed K4 state")
    packed = state.get("evidence", {}).get("packed_checkpoint_receipt_sha256")
    kld_sha = verify_k4_kld_receipt(kld_receipt, packed_checkpoint_receipt_sha256=packed)
    evidence = dict(state.get("evidence", {}))
    evidence["k4_packed_kld_receipt_sha256"] = kld_sha
    return _successor(plan, state, phase="k4_kld_qualified", evidence=evidence)


def authorize_k6(plan: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    verify_state(plan, state)
    if state.get("phase") != "k4_kld_qualified":
        raise ValueError("K6 is prohibited until packed K4 KLD qualifies")
    _require_hash(
        state.get("evidence", {}).get("k4_packed_kld_receipt_sha256"),
        "packed K4 KLD receipt",
    )
    return _successor(plan, state, phase="k6_authorized", k6_authorized=True)
