"""Separate layer-45 actual-MCG producer for the GLM-5.3 uniform-K4 campaign."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..calibration.glm53_capture import HIDDEN_SIZE, NUM_EXPERTS, TOP_K, layer_paths, verify_seal
from ..calibration.glm53_mtp_capture import MTP_CAPTURE_SCHEMA, MTP_RECEIPT_SCHEMA
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from ..evaluation.glm53_packed_k4_reader import build_mtp_packed_layer_receipt
from . import glm53_direct_k4 as direct
from .glm53_uniform_k4 import MTP_ADAPTER_RECEIPT_SCHEMA, verify_launch_plan


MTP_CONTRACT_SCHEMA = "quant-pipeline.glm53-mtp45-exl3-mcg-k4-contract.v1"
K6_MTP_CONTRACT_SCHEMA = "quant-pipeline.glm53-mtp45-exl3-mcg-k6-contract.v1"
MTP_STATE_SCHEMA = "quant-pipeline.glm53-mtp45-exl3-mcg-work-state.v1"
MTP_CLAIM_SCHEMA = "quant-pipeline.glm53-mtp45-exl3-mcg-work-claim.v1"
MTP_UNIT_RECEIPT_SCHEMA = "quant-pipeline.glm53-mtp45-exl3-mcg-work-unit-receipt.v1"
MTP_TELEMETRY_SCHEMA = "quant-pipeline.glm53-mtp45-exl3-mcg-work-unit-telemetry.v1"
MAIN_RECEIPT_SCHEMA = "quant-pipeline.glm53-exl3-mcg-main-k4-receipt.v1"
READER_ABI_SCHEMA = "quant-pipeline.glm53-exl3-reader-abi-receipt.v1"
PURE_MCG_BACKEND_SCHEMA = "quant-pipeline.glm53-pure-mcg-backend-qualification.v1"
PURE_MCG_PREPARATION_SCHEMA = "quant-pipeline.glm53-pure-mcg-preparation-qualification.v1"
MTP_LAYER = 45
REQUIRED_ROLES = ("fit", "conditional-fit", "selection", "confirmation")
_HASH = re.compile(r"[0-9a-f]{64}")


def _mtp_contract_schema(bits: int) -> str:
    direct.recipe_id_for_bits(bits)
    return MTP_CONTRACT_SCHEMA if bits == 4 else K6_MTP_CONTRACT_SCHEMA


def _main_receipt_schema(bits: int) -> str:
    direct.recipe_id_for_bits(bits)
    return f"quant-pipeline.glm53-exl3-mcg-main-k{bits}-receipt.v1"


def _mtp_adapter_schema(bits: int) -> str:
    direct.recipe_id_for_bits(bits)
    return (
        MTP_ADAPTER_RECEIPT_SCHEMA
        if bits == 4
        else "quant-pipeline.glm53-uniform-k6-mtp-adapter-receipt.v1"
    )


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify(value: Mapping[str, Any], schema: str, field: str) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"foreign {schema} receipt")
    digest = value.get(field)
    body = copy.deepcopy(dict(value))
    body.pop(field, None)
    if not isinstance(digest, str) or digest != sha256_bytes(canonical_json(body)):
        raise ValueError(f"{schema} seal differs")
    return digest


class Glm53MTP45CaptureView:
    """Memory-mapped MTP45 capture implementing the prepared backend ABI."""

    layer = MTP_LAYER

    def __init__(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root).resolve()
        receipt_path = self.root / "capture-receipt.json"
        manifest_path = self.root / "capture-manifest.json"
        receipt = verify_seal(
            json.loads(receipt_path.read_text(encoding="utf-8")),
            schema=MTP_RECEIPT_SCHEMA,
            field="receipt_sha256",
        )
        manifest = verify_seal(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            schema=MTP_CAPTURE_SCHEMA,
            field="capture_sha256",
        )
        if (
            receipt.get("complete") is not True
            or receipt.get("capture_sha256") != manifest["capture_sha256"]
            or receipt.get("capture_manifest_file_sha256") != sha256_file(manifest_path)
            or manifest.get("layer") != MTP_LAYER
            or tuple(manifest.get("roles", ())) != REQUIRED_ROLES
            or manifest.get("geometry", {}).get("hidden_size") != HIDDEN_SIZE
            or manifest.get("geometry", {}).get("experts") != NUM_EXPERTS
            or manifest.get("geometry", {}).get("top_k") != TOP_K
            or manifest.get("router_cross_check", {}).get("passed") is not True
        ):
            raise ValueError("MTP45 capture is incomplete or has a foreign ABI")
        windows = manifest.get("windows")
        rows = manifest.get("rows")
        if (
            not isinstance(windows, list)
            or not windows
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or sum(int(row.get("rows", -1)) for row in windows) != rows
        ):
            raise ValueError("MTP45 capture window/row census differs")
        offsets: list[tuple[int, int, str, str]] = []
        cursor = 0
        documents: dict[str, int] = {}
        for index, row in enumerate(windows):
            count = int(row.get("rows", -1))
            role = row.get("role")
            document = row.get("document_id")
            if (
                row.get("window_index") != index
                or count <= 0
                or role not in REQUIRED_ROLES
                or not isinstance(document, str)
                or not document
            ):
                raise ValueError("MTP45 capture window identity differs")
            documents.setdefault(document, len(documents))
            offsets.append((cursor, cursor + count, str(role), document))
            cursor += count
        paths = layer_paths(self.root, MTP_LAYER)
        records = manifest.get("files")
        if not isinstance(records, Mapping):
            raise ValueError("MTP45 capture file records are absent")
        expected_bytes = {
            "hidden_bf16": rows * HIDDEN_SIZE * 2,
            "topk_ids_u16le": rows * TOP_K * 2,
            "topk_weights_f32le": rows * TOP_K * 4,
        }
        for name, path in paths.items():
            record = records.get(name)
            if (
                not isinstance(record, Mapping)
                or not path.is_file()
                or path.is_symlink()
                or record.get("path") != path.relative_to(self.root).as_posix()
                or record.get("bytes") != expected_bytes[name]
                or path.stat().st_size != expected_bytes[name]
                or (verify_hashes and sha256_file(path) != record.get("sha256"))
            ):
                raise ValueError(f"MTP45 capture artifact differs: {name}")
        self.receipt = receipt
        self.manifest = manifest
        self.rows = rows
        self._offsets = tuple(offsets)
        self._documents = documents
        self.hidden_u16 = np.memmap(
            paths["hidden_bf16"], dtype="<u2", mode="r", shape=(rows, HIDDEN_SIZE)
        )
        self.topk_ids = np.memmap(
            paths["topk_ids_u16le"], dtype="<u2", mode="r", shape=(rows, TOP_K)
        )
        self.topk_weights = np.memmap(
            paths["topk_weights_f32le"], dtype="<f4", mode="r", shape=(rows, TOP_K)
        )
        if self.topk_ids.size and int(self.topk_ids.max()) >= NUM_EXPERTS:
            raise ValueError("MTP45 capture contains an out-of-range expert ID")

    def binding(self) -> dict[str, Any]:
        return {
            "capture_sha256": self.manifest["capture_sha256"],
            "capture_receipt_sha256": self.receipt["receipt_sha256"],
            "inventory_sha256": self.manifest["inventory_sha256"],
            "token_panel_receipt_sha256": self.manifest["token_panel_receipt_sha256"],
            "main_capture_sha256": self.manifest["main_capture_sha256"],
            "terminal_last_hidden_sha256": self.manifest["terminal_last_hidden_sha256"],
            "layer": MTP_LAYER,
            "roles": list(self.manifest["roles"]),
            "rows": self.rows,
            "route_id_abi": "uint16-little-endian",
            "semantics": self.manifest["semantics"],
        }

    def routed_rows(self, expert: int, role: str) -> direct.RoutedRows:
        if expert not in range(NUM_EXPERTS) or role not in REQUIRED_ROLES:
            raise ValueError("MTP45 routed-row expert or role is outside the sealed capture")
        role_mask = np.zeros(self.rows, dtype=np.bool_)
        document_epochs = np.empty(self.rows, dtype=np.uint32)
        for start, stop, observed, document in self._offsets:
            if observed == role:
                role_mask[start:stop] = True
            document_epochs[start:stop] = self._documents[document]
        hits = (self.topk_ids == expert) & role_mask[:, None]
        rows, slots = np.nonzero(hits)
        return direct.RoutedRows(
            row_indices=rows.astype(np.int64, copy=False),
            route_slots=slots.astype(np.int16, copy=False),
            applied_weights=np.asarray(self.topk_weights[rows, slots], dtype=np.float32),
            document_epochs=document_epochs[rows],
        )


def build_contract(
    *,
    direct_contract: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    main_receipt: Mapping[str, Any],
    capture: Glm53MTP45CaptureView,
    preparation_manifest: Mapping[str, Any],
    reader_abi_receipt: Mapping[str, Any],
    pure_mcg_backend_receipt: Mapping[str, Any],
    pure_mcg_preparation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    direct_sha = direct.verify_contract(direct_contract)
    bits = direct.contract_bits(direct_contract)
    if bits == 4:
        launch_sha = verify_launch_plan(launch_plan)
    else:
        from .glm53_uniform_k6 import verify_launch_plan as verify_k6_launch_plan

        launch_sha = verify_k6_launch_plan(launch_plan)
    main_sha = _verify(main_receipt, _main_receipt_schema(bits), "receipt_sha256")
    reader_sha = _verify(reader_abi_receipt, READER_ABI_SCHEMA, "receipt_sha256")
    reader_abi_sha = reader_abi_receipt.get("reader_sha256")
    backend_sha = verify_pure_mcg_backend_receipt(pure_mcg_backend_receipt)
    pure_preparation_sha = verify_pure_mcg_preparation_receipt(
        pure_mcg_preparation_receipt
    )
    preparation_sha = preparation_manifest.get("preparation_sha256")
    if preparation_sha is None:
        preparation_sha = sha256_bytes(canonical_json(preparation_manifest))
    if (
        direct_contract.get("launch_plan_sha256") != launch_sha
        or direct_contract.get("inventory_sha256") != launch_plan.get("inventory_sha256")
        or main_receipt.get("contract_sha256") != direct_sha
        or main_receipt.get("complete") is not True
        or main_receipt.get("matrix_count") != direct.MAIN_MATRIX_COUNT
        or capture.manifest.get("inventory_sha256") != direct_contract.get("inventory_sha256")
        or preparation_manifest.get("schema")
        != "quant-pipeline.glm53-public-shapleymcg-layer-preparation.v1"
        or preparation_manifest.get("layer") != MTP_LAYER
        or preparation_manifest.get("complete") is not True
        or preparation_manifest.get("bits") != bits
        or preparation_manifest.get("codec_family") != "exl3-mcg"
        or reader_abi_receipt.get("qualified") is not True
        or reader_abi_receipt.get("bits") != bits
        or reader_abi_receipt.get("exact_reconstruction_checked") is not True
        or pure_mcg_backend_receipt.get("bits") != bits
        or pure_mcg_preparation_receipt.get("bits") != bits
        or _HASH.fullmatch(str(reader_abi_sha or "")) is None
    ):
        raise ValueError("MTP45 MCG prerequisites are incomplete or cross-bound")
    body = {
        "schema": _mtp_contract_schema(bits),
        "direct_contract_sha256": direct_sha,
        "launch_plan_sha256": launch_sha,
        "inventory_sha256": direct_contract["inventory_sha256"],
        f"main_k{bits}_receipt_sha256": main_sha,
        "mtp_capture": capture.binding(),
        "preparation_sha256": preparation_sha,
        "reader_abi_receipt_sha256": reader_sha,
        "reader_abi_sha256": reader_abi_sha,
        "pure_mcg_backend_receipt_sha256": backend_sha,
        "pure_mcg_preparation_receipt_sha256": pure_preparation_sha,
        "layer": MTP_LAYER,
        "experts": NUM_EXPERTS,
        "matrix_count": NUM_EXPERTS * 3,
        "bits": bits,
        "allowed_bits": [bits],
        "codec_family": "exl3-mcg",
        "mcg_multiplier_hex": "0xCBAC1FED",
        "global_allocator": False,
        "candidate_rate_grid": False,
        "main_complete_before_mtp": True,
        "launch_authorized": False,
    }
    return _seal(body, "contract_sha256")


def verify_pure_mcg_backend_receipt(receipt: Mapping[str, Any]) -> str:
    digest = _verify(receipt, PURE_MCG_BACKEND_SCHEMA, "receipt_sha256")
    closure = receipt.get("source_closure")
    forbidden = (
        "glm52_fresh_sqg",
        "score_sqg",
        "bmmlaw_r7_encoder/trellis.py",
    )
    paths = [row.get("path") for row in closure] if isinstance(closure, list) else []
    if (
        receipt.get("qualified") is not True
        or receipt.get("bits", 4) not in direct.SUPPORTED_BITS
        or receipt.get("codec_family") != "exl3-mcg"
        or receipt.get("mcg_multiplier_hex") != "0xCBAC1FED"
        or receipt.get("sqg_orchestration_imported") is not False
        or receipt.get("actual_mcg_encode_pack_decode_checked") is not True
        or receipt.get("codec_class")
        != "r7_encoder.r10_codec.R10TrellisCodec"
        or receipt.get("public_codec_adapter") != "Exl3MCGCodec"
        or receipt.get("offline_reader_exact_decode_checked") is not True
        or _HASH.fullmatch(str(receipt.get("offline_reader_abi_sha256", ""))) is None
        or not isinstance(closure, list)
        or not closure
        or not any(
            isinstance(path, str) and path.endswith("r7_encoder/r10_codec.py")
            for path in paths
        )
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or any(token in row["path"] for token in forbidden)
            or _HASH.fullmatch(str(row.get("sha256", ""))) is None
            for row in closure
        )
    ):
        raise ValueError("numeric backend is not a sealed pure-MCG implementation")
    return digest


def verify_pure_mcg_preparation_receipt(receipt: Mapping[str, Any]) -> str:
    digest = _verify(receipt, PURE_MCG_PREPARATION_SCHEMA, "receipt_sha256")
    closure = receipt.get("source_closure")
    forbidden = (
        "glm52_fresh_sqg",
        "score_sqg",
        "bmmlaw_r7_encoder/trellis.py",
    )
    paths = [row.get("path") for row in closure] if isinstance(closure, list) else []
    if (
        receipt.get("qualified") is not True
        or receipt.get("bits", 4) not in direct.SUPPORTED_BITS
        or receipt.get("sqg_orchestration_imported") is not False
        or receipt.get("public_shapleymcg_run_qwen_fast_encode_structure") is not True
        or receipt.get("local_corrected_v1_numerical_order") is not True
        or receipt.get("r7_encoder_r10_codec_closure") is not True
        or receipt.get("codec_class")
        != "r7_encoder.r10_codec.R10TrellisCodec"
        or _HASH.fullmatch(str(receipt.get("local_corrected_v1_sha256", ""))) is None
        or not isinstance(closure, list)
        or not closure
        or not any(
            isinstance(path, str) and path.endswith("r7_encoder/r10_codec.py")
            for path in paths
        )
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or any(token in row["path"] for token in forbidden)
            or _HASH.fullmatch(str(row.get("sha256", ""))) is None
            for row in closure
        )
    ):
        raise ValueError("preparation is not sealed to the pure ShapleyMCG/r7 lineage")
    return digest


def verify_backend_identity(contract: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if (
        identity.get("codec_family") != "exl3-mcg"
        or identity.get("mcg_multiplier_hex") != "0xCBAC1FED"
        or identity.get("codec_class")
        != "r7_encoder.r10_codec.R10TrellisCodec"
        or identity.get("public_codec_adapter") != "Exl3MCGCodec"
        or identity.get("sqg_orchestration_imported") is not False
        or identity.get("reader_abi_sha256") != contract.get("reader_abi_sha256")
        or identity.get("pure_mcg_backend_receipt_sha256")
        != contract.get("pure_mcg_backend_receipt_sha256")
    ):
        raise ValueError("loaded backend identity is not the contract-qualified pure MCG backend")


def verify_contract(contract: Mapping[str, Any]) -> str:
    bits = int(contract.get("bits", -1))
    direct.recipe_id_for_bits(bits)
    digest = _verify(contract, _mtp_contract_schema(bits), "contract_sha256")
    if (
        contract.get("layer") != MTP_LAYER
        or contract.get("matrix_count") != NUM_EXPERTS * 3
        or contract.get("allowed_bits") != [bits]
        or contract.get("codec_family") != "exl3-mcg"
        or contract.get("global_allocator") is not False
        or contract.get("launch_authorized") is not False
        or _HASH.fullmatch(str(contract.get("pure_mcg_backend_receipt_sha256", "")))
        is None
        or _HASH.fullmatch(str(contract.get("pure_mcg_preparation_receipt_sha256", "")))
        is None
    ):
        raise ValueError("MTP45 MCG contract invariant differs")
    return digest


def build_work_units(
    contract: Mapping[str, Any], *, experts_per_unit: int = 18
) -> list[dict[str, Any]]:
    contract_sha = verify_contract(contract)
    bits = int(contract["bits"])
    if experts_per_unit <= 0 or experts_per_unit > NUM_EXPERTS:
        raise ValueError("MTP experts per unit must be in [1,288]")
    units = []
    for start in range(0, NUM_EXPERTS, experts_per_unit):
        stop = min(start + experts_per_unit, NUM_EXPERTS)
        units.append(
            _seal(
                {
                    "schema": direct.WORK_UNIT_SCHEMA,
                    "contract_sha256": contract_sha,
                    "direct_contract_sha256": contract["direct_contract_sha256"],
                    "layer": MTP_LAYER,
                    "expert_start": start,
                    "expert_stop": stop,
                    "expert_count": stop - start,
                    "matrix_count": (stop - start) * 3,
                    "bits": bits,
                    "claim_policy": "dynamic_next_unclaimed",
                    "global_allocator": False,
                },
                "work_unit_sha256",
            )
        )
    return units


def initial_state(contract: Mapping[str, Any], units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contract_sha = verify_contract(contract)
    table = {str(row["work_unit_sha256"]): dict(row) for row in units}
    coverage = {
        expert
        for row in units
        for expert in range(int(row["expert_start"]), int(row["expert_stop"]))
    }
    if len(table) != len(units) or coverage != set(range(NUM_EXPERTS)):
        raise ValueError("MTP45 work units do not close exactly 288 experts")
    return _seal(
        {
            "schema": MTP_STATE_SCHEMA,
            "contract_sha256": contract_sha,
            "sequence": 0,
            "previous_state_sha256": None,
            "units": table,
            "pending": list(table),
            "active": {},
            "completed": {},
        },
        "state_sha256",
    )


def verify_state(contract: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    contract_sha = verify_contract(contract)
    digest = _verify(state, MTP_STATE_SCHEMA, "state_sha256")
    units, pending, active, completed = (
        state.get("units"),
        state.get("pending"),
        state.get("active"),
        state.get("completed"),
    )
    active_units = [row.get("work_unit_sha256") for row in active.values()] if isinstance(active, Mapping) else []
    if (
        state.get("contract_sha256") != contract_sha
        or not isinstance(units, Mapping)
        or not isinstance(pending, list)
        or not isinstance(active, Mapping)
        or not isinstance(completed, Mapping)
        or set(pending) & set(active_units)
        or set(pending) & set(completed)
        or set(active_units) & set(completed)
        or set(pending) | set(active_units) | set(completed) != set(units)
    ):
        raise ValueError("MTP45 work-state partition differs")
    return digest


def _successor(contract: Mapping[str, Any], state: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    previous = verify_state(contract, state)
    body = copy.deepcopy(dict(state))
    del body["state_sha256"]
    body.update(copy.deepcopy(updates))
    body["sequence"] = int(state["sequence"]) + 1
    body["previous_state_sha256"] = previous
    return _seal(body, "state_sha256")


def claim_next(contract: Mapping[str, Any], state: Mapping[str, Any], worker_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_state(contract, state)
    if worker_id not in {f"b200-{index}" for index in range(4)} or worker_id in state["active"]:
        raise ValueError("MTP45 worker identity is invalid or already active")
    pending = list(state["pending"])
    if not pending:
        raise ValueError("no MTP45 work unit remains")
    unit_sha = pending.pop(0)
    unit = state["units"][unit_sha]
    claim = _seal(
        {
            "schema": MTP_CLAIM_SCHEMA,
            "contract_sha256": contract["contract_sha256"],
            "parent_state_sha256": state["state_sha256"],
            "worker_id": worker_id,
            "work_unit_sha256": unit_sha,
            "expert_start": unit["expert_start"],
            "expert_stop": unit["expert_stop"],
        },
        "claim_sha256",
    )
    active = copy.deepcopy(dict(state["active"]))
    active[worker_id] = claim
    return _successor(contract, state, pending=pending, active=active), claim


def complete(
    contract: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    worker_id: str,
    unit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_state(contract, state)
    _verify(unit_receipt, MTP_UNIT_RECEIPT_SCHEMA, "receipt_sha256")
    active = copy.deepcopy(dict(state["active"]))
    claim = active.pop(worker_id, None)
    if (
        not isinstance(claim, Mapping)
        or unit_receipt.get("contract_sha256") != contract["contract_sha256"]
        or unit_receipt.get("work_unit_sha256") != claim.get("work_unit_sha256")
        or unit_receipt.get("complete") is not True
    ):
        raise ValueError("MTP45 work-unit completion differs from its claim")
    completed = copy.deepcopy(dict(state["completed"]))
    completed[claim["work_unit_sha256"]] = {
        "worker_id": worker_id,
        "claim_sha256": claim["claim_sha256"],
        "receipt_sha256": unit_receipt["receipt_sha256"],
    }
    return _successor(contract, state, active=active, completed=completed)


def release_claim(
    contract: Mapping[str, Any], state: Mapping[str, Any], *, worker_id: str
) -> dict[str, Any]:
    """Requeue one dead worker claim while preserving the state hash chain."""

    verify_state(contract, state)
    active = copy.deepcopy(dict(state["active"]))
    claim = active.pop(worker_id, None)
    if not isinstance(claim, Mapping):
        raise ValueError("MTP45 worker has no active claim to release")
    pending = list(state["pending"])
    pending.insert(0, claim["work_unit_sha256"])
    return _successor(contract, state, active=active, pending=pending)


def direct_work_unit(contract: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    verify_contract(contract)
    bits = int(contract["bits"])
    return _seal(
        {
            "schema": direct.WORK_UNIT_SCHEMA,
            "contract_sha256": contract["direct_contract_sha256"],
            "layer": MTP_LAYER,
            "expert_start": unit["expert_start"],
            "expert_stop": unit["expert_stop"],
            "expert_count": unit["expert_count"],
            "matrix_count": unit["matrix_count"],
            "bits": bits,
            "claim_policy": "dynamic_next_unclaimed",
        },
        "work_unit_sha256",
    )


def encode_work_unit(
    *,
    contract: Mapping[str, Any],
    direct_contract: Mapping[str, Any],
    unit: Mapping[str, Any],
    source: direct.Glm53BF16Source,
    capture: Glm53MTP45CaptureView,
    backend: direct.DirectMCGBackend,
    output_root: str | Path,
    device: str = "cuda:0",
    max_inflight_experts: int = 28,
) -> dict[str, Any]:
    verify_contract(contract)
    verify_backend_identity(contract, backend.identity())
    translated = direct_work_unit(contract, unit)
    receipt = direct.encode_work_unit(
        contract=direct_contract,
        work_unit=translated,
        source=source,
        capture=capture,  # type: ignore[arg-type]
        backend=backend,
        output_root=output_root,
        device=device,
        max_inflight_experts=max_inflight_experts,
    )
    body = {
        "schema": MTP_UNIT_RECEIPT_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "direct_contract_sha256": contract["direct_contract_sha256"],
        "work_unit_sha256": unit["work_unit_sha256"],
        "direct_work_unit_sha256": translated["work_unit_sha256"],
        "direct_receipt_sha256": receipt["receipt_sha256"],
        "layer": MTP_LAYER,
        "expert_start": unit["expert_start"],
        "expert_stop": unit["expert_stop"],
        "complete": receipt.get("complete") is True,
    }
    result = _seal(body, "receipt_sha256")
    write_json(
        Path(output_root) / "mtp-work-units" / f"{unit['work_unit_sha256']}.json",
        result,
    )
    return result


def seal_mtp_layer(
    *,
    contract: Mapping[str, Any],
    direct_contract: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    state: Mapping[str, Any],
    output_root: str | Path,
    backend_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_contract(contract)
    verify_state(contract, state)
    if state["pending"] or state["active"] or len(state["completed"]) != len(state["units"]):
        raise ValueError("MTP45 cannot seal until all 288 experts complete")
    root = Path(output_root)
    packed = build_mtp_packed_layer_receipt(root=root, contract=direct_contract)
    bits = int(contract["bits"])
    write_json(root / "layers" / "layer-045.json", packed)
    telemetry = []
    for unit_sha in sorted(state["units"]):
        path = root / "mtp-telemetry" / f"{unit_sha}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        _verify(row, MTP_TELEMETRY_SCHEMA, "receipt_sha256")
        if row.get("work_unit_sha256") != unit_sha:
            raise ValueError("MTP45 telemetry work-unit binding differs")
        telemetry.append(row["receipt_sha256"])
    backend_sha = sha256_bytes(canonical_json(dict(backend_identity)))
    adapter = _seal(
        {
            "schema": _mtp_adapter_schema(bits),
            "launch_plan_sha256": launch_plan["launch_plan_sha256"],
            "inventory_sha256": contract["inventory_sha256"],
            f"main_k{bits}_receipt_sha256": contract[f"main_k{bits}_receipt_sha256"],
            "mtp_capture_sha256": contract["mtp_capture"]["capture_sha256"],
            "preparation_sha256": contract["preparation_sha256"],
            "layer": MTP_LAYER,
            "expert_count": NUM_EXPERTS,
            "matrix_count": NUM_EXPERTS * 3,
            "bits": bits,
            "codec_family": "exl3-mcg",
            "mcg_multiplier_hex": "0xCBAC1FED",
            "qualified": True,
            "tensor_names_sha256": launch_plan["mtp_work_unit"]["tensor_names_sha256"],
            "codec_adapter_sha256": backend_sha,
            "packed_payload_receipt_sha256": packed["receipt_sha256"],
            "global_allocator_invoked": False,
            "work_unit_telemetry_receipt_sha256": telemetry,
        },
        "receipt_sha256",
    )
    write_json(root / "mtp-adapter-receipt.json", adapter)
    return packed, adapter
