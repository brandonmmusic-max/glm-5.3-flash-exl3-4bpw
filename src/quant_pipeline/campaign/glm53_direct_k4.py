"""Production adapter contract for the public ShapleyMCG GLM recipe.

The numerical recipe lives in the prepared public ShapleyMCG source closure.
This module changes only GLM-5.3 geometry, names, capture access, scheduling,
resume and receipts.  It deliberately has no rate search or allocator and it
does not import CUDA at module import time.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..calibration.glm53_capture import (
    CAPTURE_SCHEMA,
    HIDDEN_SIZE,
    MAIN_ROUTED_LAYERS,
    NUM_EXPERTS,
    TOP_K,
    verify_seal,
)
from ..checkpoint.packed_payload import PackedMCGPayloadStore
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from .glm53_uniform_k4 import (
    INVENTORY_SCHEMA,
    LAUNCH_PLAN_SCHEMA,
    MTP_ADAPTER_RECEIPT_SCHEMA,
    PROJECTIONS,
    verify_launch_plan,
)


CONTRACT_SCHEMA = "quant-pipeline.glm53-direct-mcg-k4-contract.v1"
K6_CONTRACT_SCHEMA = "quant-pipeline.glm53-direct-mcg-k6-contract.v1"
WORK_UNIT_SCHEMA = "quant-pipeline.glm53-direct-mcg-work-unit.v1"
EXPERT_RECEIPT_SCHEMA = "quant-pipeline.glm53-direct-mcg-expert-receipt.v1"
LAYER_RECEIPT_SCHEMA = "quant-pipeline.glm53-direct-mcg-layer-receipt.v1"
MATERIALIZATION_PLAN_SCHEMA = "quant-pipeline.glm53-k4-materialization-plan.v1"
K6_MATERIALIZATION_PLAN_SCHEMA = "quant-pipeline.glm53-k6-materialization-plan.v1"
MATERIALIZATION_RECEIPT_SCHEMA = "quant-pipeline.glm53-k4-materialization-receipt.v1"
K6_MATERIALIZATION_RECEIPT_SCHEMA = "quant-pipeline.glm53-k6-materialization-receipt.v1"
WORK_STATE_SCHEMA = "quant-pipeline.glm53-direct-mcg-work-state.v1"
WORK_CLAIM_SCHEMA = "quant-pipeline.glm53-direct-mcg-work-claim.v1"
RECIPE_ID = "shapleymcg-public-r10-uniform-k4-candidate-conditioned-down-v1"
K6_RECIPE_ID = "shapleymcg-public-r10-uniform-k6-candidate-conditioned-down-v1"
SUPPORTED_BITS = (4, 6)
REQUIRED_ROLES = ("fit", "conditional-fit", "selection", "confirmation")
INTERMEDIATE_SIZE = 2_048
MAIN_MATRIX_COUNT = len(MAIN_ROUTED_LAYERS) * NUM_EXPERTS * len(PROJECTIONS)
MTP_LAYER = 45

_HASH = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def recipe_id_for_bits(bits: int) -> str:
    if bits == 4:
        return RECIPE_ID
    if bits == 6:
        return K6_RECIPE_ID
    raise ValueError("GLM-5.3 direct MCG supports only uniform K4 or K6")


def contract_schema_for_bits(bits: int) -> str:
    recipe_id_for_bits(bits)
    return CONTRACT_SCHEMA if bits == 4 else K6_CONTRACT_SCHEMA


def materialization_plan_schema_for_bits(bits: int) -> str:
    recipe_id_for_bits(bits)
    return MATERIALIZATION_PLAN_SCHEMA if bits == 4 else K6_MATERIALIZATION_PLAN_SCHEMA


def materialization_receipt_schema_for_bits(bits: int) -> str:
    recipe_id_for_bits(bits)
    return MATERIALIZATION_RECEIPT_SCHEMA if bits == 4 else K6_MATERIALIZATION_RECEIPT_SCHEMA


def contract_bits(contract: Mapping[str, Any]) -> int:
    bits = int(contract.get("rate", {}).get("bits", -1))
    if bits not in SUPPORTED_BITS or contract.get("schema") != contract_schema_for_bits(bits):
        raise ValueError("direct MCG contract rate/schema differs")
    return bits


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_seal(value: Mapping[str, Any], schema: str, field: str) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"foreign {schema} receipt")
    digest = _require_hash(value.get(field), field)
    body = copy.deepcopy(dict(value))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        raise ValueError(f"{schema} seal differs")
    return digest


def projection_shape(projection: str) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    if projection == "down_proj":
        return (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    raise ValueError(f"unknown routed projection: {projection}")


def tensor_name(layer: int, expert: int, projection: str) -> str:
    if layer not in (*MAIN_ROUTED_LAYERS, MTP_LAYER):
        raise ValueError("GLM-5.3 routed layer is outside main/MTP surface")
    if expert not in range(NUM_EXPERTS):
        raise ValueError("GLM-5.3 expert is outside [0,288)")
    projection_shape(projection)
    return (
        f"model.language_model.layers.{layer}.mlp.experts."
        f"{expert}.{projection}.weight"
    )


def inventory_tensor_map(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate exact official routed geometry and return the tensor rows."""

    _verify_seal(inventory, INVENTORY_SCHEMA, "inventory_sha256")
    if inventory.get("seal_mode") != "full-shard-sha256":
        raise ValueError("production K4 requires full source-shard hashes")
    if _REVISION.fullmatch(str(inventory.get("model_revision", ""))) is None:
        raise ValueError("production K4 requires an immutable BF16 revision")
    shard_hashes = inventory.get("shard_sha256")
    if not isinstance(shard_hashes, Mapping) or not shard_hashes or any(
        not isinstance(name, str)
        or not name
        or _HASH.fullmatch(str(digest)) is None
        for name, digest in shard_hashes.items()
    ):
        raise ValueError("production K4 inventory lacks full shard SHA-256 closure")
    result: dict[str, dict[str, Any]] = {}
    rows = inventory.get("tensors")
    if not isinstance(rows, list):
        raise ValueError("inventory tensor rows are absent")
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("inventory tensor row is malformed")
        name = raw.get("tensor_name")
        if not isinstance(name, str) or name in result:
            raise ValueError("inventory tensor name is absent or duplicated")
        result[name] = dict(raw)
    for layer in (*MAIN_ROUTED_LAYERS, MTP_LAYER):
        scope = "routed_expert" if layer in MAIN_ROUTED_LAYERS else "mtp_routed_expert"
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                name = tensor_name(layer, expert, projection)
                row = result.get(name)
                shape = list(projection_shape(projection))
                if (
                    row is None
                    or row.get("scope") != scope
                    or row.get("dtype") != "BF16"
                    or row.get("shape") != shape
                    or row.get("source_bytes") != math.prod(shape) * 2
                    or _HASH.fullmatch(str(row.get("source_payload_sha256", ""))) is None
                    or not isinstance(row.get("shard"), str)
                    or row.get("shard") not in shard_hashes
                ):
                    raise ValueError(f"official BF16 routed tensor differs: {name}")
    return result


@dataclass(frozen=True)
class RoutedRows:
    row_indices: np.ndarray
    route_slots: np.ndarray
    applied_weights: np.ndarray
    document_epochs: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.row_indices.size)


class Glm53CaptureView:
    """Memory-mapped adapter from the sealed u16 capture ABI to recipe rows."""

    def __init__(
        self,
        root: str | Path,
        layer: int,
        *,
        verify_hashes: bool = True,
        required_roles: Sequence[str] = REQUIRED_ROLES,
    ) -> None:
        if layer not in MAIN_ROUTED_LAYERS:
            raise ValueError("main capture adapter excludes MTP45")
        self.root = Path(root).resolve()
        self.layer = layer
        path = self.root / "capture-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        verify_seal(manifest, schema=CAPTURE_SCHEMA, field="capture_sha256")
        if (
            manifest.get("layers") != list(MAIN_ROUTED_LAYERS)
            or manifest.get("geometry", {}).get("hidden_size") != HIDDEN_SIZE
            or manifest.get("geometry", {}).get("experts") != NUM_EXPERTS
            or manifest.get("geometry", {}).get("top_k") != TOP_K
        ):
            raise ValueError("capture geometry differs from released GLM-5.3")
        roles = tuple(manifest.get("roles", ()))
        if not set(required_roles) <= set(roles):
            raise ValueError(
                "capture lacks raw selection/confirmation roles required by the preserved recipe"
            )
        windows = manifest.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("capture window journal is absent")
        offsets: list[tuple[int, int, str, str]] = []
        cursor = 0
        document_epoch: dict[str, int] = {}
        for index, raw in enumerate(windows):
            if not isinstance(raw, Mapping) or raw.get("window_index") != index:
                raise ValueError("capture window order differs")
            rows = raw.get("rows")
            role = raw.get("role")
            document = raw.get("document_id")
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
                or role not in roles
                or not isinstance(document, str)
                or not document
            ):
                raise ValueError("capture window boundary is malformed")
            document_epoch.setdefault(document, len(document_epoch))
            offsets.append((cursor, cursor + rows, str(role), document))
            cursor += rows
        if manifest.get("rows_per_layer") != cursor:
            raise ValueError("capture row census differs from window boundaries")
        files = manifest.get("files", {}).get(str(layer), {})
        paths: dict[str, Path] = {}
        for key in ("hidden_bf16", "topk_ids_u16le", "topk_weights_f32le"):
            record = files.get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"capture lacks {key}")
            artifact = (self.root / str(record.get("path", ""))).resolve()
            try:
                artifact.relative_to(self.root)
            except ValueError as error:
                raise ValueError("capture artifact escapes its root") from error
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or artifact.stat().st_size != record.get("bytes")
                or (verify_hashes and sha256_file(artifact) != record.get("sha256"))
            ):
                raise ValueError(f"capture artifact differs: {key}")
            paths[key] = artifact
        expected_bytes = {
            "hidden_bf16": cursor * HIDDEN_SIZE * 2,
            "topk_ids_u16le": cursor * TOP_K * 2,
            "topk_weights_f32le": cursor * TOP_K * 4,
        }
        if any(paths[key].stat().st_size != size for key, size in expected_bytes.items()):
            raise ValueError("capture raw-file byte geometry differs")
        self.manifest = manifest
        self.rows = cursor
        self._offsets = tuple(offsets)
        self._document_epoch = document_epoch
        self.hidden_u16 = np.memmap(paths["hidden_bf16"], dtype="<u2", mode="r", shape=(cursor, HIDDEN_SIZE))
        self.topk_ids = np.memmap(paths["topk_ids_u16le"], dtype="<u2", mode="r", shape=(cursor, TOP_K))
        self.topk_weights = np.memmap(paths["topk_weights_f32le"], dtype="<f4", mode="r", shape=(cursor, TOP_K))
        if self.topk_ids.size and int(self.topk_ids.max()) >= NUM_EXPERTS:
            raise ValueError("capture contains an out-of-range u16 expert ID")

    def binding(self) -> dict[str, Any]:
        return {
            "capture_sha256": self.manifest["capture_sha256"],
            "inventory_sha256": self.manifest["inventory_sha256"],
            "token_panel_receipt_sha256": self.manifest["token_panel_receipt_sha256"],
            "layer": self.layer,
            "roles": list(self.manifest["roles"]),
            "rows": self.rows,
            "route_id_abi": "uint16-little-endian",
        }

    def role_mask(self, role: str) -> np.ndarray:
        if role not in self.manifest["roles"]:
            raise ValueError(f"capture role is absent: {role}")
        result = np.zeros(self.rows, dtype=np.bool_)
        for start, stop, observed, _document in self._offsets:
            if observed == role:
                result[start:stop] = True
        return result

    def document_epochs(self) -> np.ndarray:
        result = np.empty(self.rows, dtype=np.uint32)
        for start, stop, _role, document in self._offsets:
            result[start:stop] = self._document_epoch[document]
        return result

    def routed_rows(self, expert: int, role: str) -> RoutedRows:
        if expert not in range(NUM_EXPERTS):
            raise ValueError("expert is outside [0,288)")
        role_mask = self.role_mask(role)
        hits = (self.topk_ids == expert) & role_mask[:, None]
        rows, slots = np.nonzero(hits)
        return RoutedRows(
            row_indices=rows.astype(np.int64, copy=False),
            route_slots=slots.astype(np.int16, copy=False),
            applied_weights=np.asarray(self.topk_weights[rows, slots], dtype=np.float32),
            document_epochs=self.document_epochs()[rows],
        )


class Glm53BF16Source:
    """Exact official-BF16 triplet loader bound to the sealed inventory."""

    def __init__(
        self,
        inventory: Mapping[str, Any],
        root: str | Path,
        *,
        verify_shards: bool = True,
    ) -> None:
        self.inventory = copy.deepcopy(dict(inventory))
        self.rows = inventory_tensor_map(inventory)
        self.root = Path(root).resolve()
        if Path(str(inventory.get("checkpoint", ""))).resolve() != self.root:
            raise ValueError("BF16 source root differs from the sealed inventory")
        if verify_shards:
            for name, expected in inventory["shard_sha256"].items():
                path = (self.root / name).resolve()
                try:
                    path.relative_to(self.root)
                except ValueError as error:
                    raise ValueError("inventory shard escapes BF16 source root") from error
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != expected
                ):
                    raise ValueError(f"official BF16 shard hash differs: {name}")

    def load_projection(self, layer: int, expert: int, projection: str, *, device: str = "cpu") -> Any:
        from safetensors import safe_open

        name = tensor_name(layer, expert, projection)
        row = self.rows[name]
        shard = (self.root / row["shard"]).resolve()
        try:
            shard.relative_to(self.root)
        except ValueError as error:
            raise ValueError("inventory shard escapes BF16 source root") from error
        if not shard.is_file() or shard.is_symlink():
            raise ValueError(f"official BF16 shard is absent: {row['shard']}")
        with safe_open(shard, framework="pt", device=device) as handle:
            value = handle.get_tensor(name)
        if list(value.shape) != row["shape"] or str(value.dtype) != "torch.bfloat16":
            raise ValueError(f"official BF16 tensor payload differs: {name}")
        return value.contiguous()

    def load_triplet(self, layer: int, expert: int, *, device: str = "cpu") -> dict[str, Any]:
        return {
            projection: self.load_projection(layer, expert, projection, device=device)
            for projection in PROJECTIONS
        }


def _validate_recipe_evidence(value: Mapping[str, Any], *, bits: int | None = None) -> None:
    target_bits = int(value.get("bits", -1)) if bits is None else bits
    recipe_id_for_bits(target_bits)
    required = {
        "recipe_id": recipe_id_for_bits(target_bits),
        "codec_family": "exl3-mcg",
        "mcg_multiplier_hex": "0xCBAC1FED",
        "bits": target_bits,
        "candidate_rate_grid": False,
        "global_allocator": False,
        "gate_up_hessian": "routed_p2_uncentered_full_hessian",
        "down_hessian": f"decoded_k{target_bits}_candidate_conditioned_routed_p2_uncentered_full_hessian",
        "down_candidate_conditioned": True,
        "profile_source": "public-run-qwen-fast-encode-defaults",
        "profile_policy": "energy_balanced",
        "scale_family": "per128-grid",
        "profile_fixed_before_encoding": True,
        "selection_used_for_profile_choice": False,
        "selection_rows_used_for_encoding": False,
        "confirmation_rows_used_for_choice": False,
        "sqg_orchestration_imported": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"public ShapleyMCG recipe semantic differs: {key}")


def build_contract(
    *,
    launch_plan: Mapping[str, Any],
    inventory: Mapping[str, Any],
    capture_manifest: Mapping[str, Any],
    prepared_source: Mapping[str, Any],
    exllama: Mapping[str, Any],
    preparation: Mapping[str, Any],
    reader_abi_sha256: str,
    pure_mcg_backend_receipt_sha256: str,
    pure_mcg_preparation_receipt_sha256: str,
    bits: int = 4,
) -> dict[str, Any]:
    """Seal the compatibility boundary without launching any GPU work."""

    recipe_id = recipe_id_for_bits(bits)
    if bits == 4:
        plan_sha = verify_launch_plan(launch_plan)
        if launch_plan.get("schema") != LAUNCH_PLAN_SCHEMA:
            raise ValueError("uniform-K4 launch plan differs")
    else:
        plan_sha = _verify_seal(
            launch_plan,
            "quant-pipeline.glm53-uniform-k6-four-b200-launch-plan.v1",
            "launch_plan_sha256",
        )
        if (
            launch_plan.get("profile") != "k6-tp4"
            or launch_plan.get("rate_contract", {}).get("allowed_bits") != [6]
            or launch_plan.get("rate_contract", {}).get("global_allocator_invoked") is not False
        ):
            raise ValueError("uniform-K6 TP4 launch plan differs")
    rows = inventory_tensor_map(inventory)
    verify_seal(capture_manifest, schema=CAPTURE_SCHEMA, field="capture_sha256")
    if (
        capture_manifest.get("inventory_sha256") != inventory["inventory_sha256"]
        or launch_plan.get("inventory_sha256") != inventory["inventory_sha256"]
        or capture_manifest.get("roles") != list(REQUIRED_ROLES)
        or capture_manifest.get("layers") != list(MAIN_ROUTED_LAYERS)
    ):
        raise ValueError("inventory/capture/launch-plan binding differs")
    if (
        prepared_source.get("recipe_id") != recipe_id
        or prepared_source.get("reviewed_glm53_entrypoint") is not True
        or _HASH.fullmatch(str(prepared_source.get("tree_sha256", ""))) is None
        or not isinstance(prepared_source.get("entrypoint"), str)
    ):
        raise ValueError("prepared source lacks a reviewed GLM53 recipe entrypoint")
    _validate_recipe_evidence(prepared_source, bits=bits)
    if (
        exllama.get("fresh_build") is not True
        or "10.0" not in exllama.get("compute_capabilities", ())
        or _HASH.fullmatch(str(exllama.get("extension_sha256", ""))) is None
    ):
        raise ValueError("fresh SM100 ExLlama extension evidence is incomplete")
    preparation_required = {
        "transform_seed_sha256",
        "fixed_profile_receipt_sha256",
        "permutation_set_sha256",
        "gate_up_p2_set_sha256",
        "gss_receipts_sha256",
        "confirmation_report_receipt_sha256",
    }
    for key in preparation_required:
        _require_hash(preparation.get(key), f"preparation.{key}")
    _require_hash(
        reader_abi_sha256,
        "reader_abi_sha256",
    )
    _require_hash(
        pure_mcg_backend_receipt_sha256,
        "pure_mcg_backend_receipt_sha256",
    )
    _require_hash(
        pure_mcg_preparation_receipt_sha256,
        "pure_mcg_preparation_receipt_sha256",
    )
    if (
        preparation.get("bits", 4) != bits
        or preparation.get("codec_family", "exl3-mcg") != "exl3-mcg"
        or preparation.get("global_allocator_invoked", False) is not False
        or preparation.get("candidate_rate_grid_invoked", False) is not False
        or preparation.get("profile_source")
        != "public-run-qwen-fast-encode-defaults"
        or preparation.get("profile_fixed_before_encoding") is not True
        or preparation.get("selection_rows_used") is not False
        or preparation.get("selection_used_for_profile_choice") is not False
        or preparation.get("confirmation_report_only") is not True
        or preparation.get("confirmation_used_for_choice") is not False
        or preparation.get("selection_used_for_final_encoding") is not False
    ):
        raise ValueError("profile selection/confirmation split semantics differ")
    main_names = [
        tensor_name(layer, expert, projection)
        for layer in MAIN_ROUTED_LAYERS
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    ]
    body = {
        "schema": contract_schema_for_bits(bits),
        "launch_plan_sha256": plan_sha,
        "inventory_sha256": inventory["inventory_sha256"],
        "capture_sha256": capture_manifest["capture_sha256"],
        "model_revision": inventory["model_revision"],
        "recipe": copy.deepcopy(dict(prepared_source)),
        "exllama": copy.deepcopy(dict(exllama)),
        "preparation": copy.deepcopy(dict(preparation)),
        "reader_abi_sha256": reader_abi_sha256,
        "pure_mcg_backend_receipt_sha256": pure_mcg_backend_receipt_sha256,
        "pure_mcg_preparation_receipt_sha256": pure_mcg_preparation_receipt_sha256,
        "geometry": {
            "layers": list(MAIN_ROUTED_LAYERS),
            "experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "projections": list(PROJECTIONS),
            "matrix_count": MAIN_MATRIX_COUNT,
            "tensor_names_sha256": sha256_bytes(canonical_json(main_names)),
        },
        "rate": {
            "bits": bits,
            "allowed_bits": [bits],
            "candidate_rate_grid": False,
            "global_allocator": False,
            "uniform_within_every_expert": True,
        },
        "mtp": {
            "layer": MTP_LAYER,
            "included": False,
            "separate_adapter_and_qualification_required": True,
        },
        "source_routed_tensor_count": sum(
            row.get("scope") in {"routed_expert", "mtp_routed_expert"}
            for row in rows.values()
        ),
        "launch_authorized": False,
    }
    return _seal(body, "contract_sha256")


def verify_contract(contract: Mapping[str, Any]) -> str:
    bits = contract_bits(contract)
    digest = _verify_seal(contract, contract_schema_for_bits(bits), "contract_sha256")
    if (
        contract.get("geometry", {}).get("matrix_count") != MAIN_MATRIX_COUNT
        or contract.get("rate", {}).get("allowed_bits") != [bits]
        or contract.get("mtp", {}).get("included") is not False
        or contract.get("launch_authorized") is not False
        or _HASH.fullmatch(
            str(contract.get("reader_abi_sha256", ""))
        )
        is None
        or _HASH.fullmatch(
            str(contract.get("pure_mcg_backend_receipt_sha256", ""))
        )
        is None
        or _HASH.fullmatch(
            str(contract.get("pure_mcg_preparation_receipt_sha256", ""))
        )
        is None
    ):
        raise ValueError("direct MCG contract invariant differs")
    _validate_recipe_evidence(contract.get("recipe", {}), bits=bits)
    return digest


def build_work_units(
    contract: Mapping[str, Any], *, experts_per_unit: int = NUM_EXPERTS
) -> list[dict[str, Any]]:
    """Build dynamic whole-layer or contiguous expert-range work units."""

    contract_sha = verify_contract(contract)
    bits = contract_bits(contract)
    if experts_per_unit <= 0 or experts_per_unit > NUM_EXPERTS:
        raise ValueError("experts_per_unit must be in [1,288]")
    units: list[dict[str, Any]] = []
    for layer in MAIN_ROUTED_LAYERS:
        for start in range(0, NUM_EXPERTS, experts_per_unit):
            stop = min(start + experts_per_unit, NUM_EXPERTS)
            body = {
                "schema": WORK_UNIT_SCHEMA,
                "contract_sha256": contract_sha,
                "layer": layer,
                "expert_start": start,
                "expert_stop": stop,
                "expert_count": stop - start,
                "matrix_count": (stop - start) * len(PROJECTIONS),
                "bits": bits,
                "claim_policy": "dynamic_next_unclaimed",
            }
            units.append(_seal(body, "work_unit_sha256"))
    return units


def initial_work_state(
    contract: Mapping[str, Any], work_units: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Create the coordinator state for four workers pulling dynamic units."""

    contract_sha = verify_contract(contract)
    pending: list[str] = []
    units: dict[str, dict[str, Any]] = {}
    coverage: set[tuple[int, int]] = set()
    for raw in work_units:
        unit = dict(raw)
        digest = _verify_seal(unit, WORK_UNIT_SCHEMA, "work_unit_sha256")
        if unit.get("contract_sha256") != contract_sha or digest in units:
            raise ValueError("work-unit set is duplicated or targets another contract")
        layer = int(unit["layer"])
        start, stop = int(unit["expert_start"]), int(unit["expert_stop"])
        if layer not in MAIN_ROUTED_LAYERS or not 0 <= start < stop <= NUM_EXPERTS:
            raise ValueError("work-unit range is outside the main routed surface")
        for expert in range(start, stop):
            if (layer, expert) in coverage:
                raise ValueError("work units overlap")
            coverage.add((layer, expert))
        pending.append(digest)
        units[digest] = unit
    expected = {(layer, expert) for layer in MAIN_ROUTED_LAYERS for expert in range(NUM_EXPERTS)}
    if coverage != expected:
        raise ValueError("work units do not close every main routed expert")
    body = {
        "schema": WORK_STATE_SCHEMA,
        "contract_sha256": contract_sha,
        "sequence": 0,
        "previous_state_sha256": None,
        "units": units,
        "pending": pending,
        "active": {},
        "completed": {},
    }
    return _seal(body, "state_sha256")


def verify_work_state(contract: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    contract_sha = verify_contract(contract)
    state_sha = _verify_seal(state, WORK_STATE_SCHEMA, "state_sha256")
    if state.get("contract_sha256") != contract_sha:
        raise ValueError("work state targets another contract")
    units = state.get("units")
    pending = state.get("pending")
    active = state.get("active")
    completed = state.get("completed")
    if not isinstance(units, Mapping) or not isinstance(pending, list) or not isinstance(active, Mapping) or not isinstance(completed, Mapping):
        raise ValueError("work-state domains are malformed")
    active_units = [row.get("work_unit_sha256") for row in active.values() if isinstance(row, Mapping)]
    if (
        len(active_units) != len(active)
        or len(pending) != len(set(pending))
        or len(active_units) != len(set(active_units))
        or set(pending) & set(active_units)
        or set(pending) & set(completed)
        or set(active_units) & set(completed)
        or set(pending) | set(active_units) | set(completed) != set(units)
    ):
        raise ValueError("work-state pending/active/completed partition differs")
    for worker, claim in active.items():
        if not isinstance(worker, str) or not isinstance(claim, Mapping):
            raise ValueError("work-state claim is malformed")
        _verify_seal(claim, WORK_CLAIM_SCHEMA, "claim_sha256")
        if claim.get("worker_id") != worker or claim.get("contract_sha256") != contract_sha:
            raise ValueError("work-state claim binding differs")
    return state_sha


def _work_successor(
    contract: Mapping[str, Any], state: Mapping[str, Any], **updates: Any
) -> dict[str, Any]:
    previous = verify_work_state(contract, state)
    body = copy.deepcopy(dict(state))
    del body["state_sha256"]
    body.update(copy.deepcopy(updates))
    body["sequence"] = int(state["sequence"]) + 1
    body["previous_state_sha256"] = previous
    return _seal(body, "state_sha256")


def claim_next_work_unit(
    contract: Mapping[str, Any], state: Mapping[str, Any], *, worker_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_work_state(contract, state)
    if worker_id not in {f"b200-{index}" for index in range(4)}:
        raise ValueError("worker must be one of the four B200 slots")
    if worker_id in state["active"]:
        raise ValueError("worker already owns an active unit")
    pending = list(state["pending"])
    if not pending:
        raise ValueError("no unclaimed work unit remains")
    unit_sha = pending.pop(0)
    unit = state["units"][unit_sha]
    claim = _seal(
        {
            "schema": WORK_CLAIM_SCHEMA,
            "contract_sha256": contract["contract_sha256"],
            "parent_state_sha256": state["state_sha256"],
            "worker_id": worker_id,
            "work_unit_sha256": unit_sha,
            "layer": unit["layer"],
            "expert_start": unit["expert_start"],
            "expert_stop": unit["expert_stop"],
        },
        "claim_sha256",
    )
    active = copy.deepcopy(dict(state["active"]))
    active[worker_id] = claim
    return _work_successor(contract, state, pending=pending, active=active), claim


def complete_work_unit(
    contract: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    worker_id: str,
    work_unit_receipt_sha256: str,
) -> dict[str, Any]:
    verify_work_state(contract, state)
    active = copy.deepcopy(dict(state["active"]))
    claim = active.pop(worker_id, None)
    if not isinstance(claim, Mapping):
        raise ValueError("worker has no active unit")
    completed = copy.deepcopy(dict(state["completed"]))
    completed[claim["work_unit_sha256"]] = {
        "worker_id": worker_id,
        "claim_sha256": claim["claim_sha256"],
        "work_unit_receipt_sha256": _require_hash(
            work_unit_receipt_sha256, "work-unit receipt"
        ),
    }
    return _work_successor(contract, state, active=active, completed=completed)


@dataclass(frozen=True)
class EncodeRequest:
    contract_sha256: str
    layer: int
    expert: int
    bits: int
    tensor_names: Mapping[str, str]
    source_weights: Mapping[str, Any]
    capture: Glm53CaptureView
    preparation: Mapping[str, Any]


class DirectMCGBackend(Protocol):
    """Reviewed prepared-source entrypoint for fixed-rate expert triplets."""

    def identity(self) -> Mapping[str, Any]: ...

    def encode_expert(self, request: EncodeRequest) -> Mapping[str, Any]: ...

    def encode_experts(
        self,
        requests: Sequence[EncodeRequest],
        *,
        max_inflight_experts: int | None = None,
    ) -> Sequence[Mapping[str, Any]]: ...


def _verify_backend(contract: Mapping[str, Any], backend: DirectMCGBackend) -> dict[str, Any]:
    bits = contract_bits(contract)
    identity = dict(backend.identity())
    if (
        identity.get("recipe_id") != recipe_id_for_bits(bits)
        or identity.get("bits") != bits
        or identity.get("codec_family") != "exl3-mcg"
        or identity.get("codec_class")
        != "r7_encoder.r10_codec.R10TrellisCodec"
        or identity.get("public_codec_adapter") != "Exl3MCGCodec"
        or identity.get("sqg_orchestration_imported") is not False
        or identity.get("mcg_multiplier_hex") != "0xCBAC1FED"
        or identity.get("prepared_source_tree_sha256") != contract["recipe"]["tree_sha256"]
        or identity.get("exllama_extension_sha256") != contract["exllama"]["extension_sha256"]
        or identity.get("reviewed_glm53_adapter") is not True
    ):
        raise ValueError("loaded numerical backend differs from the sealed GLM53 adapter")
    return identity


def _expert_path(root: Path, layer: int, expert: int) -> Path:
    return root / "experts" / f"layer-{layer:03d}" / f"expert-{expert:03d}.json"


def verify_expert_receipt(
    root: str | Path,
    receipt: Mapping[str, Any] | str | Path,
    *,
    contract_sha256: str,
    expected_bits: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    row = json.loads(Path(receipt).read_text()) if isinstance(receipt, (str, Path)) else dict(receipt)
    _verify_seal(row, EXPERT_RECEIPT_SCHEMA, "receipt_sha256")
    bits = int(row.get("bits", -1))
    recipe_id_for_bits(bits)
    if expected_bits is not None and bits != expected_bits:
        raise ValueError("expert receipt rate differs from its contract")
    if (
        row.get("contract_sha256") != contract_sha256
        or row.get("bits") != bits
        or row.get("projections") != list(PROJECTIONS)
        or row.get("candidate_rate_grid") is not False
        or row.get("global_allocator") is not False
        or row.get("down_candidate_conditioned") is not True
    ):
        raise ValueError("expert receipt semantic binding differs")
    choices = row.get("choices")
    if not isinstance(choices, Mapping) or set(choices) != set(PROJECTIONS):
        raise ValueError("expert receipt does not close its triplet")
    store = PackedMCGPayloadStore(root / "payload-store")
    for projection, choice in choices.items():
        verified = store.verify_choice(choice)
        if (
            verified.get("layer") != row.get("layer")
            or verified.get("expert") != row.get("expert")
            or verified.get("projection") != projection
            or verified.get("bits") != bits
            or verified.get("reconstruction_closure", {}).get("shape")
            != list(projection_shape(projection))
            or verified.get("param_count") != math.prod(projection_shape(projection))
        ):
            raise ValueError("expert choice binding differs")
    evidence = row.get("recipe_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("expert recipe evidence is absent")
    _validate_recipe_evidence(evidence, bits=bits)
    for key in (
        "profile_selection_sha256",
        "permutation_sha256",
        "gate_up_hessian_sha256",
        "down_hessian_sha256",
        "decoded_gate_reconstruction_sha256",
        "decoded_up_reconstruction_sha256",
    ):
        _require_hash(evidence.get(key), f"recipe_evidence.{key}")
    hessian = evidence.get("hessian_artifact")
    if not isinstance(hessian, Mapping):
        raise ValueError("expert receipt lacks its routed Hessian artifact")
    hessian_path = Path(str(hessian.get("path", ""))).resolve()
    try:
        hessian_path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("expert Hessian artifact escapes the output root") from error
    if (
        hessian.get("schema")
        != "quant-pipeline.glm53-routed-p2-hessian-pair.v1"
        or hessian.get("stored_dtype") != "float16"
        or hessian.get("gate_up_shape") != [HIDDEN_SIZE, HIDDEN_SIZE]
        or hessian.get("down_shape") != [INTERMEDIATE_SIZE, INTERMEDIATE_SIZE]
        or not hessian_path.is_file()
        or hessian_path.is_symlink()
        or hessian_path.stat().st_size != hessian.get("bytes")
        or sha256_file(hessian_path) != hessian.get("sha256")
        or hessian.get("gate_up_exact_fp32_sha256")
        != evidence["gate_up_hessian_sha256"]
        or hessian.get("down_exact_fp32_sha256")
        != evidence["down_hessian_sha256"]
    ):
        raise ValueError("expert Hessian artifact binding differs")
    return row


def encode_work_unit(
    *,
    contract: Mapping[str, Any],
    work_unit: Mapping[str, Any],
    source: Glm53BF16Source,
    capture: Glm53CaptureView,
    backend: DirectMCGBackend,
    output_root: str | Path,
    device: str = "cuda:0",
    max_inflight_experts: int = 28,
) -> dict[str, Any]:
    """Execute/resume one claimed unit; the caller owns GPU process placement."""

    contract_sha = verify_contract(contract)
    bits = int(contract.get("rate", {}).get("bits", work_unit.get("bits", -1)))
    recipe_id_for_bits(bits)
    _verify_seal(work_unit, WORK_UNIT_SCHEMA, "work_unit_sha256")
    if (
        work_unit.get("contract_sha256") != contract_sha
        or work_unit.get("layer") != capture.layer
        or work_unit.get("bits") != bits
    ):
        raise ValueError("work unit/capture/contract binding differs")
    backend_identity = _verify_backend(contract, backend)
    output_root = Path(output_root)
    store = PackedMCGPayloadStore(output_root / "payload-store")
    completed_by_expert: dict[int, str] = {}
    pending: list[tuple[int, Path, EncodeRequest]] = []
    start, stop = int(work_unit["expert_start"]), int(work_unit["expert_stop"])
    for expert in range(start, stop):
        path = _expert_path(output_root, capture.layer, expert)
        if path.exists():
            receipt = verify_expert_receipt(
                output_root,
                path,
                contract_sha256=contract_sha,
                expected_bits=bits,
            )
            completed_by_expert[expert] = receipt["receipt_sha256"]
            continue
        weights = source.load_triplet(capture.layer, expert, device=device)
        request = EncodeRequest(
            contract_sha256=contract_sha,
            layer=capture.layer,
            expert=expert,
            bits=bits,
            tensor_names={p: tensor_name(capture.layer, expert, p) for p in PROJECTIONS},
            source_weights=weights,
            capture=capture,
            preparation=contract["preparation"],
        )
        pending.append((expert, path, request))

    if pending:
        encoded = list(
            backend.encode_experts(
                [request for _expert, _path, request in pending],
                max_inflight_experts=max_inflight_experts,
            )
        )
        if len(encoded) != len(pending):
            raise ValueError("prepared backend batch result census differs")
    else:
        encoded = []

    for (expert, path, _request), raw_result in zip(pending, encoded, strict=True):
        result = dict(raw_result)
        evidence = result.get("recipe_evidence")
        payloads = result.get("projections")
        if not isinstance(evidence, Mapping) or not isinstance(payloads, Mapping):
            raise ValueError("prepared backend returned an incomplete expert triplet")
        if bits == 4:
            _validate_recipe_evidence(evidence)
        else:
            _validate_recipe_evidence(evidence, bits=bits)
        choices: dict[str, Any] = {}
        predecessor = work_unit["work_unit_sha256"]
        for projection in PROJECTIONS:
            payload = payloads.get(projection)
            if not isinstance(payload, Mapping):
                raise ValueError(f"prepared backend omitted {projection}")
            choices[projection] = store.put_choice(
                layer=capture.layer,
                expert=expert,
                projection=projection,
                choice_id=f"L{capture.layer:03d}.E{expert:03d}.{projection}.K{bits}",
                trellis=payload["trellis"],
                suh=payload["suh"],
                svh=payload["svh"],
                mcg=payload["mcg"],
                reconstruction=payload["reconstruction"],
                vector_topology=payload["vector_topology"],
                reader_abi_sha256=backend_identity["reader_abi_sha256"],
                provenance={
                    "contract_sha256": contract_sha,
                    "backend": backend_identity,
                    "source_payload_sha256": source.rows[tensor_name(capture.layer, expert, projection)]["source_payload_sha256"],
                },
                predecessor_state_hash=predecessor,
            )
            predecessor = choices[projection]["choice_sha256"]
        body = {
            "schema": EXPERT_RECEIPT_SCHEMA,
            "contract_sha256": contract_sha,
            "work_unit_sha256": work_unit["work_unit_sha256"],
            "layer": capture.layer,
            "expert": expert,
            "bits": bits,
            "projections": list(PROJECTIONS),
            "candidate_rate_grid": False,
            "global_allocator": False,
            "down_candidate_conditioned": True,
            "capture_binding": capture.binding(),
            "backend": backend_identity,
            "choices": choices,
            "recipe_evidence": dict(evidence),
        }
        receipt = _seal(body, "receipt_sha256")
        write_json(path, receipt)
        verify_expert_receipt(
            output_root,
            path,
            contract_sha256=contract_sha,
            expected_bits=bits,
        )
        completed_by_expert[expert] = receipt["receipt_sha256"]
    completed = [completed_by_expert[expert] for expert in range(start, stop)]
    unit_receipt = _seal(
        {
            "schema": "quant-pipeline.glm53-direct-mcg-work-unit-receipt.v1",
            "contract_sha256": contract_sha,
            "work_unit_sha256": work_unit["work_unit_sha256"],
            "layer": capture.layer,
            "expert_start": start,
            "expert_stop": stop,
            "expert_receipt_sha256": completed,
            "complete": len(completed) == stop - start,
        },
        "receipt_sha256",
    )
    path = output_root / "work-units" / f"{work_unit['work_unit_sha256']}.json"
    write_json(path, unit_receipt)
    return unit_receipt


def seal_layer(output_root: str | Path, contract: Mapping[str, Any], layer: int) -> dict[str, Any]:
    contract_sha = verify_contract(contract)
    bits = contract_bits(contract)
    if layer not in MAIN_ROUTED_LAYERS:
        raise ValueError("main layer receipt excludes MTP45")
    root = Path(output_root)
    receipts = []
    choices = []
    for expert in range(NUM_EXPERTS):
        receipt = verify_expert_receipt(
            root,
            _expert_path(root, layer, expert),
            contract_sha256=contract_sha,
            expected_bits=bits,
        )
        receipts.append(receipt["receipt_sha256"])
        choices.extend(receipt["choices"][p]["choice_sha256"] for p in PROJECTIONS)
    body = {
        "schema": LAYER_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha,
        "layer": layer,
        "experts": NUM_EXPERTS,
        "matrix_count": NUM_EXPERTS * len(PROJECTIONS),
        "bits": bits,
        "expert_receipt_sha256": receipts,
        "choice_sha256": choices,
        "complete": True,
    }
    receipt = _seal(body, "receipt_sha256")
    write_json(root / "layers" / f"layer-{layer:03d}.json", receipt)
    return receipt


def build_materialization_plan(
    *,
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    main_layer_receipts: Sequence[Mapping[str, Any]],
    mtp_adapter_receipt: Mapping[str, Any],
    reader_abi_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close native-copy plus all packed choices before any shard is written."""

    contract_sha = verify_contract(contract)
    bits = contract_bits(contract)
    rows = inventory_tensor_map(inventory)
    if inventory["inventory_sha256"] != contract["inventory_sha256"]:
        raise ValueError("materializer inventory differs from encode contract")
    layer_map: dict[int, Mapping[str, Any]] = {}
    all_choices: list[str] = []
    for receipt in main_layer_receipts:
        _verify_seal(receipt, LAYER_RECEIPT_SCHEMA, "receipt_sha256")
        layer = int(receipt.get("layer", -1))
        if (
            receipt.get("contract_sha256") != contract_sha
            or receipt.get("complete") is not True
            or receipt.get("matrix_count") != NUM_EXPERTS * 3
            or receipt.get("bits") != bits
            or layer in layer_map
        ):
            raise ValueError("main layer materialization receipt differs")
        layer_map[layer] = receipt
        all_choices.extend(receipt["choice_sha256"])
    if set(layer_map) != set(MAIN_ROUTED_LAYERS) or len(all_choices) != MAIN_MATRIX_COUNT:
        raise ValueError(
            "materialization lacks the complete main K4 choice surface"
            if bits == 4
            else "materialization lacks the complete main K6 choice surface"
        )
    mtp_schema = (
        MTP_ADAPTER_RECEIPT_SCHEMA
        if bits == 4
        else "quant-pipeline.glm53-uniform-k6-mtp-adapter-receipt.v1"
    )
    _verify_seal(mtp_adapter_receipt, mtp_schema, "receipt_sha256")
    if (
        mtp_adapter_receipt.get("layer") != MTP_LAYER
        or mtp_adapter_receipt.get("matrix_count") != NUM_EXPERTS * 3
        or mtp_adapter_receipt.get("bits") != bits
        or mtp_adapter_receipt.get("qualified") is not True
    ):
        raise ValueError("MTP45 is not separately adapter-qualified")
    reader_receipt_sha256 = None
    reader_qualified = False
    if reader_abi_receipt is not None:
        _verify_seal(
            reader_abi_receipt,
            "quant-pipeline.glm53-exl3-reader-abi-receipt.v1",
            "receipt_sha256",
        )
        if (
            reader_abi_receipt.get("qualified") is not True
            or reader_abi_receipt.get("bits") != bits
            or reader_abi_receipt.get("tp_sizes") != ([2, 4] if bits == 4 else [4])
            or reader_abi_receipt.get("exact_reconstruction_checked") is not True
        ):
            raise ValueError("packed GLM53 reader ABI is not qualified")
        reader_receipt_sha256 = reader_abi_receipt["receipt_sha256"]
        reader_qualified = True
    native = sorted(
        row["tensor_name"]
        for row in rows.values()
        if row.get("scope") not in {"routed_expert", "mtp_routed_expert"}
    )
    body = {
        "schema": materialization_plan_schema_for_bits(bits),
        "contract_sha256": contract_sha,
        "inventory_sha256": inventory["inventory_sha256"],
        "main_layer_receipt_sha256": [layer_map[layer]["receipt_sha256"] for layer in MAIN_ROUTED_LAYERS],
        "mtp_adapter_receipt_sha256": mtp_adapter_receipt["receipt_sha256"],
        "reader_abi_receipt_sha256": reader_receipt_sha256,
        "reader_abi_qualified": reader_qualified,
        "main_choice_count": MAIN_MATRIX_COUNT,
        "mtp_choice_count": NUM_EXPERTS * 3,
        "total_choice_count": MAIN_MATRIX_COUNT + NUM_EXPERTS * 3,
        "native_tensor_count": len(native),
        "native_tensor_names_sha256": sha256_bytes(canonical_json(native)),
        "policy": {
            "routed": "pack_exact_choice_payloads",
            "nonrouted": "copy_official_source_native",
            "atomic_shards": True,
            "resume_requires_shard_receipts": True,
            "serving_ready_only_after_reader_audit": True,
        },
        "launch_authorized": False,
    }
    return _seal(body, "plan_sha256")


def seal_materialization_receipt(
    plan: Mapping[str, Any], *, shard_receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    schema = str(plan.get("schema", ""))
    if schema == MATERIALIZATION_PLAN_SCHEMA:
        bits = 4
    elif schema == K6_MATERIALIZATION_PLAN_SCHEMA:
        bits = 6
    else:
        raise ValueError("foreign materialization plan schema")
    plan_sha = _verify_seal(plan, schema, "plan_sha256")
    if not shard_receipts:
        raise ValueError("materialization produced no shard receipts")
    names: set[str] = set()
    hashes: list[str] = []
    for row in shard_receipts:
        _verify_seal(
            row,
            f"quant-pipeline.glm53-k{bits}-materialized-shard-receipt.v1",
            "receipt_sha256",
        )
        name = row.get("shard")
        if (
            row.get("plan_sha256") != plan_sha
            or row.get("complete") is not True
            or not isinstance(name, str)
            or name in names
            or _HASH.fullmatch(str(row.get("shard_sha256", ""))) is None
        ):
            raise ValueError("materialized shard receipt differs")
        names.add(name)
        hashes.append(row["receipt_sha256"])
    body = {
        "schema": materialization_receipt_schema_for_bits(bits),
        "plan_sha256": plan_sha,
        "shards": sorted(names),
        "shard_receipt_sha256": hashes,
        "routed_choice_count": plan["total_choice_count"],
        "native_tensor_count": plan["native_tensor_count"],
        "reader_audit_required_before_publication": True,
        "complete": True,
    }
    return _seal(body, "receipt_sha256")
