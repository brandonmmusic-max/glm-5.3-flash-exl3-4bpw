"""Auditable internal Qwen EXL3/MCG checkpoint assembly.

This is the complete carrier-plus-payload assembly format, **not** upstream
``btx-atoms-v1``.  Official BTX atom emission lives in :mod:`official_btx`.
The internal format keeps packed codec payloads rather than re-encoding reconstructed
weights.  Every non-MoE source tensor is carried with a payload hash, and each
quantized expert tensor binds its selected bits, exact bytes, transform-vector
topology, reconstruction oracle, causal predecessor, and source provenance.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from ..core.artifacts import canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json
from .exact_payload import ExactCodecPayloadStore, tensor_sha256


INSTALLED_SCHEMA = "quant-pipeline.qwen-installed-layer.v3"
INTERNAL_CHECKPOINT_SCHEMA = "quant-pipeline.qwen-exl3-mcg-carrier.v1"
INTERNAL_INDEX_SCHEMA = "quant-pipeline.qwen-exl3-mcg-carrier-index.v1"
# Private compatibility constants.  They never denote upstream
# ``btx-atoms-v1`` and are intentionally not package exports.
BTX_SCHEMA = INTERNAL_CHECKPOINT_SCHEMA
BTX_INDEX_SCHEMA = INTERNAL_INDEX_SCHEMA
BYTE_SEMANTICS = "codec-payload-including-codec-vectors-excluding-container"
COST_SCHEMA = "quant-pipeline.installed-codec-cost.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_STACKED_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
_LEGAL_SHARED_GROUPS = {
    ("gate_proj", "suh"): "gate_up.suh",
    ("up_proj", "suh"): "gate_up.suh",
    ("down_proj", "svh"): "down.svh",
}


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_save_file(tensors: Mapping[str, Any], path: Path, metadata: Mapping[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(dict(tensors), temporary, metadata=dict(metadata))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _copy_choice(source: ExactCodecPayloadStore, destination: ExactCodecPayloadStore, row: Mapping[str, Any]) -> dict[str, Any]:
    verified = source.verify_choice(row)
    tensors = {name: source.load_tensor(ref) for name, ref in verified["objects"].items()}
    copied = destination.put_choice(
        layer=int(verified["layer"]),
        expert=int(verified["expert"]),
        projection=verified["projection"],
        choice_id=verified["choice_id"],
        bits=int(verified["bits"]),
        trellis=tensors["trellis"],
        suh=tensors["suh"],
        svh=tensors["svh"],
        reconstruction=tensors["reconstruction"],
        vector_topology=verified["vector_topology"],
        provenance=verified["provenance"],
        predecessor_state_hash=verified["predecessor_state_hash"],
    )
    if copied["packed_sha256"] != verified["packed_sha256"] or copied["reconstruction_sha256"] != verified["reconstruction_sha256"]:
        raise ValueError("copying an exact codec choice changed its bytes")
    return copied


def _shared_group(projection: str, role: str) -> str:
    try:
        return _LEGAL_SHARED_GROUPS[(projection, role)]
    except KeyError as error:
        raise ValueError(
            f"illegal layer-shared vector family: {projection}.{role}"
        ) from error


def installed_cost_breakdown(choices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive semantic allocator cost independently from exact choice refs."""
    private_slots: list[dict[str, Any]] = []
    shared_objects: dict[str, dict[str, Any]] = {}
    deployment_objects: dict[str, Mapping[str, Any]] = {}
    logical = 0
    for choice in choices:
        refs = choice.get("objects")
        if not isinstance(refs, Mapping) or set(refs) != {"trellis", "suh", "svh", "reconstruction"}:
            raise ValueError("installed choice exact object set is malformed")
        derived_logical = sum(int(refs[role]["bytes"]) for role in ("trellis", "suh", "svh"))
        if int(choice.get("logical_payload_bytes", -1)) != derived_logical:
            raise ValueError("installed choice logical bytes differ from exact trellis-plus-vector bytes")
        logical += derived_logical
        topology = choice.get("vector_topology")
        if not isinstance(topology, Mapping) or set(topology) != {"suh", "svh"}:
            raise ValueError("installed choice vector topology is incomplete")
        for role in ("trellis", "suh", "svh"):
            ref = refs[role]
            deployment_objects[ref["sha256"]] = ref
            scope = "expert_private" if role == "trellis" else topology[role]
            slot = {
                "expert": int(choice["expert"]),
                "projection": str(choice["projection"]),
                "role": role,
                "sha256": ref["sha256"],
                "bytes": int(ref["bytes"]),
            }
            if scope == "expert_private":
                private_slots.append(slot)
            elif scope == "layer_shared":
                group = _shared_group(str(choice["projection"]), role)
                shared_objects.setdefault(
                    group,
                    {"shared_group": group, "sha256": ref["sha256"], "bytes": int(ref["bytes"])},
                )
                if (
                    shared_objects[group]["bytes"] != int(ref["bytes"])
                    or shared_objects[group]["sha256"] != ref["sha256"]
                ):
                    raise ValueError("one semantic shared group contains different bytes")
            else:
                raise ValueError("installed choice vector topology scope is invalid")
    body = {
        "schema": COST_SCHEMA,
        "byte_semantics": BYTE_SEMANTICS,
        "codec_reported_logical_bytes": logical,
        "semantic_expert_private_slots": private_slots,
        "semantic_expert_private_bytes": sum(row["bytes"] for row in private_slots),
        "semantic_layer_shared_objects": sorted(shared_objects.values(), key=lambda row: row["shared_group"]),
        "semantic_layer_shared_bytes": sum(row["bytes"] for row in shared_objects.values()),
        "allocated_payload_bytes": sum(row["bytes"] for row in private_slots)
        + sum(row["bytes"] for row in shared_objects.values()),
        "physical_deployment_object_sha256": sorted(deployment_objects),
        "physical_deployment_bytes": sum(int(row["bytes"]) for row in deployment_objects.values()),
    }
    body["cost_breakdown_sha256"] = sha256_bytes(canonical_json(body))
    return body


def install_layer_payloads(
    *,
    output_dir: str | Path,
    layer: int,
    predecessor_state_hash: str,
    source_checkpoint_sha256: str,
    fit_sha256: str,
    candidate_ledger_sha256: str,
    selected_choices: Sequence[Mapping[str, Any]],
    production_geometry: bool = False,
    expected_allocated_payload_bytes: int | None = None,
) -> dict[str, Any]:
    """Create a self-contained causal layer install from selected payloads."""

    for label, value in (
        ("predecessor_state_hash", predecessor_state_hash),
        ("source_checkpoint_sha256", source_checkpoint_sha256),
        ("fit_sha256", fit_sha256),
        ("candidate_ledger_sha256", candidate_ledger_sha256),
    ):
        _require_hash(value, label)
    root = prepare_empty_destination(output_dir)
    store = ExactCodecPayloadStore(root / "payload-store")
    choices: list[dict[str, Any]] = []
    for raw in selected_choices:
        if "choice" in raw and "store_root" in raw:
            source = ExactCodecPayloadStore(raw["store_root"])
            choice = _copy_choice(source, store, raw["choice"])
        elif "tensors" in raw:
            tensors = raw["tensors"]
            choice = store.put_choice(
                layer=layer,
                expert=int(raw["expert"]),
                projection=str(raw["projection"]),
                choice_id=str(raw["choice_id"]),
                bits=int(raw["bits"]),
                trellis=tensors["trellis"],
                suh=tensors["suh"],
                svh=tensors["svh"],
                reconstruction=tensors["reconstruction"],
                vector_topology=raw["vector_topology"],
                provenance=raw["provenance"],
                predecessor_state_hash=predecessor_state_hash,
            )
        else:
            raise ValueError("selected choice must contain tensors or an exact-store reference")
        if int(choice["layer"]) != layer or choice["predecessor_state_hash"] != predecessor_state_hash:
            raise ValueError("selected choice causal identity mismatch")
        choices.append(choice)
    keys = [(int(row["expert"]), row["projection"]) for row in choices]
    if not choices or len(keys) != len(set(keys)):
        raise ValueError("installed layer choices must be non-empty and unique")
    if production_geometry:
        expected = {(expert, projection) for expert in range(128) for projection in ("gate_proj", "up_proj", "down_proj")}
        if set(keys) != expected:
            raise ValueError("production Qwen layer install must cover 128 experts x 3 projections")

    # The current B12X full-rotation path supports both broadcast and private
    # vectors.  Shared declarations must truly share object bytes.
    shared: dict[tuple[str, str], set[str]] = {}
    for choice in choices:
        for role in ("suh", "svh"):
            scope = choice["vector_topology"].get(role)
            if scope not in {"layer_shared", "expert_private"}:
                raise ValueError("vector topology must label each vector shared or private")
            if scope == "layer_shared":
                _shared_group(str(choice["projection"]), role)
                shared.setdefault((choice["projection"], role), set()).add(choice["objects"][role]["sha256"])
    if any(len(hashes) != 1 for hashes in shared.values()):
        raise ValueError("a layer-shared vector declaration contains different bytes")
    ordered = sorted(choices, key=lambda row: (int(row["expert"]), row["projection"]))
    cost_breakdown = installed_cost_breakdown(ordered)
    if expected_allocated_payload_bytes is not None:
        if isinstance(expected_allocated_payload_bytes, bool) or not isinstance(expected_allocated_payload_bytes, int):
            raise ValueError("expected allocator byte cost must be an integer")
        if cost_breakdown["allocated_payload_bytes"] != expected_allocated_payload_bytes:
            raise ValueError("allocator cost differs from installed exact payload cost")
    elif production_geometry:
        raise ValueError("production install requires an exact allocator byte-cost parity gate")
    payload_objects = {ref["sha256"]: ref for row in ordered for ref in row["objects"].values()}
    body = {
        "schema": INSTALLED_SCHEMA,
        "layer": int(layer),
        "predecessor_state_hash": predecessor_state_hash,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "fit_sha256": fit_sha256,
        "candidate_ledger_sha256": candidate_ledger_sha256,
        "choices": ordered,
        "logical_payload_bytes": sum(int(row["logical_payload_bytes"]) for row in ordered),
        "allocated_payload_bytes": cost_breakdown["allocated_payload_bytes"],
        "cost_breakdown": cost_breakdown,
        "physical_deployment_object_bytes": cost_breakdown["physical_deployment_bytes"],
        "physical_object_bytes": sum(int(row["bytes"]) for row in payload_objects.values()),
        "shared_object_sha256": sorted({next(iter(hashes)) for hashes in shared.values()}),
        "choice_sha256": [row["choice_sha256"] for row in ordered],
    }
    body["installed_checkpoint_sha256"] = sha256_bytes(canonical_json(body))
    write_json(root / "manifest.json", body)
    return verify_installed_layer(root)


def verify_installed_layer(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != INSTALLED_SCHEMA:
        raise ValueError("unsupported Qwen installed-layer manifest")
    expected = _require_hash(manifest.get("installed_checkpoint_sha256"), "installed layer")
    if sha256_bytes(canonical_json({key: value for key, value in manifest.items() if key != "installed_checkpoint_sha256"})) != expected:
        raise ValueError("installed-layer manifest seal mismatch")
    layer = manifest.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("installed layer identity must be a non-negative integer")
    for field in ("predecessor_state_hash", "source_checkpoint_sha256", "fit_sha256", "candidate_ledger_sha256"):
        _require_hash(manifest.get(field), field)
    raw_choices = manifest.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise ValueError("installed layer choices must be a non-empty list")
    store = ExactCodecPayloadStore(root / "payload-store")
    choices = [store.verify_choice(row) for row in raw_choices]
    keys: list[tuple[int, str]] = []
    for choice in choices:
        expert = choice.get("expert")
        projection = choice.get("projection")
        if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
            raise ValueError("installed choice expert identity must be a non-negative integer")
        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError("installed choice projection identity is invalid")
        if choice.get("layer") != layer or isinstance(choice.get("layer"), bool):
            raise ValueError("installed choice layer differs from manifest layer")
        if choice.get("predecessor_state_hash") != manifest["predecessor_state_hash"]:
            raise ValueError("installed choice predecessor differs from manifest predecessor")
        keys.append((expert, str(projection)))
    if len(keys) != len(set(keys)):
        raise ValueError("installed layer choices must be unique by expert and projection")
    if choices != sorted(choices, key=lambda row: (int(row["expert"]), row["projection"])):
        raise ValueError("installed layer choices are not in canonical expert/projection order")
    observed = [row["choice_sha256"] for row in choices]
    if observed != manifest["choice_sha256"]:
        raise ValueError("installed-layer choice list drift")
    shared: dict[tuple[str, str], set[str]] = {}
    for choice in choices:
        for role in ("suh", "svh"):
            if choice["vector_topology"][role] == "layer_shared":
                _shared_group(str(choice["projection"]), role)
                shared.setdefault((str(choice["projection"]), role), set()).add(
                    choice["objects"][role]["sha256"]
                )
    if any(len(hashes) != 1 for hashes in shared.values()):
        raise ValueError("a verified layer-shared vector declaration contains different bytes")
    expected_shared = sorted({next(iter(hashes)) for hashes in shared.values()})
    if manifest.get("shared_object_sha256") != expected_shared:
        raise ValueError("installed-layer shared-object inventory drift")
    objects = {ref["sha256"]: ref for row in choices for ref in row["objects"].values()}
    if sum(int(row["bytes"]) for row in objects.values()) != int(manifest["physical_object_bytes"]):
        raise ValueError("installed-layer physical-byte accounting drift")
    cost_breakdown = installed_cost_breakdown(choices)
    if (
        manifest.get("cost_breakdown") != cost_breakdown
        or manifest.get("logical_payload_bytes") != cost_breakdown["codec_reported_logical_bytes"]
        or manifest.get("allocated_payload_bytes") != cost_breakdown["allocated_payload_bytes"]
        or manifest.get("physical_deployment_object_bytes") != cost_breakdown["physical_deployment_bytes"]
    ):
        raise ValueError("installed-layer semantic/physical cost accounting drift")
    return manifest


def reconcile_installed_allocation(
    allocation_cost: Mapping[str, Any],
    installed_layers: Sequence[str | Path],
) -> dict[str, Any]:
    """Prove selected-ledger cost equals every installed layer and total.

    This is the accounting handoff between ``selected_allocation_cost`` and
    either checkpoint emitter. Emitters must still receive the returned total
    as ``expected_allocated_payload_bytes`` and their output must be audited.
    """

    cost = dict(allocation_cost)
    if cost.get("schema") != "quant-pipeline.selected-allocation-cost.v1":
        raise ValueError("unsupported selected-allocation cost schema")
    expected_seal = _require_hash(cost.get("allocation_cost_sha256"), "selected allocation cost")
    if sha256_bytes(canonical_json({key: value for key, value in cost.items() if key != "allocation_cost_sha256"})) != expected_seal:
        raise ValueError("selected-allocation cost seal mismatch")
    expected_rows = cost.get("selected_layer_costs")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("selected-allocation cost lacks per-layer reconciliation rows")
    expected_by_layer: dict[int, Mapping[str, Any]] = {}
    all_record_hashes: list[str] = []
    derived_private = 0
    derived_shared = 0
    derived_allocated = 0
    derived_layer_shared_costs: list[dict[str, Any]] = []
    for raw in expected_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("selected-allocation layer cost is malformed")
        layer = raw.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError("selected-allocation layer identity is invalid")
        if layer in expected_by_layer:
            raise ValueError("selected-allocation layer cost contains duplicate layers")
        private = raw.get("semantic_expert_private_bytes")
        shared = raw.get("semantic_layer_shared_bytes")
        allocated = raw.get("allocated_payload_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (private, shared, allocated)):
            raise ValueError("selected-allocation layer byte totals are invalid")
        if allocated != private + shared:
            raise ValueError("selected-allocation layer total differs from private plus shared bytes")
        objects = raw.get("semantic_layer_shared_objects")
        if not isinstance(objects, list) or sum(int(row["bytes"]) for row in objects) != shared:
            raise ValueError("selected-allocation layer shared objects differ from shared bytes")
        record_hashes = raw.get("selected_candidate_record_sha256")
        if (
            not isinstance(record_hashes, list)
            or not record_hashes
            or record_hashes != sorted(set(record_hashes))
            or any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in record_hashes)
        ):
            raise ValueError("selected-allocation layer candidate hashes are malformed")
        identities = raw.get("selected_candidate_identities")
        if identities is not None:
            if not isinstance(identities, list) or len(identities) != len(record_hashes):
                raise ValueError("selected-allocation candidate identities are malformed")
            identity_hashes = []
            for identity in identities:
                if not isinstance(identity, Mapping) or set(identity) != {"record_sha256", "candidate_id", "unit_id"}:
                    raise ValueError("selected-allocation candidate identity is malformed")
                digest = _require_hash(identity.get("record_sha256"), "selected candidate record")
                if not isinstance(identity.get("candidate_id"), str) or not identity["candidate_id"]:
                    raise ValueError("selected candidate ID is malformed")
                if not isinstance(identity.get("unit_id"), str) or not identity["unit_id"]:
                    raise ValueError("selected candidate unit ID is malformed")
                identity_hashes.append(digest)
            if sorted(identity_hashes) != record_hashes:
                raise ValueError("selected candidate identities differ from selected hashes")
        expected_by_layer[layer] = raw
        all_record_hashes.extend(record_hashes)
        derived_private += private
        derived_shared += shared
        derived_allocated += allocated
        derived_layer_shared_costs.append(
            {
                "layer": layer,
                "objects": objects,
                "semantic_layer_shared_bytes": shared,
            }
        )
    if len(all_record_hashes) != len(set(all_record_hashes)):
        raise ValueError("selected candidate record hashes occur in more than one layer")
    if cost.get("selected_candidate_record_sha256") != sorted(all_record_hashes):
        raise ValueError("selected-allocation top-level candidate hashes differ from layer rows")
    if cost.get("semantic_expert_private_bytes") != derived_private:
        raise ValueError("selected-allocation top-level private bytes differ from layer rows")
    if cost.get("semantic_layer_shared_bytes") != derived_shared:
        raise ValueError("selected-allocation top-level shared bytes differ from layer rows")
    if cost.get("allocated_payload_bytes") != derived_allocated:
        raise ValueError("selected-allocation top-level total differs from layer rows")
    if cost.get("layer_shared_costs") != derived_layer_shared_costs:
        raise ValueError("selected-allocation layer-shared summary differs from layer rows")
    installed_by_layer: dict[int, Mapping[str, Any]] = {}
    installed_roots: dict[int, str] = {}
    for raw_root in installed_layers:
        manifest = verify_installed_layer(raw_root)
        layer = int(manifest["layer"])
        if layer in installed_by_layer:
            raise ValueError("installed allocation contains duplicate layers")
        installed_by_layer[layer] = manifest
        installed_roots[layer] = str(Path(raw_root).resolve())
    if set(installed_by_layer) != set(expected_by_layer):
        raise ValueError("installed layer set differs from selected allocation")
    reconciled = []
    for layer in sorted(expected_by_layer):
        expected = expected_by_layer[layer]
        installed_manifest = installed_by_layer[layer]
        observed = installed_manifest["cost_breakdown"]
        comparisons = {
            "semantic_expert_private_bytes": observed["semantic_expert_private_bytes"],
            "semantic_layer_shared_objects": observed["semantic_layer_shared_objects"],
            "semantic_layer_shared_bytes": observed["semantic_layer_shared_bytes"],
            "allocated_payload_bytes": observed["allocated_payload_bytes"],
        }
        if any(comparisons[key] != expected.get(key) for key in comparisons):
            raise ValueError(f"installed layer {layer} cost differs from selected allocation")
        expected_identities = expected.get("selected_candidate_identities")
        installed_choices = installed_manifest["choices"]
        provenance_hashes = [
            row.get("provenance", {}).get(
                "selected_candidate_record_sha256",
                row.get("provenance", {}).get("candidate_record_sha256"),
            )
            for row in installed_choices
        ]
        for choice in installed_choices:
            causal_digest = choice.get("provenance", {}).get("causal_candidate_record_sha256")
            if causal_digest is not None:
                _require_hash(causal_digest, "causal candidate record")
        if expected_identities is not None:
            identity_by_hash = {row["record_sha256"]: row for row in expected_identities}
            if any(value not in identity_by_hash for value in provenance_hashes):
                raise ValueError(f"installed layer {layer} choices are not bound to selected candidate records")
            observed_projections: dict[str, set[str]] = {}
            for choice, digest in zip(installed_choices, provenance_hashes, strict=True):
                identity = identity_by_hash[digest]
                if choice["choice_id"] != identity["candidate_id"]:
                    raise ValueError(f"installed layer {layer} choice ID differs from selected candidate")
                expected_unit = f"L{layer}.E{choice['expert']}"
                if identity["unit_id"] != expected_unit:
                    raise ValueError(f"installed layer {layer} choice unit differs from selected candidate")
                observed_projections.setdefault(digest, set()).add(choice["projection"])
            if set(observed_projections) != set(identity_by_hash) or any(
                projections != {"gate_proj", "up_proj", "down_proj"}
                for projections in observed_projections.values()
            ):
                raise ValueError(f"installed layer {layer} does not contain three choices per selected candidate")
        elif any(value is not None for value in provenance_hashes):
            expected_hashes = set(expected["selected_candidate_record_sha256"])
            if set(provenance_hashes) != expected_hashes:
                raise ValueError(f"installed layer {layer} candidate hashes differ from selected allocation")
        reconciled.append(
            {
                "layer": layer,
                "installed_root": installed_roots[layer],
                "installed_checkpoint_sha256": installed_by_layer[layer]["installed_checkpoint_sha256"],
                "selected_candidate_record_sha256": expected["selected_candidate_record_sha256"],
                "allocated_payload_bytes": comparisons["allocated_payload_bytes"],
            }
        )
    total = sum(int(row["allocated_payload_bytes"]) for row in reconciled)
    if total != derived_allocated:
        raise ValueError("installed allocation total differs from selected allocation")
    body = {
        "schema": "quant-pipeline.installed-allocation-reconciliation.v1",
        "allocation_cost_sha256": expected_seal,
        "layers": reconciled,
        "allocated_payload_bytes": total,
    }
    body["reconciliation_sha256"] = sha256_bytes(canonical_json(body))
    return body


def _assign_reconstruction(model: Any, layer: int, expert: int, projection: str, value: Any) -> None:
    import torch

    experts = model.model.layers[layer].mlp.experts
    intermediate = experts.gate_up_proj.shape[1] // 2
    if projection == "gate_proj":
        target = experts.gate_up_proj[expert, :intermediate]
    elif projection == "up_proj":
        target = experts.gate_up_proj[expert, intermediate:]
    elif projection == "down_proj":
        target = experts.down_proj[expert]
    else:  # pragma: no cover - verified earlier
        raise ValueError(projection)
    if tuple(target.shape) != tuple(value.shape):
        raise ValueError(f"reconstruction shape mismatch at L{layer}.E{expert}.{projection}")
    with torch.no_grad():
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def replay_installed_layers(
    model: Any,
    installed_layers: Sequence[str | Path],
    *,
    expected_final_state_hash: str,
    expected_prefix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay exact FP16 reconstructions into a freshly loaded BF16 model.

    The runner's complete accepted prefix is mandatory. Every layer number,
    predecessor state, installed-checkpoint identity, and terminal external
    state is checked. The external state includes stage artifact bytes and is
    therefore supplied by the runner, but it is never returned as a purported
    value derived from model replay.
    """

    _require_hash(expected_final_state_hash, "expected final state")
    prefix = [dict(row) for row in expected_prefix]
    if len(installed_layers) != len(prefix):
        raise ValueError("installed-layer paths do not exactly cover the accepted runner prefix")
    rows = []
    for position, (root_raw, expected) in enumerate(zip(installed_layers, prefix, strict=True)):
        root = Path(root_raw)
        manifest = verify_installed_layer(root)
        layer = int(manifest["layer"])
        if int(expected.get("layer", -1)) != layer:
            raise ValueError(f"installed layer at prefix position {position} differs from runner journal")
        predecessor = _require_hash(expected.get("predecessor_state_hash"), "expected layer predecessor")
        if manifest["predecessor_state_hash"] != predecessor:
            raise ValueError(f"installed layer {layer} predecessor differs from runner journal")
        installed_identity = _require_hash(
            expected.get("installed_checkpoint_sha256"), "expected installed checkpoint"
        )
        if manifest["installed_checkpoint_sha256"] != installed_identity:
            raise ValueError(f"installed layer {layer} checkpoint identity differs from runner journal")
        installed_state = _require_hash(expected.get("installed_state_hash"), "expected installed state")
        store = ExactCodecPayloadStore(root / "payload-store")
        for choice in manifest["choices"]:
            reconstruction = store.load_tensor(choice["objects"]["reconstruction"])
            if tensor_sha256(reconstruction) != choice["reconstruction_sha256"]:
                raise ValueError("installed reconstruction changed during load")
            _assign_reconstruction(model, layer, int(choice["expert"]), choice["projection"], reconstruction)
        rows.append({
            "position": position,
            "layer": layer,
            "predecessor_state_hash": predecessor,
            "installed_checkpoint_sha256": installed_identity,
            "installed_state_hash": installed_state,
        })
    if [row["layer"] for row in rows] != sorted({row["layer"] for row in rows}):
        raise ValueError("installed layers must be unique and ordered")
    if rows and rows[-1]["installed_state_hash"] != expected_final_state_hash:
        raise ValueError("accepted prefix terminal state differs from requested predecessor state")
    body = {
        "schema": "quant-pipeline.qwen-installed-replay.v1",
        "layers": rows,
        "accepted_prefix_length": len(prefix),
        "requested_predecessor_state_hash": expected_final_state_hash,
    }
    body["replay_sha256"] = sha256_bytes(canonical_json(body))
    return body


def _source_index(source: Path) -> tuple[dict[str, str], set[str]]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        mapping = {str(k): str(v) for k, v in json.loads(index_path.read_text())["weight_map"].items()}
        return mapping, set(mapping.values())
    single = source / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError("source checkpoint has no safetensors model")
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        mapping = {key: single.name for key in handle.keys()}
    return mapping, {single.name}


def _payload_key(layer: int, expert: int, projection: str, role: str, topology: str) -> str:
    prefix = f"model.layers.{layer}.mlp.experts"
    if topology == "layer_shared":
        if projection in {"gate_proj", "up_proj"} and role == "suh":
            return f"{prefix}.r7_shared.gate_up_suh"
        if projection == "down_proj" and role == "svh":
            return f"{prefix}.r7_shared.down_svh"
    return f"{prefix}.{expert}.{projection}.{role}"


def emit_internal_qwen_checkpoint(
    *,
    source_checkpoint: str | Path,
    installed_layers: Sequence[str | Path],
    output_dir: str | Path,
    format_version: str,
    expected_allocated_payload_bytes: int | None = None,
) -> dict[str, Any]:
    """Emit a complete Qwen checkpoint carrying exact selected codec bytes."""

    from safetensors import safe_open

    source = Path(source_checkpoint).resolve()
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source checkpoint must be a local directory")
    installed = [(Path(path), verify_installed_layer(path)) for path in installed_layers]
    layers = [int(row[1]["layer"]) for row in installed]
    if layers != sorted(set(layers)):
        raise ValueError("installed layers must be ordered and unique")
    source_hashes = {row[1]["source_checkpoint_sha256"] for row in installed}
    if len(source_hashes) > 1:
        raise ValueError("installed layers disagree on source checkpoint identity")
    allocated_payload_bytes = sum(int(row[1]["allocated_payload_bytes"]) for row in installed)
    if expected_allocated_payload_bytes is not None and allocated_payload_bytes != expected_allocated_payload_bytes:
        raise ValueError("allocator cost differs from internal emitted payload cost")
    destination = prepare_empty_destination(output_dir)
    source_map, source_shards = _source_index(source)
    quantized_layers = set(layers)
    weight_map: dict[str, str] = {}
    tensor_hashes: dict[str, str] = {}
    source_shard_hashes = {name: sha256_file(source / name) for name in sorted(source_shards)}

    # Carry every non-replaced source tensor.  The tensor payload, not its new
    # container offsets, is the byte-exact invariant inherited from the 3.5
    # bpw GLM assembly path.
    for shard_index, shard_name in enumerate(sorted(source_shards), start=1):
        kept: dict[str, Any] = {}
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                match = _STACKED_EXPERT.fullmatch(name)
                if match and int(match.group(1)) in quantized_layers:
                    continue
                tensor = handle.get_tensor(name)
                kept[name] = tensor
                tensor_hashes[name] = tensor_sha256(tensor)
        if not kept:
            continue
        output_name = f"carrier-{shard_index:05d}.safetensors"
        _atomic_save_file(kept, destination / output_name, {"schema": BTX_SCHEMA, "role": "bf16-carrier"})
        for name in kept:
            weight_map[name] = output_name

    unit_records: dict[str, Any] = {}
    layer_records: dict[str, Any] = {}
    logical_payload_bytes = 0
    all_choice_hashes: list[str] = []
    for root, manifest in installed:
        layer = int(manifest["layer"])
        store = ExactCodecPayloadStore(root / "payload-store")
        tensors: dict[str, Any] = {}
        for choice in manifest["choices"]:
            unit = f"L{layer}.E{int(choice['expert'])}.{choice['projection']}"
            refs: dict[str, str] = {}
            for role in ("trellis", "suh", "svh"):
                topology = choice["vector_topology"].get(role, "expert_private") if role != "trellis" else "expert_private"
                key = _payload_key(layer, int(choice["expert"]), choice["projection"], role, topology)
                value = store.load_tensor(choice["objects"][role])
                if key in tensors and tensor_sha256(tensors[key]) != tensor_sha256(value):
                    raise ValueError(f"deduplicated BTX tensor bytes disagree: {key}")
                tensors.setdefault(key, value)
                refs[role] = key
            logical_payload_bytes += int(choice["logical_payload_bytes"])
            all_choice_hashes.append(choice["choice_sha256"])
            unit_records[unit] = {
                "layer": layer,
                "expert": int(choice["expert"]),
                "projection": choice["projection"],
                "bits": int(choice["bits"]),
                "logical_payload_bytes": int(choice["logical_payload_bytes"]),
                "packed_sha256": choice["packed_sha256"],
                "reconstruction_sha256": choice["reconstruction_sha256"],
                "payload_keys": refs,
                "vector_topology": choice["vector_topology"],
                "choice_sha256": choice["choice_sha256"],
                "predecessor_state_hash": choice["predecessor_state_hash"],
                "provenance": choice["provenance"],
            }
        output_name = f"btx-experts-layer-{layer:03d}.safetensors"
        _atomic_save_file(tensors, destination / output_name, {"schema": BTX_SCHEMA, "layer": str(layer)})
        with safe_open(destination / output_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                weight_map[name] = output_name
                tensor_hashes[name] = tensor_sha256(handle.get_tensor(name))
        layer_records[str(layer)] = {
            "file": output_name,
            "bytes": (destination / output_name).stat().st_size,
            "sha256": sha256_file(destination / output_name),
            "installed_checkpoint_sha256": manifest["installed_checkpoint_sha256"],
            "predecessor_state_hash": manifest["predecessor_state_hash"],
            "fit_sha256": manifest["fit_sha256"],
            "candidate_ledger_sha256": manifest["candidate_ledger_sha256"],
            "allocated_payload_bytes": manifest["allocated_payload_bytes"],
            "cost_breakdown_sha256": manifest["cost_breakdown"]["cost_breakdown_sha256"],
        }

    # Non-model assets are copied only after excluding all regenerated surfaces.
    regenerated = set(source_shards) | {"model.safetensors.index.json", "config.json", "quantization_config.json", "BTX_MANIFEST.json", "BTX_MANIFEST.sha256"}
    for path in sorted(source.iterdir(), key=lambda item: item.name):
        if path.name in regenerated:
            continue
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)

    config = json.loads((source / "config.json").read_text())
    quantization = {
        "quant_method": "exl3",
        "format": "btx-qwen-exl3-mcg",
        "format_version": format_version,
        "byte_semantics": BYTE_SEMANTICS,
        "bits": "mixed_tensor",
        "requires_loader_feature": "btx-qwen-exl3-mcg-v1",
    }
    config["quantization_config"] = quantization
    write_json(destination / "config.json", config)
    write_json(destination / "quantization_config.json", quantization)
    index = {
        "schema": BTX_INDEX_SCHEMA,
        "metadata": {"total_size": sum((destination / file).stat().st_size for file in set(weight_map.values()))},
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_json(destination / "model.safetensors.index.json", index)
    files = {
        path.relative_to(destination).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    body = {
        "schema": BTX_SCHEMA,
        "format_version": format_version,
        "byte_semantics": BYTE_SEMANTICS,
        "source": {
            "path": str(source),
            "config_sha256": sha256_file(source / "config.json"),
            "index_sha256": sha256_file(source / "model.safetensors.index.json") if (source / "model.safetensors.index.json").exists() else None,
            "shards": source_shard_hashes,
            "declared_identity": next(iter(source_hashes), None),
        },
        "layers": layer_records,
        "units": unit_records,
        "choice_sha256": sorted(all_choice_hashes),
        "logical_payload_bytes": logical_payload_bytes,
        "allocated_payload_bytes": allocated_payload_bytes,
        "tensor_payload_sha256": dict(sorted(tensor_hashes.items())),
        "files": files,
    }
    body["manifest_sha256"] = sha256_bytes(canonical_json(body))
    write_json(destination / "BTX_MANIFEST.json", body)
    (destination / "BTX_MANIFEST.sha256").write_text(f"{sha256_file(destination / 'BTX_MANIFEST.json')}  BTX_MANIFEST.json\n")
    return body


@runtime_checkable
class RuntimeCheckpointReader(Protocol):
    def load_checkpoint(self, root: Path) -> Mapping[str, Any]: ...


def audit_internal_qwen_checkpoint(
    root: str | Path,
    *,
    runtime_reader: RuntimeCheckpointReader | Callable[[Path], Mapping[str, Any]] | None = None,
    require_runtime_reader: bool = True,
) -> dict[str, Any]:
    """Independently audit files, tensor bytes, metadata, and runtime load."""

    from safetensors import safe_open

    root = Path(root)
    manifest_path = root / "BTX_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []
    if manifest.get("schema") != BTX_SCHEMA:
        failures.append("schema")
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or sha256_bytes(canonical_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})) != expected:
        failures.append("manifest-seal")
    seal_line = (root / "BTX_MANIFEST.sha256").read_text().strip().split()
    if not seal_line or seal_line[0] != sha256_file(manifest_path):
        failures.append("manifest-file-seal")
    declared_files = set(manifest.get("files", {})) | {"BTX_MANIFEST.json", "BTX_MANIFEST.sha256"}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != declared_files:
        failures.append("file-set")
    for relative, row in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            failures.append(f"file:{relative}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    observed: dict[str, str] = {}
    tensor_nbytes: dict[str, int] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in observed:
                    failures.append(f"duplicate-tensor:{name}")
                observed[name] = shard
                tensor = handle.get_tensor(name)
                tensor_nbytes[name] = int(tensor.numel() * tensor.element_size())
                expected_tensor = manifest["tensor_payload_sha256"].get(name)
                if expected_tensor != tensor_sha256(tensor):
                    failures.append(f"tensor:{name}")
    if observed != index["weight_map"]:
        failures.append("index-bijection")
    if manifest.get("allocated_payload_bytes") != sum(
        int(row.get("allocated_payload_bytes", -1)) for row in manifest.get("layers", {}).values()
    ):
        failures.append("allocated-payload-parity")
    semantic_private = 0
    semantic_shared: dict[tuple[int, str], tuple[str, int]] = {}
    derived_logical = 0
    for unit, row in manifest.get("units", {}).items():
        try:
            layer = int(row["layer"])
            projection = str(row["projection"])
            keys = row["payload_keys"]
            topology = row["vector_topology"]
            sizes = {role: tensor_nbytes[keys[role]] for role in ("trellis", "suh", "svh")}
            logical = sum(sizes.values())
            if logical != int(row["logical_payload_bytes"]):
                raise ValueError("logical")
            derived_logical += logical
            semantic_private += sizes["trellis"]
            for role in ("suh", "svh"):
                if topology[role] == "expert_private":
                    semantic_private += sizes[role]
                elif topology[role] == "layer_shared":
                    group = _shared_group(projection, role)
                    value = (keys[role], sizes[role])
                    incumbent = semantic_shared.setdefault((layer, group), value)
                    if incumbent != value:
                        raise ValueError("shared")
                else:
                    raise ValueError("topology")
        except Exception as error:
            failures.append(f"unit-accounting:{unit}:{error}")
    derived_allocated = semantic_private + sum(size for _, size in semantic_shared.values())
    if manifest.get("logical_payload_bytes") != derived_logical:
        failures.append("logical-payload-parity")
    if manifest.get("allocated_payload_bytes") != derived_allocated:
        failures.append("emitted-allocator-parity")
    runtime: Mapping[str, Any] | None = None
    if runtime_reader is None:
        if require_runtime_reader:
            failures.append("runtime-reader-unavailable")
    else:
        runtime = runtime_reader(root) if callable(runtime_reader) and not hasattr(runtime_reader, "load_checkpoint") else runtime_reader.load_checkpoint(root)  # type: ignore[union-attr]
        if not isinstance(runtime, Mapping) or runtime.get("ok") is not True:
            failures.append("runtime-load")
    return {
        "ok": not failures,
        "failures": failures,
        "unit_count": len(manifest.get("units", {})),
        "logical_payload_bytes": int(manifest.get("logical_payload_bytes", 0)),
        "allocated_payload_bytes": int(manifest.get("allocated_payload_bytes", 0)),
        "runtime": dict(runtime or {}),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


class InternalBTXReader:
    """Reference reader for tests and offline inspection, not production proof."""

    def load_checkpoint(self, root: Path) -> Mapping[str, Any]:
        from safetensors import safe_open

        index = json.loads((root / "model.safetensors.index.json").read_text())
        count = 0
        for shard in sorted(set(index["weight_map"].values())):
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                count += len(handle.keys())
        return {"ok": count == len(index["weight_map"]), "tensor_count": count, "reader": "internal-reference-only"}


# Pre-release source compatibility only.  Production callers and package
# exports use the unambiguous internal names above or official_btx.py.
emit_btx_qwen_checkpoint = emit_internal_qwen_checkpoint
audit_btx_qwen_checkpoint = audit_internal_qwen_checkpoint
