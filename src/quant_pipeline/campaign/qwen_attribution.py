"""Native causal attribution for decoded Qwen MoE candidates.

The implementation evaluates the full language model.  It does not call a
layer-local proxy "KLD".  Every Qwen MoE hook recomputes the selected decoded
expert function at the contemporaneous path hidden state and returns

    source_block_output + alpha_layer * (decoded_block_output - source_block_output).

This makes ``alpha=0`` an exact source control and ``alpha=1`` the actual
decoded provisional candidate.  Gradients of teacher-to-path next-token KL
with respect to each alpha are integrated by the caller.  Score-function
Fisher VJPs are projected against exact routed expert residuals, so the
quadratic expert split includes cross-expert terms without one downstream
forward pass per expert.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..calibration.qwen_capture import qwen_moe_layers
from ..candidates.payload_store import ExactPayloadStore
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from ..evaluation.kld_window import verify_kld_window
from ..scoring.attribution import (
    aumann_shapley,
    reconcile_layer_components_for_allocation,
    reconcile_signed_completeness,
    split_layer_damage,
)


PROVISIONAL_SCHEMA = "quant-pipeline.qwen-provisional-decoded-deltas.v1"
ATTRIBUTION_INPUT_SCHEMA = "quant-pipeline.qwen-native-causal-attribution-inputs.v2"
LEGACY_ATTRIBUTION_INPUT_SCHEMA = "quant-pipeline.qwen-native-causal-attribution-inputs.v1"
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_bytes(value: Any) -> bytes:
    import torch

    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    return tensor.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(value: Any) -> str:
    return sha256_bytes(_tensor_bytes(value))


def load_teacher_logits(path: Path) -> np.ndarray:
    """Load canonical LM logits from a sealed teacher/router capture."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        # Production teacher captures seal per-layer router logits alongside
        # the language-model logits.  Their presence is expected and does not
        # make the canonical KLD target ambiguous.
        if "logits" not in handle.keys():
            raise ValueError("teacher safetensors must contain a logits tensor")
        return np.asarray(handle.get_tensor("logits"))


def _dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    if name not in mapping:
        raise ValueError(f"unsupported exact payload dtype: {name}")
    return mapping[name]


def _load_exact_tensor(store: ExactPayloadStore, ref: Mapping[str, Any]):
    import torch

    store.verify_ref(ref)
    raw = bytearray((store.root / str(ref["path"])).read_bytes())
    tensor = torch.frombuffer(raw, dtype=_dtype(str(ref["dtype"]))).clone()
    expected = math.prod(int(value) for value in ref["shape"])
    if tensor.numel() != expected:
        raise ValueError("exact decoded payload shape does not match its bytes")
    return tensor.reshape(tuple(int(value) for value in ref["shape"]))


def persist_provisional_winner_deltas(
    *,
    output_dir: str | Path,
    ledger: Mapping[str, Any],
    payload_store_root: str | Path,
    checkpoint_sources: Mapping[tuple[int, int], Any],
    bit_triplet: Sequence[int] = (4, 4, 4),
) -> Path:
    """Persist source-first deltas for one explicit actual-codec control arm."""

    import torch
    from safetensors.torch import save_file

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    triplet = tuple(int(value) for value in bit_triplet)
    if len(triplet) != 3 or any(value not in (3, 4, 5) for value in triplet):
        raise ValueError("provisional attribution bit triplet must contain three K3/K4/K5 values")
    ledger_sha = _require_hash(ledger.get("ledger_sha256"), "candidate ledger")
    if _hash_json({key: value for key, value in ledger.items() if key != "ledger_sha256"}) != ledger_sha:
        raise ValueError("candidate ledger seal mismatch before provisional delta persistence")
    units = {f"L{int(layer)}.E{int(expert)}": value for (layer, expert), value in checkpoint_sources.items()}
    selected = [row for row in ledger.get("candidates", ()) if tuple(row.get("bit_triplet", ())) == triplet]
    if {str(row["unit_id"]) for row in selected} != set(units) or len(selected) != len(units):
        raise ValueError("provisional triplet does not cover the exact candidate expert inventory")
    store = ExactPayloadStore(payload_store_root)
    store_manifest_path = store.root / "manifest.json"
    if not store_manifest_path.is_file() or store_manifest_path.is_symlink():
        raise ValueError("provisional exact payload store manifest is missing")
    store_manifest = json.loads(store_manifest_path.read_text())
    store_seal = _require_hash(store_manifest.get("manifest_sha256"), "exact payload store manifest")
    if _hash_json({key: value for key, value in store_manifest.items() if key != "manifest_sha256"}) != store_seal:
        raise ValueError("exact payload store manifest seal mismatch")
    if store_seal != ledger.get("exact_payload_store", {}).get("manifest_sha256"):
        raise ValueError("candidate ledger names a different exact payload store manifest")
    by_layer: dict[int, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    layer_rows = []
    for layer, records in sorted(by_layer.items()):
        tensors: dict[str, Any] = {}
        winners = []
        for record in sorted(records, key=lambda item: int(item["expert"])):
            source_expert = units[str(record["unit_id"])]
            expert = int(record["expert"])
            projection_rows = {}
            for projection in _PROJECTIONS:
                source = torch.as_tensor(getattr(source_expert, projection)).detach().cpu().contiguous()
                ref = record["projections"][projection]["exact_payload_refs"]["reconstruction_hf"]
                decoded = _load_exact_tensor(store, ref)
                if tuple(decoded.shape) != tuple(source.shape):
                    raise ValueError(f"decoded {record['unit_id']} {projection} shape differs from source")
                delta = decoded.float() - source.float()
                key = f"expert_{expert:03d}.{projection}.delta_f32"
                tensors[key] = delta.contiguous()
                projection_rows[projection] = {
                    "delta_tensor": key,
                    "delta_sha256": _tensor_sha256(delta),
                    "source_sha256": _tensor_sha256(source),
                    "decoded_ref": dict(ref),
                    "decoded_sha256": _tensor_sha256(decoded),
                }
            winners.append({
                "unit_id": str(record["unit_id"]),
                "expert": expert,
                "candidate_id": str(record["candidate_id"]),
                "candidate_record_sha256": _require_hash(record.get("record_sha256"), "candidate record"),
                "projections": projection_rows,
            })
        target = output / f"layer-{layer:03d}.safetensors"
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        save_file(tensors, temporary)
        os.replace(temporary, target)
        layer_rows.append({
            "layer": layer,
            "file": target.name,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "winners": winners,
        })
    body = {
        "schema": PROVISIONAL_SCHEMA,
        "definition": "decoded_actual_codec_minus_exact_source_f32",
        "source_control": "alpha-zero-is-unmodified-source-block-output",
        "candidate_control": "alpha-one-is-decoded-block-output-in-model-dtype",
        "candidate_ledger_sha256": ledger_sha,
        "bit_triplet": list(triplet),
        "payload_store_manifest_sha256": ledger["exact_payload_store"]["manifest_sha256"],
        "layers": layer_rows,
    }
    body["manifest_sha256"] = _hash_json(body)
    path = output / "manifest.json"
    write_json(path, body)
    verify_provisional_winner_deltas(path, payload_store_root=store.root)
    return path


def verify_provisional_winner_deltas(
    manifest_path: str | Path,
    *,
    payload_store_root: str | Path,
) -> dict[str, Any]:
    from safetensors import safe_open

    path = Path(manifest_path).resolve()
    body = json.loads(path.read_text())
    if body.get("schema") != PROVISIONAL_SCHEMA:
        raise ValueError("unsupported provisional decoded-delta manifest")
    seal = _require_hash(body.get("manifest_sha256"), "provisional delta manifest")
    if _hash_json({key: value for key, value in body.items() if key != "manifest_sha256"}) != seal:
        raise ValueError("provisional decoded-delta manifest seal mismatch")
    store = ExactPayloadStore(payload_store_root)
    store_manifest_path = store.root / "manifest.json"
    if not store_manifest_path.is_file() or store_manifest_path.is_symlink():
        raise ValueError("provisional exact payload store manifest is missing")
    store_manifest = json.loads(store_manifest_path.read_text())
    store_seal = _require_hash(store_manifest.get("manifest_sha256"), "exact payload store manifest")
    if _hash_json({key: value for key, value in store_manifest.items() if key != "manifest_sha256"}) != store_seal:
        raise ValueError("exact payload store manifest seal mismatch")
    if store_seal != body.get("payload_store_manifest_sha256"):
        raise ValueError("provisional winner names a different exact payload store manifest")
    seen: set[tuple[int, int]] = set()
    for layer_row in body.get("layers", ()):
        layer = int(layer_row["layer"])
        target = path.parent / str(layer_row["file"])
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != int(layer_row["bytes"])
            or sha256_file(target) != layer_row["sha256"]
        ):
            raise ValueError("provisional decoded-delta tensor file is missing or drifted")
        with safe_open(target, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for winner in layer_row["winners"]:
                identity = (layer, int(winner["expert"]))
                if identity in seen:
                    raise ValueError("duplicate provisional layer/expert winner")
                seen.add(identity)
                for projection in _PROJECTIONS:
                    row = winner["projections"][projection]
                    if row["delta_tensor"] not in keys:
                        raise ValueError("provisional delta tensor is absent")
                    delta = handle.get_tensor(row["delta_tensor"])
                    if _tensor_sha256(delta) != row["delta_sha256"]:
                        raise ValueError("provisional delta tensor identity mismatch")
                    store.verify_ref(row["decoded_ref"])
                    decoded = _load_exact_tensor(store, row["decoded_ref"])
                    if _tensor_sha256(decoded) != row["decoded_sha256"]:
                        raise ValueError("provisional decoded tensor identity mismatch")
    if not seen:
        raise ValueError("provisional decoded-delta manifest has no winners")
    return body


def load_provisional_decoded_weights(
    model: Any,
    manifest_path: str | Path,
    *,
    payload_store_root: str | Path,
) -> dict[int, dict[str, Any]]:
    """Load exact decoded winners only after source and delta identities close."""

    import torch
    from safetensors import safe_open

    manifest = verify_provisional_winner_deltas(manifest_path, payload_store_root=payload_store_root)
    blocks = qwen_moe_layers(model)
    store = ExactPayloadStore(payload_store_root)
    result: dict[int, dict[str, Any]] = {}
    root = Path(manifest_path).resolve().parent
    for layer_row in manifest["layers"]:
        layer = int(layer_row["layer"])
        if layer not in blocks:
            raise ValueError(f"provisional candidate names absent Qwen MoE layer {layer}")
        experts = blocks[layer].experts
        gate, up = experts.gate_up_proj.detach().chunk(2, dim=1)
        decoded = {
            "gate_proj": gate.clone(),
            "up_proj": up.clone(),
            "down_proj": experts.down_proj.detach().clone(),
        }
        with safe_open(root / layer_row["file"], framework="pt", device="cpu") as deltas:
            for winner in layer_row["winners"]:
                expert = int(winner["expert"])
                for projection in _PROJECTIONS:
                    row = winner["projections"][projection]
                    source = gate[expert] if projection == "gate_proj" else up[expert] if projection == "up_proj" else experts.down_proj.detach()[expert]
                    if _tensor_sha256(source) != row["source_sha256"]:
                        raise RuntimeError("loaded Qwen source tensor differs from provisional source control")
                    exact = _load_exact_tensor(store, row["decoded_ref"])
                    observed_delta = exact.float() - source.detach().cpu().float()
                    stored_delta = deltas.get_tensor(row["delta_tensor"])
                    if not torch.equal(observed_delta, stored_delta):
                        raise RuntimeError("persisted decoded delta no longer equals exact decoded-minus-source")
                    decoded[projection][expert] = exact.to(decoded[projection].device, decoded[projection].dtype)
        result[layer] = decoded
    return result


def _decoded_expert_block(block: Any, hidden_states: Any, decoded: Mapping[str, Any]):
    """Return exact decoded output plus sparse expert and backend residuals."""

    import torch
    import torch.nn.functional as functional

    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    _logits, route_weights, route_indices = block.gate(flat)
    decoded_total = torch.zeros_like(flat)
    source_recomputed = torch.zeros_like(flat)
    sparse: list[tuple[Any, Any]] = []
    for expert in range(int(block.experts.num_experts)):
        positions = torch.nonzero(route_indices == expert, as_tuple=False)
        if not positions.numel():
            sparse.append((torch.empty(0, dtype=torch.long, device=flat.device), flat.new_empty((0, flat.shape[-1]))))
            continue
        token_index = positions[:, 0]
        topk_index = positions[:, 1]
        values = flat[token_index]
        source_gate_up = functional.linear(values, block.experts.gate_up_proj[expert])
        source_gate, source_up = source_gate_up.chunk(2, dim=-1)
        source_value = functional.linear(block.experts.act_fn(source_gate) * source_up, block.experts.down_proj[expert])
        candidate_gate = functional.linear(values, decoded["gate_proj"][expert])
        candidate_up = functional.linear(values, decoded["up_proj"][expert])
        candidate_value = functional.linear(
            block.experts.act_fn(candidate_gate) * candidate_up,
            decoded["down_proj"][expert],
        )
        weight = route_weights[token_index, topk_index, None]
        source_value = source_value * weight
        candidate_value = candidate_value * weight
        source_recomputed.index_add_(0, token_index, source_value.to(source_recomputed.dtype))
        decoded_total.index_add_(0, token_index, candidate_value.to(decoded_total.dtype))
        sparse.append((token_index, (candidate_value - source_value).to(flat.dtype)))
    expert_sum = decoded_total - source_recomputed
    return decoded_total.reshape(original_shape), expert_sum.reshape(original_shape), sparse


def measure_native_causal_attribution(
    *,
    model: Any,
    token_ids: Sequence[int],
    decoded_by_layer: Mapping[int, Mapping[str, Any]],
    teacher_logits: Any | None = None,
    path_nodes: int = 5,
    fisher_rank: int = 8,
    seed: int = 20260823,
) -> dict[str, np.ndarray]:
    """Measure full-model KL path gradients and routed Fisher projections."""

    import torch

    if path_nodes < 2 or fisher_rank < 1:
        raise ValueError("native attribution requires at least two path nodes and one Fisher probe")
    if len(token_ids) < 2:
        raise ValueError("native attribution requires at least two tokens")
    blocks = qwen_moe_layers(model)
    layers = tuple(sorted(int(layer) for layer in decoded_by_layer))
    if not layers or any(layer not in blocks for layer in layers):
        raise ValueError("decoded provisional winners do not name valid Qwen MoE layers")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prepared = {}
    for layer in layers:
        device = blocks[layer].experts.gate_up_proj.device
        dtype = blocks[layer].experts.gate_up_proj.dtype
        prepared[layer] = {name: value.to(device=device, dtype=dtype) for name, value in decoded_by_layer[layer].items()}
    input_device = model.get_input_embeddings().weight.device
    ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=input_device)
    if teacher_logits is None:
        with torch.inference_mode():
            teacher = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1].double()
    else:
        teacher = torch.as_tensor(teacher_logits, device=input_device).double()
        if teacher.ndim == 2:
            teacher = teacher.unsqueeze(0)
        expected_shape = (1, len(token_ids) - 1)
        if teacher.shape[:2] != expected_shape:
            raise ValueError("sealed teacher logits do not align with attribution token positions")
    teacher_log = torch.log_softmax(teacher, dim=-1).detach()
    teacher_probability = teacher_log.exp()
    nodes, path_quadrature_weights = np.polynomial.legendre.leggauss(path_nodes)
    nodes = (nodes + 1.0) / 2.0
    path_quadrature_weights = path_quadrature_weights / 2.0
    path_gradients = np.zeros((path_nodes, len(layers), 1), dtype=np.float64)
    projected = np.zeros((len(layers), int(model.config.num_experts), path_nodes, fisher_rank), dtype=np.float64)
    projected_routing = np.zeros((len(layers), path_nodes, fisher_rank), dtype=np.float64)
    node_kld = np.zeros(path_nodes, dtype=np.float64)

    def run(alpha_value: float, node_index: int | None, capture_projections: bool):
        alphas = [torch.tensor(float(alpha_value), device=blocks[layer].experts.gate_up_proj.device, requires_grad=True) for layer in layers]
        captures: dict[int, dict[str, Any]] = {}
        handles = []
        for position, layer in enumerate(layers):
            def hook(module, args, output, *, _layer=layer, _position=position):
                decoded_output, expert_sum, sparse = _decoded_expert_block(module, args[0], prepared[_layer])
                actual_delta = decoded_output - output
                blended = output + alphas[_position].to(output.device) * actual_delta
                captures[_layer] = {
                    "output": blended,
                    "expert_sum": expert_sum.detach(),
                    "routing_residual": (actual_delta - expert_sum).detach(),
                    "sparse": [(index.detach(), value.detach()) for index, value in sparse],
                }
                return blended
            handles.append(blocks[layer].register_forward_hook(hook))
        try:
            result = model(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if set(captures) != set(layers):
            raise RuntimeError("not every provisional Qwen layer executed during attribution")
        # The measured KL is accumulated in float64.  At the small deltas used
        # for path quadrature, a float32 softmax subtraction can become
        # spuriously negative and would corrupt the sign gate.
        logits = result.logits[:, :-1].double()
        log_probability = torch.log_softmax(logits, dim=-1)
        loss = torch.mean(torch.sum(teacher_probability * (teacher_log - log_probability), dim=-1))
        if node_index is not None:
            alpha_gradients = torch.autograd.grad(loss, tuple(alphas), retain_graph=capture_projections)
            for position, gradient in enumerate(alpha_gradients):
                path_gradients[node_index, position, 0] = float(gradient.detach().cpu())
            node_kld[node_index] = float(loss.detach().cpu())
        if capture_projections:
            outputs = tuple(captures[layer]["output"] for layer in layers)
            probabilities = torch.softmax(logits.detach(), dim=-1)
            generator = torch.Generator(device=logits.device).manual_seed(seed + int(node_index or 0))
            samples = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]),
                fisher_rank,
                replacement=True,
                generator=generator,
            )
            flat_log = log_probability.reshape(-1, log_probability.shape[-1])
            normalization = flat_log.shape[0] ** 0.5
            for probe in range(fisher_rank):
                score = flat_log.gather(1, samples[:, probe:probe + 1]).sum() / normalization
                gradients = torch.autograd.grad(score, outputs, retain_graph=probe + 1 < fisher_rank)
                for position, (layer, gradient) in enumerate(zip(layers, gradients, strict=True)):
                    scored = gradient[:, :-1].reshape(-1, gradient.shape[-1])
                    for expert, (indices, residual) in enumerate(captures[layer]["sparse"]):
                        keep = indices < scored.shape[0]
                        if bool(keep.any()):
                            projected[position, expert, int(node_index), probe] = float(
                                torch.sum(scored[indices[keep]] * residual[keep]).detach().cpu()
                            )
                    routing = captures[layer]["routing_residual"][:, :-1].reshape_as(scored)
                    projected_routing[position, int(node_index), probe] = float(
                        torch.sum(scored * routing).detach().cpu()
                    )
        return float(loss.detach().cpu())

    source_kld = run(0.0, None, False)
    if abs(source_kld) > 1e-12:
        raise RuntimeError(
            "alpha-zero source control does not reproduce zero teacher KL: "
            f"observed={source_kld:.17g}"
        )
    for node_index, node in enumerate(nodes):
        run(float(node), node_index, True)
    candidate_kld = run(1.0, None, False)
    if candidate_kld < -1e-12:
        raise RuntimeError("alpha-one decoded candidate produced a materially negative KL")
    layer_deltas = np.ones((len(layers), 1), dtype=np.float64)

    def gradient_at(value: float):
        matches = np.flatnonzero(np.isclose(nodes, value, rtol=0.0, atol=1e-14))
        if len(matches) != 1:
            raise ValueError("requested attribution node is absent")
        return list(path_gradients[int(matches[0])])

    layer_damage = aumann_shapley(list(layer_deltas), gradient_at, path_nodes=path_nodes)
    values = {
        "layer_indices": np.asarray(layers, dtype=np.int32),
        "layer_deltas": layer_deltas,
        "path_nodes": np.asarray(nodes, dtype=np.float64),
        "path_quadrature_weights": np.asarray(path_quadrature_weights, dtype=np.float64),
        "path_gradients": path_gradients,
        "node_kld": node_kld,
        "projected_expert_residuals": projected,
        "projected_routing_residuals": projected_routing,
        "measured_layer_damage": np.asarray(layer_damage, dtype=np.float64),
        "source_kld": np.asarray([source_kld], dtype=np.float64),
        "candidate_kld": np.asarray([candidate_kld], dtype=np.float64),
        "measured_end_to_end_delta": np.asarray([candidate_kld - source_kld], dtype=np.float64),
    }
    if any(not np.isfinite(value).all() for value in values.values()):
        raise RuntimeError("native causal attribution produced non-finite evidence")
    return values


def build_hierarchical_attribution_document(
    arrays: Mapping[str, np.ndarray],
    candidate_inventory_sha256: str,
) -> dict[str, Any]:
    """Close raw path/Fisher evidence hierarchically to measured endpoint KL."""
    inventory_sha256 = _require_hash(candidate_inventory_sha256, "candidate inventory")
    measured = float(arrays["measured_end_to_end_delta"][0])
    raw_layers = np.asarray(arrays["measured_layer_damage"], dtype=np.float64)
    layer_accounting = reconcile_signed_completeness(raw_layers, measured)
    path_weights = np.asarray(arrays["path_quadrature_weights"], dtype=np.float64)
    layers = []
    for index, layer in enumerate(arrays["layer_indices"]):
        split = split_layer_damage(
            float(raw_layers[index]),
            arrays["projected_expert_residuals"][index],
            projected_routing_residual=arrays["projected_routing_residuals"][index],
            observation_weights=path_weights,
        )
        raw_layer = float(raw_layers[index])
        reconciled_layer = float(layer_accounting.reconciled[index])
        components = reconcile_layer_components_for_allocation(
            expert_direct=split["expert_direct"],
            routing_state_shift=split["routing_state_shift"],
            explicit_residual=split["unresolved_nonlinear_remainder"],
            raw_layer_damage=raw_layer,
            reconciled_layer_damage=reconciled_layer,
        )
        layers.append({
            "layer_index": int(layer),
            "aumann_shapley_raw": raw_layer,
            "reconciled_layer_damage": reconciled_layer,
            "expert_direct_raw": split["expert_direct"],
            "routing_state_shift_raw": split["routing_state_shift"],
            "within_layer_unresolved_remainder_raw": split["unresolved_nonlinear_remainder"],
            "raw_joint_fisher_total": split["raw_total"],
            **components,
        })
    reconciled_layer_total = float(sum(row["reconciled_layer_damage"] for row in layers))
    reconciled_expert_total = float(sum(row["reconciled_expert_total"] for row in layers))
    body = {
        "schema": "quant-pipeline.qwen-hf-mcg-attribution.v4",
        "candidate_inventory_sha256": inventory_sha256,
        "path_nodes": int(len(arrays["path_nodes"])),
        "fisher_rank": int(arrays["projected_expert_residuals"].shape[-1]),
        "source_kld": float(arrays["source_kld"][0]),
        "candidate_kld": float(arrays["candidate_kld"][0]),
        "measured_end_to_end_delta": measured,
        "sum_raw_aumann_shapley_layer_damage": layer_accounting.raw_total,
        "raw_path_quadrature_and_nonlinear_remainder": layer_accounting.closure_residual,
        "global_layer_completeness_scale": measured / layer_accounting.raw_total,
        "sum_reconciled_layer_damage": reconciled_layer_total,
        "sum_reconciled_expert_damage": reconciled_expert_total,
        "closure_policy": "signed-layer-completeness-plus-explicit-within-layer-components-v2",
        "expert_score_policy": "direct-evidence-plus-explicit-nonexpert-redistribution-for-allocation-only-v1",
        "raw_evidence_is_preserved": True,
        "layers": layers,
    }
    tolerance = max(1e-15, abs(measured) * 1e-12)
    if abs(reconciled_layer_total - measured) > tolerance or abs(reconciled_expert_total - measured) > tolerance:
        raise RuntimeError("hierarchical attribution did not close to measured endpoint KLD")
    body["attribution_sha256"] = _hash_json(body)
    return body


def write_attribution_inputs(
    output_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    provenance: Mapping[str, Any],
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".npz":
        raise ValueError("attribution inputs must use a .npz archive")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    os.replace(temporary, path)
    provenance_body = dict(provenance)
    receipt = {
        "schema": ATTRIBUTION_INPUT_SCHEMA,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_bytes(np.ascontiguousarray(value).view(np.uint8).tobytes()),
            }
            for key, value in sorted(arrays.items())
        },
        "provenance": provenance_body,
        "provenance_sha256": _hash_json(provenance_body),
    }
    receipt["receipt_sha256"] = _hash_json(receipt)
    write_json(path.with_suffix(path.suffix + ".receipt.json"), receipt)
    verify_attribution_inputs(path)
    return path


def verify_attribution_inputs(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = Path(path).resolve()
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    if not target.is_file() or target.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("native attribution archive or receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    schema = receipt.get("schema")
    if schema not in {ATTRIBUTION_INPUT_SCHEMA, LEGACY_ATTRIBUTION_INPUT_SCHEMA}:
        raise ValueError("unsupported native attribution input receipt")
    seal = _require_hash(receipt.get("receipt_sha256"), "native attribution receipt")
    if _hash_json({key: value for key, value in receipt.items() if key != "receipt_sha256"}) != seal:
        raise ValueError("native attribution receipt seal mismatch")
    if _hash_json(receipt.get("provenance", {})) != _require_hash(
        receipt.get("provenance_sha256"), "native attribution provenance"
    ):
        raise ValueError("native attribution provenance identity mismatch")
    if target.stat().st_size != int(receipt["bytes"]) or sha256_file(target) != receipt["sha256"]:
        raise ValueError("native attribution archive byte identity mismatch")
    with np.load(target, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if set(arrays) != set(receipt["arrays"]):
        raise ValueError("native attribution archive inventory mismatch")
    for key, value in arrays.items():
        row = receipt["arrays"][key]
        if list(value.shape) != row["shape"] or str(value.dtype) != row["dtype"]:
            raise ValueError("native attribution array shape or dtype mismatch")
        if sha256_bytes(np.ascontiguousarray(value).view(np.uint8).tobytes()) != row["sha256"]:
            raise ValueError("native attribution array identity mismatch")
        if not np.isfinite(value).all():
            raise ValueError("native attribution archive contains non-finite values")
    if schema == LEGACY_ATTRIBUTION_INPUT_SCHEMA and "path_quadrature_weights" not in arrays:
        _legacy_nodes, legacy_weights = np.polynomial.legendre.leggauss(len(arrays["path_nodes"]))
        arrays["path_quadrature_weights"] = np.asarray(legacy_weights / 2.0, dtype=np.float64)
    required = {
        "layer_indices", "layer_deltas", "path_nodes", "path_quadrature_weights", "path_gradients", "node_kld",
        "projected_expert_residuals", "projected_routing_residuals", "measured_layer_damage",
        "source_kld", "candidate_kld", "measured_end_to_end_delta",
    }
    if set(arrays) != required:
        raise ValueError("native attribution archive lacks the exact required arrays")
    layers = len(arrays["layer_indices"])
    nodes = len(arrays["path_nodes"])
    layer_indices = arrays["layer_indices"]
    if layer_indices.dtype.kind not in "iu" or layers < 1 or np.any(layer_indices < 0) or np.any(np.diff(layer_indices) <= 0):
        raise ValueError("native attribution layer indices must be unique increasing integers")
    expected_nodes, expected_weights = np.polynomial.legendre.leggauss(nodes)
    expected_nodes = (expected_nodes + 1.0) / 2.0
    expected_weights = expected_weights / 2.0
    if nodes < 2 or not np.array_equal(arrays["path_nodes"], expected_nodes):
        raise ValueError("native attribution path nodes are not canonical Gauss-Legendre order")
    if arrays["path_quadrature_weights"].shape != (nodes,) or not np.array_equal(
        arrays["path_quadrature_weights"], expected_weights
    ):
        raise ValueError("native attribution quadrature weights are not canonical Gauss-Legendre weights")
    if arrays["layer_deltas"].shape != (layers, 1) or arrays["path_gradients"].shape != (nodes, layers, 1):
        raise ValueError("native attribution layer/path geometry mismatch")
    if not np.array_equal(arrays["layer_deltas"], np.ones((layers, 1), dtype=np.float64)):
        raise ValueError("native attribution blend deltas must be the unit source-to-decoded path")
    expert_shape = arrays["projected_expert_residuals"].shape
    if len(expert_shape) != 4 or expert_shape[0] != layers or expert_shape[1] < 1 or expert_shape[2] != nodes or expert_shape[3] < 1:
        raise ValueError("native attribution expert projection geometry mismatch")
    if arrays["projected_routing_residuals"].shape != (layers, nodes, expert_shape[3]):
        raise ValueError("native attribution routing projection geometry mismatch")
    if arrays["node_kld"].shape != (nodes,) or arrays["measured_layer_damage"].shape != (layers,):
        raise ValueError("native attribution KLD/layer damage geometry mismatch")
    if any(arrays[name].shape != (1,) for name in ("source_kld", "candidate_kld", "measured_end_to_end_delta")):
        raise ValueError("native attribution endpoint arrays must contain one scalar")
    source = float(arrays["source_kld"][0])
    candidate = float(arrays["candidate_kld"][0])
    end_to_end = float(arrays["measured_end_to_end_delta"][0])
    if abs(source) > 1e-12 or candidate < -1e-12 or not math.isclose(candidate - source, end_to_end, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("native attribution endpoint source/candidate closure failed")

    def gradient_at(value: float):
        matches = np.flatnonzero(np.isclose(expected_nodes, value, rtol=0.0, atol=1e-14))
        if len(matches) != 1:
            raise ValueError("native attribution requested a noncanonical node")
        return list(arrays["path_gradients"][int(matches[0])])

    recomputed = aumann_shapley(list(arrays["layer_deltas"]), gradient_at, path_nodes=nodes)
    if not np.allclose(recomputed, arrays["measured_layer_damage"], rtol=1e-10, atol=1e-12):
        raise ValueError("native attribution stored layer integral does not match path gradients")
    provenance = receipt["provenance"]
    if provenance.get("test_only") is not True:
        implementation = provenance.get("implementation")
        if implementation == "native-qwen-hf-uniform-k4-mcg-blend-fisher-v2":
            required_provenance = {
                "implementation", "model_revision", "kld_window_seal_sha256",
                "teacher_reference_sha256", "candidate_inventory_sha256",
                "candidate_dataset_repo", "candidate_dataset_revision",
                "provisional_bit_triplet", "path_nodes", "fisher_rank", "seed",
                "test_only",
            }
            if provenance.get("provisional_bit_triplet") != [4, 4, 4]:
                raise ValueError("HF-native attribution must use the sealed uniform-K4 anchor")
            if not isinstance(provenance.get("candidate_dataset_repo"), str) or not provenance["candidate_dataset_repo"]:
                raise ValueError("HF-native attribution dataset repository is missing")
            if not isinstance(provenance.get("candidate_dataset_revision"), str) or _REVISION.fullmatch(provenance["candidate_dataset_revision"]) is None:
                raise ValueError("HF-native attribution dataset revision is not immutable")
            provenance_hash_keys = (
                "kld_window_seal_sha256", "teacher_reference_sha256",
                "candidate_inventory_sha256",
            )
        else:
            required_provenance = {
                "implementation", "model_revision", "kld_window_seal_sha256",
                "teacher_reference_sha256", "provisional_manifest_sha256", "candidate_ledger_sha256",
                "path_nodes", "fisher_rank", "seed", "test_only",
            }
            provenance_hash_keys = (
                "kld_window_seal_sha256", "teacher_reference_sha256",
                "provisional_manifest_sha256", "candidate_ledger_sha256",
            )
        if set(provenance) != required_provenance:
            raise ValueError("production native attribution provenance is incomplete")
        if not isinstance(provenance["model_revision"], str) or _REVISION.fullmatch(provenance["model_revision"]) is None:
            raise ValueError("production native attribution model revision is not immutable")
        for key in provenance_hash_keys:
            _require_hash(provenance[key], f"native attribution provenance {key}")
        if int(provenance["path_nodes"]) != nodes or int(provenance["fisher_rank"]) != expert_shape[3]:
            raise ValueError("native attribution provenance geometry differs from arrays")
    return arrays, receipt


def produce_qwen_attribution_inputs_from_local(
    *,
    source_checkpoint: str | Path,
    model_revision: str,
    kld_window: str | Path,
    teacher_reference: str | Path,
    provisional_manifest: str | Path,
    payload_store_root: str | Path,
    output_path: str | Path,
    device_map: Any = "auto",
    attn_implementation: str = "eager",
    path_nodes: int = 5,
    fisher_rank: int = 8,
    seed: int = 20260823,
) -> Path:
    """Load pinned local Qwen bytes and emit sealed measured attribution input."""

    import torch
    from transformers import AutoModelForCausalLM

    if not isinstance(model_revision, str) or _REVISION.fullmatch(model_revision) is None:
        raise ValueError("native attribution model revision must be immutable 40-hex")
    window_root = Path(kld_window).resolve()
    window = json.loads((window_root / "kld-window.json").read_text())
    verify_kld_window(window, window_root)
    manifest = verify_provisional_winner_deltas(
        provisional_manifest,
        payload_store_root=payload_store_root,
    )
    model = AutoModelForCausalLM.from_pretrained(
        Path(source_checkpoint).resolve(),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=attn_implementation,
    ).eval()
    decoded = load_provisional_decoded_weights(
        model,
        provisional_manifest,
        payload_store_root=payload_store_root,
    )
    teacher_path = Path(teacher_reference).resolve()
    if not teacher_path.is_file() or teacher_path.is_symlink():
        raise ValueError("sealed teacher reference is missing")
    teacher = np.load(teacher_path, allow_pickle=False)
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([window["token_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode():
        observed_teacher = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1].float().cpu().numpy()
    expected_teacher = np.asarray(teacher).reshape(observed_teacher.shape)
    if not np.array_equal(observed_teacher, expected_teacher):
        raise RuntimeError("sealed teacher logits differ from pinned source model and KLD window")
    arrays = measure_native_causal_attribution(
        model=model,
        token_ids=window["token_ids"],
        decoded_by_layer=decoded,
        teacher_logits=teacher,
        path_nodes=path_nodes,
        fisher_rank=fisher_rank,
        seed=seed,
    )
    return write_attribution_inputs(
        output_path,
        arrays,
        provenance={
            "implementation": "native-qwen-decoded-output-blend-fisher-vjp-v1",
            "model_revision": model_revision,
            "kld_window_seal_sha256": window["seal_sha256"],
            "teacher_reference_sha256": sha256_file(teacher_path),
            "provisional_manifest_sha256": manifest["manifest_sha256"],
            "candidate_ledger_sha256": manifest["candidate_ledger_sha256"],
            "path_nodes": int(path_nodes),
            "fisher_rank": int(fisher_rank),
            "seed": int(seed),
            "test_only": False,
        },
    )
