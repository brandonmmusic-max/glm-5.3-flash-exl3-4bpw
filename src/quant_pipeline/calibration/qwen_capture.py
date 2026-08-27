"""Exact routed-activation capture for Qwen3 MoE checkpoints.

This module deliberately keeps model loading outside the capture arithmetic so
the same path can be tested with a tiny Qwen-shaped module.  Production loading
is local-only and starts from the official BF16 checkpoint; installed causal
layers are replayed only after their manifests and reconstructed tensor bytes
have been verified.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json


CAPTURE_SCHEMA = "quant-pipeline.qwen-routed-capture.v2"
CHUNK_SCHEMA = "quant-pipeline.qwen-routed-chunk.v2"
CHUNK_RECEIPT_SCHEMA = "quant-pipeline.qwen-routed-chunk-receipt.v1"
QWEN_PRODUCTION_GEOMETRY = {
    "layers": 48,
    "experts": 128,
    "top_k": 8,
    "hidden_size": 2048,
    "intermediate_size": 768,
}
_HASH = re.compile(r"[0-9a-f]{64}")


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_save_tensors(path: Path, tensors: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except Exception as error:  # pragma: no cover - optional dependency
        raise RuntimeError("safetensors and torch are required for routed capture") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    # Route arrays often have view aliases (for example flattened assignment
    # weights).  Persist independent contiguous values so safetensors cannot
    # encode accidental storage aliasing as part of the format contract.
    save_file(
        {name: value.detach().contiguous().clone() for name, value in tensors.items()},
        temporary,
        metadata={"qp": encoded},
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _read_tensor_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from safetensors import safe_open
    except Exception as error:  # pragma: no cover
        raise RuntimeError("safetensors is required for routed capture") from error
    with safe_open(path, framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get("qp")
        metadata = json.loads(raw) if raw else {}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return metadata, tensors


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _chunk_receipt(path: Path, *, metadata: Mapping[str, Any], tensors: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": CHUNK_RECEIPT_SCHEMA,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "metadata": dict(metadata),
        "tensors": {name: _tensor_record(value) for name, value in sorted(tensors.items())},
    }
    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    return body


def _receipt_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".receipt.json")


def _persist_capture_chunk(
    path: Path,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    window: "CaptureWindow",
    window_index: int,
) -> dict[str, Any]:
    """Seal one CPU-resident chunk outside the GPU producer's hot path."""

    _atomic_save_tensors(path, tensors, metadata)
    receipt = _chunk_receipt(path, metadata=metadata, tensors=tensors)
    write_json(_receipt_path(path), receipt)
    # Full immediate readback remains mandatory, but callers may run it in a
    # background writer pool while the next model forward is executing.
    verify_capture_chunk(path, metadata)
    return {
        "file": path.relative_to(path.parents[1]).as_posix(),
        "receipt": _receipt_path(path).relative_to(path.parents[1]).as_posix(),
        "bytes": receipt["bytes"],
        "sha256": receipt["sha256"],
        "window_index": window_index,
        "token_sha256": window.token_sha256,
    }


def _verify_existing_capture_chunk(
    path: Path,
    metadata: Mapping[str, Any],
    window: "CaptureWindow",
    window_index: int,
) -> dict[str, Any]:
    """Audit a resumable chunk in the same bounded I/O pool as new writes."""

    receipt = verify_capture_chunk(path, metadata)
    return {
        "file": path.relative_to(path.parents[1]).as_posix(),
        "receipt": _receipt_path(path).relative_to(path.parents[1]).as_posix(),
        "bytes": receipt["bytes"],
        "sha256": receipt["sha256"],
        "window_index": window_index,
        "token_sha256": window.token_sha256,
    }


def _cleanup_fisher_scratch_after(futures: Sequence[Future[dict[str, Any]]], scratch: Path) -> None:
    """Keep file-backed Fisher tensors alive until every consumer seals."""

    for future in futures:
        future.result()
    for temporary in scratch.iterdir():
        temporary.unlink()
    scratch.rmdir()


def _capture_request_identity(
    *,
    role: str,
    layers: Sequence[int],
    predecessor_state_hash: str,
    windows: Sequence["CaptureWindow"],
    geometry: Mapping[str, Any],
    norm_topk_prob: bool,
    fisher_rank: int,
    seed: int,
    model_revision: str | None,
    replay_report: Mapping[str, Any] | None,
    capture_checkpoint_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(role, str) or not role:
        raise ValueError("capture role must be a non-empty string")
    if model_revision is not None and re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("capture model_revision must be immutable 40-hex")
    replay = None
    if replay_report is not None:
        replay = dict(replay_report)
        replay_seal = _require_hash(replay.get("replay_sha256"), "installed replay")
        replay_body = {key: value for key, value in replay.items() if key != "replay_sha256"}
        if replay.get("schema") != "quant-pipeline.qwen-installed-replay.v1" or sha256_bytes(canonical_json(replay_body)) != replay_seal:
            raise ValueError("installed replay report is malformed or unsealed")
        if replay.get("requested_predecessor_state_hash") != predecessor_state_hash:
            raise ValueError("installed replay targets a different predecessor state")
    capture_checkpoint = None
    if capture_checkpoint_identity is not None:
        capture_checkpoint = dict(capture_checkpoint_identity)
        capture_seal = _require_hash(
            capture_checkpoint.get("capture_checkpoint_sha256"),
            "capture checkpoint identity",
        )
        capture_body = {
            key: value
            for key, value in capture_checkpoint.items()
            if key != "capture_checkpoint_sha256"
        }
        if (
            capture_checkpoint.get("schema")
            != "quant-pipeline.qwen-capture-checkpoint-identity.v1"
            or sha256_bytes(canonical_json(capture_body)) != capture_seal
        ):
            raise ValueError("capture checkpoint identity is malformed or unsealed")
        if capture_checkpoint.get("source_revision") != model_revision:
            raise ValueError("capture checkpoint derives from a different source revision")
    body = {
        "schema": "quant-pipeline.qwen-routed-capture-request.v1",
        "role": role,
        "layers": [int(layer) for layer in layers],
        "predecessor_state_hash": _require_hash(predecessor_state_hash, "predecessor state"),
        "geometry": dict(geometry),
        "norm_topk_prob": bool(norm_topk_prob),
        "fisher_rank": int(fisher_rank),
        "seed": int(seed),
        "model_revision": model_revision,
        "installed_replay": replay,
        "capture_checkpoint": capture_checkpoint,
        "windows": [
            {
                "window_index": index,
                "token_sha256": str(window.token_sha256),
                "token_ids_sha256": sha256_bytes(canonical_json([int(token) for token in window.token_ids])),
                "token_count": len(window.token_ids),
                "document_id": str(window.document_id),
                "start_token": int(window.start_token),
            }
            for index, window in enumerate(windows)
        ],
    }
    body["request_sha256"] = sha256_bytes(canonical_json(body))
    return body


def _quarantine_incomplete_chunk(path: Path, root: Path) -> dict[str, Any]:
    """Move an orphaned chunk/receipt aside before deterministic regeneration."""

    receipt = _receipt_path(path)
    existing = [candidate for candidate in (path, receipt) if candidate.exists()]
    if not existing:
        raise FileNotFoundError("no incomplete capture artifacts to quarantine")
    if any(not candidate.is_file() or candidate.is_symlink() for candidate in existing):
        raise ValueError(f"incomplete capture artifact is not a regular file: {path}")
    entries = [
        {"name": candidate.name, "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
        for candidate in existing
    ]
    identity = sha256_bytes(canonical_json(entries))
    quarantine = root / ".quarantine" / f"{path.parent.name}-{path.stem}-{identity}"
    quarantine.mkdir(parents=True, exist_ok=False)
    for candidate in existing:
        os.replace(candidate, quarantine / candidate.name)
    write_json(
        quarantine / "quarantine.json",
        {
            "schema": "quant-pipeline.qwen-capture-quarantine.v1",
            "reason": "chunk-receipt-pair-incomplete",
            "original": path.relative_to(root).as_posix(),
            "entries": entries,
            "identity_sha256": identity,
        },
    )
    for directory in (quarantine, quarantine.parent, path.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"path": quarantine.relative_to(root).as_posix(), "identity_sha256": identity}


def verify_capture_chunk(path: str | Path, expected_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Verify the complete chunk bytes, its tensor bytes, schema and receipt.

    Resume never trusts metadata alone.  A byte change in the safetensors
    header or payload, a changed tensor dtype/shape, or a receipt edit fails.
    """

    target = Path(path)
    receipt_path = _receipt_path(target)
    if not target.is_file() or target.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError(f"capture chunk or receipt missing: {target}")
    receipt = json.loads(receipt_path.read_text())
    expected_seal = _require_hash(receipt.get("receipt_sha256"), "chunk receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if sha256_bytes(canonical_json(body)) != expected_seal:
        raise ValueError(f"capture chunk receipt seal mismatch: {target}")
    if receipt.get("schema") != CHUNK_RECEIPT_SCHEMA:
        raise ValueError(f"unsupported capture chunk receipt: {target}")
    if receipt.get("file") != target.name or receipt.get("bytes") != target.stat().st_size or receipt.get("sha256") != sha256_file(target):
        raise ValueError(f"capture chunk byte identity mismatch: {target}")
    metadata, tensors = _read_tensor_file(target)
    if metadata != receipt.get("metadata"):
        raise ValueError(f"capture chunk metadata mismatch: {target}")
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise ValueError(f"capture chunk belongs to a different request: {target}")
    observed = {name: _tensor_record(value) for name, value in sorted(tensors.items())}
    if observed != receipt.get("tensors"):
        raise ValueError(f"capture chunk tensor identity mismatch: {target}")
    required = {
        "hidden_states",
        "router_logits",
        "expert_ids",
        "router_weights",
        "routed_hidden_states",
        "routed_down_inputs",
        "assignment_expert_ids",
        "assignment_token_offsets",
        "assignment_router_weights",
    }
    if not required <= set(tensors):
        raise ValueError(f"capture chunk lacks required tensors: {sorted(required - set(tensors))}")
    import torch

    hidden = tensors["hidden_states"]
    routes = tensors["expert_ids"]
    weights = tensors["router_weights"]
    logits = tensors["router_logits"]
    routed_hidden = tensors["routed_hidden_states"]
    down = tensors["routed_down_inputs"]
    assignment_expert = tensors["assignment_expert_ids"]
    assignment_offset = tensors["assignment_token_offsets"]
    assignment_weight = tensors["assignment_router_weights"]
    if hidden.ndim != 2 or routes.ndim != 2 or weights.shape != routes.shape or logits.ndim != 2:
        raise ValueError(f"capture chunk base tensor shapes are invalid: {target}")
    if hidden.shape[0] != routes.shape[0] or logits.shape[0] != hidden.shape[0]:
        raise ValueError(f"capture chunk token domains disagree: {target}")
    assignments = assignment_expert.numel()
    if (
        routed_hidden.ndim != 2
        or down.ndim != 2
        or routed_hidden.shape[0] != assignments
        or down.shape[0] != assignments
        or assignment_offset.numel() != assignments
        or assignment_weight.numel() != assignments
    ):
        raise ValueError(f"capture chunk routed assignment shapes are invalid: {target}")
    # Routed residual/Fisher scoring uses exactly next-token rows 0..T-2.
    # The last input token has no next-token logit in the sealed KLD control.
    if assignments and (int(assignment_offset.min()) < 0 or int(assignment_offset.max()) >= hidden.shape[0] - 1):
        raise ValueError(f"capture chunk contains an unscored final-token assignment: {target}")
    floating = [hidden, weights, logits, routed_hidden, down, assignment_weight]
    if "fisher_gradients" in tensors:
        fisher = tensors["fisher_gradients"]
        if fisher.ndim != 4 or fisher.shape[-2] != hidden.shape[0] - 1 or fisher.shape[-1] != hidden.shape[1]:
            raise ValueError(f"capture chunk Fisher tensor is not aligned to T-1 rows: {target}")
        floating.append(fisher)
    if any(not torch.isfinite(value.float()).all() for value in floating):
        raise ValueError(f"capture chunk contains non-finite numeric data: {target}")
    return receipt


def recompute_routing(gate_weight: Any, hidden_flat: Any, top_k: int, norm_topk_prob: bool):
    """Mirror Qwen3MoeTopKRouter's linear/FP32-softmax/top-k arithmetic."""

    import torch
    import torch.nn.functional as functional

    logits = functional.linear(hidden_flat, gate_weight)
    probabilities = torch.softmax(logits, dtype=torch.float32, dim=-1)
    weights, indices = torch.topk(probabilities, int(top_k), dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return logits, weights.to(logits.dtype), indices


def _split_gate_up(experts: Any, expert: int) -> tuple[Any, Any]:
    weight = experts.gate_up_proj[expert]
    if weight.shape[0] % 2:
        raise ValueError("Qwen gate_up_proj has an odd stacked output dimension")
    return weight.chunk(2, dim=0)


def routed_expert_rows(
    hidden: Any,
    expert_ids: Any,
    router_weights: Any,
    experts: Any,
    *,
    next_token_only: bool = True,
) -> dict[str, Any]:
    """Expand token/top-k routes and compute exact post-SwiGLU down inputs."""

    import torch
    import torch.nn.functional as functional

    hidden = hidden.reshape(-1, hidden.shape[-1])
    expert_ids = expert_ids.reshape(-1, expert_ids.shape[-1])
    router_weights = router_weights.reshape(-1, router_weights.shape[-1])
    if next_token_only:
        # KLD/Fisher/student captures have T-1 scored positions.  Filtering
        # before route expansion makes it impossible for a distinct final
        # token to enter expert residual or Fisher attribution.
        hidden = hidden[:-1]
        expert_ids = expert_ids[:-1]
        router_weights = router_weights[:-1]
    token_index = torch.arange(hidden.shape[0], device=hidden.device).repeat_interleave(expert_ids.shape[1])
    flat_expert = expert_ids.reshape(-1)
    flat_weight = router_weights.reshape(-1)
    routed_hidden = hidden[token_index]
    intermediate = experts.down_proj.shape[-1]
    down = torch.empty((routed_hidden.shape[0], intermediate), dtype=hidden.dtype, device=hidden.device)
    for expert in torch.unique(flat_expert).tolist():
        selected = torch.nonzero(flat_expert == expert, as_tuple=False).flatten()
        gate, up = _split_gate_up(experts, int(expert))
        values = routed_hidden[selected]
        down[selected] = functional.silu(functional.linear(values, gate)) * functional.linear(values, up)
    return {
        "routed_hidden_states": routed_hidden,
        "routed_down_inputs": down,
        "assignment_expert_ids": flat_expert,
        "assignment_token_offsets": token_index,
        "assignment_router_weights": flat_weight,
    }


def qwen_moe_layers(model: Any) -> dict[int, Any]:
    layers: dict[int, Any] = {}
    for index, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            layers[index] = mlp
    if not layers:
        raise ValueError("model exposes no Qwen-shaped sparse MoE layers")
    return layers


def _capture_fisher_all(
    model: Any,
    blocks: Mapping[int, Any],
    input_ids: Any,
    rank: int,
    seed: int,
    scratch_dir: Path,
) -> dict[int, tuple[Path, tuple[int, ...]]]:
    """Capture all requested layers in one forward and ``rank`` backwards.

    Each probe calls ``autograd.grad`` once with the tuple of all MoE block
    outputs. Gradients are immediately copied to layer-local disk memmaps, so
    the 48-layer production case does not retain every Fisher tensor in RAM.
    """

    import torch

    if rank < 1:
        raise ValueError("Fisher rank must be positive")
    import numpy as np

    saved: dict[int, Any] = {}

    def embed_hook(_module, _args, output):
        return output.detach().clone().requires_grad_(True)

    embed_handle = model.get_input_embeddings().register_forward_hook(embed_hook)
    block_handles = []
    for layer, block in blocks.items():
        def block_hook(_module, _args, output, *, _layer=layer):
            saved[_layer] = output
            return output
        block_handles.append(block.register_forward_hook(block_hook))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    maps: dict[int, Any] = {}
    records: dict[int, tuple[Path, tuple[int, ...]]] = {}
    try:
        with torch.enable_grad():
            output = model(input_ids=input_ids, use_cache=False, return_dict=True)
            if set(saved) != set(blocks):
                raise RuntimeError("not every requested MoE block produced a Fisher graph output")
            logits = output.logits[:, :-1]
            probabilities = torch.softmax(logits.float(), dim=-1)
            generator = torch.Generator(device=logits.device).manual_seed(seed)
            samples = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]),
                rank,
                replacement=True,
                generator=generator,
            )
            flat_log_probs = torch.log_softmax(logits.float(), dim=-1).reshape(-1, logits.shape[-1])
            ordered_layers = tuple(sorted(saved))
            outputs = tuple(saved[layer] for layer in ordered_layers)
            for layer, value in zip(ordered_layers, outputs, strict=True):
                shape = (rank, value.shape[0], value.shape[1] - 1, value.shape[2])
                path = scratch_dir / f"layer-{layer:03d}.f16"
                maps[layer] = np.memmap(path, dtype=np.float16, mode="w+", shape=shape)
                records[layer] = (path, shape)
            for probe in range(rank):
                selected = flat_log_probs.gather(1, samples[:, probe : probe + 1]).sum() / (flat_log_probs.shape[0] ** 0.5)
                gradients = torch.autograd.grad(selected, outputs, retain_graph=probe + 1 < rank)
                for layer, gradient in zip(ordered_layers, gradients, strict=True):
                    maps[layer][probe] = gradient[:, :-1].detach().to(torch.float16).cpu().numpy()
            for layer, value in maps.items():
                value.flush()
                descriptor = os.open(records[layer][0], os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    finally:
        embed_handle.remove()
        for handle in block_handles:
            handle.remove()
    maps.clear()
    return records


def _load_fisher_memmap(record: tuple[Path, tuple[int, ...]]):
    import math
    import torch

    path, shape = record
    return torch.from_file(str(path), shared=False, size=math.prod(shape), dtype=torch.float16).reshape(shape)


@dataclass(frozen=True)
class CaptureWindow:
    token_ids: Sequence[int]
    token_sha256: str
    document_id: str
    start_token: int = 0


def capture_loaded_qwen(
    *,
    model: Any,
    windows: Iterable[CaptureWindow],
    role: str,
    layers: Sequence[int],
    predecessor_state_hash: str,
    output_dir: str | Path,
    fisher_rank: int = 0,
    seed: int = 20260823,
    production_geometry: bool = False,
    model_revision: str | None = None,
    replay_report: Mapping[str, Any] | None = None,
    capture_checkpoint_identity: Mapping[str, Any] | None = None,
    writer_workers: int = 1,
    max_inflight_chunks: int | None = None,
) -> dict[str, Any]:
    """Capture a loaded Qwen model with exact routing and crash-safe resume."""

    import torch

    predecessor_state_hash = _require_hash(predecessor_state_hash, "predecessor state")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    blocks = qwen_moe_layers(model)
    target_layers = tuple(sorted(int(layer) for layer in layers))
    if not target_layers or any(layer not in blocks for layer in target_layers):
        raise ValueError("capture layers must be existing Qwen MoE layers")
    config = model.config
    geometry = {
        "layers": len(blocks),
        "experts": int(config.num_experts),
        "top_k": int(config.num_experts_per_tok),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.moe_intermediate_size),
    }
    if production_geometry and geometry != QWEN_PRODUCTION_GEOMETRY:
        raise ValueError(f"Qwen production geometry mismatch: {geometry}")
    norm_topk_prob = bool(config.norm_topk_prob)
    input_device = model.get_input_embeddings().weight.device
    windows = tuple(windows)
    request_identity = _capture_request_identity(
        role=role,
        layers=target_layers,
        predecessor_state_hash=predecessor_state_hash,
        windows=windows,
        geometry=geometry,
        norm_topk_prob=norm_topk_prob,
        fisher_rank=fisher_rank,
        seed=seed,
        model_revision=model_revision,
        replay_report=replay_report,
        capture_checkpoint_identity=capture_checkpoint_identity,
    )
    if (root / "capture-manifest.json").exists():
        return verify_capture_manifest(root, expected_request_sha256=request_identity["request_sha256"])
    captures: dict[int, dict[str, Any]] = {}
    handles = []

    def hook(layer: int):
        def _hook(module, args, output):
            hidden = args[0].detach().reshape(-1, args[0].shape[-1])
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise ValueError("Qwen router hook expected (logits, weights, indices)")
            captures[layer] = {
                "hidden": hidden,
                "logits": output[0].detach(),
                "weights": output[1].detach(),
                "indices": output[2].detach(),
                "gate_weight": module.weight.detach(),
            }
        return _hook

    for layer in target_layers:
        handles.append(blocks[layer].gate.register_forward_hook(hook(layer)))
    records: dict[str, list[dict[str, Any]]] = {str(layer): [] for layer in target_layers}
    writer_workers = int(writer_workers)
    if writer_workers < 1:
        raise ValueError("writer_workers must be positive")
    if max_inflight_chunks is None:
        max_inflight_chunks = max(writer_workers * 12, len(target_layers) * 2)
    max_inflight_chunks = int(max_inflight_chunks)
    if max_inflight_chunks < writer_workers:
        raise ValueError("max_inflight_chunks must be at least writer_workers")
    pending_writes: list[tuple[int, Future[dict[str, Any]]]] = []
    scratch_cleanups: list[Future[None]] = []

    def drain_writes(*, all_pending: bool = False) -> None:
        target = 0 if all_pending else max_inflight_chunks - 1
        while len(pending_writes) > target:
            layer, future = pending_writes.pop(0)
            records[str(layer)].append(future.result())

    try:
        with ThreadPoolExecutor(max_workers=writer_workers, thread_name_prefix="qwen-capture-writer") as writers:
            for window_index, window in enumerate(windows):
                # Bound host memory while allowing one or more prior windows to
                # serialize concurrently with this window's GPU work.
                drain_writes()
                input_ids = torch.tensor([list(window.token_ids)], dtype=torch.long, device=input_device)
                pending: list[tuple[int, Path, dict[str, Any]]] = []
                for layer in target_layers:
                    path = root / f"layer-{layer:03d}" / f"window-{window_index:04d}.safetensors"
                    metadata = {
                        "schema": CHUNK_SCHEMA,
                        "role": role,
                        "layer": layer,
                        "window_index": window_index,
                        "token_sha256": window.token_sha256,
                        "document_id": window.document_id,
                        "start_token": int(window.start_token),
                        "predecessor_state_hash": predecessor_state_hash,
                        "capture_request_sha256": request_identity["request_sha256"],
                    }
                    pair = (path.exists(), _receipt_path(path).exists())
                    if pair[0] != pair[1]:
                        _quarantine_incomplete_chunk(path, root)
                    elif pair == (True, True):
                        pending_writes.append((
                            layer,
                            writers.submit(_verify_existing_capture_chunk, path, metadata, window, window_index),
                        ))
                        drain_writes()
                        continue
                    pending.append((layer, path, metadata))
                if pending:
                    captures.clear()
                    with torch.inference_mode():
                        model(input_ids=input_ids, use_cache=False, return_dict=True)
                base_captures = {layer: dict(row) for layer, row in captures.items()}
                fisher_records: dict[int, tuple[Path, tuple[int, ...]]] = {}
                scratch = root / f".fisher-scratch-{window_index:04d}"
                if pending and fisher_rank:
                    if scratch.exists():
                        for stale in scratch.iterdir():
                            stale.unlink()
                    fisher_records = _capture_fisher_all(
                        model,
                        {layer: blocks[layer] for layer, _path, _metadata in pending},
                        input_ids,
                        fisher_rank,
                        seed + window_index,
                        scratch,
                    )
                window_writes: list[Future[dict[str, Any]]] = []
                for layer, path, metadata in pending:
                    if layer not in base_captures:
                        raise RuntimeError(f"router hook did not fire for layer {layer}")
                    row = base_captures[layer]
                    recomputed = recompute_routing(row["gate_weight"], row["hidden"], geometry["top_k"], norm_topk_prob)
                    if not torch.equal(recomputed[0], row["logits"]):
                        raise RuntimeError(f"router logits diverged from Qwen arithmetic at layer {layer}")
                    if not torch.equal(recomputed[1], row["weights"]) or not torch.equal(recomputed[2], row["indices"]):
                        raise RuntimeError(f"top-k routing diverged from Qwen arithmetic at layer {layer}")
                    routed = routed_expert_rows(row["hidden"], row["indices"], row["weights"], blocks[layer].experts)
                    tensors = {
                        "hidden_states": row["hidden"].to(torch.bfloat16).cpu(),
                        "router_logits": row["logits"].to(torch.float32).cpu(),
                        "expert_ids": row["indices"].to(torch.int32).cpu(),
                        "router_weights": row["weights"].to(torch.float32).cpu(),
                        **{
                            key: value.to(torch.float32 if "weight" in key else torch.int32 if "ids" in key or "offsets" in key else torch.bfloat16).cpu()
                            for key, value in routed.items()
                        },
                    }
                    if fisher_rank:
                        tensors["fisher_gradients"] = _load_fisher_memmap(fisher_records[layer])
                    future = writers.submit(_persist_capture_chunk, path, tensors, metadata, window, window_index)
                    pending_writes.append((layer, future))
                    window_writes.append(future)
                    drain_writes()
                if scratch.exists():
                    scratch_cleanups.append(writers.submit(_cleanup_fisher_scratch_after, tuple(window_writes), scratch))
            drain_writes(all_pending=True)
            for cleanup in scratch_cleanups:
                cleanup.result()
    finally:
        for handle in handles:
            handle.remove()
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "role": role,
        "predecessor_state_hash": predecessor_state_hash,
        "geometry": geometry,
        "norm_topk_prob": norm_topk_prob,
        "layers": list(target_layers),
        "fisher_rank": int(fisher_rank),
        "next_token_positions_per_window": [max(0, len(window.token_ids) - 1) for window in windows],
        "records": records,
        "routing_cross_check": "bit-exact",
        "request": request_identity,
        "request_sha256": request_identity["request_sha256"],
    }
    manifest["capture_sha256"] = sha256_bytes(canonical_json(manifest))
    write_json(root / "capture-manifest.json", manifest)
    # Every record admitted above came from a completed writer future, and
    # _persist_capture_chunk performs a full hash/tensor/numeric readback before
    # that future resolves.  Re-reading every new chunk again here is therefore
    # byte-neutral duplicate I/O.  Validate the newly assembled manifest and
    # record inventory now; the existing-manifest resume branch above retains
    # the full chunk sweep so data at rest is never trusted after a restart.
    return verify_capture_manifest(root, verify_chunks=False)


def _record(path: Path, window: CaptureWindow, index: int) -> dict[str, Any]:
    return {
        "file": path.relative_to(path.parents[1]).as_posix(),
        "receipt": _receipt_path(path).relative_to(path.parents[1]).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "window_index": index,
        "token_sha256": window.token_sha256,
    }


def verify_capture_manifest(
    root: str | Path,
    *,
    expected_request_sha256: str | None = None,
    verify_chunks: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "capture-manifest.json").read_text())
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("unsupported Qwen routed-capture manifest")
    expected = _require_hash(manifest.get("capture_sha256"), "capture manifest")
    body = {key: value for key, value in manifest.items() if key != "capture_sha256"}
    if sha256_bytes(canonical_json(body)) != expected:
        raise ValueError("Qwen routed-capture manifest seal mismatch")
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise ValueError("Qwen routed-capture manifest lacks its full request identity")
    request_sha256 = _require_hash(request.get("request_sha256"), "capture request")
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    if sha256_bytes(canonical_json(request_body)) != request_sha256 or manifest.get("request_sha256") != request_sha256:
        raise ValueError("Qwen routed-capture request identity mismatch")
    if expected_request_sha256 is not None and request_sha256 != _require_hash(expected_request_sha256, "expected capture request"):
        raise ValueError("completed Qwen routed capture belongs to a different request")
    layers = manifest.get("layers")
    records = manifest.get("records")
    if (
        not isinstance(layers, list)
        or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers)
        or len(set(layers)) != len(layers)
        or not isinstance(records, dict)
        or set(records) != {str(layer) for layer in layers}
        or any(not isinstance(rows, list) for rows in records.values())
    ):
        raise ValueError("Qwen routed-capture layer/record inventory is malformed")
    if verify_chunks:
        for rows in records.values():
            for row in rows:
                path = root / row["file"]
                receipt = verify_capture_chunk(path)
                if receipt.get("metadata", {}).get("capture_request_sha256") != request_sha256:
                    raise ValueError("Qwen routed-capture chunk is bound to a different request")
                if row["sha256"] != receipt["sha256"] or row["bytes"] != receipt["bytes"]:
                    raise ValueError("Qwen routed-capture record identity mismatch")
    return manifest


def load_capture_windows(sealed: Mapping[str, Any], role: str) -> tuple[CaptureWindow, ...]:
    windows = sealed["windows"][role]
    return tuple(
        CaptureWindow(
            token_ids=tuple(row["token_ids"]),
            token_sha256=row["token_sha256"],
            document_id=str(row.get("document_id", row.get("document_sha256", f"window-{index}"))),
            start_token=int(row.get("start_token", row.get("token_offset", row.get("offset", 0)))),
        )
        for index, row in enumerate(windows)
    )


def capture_moe_from_local_bf16(
    *,
    source_checkpoint: str | Path,
    model_revision: str,
    sealed_corpus: str | Path,
    role: str,
    layers: Sequence[int],
    predecessor_state_hash: str,
    output_dir: str | Path,
    installed_layers: Sequence[str | Path] = (),
    installed_layer_prefix: Sequence[Mapping[str, Any]] = (),
    device_map: Any = "auto",
    fisher_rank: int = 0,
    seed: int = 20260823,
    production_geometry: bool = True,
    writer_workers: int = 1,
    max_inflight_chunks: int | None = None,
    capture_checkpoint: str | Path | None = None,
    capture_checkpoint_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single-role compatibility wrapper over the one-load multi-role path."""

    return capture_roles_from_local_bf16(
        source_checkpoint=source_checkpoint,
        model_revision=model_revision,
        sealed_corpus=sealed_corpus,
        captures=[{
            "purpose": "capture",
            "role": role,
            "output_dir": str(output_dir),
            "fisher_rank": int(fisher_rank),
            "seed": int(seed),
        }],
        layers=layers,
        predecessor_state_hash=predecessor_state_hash,
        installed_layers=installed_layers,
        installed_layer_prefix=installed_layer_prefix,
        device_map=device_map,
        production_geometry=production_geometry,
        writer_workers=writer_workers,
        max_inflight_chunks=max_inflight_chunks,
        capture_checkpoint=capture_checkpoint,
        capture_checkpoint_identity=capture_checkpoint_identity,
    )["capture"]


def capture_roles_from_local_bf16(
    *,
    source_checkpoint: str | Path,
    model_revision: str,
    sealed_corpus: str | Path,
    captures: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    predecessor_state_hash: str,
    installed_layers: Sequence[str | Path] = (),
    installed_layer_prefix: Sequence[Mapping[str, Any]] = (),
    device_map: Any = "auto",
    attn_implementation: str = "eager",
    production_geometry: bool = True,
    writer_workers: int = 1,
    max_inflight_chunks: int | None = None,
    capture_checkpoint: str | Path | None = None,
    capture_checkpoint_identity: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load/replay one immutable model and capture all disjoint corpus roles."""

    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("model_revision must be immutable 40-hex")
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as error:  # pragma: no cover
        raise RuntimeError("torch and transformers are required for production Qwen capture") from error
    from ..calibration.windows import verify_sealed_corpus
    from ..checkpoint.btx_qwen import replay_installed_layers

    source = Path(source_checkpoint).resolve()
    if not source.is_dir() or source.is_symlink():
        raise ValueError("source_checkpoint must be a local immutable directory")
    capture_source = (
        source
        if capture_checkpoint is None
        else Path(capture_checkpoint).resolve()
    )
    if not capture_source.is_dir() or capture_source.is_symlink():
        raise ValueError("capture_checkpoint must be a local immutable directory")
    if capture_source != source and capture_checkpoint_identity is None:
        raise ValueError(
            "a distinct capture checkpoint requires a sealed capture checkpoint identity"
        )
    sealed = json.loads(Path(sealed_corpus).read_text())
    verify_sealed_corpus(sealed)
    requests = [dict(row) for row in captures]
    purposes = [row.get("purpose") for row in requests]
    if not requests or any(not isinstance(item, str) or not item for item in purposes) or len(set(purposes)) != len(purposes):
        raise ValueError("multi-role capture purposes must be nonempty and unique")
    for row in requests:
        if not isinstance(row.get("role"), str) or row["role"] not in sealed["windows"]:
            raise ValueError("multi-role capture references an unknown sealed-corpus role")
        if not isinstance(row.get("output_dir"), str) or not row["output_dir"]:
            raise ValueError("multi-role capture requires an output directory per purpose")
    model = AutoModelForCausalLM.from_pretrained(
        capture_source,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replay = replay_installed_layers(
        model,
        installed_layers,
        expected_final_state_hash=predecessor_state_hash,
        expected_prefix=installed_layer_prefix,
    )
    result = {}
    for row in requests:
        purpose = str(row["purpose"])
        result[purpose] = capture_loaded_qwen(
            model=model,
            windows=load_capture_windows(sealed, str(row["role"])),
            role=str(row["role"]),
            layers=layers,
            predecessor_state_hash=predecessor_state_hash,
            output_dir=str(row["output_dir"]),
            fisher_rank=int(row.get("fisher_rank", 0)),
            seed=int(row.get("seed", 20260823)),
            production_geometry=production_geometry,
            model_revision=model_revision,
            replay_report=replay,
            capture_checkpoint_identity=capture_checkpoint_identity,
            writer_workers=writer_workers,
            max_inflight_chunks=max_inflight_chunks,
        )
    return result
