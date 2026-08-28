#!/usr/bin/env python3
"""One-shot streaming capture of every MoE layer from the UNQUANTIZED source.

The sequential walk captures layer ``L`` only after layers ``3..L-1`` have been
encoded and installed, so its captures are serial by construction.  This module
is the deliberate opposite: it streams layers ``0..77`` **once**, always through
the BF16 source experts, and writes one :class:`~r7_encoder.flat_capture`
payload trio per MoE layer.  Nothing here ever installs a quantized expert or
reads a previous layer's encode, which is exactly what makes the 75 layer
encodes independent and lets the layer-parallel driver run them at once.

What is captured, per MoE layer, in canonical corpus-plan order:

    x.bin        the exact tensor the routed experts consume  (bf16 [n,6144])
    ids.bin      that layer's router top-8 expert ids         (uint8 [n,8])
    weights.bin  that layer's router top-8 weights            (f32   [n,8])

Both routing tensors come from the model's own router (``mlp.gate`` followed by
``mlp.route_tokens_to_experts``), which is the same call pair
``TransformersSequentialRuntime.route_exact`` uses, so the captured convention
is identical to the walk's -- including that each row sums to
``config.routed_scaling_factor`` rather than to 1.0.  A separate
``sigmoid(logits) + e_score_correction_bias`` top-k recomputation cross-checks
the selected expert set on the first prompts of every layer, and
:class:`~r7_encoder.routing.RoutedMassAccumulator` audits the float32 mass of
every row of every layer.  Any disagreement aborts the pass.

Restart contract: the unit of atomicity is one layer. In disk-state mode, a
layer's input is retired only after its successor and capture are sealed. In
RAM-state mode, the layer-3 prefix remains sealed as the durable replay anchor
while successor state rolls through host memory. A partially written capture is
discarded, never resumed, and every adopted capture must match the run bindings.

Importing this module is inert: no torch, no transformers, no CUDA, no
checkpoint.  Everything heavy is imported inside the functions that need it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import (
    DEFAULT_SIGMA_REG,
    FIRST_MOE_LAYER,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAST_MOE_LAYER,
    MOE_LAYERS,
    NUM_EXPERTS,
    PROJECTIONS,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TOP_K,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .flat_capture import (
    FlatCaptureReader,
    FlatCaptureWriter,
    MANIFEST_FILE,
    layer_capture_dir,
)
from .inventory import load_checkpoint_inventory
from .parallel_driver import parse_layers
from .routing import RoutedMassAccumulator
from .safetensors_io import SafeTensorReader, read_torch_tensor, read_torch_tensor_mmap

SCHEMA = "r7-capture-pass-v1"
STATE_SCHEMA = "r7-capture-pass-state-v1"
PROGRESS_FILE = "CAPTURE_PASS.json"
STATE_ROOT = "_state"
STATE_MANIFEST = "STATE.json"
RUNTIME_FACTORY = "r7_encoder.transformers_runtime:factory"

# Layers 0..77 of the main model; 78 is the MTP head and is never streamed.
FIRST_LAYER = 0
LAST_LAYER = LAST_MOE_LAYER

# How many prompts per layer get the independent sigmoid+bias top-k router
# cross-check.  The check is a per-layer convention gate, not a per-token one,
# so a small number is sufficient and costs nothing measurable.
ROUTER_CROSSCHECK_PROMPTS = 2

_EXPERT_STACK_SHAPES = {
    "gate_up_proj": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
    "down_proj": (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
}
_EXPERT_HF_SHAPES = {
    "gate_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
    "up_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
    "down_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def state_dir(capture_dir: str | Path, layer: int) -> Path:
    """Directory holding the per-prompt hidden state *entering* ``layer``."""

    value = int(layer)
    if not FIRST_LAYER <= value <= LAST_LAYER + 1:
        raise ValueError(f"state layer {value} outside [{FIRST_LAYER},{LAST_LAYER + 1}]")
    return Path(capture_dir) / STATE_ROOT / f"input-layer-{value:03d}"


def progress_path(capture_dir: str | Path) -> Path:
    return Path(capture_dir) / PROGRESS_FILE


def _sealed_inventory_sha256(path: Path, *, label: str) -> str:
    """Return a manifest's verified seal without reading any indexed payload."""

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} inventory is not a JSON object: {path}")
    digest = payload.pop("inventory_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError(f"{label} inventory seal mismatch: {path}")
    return str(digest)


def _require_checkpoint_root(
    inventory: Mapping[str, object], *, root: Path, label: str
) -> None:
    """Bind an inventory to its resolved checkpoint root without payload I/O."""

    declared = inventory.get("checkpoint")
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"{label} inventory lacks its checkpoint root")
    if Path(declared).expanduser().resolve() != root:
        raise ValueError(
            f"{label} inventory belongs to {Path(declared).expanduser().resolve()}, "
            f"not the requested checkpoint {root}"
        )


# ---------------------------------------------------------------------------
# BF16 source experts
# ---------------------------------------------------------------------------


def _payload_sha256(tensor) -> str:
    """SHA-256 of a tensor's raw payload bytes, without re-reading the shard."""

    import hashlib
    import torch

    flat = tensor.contiguous().view(torch.uint8).reshape(-1)
    return hashlib.sha256(memoryview(flat.numpy())).hexdigest()


class _Bf16ExpertSource:
    """Inventory-checked reader for the unquantized routed experts.

    Tensors are read through a private file mapping and copied straight into a
    preallocated device stack, so peak host residency is one expert, not one
    layer.  Reads are issued in on-disk order to keep the 1.4 TB source
    sequential.
    """

    def __init__(
        self,
        root: str | Path,
        inventory: Mapping[str, object],
        *,
        verify_payloads: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        entries = inventory.get("entries")
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError("BF16 source inventory carries no tensor entries")
        self.entries: Mapping[str, Mapping[str, object]] = entries  # type: ignore[assignment]
        self.verify_payloads = bool(verify_payloads)
        self._reader: SafeTensorReader | None = None
        self._reader_shard: str | None = None
        self.verified_tensors = 0

    def _record(self, name: str) -> Mapping[str, object]:
        try:
            record = self.entries[name]
        except KeyError as exc:
            raise KeyError(f"BF16 source inventory lacks {name}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"malformed BF16 source inventory record: {name}")
        return record

    def _open(self, shard_name: str) -> SafeTensorReader:
        if self._reader_shard != shard_name or self._reader is None:
            # One reader at a time: each holds a whole-shard private mapping.
            self._reader = SafeTensorReader(self.root / shard_name)
            self._reader_shard = shard_name
        return self._reader

    def close(self) -> None:
        self._reader = None
        self._reader_shard = None

    def _tensor(self, name: str, expected_shape: tuple[int, ...]):
        record = self._record(name)
        if str(record.get("dtype")) != "BF16":
            raise ValueError(
                f"{name}: BF16 source declares dtype {record.get('dtype')!r}; "
                "quant-of-quant is forbidden"
            )
        if tuple(int(value) for value in record.get("shape", ())) != expected_shape:
            raise ValueError(
                f"{name}: inventory shape {record.get('shape')} != {list(expected_shape)}"
            )
        reader = self._open(str(record["shard"]))
        info = reader.tensors.get(name)
        if info is None:
            raise ValueError(f"{name}: absent from shard {record['shard']}")
        if (
            info.dtype != "BF16"
            or tuple(info.shape) != expected_shape
            or info.payload.start != int(record["payload_start"])
            or info.payload.end != int(record["payload_end"])
        ):
            raise ValueError(f"{name}: BF16 source header differs from inventory")
        tensor = read_torch_tensor_mmap(reader, name)
        if self.verify_payloads:
            if _payload_sha256(tensor) != str(record["payload_sha256"]):
                raise ValueError(
                    f"{name}: BF16 source payload changed after inventory"
                )
            self.verified_tensors += 1
        return tensor

    def layer_plan(self, layer: int) -> list[tuple[int, str, str]]:
        """``(expert, projection, tensor_name)`` sorted by on-disk position."""

        plan: list[tuple[str, int, int, str, str]] = []
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                name = (
                    f"model.layers.{int(layer)}.mlp.experts.{expert}."
                    f"{projection}.weight"
                )
                record = self._record(name)
                plan.append(
                    (
                        str(record["shard"]),
                        int(record["payload_start"]),
                        expert,
                        projection,
                        name,
                    )
                )
        plan.sort()
        return [(expert, projection, name) for _, _, expert, projection, name in plan]

    def build_layer_stacks(self, layer: int, device) -> tuple[Any, Any]:
        """Return ``(gate_up_proj, down_proj)`` device stacks for one layer.

        The packing matches ``TransformersSequentialRuntime._layer_state``:
        ``gate_up_proj[e]`` is ``[gate; up]`` with each half ``[I,H]`` and
        ``down_proj[e]`` is ``[H,I]`` -- i.e. exactly the HF ``nn.Linear``
        ``[out,in]`` orientation of the source tensors, no transpose.
        """

        import torch

        gate_up = torch.empty(
            _EXPERT_STACK_SHAPES["gate_up_proj"], dtype=torch.bfloat16, device=device
        )
        down = torch.empty(
            _EXPERT_STACK_SHAPES["down_proj"], dtype=torch.bfloat16, device=device
        )
        seen: set[tuple[int, str]] = set()
        try:
            for expert, projection, name in self.layer_plan(layer):
                tensor = self._tensor(name, _EXPERT_HF_SHAPES[projection])
                if projection == "gate_proj":
                    gate_up[expert, :INTERMEDIATE_SIZE].copy_(tensor)
                elif projection == "up_proj":
                    gate_up[expert, INTERMEDIATE_SIZE:].copy_(tensor)
                else:
                    down[expert].copy_(tensor)
                seen.add((expert, projection))
                del tensor
        finally:
            self.close()
        expected = {
            (expert, projection)
            for expert in range(NUM_EXPERTS)
            for projection in PROJECTIONS
        }
        if seen != expected:
            raise ValueError(
                f"layer {layer}: BF16 expert stack is incomplete "
                f"({len(seen)}/{len(expected)} tensors)"
            )
        return gate_up, down


# ---------------------------------------------------------------------------
# MoE tap
# ---------------------------------------------------------------------------


class _MoeTap:
    """Forward pre-hook on ``mlp.experts``: the routed input and its routing.

    The hook sees the *exact* tensor the routed experts consume -- the same
    ``post_attention_layernorm`` output the walk names ``moe_hidden`` -- without
    reimplementing any part of the decoder layer.  Routing is then produced by
    the model's own ``mlp.gate`` / ``mlp.route_tokens_to_experts`` pair on that
    same tensor, which is the call sequence ``route_exact`` uses.
    """

    def __init__(self, mlp, *, layer: int, routed_scaling_factor: float) -> None:
        self.mlp = mlp
        self.layer = int(layer)
        self.routed_scaling_factor = float(routed_scaling_factor)
        if not hasattr(mlp, "gate") or not hasattr(mlp, "route_tokens_to_experts"):
            raise ValueError(
                f"layer {self.layer}: mlp lacks the gate/route_tokens_to_experts "
                "pair the runtime routes with"
            )
        self.crosscheck = False
        self.tokens = 0
        self.calls = 0
        self._hidden: list[Any] = []
        self._ids: list[Any] = []
        self._weights: list[Any] = []

    def reset(self, *, tokens: int, crosscheck: bool) -> None:
        self.tokens = int(tokens)
        self.crosscheck = bool(crosscheck)
        self.calls = 0
        self._hidden = []
        self._ids = []
        self._weights = []

    # -- routing cross-check --------------------------------------------

    def _crosscheck_ids(self, hidden, ids) -> None:
        import torch
        import torch.nn.functional as functional

        gate = self.mlp.gate
        weight = getattr(gate, "weight", None)
        bias = getattr(gate, "e_score_correction_bias", None)
        if weight is None or bias is None:
            raise ValueError(
                f"layer {self.layer}: router lacks weight/e_score_correction_bias; "
                "the independent top-k cross-check cannot be evaluated"
            )
        if tuple(weight.shape) != (NUM_EXPERTS, HIDDEN_SIZE):
            raise ValueError(
                f"layer {self.layer}: router weight {tuple(weight.shape)} != "
                f"{(NUM_EXPERTS, HIDDEN_SIZE)}"
            )
        flat_bias = bias.detach().to(torch.float32).flatten()
        if tuple(flat_bias.shape) != (NUM_EXPERTS,):
            raise ValueError(
                f"layer {self.layer}: router bias {tuple(bias.shape)} is not "
                f"[{NUM_EXPERTS}]"
            )
        logits = functional.linear(
            hidden.to(torch.float32), weight.detach().to(torch.float32)
        )
        scores = torch.sigmoid(logits) + flat_bias
        reference = torch.topk(scores, TOP_K, dim=-1, sorted=False).indices
        mismatched = int(
            (
                reference.sort(dim=-1).values
                != ids.to(reference.dtype).sort(dim=-1).values
            )
            .any(dim=-1)
            .sum()
            .item()
        )
        if mismatched:
            raise ValueError(
                f"layer {self.layer}: router convention mismatch -- "
                f"sigmoid(logits)+bias top-{TOP_K} disagrees with the model router "
                f"on {mismatched} of {int(ids.shape[0])} tokens"
            )

    # -- the hook --------------------------------------------------------

    def __call__(self, module, args, kwargs):
        import torch

        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if hidden is None or not torch.is_tensor(hidden):
            raise ValueError(
                f"layer {self.layer}: routed-expert input is not a tensor; "
                "the MoE dispatch signature drifted"
            )
        if hidden.ndim < 2 or int(hidden.shape[-1]) != HIDDEN_SIZE:
            raise ValueError(
                f"layer {self.layer}: routed-expert input {tuple(hidden.shape)} is "
                f"not [...,{HIDDEN_SIZE}]"
            )
        flat = hidden.detach().reshape(-1, HIDDEN_SIZE)
        rows = int(flat.shape[0])
        if rows <= 0 or rows > self.tokens:
            raise ValueError(
                f"layer {self.layer}: routed-expert input has {rows} rows, "
                f"expected at most {self.tokens}"
            )
        if not torch.isfinite(flat).all():
            raise ValueError(f"layer {self.layer}: routed-expert input is non-finite")

        logits = self.mlp.gate(flat)
        routed = self.mlp.route_tokens_to_experts(logits)
        if not isinstance(routed, tuple) or len(routed) != 2:
            raise ValueError(
                f"layer {self.layer}: route_tokens_to_experts did not return "
                "(ids, weights)"
            )
        ids, weights = routed
        if not torch.is_tensor(ids) or not torch.is_tensor(weights):
            raise ValueError(f"layer {self.layer}: router returned non-tensors")
        if tuple(ids.shape) != (rows, TOP_K) or tuple(weights.shape) != (rows, TOP_K):
            raise ValueError(
                f"layer {self.layer}: router returned ids={tuple(ids.shape)} "
                f"weights={tuple(weights.shape)}, expected [{rows},{TOP_K}]"
            )
        ids = ids.detach().to(torch.int64)
        if int(ids.min()) < 0 or int(ids.max()) >= NUM_EXPERTS:
            raise ValueError(
                f"layer {self.layer}: routed expert id outside [0,{NUM_EXPERTS})"
            )
        weights = weights.detach().to(torch.float32)
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError(
                f"layer {self.layer}: router weights must be finite and nonnegative"
            )
        if self.crosscheck and self.calls == 0:
            self._crosscheck_ids(flat, ids)

        # Materialize as owned numpy buffers *inside* the inference-mode region
        # so the writer never touches an inference tensor.
        self._hidden.append(
            flat.to("cpu", dtype=torch.bfloat16)
            .contiguous()
            .view(torch.int16)
            .numpy()
            .copy()
        )
        self._ids.append(ids.to("cpu").contiguous().numpy().copy())
        self._weights.append(weights.to("cpu").contiguous().numpy().copy())
        self.calls += 1
        return None

    # -- results ---------------------------------------------------------

    def result(self):
        """``(hidden bf16 [n,6144], ids int64 [n,8], weights f32 [n,8])``."""

        import numpy as np
        import torch

        if not self.calls:
            raise ValueError(
                f"layer {self.layer}: the routed-expert module never ran; nothing "
                "was captured"
            )
        hidden = (
            self._hidden[0] if self.calls == 1 else np.concatenate(self._hidden, axis=0)
        )
        ids = self._ids[0] if self.calls == 1 else np.concatenate(self._ids, axis=0)
        weights = (
            self._weights[0]
            if self.calls == 1
            else np.concatenate(self._weights, axis=0)
        )
        if hidden.shape[0] != self.tokens:
            raise ValueError(
                f"layer {self.layer}: captured {hidden.shape[0]} routed rows for a "
                f"{self.tokens}-token prompt"
            )
        if ids.shape[0] != self.tokens or weights.shape[0] != self.tokens:
            raise ValueError(f"layer {self.layer}: routing/hidden row-count drift")
        return (
            torch.from_numpy(hidden).view(torch.bfloat16),
            torch.from_numpy(ids),
            torch.from_numpy(weights),
        )


# ---------------------------------------------------------------------------
# rolling state
# ---------------------------------------------------------------------------


def _record_from_runtime(directory: Path, record: Sequence[Any]) -> dict[str, object]:
    shard_id, hidden_path, metadata_path, tokens, hidden_size = record
    hidden_path = Path(hidden_path)
    metadata_path = Path(metadata_path)
    if (
        hidden_path.parent.resolve() != directory.resolve()
        or metadata_path.parent.resolve() != directory.resolve()
    ):
        raise ValueError("runtime state artifact escaped the capture state directory")
    if int(hidden_size) != HIDDEN_SIZE:
        raise ValueError("runtime state hidden size drift")
    return {
        "shard_id": str(shard_id),
        "hidden": hidden_path.name,
        "metadata": metadata_path.name,
        "tokens": int(tokens),
    }


def _seal_state(
    directory: Path,
    *,
    layer: int,
    bindings: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    expected: Mapping[str, int],
) -> dict[str, object]:
    order = [str(item["shard_id"]) for item in records]
    if sorted(order) != order or len(set(order)) != len(order):
        raise ValueError("capture state prompts are not in canonical order")
    if {key: int(item["tokens"]) for key, item in zip(order, records)} != dict(expected):
        raise ValueError("capture state does not cover the sealed prompt domain")
    for item in records:
        for key in ("hidden", "metadata"):
            path = directory / str(item[key])
            if Path(str(item[key])).name != str(item[key]) or not path.is_file():
                raise ValueError(f"capture state artifact missing: {item[key]}")
    payload = {
        "schema": STATE_SCHEMA,
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "layer": int(layer),
        "bindings": dict(bindings),
        "tokens": sum(int(item["tokens"]) for item in records),
        "prompts": [dict(item) for item in records],
    }
    payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_write_json(directory / STATE_MANIFEST, payload)
    return payload


def _load_state(
    directory: Path,
    *,
    layer: int,
    bindings: Mapping[str, object],
    expected: Mapping[str, int],
) -> list[dict[str, object]] | None:
    manifest = directory / STATE_MANIFEST
    if not manifest.is_file():
        return None
    payload = read_json(manifest)
    if not isinstance(payload, dict):
        raise ValueError(f"malformed capture state manifest: {manifest}")
    content = payload.pop("content_sha256", None)
    if content != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError(f"capture state manifest digest mismatch: {manifest}")
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("marker") != RECIPE_MARKER
        or int(payload.get("layer", -1)) != int(layer)
    ):
        raise ValueError(f"foreign capture state manifest: {manifest}")
    if payload.get("bindings") != dict(bindings):
        raise ValueError(
            f"capture state at {directory} belongs to different inputs; delete "
            f"{directory.parent} to rebuild the rolling state"
        )
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"capture state manifest has no prompts: {manifest}")
    records = [dict(item) for item in prompts]
    order = [str(item["shard_id"]) for item in records]
    if sorted(order) != order or len(set(order)) != len(order):
        raise ValueError("capture state prompts are not in canonical order")
    if {key: int(item["tokens"]) for key, item in zip(order, records)} != dict(expected):
        raise ValueError("capture state does not cover the sealed prompt domain")
    for item in records:
        for key in ("hidden", "metadata"):
            name = str(item[key])
            if Path(name).name != name or not (directory / name).is_file():
                raise ValueError(f"capture state artifact missing: {directory / name}")
    return records


def _read_prompt(directory: Path, record: Mapping[str, object]):
    """Return ``(hidden, metadata)`` ready for ``runtime._advance``."""

    tokens = int(record["tokens"])
    shard_id = str(record["shard_id"])
    reader = SafeTensorReader(directory / str(record["hidden"]))
    if set(reader.tensors) != {"hidden", "prev_topk_indices"}:
        raise ValueError(f"capture state shard {shard_id} has a foreign tensor set")
    info = reader.tensors["hidden"]
    if info.dtype != "BF16" or tuple(info.shape) != (tokens, HIDDEN_SIZE):
        raise ValueError(
            f"capture state shard {shard_id}: hidden must be BF16 "
            f"[{tokens},{HIDDEN_SIZE}], got {info.dtype} {tuple(info.shape)}"
        )
    previous_info = reader.tensors["prev_topk_indices"]
    if previous_info.dtype != "I32" or previous_info.shape[0] != tokens:
        raise ValueError(
            f"capture state shard {shard_id}: prev_topk_indices must be I32 "
            f"[{tokens},*]"
        )
    hidden = read_torch_tensor(reader, "hidden")
    previous = read_torch_tensor(reader, "prev_topk_indices")
    metadata = dict(read_json(directory / str(record["metadata"])))
    if (
        metadata.get("schema") != "r7-state-metadata-v2"
        or str(metadata.get("shard_id")) != shard_id
        or int(metadata.get("tokens", -1)) != tokens
        or metadata.get("sequence_lengths") != [tokens]
    ):
        raise ValueError(f"capture state metadata drift for shard {shard_id}")
    metadata["prev_topk_indices"] = previous
    return hidden, metadata


def _memory_prompt(record: Mapping[str, object]):
    """Read an R10 rolling state held in host RAM instead of 1,773 files."""

    import torch

    hidden = record.get("hidden_tensor")
    metadata = record.get("metadata_value")
    if not torch.is_tensor(hidden) or not isinstance(metadata, Mapping):
        raise ValueError("malformed in-memory rolling prompt")
    tokens = int(record["tokens"])
    if tuple(hidden.shape) != (tokens, HIDDEN_SIZE) or hidden.dtype != torch.bfloat16:
        raise ValueError("in-memory rolling hidden geometry drift")
    value = dict(metadata)
    previous = value.get("prev_topk_indices")
    if not torch.is_tensor(previous) or int(previous.shape[0]) != tokens:
        raise ValueError("in-memory rolling DSA state geometry drift")
    return hidden, value


def _remove_tree(directory: Path) -> None:
    import shutil

    if directory.exists():
        shutil.rmtree(directory)


# ---------------------------------------------------------------------------
# capture validation
# ---------------------------------------------------------------------------


def _capture_is_complete(
    capture_dir: Path,
    layer: int,
    expected_tokens: int,
    *,
    expected_bindings: Mapping[str, object],
    expected_verification: Mapping[str, object],
    verify_payloads: bool,
    log,
) -> dict[str, object] | None:
    manifest = layer_capture_dir(capture_dir, layer) / MANIFEST_FILE
    if not manifest.is_file():
        return None
    try:
        with FlatCaptureReader(
            manifest,
            expected_layer=layer,
            expected_bindings=expected_bindings,
            expected_verification=expected_verification,
            verify_payloads=verify_payloads,
            verify_structure=verify_payloads,
        ) as reader:
            if reader.tokens != int(expected_tokens):
                raise ValueError(
                    f"layer {layer}: capture holds {reader.tokens} tokens, the "
                    f"corpus plan declares {int(expected_tokens)}"
                )
            if reader.hidden_size != HIDDEN_SIZE or reader.top_k != TOP_K:
                raise ValueError(f"layer {layer}: capture geometry drift")
            summary = {
                "tokens": reader.tokens,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "content_sha256": str(reader.manifest.get("content_sha256", "")),
                "verification": dict(reader.verification),
            }
        return summary
    except (ValueError, OSError, KeyError) as exc:
        log(f"layer {layer}: existing capture rejected ({exc}); it will be rebuilt")
        return None


# ---------------------------------------------------------------------------
# layer module
# ---------------------------------------------------------------------------


def _build_layer_module(runtime, layer: int, experts: _Bf16ExpertSource | None):
    """One decoder layer: carrier dense/attention, BF16 *source* routed experts."""

    import torch

    module = runtime._layer_state(int(layer), include_experts=False)
    if int(layer) < FIRST_MOE_LAYER:
        return module
    if experts is None:
        raise ValueError("MoE layers require a BF16 expert source")
    holder = getattr(getattr(module, "mlp", None), "experts", None)
    if holder is None:
        raise ValueError(f"layer {layer}: official module has no mlp.experts")
    for name, shape in _EXPERT_STACK_SHAPES.items():
        parameter = getattr(holder, name, None)
        if parameter is None or tuple(parameter.shape) != shape:
            raise ValueError(
                f"layer {layer}: mlp.experts.{name} is "
                f"{None if parameter is None else tuple(parameter.shape)}, "
                f"expected {shape}; the official expert layout drifted"
            )
    gate_up, down = experts.build_layer_stacks(int(layer), runtime.device)
    incompatible = module.load_state_dict(
        {"mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down},
        strict=False,
        assign=True,
    )
    if incompatible.unexpected_keys:
        raise ValueError(
            f"layer {layer}: BF16 expert install rejected "
            f"{incompatible.unexpected_keys}"
        )
    missing = set(incompatible.missing_keys)
    if missing & set(f"mlp.experts.{name}" for name in _EXPERT_STACK_SHAPES):
        raise ValueError(f"layer {layer}: BF16 expert install did not bind")
    for name, tensor in module.state_dict().items():
        if torch.is_tensor(tensor) and tensor.is_meta:
            raise ValueError(f"layer {layer}: {name} is still on meta after loading")
    module.eval()
    return module


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_capture(
    *,
    carrier: str | Path,
    bf16_source: str | Path,
    corpus: str | Path,
    capture_dir: str | Path,
    carrier_inventory: str | Path,
    source_inventory: str | Path,
    numeric_inventory: str | Path,
    runtime_inventory: str | Path,
    layers: str | Iterable[int] = "3-77",
    device: str = "cuda:0",
    capture_devices: tuple = (),
    sigma_reg: float = DEFAULT_SIGMA_REG,
    verify_expert_payloads: bool = True,
    verify_resume_payloads: bool = False,
    retire_state: bool = True,
    memory_state: bool = False,
    verify_routing: bool = True,
    verify_runtime_files: bool = True,
    verify_carrier_files: bool = True,
    verify_carrier_payloads: bool = True,
    verify_numeric_files: bool = True,
    log=print,
) -> dict[str, object]:
    """Stream layers 0..N once and write one flat capture per requested layer.

    Every requested layer is captured against the BF16 source experts, so no
    layer depends on any other layer's encode.  Returns a JSON-ready summary.
    """

    from types import SimpleNamespace

    import torch

    from .transformers_runtime import factory as runtime_factory

    requested = (
        parse_layers(layers)
        if isinstance(layers, str)
        else sorted({int(value) for value in layers})
    )
    if not requested or any(value not in MOE_LAYERS for value in requested):
        raise ValueError(
            f"layers must be a nonempty subset of {MOE_LAYERS[0]}..{MOE_LAYERS[-1]}"
        )
    capture_dir = Path(capture_dir).resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)
    carrier = Path(carrier).resolve()
    bf16_source = Path(bf16_source).resolve()
    corpus = Path(corpus).resolve()

    config = SimpleNamespace(
        carrier=carrier,
        bf16_source=bf16_source,
        corpus=corpus,
        work=capture_dir,
        capture_dir=capture_dir,
        carrier_inventory=Path(carrier_inventory).resolve(),
        source_inventory=Path(source_inventory).resolve(),
        numeric_inventory=Path(numeric_inventory).resolve(),
        runtime_inventory=Path(runtime_inventory).resolve(),
        device=str(device),
        devices=tuple(capture_devices) if capture_devices else (str(device),),
        sigma_reg=float(sigma_reg),
        runtime_factory=RUNTIME_FACTORY,
        verify_runtime_files=bool(verify_runtime_files),
        verify_carrier_files=bool(verify_carrier_files),
        verify_carrier_payloads=bool(verify_carrier_payloads),
        verify_numeric_files=bool(verify_numeric_files),
    )
    carrier_checkpoint = load_checkpoint_inventory(
        config.carrier_inventory, role="carrier"
    )
    source_checkpoint = load_checkpoint_inventory(
        config.source_inventory, role="bf16-source", require_routed_bf16=True
    )
    _require_checkpoint_root(carrier_checkpoint, root=carrier, label="carrier")
    _require_checkpoint_root(
        source_checkpoint, root=bf16_source, label="BF16 source"
    )
    numeric_inventory_sha256 = _sealed_inventory_sha256(
        config.numeric_inventory, label="numeric environment"
    )

    log(f"capture pass: building runtime on {device}")
    runtime = runtime_factory(config)
    if (
        runtime.carrier_inventory.get("inventory_sha256")
        != carrier_checkpoint.get("inventory_sha256")
    ):
        raise RuntimeError("runtime loaded a different carrier inventory seal")
    experts = _Bf16ExpertSource(
        bf16_source, source_checkpoint, verify_payloads=bool(verify_expert_payloads)
    )

    plan = runtime.prepare_corpus_plan(corpus=corpus)
    corpus_plan_sha256 = str(plan["corpus_plan_sha256"])
    selected = tuple(runtime._corpus_plan_payload["selected"])
    order = [str(raw["shard_id"]) for raw in selected]
    if sorted(order) != order:
        raise ValueError(
            "corpus-plan prompt IDs are not lexicographically ordered; flat "
            "capture rows would not follow global row order"
        )
    expected_shards = {
        str(raw["shard_id"]): int(raw["tokens"]) for raw in selected
    }
    expected_tokens = sum(expected_shards.values())
    routed_scaling_factor = float(runtime.config.routed_scaling_factor)
    if not routed_scaling_factor > 0:
        raise ValueError("routed_scaling_factor must be positive")

    verification = {
        "verify_runtime_files": bool(verify_runtime_files),
        "verify_carrier_files": bool(verify_carrier_files),
        "verify_carrier_payloads": bool(verify_carrier_payloads),
        "verify_numeric_files": bool(verify_numeric_files),
        "verify_expert_payloads": bool(verify_expert_payloads),
        "verify_resume_payloads": bool(verify_resume_payloads),
        "verify_routing": bool(verify_routing),
    }
    bindings = {
        "capture_pass_sha256": sha256_file(Path(__file__)),
        "corpus_plan_sha256": corpus_plan_sha256,
        "runtime_fingerprint": str(runtime.fingerprint),
        "carrier_inventory_sha256": str(runtime.carrier_inventory["inventory_sha256"]),
        "source_inventory_sha256": str(source_checkpoint["inventory_sha256"]),
        "numeric_inventory_sha256": numeric_inventory_sha256,
        "carrier_checkpoint": str(carrier),
        "source_checkpoint": str(bf16_source),
        "capture_device_pool": [str(item) for item in config.devices],
        "dispatch_audit_sha256": str(runtime.dispatch_audit_sha256),
        "routed_scaling_factor": format(routed_scaling_factor, ".17g"),
        "expert_source": "bf16-source-unquantized",
    }

    progress_file = progress_path(capture_dir)
    progress: dict[str, object] = {
        "schema": SCHEMA,
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "bindings": bindings,
        "verification": verification,
        "expected_tokens": expected_tokens,
        "state_layer": FIRST_LAYER,
        "layers": {},
    }
    if progress_file.exists():
        prior = read_json(progress_file)
        if not isinstance(prior, dict) or prior.get("schema") != SCHEMA:
            raise ValueError(f"foreign capture-pass progress file: {progress_file}")
        if prior.get("bindings") != bindings:
            raise ValueError(
                "capture-pass progress belongs to different inputs; choose a new "
                f"--capture-dir or remove the entire capture directory {capture_dir}, "
                "including every layer_* payload, before restarting"
            )
        if int(prior.get("expected_tokens", -1)) != expected_tokens:
            raise ValueError("capture-pass progress token-domain drift")
        progress["state_layer"] = int(prior.get("state_layer", FIRST_LAYER))
        progress["layers"] = dict(prior.get("layers", {}))
        progress["verification"] = verification

    def _commit(state_layer: int) -> None:
        progress["state_layer"] = int(state_layer)
        atomic_write_json(progress_file, progress)

    final_layer = max(requested)
    captured: dict[int, dict[str, object]] = {}
    mass_audits: dict[int, str] = {}

    # ---- resume point ------------------------------------------------------
    state_layer = FIRST_LAYER
    for candidate in range(LAST_LAYER + 1, FIRST_LAYER, -1):
        records = _load_state(
            state_dir(capture_dir, candidate),
            layer=candidate,
            bindings=bindings,
            expected=expected_shards,
        )
        if records is not None:
            state_layer = candidate
            break
    # ---- carried prefix: layers 0..2 --------------------------------------
    if state_layer < FIRST_MOE_LAYER:
        prefix_dir = state_dir(capture_dir, FIRST_MOE_LAYER)
        prefix_dir.mkdir(parents=True, exist_ok=True)
        log(f"capture pass: forwarding carried layers 0..{FIRST_MOE_LAYER - 1}")
        prefix_records = [
            _record_from_runtime(prefix_dir, record)
            for record in runtime.initialize_carried_state(
                carrier=carrier,
                corpus=corpus,
                output_partial=prefix_dir,
                completed_shard_ids=frozenset(),
            )
        ]
        prefix_records.sort(key=lambda item: str(item["shard_id"]))
        _seal_state(
            prefix_dir,
            layer=FIRST_MOE_LAYER,
            bindings=bindings,
            records=prefix_records,
            expected=expected_shards,
        )
        state_layer = FIRST_MOE_LAYER
        _commit(state_layer)
        log(f"capture pass: carried prefix sealed -> {prefix_dir}")

    memory_records: list[dict[str, object]] | None = None
    if memory_state:
        # Convert the durable prefix checkpoint into host-RAM records once. Every
        # subsequent layer already returns CPU BF16 hidden/top-k tensors, so
        # retaining them avoids ~18.5 GB and 3,546 metadata/data file writes per
        # layer. Keep the prefix sealed on disk: it is the restart anchor after
        # an interrupted RAM-state pass, while completed flat captures remain
        # independently adoptable through their exact input bindings.
        prefix_dir = state_dir(capture_dir, FIRST_MOE_LAYER)
        prefix_records = _load_state(
            prefix_dir,
            layer=FIRST_MOE_LAYER,
            bindings=bindings,
            expected=expected_shards,
        )
        if prefix_records is None:
            raise ValueError("R10 memory-state prefix is missing")
        memory_records = []
        for record in prefix_records:
            hidden, metadata = _read_prompt(prefix_dir, record)
            memory_records.append(
                {
                    "shard_id": str(record["shard_id"]),
                    "tokens": int(record["tokens"]),
                    "hidden_tensor": hidden,
                    "metadata_value": metadata,
                }
            )
        resume_layer = FIRST_MOE_LAYER
        log(
            "capture pass: R10 rolling state is host RAM; durable layer-3 prefix "
            "retained, per-layer successor safetensors/JSON writes disabled"
        )

    # A requested layer below the rolling front is fine only if its capture is
    # already sealed, or its input state was retained (``--keep-states``).
    resume_layer = FIRST_MOE_LAYER if memory_state else state_layer
    for value in sorted(value for value in requested if value < state_layer):
        summary = _capture_is_complete(
            capture_dir,
            value,
            expected_tokens,
            expected_bindings=bindings,
            expected_verification=verification,
            verify_payloads=bool(verify_resume_payloads),
            log=log,
        )
        if summary is not None:
            captured[value] = summary
            continue
        if memory_state:
            continue
        if (
            _load_state(
                state_dir(capture_dir, value),
                layer=value,
                bindings=bindings,
                expected=expected_shards,
            )
            is None
        ):
            raise ValueError(
                f"layer {value} was requested but its input state has already been "
                f"retired (rolling state is at layer {state_layer}); delete "
                f"{capture_dir / STATE_ROOT} to re-stream from the corpus"
            )
        resume_layer = min(resume_layer, value)
    log(
        f"capture pass: {len(requested)} requested layers "
        f"({requested[0]}..{requested[-1]}), resuming at layer {resume_layer} "
        f"(rolling front {state_layer}), {expected_tokens} tokens over "
        f"{len(selected)} prompts"
    )

    # ---- streaming MoE layers ---------------------------------------------
    for layer in range(resume_layer, final_layer + 1):
        source_state = None if memory_state else state_dir(capture_dir, layer)
        if memory_state:
            input_records = memory_records
        else:
            assert source_state is not None
            input_records = _load_state(
                source_state, layer=layer, bindings=bindings, expected=expected_shards
            )
        if input_records is None:
            raise ValueError(f"layer {layer}: rolling input state is missing")
        wants_capture = layer in requested
        has_successor = layer < final_layer and layer < LAST_LAYER
        successor_dir = state_dir(capture_dir, layer + 1) if has_successor else None

        capture_summary = (
            _capture_is_complete(
                capture_dir,
                layer,
                expected_tokens,
                expected_bindings=bindings,
                expected_verification=verification,
                verify_payloads=bool(verify_resume_payloads),
                log=log,
            )
            if wants_capture
            else None
        )
        successor_ready = memory_state or successor_dir is None or (
            _load_state(
                successor_dir,
                layer=layer + 1,
                bindings=bindings,
                expected=expected_shards,
            )
            is not None
        )
        needs_capture = wants_capture and capture_summary is None
        needs_successor = has_successor if memory_state else (
            successor_dir is not None and not successor_ready
        )
        if not needs_capture and not needs_successor:
            entry: dict[str, object] = {"captured": bool(wants_capture)}
            if capture_summary is not None:
                captured[layer] = capture_summary
                entry.update(capture_summary)
            assert isinstance(progress["layers"], dict)
            progress["layers"][f"{layer:03d}"] = entry
            log(f"layer {layer}: already complete, skipping")
            if retire_state and successor_dir is not None and source_state is not None:
                _remove_tree(source_state)
            _commit(layer + 1 if has_successor else layer)
            continue

        # A still-valid successor state is never rebuilt just to re-capture.
        target_state = (
            successor_dir if needs_successor and not memory_state else None
        )

        if not needs_capture and capture_summary is not None:
            captured[layer] = capture_summary
        # One layer is the unit of atomicity: discard any partial artifacts.
        if needs_capture:
            _remove_tree(layer_capture_dir(capture_dir, layer))
        if target_state is not None:
            _remove_tree(target_state)
            target_state.mkdir(parents=True, exist_ok=True)

        log(
            f"layer {layer}: loading carrier dense + 768 BF16 source expert tensors"
            + (" (payload-verified)" if verify_expert_payloads else "")
        )
        module = _build_layer_module(runtime, layer, experts)
        decode_audit = str(runtime.loader.audit_sha256)
        writer: FlatCaptureWriter | None = None
        accumulator: RoutedMassAccumulator | None = None
        tap: _MoeTap | None = None
        handle = None
        successor_records: list[dict[str, object]] = []
        replicas = []
        replica_taps = []
        replica_handles = []
        executors = []
        units = []
        pending_futures = []
        iterator = None
        replica = None
        rtap = None
        _forward_one = None
        _ordered_results = None
        try:
            if needs_capture:
                writer = FlatCaptureWriter(
                    capture_dir,
                    layer=layer,
                    hidden_size=HIDDEN_SIZE,
                    top_k=TOP_K,
                    num_experts=NUM_EXPERTS,
                    bindings=bindings,
                    verification=verification,
                    overwrite=True,
                )
                accumulator = (
                    RoutedMassAccumulator(layer) if verify_routing else None
                )
                tap = _MoeTap(
                    module.mlp,
                    layer=layer,
                    routed_scaling_factor=routed_scaling_factor,
                )
            rows = 0
            # ---- multi-GPU prompt sharding -------------------------------
            # The capture forward is the same module applied to independent
            # prompts, so it replicates cleanly. Build one replica per extra
            # device once per layer, then round-robin prompts across them.  A
            # single-thread executor owns each unit: a general thread pool can
            # otherwise schedule two prompts concurrently on the same module
            # and race that unit's mutable tap buffers. Each serial queue keeps
            # at most one prompt outstanding so completed tensors cannot pile
            # up for the full corpus behind an earlier long prompt.
            # The MoE tap and the writer stay on THIS thread in prompt order,
            # so captured rows, mass audit, and successor state are in exactly
            # the same order as the single-device path.
            pool_devices = list(getattr(runtime, "capture_devices", ()) or ())
            if len(pool_devices) > 1:
                import copy as _copy

                for extra in pool_devices[1:]:
                    replica = _copy.deepcopy(module).to(extra)
                    if tap is not None:
                        rtap = _MoeTap(
                            replica.mlp,
                            layer=layer,
                            routed_scaling_factor=routed_scaling_factor,
                        )
                        replica_handles.append(
                            replica.mlp.experts.register_forward_pre_hook(
                                rtap, with_kwargs=True
                            )
                        )
                        replica_taps.append(rtap)
                    replicas.append((replica, extra))
                log(
                    f"layer {layer}: sharding prompts over {len(replicas)+1} devices"
                )
            if tap is not None:
                # Register only after deepcopy so replicas do not inherit a
                # second, redundant copy of the primary tap.
                handle = module.mlp.experts.register_forward_pre_hook(
                    tap, with_kwargs=True
                )
            units.extend(
                [(module, tap, None)]
                + [
                    (rep, replica_taps[i] if replica_taps else None, dev)
                    for i, (rep, dev) in enumerate(replicas)
                ]
            )

            from concurrent.futures import ThreadPoolExecutor

            def _forward_one(job):
                idx, rec = job
                unit_module, unit_tap, unit_device = units[idx % len(units)]
                tk = int(rec["tokens"])
                hid, meta = (
                    _memory_prompt(rec)
                    if memory_state
                    else _read_prompt(source_state, rec)
                )
                if unit_tap is not None:
                    unit_tap.reset(
                        tokens=tk,
                        crosscheck=(
                            verify_routing and idx < ROUTER_CROSSCHECK_PROMPTS
                        ),
                    )
                out, prev = runtime._advance(
                    unit_module, hid, meta, device=unit_device
                )
                prompt_capture = unit_tap.result() if unit_tap is not None else None
                return rec, meta, out, prev, prompt_capture

            if len(units) > 1:
                executors = [ThreadPoolExecutor(max_workers=1) for _ in units]
                pending_futures = [None] * len(units)
                for initial_index in range(min(len(units), len(input_records))):
                    pending_futures[initial_index] = executors[initial_index].submit(
                        _forward_one, (initial_index, input_records[initial_index])
                    )

                def _ordered_results():
                    # Each device owns at most one running/queued future. Once
                    # its result is available, enqueue that same device's next
                    # round-robin prompt before yielding in canonical order.
                    for future_index in range(len(input_records)):
                        worker = future_index % len(units)
                        future = pending_futures[worker]
                        if future is None:
                            raise AssertionError("capture future queue lost a prompt")
                        result = future.result()
                        next_index = future_index + len(units)
                        if next_index < len(input_records):
                            pending_futures[worker] = executors[worker].submit(
                                _forward_one,
                                (next_index, input_records[next_index]),
                            )
                        else:
                            pending_futures[worker] = None
                        yield result
                        del result, future

                iterator = _ordered_results()
            else:
                iterator = (_forward_one(job) for job in enumerate(input_records))
            index = -1
            for record, metadata, output, previous_topk, prompt_capture in iterator:
                index += 1
                if (
                    prompt_capture is not None
                    and writer is not None
                ):
                    moe_hidden, ids, weights = prompt_capture
                    if accumulator is not None:
                        accumulator.add(ids, weights, routed_scaling_factor)
                    rows += writer.append(moe_hidden, ids, weights)
                    del moe_hidden, ids, weights
                if target_state is not None or (memory_state and has_successor):
                    metadata = dict(metadata)
                    metadata.pop("prev_topk_indices", None)
                    metadata["producer"] = (
                        "capture-pass-bf16-source-layers-0-through-"
                        f"{layer}"
                    )
                    metadata["carrier_decode_audit_sha256"] = decode_audit
                    if memory_state:
                        metadata["prev_topk_indices"] = previous_topk
                        successor_records.append(
                            {
                                "shard_id": str(record["shard_id"]),
                                "tokens": int(record["tokens"]),
                                "hidden_tensor": output,
                                "metadata_value": metadata,
                            }
                        )
                    else:
                        successor_records.append(
                            _record_from_runtime(
                                target_state,
                                runtime._write_state(
                                    output_partial=target_state,
                                    layer_input=layer + 1,
                                    shard_id=str(record["shard_id"]),
                                    hidden=output,
                                    previous_topk=previous_topk,
                                    metadata=metadata,
                                ),
                            )
                        )
                del output, previous_topk, prompt_capture
                if index and index % 200 == 0:
                    log(
                        f"layer {layer}: {index}/{len(input_records)} prompts, "
                        f"{rows} captured rows"
                    )
            # The mass audit must pass BEFORE the manifest is sealed: a sealed
            # manifest is what a restart trusts to skip this layer.
            audit = accumulator.finish() if accumulator is not None else None
            if writer is not None:
                if rows != expected_tokens:
                    raise ValueError(
                        f"layer {layer}: captured {rows} rows, the corpus plan "
                        f"declares {expected_tokens}"
                    )
                manifest = writer.finalize()
                assert writer.manifest_path is not None
                captured[layer] = {
                    "tokens": int(manifest["tokens"]),
                    "manifest": str(writer.manifest_path),
                    "manifest_sha256": sha256_file(writer.manifest_path),
                    "content_sha256": str(manifest["content_sha256"]),
                    "verification": dict(verification),
                }
                writer = None
            if audit is not None:
                mass_audits[layer] = audit.digest_without_self()
                atomic_write_json(
                    layer_capture_dir(capture_dir, layer) / "MASS_AUDIT.json",
                    audit.to_json(),
                )
        finally:
            for future in pending_futures:
                if future is not None:
                    future.cancel()
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
            executors.clear()
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            pending_futures.clear()
            if handle is not None:
                handle.remove()
            for replica_handle in replica_handles:
                replica_handle.remove()
            replica_handles.clear()
            if writer is not None:
                writer.abort()
            # Hooks retain taps, taps retain their MLP, and the forwarding
            # closure retains ``units``. Break every path before empty_cache so
            # no full replica survives into construction of the next layer.
            units.clear()
            replicas.clear()
            replica_taps.clear()
            tap = None
            rtap = None
            replica = None
            replica_handle = None
            future = None
            executor = None
            handle = None
            close = None
            iterator = None
            _ordered_results = None
            _forward_one = None
            del module
            torch.cuda.empty_cache()

        if target_state is not None:
            successor_records.sort(key=lambda item: str(item["shard_id"]))
            _seal_state(
                target_state,
                layer=layer + 1,
                bindings=bindings,
                records=successor_records,
                expected=expected_shards,
            )
        if memory_state and has_successor:
            successor_records.sort(key=lambda item: str(item["shard_id"]))
            memory_records = successor_records
        entry: dict[str, object] = {"captured": bool(wants_capture)}
        if layer in captured:
            entry.update(captured[layer])
        if layer in mass_audits:
            entry["mass_audit_digest"] = mass_audits[layer]
        assert isinstance(progress["layers"], dict)
        progress["layers"][f"{layer:03d}"] = entry
        _commit(layer + 1 if has_successor else layer)
        if retire_state and successor_dir is not None and source_state is not None:
            _remove_tree(source_state)
        log(
            f"layer {layer}: done"
            + (f" -- capture {captured[layer]['tokens']} tokens" if layer in captured else "")
        )

    missing = sorted(value for value in requested if value not in captured)
    if missing:
        raise ValueError(f"capture pass finished without layers {missing}")
    summary = {
        "schema": SCHEMA,
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "capture_dir": str(capture_dir),
        "bindings": bindings,
        "verification": verification,
        "device": str(device),
        "prompts": len(selected),
        "tokens": expected_tokens,
        "layers": [int(value) for value in requested],
        "captures": {f"{key:03d}": captured[key] for key in sorted(captured)},
        "mass_audit_digests": {
            f"{key:03d}": mass_audits[key] for key in sorted(mass_audits)
        },
        "expert_tensors_payload_verified": experts.verified_tensors,
        "progress": str(progress_file),
    }
    atomic_write_json(capture_dir / "CAPTURE_PASS_SUMMARY.json", summary)
    log(
        f"capture pass complete: {len(captured)} layers x {expected_tokens} tokens "
        f"-> {capture_dir}"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m r7_encoder.capture_pass",
        description=(
            "One streaming pass over the unquantized GLM-5.2 source that writes a "
            "flat routed-expert capture for every requested MoE layer."
        ),
    )
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True, dest="src")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--carrier-inventory", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--numeric-inventory", type=Path, required=True)
    parser.add_argument("--runtime-inventory", type=Path, required=True)
    parser.add_argument("--layers", default="3-77")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-devices", default="",
        help="comma-separated device pool for prompt-sharded capture")
    parser.add_argument("--sigma-reg", type=float, default=DEFAULT_SIGMA_REG)
    parser.add_argument(
        "--no-verify-expert-payloads",
        action="store_true",
        help=(
            "skip the per-tensor SHA-256 of the BF16 routed experts; the inventory "
            "dtype/shape/offset checks still run"
        ),
    )
    parser.add_argument(
        "--verify-resume-payloads",
        action="store_true",
        help="re-hash an already finalized capture before skipping its layer",
    )
    parser.add_argument(
        "--keep-states",
        action="store_true",
        help="retain every rolling per-layer state directory instead of retiring it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_capture(
        carrier=args.carrier,
        bf16_source=args.src,
        corpus=args.corpus,
        capture_dir=args.capture_dir,
        carrier_inventory=args.carrier_inventory,
        source_inventory=args.source_inventory,
        numeric_inventory=args.numeric_inventory,
        runtime_inventory=args.runtime_inventory,
        layers=args.layers,
        device=args.device,
        capture_devices=tuple(p.strip() for p in args.capture_devices.split(',') if p.strip()),
        sigma_reg=args.sigma_reg,
        verify_expert_payloads=not args.no_verify_expert_payloads,
        verify_resume_payloads=args.verify_resume_payloads,
        retire_state=not args.keep_states,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
