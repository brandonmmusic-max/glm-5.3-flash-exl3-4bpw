"""Sealed GLM5Next layer-45 NextN calibration primitives.

Layer 45 is not a 46th mHC target-model layer.  The released checkpoint stores
one classic-residual NextN head: shifted token embedding and target terminal
hidden fusion, a standalone DSA attention block, then the MTP MoE.  This module
defines that source-key ABI and a crash-atomic single-layer capture journal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from .glm53_capture import (
    CAPTURE_SCHEMA as MAIN_CAPTURE_SCHEMA,
    HIDDEN_SIZE,
    MTP_LAYER,
    NUM_EXPERTS,
    TOP_K,
    CapturedLayerWindow,
    expected_layer_bytes,
    layer_paths,
    verify_seal,
)


MTP_PROGRESS_SCHEMA = "quant-pipeline.glm53-mtp45-calibration-progress.v1"
MTP_CAPTURE_SCHEMA = "quant-pipeline.glm53-mtp45-calibration-capture.v1"
MTP_RECEIPT_SCHEMA = "quant-pipeline.glm53-mtp45-calibration-capture-receipt.v1"
MAIN_RECEIPT_SCHEMA = "quant-pipeline.glm53-main-calibration-capture-receipt.v1"
LAYER_PREFIX = "model.language_model.layers.45"
EMBEDDING_NAME = "model.language_model.embed_tokens.weight"


MTP45_NONEXPERT_NAMES = frozenset(
    {
        f"{LAYER_PREFIX}.eh_proj.weight",
        f"{LAYER_PREFIX}.enorm.weight",
        f"{LAYER_PREFIX}.hnorm.weight",
        f"{LAYER_PREFIX}.input_layernorm.weight",
        f"{LAYER_PREFIX}.post_attention_layernorm.weight",
        f"{LAYER_PREFIX}.mlp.gate.weight",
        f"{LAYER_PREFIX}.mlp.gate.e_score_correction_bias",
        f"{LAYER_PREFIX}.mlp.shared_experts.gate_proj.weight",
        f"{LAYER_PREFIX}.mlp.shared_experts.up_proj.weight",
        f"{LAYER_PREFIX}.mlp.shared_experts.down_proj.weight",
        f"{LAYER_PREFIX}.self_attn.q_a_proj.weight",
        f"{LAYER_PREFIX}.self_attn.q_a_layernorm.weight",
        f"{LAYER_PREFIX}.self_attn.q_b_proj.weight",
        f"{LAYER_PREFIX}.self_attn.kv_a_proj_with_mqa.weight",
        f"{LAYER_PREFIX}.self_attn.kv_a_layernorm.weight",
        f"{LAYER_PREFIX}.self_attn.kv_b_proj.weight",
        f"{LAYER_PREFIX}.self_attn.o_proj.weight",
        f"{LAYER_PREFIX}.self_attn.indexer.wq_b.weight",
        f"{LAYER_PREFIX}.self_attn.indexer.wk.weight",
        f"{LAYER_PREFIX}.self_attn.indexer.k_norm.weight",
        f"{LAYER_PREFIX}.self_attn.indexer.k_norm.bias",
        f"{LAYER_PREFIX}.self_attn.indexer.weights_proj.weight",
        f"{LAYER_PREFIX}.self_attn.indexer.index_kpool_compress_ape",
        f"{LAYER_PREFIX}.self_attn.indexer.index_kpool_compress_gate",
        f"{LAYER_PREFIX}.shared_head.norm.weight",
    }
)


def mtp45_expert_names() -> frozenset[str]:
    return frozenset(
        f"{LAYER_PREFIX}.mlp.experts.{expert}.{projection}.weight"
        for expert in range(NUM_EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )


def mtp45_prefix_names() -> frozenset[str]:
    """Tensors evaluated before the routed experts during calibration."""

    return frozenset(
        {EMBEDDING_NAME}
        | {
            name
            for name in MTP45_NONEXPERT_NAMES
            if ".mlp.shared_experts." not in name and ".shared_head." not in name
        }
    )


def validate_mtp45_inventory(
    inventory: Mapping[str, Any], *, model_root: Path, revision: str
) -> dict[str, dict[str, Any]]:
    """Fail closed against the exact 889-tensor released layer-45 surface."""

    body = dict(inventory)
    seal = body.pop("inventory_sha256", None)
    rows = inventory.get("tensors")
    if (
        inventory.get("schema") != "quant-pipeline.glm-release-inventory.v1"
        or seal != sha256_bytes(canonical_json(body))
        or inventory.get("seal_mode") != "full-shard-sha256"
        or inventory.get("model_revision") != revision
        or Path(str(inventory.get("checkpoint", ""))).resolve() != model_root.resolve()
        or not isinstance(rows, list)
    ):
        raise ValueError("MTP45 requires a sealed full-payload official inventory")
    entries = {
        str(row.get("tensor_name")): dict(row)
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("tensor_name"), str)
    }
    if len(entries) != len(rows):
        raise ValueError("inventory contains malformed or duplicate tensor names")
    actual_layer = {name for name in entries if name.startswith(LAYER_PREFIX + ".")}
    expected_layer = MTP45_NONEXPERT_NAMES | mtp45_expert_names()
    if actual_layer != expected_layer:
        missing = sorted(expected_layer - actual_layer)
        extra = sorted(actual_layer - expected_layer)
        raise ValueError(
            f"released MTP45 key inventory drifted: {len(missing)} missing, {len(extra)} extra"
        )
    if len(actual_layer) != 889 or EMBEDDING_NAME not in entries:
        raise ValueError("released MTP45 tensor count or shared embedding is absent")
    for name in actual_layer | {EMBEDDING_NAME}:
        row = entries[name]
        if (
            row.get("dtype") not in {"BF16", "F32"}
            or not isinstance(row.get("shape"), list)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_payload_sha256"))) is None
            or not isinstance(row.get("shard"), str)
        ):
            raise ValueError(f"MTP45 source tensor lacks BF16/F32 payload attestation: {name}")
        if name.endswith(".weight") and not name.endswith("k_norm.weight") and row["dtype"] != "BF16":
            raise ValueError(f"released MTP45 matrix is not BF16: {name}")
    return entries


@dataclass(frozen=True)
class MainTerminalCapture:
    receipt: dict[str, Any]
    manifest: dict[str, Any]
    hidden_path: Path
    windows: tuple[dict[str, Any], ...]


def load_main_terminal_capture(
    receipt_path: str | Path,
    *,
    revision: str,
    inventory_sha256: str,
    token_panel_receipt_sha256: str,
) -> MainTerminalCapture:
    """Verify and expose the main teacher's post-mHC terminal stream."""

    receipt_path = Path(receipt_path).resolve()
    receipt = verify_seal(
        json.loads(receipt_path.read_text()),
        schema=MAIN_RECEIPT_SCHEMA,
        field="receipt_sha256",
    )
    manifest_path = receipt_path.parent / "capture-manifest.json"
    if (
        receipt.get("complete") is not True
        or receipt.get("model_revision_receipt_sha256") is None
        or receipt.get("inventory_sha256") != inventory_sha256
        or receipt.get("token_panel_receipt_sha256") != token_panel_receipt_sha256
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or sha256_file(manifest_path) != receipt.get("capture_manifest_file_sha256")
    ):
        raise ValueError("main calibration receipt is incomplete or belongs to different inputs")
    manifest = verify_seal(
        json.loads(manifest_path.read_text()),
        schema=MAIN_CAPTURE_SCHEMA,
        field="capture_sha256",
    )
    artifact = manifest.get("terminal_last_hidden")
    windows = manifest.get("windows")
    rows = manifest.get("rows_per_layer")
    if (
        manifest.get("model_revision") != revision
        or manifest.get("inventory_sha256") != inventory_sha256
        or manifest.get("token_panel_receipt_sha256") != token_panel_receipt_sha256
        or manifest.get("capture_sha256") != receipt.get("capture_sha256")
        or not isinstance(artifact, dict)
        or set(artifact) != {"path", "bytes", "sha256"}
        or not isinstance(windows, list)
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or sum(int(window.get("rows", -1)) for window in windows) != rows
    ):
        raise ValueError("main terminal-state manifest is malformed or not source-bound")
    hidden_path = (receipt_path.parent / str(artifact["path"])).resolve()
    expected_bytes = rows * HIDDEN_SIZE * 2
    if (
        receipt_path.parent not in hidden_path.parents
        or not hidden_path.is_file()
        or hidden_path.is_symlink()
        or int(artifact["bytes"]) != expected_bytes
        or hidden_path.stat().st_size != expected_bytes
        or sha256_file(hidden_path) != artifact["sha256"]
        or receipt.get("terminal_last_hidden_sha256") != artifact["sha256"]
    ):
        raise ValueError("main terminal hidden-state artifact identity differs")
    return MainTerminalCapture(receipt, manifest, hidden_path, tuple(dict(row) for row in windows))


def terminal_window_offsets(windows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for row in windows:
        identity = str(row.get("window_id", ""))
        count = row.get("rows")
        if (
            not identity
            or identity in offsets
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 2
        ):
            raise ValueError("terminal-state window journal is malformed")
        offsets[identity] = (cursor, count)
        cursor += count
    return offsets


def read_terminal_window(path: Path, *, offset: int, rows: int) -> Any:
    """Read BF16 raw bits without any numerical conversion."""

    import numpy as np
    import torch

    if offset < 0 or rows < 2:
        raise ValueError("terminal-state window extent is invalid")
    words = np.memmap(path, mode="r", dtype="<u2", shape=(path.stat().st_size // 2,))
    start = offset * HIDDEN_SIZE
    stop = (offset + rows) * HIDDEN_SIZE
    if stop > words.size:
        raise ValueError("terminal-state window extends past the sealed payload")
    owned = np.array(words[start:stop], dtype=np.uint16, copy=True)
    return torch.from_numpy(owned).view(torch.bfloat16).reshape(rows, HIDDEN_SIZE)


def transition_mask(mask: Any) -> Any:
    import torch

    value = torch.as_tensor(mask)
    if value.ndim != 1 or value.numel() < 2 or not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("MTP45 attention mask must be a binary rank-one vector")
    result = value[:-1].bool() & value[1:].bool()
    if not bool(result.any()):
        raise ValueError("MTP45 window has no valid teacher-forced transition")
    return result


class Mtp45CaptureStore:
    """Crash-atomic flat journal for the single routed MTP layer."""

    def __init__(self, root: str | Path, *, run_identity_sha256: str, resume: bool) -> None:
        self.root = Path(root)
        self.identity = run_identity_sha256
        self.progress_path = self.root / "progress.json"
        if re.fullmatch(r"[0-9a-f]{64}", self.identity) is None:
            raise ValueError("run identity must be a SHA-256 digest")
        if resume:
            self.progress = self._load()
            self._truncate()
        else:
            if self.progress_path.exists() or (self.root / "layers").exists():
                raise FileExistsError("MTP45 capture output already exists; use --resume")
            for path in layer_paths(self.root, MTP_LAYER).values():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb"):
                    pass
            self.progress = {
                "schema": MTP_PROGRESS_SCHEMA,
                "run_identity_sha256": self.identity,
                "layer": MTP_LAYER,
                "committed_windows": [],
                "committed_rows": 0,
                "route_counts": [0] * NUM_EXPERTS,
                "applied_weight_sum": 0.0,
                "router_cross_checked": False,
            }
            self._write()

    def _write(self) -> None:
        body = dict(self.progress)
        body.pop("progress_sha256", None)
        body["progress_sha256"] = sha256_bytes(canonical_json(body))
        self.progress = body
        write_json(self.progress_path, body)

    def _load(self) -> dict[str, Any]:
        payload = verify_seal(
            json.loads(self.progress_path.read_text()),
            schema=MTP_PROGRESS_SCHEMA,
            field="progress_sha256",
        )
        windows = payload.get("committed_windows")
        rows = payload.get("committed_rows")
        counts = payload.get("route_counts")
        if (
            payload.get("run_identity_sha256") != self.identity
            or payload.get("layer") != MTP_LAYER
            or not isinstance(windows, list)
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or sum(int(row.get("rows", -1)) for row in windows) != rows
            or not isinstance(counts, list)
            or len(counts) != NUM_EXPERTS
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts)
            or sum(counts) != rows * TOP_K
            or not math.isfinite(float(payload.get("applied_weight_sum", -1)))
        ):
            raise ValueError("MTP45 resume progress is malformed")
        return payload

    def _truncate(self) -> None:
        expected = expected_layer_bytes(int(self.progress["committed_rows"]))
        for key, path in layer_paths(self.root, MTP_LAYER).items():
            if not path.is_file() or path.is_symlink() or path.stat().st_size < expected[key]:
                raise ValueError("MTP45 capture is shorter than its committed boundary")
            if path.stat().st_size > expected[key]:
                descriptor = os.open(path, os.O_WRONLY)
                try:
                    os.ftruncate(descriptor, expected[key])
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

    @property
    def committed_windows(self) -> int:
        return len(self.progress["committed_windows"])

    def append_window(self, capture: CapturedLayerWindow, *, window: Mapping[str, Any]) -> None:
        expected = expected_layer_bytes(capture.rows)
        payloads = {
            "hidden_bf16": capture.hidden_bf16,
            "topk_ids_u16le": capture.topk_ids_u16le,
            "topk_weights_f32le": capture.topk_weights_f32le,
        }
        if capture.rows <= 0 or {key: len(value) for key, value in payloads.items()} != expected:
            raise ValueError("MTP45 capture byte geometry differs")
        handles = []
        try:
            for key, path in layer_paths(self.root, MTP_LAYER).items():
                handle = path.open("ab", buffering=16 << 20)
                handles.append(handle)
                handle.write(payloads[key])
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in handles:
                handle.close()
        body = dict(self.progress)
        body.pop("progress_sha256", None)
        body["committed_rows"] = int(body["committed_rows"]) + capture.rows
        body["committed_windows"] = list(body["committed_windows"]) + [dict(window) | {"rows": capture.rows}]
        body["route_counts"] = [
            int(left) + int(right) for left, right in zip(body["route_counts"], capture.route_counts)
        ]
        body["applied_weight_sum"] = float(body["applied_weight_sum"]) + capture.applied_weight_sum
        body["router_cross_checked"] = bool(body["router_cross_checked"] or capture.router_cross_checked)
        self.progress = body
        self._write()
