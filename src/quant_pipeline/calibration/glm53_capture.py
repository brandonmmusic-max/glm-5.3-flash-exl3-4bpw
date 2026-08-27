"""Crash-safe flat capture primitives for the released GLM5Next main model.

This module deliberately contains no model loader.  The torchrun entrypoint owns
the official Transformers forward while these helpers own the byte ABI,
independent router oracle, and atomic window journal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..core.artifacts import canonical_json, sha256_bytes, write_json


PROGRESS_SCHEMA = "quant-pipeline.glm53-main-calibration-progress.v1"
CAPTURE_SCHEMA = "quant-pipeline.glm53-main-calibration-capture.v1"
RECEIPT_SCHEMA = "quant-pipeline.glm53-main-calibration-capture-receipt.v1"
HIDDEN_SIZE = 4_096
NUM_EXPERTS = 288
TOP_K = 8
MAIN_ROUTED_LAYERS = tuple(range(3, 45))
MTP_LAYER = 45


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = sha256_bytes(canonical_json(result))
    return result


def verify_seal(payload: Mapping[str, Any], *, schema: str, field: str) -> dict[str, Any]:
    body = dict(payload)
    seal = body.pop(field, None)
    if payload.get("schema") != schema or seal != sha256_bytes(canonical_json(body)):
        raise ValueError(f"invalid sealed {schema}")
    return dict(payload)


def independent_router(
    hidden: Any,
    weight: Any,
    correction_bias: Any,
    *,
    top_k: int,
    num_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> tuple[Any, Any, Any]:
    """Recompute the released GLM5Next router without calling its forward."""

    import torch
    import torch.nn.functional as functional

    hidden = torch.as_tensor(hidden)
    weight = torch.as_tensor(weight, device=hidden.device)
    correction_bias = torch.as_tensor(correction_bias, device=hidden.device)
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("router hidden/weight geometry mismatch")
    experts = int(weight.shape[0])
    if (
        correction_bias.shape != (experts,)
        or top_k <= 0
        or top_k > experts
        or num_group <= 0
        or experts % num_group
        or topk_group <= 0
        or topk_group > num_group
    ):
        raise ValueError("router configuration is invalid")
    logits = functional.linear(hidden.float(), weight.float())
    scores = logits.sigmoid()
    scores_for_choice = scores + correction_bias.float()
    group_scores = (
        scores_for_choice.view(-1, num_group, experts // num_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_index = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_index, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, num_group, experts // num_group)
        .reshape(-1, experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    applied = scores.gather(1, indices)
    if norm_topk_prob:
        applied = applied / (applied.sum(dim=-1, keepdim=True) + 1e-20)
    applied = applied * float(routed_scaling_factor)
    return logits, applied, indices


def local_tensor(value: Any) -> Any:
    """Return the local payload of a DTensor, or the tensor unchanged."""

    return value.to_local() if hasattr(value, "to_local") else value


def reconstruct_ep4_routes(
    rank_ids: Sequence[Any], rank_weights: Sequence[Any]
) -> tuple[Any, Any]:
    """Undo Transformers ``EpRouterParallel`` masking/remapping.

    The released EP adapter leaves exactly one owner for each routed slot.  On
    rank ``r`` its local expert IDs are ``global_id % 72`` and every non-owner
    receives the sentinel ID 72 with a zero applied weight.
    """

    import torch

    if len(rank_ids) != 4 or len(rank_weights) != 4:
        raise ValueError("GLM5Next route reconstruction requires exactly four EP ranks")
    ids = [local_tensor(value).detach().to(torch.int64) for value in rank_ids]
    weights = [local_tensor(value).detach().to(torch.float32) for value in rank_weights]
    shape = ids[0].shape
    if len(shape) != 2 or shape[-1] != TOP_K:
        raise ValueError("EP route tensors must be [rows,8]")
    if any(value.shape != shape for value in ids + weights):
        raise ValueError("EP ranks disagree on route tensor geometry")
    local_experts = NUM_EXPERTS // 4
    owners = torch.stack([value < local_experts for value in ids], dim=0)
    if not torch.equal(owners.sum(dim=0), torch.ones(shape, dtype=torch.int64, device=ids[0].device)):
        raise RuntimeError("EP router did not leave exactly one owner per routed slot")
    for rank, (rank_ids_value, rank_weights_value) in enumerate(zip(ids, weights)):
        if bool(((~owners[rank]) & (rank_ids_value != local_experts)).any()):
            raise RuntimeError("EP router non-owner ID is not the expected sentinel")
        if bool(((~owners[rank]) & (rank_weights_value != 0)).any()):
            raise RuntimeError("EP router non-owner retained a nonzero applied weight")
    global_ids = torch.zeros_like(ids[0])
    global_weights = torch.zeros_like(weights[0])
    for rank, (rank_ids_value, rank_weights_value) in enumerate(zip(ids, weights)):
        global_ids = torch.where(
            owners[rank], rank_ids_value + rank * local_experts, global_ids
        )
        global_weights = global_weights + rank_weights_value
    if int(global_ids.min()) < 0 or int(global_ids.max()) >= NUM_EXPERTS:
        raise RuntimeError("reconstructed global expert ID is out of range")
    return global_ids, global_weights


@dataclass(frozen=True)
class CapturedLayerWindow:
    rows: int
    hidden_bf16: bytes
    topk_ids_u16le: bytes
    topk_weights_f32le: bytes
    route_counts: tuple[int, ...]
    applied_weight_sum: float
    router_cross_checked: bool


def captured_window_from_tensors(
    hidden: Any,
    ids: Any,
    weights: Any,
    *,
    router: Any | None,
    crosscheck: bool,
    geometry: Mapping[str, Any],
) -> CapturedLayerWindow:
    """Validate and serialize the exact tensors entering one expert collection."""

    import torch

    hidden = local_tensor(hidden).detach()
    ids = local_tensor(ids).detach()
    weights = local_tensor(weights).detach()
    if hidden.ndim != 2 or tuple(hidden.shape[1:]) != (HIDDEN_SIZE,):
        raise ValueError(f"GLM5Next expert input must be [rows,{HIDDEN_SIZE}]")
    rows = int(hidden.shape[0])
    if rows <= 0 or ids.shape != (rows, TOP_K) or weights.shape != (rows, TOP_K):
        raise ValueError("GLM5Next expert routing tensors have the wrong shape")
    if hidden.dtype != torch.bfloat16:
        raise TypeError("pre-expert hidden states must already be BF16; conversion is forbidden")
    if weights.dtype != torch.float32:
        raise TypeError("applied router weights must already be float32")
    ids64 = ids.to(torch.int64)
    if int(ids64.min()) < 0 or int(ids64.max()) >= NUM_EXPERTS:
        raise ValueError("routed expert ID is outside [0,288)")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("applied router weights must be finite and nonnegative")
    expected_mass = torch.full(
        (rows,),
        float(geometry["routed_scaling_factor"]),
        dtype=torch.float32,
        device=weights.device,
    )
    if not torch.allclose(weights.sum(dim=-1), expected_mass, rtol=1e-6, atol=1e-6):
        raise ValueError("applied router weights do not sum to the routed scaling factor")
    if bool((ids64.sort(dim=-1).values[:, 1:] == ids64.sort(dim=-1).values[:, :-1]).any()):
        raise ValueError("a token routes to the same expert more than once")

    checked = False
    if crosscheck:
        if router is None:
            raise ValueError("first-window router cross-check requires the official router module")
        router_weight = local_tensor(router.weight).detach()
        correction = local_tensor(router.e_score_correction_bias).detach()
        expected_geometry = (NUM_EXPERTS, HIDDEN_SIZE)
        if tuple(router_weight.shape) != expected_geometry or tuple(correction.shape) != (NUM_EXPERTS,):
            raise ValueError("official router parameters have unexpected geometry")
        _, expected_weights, expected_ids = independent_router(
            hidden,
            router_weight,
            correction,
            top_k=int(geometry["top_k"]),
            num_group=int(geometry["n_group"]),
            topk_group=int(geometry["topk_group"]),
            norm_topk_prob=bool(geometry["norm_topk_prob"]),
            routed_scaling_factor=float(geometry["routed_scaling_factor"]),
        )
        actual_order = ids64.argsort(dim=-1)
        expected_order = expected_ids.to(torch.int64).argsort(dim=-1)
        actual_ids = ids64.gather(1, actual_order)
        expected_ids = expected_ids.to(torch.int64).gather(1, expected_order)
        actual_weights = weights.gather(1, actual_order)
        expected_weights = expected_weights.gather(1, expected_order)
        if not torch.equal(actual_ids, expected_ids):
            raise RuntimeError("independent GLM5Next router ID cross-check failed")
        if not torch.allclose(actual_weights, expected_weights, rtol=1e-6, atol=1e-7):
            maximum = float((actual_weights - expected_weights).abs().max())
            raise RuntimeError(f"independent GLM5Next applied-weight cross-check failed: {maximum}")
        checked = True

    hidden_words = (
        hidden.to("cpu").contiguous().view(torch.uint16).numpy().astype("<u2", copy=False)
    )
    ids_words = ids64.to("cpu", dtype=torch.int32).numpy().astype("<u2", copy=True)
    weight_words = weights.to("cpu").contiguous().numpy().astype("<f4", copy=False)
    counts = torch.bincount(ids64.flatten(), minlength=NUM_EXPERTS).to("cpu").tolist()
    return CapturedLayerWindow(
        rows=rows,
        hidden_bf16=hidden_words.tobytes(order="C"),
        topk_ids_u16le=ids_words.tobytes(order="C"),
        topk_weights_f32le=weight_words.tobytes(order="C"),
        route_counts=tuple(int(value) for value in counts),
        applied_weight_sum=float(weights.double().sum().item()),
        router_cross_checked=checked,
    )


def layer_paths(root: Path, layer: int) -> dict[str, Path]:
    directory = root / "layers" / f"layer-{layer:03d}"
    return {
        "hidden_bf16": directory / "hidden.bf16.bin",
        "topk_ids_u16le": directory / "topk_ids.u16le.bin",
        "topk_weights_f32le": directory / "topk_weights.f32le.bin",
    }


def expected_layer_bytes(rows: int) -> dict[str, int]:
    if rows < 0:
        raise ValueError("committed row count cannot be negative")
    return {
        "hidden_bf16": rows * HIDDEN_SIZE * 2,
        "topk_ids_u16le": rows * TOP_K * 2,
        "topk_weights_f32le": rows * TOP_K * 4,
    }


def terminal_hidden_path(root: Path) -> Path:
    """Path of the official post-mHC-head/post-final-norm teacher state."""

    return root / "terminal" / "last_hidden.bf16.bin"


def expected_terminal_hidden_bytes(rows: int) -> int:
    if rows < 0:
        raise ValueError("committed row count cannot be negative")
    return rows * HIDDEN_SIZE * 2


class FlatCaptureStore:
    """Append all layers for one window, then atomically advance progress."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_identity_sha256: str,
        layers: Sequence[int] = MAIN_ROUTED_LAYERS,
        resume: bool,
    ) -> None:
        self.root = Path(root)
        self.layers = tuple(int(layer) for layer in layers)
        if self.layers != MAIN_ROUTED_LAYERS:
            raise ValueError("main capture is pinned to routed layers 3..44; MTP45 is separate")
        if re.fullmatch(r"[0-9a-f]{64}", run_identity_sha256) is None:
            raise ValueError("run identity must be a SHA-256 digest")
        self.run_identity_sha256 = run_identity_sha256
        self.progress_path = self.root / "progress.json"
        if resume:
            self.progress = self._load_progress()
            self._truncate_to_commit()
        else:
            if (
                self.progress_path.exists()
                or (self.root / "layers").exists()
                or (self.root / "terminal").exists()
            ):
                raise FileExistsError("capture output already has progress; use --resume")
            for layer in self.layers:
                for path in layer_paths(self.root, layer).values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb"):
                        pass
            terminal = terminal_hidden_path(self.root)
            terminal.parent.mkdir(parents=True, exist_ok=True)
            with terminal.open("xb"):
                pass
            self.progress = self._new_progress()
            self._write_progress()

    def _new_progress(self) -> dict[str, Any]:
        return {
            "schema": PROGRESS_SCHEMA,
            "run_identity_sha256": self.run_identity_sha256,
            "layers": list(self.layers),
            "committed_windows": [],
            "committed_rows": 0,
            "route_counts": {str(layer): [0] * NUM_EXPERTS for layer in self.layers},
            "applied_weight_sum": {str(layer): 0.0 for layer in self.layers},
            "router_cross_checked_layers": [],
        }

    def _load_progress(self) -> dict[str, Any]:
        if not self.progress_path.is_file() or self.progress_path.is_symlink():
            raise FileNotFoundError("resume requires a regular progress.json")
        payload = verify_seal(
            json.loads(self.progress_path.read_text()),
            schema=PROGRESS_SCHEMA,
            field="progress_sha256",
        )
        if payload.get("run_identity_sha256") != self.run_identity_sha256:
            raise ValueError("resume run identity differs from progress")
        if payload.get("layers") != list(self.layers):
            raise ValueError("resume layer domain differs from progress")
        rows = payload.get("committed_rows")
        windows = payload.get("committed_windows")
        counts = payload.get("route_counts")
        sums = payload.get("applied_weight_sum")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(windows, list)
            or any(
                not isinstance(row, dict)
                or isinstance(row.get("rows"), bool)
                or not isinstance(row.get("rows"), int)
                or int(row["rows"]) <= 0
                for row in windows
            )
            or not isinstance(counts, dict)
            or set(counts) != {str(layer) for layer in self.layers}
            or any(not isinstance(value, list) or len(value) != NUM_EXPERTS for value in counts.values())
            or not isinstance(sums, dict)
            or set(sums) != set(counts)
        ):
            raise ValueError("resume progress geometry is malformed")
        if sum(int(row["rows"]) for row in windows) != rows:
            raise ValueError("resume progress window rows do not close")
        for layer in self.layers:
            values = counts[str(layer)]
            if (
                any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
                or sum(values) != rows * TOP_K
                or not math.isfinite(float(sums[str(layer)]))
                or float(sums[str(layer)]) < 0
            ):
                raise ValueError("resume progress route totals do not close")
        checked = payload.get("router_cross_checked_layers")
        if (
            not isinstance(checked, list)
            or len(checked) != len(set(checked))
            or any(value not in self.layers for value in checked)
        ):
            raise ValueError("resume progress router cross-check domain is malformed")
        return payload

    def _write_progress(self) -> None:
        payload = dict(self.progress)
        payload.pop("progress_sha256", None)
        self.progress = _sealed(payload, "progress_sha256")
        write_json(self.progress_path, self.progress)

    def _truncate_to_commit(self) -> None:
        expected = expected_layer_bytes(int(self.progress["committed_rows"]))
        for layer in self.layers:
            for key, path in layer_paths(self.root, layer).items():
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(f"resume capture payload is absent or unsafe: {path}")
                size = path.stat().st_size
                if size < expected[key]:
                    raise ValueError(f"resume capture payload is shorter than its committed boundary: {path}")
                if size > expected[key]:
                    descriptor = os.open(path, os.O_WRONLY)
                    try:
                        os.ftruncate(descriptor, expected[key])
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        terminal = terminal_hidden_path(self.root)
        terminal_expected = expected_terminal_hidden_bytes(int(self.progress["committed_rows"]))
        if not terminal.is_file() or terminal.is_symlink():
            raise FileNotFoundError(f"resume terminal-state payload is absent or unsafe: {terminal}")
        size = terminal.stat().st_size
        if size < terminal_expected:
            raise ValueError("resume terminal-state payload is shorter than its committed boundary")
        if size > terminal_expected:
            descriptor = os.open(terminal, os.O_WRONLY)
            try:
                os.ftruncate(descriptor, terminal_expected)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @property
    def committed_windows(self) -> int:
        return len(self.progress["committed_windows"])

    def append_window(
        self,
        captures: Mapping[int, CapturedLayerWindow],
        *,
        window: Mapping[str, Any],
        terminal_hidden_bf16: bytes,
    ) -> None:
        if set(captures) != set(self.layers):
            raise ValueError("one atomic window must contain every main routed layer")
        rows = {capture.rows for capture in captures.values()}
        if len(rows) != 1:
            raise ValueError("captured layers disagree on window row count")
        row_count = rows.pop()
        if row_count <= 0:
            raise ValueError("captured window must contain at least one row")
        expected = expected_layer_bytes(row_count)
        if len(terminal_hidden_bf16) != expected_terminal_hidden_bytes(row_count):
            raise ValueError("terminal hidden-state byte geometry differs")
        handles = []
        try:
            for layer in self.layers:
                capture = captures[layer]
                payloads = {
                    "hidden_bf16": capture.hidden_bf16,
                    "topk_ids_u16le": capture.topk_ids_u16le,
                    "topk_weights_f32le": capture.topk_weights_f32le,
                }
                if {key: len(value) for key, value in payloads.items()} != expected:
                    raise ValueError(f"layer {layer} capture byte geometry differs")
                for key, path in layer_paths(self.root, layer).items():
                    handle = path.open("ab", buffering=16 << 20)
                    handles.append(handle)
                    handle.write(payloads[key])
            terminal_handle = terminal_hidden_path(self.root).open("ab", buffering=16 << 20)
            handles.append(terminal_handle)
            terminal_handle.write(terminal_hidden_bf16)
            for handle in handles:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in handles:
                handle.close()

        progress = dict(self.progress)
        progress.pop("progress_sha256", None)
        progress["committed_rows"] = int(progress["committed_rows"]) + row_count
        records = list(progress["committed_windows"])
        records.append(dict(window) | {"rows": row_count})
        progress["committed_windows"] = records
        route_counts = {key: list(value) for key, value in progress["route_counts"].items()}
        weight_sums = dict(progress["applied_weight_sum"])
        checked = set(int(value) for value in progress["router_cross_checked_layers"])
        for layer, capture in captures.items():
            route_counts[str(layer)] = [
                int(left) + int(right)
                for left, right in zip(route_counts[str(layer)], capture.route_counts)
            ]
            weight_sums[str(layer)] = float(weight_sums[str(layer)]) + capture.applied_weight_sum
            if capture.router_cross_checked:
                checked.add(layer)
        progress["route_counts"] = route_counts
        progress["applied_weight_sum"] = weight_sums
        progress["router_cross_checked_layers"] = sorted(checked)
        self.progress = progress
        self._write_progress()
