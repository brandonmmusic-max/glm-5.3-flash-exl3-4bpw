from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.artifacts import canonical_json, sha256_bytes, sha256_file


PANEL_RECEIPT_SCHEMA = "quant-pipeline.glm53-token-panel-receipt.v1"
PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"
CAPTURE_SCHEMA = "quant-pipeline.glm53-logit-capture.v1"
KLD_SCHEMA = "quant-pipeline.glm53-packed-student-kld.v1"


@dataclass(frozen=True)
class PanelWindow:
    window_id: str
    document_id: str
    domain: str
    role: str
    token_path: Path
    attention_mask_path: Path
    token_sha256: str
    attention_mask_sha256: str
    prediction_positions: int


def sealed_json(path: str | Path, schema: str, seal_field: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    body = dict(payload)
    seal = body.pop(seal_field, None)
    if (
        not isinstance(payload, dict)
        or payload.get("example_only") is True
        or payload.get("schema") != schema
        or seal != sha256_bytes(canonical_json(body))
    ):
        raise ValueError(f"invalid sealed {schema} artifact: {path}")
    return payload


def verified_artifacts(rows: Any) -> dict[str, Path]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("receipt must contain a nonempty artifacts list")
    result: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("artifact rows must contain exactly path, bytes, and sha256")
        path = Path(row["path"])
        digest = row["sha256"]
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != digest
        ):
            raise ValueError(f"artifact identity mismatch: {path}")
        if digest in result:
            raise ValueError(f"duplicate artifact digest: {digest}")
        result[digest] = path
    return result


def _integer_vector(path: Path, *, label: str, dtypes: tuple[np.dtype, ...]) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"{label} must be an uncompressed .npy array")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 1 or values.dtype not in dtypes or values.size < 2:
        raise ValueError(f"{label} must be a rank-1 integer vector with at least two entries")
    return values


def load_panel_windows(
    receipt_path: str | Path,
    *,
    roles: Iterable[str] = ("final",),
    vocab_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[PanelWindow]]:
    receipt = sealed_json(receipt_path, PANEL_RECEIPT_SCHEMA, "receipt_sha256")
    artifacts = verified_artifacts(receipt.get("artifacts"))
    panel_digest = receipt.get("token_panel_artifact_sha256")
    if panel_digest not in artifacts:
        raise ValueError("token-panel receipt does not resolve its panel artifact")
    panel = json.loads(artifacts[panel_digest].read_text())
    if not isinstance(panel, dict) or panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("token-panel artifact has the wrong schema")
    requested = frozenset(str(role) for role in roles)
    if not requested:
        raise ValueError("at least one token-panel role is required")
    windows: list[PanelWindow] = []
    for row in panel.get("windows", ()):
        if not isinstance(row, dict) or row.get("role") not in requested:
            continue
        token_digest = row.get("token_ids_sha256")
        mask_digest = row.get("attention_mask_sha256")
        if token_digest not in artifacts or mask_digest not in artifacts:
            raise ValueError("panel window token or mask artifact is absent")
        tokens = _integer_vector(
            artifacts[token_digest],
            label="token IDs",
            dtypes=(np.dtype("int32"), np.dtype("int64")),
        )
        mask = _integer_vector(
            artifacts[mask_digest],
            label="attention mask",
            dtypes=(np.dtype("uint8"), np.dtype("bool"), np.dtype("int32"), np.dtype("int64")),
        )
        if tokens.shape != mask.shape or not np.isin(mask, (0, 1)).all():
            raise ValueError("panel window token IDs and binary mask are misaligned")
        if vocab_size is not None and (int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size):
            raise ValueError("panel token ID is outside the released vocabulary")
        prediction_mask = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
        prediction_positions = int(prediction_mask.sum())
        if prediction_positions <= 0 or row.get("prediction_positions") != prediction_positions:
            raise ValueError("panel prediction-position count differs from the exact causal mask")
        windows.append(
            PanelWindow(
                window_id=str(row["window_id"]),
                document_id=str(row["document_id"]),
                domain=str(row["domain"]),
                role=str(row["role"]),
                token_path=artifacts[token_digest],
                attention_mask_path=artifacts[mask_digest],
                token_sha256=token_digest,
                attention_mask_sha256=mask_digest,
                prediction_positions=prediction_positions,
            )
        )
    if not windows:
        raise ValueError(f"token panel has no windows for roles {sorted(requested)}")
    identities = [window.window_id for window in windows]
    if len(identities) != len(set(identities)):
        raise ValueError("token-panel window identities are not unique")
    return receipt, panel, windows


def load_capture_receipt(path: str | Path, *, expected_role: str | None = None) -> dict[str, Any]:
    receipt = sealed_json(path, CAPTURE_SCHEMA, "receipt_sha256")
    if expected_role is not None and receipt.get("capture_role") != expected_role:
        raise ValueError(f"expected {expected_role} capture, got {receipt.get('capture_role')}")
    files = receipt.get("logit_files")
    if not isinstance(files, list) or not files:
        raise ValueError("capture receipt has no logit files")
    seen: set[str] = set()
    for row in files:
        required = {
            "window_id",
            "document_id",
            "domain",
            "role",
            "token_ids_sha256",
            "attention_mask_sha256",
            "prediction_positions",
            "path",
            "bytes",
            "sha256",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("logit-file record has unexpected fields")
        logit_path = Path(row["path"])
        if (
            not logit_path.is_absolute()
            or not logit_path.is_file()
            or logit_path.is_symlink()
            or logit_path.stat().st_size != int(row["bytes"])
            or sha256_file(logit_path) != row["sha256"]
        ):
            raise ValueError(f"logit artifact identity mismatch: {logit_path}")
        if row["window_id"] in seen:
            raise ValueError("capture contains a duplicate window")
        seen.add(row["window_id"])
    return receipt


def token_kld_chunk(teacher: np.ndarray, student: np.ndarray) -> np.ndarray:
    if teacher.shape != student.shape or teacher.ndim != 2 or teacher.shape[1] <= 1:
        raise ValueError("teacher/student logit geometry mismatch")
    teacher64 = np.asarray(teacher, dtype=np.float64)
    student64 = np.asarray(student, dtype=np.float64)
    if not np.isfinite(teacher64).all() or not np.isfinite(student64).all():
        raise ValueError("teacher/student logits must be finite")
    teacher64 -= np.max(teacher64, axis=-1, keepdims=True)
    student64 -= np.max(student64, axis=-1, keepdims=True)
    teacher64 -= np.logaddexp.reduce(teacher64, axis=-1, keepdims=True)
    student64 -= np.logaddexp.reduce(student64, axis=-1, keepdims=True)
    return np.sum(np.exp(teacher64) * (teacher64 - student64), axis=-1)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("cannot summarize empty or non-finite KLD values")
    ordered = np.sort(vector)
    tail = ordered[max(0, int(np.floor(0.95 * ordered.size))) :]
    return {
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "std": float(vector.std(ddof=1)) if vector.size > 1 else 0.0,
        "p50": float(np.quantile(vector, 0.50)),
        "p95": float(np.quantile(vector, 0.95)),
        "p99": float(np.quantile(vector, 0.99)),
        "cvar95": float(tail.mean()),
        "max": float(ordered[-1]),
    }
