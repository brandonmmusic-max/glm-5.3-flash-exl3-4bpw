"""Content-addressed storage for exact codec tensors.

The store writes the actual byte sequence used for hashing.  JSON ledgers may
refer to these objects, but never stand in for the packed trellis, transform
vectors, or reconstruction oracle themselves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..core.artifacts import canonical_json, sha256_bytes, sha256_file


SCHEMA_PAYLOAD_REF = "quant-pipeline.exact-payload-ref.v1"
SCHEMA_PAYLOAD_MANIFEST = "quant-pipeline.exact-payload-manifest.v1"


def _tensor_bytes(value: Any) -> bytes:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            return tensor.view(torch.uint8).numpy().tobytes()
    except ImportError:  # pragma: no cover
        pass
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be exact payloads")
    return array.view(np.uint8).tobytes()


def _shape(value: Any) -> list[int]:
    return list(map(int, value.shape))


def _dtype(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ExactPayloadStore:
    """Append-only SHA-256 object store with self-describing tensor refs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def put_tensor(self, value: Any, *, role: str) -> dict[str, Any]:
        if not isinstance(role, str) or not role:
            raise ValueError("payload role must be non-empty")
        raw = _tensor_bytes(value)
        digest = sha256_bytes(raw)
        relative = Path("objects") / digest[:2] / f"{digest}.bin"
        path = self.root / relative
        if path.exists():
            if path.stat().st_size != len(raw) or sha256_file(path) != digest:
                raise RuntimeError(f"content-addressed payload collision or corruption: {digest}")
        else:
            _atomic_write(path, raw)
        return {
            "schema": SCHEMA_PAYLOAD_REF,
            "role": role,
            "sha256": digest,
            "bytes": len(raw),
            "dtype": _dtype(value),
            "shape": _shape(value),
            "path": relative.as_posix(),
        }

    def verify_ref(self, ref: Mapping[str, Any]) -> None:
        if ref.get("schema") != SCHEMA_PAYLOAD_REF:
            raise ValueError("unsupported exact payload reference")
        digest = ref.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("payload reference has invalid SHA-256")
        expected_path = (Path("objects") / digest[:2] / f"{digest}.bin").as_posix()
        if ref.get("path") != expected_path:
            raise ValueError("payload reference path is not content-addressed")
        path = self.root / expected_path
        if not path.is_file() or path.stat().st_size != ref.get("bytes") or sha256_file(path) != digest:
            raise ValueError(f"payload object is missing or corrupt: {digest}")

    def manifest(self, refs: list[Mapping[str, Any]]) -> dict[str, Any]:
        unique: dict[str, dict[str, Any]] = {}
        for raw in refs:
            ref = dict(raw)
            self.verify_ref(ref)
            digest = ref["sha256"]
            # The object is raw bytes. Shape/dtype remain on each typed
            # reference because one byte string can legally have more than one
            # tensor view without being duplicated on disk.
            stable = {key: ref[key] for key in ("sha256", "bytes", "path")}
            incumbent = unique.setdefault(digest, stable)
            if incumbent != stable:
                raise ValueError(f"inconsistent payload descriptors for {digest}")
        objects = [unique[key] for key in sorted(unique)]
        body = {
            "schema": SCHEMA_PAYLOAD_MANIFEST,
            "objects": objects,
            "physical_bytes": sum(row["bytes"] for row in objects),
        }
        body["manifest_sha256"] = sha256_bytes(canonical_json(body))
        manifest_path = self.root / "manifest.json"
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
        if manifest_path.exists():
            # A later ledger can legitimately extend the same append-only store.
            if json.loads(manifest_path.read_text()) != body:
                _atomic_write(manifest_path, encoded)
        else:
            _atomic_write(manifest_path, encoded)
        return body
