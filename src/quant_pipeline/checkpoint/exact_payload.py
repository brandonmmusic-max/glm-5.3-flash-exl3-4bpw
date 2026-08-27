"""Content-addressed exact-codec payload storage.

Packed trellis bytes, transform vectors, and the FP16 reconstruction oracle
are persisted together. Hash-only ledgers cannot satisfy this protocol. The
packed identity is canonically role- and length-framed; ambiguous legacy v1
concatenation choices are deliberately rejected rather than auto-upgraded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json


OBJECT_SCHEMA = "quant-pipeline.exact-codec-object.v1"
CHOICE_SCHEMA = "quant-pipeline.exact-codec-choice.v2"
PACKED_HASH_SCHEMA = "quant-pipeline.packed-payload-length-framed.v1"
_HASH = re.compile(r"[0-9a-f]{64}")


def _torch_tensor(value: Any):
    import torch

    return torch.as_tensor(value).detach().contiguous().cpu()


def tensor_bytes(value: Any) -> bytes:
    tensor = _torch_tensor(value)
    return tensor.view(__import__("torch").uint8).numpy().tobytes()


def tensor_sha256(value: Any) -> str:
    return sha256_bytes(tensor_bytes(value))


def packed_payload_sha256(values: Mapping[str, Any]) -> str:
    """Hash trellis/suh/svh with canonical labels and uint64 byte lengths."""

    names = ("trellis", "suh", "svh")
    if set(values) != set(names):
        raise ValueError("packed payload must contain exactly trellis, suh, and svh")
    framed = bytearray(PACKED_HASH_SCHEMA.encode("ascii"))
    framed.extend(len(names).to_bytes(2, "big"))
    for name in names:
        raw = values[name] if isinstance(values[name], bytes) else tensor_bytes(values[name])
        label = name.encode("ascii")
        framed.extend(len(label).to_bytes(2, "big"))
        framed.extend(label)
        framed.extend(len(raw).to_bytes(8, "big"))
        framed.extend(raw)
    return sha256_bytes(bytes(framed))


_DTYPE_FROM_NAME: dict[str, str] = {
    "bool": "bool",
    "uint8": "uint8",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
}


@dataclass(frozen=True)
class PayloadObjectRef:
    sha256: str
    bytes: int
    dtype: str
    shape: tuple[int, ...]
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OBJECT_SCHEMA,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "path": self.path,
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PayloadObjectRef":
        if row.get("schema") != OBJECT_SCHEMA:
            raise ValueError("unsupported exact-codec object reference")
        digest = row.get("sha256")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("invalid exact-codec object SHA-256")
        return cls(digest, int(row["bytes"]), str(row["dtype"]), tuple(int(x) for x in row["shape"]), str(row["path"]))


class ExactCodecPayloadStore:
    """Append-only object store with typed tensor reconstruction."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.choices = self.root / "choices"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.choices.mkdir(parents=True, exist_ok=True)

    def put_tensor(self, value: Any) -> PayloadObjectRef:
        tensor = _torch_tensor(value)
        raw = tensor_bytes(tensor)
        digest = sha256_bytes(raw)
        relative = Path("objects") / digest[:2] / f"{digest}.bin"
        path = self.root / relative
        if path.exists():
            if path.is_symlink() or path.stat().st_size != len(raw) or sha256_file(path) != digest:
                raise ValueError(f"content-addressed object collision/corruption: {digest}")
        else:
            atomic_write(path, raw)
        return PayloadObjectRef(
            sha256=digest,
            bytes=len(raw),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(int(x) for x in tensor.shape),
            path=relative.as_posix(),
        )

    def verify(self, ref: PayloadObjectRef | Mapping[str, Any]) -> PayloadObjectRef:
        ref = ref if isinstance(ref, PayloadObjectRef) else PayloadObjectRef.from_mapping(ref)
        expected = (Path("objects") / ref.sha256[:2] / f"{ref.sha256}.bin").as_posix()
        if ref.path != expected or ref.dtype not in _DTYPE_FROM_NAME or ref.bytes < 0:
            raise ValueError("malformed exact-codec object reference")
        path = self.root / ref.path
        if not path.is_file() or path.is_symlink() or path.stat().st_size != ref.bytes or sha256_file(path) != ref.sha256:
            raise ValueError(f"exact-codec object missing or corrupt: {ref.sha256}")
        return ref

    def load_tensor(self, ref: PayloadObjectRef | Mapping[str, Any]):
        import torch

        ref = self.verify(ref)
        dtype = getattr(torch, _DTYPE_FROM_NAME[ref.dtype])
        raw = bytearray((self.root / ref.path).read_bytes())
        value = torch.frombuffer(raw, dtype=dtype).clone()
        expected = 1
        for size in ref.shape:
            expected *= size
        if value.numel() != expected:
            raise ValueError("exact-codec object byte count contradicts dtype/shape")
        return value.reshape(ref.shape)

    def put_choice(
        self,
        *,
        layer: int,
        expert: int,
        projection: str,
        choice_id: str,
        bits: int,
        trellis: Any,
        suh: Any,
        svh: Any,
        reconstruction: Any,
        vector_topology: Mapping[str, str],
        provenance: Mapping[str, Any],
        predecessor_state_hash: str,
    ) -> dict[str, Any]:
        import torch

        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError("unknown Qwen expert projection")
        if not isinstance(bits, int) or isinstance(bits, bool) or bits not in {3, 4, 5}:
            raise ValueError("competitive exact-codec bits must be 3, 4, or 5")
        if not isinstance(predecessor_state_hash, str) or not _HASH.fullmatch(predecessor_state_hash):
            raise ValueError("choice predecessor state must be a SHA-256")
        tensors = {name: _torch_tensor(value) for name, value in {
            "trellis": trellis,
            "suh": suh,
            "svh": svh,
            "reconstruction": reconstruction,
        }.items()}
        if tensors["trellis"].dtype != torch.int16:
            raise ValueError("EXL3 trellis must be int16")
        if tensors["suh"].dtype != torch.float16 or tensors["svh"].dtype != torch.float16 or tensors["reconstruction"].dtype != torch.float16:
            raise ValueError("stored transform vectors and reconstruction must be FP16")
        if tensors["reconstruction"].ndim != 2 or tensors["suh"].ndim != 1 or tensors["svh"].ndim != 1:
            raise ValueError("exact-codec tensor ranks are invalid")
        n, k = tensors["reconstruction"].shape
        if tensors["suh"].numel() != k or tensors["svh"].numel() != n:
            raise ValueError("transform vector lengths disagree with reconstruction")
        refs = {name: self.put_tensor(value).as_dict() for name, value in tensors.items()}
        packed_bytes = sum(refs[name]["bytes"] for name in ("trellis", "suh", "svh"))
        body = {
            "schema": CHOICE_SCHEMA,
            "layer": int(layer),
            "expert": int(expert),
            "projection": projection,
            "choice_id": choice_id,
            "bits": bits,
            "predecessor_state_hash": predecessor_state_hash,
            "objects": refs,
            "packed_hash_schema": PACKED_HASH_SCHEMA,
            "packed_sha256": packed_payload_sha256(
                {name: tensors[name] for name in ("trellis", "suh", "svh")}
            ),
            "reconstruction_sha256": refs["reconstruction"]["sha256"],
            "logical_payload_bytes": packed_bytes,
            "param_count": int(n * k),
            "vector_topology": dict(vector_topology),
            "provenance": dict(provenance),
        }
        body["choice_sha256"] = sha256_bytes(canonical_json(body))
        path = self.choices / f"{body['choice_sha256']}.json"
        if path.exists():
            if json.loads(path.read_text()) != body:
                raise ValueError("choice hash collision")
        else:
            write_json(path, body)
        return body

    def verify_choice(self, choice: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(choice, (str, Path)):
            row = json.loads(Path(choice).read_text())
        else:
            row = dict(choice)
        if row.get("schema") != CHOICE_SCHEMA:
            raise ValueError("unsupported exact-codec choice; legacy unframed v1 choices must be regenerated")
        if row.get("packed_hash_schema") != PACKED_HASH_SCHEMA:
            raise ValueError("unsupported packed payload hash framing")
        expected = row.get("choice_sha256")
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise ValueError("invalid exact-codec choice seal")
        if sha256_bytes(canonical_json({key: value for key, value in row.items() if key != "choice_sha256"})) != expected:
            raise ValueError("exact-codec choice seal mismatch")
        for ref in row["objects"].values():
            self.verify(ref)
        packed = packed_payload_sha256(
            {
                name: (self.root / row["objects"][name]["path"]).read_bytes()
                for name in ("trellis", "suh", "svh")
            }
        )
        if packed != row["packed_sha256"] or row["objects"]["reconstruction"]["sha256"] != row["reconstruction_sha256"]:
            raise ValueError("choice packed/reconstruction identity mismatch")
        return row
