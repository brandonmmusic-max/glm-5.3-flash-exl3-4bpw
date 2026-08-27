"""Packed-only EXL3/MCG payload storage for the GLM-5.3 campaign.

The encoder performs an exact FP16 reconstruction closure but this production
store retains only the deployable trellis, scale vectors and MCG marker.  A
reader must decode those objects and match ``reconstruction_closure`` before a
choice can be injected into a model.  The dense reconstruction is therefore
authenticated without adding a second roughly-source-sized artifact.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from .exact_payload import (
    PACKED_HASH_SCHEMA,
    ExactCodecPayloadStore,
    packed_payload_sha256,
    tensor_sha256,
)
from ..core.artifacts import canonical_json, sha256_bytes, write_json


PACKED_CHOICE_SCHEMA = "quant-pipeline.exl3-mcg-packed-choice.v1"
CHECKPOINT_HASH_SCHEMA = "quant-pipeline.exl3-mcg-checkpoint-framed.v1"
RECONSTRUCTION_CLOSURE_SCHEMA = "quant-pipeline.exl3-mcg-fp16-closure.v1"
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
_HASH = re.compile(r"[0-9a-f]{64}")
_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


def _tensor(value: Any):
    import torch

    result = torch.as_tensor(value).detach().contiguous().cpu()
    # PyTorch forbids a dtype-changing byte view of a scalar. The marker is a
    # one-element payload in safetensors as well, so normalize only rank zero.
    return result.reshape(1) if result.ndim == 0 else result


def checkpoint_payload_sha256(values: Mapping[str, Any]) -> str:
    names = ("trellis", "suh", "svh", "mcg")
    if set(values) != set(names):
        raise ValueError("EXL3/MCG payload object census differs")
    import torch

    framed = bytearray(CHECKPOINT_HASH_SCHEMA.encode("ascii"))
    framed.extend(len(names).to_bytes(2, "big"))
    for name in names:
        value = _tensor(values[name])
        raw = value.view(torch.uint8).numpy().tobytes()
        label = name.encode("ascii")
        framed.extend(len(label).to_bytes(2, "big"))
        framed.extend(label)
        framed.extend(len(raw).to_bytes(8, "big"))
        framed.extend(raw)
    return sha256_bytes(bytes(framed))


class PackedMCGPayloadStore:
    """Append-only EXL3/MCG objects plus a non-persisted FP16 closure."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = ExactCodecPayloadStore(self.root)
        self.choices = self.root / "choices"
        self.choices.mkdir(parents=True, exist_ok=True)

    def put_choice(
        self,
        *,
        layer: int,
        expert: int,
        projection: str,
        choice_id: str,
        trellis: Any,
        suh: Any,
        svh: Any,
        mcg: Any,
        reconstruction: Any,
        vector_topology: Mapping[str, str],
        reader_abi_sha256: str,
        provenance: Mapping[str, Any],
        predecessor_state_hash: str,
    ) -> dict[str, Any]:
        import torch

        if projection not in _PROJECTIONS:
            raise ValueError("unknown routed expert projection")
        if _HASH.fullmatch(str(predecessor_state_hash)) is None:
            raise ValueError("choice predecessor state must be SHA-256")
        if _HASH.fullmatch(str(reader_abi_sha256)) is None:
            raise ValueError("EXL3/MCG reader ABI must be SHA-256")
        values = {
            name: _tensor(value)
            for name, value in {
                "trellis": trellis,
                "suh": suh,
                "svh": svh,
                "mcg": mcg,
                "reconstruction": reconstruction,
            }.items()
        }
        if values["trellis"].dtype != torch.int16:
            raise ValueError("EXL3 trellis must be int16")
        if any(values[name].dtype != torch.float16 for name in ("suh", "svh", "reconstruction")):
            raise ValueError("EXL3 scales and closure reconstruction must be FP16")
        if (
            values["mcg"].dtype != torch.int32
            or values["mcg"].numel() != 1
            or int(values["mcg"].reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32
        ):
            raise ValueError("choice is not marked with MCG 0xCBAC1FED")
        reconstruction = values["reconstruction"]
        if reconstruction.ndim != 2 or values["suh"].ndim != 1 or values["svh"].ndim != 1:
            raise ValueError("EXL3/MCG choice tensor ranks differ")
        n, k = map(int, reconstruction.shape)
        if values["suh"].numel() != k or values["svh"].numel() != n:
            raise ValueError("EXL3/MCG vectors disagree with reconstruction geometry")
        if set(vector_topology) != {"suh", "svh"} or any(
            item not in {"layer_shared", "expert_private"}
            for item in vector_topology.values()
        ):
            raise ValueError("EXL3/MCG vector topology differs")
        stored = {name: values[name] for name in ("trellis", "suh", "svh", "mcg")}
        refs = {name: self.objects.put_tensor(value).as_dict() for name, value in stored.items()}
        body = {
            "schema": PACKED_CHOICE_SCHEMA,
            "layer": int(layer),
            "expert": int(expert),
            "projection": projection,
            "choice_id": str(choice_id),
            "bits": 4,
            "predecessor_state_hash": str(predecessor_state_hash),
            "objects": refs,
            "packed_hash_schema": PACKED_HASH_SCHEMA,
            "packed_sha256": packed_payload_sha256(
                {name: stored[name] for name in ("trellis", "suh", "svh")}
            ),
            "checkpoint_hash_schema": CHECKPOINT_HASH_SCHEMA,
            "checkpoint_payload_sha256": checkpoint_payload_sha256(stored),
            "logical_payload_bytes": sum(int(ref["bytes"]) for ref in refs.values()),
            "param_count": n * k,
            "vector_topology": dict(vector_topology),
            "reconstruction_closure": {
                "schema": RECONSTRUCTION_CLOSURE_SCHEMA,
                "dtype": "float16",
                "shape": [n, k],
                "orientation": "huggingface_out_in",
                "payload_sha256": tensor_sha256(reconstruction),
                "persisted": False,
                "encoder_full_decode_closure": True,
            },
            "decoder": {
                "codec_family": "exl3-mcg",
                "mcg_multiplier_hex": "0xCBAC1FED",
                "mcg_marker_signed_int32": MCG_MARKER_SIGNED_INT32,
                "reader_abi_sha256": str(reader_abi_sha256),
            },
            "provenance": copy.deepcopy(dict(provenance)),
        }
        body["choice_sha256"] = sha256_bytes(canonical_json(body))
        path = self.choices / f"{body['choice_sha256']}.json"
        if path.exists():
            import json

            if json.loads(path.read_text(encoding="utf-8")) != body:
                raise ValueError("EXL3/MCG choice hash collision")
        else:
            write_json(path, body)
        return body

    def verify_choice(self, choice: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        import json
        import torch

        row = (
            json.loads(Path(choice).read_text(encoding="utf-8"))
            if isinstance(choice, (str, Path))
            else copy.deepcopy(dict(choice))
        )
        expected = row.get("choice_sha256")
        unsigned = {key: value for key, value in row.items() if key != "choice_sha256"}
        if (
            row.get("schema") != PACKED_CHOICE_SCHEMA
            or _HASH.fullmatch(str(expected)) is None
            or sha256_bytes(canonical_json(unsigned)) != expected
            or row.get("bits") != 4
            or row.get("projection") not in _PROJECTIONS
        ):
            raise ValueError("EXL3/MCG packed-choice seal differs")
        objects = row.get("objects")
        if not isinstance(objects, Mapping) or set(objects) != {"trellis", "suh", "svh", "mcg"}:
            raise ValueError("EXL3/MCG packed-choice object census differs")
        values = {name: self.objects.load_tensor(ref) for name, ref in objects.items()}
        if (
            values["trellis"].dtype != torch.int16
            or values["suh"].dtype != torch.float16
            or values["svh"].dtype != torch.float16
            or values["mcg"].dtype != torch.int32
            or values["mcg"].numel() != 1
            or int(values["mcg"].reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32
        ):
            raise ValueError("EXL3/MCG stored dtype or marker differs")
        if (
            row.get("packed_hash_schema") != PACKED_HASH_SCHEMA
            or packed_payload_sha256({name: values[name] for name in ("trellis", "suh", "svh")}) != row.get("packed_sha256")
            or row.get("checkpoint_hash_schema") != CHECKPOINT_HASH_SCHEMA
            or checkpoint_payload_sha256(values) != row.get("checkpoint_payload_sha256")
        ):
            raise ValueError("EXL3/MCG packed payload hash differs")
        closure = row.get("reconstruction_closure")
        decoder = row.get("decoder")
        if (
            not isinstance(closure, Mapping)
            or closure.get("schema") != RECONSTRUCTION_CLOSURE_SCHEMA
            or closure.get("dtype") != "float16"
            or closure.get("orientation") != "huggingface_out_in"
            or closure.get("persisted") is not False
            or closure.get("encoder_full_decode_closure") is not True
            or _HASH.fullmatch(str(closure.get("payload_sha256", ""))) is None
            or not isinstance(decoder, Mapping)
            or decoder.get("codec_family") != "exl3-mcg"
            or decoder.get("mcg_multiplier_hex") != "0xCBAC1FED"
            or decoder.get("mcg_marker_signed_int32") != MCG_MARKER_SIGNED_INT32
            or _HASH.fullmatch(str(decoder.get("reader_abi_sha256", ""))) is None
        ):
            raise ValueError("EXL3/MCG reconstruction/decoder closure differs")
        shape = closure.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
            or row.get("param_count") != shape[0] * shape[1]
            or values["suh"].numel() != shape[1]
            or values["svh"].numel() != shape[0]
            or row.get("logical_payload_bytes") != sum(int(ref["bytes"]) for ref in objects.values())
        ):
            raise ValueError("EXL3/MCG packed-choice geometry/accounting differs")
        return row
