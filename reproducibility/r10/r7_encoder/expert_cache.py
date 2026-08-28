"""Crash-resumable, content-bound final expert mini-shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .constants import PROJECTIONS, RECIPE_MARKER, RECIPE_VERSION, TensorId
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .safetensors_io import (
    SafeTensorReader,
    read_torch_tensor,
    torch_tensor_entry,
    write_safetensors_atomic,
)
from .trellis import Exl3TrellisCodec
from .types import EncodedTensor


def _tensor_sha256(tensor) -> str:
    import torch

    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    return sha256_bytes(value.view(torch.uint8).numpy().tobytes())


@dataclass(frozen=True)
class CachedExpert:
    encoded: tuple[EncodedTensor, EncodedTensor, EncodedTensor]
    gate_up_sha256: str
    final_loss: float
    holdout_row_ids_sha256: str
    permutation_audit: Mapping[str, object]
    cold_audit: tuple[Mapping[str, object], ...]


def _paths(root: Path, expert: int) -> tuple[Path, Path]:
    return (
        root / f"expert-{expert:03d}.safetensors",
        root / f"expert-{expert:03d}.json",
    )


def _discard_derivative_pair(shard: Path, manifest: Path) -> None:
    shard.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)


def write_cached_expert(
    root: str | Path,
    *,
    encoded: Sequence[EncodedTensor],
    bindings: Mapping[str, str],
    gate_up_sha256: str,
    final_loss: float,
    holdout_row_ids_sha256: str,
    permutation_audit: Mapping[str, object],
    cold_audit: Sequence[Mapping[str, object]] = (),
) -> str:
    values = tuple(
        sorted(encoded, key=lambda item: PROJECTIONS.index(item.tensor_id.projection))
    )
    if (
        len(values) != 3
        or tuple(item.tensor_id.projection for item in values) != PROJECTIONS
    ):
        raise ValueError("expert cache requires gate/up/down")
    layer = values[0].tensor_id.layer
    expert = values[0].tensor_id.expert
    if any(
        item.tensor_id.layer != layer or item.tensor_id.expert != expert
        for item in values
    ):
        raise ValueError("expert cache contains mixed identities")
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    shard, manifest_path = _paths(directory, expert)
    if shard.exists() or manifest_path.exists():
        raise FileExistsError("expert-cache writer requires a clean transaction pair")
    entries = []
    for item in values:
        prefix = item.tensor_id.projection
        entries.extend(
            (
                torch_tensor_entry(f"{prefix}.trellis", item.trellis),
                torch_tensor_entry(f"{prefix}.suh", item.suh),
                torch_tensor_entry(f"{prefix}.svh", item.svh),
            )
        )
    payload_hashes, shard_hash = write_safetensors_atomic(
        shard,
        entries,
        metadata={
            "r7_schema": "r7-final-expert-cache-v1",
            "layer": str(layer),
            "expert": str(expert),
        },
    )
    manifest = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-final-expert-cache-v1",
        "layer": layer,
        "expert": expert,
        "shard": shard.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
        "bindings": dict(sorted(bindings.items())),
        "gate_up_sha256": gate_up_sha256,
        "final_loss": format(final_loss, ".17g"),
        "holdout_row_ids_sha256": holdout_row_ids_sha256,
        "permutation_audit": dict(permutation_audit),
        "cold_audit": [dict(record) for record in cold_audit],
        "tensors": {
            item.tensor_id.projection: {
                "bits": item.bits,
                "proxy_loss": format(item.proxy_loss, ".17g"),
                "packed_sha256": item.packed_sha256,
                "reconstruction_sha256": item.reconstruction_sha256,
                "provenance": dict(item.provenance),
            }
            for item in values
        },
    }
    manifest["content_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    atomic_write_json(manifest_path, manifest)
    return sha256_file(manifest_path)


def load_cached_expert(
    root: str | Path,
    *,
    layer: int,
    expert: int,
    bits: Mapping[str, int],
    bindings: Mapping[str, str],
    codec: Exl3TrellisCodec,
    reconstruct: bool = True,
) -> CachedExpert | None:
    """Load and authenticate one final-expert cache entry.

    Packed tensors and both stored vectors are always authenticated against the
    sealed mini-shard.  ``reconstruct=False`` deliberately defers the expensive
    packed reconstruction oracle to the runtime install path, which decodes the
    same authenticated payload and checks ``reconstruction_sha256`` before the
    value can become active.  Other callers retain the original eager oracle by
    default.
    """

    shard, manifest_path = _paths(Path(root), expert)
    if not manifest_path.exists() and not shard.exists():
        return None
    if not manifest_path.is_file() or not shard.is_file():
        _discard_derivative_pair(shard, manifest_path)
        return None
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, TypeError):
        _discard_derivative_pair(shard, manifest_path)
        return None
    if (
        manifest.get("marker") != RECIPE_MARKER
        or manifest.get("schema") != "r7-final-expert-cache-v1"
        or int(manifest.get("layer", -1)) != layer
        or int(manifest.get("expert", -1)) != expert
        or manifest.get("bindings") != dict(sorted(bindings.items()))
    ):
        raise ValueError("final expert cache identity/binding drift")
    content_hash = manifest.pop("content_sha256", None)
    if content_hash != sha256_bytes(canonical_json_bytes(manifest)):
        _discard_derivative_pair(shard, manifest_path)
        return None
    manifest["content_sha256"] = content_hash
    if sha256_file(shard) != manifest["shard_sha256"]:
        _discard_derivative_pair(shard, manifest_path)
        return None
    try:
        reader = SafeTensorReader(shard)
    except (OSError, ValueError, TypeError):
        _discard_derivative_pair(shard, manifest_path)
        return None
    if reader.metadata.get("r7_schema") != "r7-final-expert-cache-v1":
        _discard_derivative_pair(shard, manifest_path)
        return None
    if set(reader.tensors) != set(manifest["payload_sha256"]):
        _discard_derivative_pair(shard, manifest_path)
        return None
    raw_cold_audit = manifest.get("cold_audit")
    if not isinstance(raw_cold_audit, list) or any(
        not isinstance(record, Mapping)
        or int(record.get("layer", -1)) != layer
        for record in raw_cold_audit
    ):
        raise ValueError("final expert cache cold-fallback evidence is malformed")
    values = []
    for projection in PROJECTIONS:
        tensor_id = TensorId(layer, expert, projection)
        record = manifest["tensors"][projection]
        if int(record["bits"]) != int(bits[tensor_id.key]):
            raise ValueError("final expert cache allocation drift")
        for suffix in ("trellis", "suh", "svh"):
            name = f"{projection}.{suffix}"
            if (
                reader.tensors[name].payload.sha256()
                != manifest["payload_sha256"][name]
            ):
                _discard_derivative_pair(shard, manifest_path)
                return None
        packed = read_torch_tensor(reader, f"{projection}.trellis")
        suh = read_torch_tensor(reader, f"{projection}.suh")
        svh = read_torch_tensor(reader, f"{projection}.svh")
        if _tensor_sha256(packed) != record["packed_sha256"]:
            raise ValueError("cached packed tensor hash drift")
        reconstructed = None
        if reconstruct:
            reconstructed = (
                codec.decode_to_original(
                    packed.to(codec.config.device), suh, svh, int(record["bits"])
                )
                .detach()
                .cpu()
            )
            if (
                _tensor_sha256(reconstructed.half())
                != record["reconstruction_sha256"]
            ):
                raise ValueError("cached reconstruction oracle failed")
        values.append(
            EncodedTensor(
                tensor_id=tensor_id,
                bits=int(record["bits"]),
                trellis=packed,
                suh=suh,
                svh=svh,
                reconstructed_kn=reconstructed,
                proxy_loss=float(record["proxy_loss"]),
                packed_sha256=str(record["packed_sha256"]),
                reconstruction_sha256=str(record["reconstruction_sha256"]),
                provenance=dict(record["provenance"]),
            )
        )
    return CachedExpert(
        encoded=(values[0], values[1], values[2]),
        gate_up_sha256=str(manifest["gate_up_sha256"]),
        final_loss=float(manifest["final_loss"]),
        holdout_row_ids_sha256=str(manifest["holdout_row_ids_sha256"]),
        permutation_audit=dict(manifest["permutation_audit"]),
        cold_audit=tuple(dict(record) for record in raw_cold_audit),
    )
