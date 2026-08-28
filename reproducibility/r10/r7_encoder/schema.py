"""Topology-neutral schema v2 emission and shape/allocation audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .allocation import audit_allocation
from .constants import (
    MCG_MULT,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    PROJECTIONS,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TP_SLICE_QUANTUM,
    TRELLIS_TILE,
    TensorId,
)
from .determinism import atomic_write_json, sha256_file
from .safetensors_io import TensorEntry, torch_tensor_entry, write_safetensors_atomic
from .types import EncodedTensor

SCHEMA_VERSION = 2


def layer_shared_prefix(layer: int) -> str:
    return f"model.layers.{layer}.mlp.experts.r7_shared"


def shared_gate_up_suh_name(layer: int) -> str:
    return f"{layer_shared_prefix(layer)}.gate_up_suh"


def shared_down_svh_name(layer: int) -> str:
    return f"{layer_shared_prefix(layer)}.down_svh"


def tensor_name(tensor_id: TensorId, suffix: str) -> str:
    return f"{tensor_id.hf_prefix}.{suffix}"


def validate_encoded_tensor(encoded: EncodedTensor) -> None:
    shape = tuple(encoded.trellis.shape)
    expected = (
        encoded.tensor_id.k // TRELLIS_TILE,
        encoded.tensor_id.n // TRELLIS_TILE,
        TRELLIS_TILE * encoded.bits,
    )
    if shape != expected:
        raise ValueError(f"{encoded.tensor_id.key}: trellis {shape} != {expected}")
    if tuple(encoded.suh.shape) != (encoded.tensor_id.k,):
        raise ValueError(f"{encoded.tensor_id.key}: suh shape mismatch")
    if tuple(encoded.svh.shape) != (encoded.tensor_id.n,):
        raise ValueError(f"{encoded.tensor_id.key}: svh shape mismatch")


@dataclass(frozen=True)
class LoadSlice:
    rank: int
    trellis_axis: int
    trellis_start: int
    trellis_end: int
    vector_axis: str
    vector_start: int
    vector_end: int


def load_time_tp_slices(tensor_id: TensorId, tp_size: int) -> tuple[LoadSlice, ...]:
    if tp_size <= 0:
        raise ValueError("TP size must be positive")
    split_size = tensor_id.n if tensor_id.projection != "down_proj" else tensor_id.k
    if split_size % tp_size or split_size // tp_size % TP_SLICE_QUANTUM:
        raise ValueError(
            f"{tensor_id.key}: TP={tp_size} violates {TP_SLICE_QUANTUM}-element boundary"
        )
    per_rank = split_size // tp_size
    per_rank_tiles = per_rank // TRELLIS_TILE
    trellis_axis = 1 if tensor_id.projection != "down_proj" else 0
    vector_axis = "svh" if tensor_id.projection != "down_proj" else "suh"
    return tuple(
        LoadSlice(
            rank=rank,
            trellis_axis=trellis_axis,
            trellis_start=rank * per_rank_tiles,
            trellis_end=(rank + 1) * per_rank_tiles,
            vector_axis=vector_axis,
            vector_start=rank * per_rank,
            vector_end=(rank + 1) * per_rank,
        )
        for rank in range(tp_size)
    )


def _same_tensor_bytes(left, right) -> bool:
    import torch

    a = torch.as_tensor(left).detach().contiguous().cpu()
    b = torch.as_tensor(right).detach().contiguous().cpu()
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)


def emit_layer_v2(
    output_dir: str | Path,
    *,
    layer: int,
    encoded_tensors: Iterable[EncodedTensor],
    shared_gate_up_suh,
    shared_down_svh,
    allocation_bits: Mapping[str, int],
    layer_provenance: Mapping[str, object],
    permutations: Mapping[int, Iterable[int]],
    permutation_policies: Mapping[int, str],
    final_expert_artifacts: Mapping[int, str],
) -> dict[str, object]:
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoded = tuple(sorted(encoded_tensors, key=lambda item: item.tensor_id.key))
    expected_count = NUM_EXPERTS * len(PROJECTIONS)
    if (
        len(encoded) != expected_count
        or len({item.tensor_id for item in encoded}) != expected_count
    ):
        raise ValueError(
            f"layer {layer}: need exactly {expected_count} encoded tensors"
        )
    if any(item.tensor_id.layer != layer for item in encoded):
        raise ValueError("foreign layer tensor in v2 emission")
    audit_allocation(allocation_bits)
    if set(permutations) != set(range(NUM_EXPERTS)) or set(permutation_policies) != set(
        range(NUM_EXPERTS)
    ):
        raise ValueError("schema v2 requires permutation provenance for all experts")
    if set(final_expert_artifacts) != set(range(NUM_EXPERTS)):
        raise ValueError("schema v2 requires all final expert mini-shard seals")

    shared_gu = torch.as_tensor(shared_gate_up_suh).detach().half().contiguous().cpu()
    shared_down = torch.as_tensor(shared_down_svh).detach().half().contiguous().cpu()
    if tuple(shared_gu.shape) != (HIDDEN_SIZE,) or tuple(shared_down.shape) != (
        HIDDEN_SIZE,
    ):
        raise ValueError("layer-shared vector geometry drift")
    if (
        not torch.isfinite(shared_gu).all()
        or not torch.isfinite(shared_down).all()
        or (shared_gu == 0).any()
        or (shared_down == 0).any()
    ):
        raise ValueError("layer-shared vectors are invalid after FP16 storage")
    entries: list[TensorEntry] = [
        torch_tensor_entry(shared_gate_up_suh_name(layer), shared_gu),
        torch_tensor_entry(shared_down_svh_name(layer), shared_down),
    ]
    bit_map: dict[str, int] = {}
    vector_refs: dict[str, dict[str, str]] = {}
    tensor_hashes: dict[str, dict[str, str]] = {}
    tensor_provenance: dict[str, dict[str, object]] = {}
    marker = torch.tensor(MCG_MULT, dtype=torch.uint32).view(torch.int32)
    for item in encoded:
        validate_encoded_tensor(item)
        key = item.tensor_id.key
        if allocation_bits.get(key) != item.bits:
            raise ValueError(f"{key}: encoded bits disagree with allocation")
        if item.tensor_id.projection in ("gate_proj", "up_proj"):
            if not _same_tensor_bytes(item.suh, shared_gu):
                raise ValueError(
                    f"{key}: gate/up suh is not the sealed layer-shared vector"
                )
            entries.append(
                torch_tensor_entry(tensor_name(item.tensor_id, "svh"), item.svh)
            )
            refs = {
                "suh": shared_gate_up_suh_name(layer),
                "svh": tensor_name(item.tensor_id, "svh"),
            }
        else:
            if not _same_tensor_bytes(item.svh, shared_down):
                raise ValueError(
                    f"{key}: down svh is not the sealed layer-shared vector"
                )
            entries.append(
                torch_tensor_entry(tensor_name(item.tensor_id, "suh"), item.suh)
            )
            refs = {
                "suh": tensor_name(item.tensor_id, "suh"),
                "svh": shared_down_svh_name(layer),
            }
        entries.extend(
            [
                torch_tensor_entry(
                    tensor_name(item.tensor_id, "trellis"), item.trellis
                ),
                torch_tensor_entry(tensor_name(item.tensor_id, "mcg"), marker),
            ]
        )
        bit_map[item.tensor_id.hf_prefix] = item.bits
        vector_refs[item.tensor_id.hf_prefix] = refs
        tensor_hashes[item.tensor_id.hf_prefix] = {
            "packed_sha256": item.packed_sha256,
            "reconstruction_sha256": item.reconstruction_sha256,
        }
        required_provenance = {
            "bf16_sha256",
            "source_name",
            "source_shard",
            "source_payload_start",
            "source_payload_end",
            "source_inventory_sha256",
            "numeric_environment_sha256",
            "runtime_inventory_sha256",
            "backend_fingerprint",
            "state_sha256",
            "capture_sha256",
            "search_sha256",
            "allocation_sha256",
            "probe_sha256",
            "fit_row_ids_sha256",
            "down_fit_row_ids_sha256",
            "holdout_row_ids_sha256",
            "permutation_sha256",
            "permutation_policy",
            "vector_sha256",
            "covariance_sha256",
            "full_k",
            "full_n",
            "mcg",
            "codebook_scale",
            "sigma_reg",
        }
        if not required_provenance <= set(item.provenance):
            missing = sorted(required_provenance - set(item.provenance))
            raise ValueError(f"{key}: incomplete final provenance {missing}")
        tensor_provenance[item.tensor_id.hf_prefix] = dict(item.provenance)

    shard = output / f"r7-experts-layer-{layer:03d}.safetensors"
    payload_hashes, shard_hash = write_safetensors_atomic(
        shard,
        entries,
        metadata={
            "format": "pt",
            "r7_schema": str(SCHEMA_VERSION),
            "r7_layer": str(layer),
            "r7_marker": RECIPE_MARKER,
        },
    )
    manifest = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "shard": shard.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
        "bit_map": dict(sorted(bit_map.items())),
        "vector_refs": dict(sorted(vector_refs.items())),
        "roundtrip_hashes": dict(sorted(tensor_hashes.items())),
        "tensor_provenance": dict(sorted(tensor_provenance.items())),
        "allocation_bit_units": sum(bit_map.values()),
        "allocation_target_bpw": "3.5",
        "shared_vectors": {
            "gate_up_suh": shared_gate_up_suh_name(layer),
            "down_svh": shared_down_svh_name(layer),
        },
        "provenance": dict(layer_provenance),
        "permutations": {
            str(expert): {
                "new_to_old": [int(value) for value in permutations[expert]],
                "policy": str(permutation_policies[expert]),
            }
            for expert in range(NUM_EXPERTS)
        },
        "final_expert_artifact_sha256": {
            str(expert): str(final_expert_artifacts[expert])
            for expert in range(NUM_EXPERTS)
        },
    }
    manifest_path = output / f"r7-experts-layer-{layer:03d}.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
