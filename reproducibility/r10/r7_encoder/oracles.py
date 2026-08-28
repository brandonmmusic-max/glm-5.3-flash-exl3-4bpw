"""Round 7 exactness and manifest oracles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .allocation import audit_allocation
from .constants import (
    FIRST_MOE_LAYER,
    HIDDEN_SIZE,
    LAST_MOE_LAYER,
    MCG_MULT,
    NUM_EXPERTS,
    PROJECTIONS,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TensorId,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .permutation import validate_permutation
from .safetensors_io import SafeTensorReader, read_torch_tensor
from .schema import (
    load_time_tp_slices,
    shared_down_svh_name,
    shared_gate_up_suh_name,
    tensor_name,
)
from .trellis import Exl3TrellisCodec
from .types import EncodedTensor


def allocation_oracle(bit_map: Mapping[str, int], layer: int) -> None:
    expected = {
        TensorId(layer, expert, projection).hf_prefix
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    if set(bit_map) != expected or any(
        type(value) is not int or value not in (3, 4, 5) for value in bit_map.values()
    ):
        raise ValueError("allocation bit map is not the exact 768-key integer domain")
    internal = {
        TensorId(layer, expert, projection).key: int(
            bit_map[TensorId(layer, expert, projection).hf_prefix]
        )
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    audit_allocation(internal)


def shared_vector_oracle(encoded: Iterable[EncodedTensor]) -> None:
    import torch

    values = tuple(encoded)
    if not values:
        raise ValueError("empty encoded layer")
    gate_up = [
        torch.as_tensor(item.suh).cpu()
        for item in values
        if item.tensor_id.projection in ("gate_proj", "up_proj")
    ]
    down = [
        torch.as_tensor(item.svh).cpu()
        for item in values
        if item.tensor_id.projection == "down_proj"
    ]
    if len(gate_up) != NUM_EXPERTS * 2 or len(down) != NUM_EXPERTS:
        raise ValueError("shared-vector oracle has incomplete layer")
    if any(not torch.equal(gate_up[0], value) for value in gate_up[1:]):
        raise AssertionError("gate/up layer-shared suh differs across experts")
    if any(not torch.equal(down[0], value) for value in down[1:]):
        raise AssertionError("down layer-shared svh differs across experts")


def sliced_reconstruction_oracle(
    encoded: EncodedTensor,
    codec: Exl3TrellisCodec,
    tp_size: int,
) -> None:
    import torch

    full = codec.decode_to_original(
        encoded.trellis.to(codec.config.device),
        encoded.suh,
        encoded.svh,
        encoded.bits,
    ).half()
    pieces = []
    for split in load_time_tp_slices(encoded.tensor_id, tp_size):
        if encoded.tensor_id.projection != "down_proj":
            packed = encoded.trellis[
                :, split.trellis_start : split.trellis_end, :
            ].contiguous()
            suh = encoded.suh
            svh = encoded.svh[split.vector_start : split.vector_end]
        else:
            packed = encoded.trellis[
                split.trellis_start : split.trellis_end, :, :
            ].contiguous()
            suh = encoded.suh[split.vector_start : split.vector_end]
            svh = encoded.svh
        pieces.append(
            codec.decode_to_original(
                packed.to(codec.config.device), suh, svh, encoded.bits
            ).half()
        )
    axis = 1 if encoded.tensor_id.projection != "down_proj" else 0
    sliced = torch.cat(pieces, dim=axis)
    if not torch.equal(full, sliced):
        raise AssertionError(
            f"{encoded.tensor_id.key}: full reconstruction != TP-sliced reconstruction"
        )


def _validate_parallel_reconstruction_results(
    manifest: Mapping[str, object],
    *,
    layer: int,
    tp_sizes: tuple[int, ...],
    results: Iterable[tuple[int, object]],
) -> tuple[Mapping[str, object], ...]:
    """Validate and canonicalize the complete process-worker oracle domain."""

    values = tuple(results)
    experts = tuple(int(expert) for expert, _ in values)
    if experts != tuple(range(NUM_EXPERTS)):
        raise ValueError("parallel V2 reconstruction expert domain/order is incomplete")
    canonical = []
    for expert, raw in values:
        if not isinstance(raw, Mapping) or set(raw) != {
            "layer",
            "expert",
            "projections",
            "passed",
        }:
            raise ValueError("malformed parallel V2 reconstruction result")
        if (
            int(raw["layer"]) != layer
            or int(raw["expert"]) != expert
            or raw["passed"] is not True
        ):
            raise ValueError("parallel V2 reconstruction result binding drift")
        projections = raw["projections"]
        if not isinstance(projections, Mapping) or set(projections) != set(PROJECTIONS):
            raise ValueError(
                "parallel V2 reconstruction projection domain is incomplete"
            )
        ordered_projections = {}
        for projection in PROJECTIONS:
            record = projections[projection]
            if not isinstance(record, Mapping) or set(record) != {
                "packed_sha256",
                "reconstruction_sha256",
                "tp_sizes",
                "passed",
            }:
                raise ValueError("malformed parallel V2 projection result")
            tensor_id = TensorId(layer, expert, projection)
            prefix = tensor_id.hf_prefix
            packed_name = tensor_name(tensor_id, "trellis")
            if (
                record["passed"] is not True
                or tuple(int(value) for value in record["tp_sizes"]) != tp_sizes
                or record["packed_sha256"] != manifest["payload_sha256"][packed_name]  # type: ignore[index]
                or record["packed_sha256"]
                != manifest["roundtrip_hashes"][prefix]["packed_sha256"]  # type: ignore[index]
                or record["reconstruction_sha256"]
                != manifest["roundtrip_hashes"][prefix][  # type: ignore[index]
                    "reconstruction_sha256"
                ]
            ):
                raise ValueError(
                    f"{tensor_id.key}: parallel reconstruction audit drift"
                )
            ordered_projections[projection] = dict(record)
        canonical.append(
            {
                "layer": layer,
                "expert": expert,
                "projections": ordered_projections,
                "passed": True,
            }
        )
    return tuple(canonical)


def audit_v2_layer(
    manifest_path: str | Path,
    *,
    codec: Exl3TrellisCodec | None = None,
    tp_sizes: Iterable[int] = (),
    process_pool=None,
) -> dict[str, object]:
    path = Path(manifest_path)
    tp_sizes = tuple(int(value) for value in tp_sizes)
    manifest = json.loads(path.read_text())
    layer = int(manifest["layer"])
    if (
        manifest.get("marker") != RECIPE_MARKER
        or manifest.get("recipe_version") != RECIPE_VERSION
        or int(manifest.get("schema_version", -1)) != 2
        or not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER
    ):
        raise ValueError("not schema v2")
    shard = path.parent / manifest["shard"]
    if sha256_file(shard) != manifest["shard_sha256"]:
        raise AssertionError("v2 shard file hash mismatch")
    reader = SafeTensorReader(shard)
    if reader.metadata != {
        "format": "pt",
        "r7_layer": str(layer),
        "r7_marker": RECIPE_MARKER,
        "r7_schema": "2",
    }:
        raise ValueError("v2 shard metadata mismatch")
    allocation_oracle(manifest["bit_map"], layer)
    if (
        int(manifest.get("allocation_bit_units", -1)) != 2688
        or manifest.get("allocation_target_bpw") != "3.5"
    ):
        raise ValueError("v2 allocation arithmetic metadata drift")
    expected_prefixes = {
        TensorId(layer, expert, projection).hf_prefix
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    for field in ("vector_refs", "roundtrip_hashes", "tensor_provenance"):
        if set(manifest.get(field, {})) != expected_prefixes:
            raise ValueError(f"v2 {field} does not cover the exact tensor domain")
    if set(manifest.get("permutations", {})) != {
        str(expert) for expert in range(NUM_EXPERTS)
    }:
        raise ValueError("v2 permutation map is incomplete")
    if set(manifest.get("final_expert_artifact_sha256", {})) != {
        str(expert) for expert in range(NUM_EXPERTS)
    }:
        raise ValueError("v2 final-expert cache provenance is incomplete")
    audits = manifest.get("provenance", {}).get("permutation_audit", {})
    if set(audits) not in (
        set(range(NUM_EXPERTS)),
        {str(expert) for expert in range(NUM_EXPERTS)},
    ):
        raise ValueError("v2 permutation oracle report is incomplete")
    install_digest = manifest.get("provenance", {}).get("install_audit_sha256")
    if (
        not isinstance(install_digest, str)
        or len(install_digest) != 64
        or any(character not in "0123456789abcdef" for character in install_digest)
    ):
        raise ValueError("v2 manifest lacks packed-install arithmetic provenance")
    for expert in range(NUM_EXPERTS):
        permutation = validate_permutation(
            manifest["permutations"][str(expert)]["new_to_old"]
        )
        audit = audits.get(str(expert), audits.get(expert))
        if not isinstance(audit, dict) or not audit.get("passed"):
            raise ValueError(f"permutation oracle did not pass for expert {expert}")
        digest = sha256_bytes(canonical_json_bytes(list(permutation)))
        for projection in PROJECTIONS:
            prefix = TensorId(layer, expert, projection).hf_prefix
            if (
                manifest["tensor_provenance"][prefix].get("permutation_sha256")
                != digest
            ):
                raise ValueError(f"permutation provenance mismatch for {prefix}")
    required = {shared_gate_up_suh_name(layer), shared_down_svh_name(layer)}
    shared_gu = reader.tensors[shared_gate_up_suh_name(layer)]
    shared_down = reader.tensors[shared_down_svh_name(layer)]
    if (
        shared_gu.dtype != "F16"
        or shared_gu.shape != (HIDDEN_SIZE,)
        or shared_down.dtype != "F16"
        or shared_down.shape != (HIDDEN_SIZE,)
    ):
        raise ValueError("v2 shared-vector dtype/shape drift")
    import torch

    for info in (shared_gu, shared_down):
        value = read_torch_tensor(reader, info.name)
        if not torch.isfinite(value).all() or (value == 0).any():
            raise ValueError("v2 shared vector invalid after storage")
    if process_pool is not None and codec is None:
        raise ValueError(
            "parallel V2 reconstruction requires an explicit codec contract"
        )
    parallel_reconstruction = codec is not None and process_pool is not None
    for expert in range(NUM_EXPERTS):
        for projection in PROJECTIONS:
            tensor_id = TensorId(layer, expert, projection)
            required.add(tensor_name(tensor_id, "trellis"))
            required.add(tensor_name(tensor_id, "mcg"))
            required.add(
                tensor_name(tensor_id, "svh" if projection != "down_proj" else "suh")
            )
            refs = manifest["vector_refs"][tensor_id.hf_prefix]
            expected_suh = (
                shared_gate_up_suh_name(layer)
                if projection != "down_proj"
                else tensor_name(tensor_id, "suh")
            )
            expected_svh = (
                tensor_name(tensor_id, "svh")
                if projection != "down_proj"
                else shared_down_svh_name(layer)
            )
            if refs != {"suh": expected_suh, "svh": expected_svh}:
                raise AssertionError(f"vector reference drift for {tensor_id.key}")
            bits = int(manifest["bit_map"][tensor_id.hf_prefix])
            expected_shape = (
                tensor_id.k // 16,
                tensor_id.n // 16,
                16 * bits,
            )
            actual_shape = reader.tensors[tensor_name(tensor_id, "trellis")].shape
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"{tensor_id.key}: trellis shape {actual_shape} != {expected_shape}"
                )
            trellis_info = reader.tensors[tensor_name(tensor_id, "trellis")]
            mcg_info = reader.tensors[tensor_name(tensor_id, "mcg")]
            vector_suffix = "svh" if projection != "down_proj" else "suh"
            vector_info = reader.tensors[tensor_name(tensor_id, vector_suffix)]
            expected_vector = tensor_id.n if projection != "down_proj" else tensor_id.k
            if trellis_info.dtype != "I16":
                raise ValueError(f"{tensor_id.key}: trellis dtype is not I16")
            if mcg_info.dtype != "I32" or mcg_info.shape != ():
                raise ValueError(f"{tensor_id.key}: MCG marker dtype/shape drift")
            if (
                int(read_torch_tensor(reader, mcg_info.name).item()) & 0xFFFFFFFF
                != MCG_MULT
            ):
                raise ValueError(f"{tensor_id.key}: MCG marker value drift")
            if vector_info.dtype != "F16" or vector_info.shape != (expected_vector,):
                raise ValueError(f"{tensor_id.key}: unique vector dtype/shape drift")
            vector_value = read_torch_tensor(reader, vector_info.name)
            if not torch.isfinite(vector_value).all() or (vector_value == 0).any():
                raise ValueError(
                    f"{tensor_id.key}: unique vector invalid after storage"
                )
            provenance = manifest["tensor_provenance"][tensor_id.hf_prefix]
            for binding in (
                "source_inventory_sha256",
                "numeric_environment_sha256",
                "runtime_inventory_sha256",
                "backend_fingerprint",
                "state_sha256",
                "capture_sha256",
                "search_sha256",
                "allocation_sha256",
                "probe_sha256",
            ):
                if provenance.get(binding) != manifest.get("provenance", {}).get(
                    binding
                ):
                    raise ValueError(f"{tensor_id.key}: layer/tensor {binding} drift")
            if provenance.get("source_name") != f"{tensor_id.hf_prefix}.weight":
                raise ValueError(f"{tensor_id.key}: BF16 source name drift")
            if (
                not provenance.get("source_shard")
                or type(provenance.get("source_payload_start")) is not int
                or type(provenance.get("source_payload_end")) is not int
                or provenance["source_payload_end"]
                <= provenance["source_payload_start"]
                or provenance["source_payload_end"] - provenance["source_payload_start"]
                != tensor_id.k * tensor_id.n * 2
            ):
                raise ValueError(f"{tensor_id.key}: BF16 source byte range missing")
            bf16_hash = provenance.get("bf16_sha256")
            if (
                not isinstance(bf16_hash, str)
                or len(bf16_hash) != 64
                or any(character not in "0123456789abcdef" for character in bf16_hash)
            ):
                raise ValueError(f"{tensor_id.key}: BF16 payload digest malformed")
            if provenance.get("mcg") != f"0x{MCG_MULT:08X}":
                raise ValueError(f"{tensor_id.key}: provenance codebook drift")
            if (
                int(provenance.get("full_k", -1)) != tensor_id.k
                or int(provenance.get("full_n", -1)) != tensor_id.n
            ):
                raise ValueError(f"{tensor_id.key}: provenance geometry drift")
    if set(reader.tensors) != required:
        missing = sorted(required - set(reader.tensors))
        extra = sorted(set(reader.tensors) - required)
        raise AssertionError(
            f"v2 tensor set mismatch missing={missing[:3]} extra={extra[:3]}"
        )
    payload_hashes = manifest["payload_sha256"]
    if set(payload_hashes) != set(reader.tensors):
        raise AssertionError("manifest payload hash set differs from shard")
    for name, info in reader.tensors.items():
        if info.payload.sha256() != payload_hashes[name]:
            raise AssertionError(f"payload hash mismatch: {name}")
    for expert in range(NUM_EXPERTS):
        for projection in PROJECTIONS:
            tensor_id = TensorId(layer, expert, projection)
            packed_name = tensor_name(tensor_id, "trellis")
            if (
                manifest["roundtrip_hashes"][tensor_id.hf_prefix]["packed_sha256"]
                != payload_hashes[packed_name]
            ):
                raise ValueError(f"{tensor_id.key}: packed hash provenance drift")
            if codec is not None and not parallel_reconstruction:
                packed = read_torch_tensor(reader, packed_name)
                refs = manifest["vector_refs"][tensor_id.hf_prefix]
                suh = read_torch_tensor(reader, refs["suh"])
                svh = read_torch_tensor(reader, refs["svh"])
                reconstructed = codec.decode_to_original(
                    packed.to(codec.config.device),
                    suh,
                    svh,
                    int(manifest["bit_map"][tensor_id.hf_prefix]),
                ).half()
                reconstructed_hash = sha256_bytes(
                    reconstructed.detach()
                    .contiguous()
                    .cpu()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                if (
                    reconstructed_hash
                    != manifest["roundtrip_hashes"][tensor_id.hf_prefix][
                        "reconstruction_sha256"
                    ]
                ):
                    raise ValueError(f"{tensor_id.key}: reconstruction hash drift")
                encoded = EncodedTensor(
                    tensor_id=tensor_id,
                    bits=int(manifest["bit_map"][tensor_id.hf_prefix]),
                    trellis=packed,
                    suh=suh,
                    svh=svh,
                    reconstructed_kn=None,
                    proxy_loss=0.0,
                    packed_sha256=payload_hashes[packed_name],
                    reconstruction_sha256=reconstructed_hash,
                    provenance=manifest["tensor_provenance"][tensor_id.hf_prefix],
                )
                for tp_size in tp_sizes:
                    sliced_reconstruction_oracle(encoded, codec, int(tp_size))
    if parallel_reconstruction:
        results = process_pool.map(
            "audit_v2",
            range(NUM_EXPERTS),
            {
                "manifest_path": path.resolve(),
                "manifest_sha256": sha256_file(path),
                "shard_path": shard.resolve(),
                "shard_sha256": str(manifest["shard_sha256"]),
                "layer": layer,
                "tp_sizes": tp_sizes,
            },
        )
        _validate_parallel_reconstruction_results(
            manifest,
            layer=layer,
            tp_sizes=tp_sizes,
            results=results,
        )
    return {
        "layer": layer,
        "tensor_count": len(reader.tensors),
        "bit_units": sum(int(value) for value in manifest["bit_map"].values()),
        "shared_gate_up_suh": shared_gate_up_suh_name(layer),
        "shared_down_svh": shared_down_svh_name(layer),
        "passed": True,
        "reconstruction_oracle": codec is not None,
        "tp_sizes": [int(value) for value in tp_sizes],
    }


def write_oracle_report(
    path: str | Path, reports: Iterable[Mapping[str, object]]
) -> None:
    values = list(reports)
    if not values or any(not value.get("passed") for value in values):
        raise ValueError("cannot seal incomplete oracle report")
    atomic_write_json(path, {"schema": "r7-oracles-v1", "reports": values})
