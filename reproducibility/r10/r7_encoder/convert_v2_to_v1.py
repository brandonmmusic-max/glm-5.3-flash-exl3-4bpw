#!/usr/bin/env python3
"""Convert topology-neutral R7 schema v2 to TP rank-sliced schema v1.

The converter performs no quantization. It slices packed tiles and the matching
vector on exact 128-coordinate boundaries, and replicates the two layer-shared
vectors into the historical per-expert/rank names. Asymmetric projection bits
require the loader-only dispatch described in R7_SERVING_SPEC.md; use
`--assert-unmodified-r13` to fail unless a checkpoint is genuinely scalar-bit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Callable

from .constants import (
    FIRST_MOE_LAYER,
    LAST_MOE_LAYER,
    MCG_MULT,
    NUM_EXPERTS,
    PROJECTIONS,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TensorId,
)
from .determinism import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .oracles import audit_v2_layer
from .safetensors_io import SafeTensorReader, TensorEntry, write_safetensors_atomic
from .schema import (
    load_time_tp_slices,
    shared_down_svh_name,
    shared_gate_up_suh_name,
    tensor_name,
)

INDEX = "model.safetensors.index.json"
CONVERSION_TRANSACTION = "CONVERSION_TRANSACTION.json"
REGENERATED = {
    INDEX,
    "config.json",
    "quantization_config.json",
    "MANIFEST.json",
    "MANIFEST.sha256",
    "ASSEMBLY_TRANSACTION.json",
    CONVERSION_TRANSACTION,
}


def _tensor_payload_factory(path: Path, name: str, slicer: Callable | None = None):
    def payload() -> bytes:
        import torch
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
        if slicer is not None:
            value = slicer(value)
        return value.detach().contiguous().view(torch.uint8).numpy().tobytes()

    return payload


def _rank_name(tensor_id: TensorId, rank: int, suffix: str) -> str:
    return f"{tensor_id.hf_prefix}.rank{rank}.{suffix}"


def _assert_scalar_bit_compat(bit_map: dict[str, int]) -> int:
    if (
        len(bit_map) != NUM_EXPERTS * len(PROJECTIONS)
        or any(
            type(bits) is not int or bits not in (3, 4, 5) for bits in bit_map.values()
        )
        or sum(bit_map.values()) != 2688
    ):
        raise ValueError(
            "unmodified-r13 compatibility may be assessed only for a valid "
            "768-entry Round 7 map with widths {3,4,5} and sum 2688"
        )
    raise ValueError(
        "exact 3.5 bpw over 768 integer-width tensors cannot be represented "
        "by unmodified r13's single scalar K: a uniform integer width sums "
        "to 2304 (K=3), 3072 (K=4), or 3840 (K=5). Install the mixed-bit "
        "v1 loader shim described in R7_SERVING_SPEC.md."
    )


def convert_layer(
    *,
    manifest_path: Path,
    output_dir: Path,
    tp_size: int,
    assert_unmodified_r13: bool,
) -> dict[str, object]:
    audit_v2_layer(manifest_path)
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError(f"{manifest_path}: not R7 schema v2")
    layer = int(manifest["layer"])
    source_shard = manifest_path.parent / manifest["shard"]
    bit_map = {str(key): int(value) for key, value in manifest["bit_map"].items()}
    if (
        len(bit_map) != NUM_EXPERTS * len(PROJECTIONS)
        or sum(bit_map.values()) != 2688
        or any(value not in (3, 4, 5) for value in bit_map.values())
    ):
        raise ValueError("v2->v1 conversion requires exact per-layer 3.5 bpw")
    scalar_bits = _assert_scalar_bit_compat(bit_map) if assert_unmodified_r13 else None
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_sha256 = sha256_file(manifest_path)
    result_path = output_dir / f"r7-v1-layer-{layer:03d}.json"
    if result_path.exists():
        incumbent = read_json(result_path)
        if (
            incumbent.get("layer") != layer
            or incumbent.get("tp_size") != tp_size
            or incumbent.get("source_manifest_sha256") != source_manifest_sha256
        ):
            raise ValueError("existing v1 layer conversion binding drift")
        shards = incumbent.get("shards")
        if not isinstance(shards, list) or len(shards) != tp_size:
            raise ValueError("existing v1 layer conversion is incomplete")
        for record in shards:
            path = output_dir / str(record["file"])
            reader = SafeTensorReader(path)
            if (
                sha256_file(path) != record.get("sha256")
                or set(reader.tensors) != set(record.get("payload_sha256", {}))
                or any(
                    reader.tensors[name].payload.sha256() != digest
                    for name, digest in record["payload_sha256"].items()
                )
            ):
                raise ValueError("existing v1 layer shard failed adoption audit")
        return incumbent

    shard_records: list[dict[str, object]] = []
    for rank in range(tp_size):
        entries: list[TensorEntry] = []
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                tensor_id = TensorId(layer, expert, projection)
                bits = bit_map.get(tensor_id.hf_prefix)
                if bits is None:
                    raise ValueError(f"missing bit-map entry {tensor_id.hf_prefix}")
                split = load_time_tp_slices(tensor_id, tp_size)[rank]
                if projection != "down_proj":
                    trellis_shape = (
                        tensor_id.k // 16,
                        (tensor_id.n // tp_size) // 16,
                        16 * bits,
                    )

                    def trellis_slice(value, s=split):
                        return value[:, s.trellis_start : s.trellis_end, :]

                    def vector_slice(value, s=split):
                        return value[s.vector_start : s.vector_end]

                    suh_source = shared_gate_up_suh_name(layer)
                    svh_source = tensor_name(tensor_id, "svh")
                    entries.extend(
                        [
                            TensorEntry(
                                _rank_name(tensor_id, rank, "trellis"),
                                "I16",
                                trellis_shape,
                                _tensor_payload_factory(
                                    source_shard,
                                    tensor_name(tensor_id, "trellis"),
                                    trellis_slice,
                                ),
                            ),
                            TensorEntry(
                                _rank_name(tensor_id, rank, "suh"),
                                "F16",
                                (tensor_id.k,),
                                _tensor_payload_factory(source_shard, suh_source),
                            ),
                            TensorEntry(
                                _rank_name(tensor_id, rank, "svh"),
                                "F16",
                                (tensor_id.n // tp_size,),
                                _tensor_payload_factory(
                                    source_shard, svh_source, vector_slice
                                ),
                            ),
                        ]
                    )
                else:
                    trellis_shape = (
                        (tensor_id.k // tp_size) // 16,
                        tensor_id.n // 16,
                        16 * bits,
                    )

                    def trellis_slice(value, s=split):
                        return value[s.trellis_start : s.trellis_end, :, :]

                    def vector_slice(value, s=split):
                        return value[s.vector_start : s.vector_end]

                    entries.extend(
                        [
                            TensorEntry(
                                _rank_name(tensor_id, rank, "trellis"),
                                "I16",
                                trellis_shape,
                                _tensor_payload_factory(
                                    source_shard,
                                    tensor_name(tensor_id, "trellis"),
                                    trellis_slice,
                                ),
                            ),
                            TensorEntry(
                                _rank_name(tensor_id, rank, "suh"),
                                "F16",
                                (tensor_id.k // tp_size,),
                                _tensor_payload_factory(
                                    source_shard,
                                    tensor_name(tensor_id, "suh"),
                                    vector_slice,
                                ),
                            ),
                            TensorEntry(
                                _rank_name(tensor_id, rank, "svh"),
                                "F16",
                                (tensor_id.n,),
                                _tensor_payload_factory(
                                    source_shard, shared_down_svh_name(layer)
                                ),
                            ),
                        ]
                    )
                marker_bytes = int(MCG_MULT).to_bytes(4, "little", signed=False)
                entries.append(
                    TensorEntry(
                        _rank_name(tensor_id, rank, "mcg"),
                        "I32",
                        (),
                        marker_bytes,
                    )
                )

        destination = (
            output_dir / f"r7-v1-layer-{layer:03d}-rank-{rank:02d}.safetensors"
        )
        payload_hashes, file_hash = write_safetensors_atomic(
            destination,
            entries,
            metadata={
                "format": "pt",
                "r7_schema": "1",
                "r7_source_schema": "2",
                "r7_tp_size": str(tp_size),
                "r7_tp_rank": str(rank),
            },
        )
        shard_records.append(
            {
                "rank": rank,
                "file": destination.name,
                "sha256": file_hash,
                "payload_sha256": payload_hashes,
            }
        )

    # Reassemble every emitted packed tensor from rank files and compare it to
    # the full schema-v2 source; also verify replicated shared-vector bytes and
    # the actual MCG marker value, not merely its presence.
    import torch
    from safetensors import safe_open

    rank_paths = [output_dir / record["file"] for record in shard_records]
    for expert in range(NUM_EXPERTS):
        for projection in PROJECTIONS:
            tensor_id = TensorId(layer, expert, projection)
            source_name = tensor_name(tensor_id, "trellis")
            with safe_open(source_shard, framework="pt", device="cpu") as handle:
                source_trellis = handle.get_tensor(source_name)
                source_unique = handle.get_tensor(
                    tensor_name(
                        tensor_id, "svh" if projection != "down_proj" else "suh"
                    )
                )
                source_shared = handle.get_tensor(
                    shared_gate_up_suh_name(layer)
                    if projection != "down_proj"
                    else shared_down_svh_name(layer)
                )
            pieces = []
            vector_pieces = []
            for rank, rank_path in enumerate(rank_paths):
                with safe_open(rank_path, framework="pt", device="cpu") as handle:
                    pieces.append(
                        handle.get_tensor(_rank_name(tensor_id, rank, "trellis"))
                    )
                    vector_pieces.append(
                        handle.get_tensor(
                            _rank_name(
                                tensor_id,
                                rank,
                                "svh" if projection != "down_proj" else "suh",
                            )
                        )
                    )
                    replicated = handle.get_tensor(
                        _rank_name(
                            tensor_id,
                            rank,
                            "suh" if projection != "down_proj" else "svh",
                        )
                    )
                    if not torch.equal(replicated, source_shared):
                        raise AssertionError(
                            f"{tensor_id.key}: shared-vector replication drift"
                        )
                    marker = int(
                        handle.get_tensor(_rank_name(tensor_id, rank, "mcg")).item()
                    )
                    if marker & 0xFFFFFFFF != MCG_MULT:
                        raise ValueError("v1 converter emitted an invalid MCG marker")
            axis = 1 if projection != "down_proj" else 0
            if not torch.equal(torch.cat(pieces, dim=axis), source_trellis):
                raise AssertionError(
                    f"{tensor_id.key}: rank-sliced trellis reassembly drift"
                )
            if not torch.equal(torch.cat(vector_pieces), source_unique):
                raise AssertionError(
                    f"{tensor_id.key}: rank-sliced vector reassembly drift"
                )
    for record in shard_records:
        reader = SafeTensorReader(output_dir / record["file"])
        if sha256_file(output_dir / record["file"]) != record["sha256"]:
            raise AssertionError("v1 rank shard file hash drift after publication")
        if set(reader.tensors) != set(record["payload_sha256"]):
            raise AssertionError("v1 rank shard tensor set drift")

    result = {
        "layer": layer,
        "schema_version": 1,
        "source_schema_version": 2,
        "source_manifest_sha256": source_manifest_sha256,
        "source_shard_sha256": str(manifest["shard_sha256"]),
        "tp_size": tp_size,
        "bits": scalar_bits if scalar_bits is not None else "mixed_tensor",
        "k_values": sorted(set(bit_map.values())),
        "per_tensor_bits": dict(sorted(bit_map.items())),
        "shared_vectors_replicated": True,
        "requires_loader_feature": (
            None if scalar_bits is not None else "r7-asymmetric-two-stack"
        ),
        "loader_feature_version": 1,
        "shards": shard_records,
    }
    atomic_write_json(result_path, result)
    return result


def _audit_complete_input(checkpoint: Path) -> dict[str, object]:
    seal = checkpoint / "MANIFEST.sha256"
    manifest_path = checkpoint / "MANIFEST.json"
    if not seal.is_file() or not manifest_path.is_file():
        raise ValueError("v2->v1 conversion requires a sealed assembled checkpoint")
    expected = seal.read_text().strip().split()[0]
    if sha256_file(manifest_path) != expected:
        raise ValueError("assembled checkpoint manifest seal mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("marker") != RECIPE_MARKER or manifest.get("schema") not in {
        "r7-complete-v2-checkpoint-v1",
        "r7-complete-v1-checkpoint-v1",
    }:
        raise ValueError("foreign assembled checkpoint")
    for relative, digest in manifest.get("files_sha256", {}).items():
        if sha256_file(checkpoint / relative) != digest:
            raise ValueError(f"assembled checkpoint file drift: {relative}")
    quantization = read_json(checkpoint / "quantization_config.json")
    if read_json(checkpoint / "config.json").get("quantization_config") != quantization:
        raise ValueError("checkpoint embedded/external quantization metadata drift")
    r7 = quantization.get("r7_routed_experts")
    expected_schema_version = (
        2 if manifest.get("schema") == "r7-complete-v2-checkpoint-v1" else 1
    )
    manifest_prefix = (
        "r7-experts-layer" if expected_schema_version == 2 else "r7-v1-layer"
    )
    expected_manifests = [
        f"{manifest_prefix}-{layer:03d}.json"
        for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)
    ]
    if (
        not isinstance(r7, dict)
        or r7.get("feature") != "r7-asymmetric-two-stack"
        or r7.get("feature_version") != 1
        or r7.get("requires_loader_feature") != "r7-asymmetric-two-stack"
        or r7.get("schema_version") != expected_schema_version
        or r7.get("bits") != "mixed_tensor"
        or r7.get("k_values") != [3, 4, 5]
        or r7.get("target_bpw") != "3.5"
        or r7.get("mtp_layer_78") != "carried"
        or r7.get("bit_map_manifests") != expected_manifests
    ):
        raise ValueError("checkpoint asymmetric-loader metadata contract drift")
    index = read_json(checkpoint / INDEX)
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    observed = {}
    for shard_name in sorted(set(weight_map.values())):
        reader = SafeTensorReader(checkpoint / shard_name)
        for name in reader.tensors:
            if name in observed:
                raise ValueError(f"checkpoint tensor appears twice: {name}")
            observed[name] = shard_name
    if observed != weight_map:
        raise ValueError("checkpoint index/shard bijection failed")
    for name, digest in manifest.get("carried_tensor_payload_sha256", {}).items():
        if (
            SafeTensorReader(checkpoint / weight_map[name])
            .tensors[name]
            .payload.sha256()
            != digest
        ):
            raise ValueError(f"checkpoint carried payload drift: {name}")
    if manifest.get("schema") == "r7-complete-v1-checkpoint-v1":
        if set(manifest.get("layers", {})) != {
            str(layer) for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)
        }:
            raise ValueError("complete v1 conversion lacks layers 3..77")
    return manifest


def _bind_conversion_transaction(
    partial: Path, *, source: Path, source_manifest_sha256: str, tp_size: int
) -> dict[str, object]:
    binding = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-conversion-transaction-v1",
        "source": str(source),
        "source_manifest_sha256": source_manifest_sha256,
        "tp_size": tp_size,
        "converter_sha256": sha256_file(Path(__file__)),
    }
    binding["transaction_id"] = sha256_bytes(canonical_json_bytes(binding))
    path = partial / CONVERSION_TRANSACTION
    if path.exists():
        if read_json(path) != binding:
            raise ValueError("conversion partial belongs to a different transaction")
    else:
        if any(partial.iterdir()):
            raise ValueError("unbound nonempty conversion partial cannot be resumed")
        atomic_write_json(path, binding)
    return binding


def convert_checkpoint(
    *,
    checkpoint: Path,
    output_dir: Path,
    tp_size: int,
    assert_unmodified_r13: bool = False,
) -> dict[str, object]:
    """Convert all 75 v2 layers and publish a complete loadable checkpoint."""

    source = checkpoint.resolve()
    final = output_dir.resolve()
    if final == source or final in source.parents or source in final.parents:
        raise ValueError("conversion output must be disjoint from the v2 checkpoint")
    source_manifest = _audit_complete_input(source)
    if source_manifest.get("schema") != "r7-complete-v2-checkpoint-v1":
        raise ValueError("complete conversion input must use Round 7 schema v2")
    if final.exists():
        existing = _audit_complete_input(final)
        if existing.get("conversion_source_manifest_sha256") != sha256_file(
            source / "MANIFEST.json"
        ):
            raise ValueError("existing v1 output was converted from different bytes")
        return existing
    partial = final.with_name(f".{final.name}.partial")
    partial.mkdir(parents=True, exist_ok=True)
    source_manifest_sha256 = sha256_file(source / "MANIFEST.json")
    transaction = _bind_conversion_transaction(
        partial,
        source=source,
        source_manifest_sha256=source_manifest_sha256,
        tp_size=tp_size,
    )

    layer_manifests = []
    replaced_names: set[str] = set()
    v2_shards: set[str] = set()
    for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1):
        manifest_path = source / f"r7-experts-layer-{layer:03d}.json"
        audit_v2_layer(manifest_path)
        manifest = read_json(manifest_path)
        bit_map = {str(key): int(value) for key, value in manifest["bit_map"].items()}
        if assert_unmodified_r13:
            _assert_scalar_bit_compat(bit_map)
        if len(bit_map) != 768 or sum(bit_map.values()) != 2688:
            raise ValueError(f"layer {layer}: conversion input is not exact 3.5 bpw")
        reader = SafeTensorReader(source / manifest["shard"])
        replaced_names.update(reader.tensors)
        v2_shards.add(str(manifest["shard"]))
        layer_manifests.append((manifest_path, manifest))

    source_index = read_json(source / INDEX)
    source_map = {
        str(name): str(shard) for name, shard in source_index["weight_map"].items()
    }
    source_shards = set(source_map.values())
    if not replaced_names <= set(source_map):
        raise ValueError("assembled index omits v2 tensors")
    output_map: dict[str, str] = {}
    carried_hashes: dict[str, str] = {}
    file_hashes: dict[str, str] = {}

    for path in sorted(source.iterdir(), key=lambda item: item.name):
        if (
            path.name in REGENERATED
            or path.name in source_shards
            or path.name in v2_shards
            or path.name.startswith("r7-experts-layer-")
        ):
            continue
        destination = partial / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)

    for shard_name in sorted(source_shards):
        source_shard = source / shard_name
        reader = SafeTensorReader(source_shard)
        keep = sorted(name for name in reader.tensors if name not in replaced_names)
        if not keep:
            continue
        destination = partial / shard_name
        if len(keep) == len(reader.tensors):
            shutil.copy2(source_shard, destination)
        else:
            entries = [
                TensorEntry(
                    name,
                    reader.tensors[name].dtype,
                    reader.tensors[name].shape,
                    reader.tensors[name].payload,
                )
                for name in keep
            ]
            write_safetensors_atomic(destination, entries, metadata=reader.metadata)
        destination_reader = SafeTensorReader(destination)
        for name in keep:
            source_hash = reader.tensors[name].payload.sha256()
            if destination_reader.tensors[name].payload.sha256() != source_hash:
                raise AssertionError(f"v1 conversion changed carried tensor {name}")
            carried_hashes[name] = source_hash
            output_map[name] = shard_name
        file_hashes[shard_name] = sha256_file(destination)

    conversions = {}
    for manifest_path, manifest in layer_manifests:
        result = convert_layer(
            manifest_path=manifest_path,
            output_dir=partial,
            tp_size=tp_size,
            assert_unmodified_r13=False,
        )
        conversions[str(result["layer"])] = {
            "manifest": f"r7-v1-layer-{int(result['layer']):03d}.json",
            "manifest_sha256": sha256_file(
                partial / f"r7-v1-layer-{int(result['layer']):03d}.json"
            ),
            "source_manifest_sha256": sha256_file(manifest_path),
        }
        for shard_record in result["shards"]:
            shard_name = str(shard_record["file"])
            reader = SafeTensorReader(partial / shard_name)
            file_hashes[shard_name] = sha256_file(partial / shard_name)
            for name in reader.tensors:
                if name in output_map:
                    raise ValueError(f"v1 tensor collision: {name}")
                output_map[name] = shard_name

    config = read_json(source / "config.json")
    quantization = dict(config.get("quantization_config") or {})
    r7 = dict(quantization.get("r7_routed_experts") or {})
    r7.update(
        {
            "feature": "r7-asymmetric-two-stack",
            "feature_version": 1,
            "schema_version": 1,
            "source_schema_version": 2,
            "tp_size": tp_size,
            "bits": "mixed_tensor",
            "target_bpw": "3.5",
            "requires_loader_feature": "r7-asymmetric-two-stack",
            "k_values": [3, 4, 5],
            "mtp_layer_78": "carried",
            "bit_map_manifests": [
                f"r7-v1-layer-{layer:03d}.json"
                for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)
            ],
        }
    )
    quantization["r7_routed_experts"] = r7
    config["quantization_config"] = quantization
    atomic_write_json(partial / "config.json", config)
    atomic_write_json(partial / "quantization_config.json", quantization)
    if read_json(partial / "config.json")["quantization_config"] != read_json(
        partial / "quantization_config.json"
    ):
        raise AssertionError("v1 embedded/external quantization metadata drift")
    file_hashes["config.json"] = sha256_file(partial / "config.json")
    file_hashes["quantization_config.json"] = sha256_file(
        partial / "quantization_config.json"
    )

    observed = {}
    total_size = 0
    for shard_name in sorted(set(output_map.values())):
        reader = SafeTensorReader(partial / shard_name)
        for name, info in reader.tensors.items():
            if name in observed:
                raise ValueError(f"v1 tensor appears in multiple shards: {name}")
            observed[name] = shard_name
            total_size += info.nbytes
    if observed != output_map:
        raise ValueError("v1 index map does not exactly equal output tensor set")
    atomic_write_json(
        partial / INDEX,
        {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(output_map.items())),
        },
    )
    file_hashes[INDEX] = sha256_file(partial / INDEX)
    for path in sorted(partial.rglob("*")):
        if (
            path.is_file()
            and path.parent == partial
            and path.name
            not in {
                "MANIFEST.json",
                "MANIFEST.sha256",
            }
        ):
            file_hashes.setdefault(path.name, sha256_file(path))
    result = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-complete-v1-checkpoint-v1",
        "conversion_source": str(source),
        "conversion_source_manifest_sha256": sha256_file(source / "MANIFEST.json"),
        "source_walk_manifest_sha256": source_manifest.get("walk_manifest_sha256"),
        "tp_size": tp_size,
        "layers": conversions,
        "carried_tensor_payload_sha256": dict(sorted(carried_hashes.items())),
        "tensor_count": len(output_map),
        "files_sha256": dict(sorted(file_hashes.items())),
        "requires_loader_feature": "r7-asymmetric-two-stack",
        "loader_feature_version": 1,
        "conversion_transaction_id": transaction["transaction_id"],
    }
    atomic_write_json(partial / "MANIFEST.json", result)
    manifest_hash = sha256_file(partial / "MANIFEST.json")
    atomic_write_bytes(
        partial / "MANIFEST.sha256",
        f"{manifest_hash}  MANIFEST.json\n".encode("ascii"),
    )
    _audit_complete_input(partial)
    _audit_complete_input(source)
    os.replace(partial, final)
    parent_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    _audit_complete_input(final)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--assert-unmodified-r13", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint is not None:
        result = convert_checkpoint(
            checkpoint=args.checkpoint,
            output_dir=args.out,
            tp_size=args.tp,
            assert_unmodified_r13=args.assert_unmodified_r13,
        )
    else:
        result = convert_layer(
            manifest_path=args.manifest,
            output_dir=args.out,
            tp_size=args.tp,
            assert_unmodified_r13=args.assert_unmodified_r13,
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "shards"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
