#!/usr/bin/env python3
"""Assemble a complete checkpoint while hashing every carried tensor payload."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from .constants import (
    FIRST_MOE_LAYER,
    LAST_MOE_LAYER,
    NUM_EXPERTS,
    RECIPE_MARKER,
    RECIPE_VERSION,
)
from .determinism import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .inventory import load_checkpoint_inventory, verify_checkpoint_inventory
from .oracles import audit_v2_layer
from .safetensors_io import SafeTensorReader, TensorEntry, write_safetensors_atomic

INDEX = "model.safetensors.index.json"
ASSEMBLY_TRANSACTION = "ASSEMBLY_TRANSACTION.json"
REGENERATED = {
    INDEX,
    "config.json",
    "quantization_config.json",
    "MANIFEST.json",
    "MANIFEST.sha256",
    ASSEMBLY_TRANSACTION,
}
ROUTED_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\."
)


def is_replaced_routed_tensor(name: str) -> bool:
    match = ROUTED_EXPERT.match(name)
    if match is None:
        return False
    layer = int(match.group(1))
    return FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER


def _validate_walk(walk_manifest: Path, v2: Path) -> dict[str, object]:
    walk = read_json(walk_manifest)
    if (
        walk.get("marker") != RECIPE_MARKER
        or walk.get("schema") != "r7-walk-complete-v1"
    ):
        raise ValueError("assembly requires a sealed complete Round 7 walk")
    expected = {str(layer) for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)}
    if set(walk.get("layers", {})) != expected:
        raise ValueError("walk completion manifest does not cover layers 3..77")
    for layer, record in walk["layers"].items():
        manifest = walk_manifest.parent / record["manifest"]
        oracle = walk_manifest.parent / record["oracle"]
        interaction = walk_manifest.parent / record["interaction_audit"]
        install = walk_manifest.parent / record["install_audit"]
        if manifest.parent.resolve() != v2.resolve():
            raise ValueError("walk manifest points outside the selected v2 directory")
        for path, key in (
            (manifest, "manifest_sha256"),
            (oracle, "oracle_sha256"),
            (interaction, "interaction_audit_sha256"),
            (install, "install_audit_sha256"),
        ):
            if sha256_file(path) != record[key]:
                raise ValueError(f"walk artifact hash drift for layer {layer}: {path}")
        report = read_json(oracle)
        if not report.get("reports") or any(
            not item.get("passed") for item in report["reports"]
        ):
            raise ValueError(f"layer {layer} lacks a passing oracle report")
        install_report = read_json(install)
        if (
            install_report.get("schema") != "r7-layer-install-audit-v2"
            or install_report.get("complete") is not True
            or set(install_report.get("experts", {}))
            != {str(expert) for expert in range(NUM_EXPERTS)}
            or any(
                not item.get("passed")
                for item in install_report.get("experts", {}).values()
            )
            or not isinstance(install_report.get("official_layer_audit"), dict)
            or install_report["official_layer_audit"].get("schema")
            != "r7-official-installed-layer-audit-v1"
            or install_report["official_layer_audit"].get("passed") is not True
        ):
            raise ValueError(
                f"layer {layer} lacks complete install arithmetic evidence"
            )
        audit_v2_layer(manifest)
        layer_manifest = read_json(manifest)
        provenance = layer_manifest.get("provenance", {})
        if provenance.get("install_audit_sha256") != record["install_audit_sha256"]:
            raise ValueError(f"layer {layer} install audit provenance drift")
        for key in (
            "carrier_inventory_sha256",
            "source_inventory_sha256",
            "numeric_environment_sha256",
            "runtime_inventory_sha256",
        ):
            if provenance.get(key) != walk.get(key):
                raise ValueError(
                    f"layer {layer} provenance differs from completed walk: {key}"
                )
    return walk


def _prepare_paths(carrier: Path, v2: Path, output: Path) -> Path:
    carrier = carrier.resolve()
    v2 = v2.resolve()
    output = output.resolve()
    for protected in (carrier, v2):
        if (
            output == protected
            or protected in output.parents
            or output in protected.parents
        ):
            raise ValueError(
                f"output must be disjoint from read-only source {protected}"
            )
    if output.exists():
        seal = output / "MANIFEST.sha256"
        manifest = output / "MANIFEST.json"
        if not seal.is_file() or not manifest.is_file():
            raise ValueError(f"unsealed output directory already exists: {output}")
        expected = seal.read_text().strip().split()[0]
        if sha256_file(manifest) != expected:
            raise ValueError("existing assembly manifest seal mismatch")
        _audit_assembled_output(output)
        return output
    partial = output.with_name(f".{output.name}.partial")
    partial.mkdir(parents=True, exist_ok=True)
    return partial


def _audit_assembled_output(output: Path) -> dict[str, object]:
    manifest = read_json(output / "MANIFEST.json")
    if (
        manifest.get("marker") != RECIPE_MARKER
        or manifest.get("schema") != "r7-complete-v2-checkpoint-v1"
    ):
        raise ValueError("foreign assembled output")
    for relative, expected in manifest.get("files_sha256", {}).items():
        if sha256_file(output / relative) != expected:
            raise ValueError(f"assembled output file drift: {relative}")
    quantization = read_json(output / "quantization_config.json")
    if read_json(output / "config.json").get("quantization_config") != quantization:
        raise ValueError("assembled output metadata surfaces differ")
    r7 = quantization.get("r7_routed_experts")
    expected_manifests = [
        f"r7-experts-layer-{layer:03d}.json"
        for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)
    ]
    if (
        not isinstance(r7, dict)
        or r7.get("feature") != "r7-asymmetric-two-stack"
        or r7.get("feature_version") != 1
        or r7.get("requires_loader_feature") != "r7-asymmetric-two-stack"
        or r7.get("schema_version") != 2
        or r7.get("bits") != "mixed_tensor"
        or r7.get("k_values") != [3, 4, 5]
        or r7.get("target_bpw") != "3.5"
        or r7.get("mtp_layer_78") != "carried"
        or r7.get("bit_map_manifests") != expected_manifests
    ):
        raise ValueError("assembled schema-v2 loader contract drift")
    index = read_json(output / INDEX)
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    observed = {}
    for shard_name in sorted(set(weight_map.values())):
        reader = SafeTensorReader(output / shard_name)
        for name in reader.tensors:
            if name in observed:
                raise ValueError(f"assembled tensor appears twice: {name}")
            observed[name] = shard_name
    if observed != weight_map:
        raise ValueError("assembled output index/shard bijection failed")
    for name, expected in manifest.get("carried_tensor_payload_sha256", {}).items():
        shard = output / weight_map[name]
        if SafeTensorReader(shard).tensors[name].payload.sha256() != expected:
            raise ValueError(f"assembled carried payload drift: {name}")
    for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1):
        audit_v2_layer(output / f"r7-experts-layer-{layer:03d}.json")
    return manifest


def _bind_assembly_transaction(
    partial: Path,
    *,
    carrier: Path,
    v2: Path,
    inventory_sha256: str,
    walk_manifest: Path,
    layer_manifest_sha256: dict[str, str],
) -> dict[str, object]:
    binding = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-assembly-transaction-v1",
        "carrier": str(carrier),
        "v2": str(v2),
        "carrier_inventory_sha256": inventory_sha256,
        "walk_manifest_sha256": sha256_file(walk_manifest),
        "layer_manifest_sha256": dict(sorted(layer_manifest_sha256.items())),
        "assembler_sha256": sha256_file(Path(__file__)),
    }
    binding["transaction_id"] = sha256_bytes(canonical_json_bytes(binding))
    path = partial / ASSEMBLY_TRANSACTION
    if path.exists():
        if read_json(path) != binding:
            raise ValueError("assembly partial belongs to a different transaction")
    else:
        if any(partial.iterdir()):
            raise ValueError("unbound nonempty assembly partial cannot be resumed")
        atomic_write_json(path, binding)
    return binding


def _load_v2(v2: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    manifests = []
    weight_map: dict[str, str] = {}
    for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1):
        path = v2 / f"r7-experts-layer-{layer:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        audit_v2_layer(path)
        manifest = json.loads(path.read_text())
        if (
            int(manifest.get("schema_version", -1)) != 2
            or int(manifest["layer"]) != layer
        ):
            raise ValueError(f"invalid v2 manifest {path}")
        if int(manifest.get("allocation_bit_units", -1)) != 2688:
            raise ValueError(f"layer {layer}: allocation is not exact 3.5 bpw")
        bit_values = tuple(manifest.get("bit_map", {}).values())
        if (
            len(bit_values) != 768
            or any(
                type(value) is not int or value not in (3, 4, 5) for value in bit_values
            )
            or sum(bit_values) != 2688
        ):
            raise ValueError(f"layer {layer}: invalid per-tensor bit map")
        shard = v2 / str(manifest["shard"])
        if sha256_file(shard) != manifest["shard_sha256"]:
            raise ValueError(f"v2 shard hash mismatch: {shard}")
        reader = SafeTensorReader(shard)
        for name in reader.tensors:
            if name in weight_map:
                raise ValueError(f"duplicate v2 tensor {name}")
            weight_map[name] = shard.name
        manifests.append(manifest)
    return manifests, weight_map


def assemble(
    carrier: Path,
    v2: Path,
    output: Path,
    *,
    carrier_inventory: Path,
    walk_manifest: Path,
) -> dict[str, object]:
    carrier = carrier.resolve()
    v2 = v2.resolve()
    final_output = output.resolve()
    output = _prepare_paths(carrier, v2, final_output)
    if output == final_output:
        return read_json(final_output / "MANIFEST.json")
    inventory = load_checkpoint_inventory(carrier_inventory, role="carrier")
    verify_checkpoint_inventory(carrier, inventory)
    walk = _validate_walk(walk_manifest.resolve(), v2)
    source_index = json.loads((carrier / INDEX).read_text())
    source_map = {
        str(key): str(value) for key, value in source_index["weight_map"].items()
    }
    manifests, v2_map = _load_v2(v2)
    layer_manifest_sha256 = {
        str(item["layer"]): sha256_file(
            v2 / f"r7-experts-layer-{int(item['layer']):03d}.json"
        )
        for item in manifests
    }
    transaction = _bind_assembly_transaction(
        output,
        carrier=carrier,
        v2=v2,
        inventory_sha256=str(inventory["inventory_sha256"]),
        walk_manifest=walk_manifest.resolve(),
        layer_manifest_sha256=layer_manifest_sha256,
    )
    expected_top_level = {
        ASSEMBLY_TRANSACTION,
        "config.json",
        "quantization_config.json",
        INDEX,
        "MANIFEST.json",
        "MANIFEST.sha256",
    }

    # Copy non-model assets first. All loader metadata, indexes, and manifests
    # are regenerated; stale carrier copies are never allowed to survive.
    source_shards = set(source_map.values())
    for path in sorted(carrier.iterdir(), key=lambda item: item.name):
        if path.name in REGENERATED or path.name in source_shards:
            continue
        destination = output / path.name
        expected_top_level.add(path.name)
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
            source_files = {
                item.relative_to(path) for item in path.rglob("*") if item.is_file()
            }
            destination_files = {
                item.relative_to(destination)
                for item in destination.rglob("*")
                if item.is_file()
            }
            if destination_files != source_files:
                raise ValueError(
                    f"resumed asset directory has stale files: {path.name}"
                )
            for source_file in sorted(
                item for item in path.rglob("*") if item.is_file()
            ):
                relative = source_file.relative_to(path)
                if sha256_file(destination / relative) != sha256_file(source_file):
                    raise AssertionError(
                        f"non-model asset copy drift: {path.name}/{relative}"
                    )
        else:
            shutil.copy2(path, destination)

    carried_hashes: dict[str, str] = {}
    output_map: dict[str, str] = {}
    output_file_hashes: dict[str, str] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in source_map.items():
        if not is_replaced_routed_tensor(name):
            by_shard.setdefault(shard, []).append(name)

    for shard_name in sorted(source_shards):
        source_shard = carrier / shard_name
        reader = SafeTensorReader(source_shard)
        keep = sorted(by_shard.get(shard_name, []))
        dropped = set(reader.tensors) - set(keep)
        if any(not is_replaced_routed_tensor(name) for name in dropped):
            unexpected = sorted(
                name for name in dropped if not is_replaced_routed_tensor(name)
            )
            raise ValueError(f"assembly would drop carried tensors: {unexpected[:3]}")
        destination = output / shard_name
        if not keep:
            # The source shard contained only replaced layers 3..77 expert
            # payload. It must not survive as an unindexed empty model shard.
            continue
        expected_top_level.add(shard_name)
        if not dropped:
            shutil.copy2(source_shard, destination)
            if sha256_file(source_shard) != sha256_file(destination):
                raise AssertionError(f"byte-exact shard copy failed: {shard_name}")
            destination_reader = SafeTensorReader(destination)
            for name in keep:
                source_hash = reader.tensors[name].payload.sha256()
                reread_hash = destination_reader.tensors[name].payload.sha256()
                if reread_hash != source_hash:
                    raise AssertionError(f"carried tensor reread failed: {name}")
                carried_hashes[name] = source_hash
                output_map[name] = shard_name
            output_file_hashes[shard_name] = sha256_file(destination)
            continue

        entries = [
            TensorEntry(
                name,
                reader.tensors[name].dtype,
                reader.tensors[name].shape,
                reader.tensors[name].payload,
            )
            for name in keep
        ]
        written_hashes, file_hash = write_safetensors_atomic(
            destination, entries, metadata=reader.metadata
        )
        destination_reader = SafeTensorReader(destination)
        for name in keep:
            source_hash = reader.tensors[name].payload.sha256()
            if written_hashes[name] != source_hash:
                raise AssertionError(f"carried tensor payload changed: {name}")
            if destination_reader.tensors[name].payload.sha256() != source_hash:
                raise AssertionError(f"rewritten carried tensor reread failed: {name}")
            carried_hashes[name] = source_hash
            output_map[name] = shard_name
        output_file_hashes[shard_name] = file_hash

    # R7 layer shards are copied byte-exact; their internal tensors become new
    # index entries. No existing carried tensor may share a name.
    for shard_name in sorted(set(v2_map.values())):
        source_shard = v2 / shard_name
        destination = output / shard_name
        expected_top_level.add(shard_name)
        shutil.copy2(source_shard, destination)
        if sha256_file(source_shard) != sha256_file(destination):
            raise AssertionError(f"v2 shard copy failed: {shard_name}")
        output_file_hashes[shard_name] = sha256_file(destination)
    for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1):
        manifest_name = f"r7-experts-layer-{layer:03d}.json"
        source_manifest = v2 / manifest_name
        destination_manifest = output / manifest_name
        expected_top_level.add(manifest_name)
        shutil.copy2(source_manifest, destination_manifest)
        if sha256_file(source_manifest) != sha256_file(destination_manifest):
            raise AssertionError(f"v2 manifest copy failed: {manifest_name}")
        output_file_hashes[manifest_name] = sha256_file(destination_manifest)
    collisions = set(output_map) & set(v2_map)
    if collisions:
        raise ValueError(
            f"v2 tensor collides with carried tensor: {sorted(collisions)[:3]}"
        )
    output_map.update(v2_map)

    inventory_entries = inventory["entries"]
    for name, digest in carried_hashes.items():
        record = inventory_entries.get(name)
        if not isinstance(record, dict) or record.get("payload_sha256") != digest:
            raise ValueError(
                f"carried tensor differs from sealed carrier inventory: {name}"
            )
    mtp78_hashes = {
        name: digest
        for name, digest in carried_hashes.items()
        if name.startswith("model.layers.78.")
    }
    if not mtp78_hashes:
        raise ValueError("carrier inventory/output contains no layer-78 payload")

    config = json.loads((carrier / "config.json").read_text())
    legacy_tail = config.get("hybrid_tr3_tail")
    if not isinstance(legacy_tail, dict):
        raise ValueError("carrier lacks the layer-78 legacy tail metadata")
    # Layer 78 is owner-locked carried payload. Keep the old loader metadata
    # only for that layer; leaving [3,78] would make it hunt for deleted v1
    # names in the newly schema-v2 layers.
    legacy_tail = dict(legacy_tail)
    legacy_tail["moe_layers"] = [78, 78]
    legacy_tail["scope"] = {
        "carried": "layer 78 MTP routed experts, byte-exact from carrier",
        "r7_replaced": "layers 3..77 are described by r7_routed_experts",
    }
    legacy_tail["r7_compat_role"] = "carried_mtp78_only"
    config["hybrid_tr3_tail"] = legacy_tail
    quantization = dict(config.get("quantization_config") or {})
    quantization["r7_routed_experts"] = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-complete-v2-checkpoint-v1",
        "feature": "r7-asymmetric-two-stack",
        "feature_version": 1,
        "requires_loader_feature": "r7-asymmetric-two-stack",
        "schema_version": 2,
        "moe_layers": [FIRST_MOE_LAYER, LAST_MOE_LAYER],
        "mtp_layer_78": "carried",
        "bits": "mixed_tensor",
        "k_values": [3, 4, 5],
        "target_bpw": "3.5",
        "codebook": "mcg",
        "tp_slice_quantum": 128,
        "bit_map_manifests": [
            f"r7-experts-layer-{layer:03d}.json"
            for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)
        ],
    }
    config["quantization_config"] = quantization
    atomic_write_json(output / "config.json", config)
    output_file_hashes["config.json"] = sha256_file(output / "config.json")
    atomic_write_json(output / "quantization_config.json", quantization)
    output_file_hashes["quantization_config.json"] = sha256_file(
        output / "quantization_config.json"
    )
    if read_json(output / "config.json")["quantization_config"] != read_json(
        output / "quantization_config.json"
    ):
        raise AssertionError("embedded and external quantization metadata differ")

    # Index audit: every mapping resolves and every shard tensor is indexed.
    total_payload_bytes = 0
    observed: dict[str, str] = {}
    for shard_name in sorted(set(output_map.values())):
        reader = SafeTensorReader(output / shard_name)
        for name, info in reader.tensors.items():
            if name in observed:
                raise ValueError(f"tensor appears in multiple output shards: {name}")
            observed[name] = shard_name
            total_payload_bytes += info.nbytes
    if observed != output_map:
        missing = sorted(set(output_map) - set(observed))
        extra = sorted(set(observed) - set(output_map))
        raise ValueError(
            f"index/shard mismatch missing={missing[:3]} extra={extra[:3]}"
        )
    index = {
        "metadata": {"total_size": total_payload_bytes},
        "weight_map": dict(sorted(output_map.items())),
    }
    atomic_write_json(output / INDEX, index)
    output_file_hashes[INDEX] = sha256_file(output / INDEX)

    observed_top_level = {path.name for path in output.iterdir()}
    if not observed_top_level <= expected_top_level:
        raise ValueError(
            "assembly partial contains stale top-level entries: "
            f"{sorted(observed_top_level - expected_top_level)[:8]}"
        )
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if (
            path.is_file()
            and path.name not in output_file_hashes
            and path.name != "MANIFEST.json"
        ):
            output_file_hashes[path.name] = sha256_file(path)
    manifest = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-complete-v2-checkpoint-v1",
        "carrier": str(carrier),
        "carrier_inventory_sha256": inventory["inventory_sha256"],
        "carrier_index_sha256": inventory["index_sha256"],
        "walk_manifest_sha256": sha256_file(walk_manifest),
        "source_inventory_sha256": walk["source_inventory_sha256"],
        "numeric_environment_sha256": walk["numeric_environment_sha256"],
        "carried_tensor_payload_sha256": dict(sorted(carried_hashes.items())),
        "files_sha256": dict(sorted(output_file_hashes.items())),
        "tensor_count": len(output_map),
        "carried_tensor_count": len(carried_hashes),
        "r7_tensor_count": len(v2_map),
        "mtp78_carried_tensor_payload_sha256": dict(sorted(mtp78_hashes.items())),
        "assembly_transaction_id": transaction["transaction_id"],
        "r7_layer_manifest_sha256": layer_manifest_sha256,
    }
    atomic_write_json(output / "MANIFEST.json", manifest)
    manifest_hash = sha256_file(output / "MANIFEST.json")
    atomic_write_bytes(
        output / "MANIFEST.sha256",
        f"{manifest_hash}  MANIFEST.json\n".encode("ascii"),
    )
    _audit_assembled_output(output)
    verify_checkpoint_inventory(carrier, inventory)
    os.replace(output, final_output)
    parent_fd = os.open(final_output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    _audit_assembled_output(final_output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--carrier-inventory", type=Path, required=True)
    parser.add_argument("--walk-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = assemble(
        args.carrier,
        args.v2,
        args.out,
        carrier_inventory=args.carrier_inventory,
        walk_manifest=args.walk_manifest,
    )
    print(
        json.dumps(
            {
                "tensor_count": manifest["tensor_count"],
                "carried_tensor_count": manifest["carried_tensor_count"],
                "r7_tensor_count": manifest["r7_tensor_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
