from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..codecs.uniform import QuantizedArray, dequantize, quantize
from ..core.artifacts import prepare_empty_destination, sha256_file, write_json
from ..models.inventory import QuantUnit, load_inventory


def _dependency_error(package: str, error: Exception) -> RuntimeError:
    return RuntimeError(f"{package} is required for checkpoint operations; install quant-pipeline[hf]: {error}")


def _safe_key(unit_id: str, field: str) -> str:
    return f"{unit_id}.{field}"


def encode_reference_checkpoint(
    model_path: str | Path,
    family: str,
    allocation_path: str | Path,
    output_dir: str | Path,
    group_size: int,
) -> dict:
    """Pack expert weights with the deterministic reference codec.

    This is a format/lineage implementation, not the competitive MCG encoder.
    It makes allocation, byte accounting, reconstruction and scientific tests
    runnable before an accelerated codec adapter is selected.
    """
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except Exception as error:  # pragma: no cover - optional dependency
        raise _dependency_error("torch and safetensors", error)

    source = Path(model_path)
    destination = prepare_empty_destination(output_dir)
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    allocation = json.loads(Path(allocation_path).read_text())
    semantics = allocation.get("byte_semantics")
    if semantics not in (None, "codec-payload-including-codec-vectors-excluding-container"):
        raise ValueError(f"unsupported allocation byte semantics: {semantics!r}")
    choices = allocation.get("choices", allocation)
    units = load_inventory(config_path, family)
    by_layer: dict[int, list[QuantUnit]] = defaultdict(list)
    for unit in units:
        if unit.unit_id in choices:
            by_layer[unit.layer].append(unit)
    expected = set(choices)
    resolved = {unit.unit_id for rows in by_layer.values() for unit in rows}
    if missing := expected - resolved:
        raise ValueError(f"allocation contains unknown units: {sorted(missing)[:5]}")

    source_handles = {}
    manifest = {
        "schema": "quant-pipeline.reference-packed-checkpoint.v1",
        "family": family,
        "codec": {"name": "uniform-symmetric-reference", "group_size": group_size},
        "source": {
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
            "shards": {},
        },
        "allocation_sha256": sha256_file(allocation_path),
        "layers": {},
        "units": {},
    }
    try:
        for layer, layer_units in sorted(by_layer.items()):
            tensors: dict[str, torch.Tensor] = {}
            for unit in sorted(layer_units, key=lambda item: item.unit_id):
                shard_name = index[unit.tensor_name]
                if shard_name not in source_handles:
                    shard_path = source / shard_name
                    source_handles[shard_name] = safe_open(shard_path, framework="pt", device="cpu")
                    manifest["source"]["shards"][shard_name] = {
                        "bytes": shard_path.stat().st_size,
                        "sha256": sha256_file(shard_path),
                    }
                source_tensor = source_handles[shard_name].get_tensor(unit.tensor_name).float().numpy()
                if family == "gemma4":
                    if unit.expert is None:
                        raise ValueError(f"Gemma unit lacks expert index: {unit.unit_id}")
                    source_tensor = source_tensor[unit.expert]
                    if unit.projection in ("gate_slice", "up_slice"):
                        if source_tensor.shape[0] % 2:
                            raise ValueError(f"odd Gemma gate/up stack for {unit.unit_id}")
                        gate, up = np.split(source_tensor, 2, axis=0)
                        source_tensor = gate if unit.projection == "gate_slice" else up
                bits = int(choices[unit.unit_id]["bits"] if isinstance(choices[unit.unit_id], dict) else choices[unit.unit_id])
                encoded = quantize(source_tensor, bits, group_size)
                declared = choices[unit.unit_id].get("stored_bytes") if isinstance(choices[unit.unit_id], dict) else None
                if declared is not None and int(declared) != encoded.stored_bytes:
                    raise ValueError(
                        f"allocation byte claim differs from actual codec payload for {unit.unit_id}: "
                        f"declared={declared}, actual={encoded.stored_bytes}"
                    )
                tensors[_safe_key(unit.unit_id, "packed")] = torch.from_numpy(encoded.packed.copy())
                tensors[_safe_key(unit.unit_id, "scales")] = torch.from_numpy(encoded.scales.copy())
                manifest["units"][unit.unit_id] = {
                    "tensor_name": unit.tensor_name,
                    "projection": unit.projection,
                    "shape": list(encoded.shape),
                    "count": encoded.count,
                    "bits": bits,
                    "group_size": group_size,
                    "payload_bytes": encoded.stored_bytes,
                }
            layer_file = destination / f"experts-layer-{layer:03d}.safetensors"
            save_file(tensors, layer_file, metadata={"schema": manifest["schema"], "layer": str(layer)})
            manifest["layers"][str(layer)] = {
                "file": layer_file.name,
                "bytes": layer_file.stat().st_size,
                "sha256": sha256_file(layer_file),
            }
    finally:
        source_handles.clear()
    manifest["stored_bytes"] = sum(row["bytes"] for row in manifest["layers"].values())
    manifest["payload_bytes"] = sum(row["payload_bytes"] for row in manifest["units"].values())
    declared_total = allocation.get("stored_bytes")
    if declared_total is not None and int(declared_total) != manifest["payload_bytes"]:
        raise ValueError(
            f"allocation total byte claim differs from actual codec payload: "
            f"declared={declared_total}, actual={manifest['payload_bytes']}"
        )
    write_json(destination / "manifest.json", manifest)
    return manifest


def decode_unit(packed_dir: str | Path, unit_id: str) -> np.ndarray:
    try:
        from safetensors import safe_open
    except Exception as error:  # pragma: no cover
        raise _dependency_error("safetensors", error)
    root = Path(packed_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    unit = manifest["units"][unit_id]
    layer = int(re.search(r"L(\d+)", unit_id).group(1))
    path = root / manifest["layers"][str(layer)]["file"]
    with safe_open(path, framework="np") as handle:
        value = QuantizedArray(
            packed=handle.get_tensor(_safe_key(unit_id, "packed")),
            scales=handle.get_tensor(_safe_key(unit_id, "scales")),
            shape=tuple(unit["shape"]),
            bits=int(unit["bits"]),
            group_size=int(unit["group_size"]),
            count=int(unit["count"]),
        )
    return dequantize(value)


def audit_packed_checkpoint(packed_dir: str | Path) -> dict:
    try:
        from safetensors import safe_open
    except Exception as error:  # pragma: no cover
        raise _dependency_error("safetensors", error)
    root = Path(packed_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    failures: list[str] = []
    units_by_layer: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for unit_id, unit in manifest["units"].items():
        match = re.fullmatch(r"L(\d+)\..+", unit_id)
        if not match:
            failures.append(f"invalid-unit-id:{unit_id}")
            continue
        units_by_layer[str(int(match.group(1)))].append((unit_id, unit))
    actual_layer_files = {path.name for path in root.glob("experts-layer-*.safetensors")}
    declared_layer_files = {row["file"] for row in manifest["layers"].values()}
    if actual_layer_files != declared_layer_files:
        failures.append("layer-file-set-mismatch")
    actual_file_total = 0
    decoded_samples = 0
    for layer, row in manifest["layers"].items():
        path = root / row["file"]
        if not path.exists():
            failures.append(f"missing-layer:{layer}")
            continue
        actual_file_total += path.stat().st_size
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            failures.append(f"layer-identity:{layer}")
            continue
        expected_keys = {
            _safe_key(unit_id, field)
            for unit_id, _ in units_by_layer.get(layer, [])
            for field in ("packed", "scales")
        }
        sampled_bits: set[int] = set()
        with safe_open(path, framework="np") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("schema") != manifest["schema"] or metadata.get("layer") != layer:
                failures.append(f"layer-metadata:{layer}")
            if set(handle.keys()) != expected_keys:
                failures.append(f"layer-keys:{layer}")
                continue
            for unit_id, unit in units_by_layer.get(layer, []):
                packed = handle.get_tensor(_safe_key(unit_id, "packed"))
                scales = handle.get_tensor(_safe_key(unit_id, "scales"))
                count = int(unit["count"])
                bits = int(unit["bits"])
                group_size = int(unit["group_size"])
                expected_packed = (count * bits + 7) // 8
                expected_scales = (count + group_size - 1) // group_size
                actual_payload = packed.nbytes + scales.nbytes
                if packed.dtype != np.uint8 or scales.dtype != np.float32:
                    failures.append(f"unit-dtype:{unit_id}")
                if packed.size != expected_packed or scales.size != expected_scales:
                    failures.append(f"unit-size:{unit_id}")
                if actual_payload != int(unit["payload_bytes"]):
                    failures.append(f"unit-payload:{unit_id}")
                if not np.isfinite(scales).all() or np.any(scales <= 0):
                    failures.append(f"unit-scales:{unit_id}")
                if bits not in sampled_bits:
                    value = QuantizedArray(packed, scales, tuple(unit["shape"]), bits, group_size, count)
                    decoded = dequantize(value)
                    if decoded.shape != tuple(unit["shape"]) or not np.isfinite(decoded).all():
                        failures.append(f"unit-decode:{unit_id}")
                    sampled_bits.add(bits)
                    decoded_samples += 1
    payload_sum = sum(int(unit["payload_bytes"]) for unit in manifest["units"].values())
    if payload_sum != int(manifest.get("payload_bytes", -1)):
        failures.append("manifest-payload-total")
    if actual_file_total != int(manifest.get("stored_bytes", -1)):
        failures.append("manifest-file-total")
    return {
        "ok": not failures,
        "failures": failures,
        "file_bytes": actual_file_total,
        "payload_bytes": payload_sum,
        "container_overhead_bytes": actual_file_total - payload_sum,
        "unit_count": len(manifest["units"]),
        "decoded_samples": decoded_samples,
    }
