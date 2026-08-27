"""Crash-safe GLM-5.3 selective EXL3/MCG checkpoint materialization.

Only routed expert ``.weight`` tensors are replaced.  Each replacement uses
the native ExLlamaV3 storage group ``.{trellis,suh,svh,mcg}``; every other
tensor is copied byte-for-byte in its source dtype.  This module seals a
storage checkpoint.  It deliberately does not claim that ExLlamaV3 has a
GLM-5.3 TP2/TP4 model implementation.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..campaign.glm53_direct_k4 import (
    MTP_LAYER,
    MAIN_ROUTED_LAYERS,
    NUM_EXPERTS,
    PROJECTIONS,
    inventory_tensor_map,
    projection_shape,
    materialization_plan_schema_for_bits,
    materialization_receipt_schema_for_bits,
)
from ..checkpoint.exact_payload import tensor_sha256
from ..core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json
from ..evaluation.glm53_packed_k4_reader import PackedK4Surface, load_complete_surface


SHARD_RECEIPT_SCHEMA = "quant-pipeline.glm53-k4-materialized-shard-receipt.v1"
STORAGE_ABI_SCHEMA = "quant-pipeline.glm53-exl3-mcg-storage-abi.v1"
EXLLAMAV3_VERSION = "0.0.43"
EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
MCG_MULTIPLIER_HEX = "0xCBAC1FED"
PACKED_SUFFIXES = ("trellis", "suh", "svh", "mcg")
ROUTED_SCOPES = {"routed_expert", "mtp_routed_expert"}
_HASH = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PRODUCER_REFERENCES = (
    "glm52_fresh_sqg",
    "score_sqg",
    "encode_uniform_sqg",
)

_INVENTORY_TO_TORCH = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "U16": "torch.uint16",
    "I16": "torch.int16",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "U32": "torch.uint32",
    "I32": "torch.int32",
    "F32": "torch.float32",
    "U64": "torch.uint64",
    "I64": "torch.int64",
    "F64": "torch.float64",
}


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_seal(value: Mapping[str, Any], schema: str, field: str) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    digest = value.get(field)
    if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise ValueError(f"{schema}.{field} is not SHA-256")
    body = copy.deepcopy(dict(value))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        raise ValueError(f"{schema} seal differs")
    return digest


def packed_tensor_name(source_weight_name: str, suffix: str) -> str:
    """Map an official HF linear weight to ExLlamaV3's loader key."""

    if not source_weight_name.endswith(".weight") or suffix not in PACKED_SUFFIXES:
        raise ValueError("packed tensor requires an official .weight and EXL3 suffix")
    return f"{source_weight_name[:-len('.weight')]}.{suffix}"


def verify_direct_mcg_producer(choice: Mapping[str, Any]) -> None:
    """Reject MCG-labelled payloads produced through an SQG source bridge.

    A marker and valid trellis are necessary but do not prove which numerical
    process produced them.  Production choices therefore have to disclose the
    exact imported source closure, with hashes, and positively attest that the
    reviewed entrypoint calls MCG directly.
    """

    provenance = choice.get("provenance")
    backend = provenance.get("backend") if isinstance(provenance, Mapping) else None
    closure = backend.get("producer_source_closure") if isinstance(backend, Mapping) else None
    structure = backend.get("shapleymcg_process_structure") if isinstance(backend, Mapping) else None
    if (
        not isinstance(closure, list)
        or not closure
        or backend.get("direct_mcg_entrypoint_reviewed") is not True
        or backend.get("source_closure_sqg_free") is not True
    ):
        raise ValueError("packed choice lacks a sealed direct-MCG producer source closure")
    for row in closure:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or _HASH.fullmatch(str(row.get("sha256", ""))) is None
        ):
            raise ValueError("direct-MCG producer source-closure row is malformed")
    disclosed = canonical_json(closure).decode("utf-8").lower()
    if any(reference in disclosed for reference in _FORBIDDEN_PRODUCER_REFERENCES):
        raise ValueError("packed choice producer source closure imports an SQG implementation")
    expected_structure = {
        "driver": "scripts/run_qwen_fast_encode.py",
        "normalization": "src/quant_pipeline/normalization/streaming_v31.py",
        "codec_adapter": "src/quant_pipeline/codecs/exl3_mcg.py",
        "operation_order": "reproducibility/local-corrected-v1",
        "numeric_closure": "r7_encoder",
    }
    if structure != expected_structure:
        raise ValueError("packed choice does not bind the public ShapleyMCG process structure")
    paths = {str(row["path"]).lower() for row in closure}
    required_suffixes = {
        expected_structure["driver"],
        expected_structure["normalization"],
        expected_structure["codec_adapter"],
    }
    if not all(any(path.endswith(required) for path in paths) for required in required_suffixes):
        raise ValueError("direct-MCG producer closure omits the public ShapleyMCG structure")
    if not any("r7_encoder/" in path or "bmmlaw_r7_encoder/" in path for path in paths):
        raise ValueError("direct-MCG producer closure omits the pure r7_encoder numeric core")


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("example_only") is True:
        raise ValueError(f"not an executable JSON receipt: {path}")
    return value


def _source_index(source_root: Path, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    index = _json(source_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError("official source index has no valid weight_map")
    if set(weight_map) != set(rows):
        raise ValueError("official source index and sealed inventory tensor census differ")
    for name, shard in weight_map.items():
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"official source index shard path is unsafe: {shard}")
        if rows[name].get("shard") != shard:
            raise ValueError(f"official source index/inventory shard differs: {name}")
    return dict(weight_map)


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    surface: PackedK4Surface,
    mtp_adapter_receipt: Mapping[str, Any],
) -> str:
    plan_sha = _verify_seal(
        plan, materialization_plan_schema_for_bits(surface.bits), "plan_sha256"
    )
    expected_choices = (len(MAIN_ROUTED_LAYERS) + 1) * NUM_EXPERTS * len(PROJECTIONS)
    if (
        plan.get("contract_sha256") != surface.contract_sha256
        or plan.get("inventory_sha256") != inventory.get("inventory_sha256")
        or plan.get("main_layer_receipt_sha256") != list(surface.main_layer_receipt_sha256)
        or plan.get("mtp_adapter_receipt_sha256") != mtp_adapter_receipt.get("receipt_sha256")
        or plan.get("main_choice_count") != len(MAIN_ROUTED_LAYERS) * NUM_EXPERTS * 3
        or plan.get("mtp_choice_count") != NUM_EXPERTS * 3
        or plan.get("total_choice_count") != expected_choices
        or len(surface.choices) != expected_choices
    ):
        raise ValueError("materialization plan differs from complete main plus MTP packed surface")
    return plan_sha


def storage_abi_receipt(
    *, plan_sha256: str, surface: PackedK4Surface, output_tensor_names: Sequence[str]
) -> dict[str, Any]:
    """Record the real storage ABI without manufacturing serving evidence."""

    names = sorted(output_tensor_names)
    bits = int(getattr(surface, "bits", 4))
    return _seal(
        {
            "schema": STORAGE_ABI_SCHEMA,
            "plan_sha256": plan_sha256,
            "codec_family": "exl3-mcg",
            "mcg_multiplier_hex": MCG_MULTIPLIER_HEX,
            "bits": bits,
            "exllamav3": {
                "version": EXLLAMAV3_VERSION,
                "git_commit": EXLLAMAV3_COMMIT,
                "linear_storage_group": [
                    ["su", "suh"],
                    ["sv", "svh"],
                    "trellis",
                ],
                "written_suffixes": list(PACKED_SUFFIXES),
                "module_key_rule": "official_weight_name_without_.weight",
            },
            "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
            "output_tensor_names_sha256": sha256_bytes(canonical_json(names)),
            "output_tensor_count": len(names),
            "storage_checkpoint_verified": True,
            "serving_reader_qualified": False,
            "qualified_tp_sizes": [],
            "reason": "ExLlamaV3 v0.0.43 has no audited GLM-5.3 TP model load/inference receipt",
        },
        "receipt_sha256",
    )


def _tensor_record(
    *,
    name: str,
    value: Any,
    origin: str,
    source_tensor_name: str,
    choice_sha256: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "bytes": int(value.numel() * value.element_size()),
        "payload_sha256": tensor_sha256(value),
        "origin": origin,
        "source_tensor_name": source_tensor_name,
        "choice_sha256": choice_sha256,
    }


def _load_shard_tensors(
    *,
    source_root: Path,
    source_shard: str,
    shard_rows: Sequence[Mapping[str, Any]],
    surface: PackedK4Surface,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    from safetensors import safe_open

    source_path = source_root / source_shard
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError(f"official source shard is absent or symlinked: {source_shard}")
    tensors: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    native_count = 0
    choice_count = 0
    store = None
    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        expected_source_keys = {str(row["tensor_name"]) for row in shard_rows}
        if source_keys != expected_source_keys:
            raise ValueError(f"source shard header/inventory census differs: {source_shard}")
        for row in sorted(shard_rows, key=lambda item: str(item["tensor_name"])):
            source_name = str(row["tensor_name"])
            if row.get("scope") not in ROUTED_SCOPES:
                value = source.get_tensor(source_name).contiguous()
                if (
                    str(value.dtype) != _INVENTORY_TO_TORCH.get(str(row.get("dtype")))
                    or list(value.shape) != row.get("shape")
                    or value.numel() * value.element_size() != row.get("source_bytes")
                    or tensor_sha256(value) != row.get("source_payload_sha256")
                ):
                    raise ValueError(f"official native tensor payload differs: {source_name}")
                tensors[source_name] = value
                records.append(
                    _tensor_record(
                        name=source_name,
                        value=value,
                        origin="official_bf16_native_copy",
                        source_tensor_name=source_name,
                        choice_sha256=None,
                    )
                )
                native_count += 1
                continue

            layer = int(row["layer"])
            expert = int(row["expert"])
            projection = str(row["projection"])
            if (
                layer not in (*MAIN_ROUTED_LAYERS, MTP_LAYER)
                or expert not in range(NUM_EXPERTS)
                or projection not in PROJECTIONS
                or row.get("shape") != list(projection_shape(projection))
            ):
                raise ValueError(f"routed inventory geometry differs: {source_name}")
            choice = surface.choice(layer, expert, projection)
            verify_direct_mcg_producer(choice)
            if store is None:
                store = surface.store
            store.verify_choice(choice)
            for suffix in PACKED_SUFFIXES:
                value = store.objects.load_tensor(choice["objects"][suffix]).contiguous()
                output_name = packed_tensor_name(source_name, suffix)
                if output_name in tensors:
                    raise ValueError(f"output tensor name collision: {output_name}")
                tensors[output_name] = value
                records.append(
                    _tensor_record(
                        name=output_name,
                        value=value,
                        origin="sealed_exl3_mcg_packed_choice",
                        source_tensor_name=source_name,
                        choice_sha256=choice["choice_sha256"],
                    )
                )
            choice_count += 1
    if set(tensors) != {row["name"] for row in records}:
        raise ValueError("materialized shard tensor record census differs")
    return tensors, records, native_count, choice_count


def _verify_output_shard(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    from safetensors import safe_open

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"materialized shard is absent or symlinked: {path}")
    expected = {str(row["name"]): row for row in records}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError(f"materialized shard tensor census differs: {path.name}")
        for name, row in expected.items():
            value = handle.get_tensor(name)
            if (
                str(value.dtype) != row["dtype"]
                or list(value.shape) != row["shape"]
                or value.numel() * value.element_size() != row["bytes"]
                or tensor_sha256(value) != row["payload_sha256"]
            ):
                raise ValueError(f"materialized tensor payload differs: {name}")


def _write_safetensors_atomic(path: Path, tensors: Mapping[str, Any], plan_sha256: str) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        save_file(
            dict(tensors),
            temporary,
            metadata={
                "format": "pt",
                "codec": "exl3-mcg",
                "materialization_plan_sha256": plan_sha256,
            },
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _shard_receipt_path(output_root: Path, source_shard: str) -> Path:
    return output_root / ".materialization" / "shards" / f"{source_shard}.json"


def materialize_shard(
    *,
    plan_sha256: str,
    source_root: Path,
    output_root: Path,
    source_shard: str,
    shard_rows: Sequence[Mapping[str, Any]],
    surface: PackedK4Surface,
) -> dict[str, Any]:
    """Write or resume one source-aligned output shard."""

    bits = int(getattr(surface, "bits", 4))
    output_path = output_root / source_shard
    receipt_path = _shard_receipt_path(output_root, source_shard)
    if receipt_path.exists():
        receipt = _json(receipt_path)
        _verify_seal(
            receipt,
            f"quant-pipeline.glm53-k{bits}-materialized-shard-receipt.v1",
            "receipt_sha256",
        )
        if (
            receipt.get("plan_sha256") != plan_sha256
            or receipt.get("source_shard") != source_shard
            or receipt.get("shard") != source_shard
            or receipt.get("complete") is not True
            or sha256_file(output_path) != receipt.get("shard_sha256")
        ):
            raise ValueError(f"resumable shard receipt differs: {source_shard}")
        return receipt

    tensors, records, native_count, choice_count = _load_shard_tensors(
        source_root=source_root,
        source_shard=source_shard,
        shard_rows=shard_rows,
        surface=surface,
    )
    if output_path.exists():
        _verify_output_shard(output_path, records)
    else:
        _write_safetensors_atomic(output_path, tensors, plan_sha256)
        _verify_output_shard(output_path, records)
    body = {
        "schema": f"quant-pipeline.glm53-k{bits}-materialized-shard-receipt.v1",
        "plan_sha256": plan_sha256,
        "source_shard": source_shard,
        "shard": source_shard,
        "shard_bytes": output_path.stat().st_size,
        "shard_sha256": sha256_file(output_path),
        "native_tensor_count": native_count,
        "routed_choice_count": choice_count,
        "output_tensor_count": len(records),
        "output_logical_bytes": sum(int(row["bytes"]) for row in records),
        "tensors": records,
        "complete": True,
    }
    receipt = _seal(body, "receipt_sha256")
    write_json(receipt_path, receipt)
    return receipt


def _quantization_config(
    source_config: Mapping[str, Any],
    tensor_rows: Sequence[Mapping[str, Any]],
    *,
    bits: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(dict(source_config))
    prior = config.get("quantization_config")
    qcfg: dict[str, Any] = {
        "quant_method": "exl3",
        "version": EXLLAMAV3_VERSION,
        "bits": bits,
        "head_bits": 16,
        "codebook": "mcg",
        "scope": "glm53_routed_experts_only",
        "non_routed_dtype_policy": "official_source_native",
        "serving_reader_qualified": False,
    }
    if prior is not None:
        qcfg["original_quantization_config"] = copy.deepcopy(prior)
    config["quantization_config"] = qcfg

    by_module: dict[str, dict[str, Any]] = {}
    for row in tensor_rows:
        if row["origin"] != "sealed_exl3_mcg_packed_choice":
            continue
        name = str(row["name"])
        module = name.rsplit(".", 1)[0]
        item = by_module.setdefault(module, {"stored_tensors": {}})
        item["stored_tensors"][name] = {
            "shape": row["shape"],
            "n_bytes": row["bytes"],
            "dtype": row["dtype"],
        }
    for module, item in by_module.items():
        if set(item["stored_tensors"]) != {
            f"{module}.{suffix}" for suffix in PACKED_SUFFIXES
        }:
            raise ValueError(f"EXL3 stored tensor group is incomplete: {module}")
        item.update(
            {
                "quant_format": "exl3",
                "bits_per_weight": bits,
                "mcg_multiplier": int(MCG_MULTIPLIER_HEX, 16),
            }
        )
    quantization = copy.deepcopy(qcfg)
    quantization["tensor_storage"] = by_module
    return config, quantization


def _copy_auxiliary_files(source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    excluded = {"config.json", "model.safetensors.index.json", "quantization_config.json"}
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if (
            not source.is_file()
            or source.is_symlink()
            or source.name in excluded
            or source.name.endswith(".safetensors")
            or source.suffix in {".bin", ".ckpt", ".pt", ".pth"}
        ):
            continue
        destination = output_root / source.name
        expected = sha256_file(source)
        if destination.exists():
            if not destination.is_file() or destination.is_symlink() or sha256_file(destination) != expected:
                raise ValueError(f"existing auxiliary file differs: {destination}")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=output_root)
            os.close(descriptor)
            try:
                shutil.copyfile(source, temporary)
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        copied.append({"path": source.name, "bytes": destination.stat().st_size, "sha256": expected})
    return copied


def _auxiliary_source_names(source_root: Path) -> set[str]:
    excluded = {"config.json", "model.safetensors.index.json", "quantization_config.json"}
    return {
        path.name
        for path in source_root.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and path.name not in excluded
        and not path.name.endswith(".safetensors")
        and path.suffix not in {".bin", ".ckpt", ".pt", ".pth"}
    }


def verify_materialized_checkpoint(
    *,
    output_root: str | Path,
    inventory: Mapping[str, Any],
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Replay the receipt/index/config census; optionally stream every shard."""

    output_root = Path(output_root).resolve()
    receipt = _json(output_root / "materialization-receipt.json")
    bits = int(receipt.get("bits", -1))
    _verify_seal(
        receipt, materialization_receipt_schema_for_bits(bits), "receipt_sha256"
    )
    rows = inventory_tensor_map(inventory)
    if (
        receipt.get("source_inventory_sha256") != inventory.get("inventory_sha256")
        or receipt.get("source_model_revision") != inventory.get("model_revision")
        or receipt.get("source_tensor_count") != len(rows)
        or receipt.get("complete") is not True
        or receipt.get("codec_family") != "exl3-mcg"
        or receipt.get("mcg_multiplier_hex") != MCG_MULTIPLIER_HEX
        or receipt.get("bits") != bits
        or receipt.get("nonrouted_native_exact") is not True
        or receipt.get("main_and_mtp_complete") is not True
        or receipt.get("serving_reader_qualified") is not False
        or receipt.get("qualified_tp_sizes") != []
    ):
        raise ValueError("materialized checkpoint receipt semantics differ")
    shards = receipt.get("shards")
    shard_hashes = receipt.get("shard_sha256")
    shard_receipt_hashes = receipt.get("shard_receipt_sha256")
    if (
        not isinstance(shards, list)
        or not shards
        or len(shards) != len(set(shards))
        or not isinstance(shard_hashes, Mapping)
        or set(shard_hashes) != set(shards)
        or not isinstance(shard_receipt_hashes, list)
        or len(shard_receipt_hashes) != len(shards)
    ):
        raise ValueError("materialized shard receipt census differs")
    tensor_rows: list[dict[str, Any]] = []
    observed_receipt_hashes: list[str] = []
    for shard in shards:
        shard_receipt = _json(_shard_receipt_path(output_root, shard))
        shard_receipt_sha = _verify_seal(
            shard_receipt,
            f"quant-pipeline.glm53-k{bits}-materialized-shard-receipt.v1",
            "receipt_sha256",
        )
        if (
            shard_receipt.get("plan_sha256") != receipt.get("plan_sha256")
            or shard_receipt.get("shard") != shard
            or shard_receipt.get("complete") is not True
            or shard_receipt.get("shard_sha256") != shard_hashes[shard]
            or not isinstance(shard_receipt.get("tensors"), list)
            or shard_receipt.get("output_tensor_count") != len(shard_receipt["tensors"])
            or shard_receipt.get("output_logical_bytes")
            != sum(int(row["bytes"]) for row in shard_receipt["tensors"])
        ):
            raise ValueError(f"materialized shard closure differs: {shard}")
        shard_path = output_root / shard
        if (
            not shard_path.is_file()
            or shard_path.is_symlink()
            or shard_path.stat().st_size != shard_receipt.get("shard_bytes")
            or (verify_shard_hashes and sha256_file(shard_path) != shard_hashes[shard])
        ):
            raise ValueError(f"materialized shard file differs: {shard}")
        tensor_rows.extend(copy.deepcopy(shard_receipt["tensors"]))
        observed_receipt_hashes.append(shard_receipt_sha)
    if observed_receipt_hashes != shard_receipt_hashes:
        raise ValueError("materialized shard receipt ordering/hash census differs")

    names = [str(row.get("name")) for row in tensor_rows]
    if (
        len(names) != len(set(names))
        or len(names) != receipt.get("output_tensor_count")
        or sha256_bytes(canonical_json(sorted(names)))
        != receipt.get("output_tensor_names_sha256")
        or sum(int(row["bytes"]) for row in tensor_rows)
        != receipt.get("output_logical_bytes")
    ):
        raise ValueError("materialized output tensor census differs")
    by_name = {str(row["name"]): row for row in tensor_rows}
    native_names = {
        name for name, row in rows.items() if row.get("scope") not in ROUTED_SCOPES
    }
    routed_names = {name for name, row in rows.items() if row.get("scope") in ROUTED_SCOPES}
    packed_names = {
        packed_tensor_name(name, suffix)
        for name in routed_names
        for suffix in PACKED_SUFFIXES
    }
    if set(by_name) != native_names | packed_names:
        raise ValueError("materialized tensor names do not exactly replace routed weights")
    for name in native_names:
        output = by_name[name]
        source = rows[name]
        if (
            output.get("origin") != "official_bf16_native_copy"
            or output.get("source_tensor_name") != name
            or output.get("choice_sha256") is not None
            or output.get("payload_sha256") != source.get("source_payload_sha256")
            or output.get("dtype") != _INVENTORY_TO_TORCH.get(str(source.get("dtype")))
            or output.get("shape") != source.get("shape")
            or output.get("bytes") != source.get("source_bytes")
        ):
            raise ValueError(f"materialized native tensor closure differs: {name}")
    for name in packed_names:
        output = by_name[name]
        if (
            output.get("origin") != "sealed_exl3_mcg_packed_choice"
            or _HASH.fullmatch(str(output.get("choice_sha256", ""))) is None
            or _HASH.fullmatch(str(output.get("payload_sha256", ""))) is None
        ):
            raise ValueError(f"materialized packed tensor closure differs: {name}")
    if (
        receipt.get("native_tensor_count") != len(native_names)
        or receipt.get("routed_choice_count") != len(routed_names)
        or receipt.get("packed_tensor_count") != len(packed_names)
    ):
        raise ValueError("materialized native/routed accounting differs")

    index_path = output_root / "model.safetensors.index.json"
    config_path = output_root / "config.json"
    quantization_path = output_root / "quantization_config.json"
    if (
        sha256_file(index_path) != receipt.get("index_sha256")
        or sha256_file(config_path) != receipt.get("config_sha256")
        or sha256_file(quantization_path) != receipt.get("quantization_config_sha256")
    ):
        raise ValueError("materialized config/index hash differs")
    index = _json(index_path)
    expected_map = {
        str(row["name"]): shard
        for shard in shards
        for row in _json(_shard_receipt_path(output_root, shard))["tensors"]
    }
    if (
        index.get("weight_map") != expected_map
        or index.get("metadata", {}).get("total_size") != receipt.get("output_logical_bytes")
    ):
        raise ValueError("materialized safetensors index differs")
    config = _json(config_path)
    quantization = _json(quantization_path)
    if (
        config.get("quantization_config", {}).get("quant_method") != "exl3"
        or config.get("quantization_config", {}).get("codebook") != "mcg"
        or config.get("quantization_config", {}).get("serving_reader_qualified") is not False
        or quantization.get("quant_method") != "exl3"
        or quantization.get("codebook") != "mcg"
        or config.get("quantization_config", {}).get("bits") != bits
        or quantization.get("bits") != bits
        or len(quantization.get("tensor_storage", {})) != len(routed_names)
    ):
        raise ValueError("materialized EXL3/MCG config semantics differ")
    storage = _json(output_root / "exl3-mcg-storage-abi.json")
    storage_sha = _verify_seal(storage, STORAGE_ABI_SCHEMA, "receipt_sha256")
    if (
        storage_sha != receipt.get("storage_abi_receipt_sha256")
        or storage.get("serving_reader_qualified") is not False
        or storage.get("qualified_tp_sizes") != []
    ):
        raise ValueError("materialized storage ABI evidence differs")
    for auxiliary in receipt.get("auxiliary_files", []):
        path = output_root / str(auxiliary["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != auxiliary.get("bytes")
            or sha256_file(path) != auxiliary.get("sha256")
        ):
            raise ValueError(f"materialized auxiliary file differs: {path.name}")
    expected_files = {
        *shards,
        "config.json",
        "model.safetensors.index.json",
        "quantization_config.json",
        "exl3-mcg-storage-abi.json",
        "materialization-receipt.json",
        *(str(row["path"]) for row in receipt.get("auxiliary_files", [])),
        *(
            str(_shard_receipt_path(output_root, shard).relative_to(output_root))
            for shard in shards
        ),
    }
    observed_files = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError("materialized checkpoint contains undeclared or missing files")
    return receipt


def materialize_checkpoint(
    *,
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    mtp_adapter_receipt: Mapping[str, Any],
    packed_root: str | Path,
    source_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize all shards and seal the exact tensor/index/config census."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    packed_root = Path(packed_root).resolve()
    if output_root in {source_root, packed_root}:
        raise ValueError("materialized checkpoint must use a distinct output root")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = inventory_tensor_map(inventory)
    surface = load_complete_surface(
        root=packed_root,
        contract=contract,
        mtp_adapter_receipt=mtp_adapter_receipt,
    )
    plan_sha = _validate_plan(
        plan,
        inventory=inventory,
        surface=surface,
        mtp_adapter_receipt=mtp_adapter_receipt,
    )
    weight_map = _source_index(source_root, rows)
    allowed_top_level = {
        *set(weight_map.values()),
        *_auxiliary_source_names(source_root),
        "config.json",
        "model.safetensors.index.json",
        "quantization_config.json",
        "exl3-mcg-storage-abi.json",
        "materialization-receipt.json",
        ".materialization",
    }
    unexpected = {path.name for path in output_root.iterdir()} - allowed_top_level
    if unexpected:
        raise ValueError(f"materialization output contains undeclared paths: {sorted(unexpected)}")
    by_shard: dict[str, list[Mapping[str, Any]]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(rows[name])

    receipts = [
        materialize_shard(
            plan_sha256=plan_sha,
            source_root=source_root,
            output_root=output_root,
            source_shard=shard,
            shard_rows=by_shard[shard],
            surface=surface,
        )
        for shard in sorted(by_shard)
    ]
    tensor_rows = [row for receipt in receipts for row in receipt["tensors"]]
    output_names = [str(row["name"]) for row in tensor_rows]
    if len(output_names) != len(set(output_names)):
        raise ValueError("materialized checkpoint has duplicate tensor names")
    native_rows = [row for row in tensor_rows if row["origin"] == "official_bf16_native_copy"]
    packed_rows = [row for row in tensor_rows if row["origin"] == "sealed_exl3_mcg_packed_choice"]
    expected_choices = int(plan["total_choice_count"])
    if (
        len(native_rows) != plan["native_tensor_count"]
        or len(packed_rows) != expected_choices * len(PACKED_SUFFIXES)
        or {row["name"] for row in native_rows}
        != {name for name, row in rows.items() if row.get("scope") not in ROUTED_SCOPES}
    ):
        raise ValueError("final native/packed tensor census differs")
    for row in native_rows:
        source = rows[str(row["source_tensor_name"])]
        if row["payload_sha256"] != source["source_payload_sha256"]:
            raise ValueError(f"native output is not an exact source copy: {row['name']}")

    output_weight_map = {
        str(row["name"]): str(receipt["shard"])
        for receipt in receipts
        for row in receipt["tensors"]
    }
    total_size = sum(int(row["bytes"]) for row in tensor_rows)
    index = {"metadata": {"total_size": total_size}, "weight_map": output_weight_map}
    source_config = _json(source_root / "config.json")
    bits = int(getattr(surface, "bits", 4))
    config, quantization = _quantization_config(
        source_config, tensor_rows, bits=bits
    )
    atomic_write(output_root / "model.safetensors.index.json", json.dumps(index, indent=2, sort_keys=True).encode() + b"\n")
    atomic_write(output_root / "config.json", json.dumps(config, indent=2, sort_keys=True).encode() + b"\n")
    atomic_write(output_root / "quantization_config.json", json.dumps(quantization, indent=2, sort_keys=True).encode() + b"\n")
    auxiliaries = _copy_auxiliary_files(source_root, output_root)
    storage = storage_abi_receipt(
        plan_sha256=plan_sha,
        surface=surface,
        output_tensor_names=output_names,
    )
    write_json(output_root / "exl3-mcg-storage-abi.json", storage)
    final = _seal(
        {
            "schema": materialization_receipt_schema_for_bits(bits),
            "plan_sha256": plan_sha,
            "source_inventory_sha256": inventory["inventory_sha256"],
            "source_model_revision": inventory["model_revision"],
            "packed_root": str(packed_root),
            "output_root": str(output_root),
            "shards": [receipt["shard"] for receipt in receipts],
            "shard_receipt_sha256": [receipt["receipt_sha256"] for receipt in receipts],
            "shard_sha256": {receipt["shard"]: receipt["shard_sha256"] for receipt in receipts},
            "source_tensor_count": len(rows),
            "native_tensor_count": len(native_rows),
            "routed_choice_count": expected_choices,
            "packed_tensor_count": len(packed_rows),
            "output_tensor_count": len(output_names),
            "output_tensor_names_sha256": sha256_bytes(canonical_json(sorted(output_names))),
            "output_logical_bytes": total_size,
            "index_sha256": sha256_file(output_root / "model.safetensors.index.json"),
            "config_sha256": sha256_file(output_root / "config.json"),
            "quantization_config_sha256": sha256_file(output_root / "quantization_config.json"),
            "storage_abi_receipt_sha256": storage["receipt_sha256"],
            "auxiliary_files": auxiliaries,
            "codec_family": "exl3-mcg",
            "mcg_multiplier_hex": MCG_MULTIPLIER_HEX,
            "bits": bits,
            "nonrouted_native_exact": True,
            "main_and_mtp_complete": True,
            "serving_reader_qualified": False,
            "qualified_tp_sizes": [],
            "reader_audit_required_before_publication_as_serving_ready": True,
            "complete": True,
        },
        "receipt_sha256",
    )
    write_json(output_root / "materialization-receipt.json", final)
    return verify_materialized_checkpoint(
        output_root=output_root,
        inventory=inventory,
        verify_shard_hashes=False,
    )
