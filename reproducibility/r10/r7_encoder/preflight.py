"""Read-only model geometry and BF16-source header validation."""

from __future__ import annotations

import json
from pathlib import Path

from .constants import (
    FIRST_MOE_LAYER,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAST_MOE_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
    TOP_K,
    TensorId,
)
from .determinism import sha256_file
from .safetensors_io import SafeTensorReader

INDEX = "model.safetensors.index.json"


def _config_value(config: dict, key: str):
    if key in config:
        return config[key]
    text = config.get("text_config")
    if isinstance(text, dict) and key in text:
        return text[key]
    raise KeyError(key)


def _bf16_weight_name(tensor_id: TensorId) -> str:
    return f"{tensor_id.hf_prefix}.weight"


def preflight(carrier: Path, bf16_source: Path) -> dict[str, object]:
    carrier = carrier.resolve()
    source = bf16_source.resolve()
    if carrier == source:
        raise ValueError("carrier and BF16 source roles must be separate")
    carrier_config = json.loads((carrier / "config.json").read_text())
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "n_routed_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "first_k_dense_replace": FIRST_MOE_LAYER,
    }
    for key, value in expected.items():
        if int(_config_value(carrier_config, key)) != value:
            raise ValueError(f"carrier config {key} drift")

    source_index_path = source / INDEX
    source_index = json.loads(source_index_path.read_text())
    weight_map = source_index["weight_map"]
    readers: dict[str, SafeTensorReader] = {}
    checked = 0
    by_projection = {projection: 0 for projection in PROJECTIONS}
    for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1):
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                tensor_id = TensorId(layer, expert, projection)
                name = _bf16_weight_name(tensor_id)
                shard_name = weight_map.get(name)
                if shard_name is None:
                    raise ValueError(f"BF16 source missing {name}")
                reader = readers.setdefault(
                    shard_name, SafeTensorReader(source / shard_name)
                )
                info = reader.tensors.get(name)
                if info is None or info.dtype != "BF16":
                    raise ValueError(f"{name}: expected BF16 payload")
                expected_shape = (tensor_id.n, tensor_id.k)
                if info.shape != expected_shape:
                    raise ValueError(f"{name}: shape {info.shape} != {expected_shape}")
                checked += 1
                by_projection[projection] += 1
    return {
        "carrier_config_sha256": sha256_file(carrier / "config.json"),
        "carrier_index_sha256": sha256_file(carrier / INDEX),
        "bf16_source_index_sha256": sha256_file(source_index_path),
        "layers": [FIRST_MOE_LAYER, LAST_MOE_LAYER],
        "mtp_layer_78": "carried",
        "bf16_expert_tensors": checked,
        "by_projection": by_projection,
        "quant_of_quant_rejected": True,
        "passed": True,
    }
