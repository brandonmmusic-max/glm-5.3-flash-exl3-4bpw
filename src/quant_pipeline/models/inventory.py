from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantUnit:
    unit_id: str
    tensor_name: str
    layer: int
    expert: int | None
    projection: str
    scope: str = "routed_expert"
    coupling_group: str = ""
    allowed_bits: tuple[int, ...] = (3, 4)


def qwen3_moe_units(config: dict) -> list[QuantUnit]:
    layers = int(config["num_hidden_layers"])
    experts = int(config["num_experts"])
    units: list[QuantUnit] = []
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.experts"
        for expert in range(experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensor = f"{prefix}.{expert}.{projection}.weight"
                units.append(QuantUnit(f"L{layer:03d}.E{expert:03d}.{projection}", tensor, layer, expert, projection))
    return units


def gemma4_moe_units(config: dict) -> list[QuantUnit]:
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    units: list[QuantUnit] = []
    # Gemma4 stores experts stacked; slicing is recorded in unit_id rather than tensor name.
    for layer in range(layers):
        for expert in range(int(text["num_experts"])):
            base = f"model.language_model.layers.{layer}.experts"
            units.extend(
                [
                    QuantUnit(f"L{layer:03d}.E{expert:03d}.gate", f"{base}.gate_up_proj", layer, expert, "gate_slice"),
                    QuantUnit(f"L{layer:03d}.E{expert:03d}.up", f"{base}.gate_up_proj", layer, expert, "up_slice"),
                    QuantUnit(f"L{layer:03d}.E{expert:03d}.down", f"{base}.down_proj", layer, expert, "down_slice"),
                ]
            )
    return units


def glm_moe_dsa_units(config: dict) -> list[QuantUnit]:
    """Enumerate the expected GLM MoE/DSA quantization surface.

    This is a configuration-derived expectation, not a checkpoint attestation.
    A production campaign must compare it with the safetensors inventory before
    any calibration or encoding work begins.  The optional next-token-prediction
    layers are included because GLM-5.2 stores its MTP layer after the main
    transformer stack.
    """

    layers = int(config["num_hidden_layers"])
    mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    total_layers = layers + mtp_layers
    experts = int(config.get("n_routed_experts", config.get("num_experts", 0)))
    first_moe = int(config.get("first_k_dense_replace", 0))
    if layers < 1 or experts < 1 or not 0 <= first_moe <= layers:
        raise ValueError("invalid GLM MoE geometry")

    units: list[QuantUnit] = []
    for layer in range(total_layers):
        prefix = f"model.layers.{layer}"
        is_mtp = layer >= layers
        is_moe = layer >= first_moe
        if is_moe:
            for expert in range(experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    tensor = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
                    units.append(
                        QuantUnit(
                            f"L{layer:03d}.E{expert:03d}.{projection}",
                            tensor,
                            layer,
                            expert,
                            projection,
                            scope="mtp_routed_expert" if is_mtp else "routed_expert",
                            coupling_group=f"L{layer:03d}.E{expert:03d}",
                            allowed_bits=(3, 4, 5),
                        )
                    )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                units.append(
                    QuantUnit(
                        f"L{layer:03d}.shared.{projection}",
                        f"{prefix}.mlp.shared_experts.{projection}.weight",
                        layer,
                        None,
                        projection,
                        scope="mtp_shared_expert" if is_mtp else "shared_expert",
                        coupling_group=f"L{layer:03d}.shared",
                        allowed_bits=(6, 7, 8, 16),
                    )
                )
        else:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                units.append(
                    QuantUnit(
                        f"L{layer:03d}.dense_mlp.{projection}",
                        f"{prefix}.mlp.{projection}.weight",
                        layer,
                        None,
                        projection,
                        scope="dense_mlp",
                        coupling_group=f"L{layer:03d}.dense_mlp",
                        allowed_bits=(6, 7, 8, 16),
                    )
                )

        for projection in ("q_b_proj", "o_proj"):
            units.append(
                QuantUnit(
                    f"L{layer:03d}.attention.{projection}",
                    f"{prefix}.self_attn.{projection}.weight",
                    layer,
                    None,
                    projection,
                    scope="mtp_attention_output" if is_mtp else "attention_output",
                    coupling_group=f"L{layer:03d}.attention_dense6",
                    allowed_bits=(6, 7, 8, 16),
                )
            )
        for projection in ("q_a_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
            units.append(
                QuantUnit(
                    f"L{layer:03d}.attention.{projection}",
                    f"{prefix}.self_attn.{projection}.weight",
                    layer,
                    None,
                    projection,
                    scope="mtp_attention_latent" if is_mtp else "attention_latent",
                    coupling_group=f"L{layer:03d}.attention_latent",
                    allowed_bits=(6, 7, 8, 16),
                )
            )
        for projection in ("wq_b", "wk", "weights_proj"):
            units.append(
                QuantUnit(
                    f"L{layer:03d}.indexer.{projection}",
                    f"{prefix}.self_attn.indexer.{projection}.weight",
                    layer,
                    None,
                    projection,
                    scope="mtp_dsa_indexer" if is_mtp else "dsa_indexer",
                    coupling_group=f"L{layer:03d}.indexer",
                    allowed_bits=(8, 16),
                )
            )
    return units


def glm5_next_units(config: dict) -> list[QuantUnit]:
    """Enumerate GLM-5.3 ``glm5_next`` MLP quantization units.

    GLM-5.3 nests its language-model geometry under ``text_config`` and its
    tensors under ``model.language_model.layers``.  Main-layer sparsity is
    declared explicitly by ``mlp_layer_types``; the configured NextN layer is
    stored immediately after the main stack and is sparse as well.

    This inventory deliberately describes the matrices whose expert-function
    calibration is model-aware.  The released-checkpoint inspector separately
    inventories attention, vision, embeddings, norms, biases, and mHC tensors
    so each uniform K4/K6 routed-expert profile can give every physical tensor
    an explicit quantized-or-native disposition.
    """

    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "glm5_next_text":
        raise ValueError("GLM-5.3 inventory requires text_config.model_type='glm5_next_text'")
    layers = int(text["num_hidden_layers"])
    mtp_layers = int(text.get("num_nextn_predict_layers", 0))
    experts = int(text.get("n_routed_experts", 0))
    mlp_types = text.get("mlp_layer_types")
    if (
        layers < 1
        or mtp_layers < 0
        or experts < 1
        or not isinstance(mlp_types, list)
        or len(mlp_types) != layers
        or any(kind not in {"dense", "sparse"} for kind in mlp_types)
    ):
        raise ValueError("invalid GLM-5.3 text geometry")

    units: list[QuantUnit] = []
    for layer in range(layers + mtp_layers):
        prefix = f"model.language_model.layers.{layer}"
        is_mtp = layer >= layers
        is_sparse = is_mtp or mlp_types[layer] == "sparse"
        if is_sparse:
            for expert in range(experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    units.append(
                        QuantUnit(
                            f"L{layer:03d}.E{expert:03d}.{projection}",
                            f"{prefix}.mlp.experts.{expert}.{projection}.weight",
                            layer,
                            expert,
                            projection,
                            scope="mtp_routed_expert" if is_mtp else "routed_expert",
                            coupling_group=f"L{layer:03d}.E{expert:03d}",
                            allowed_bits=(4, 6),
                        )
                    )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                units.append(
                    QuantUnit(
                        f"L{layer:03d}.shared.{projection}",
                        f"{prefix}.mlp.shared_experts.{projection}.weight",
                        layer,
                        None,
                        projection,
                        scope="mtp_shared_expert" if is_mtp else "shared_expert",
                        coupling_group=f"L{layer:03d}.shared",
                        allowed_bits=(16,),
                    )
                )
        else:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                units.append(
                    QuantUnit(
                        f"L{layer:03d}.dense_mlp.{projection}",
                        f"{prefix}.mlp.{projection}.weight",
                        layer,
                        None,
                        projection,
                        scope="dense_mlp",
                        coupling_group=f"L{layer:03d}.dense_mlp",
                        allowed_bits=(16,),
                    )
                )
    return units


def load_inventory(config_path: str | Path, family: str) -> list[QuantUnit]:
    config = json.loads(Path(config_path).read_text())
    if family == "qwen3_moe":
        return qwen3_moe_units(config)
    if family == "gemma4":
        return gemma4_moe_units(config)
    if family == "glm_moe_dsa":
        return glm_moe_dsa_units(config)
    if family == "glm5_next":
        return glm5_next_units(config)
    raise ValueError(f"unsupported model family {family!r}")
