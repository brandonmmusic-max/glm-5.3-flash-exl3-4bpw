"""Exact-architecture Transformers TP2 adapter for selective GLM-5.3 EXL3/MCG.

This is deliberately a custom Transformers runtime, not a stock-vLLM or stock-
ExLlamaV3 compatibility claim.  Transformers owns the exact GLM-5.3 model and
its native BF16 body.  Sparse expert containers in main layers 3..44 are
replaced *before checkpoint loading* by packed EXL3 projections.  The adapter
uses DTensor only to stream the correct TP shard from safetensors:

* gate/up: trellis and ``svh`` are column-sharded;
* down: trellis and ``suh`` are row-sharded;
* the down result is all-reduced.

No routed ``.weight`` parameter exists after replacement.  MTP layer 45 is
stored by the checkpoint but intentionally not executed by this text runtime.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import Any, Mapping, Sequence

from ..campaign.glm53_direct_k4 import (
    MAIN_ROUTED_LAYERS,
    MTP_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
    materialization_receipt_schema_for_bits,
)
from ..checkpoint.packed_payload import MCG_MARKER_SIGNED_INT32
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file


TRANSFORMERS_VERSION = "5.16.1"
EXLLAMAV3_VERSION = "0.0.43"
EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
GLM53_TP2_RUNTIME_SCHEMA = "quant-pipeline.glm53-custom-transformers-exl3-mcg-tp2-runtime.v1"
GLM53_TP4_RUNTIME_SCHEMA = "quant-pipeline.glm53-custom-transformers-exl3-mcg-tp4-runtime.v1"
BITS = 4
TP_SIZE = 2
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
TOP_K = 8
SWIGLU_LIMIT = 10.0
ROUTED_SCALING_FACTOR = 2.5
EXPECTED_MAIN_MATRIX_COUNT = len(MAIN_ROUTED_LAYERS) * NUM_EXPERTS * len(PROJECTIONS)
_HASH = re.compile(r"[0-9a-f]{64}")
_LAYER_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")

_LINEAR_ATTN_TP_PLAN = {
    # The released base plan only covers MLA names plus o_proj.  A KDA layer
    # must keep every head-local producer on the same shard as that rowwise
    # o_proj input; otherwise o_proj receives the full 8192 channels while its
    # local weight accepts 4096 channels at TP2.
    "layers.*.self_attn": "glm53_linear_attention",
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.conv1d": "glm53_linear_attention_conv1d",
    "layers.*.self_attn.forget_gate": "glm53_linear_attention_forget_gate",
    "layers.*.self_attn.forget_gate.f_b_proj": "colwise",
    "layers.*.self_attn.b_proj": "colwise",
    "layers.*.self_attn.g_b_proj": "colwise",
}


def target_tp_size_for_bits(bits: int) -> int:
    """Return the campaign's deliberately qualified deployment width."""

    if bits == 4:
        return 2
    if bits == 6:
        return 4
    raise ValueError("custom GLM-5.3 runtime supports only uniform K4/TP2 or K6/TP4")


def runtime_schema_for_bits(bits: int) -> str:
    target_tp_size_for_bits(bits)
    return GLM53_TP2_RUNTIME_SCHEMA if bits == 4 else GLM53_TP4_RUNTIME_SCHEMA


def _torch():
    return importlib.import_module("torch")


def _seal(value: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_json_seal(value: Mapping[str, Any], *, field: str = "receipt_sha256") -> str:
    digest = value.get(field)
    if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise ValueError(f"{field} is not SHA-256")
    body = copy.deepcopy(dict(value))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        raise ValueError(f"{field} seal differs")
    return digest


def validate_glm53_text_config(config: Any) -> None:
    """Reject any geometry or routing behavior other than released GLM-5.3."""

    expected = {
        "model_type": "glm5_next_text",
        "num_hidden_layers": 45,
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "hidden_act": "silu",
        "swiglu_limit": SWIGLU_LIMIT,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "linear_num_heads": 64,
        "linear_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    for field, wanted in expected.items():
        observed = getattr(config, field, None)
        if observed != wanted:
            raise ValueError(f"GLM-5.3 runtime requires {field}={wanted!r}; found {observed!r}")
    if getattr(config, "scoring_func", "sigmoid") != "sigmoid":
        raise ValueError("GLM-5.3 runtime requires sigmoid router scoring")
    if getattr(config, "topk_method", "noaux_tc") != "noaux_tc":
        raise ValueError("GLM-5.3 runtime requires noaux_tc routing")
    layer_types = getattr(config, "mlp_layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or len(layer_types) != 45:
        raise ValueError("GLM-5.3 runtime requires the sealed 45-entry MLP layer plan")
    sparse = tuple(i for i, value in enumerate(layer_types) if value == "sparse")
    if sparse != tuple(MAIN_ROUTED_LAYERS):
        raise ValueError(f"GLM-5.3 sparse-layer plan differs: {sparse}")


def augment_glm53_linear_attention_tp_plan(
    plan: Mapping[str, str] | None,
) -> dict[str, str]:
    """Add the missing coherent head-wise KDA tensor-parallel plan."""

    result = dict(plan or {})
    result.update(_LINEAR_ATTN_TP_PLAN)
    return result


def augment_glm53_tp_plan(plan: Mapping[str, str] | None) -> dict[str, str]:
    """Return the complete GLM-5.3 native-body plus packed-expert TP plan."""

    result = augment_glm53_linear_attention_tp_plan(plan)
    for key in (
        "layers.*.mlp.experts.gate_up_proj",
        "layers.*.mlp.experts.down_proj",
        "layers.*.mlp.experts",
    ):
        result.pop(key, None)
    result.update(
        {
            "layers.*.mlp.experts.*.gate_proj": "glm53_exl3_colwise",
            "layers.*.mlp.experts.*.up_proj": "glm53_exl3_colwise",
            "layers.*.mlp.experts.*.down_proj": "glm53_exl3_rowwise",
        }
    )
    return result


def _divide_exact(value: int, divisor: int, *, name: str) -> int:
    if divisor <= 0 or value % divisor:
        raise ValueError(f"{name}={value} is not divisible by TP size {divisor}")
    return value // divisor


def _localize_linear_attention_attributes(module: Any, tp_size: int) -> None:
    """Make KDA reshape metadata describe this rank's head shard."""

    if type(module).__name__ != "Glm5NextTextLinearAttention":
        return
    if getattr(module, "_glm53_linear_attention_tp_size", None) is not None:
        if module._glm53_linear_attention_tp_size != tp_size:
            raise RuntimeError("linear-attention module was localized for another TP size")
        return
    module.num_heads = _divide_exact(int(module.num_heads), tp_size, name="linear num_heads")
    module.qkv_dim = _divide_exact(int(module.qkv_dim), tp_size, name="linear qkv_dim")
    module.conv_dim = _divide_exact(int(module.conv_dim), tp_size, name="linear conv_dim")
    module._glm53_linear_attention_tp_size = tp_size


def _localize_forget_gate_attributes(module: Any, tp_size: int) -> None:
    if type(module).__name__ != "Glm5NextTextForgetGate":
        raise TypeError(f"forget-gate TP style received {type(module).__name__}")
    if getattr(module, "_glm53_linear_attention_tp_size", None) is not None:
        if module._glm53_linear_attention_tp_size != tp_size:
            raise RuntimeError("forget gate was localized for another TP size")
        return
    module.num_heads = _divide_exact(int(module.num_heads), tp_size, name="forget-gate num_heads")
    module.qkv_dim = _divide_exact(int(module.qkv_dim), tp_size, name="forget-gate qkv_dim")
    module._glm53_linear_attention_tp_size = tp_size


def _localize_depthwise_conv1d_attributes(module: Any, tp_size: int) -> None:
    torch = _torch()
    if not isinstance(module, torch.nn.Conv1d):
        raise TypeError(f"linear-attention conv TP style received {type(module).__name__}")
    if not (module.in_channels == module.out_channels == module.groups):
        raise ValueError("GLM-5.3 linear-attention conv1d must remain depthwise")
    if getattr(module, "_glm53_linear_attention_tp_size", None) is not None:
        if module._glm53_linear_attention_tp_size != tp_size:
            raise RuntimeError("linear-attention conv1d was localized for another TP size")
        return
    channels = _divide_exact(int(module.in_channels), tp_size, name="linear conv1d channels")
    module.in_channels = channels
    module.out_channels = channels
    module.groups = channels
    module._glm53_linear_attention_tp_size = tp_size


def _shard_concatenated_qkv_channels(value, *, tp_size: int, tp_rank: int):
    """Select one head shard from each member of a concatenated Q/K/V tensor.

    GLM stores the depthwise-convolution channels as ``[Q_all, K_all, V_all]``
    while the TP forward constructs ``[Q_local, K_local, V_local]``.  A plain
    contiguous shard of the stored tensor therefore has the wrong channel
    order.  Keep the three equal regions distinct while selecting a rank.
    """

    if tp_size <= 0 or tp_rank not in range(tp_size):
        raise ValueError(f"invalid TP coordinate: size={tp_size}, rank={tp_rank}")
    if value.ndim < 1 or int(value.shape[0]) % (3 * tp_size):
        raise ValueError(
            f"concatenated QKV channels={int(value.shape[0]) if value.ndim else 0} "
            f"are not divisible by 3 * TP size {tp_size}"
        )
    q, k, v = value.chunk(3, dim=0)
    return _torch().cat(
        [part.chunk(tp_size, dim=0)[tp_rank] for part in (q, k, v)], dim=0
    ).contiguous()


def _finalize_linear_attention_local_parameters(model: Any) -> dict[str, int]:
    """Detach head-sharded direct-use parameters from DTensor after loading.

    GLM's KDA forward reads ``conv1d.weight`` directly instead of calling the
    Conv1d module.  Consequently the normal TP forward context never gets a
    chance to expose the local shard, and an inference-mode view on the DTensor
    fails before the convolution.  The checkpoint loader has already selected
    the correct shard at this point, so retain that shard as an ordinary local
    Parameter for the runtime-only model.
    """

    torch = _torch()
    from torch.distributed.tensor import DTensor

    counts = {"conv1d": 0, "forget_gate": 0, "localized_parameters": 0}

    def local_parameter(
        value, *, concatenated_qkv: bool = False, tp_size: int | None = None
    ):
        if not isinstance(value, DTensor):
            return value
        source = value.to_local()
        if concatenated_qkv:
            if tp_size is None:
                raise RuntimeError("QKV convolution localization lacks its TP size")
            tp_rank = torch.distributed.get_rank(group=value.device_mesh.get_group())
            source = _shard_concatenated_qkv_channels(
                source, tp_size=tp_size, tp_rank=tp_rank
            )
        # A tensor materialized under inference mode has no version counter;
        # allocate a fresh ordinary tensor so downstream view ops are legal.
        with torch.inference_mode(False), torch.no_grad():
            local = torch.empty(source.shape, dtype=source.dtype, device=source.device)
            local.copy_(source)
        counts["localized_parameters"] += 1
        return torch.nn.Parameter(local, requires_grad=value.requires_grad)

    for _, module in model.named_modules():
        if getattr(module, "_glm53_linear_attention_tp_size", None) is None:
            continue
        if isinstance(module, torch.nn.Conv1d):
            counts["conv1d"] += 1
            for name in ("weight", "bias"):
                value = module._parameters.get(name)
                if value is not None:
                    module._parameters[name] = local_parameter(
                        value,
                        concatenated_qkv=True,
                        tp_size=int(module._glm53_linear_attention_tp_size),
                    )
            if int(module.weight.shape[0]) != int(module.out_channels):
                raise RuntimeError(
                    "localized QKV convolution weight channels differ: "
                    f"weight={int(module.weight.shape[0])}, out_channels={int(module.out_channels)}"
                )
        elif type(module).__name__ == "Glm5NextTextForgetGate":
            counts["forget_gate"] += 1
            for name in ("dt_bias", "A_log"):
                value = module._parameters.get(name)
                if value is None:
                    raise RuntimeError(f"linear-attention forget gate lacks {name}")
                module._parameters[name] = local_parameter(value)
    if (
        counts["conv1d"] <= 0
        or counts["conv1d"] != counts["forget_gate"]
        or counts["localized_parameters"] != counts["conv1d"] * 3
    ):
        raise RuntimeError(f"linear-attention local-parameter census differs: {counts}")
    return counts


def verify_exllamav3_source(source: str | Path) -> Path:
    """Bind the runtime to the reviewed ExLlamaV3 commit without mutating it."""

    root = Path(source).expanduser().resolve()
    if not (root / "exllamav3/modules/quant/exl3.py").is_file():
        raise ValueError(f"not an ExLlamaV3 source checkout: {root}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    if head != EXLLAMAV3_COMMIT:
        raise ValueError(f"ExLlamaV3 commit differs: expected {EXLLAMAV3_COMMIT}, found {head}")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=root, check=False).returncode != 0:
            raise ValueError("reviewed ExLlamaV3 checkout has tracked modifications")
    return root


def _install_minimal_exllamav3_package(root: Path) -> None:
    """Expose only the reviewed EXL3 kernel path, not optional serving extras.

    ExLlamaV3's top-level package eagerly imports FlashAttention and Formatron,
    neither of which is used by this Transformers adapter.  Namespace parents
    let Python load ``modules.quant.exl3`` and its CUDA extension directly from
    the verified checkout without claiming those optional serving dependencies.
    """

    package_root = root / "exllamav3"
    loaded = sys.modules.get("exllamav3")
    if loaded is not None:
        loaded_root = Path(str(getattr(loaded, "__file__", ""))).resolve().parent
        if loaded_root != package_root:
            raise RuntimeError(f"another ExLlamaV3 source is already imported: {loaded_root}")
        return
    unexpected = [name for name in sys.modules if name.startswith("exllamav3.")]
    if unexpected:
        raise RuntimeError(f"partial ExLlamaV3 import already exists: {unexpected[:3]}")
    package_paths = {
        "exllamav3": package_root,
        "exllamav3.modules": package_root / "modules",
        "exllamav3.modules.quant": package_root / "modules/quant",
        "exllamav3.model": package_root / "model",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module
    config = types.ModuleType("exllamav3.model.config")
    config.__file__ = str(package_root / "model/config.py")
    config.__package__ = "exllamav3.model"
    config.Config = type("Config", (), {})
    sys.modules[config.__name__] = config


def load_tensor_storage(model_root: str | Path) -> dict[str, Mapping[str, Any]]:
    root = Path(model_root).resolve()
    config = _read_json(root / "config.json")
    qcfg = config.get("quantization_config")
    storage_file = _read_json(root / "quantization_config.json")
    materialization = _read_json(root / "materialization-receipt.json")
    storage_abi = _read_json(root / "exl3-mcg-storage-abi.json")
    _verify_json_seal(materialization)
    storage_abi_sha = _verify_json_seal(storage_abi)
    bits = qcfg.get("bits") if isinstance(qcfg, Mapping) else None
    if bits not in (4, 6):
        raise ValueError("checkpoint does not declare uniform routed K4 or K6")
    if (
        not isinstance(qcfg, Mapping)
        or qcfg.get("quant_method") != "exl3"
        or qcfg.get("version") != EXLLAMAV3_VERSION
        or qcfg.get("bits") != bits
        or qcfg.get("codebook") != "mcg"
        or qcfg.get("scope") != "glm53_routed_experts_only"
        or qcfg.get("non_routed_dtype_policy") != "official_source_native"
        or storage_file.get("quant_method") != "exl3"
        or storage_file.get("bits") != bits
        or storage_file.get("codebook") != "mcg"
        or materialization.get("schema") != materialization_receipt_schema_for_bits(bits)
        or materialization.get("bits") != bits
        or materialization.get("codec_family") != "exl3-mcg"
        or materialization.get("nonrouted_native_exact") is not True
        or materialization.get("main_and_mtp_complete") is not True
        or materialization.get("complete") is not True
        or materialization.get("storage_abi_receipt_sha256") != storage_abi_sha
        or materialization.get("config_sha256") != sha256_file(root / "config.json")
        or materialization.get("quantization_config_sha256")
        != sha256_file(root / "quantization_config.json")
        or storage_abi.get("schema") != "quant-pipeline.glm53-exl3-mcg-storage-abi.v1"
        or storage_abi.get("bits") != bits
        or storage_abi.get("storage_checkpoint_verified") is not True
        or storage_abi.get("serving_reader_qualified") is not False
        or storage_abi.get("qualified_tp_sizes") != []
    ):
        raise ValueError("checkpoint does not declare the selective GLM-5.3 EXL3/MCG contract")
    storage = storage_file.get("tensor_storage")
    if not isinstance(storage, dict):
        raise ValueError("quantization_config.json has no tensor_storage map")
    expected_modules = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in MAIN_ROUTED_LAYERS
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    expected_mtp_modules = {
        f"model.language_model.layers.{MTP_LAYER}.mlp.experts.{expert}.{projection}"
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    observed_main = {name for name in storage if f".layers.{MTP_LAYER}." not in name}
    if observed_main != expected_modules or set(storage) != expected_modules | expected_mtp_modules:
        missing = len(expected_modules - observed_main)
        extra = len(observed_main - expected_modules)
        raise ValueError(
            f"main/MTP packed tensor-storage census differs: main_missing={missing}, main_extra={extra}"
        )
    for module in expected_modules | expected_mtp_modules:
        row = storage[module]
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid packed group: {module}")
        tensors = row.get("stored_tensors")
        if (
            row.get("quant_format") != "exl3"
            or row.get("bits_per_weight") != bits
            or row.get("mcg_multiplier") != int("CBAC1FED", 16)
            or not isinstance(tensors, Mapping)
            or set(tensors) != {f"{module}.{suffix}" for suffix in ("trellis", "suh", "svh", "mcg")}
        ):
            raise ValueError(f"incomplete packed group: {module}")
    return storage


def _dtype(name: str):
    torch = _torch()
    table = {
        "torch.int16": torch.int16,
        "torch.int32": torch.int32,
        "torch.float16": torch.float16,
        "int16": torch.int16,
        "int32": torch.int32,
        "float16": torch.float16,
    }
    try:
        return table[name]
    except KeyError as error:
        raise ValueError(f"unsupported EXL3 storage dtype: {name}") from error


class PackedMCGLinear(_torch().nn.Module):
    """One packed projection; no dense routed weight is ever registered."""

    def __init__(self, in_features: int, out_features: int, metadata: Mapping[str, Any], role: str):
        super().__init__()
        torch = _torch()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.role = role
        self._hf_quantized_needs_local_tp = True
        stored = metadata.get("stored_tensors")
        if not isinstance(stored, Mapping) or len(stored) != 4:
            raise ValueError("packed linear metadata is incomplete")
        suffix_rows = {name.rsplit(".", 1)[-1]: row for name, row in stored.items()}
        if set(suffix_rows) != {"trellis", "suh", "svh", "mcg"}:
            raise ValueError("packed linear suffix census differs")
        for suffix, row in suffix_rows.items():
            shape = row.get("shape")
            if not isinstance(shape, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in shape):
                raise ValueError(f"invalid {suffix} shape metadata")
            tensor = torch.empty(tuple(shape), dtype=_dtype(str(row.get("dtype"))), device="meta")
            self.register_parameter(suffix, torch.nn.Parameter(tensor, requires_grad=False))
        trellis_words = int(suffix_rows["trellis"]["shape"][-1])
        if trellis_words not in (64, 96):
            raise ValueError(f"packed trellis rate differs: {trellis_words} words")
        self.bits = trellis_words // 16
        self._inner = None
        self._inner_signature: tuple[int, ...] | None = None

    @property
    def weight(self):
        # Some Transformers paths inspect .weight.dtype.  A meta scalar does not
        # hold routed values and is deliberately not a registered parameter.
        return _torch().empty((), dtype=_torch().float16, device="meta")

    def local_tensors(self) -> dict[str, Any]:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            DTensor = ()
        result = {}
        for name in ("trellis", "suh", "svh", "mcg"):
            value = self._parameters[name]
            result[name] = value.to_local() if isinstance(value, DTensor) else value
        return result

    def validate_loaded(self, *, tp_size: int) -> None:
        torch = _torch()
        tensors = self.local_tensors()
        if any(value.is_meta for value in tensors.values()):
            raise RuntimeError("packed projection was not loaded from checkpoint")
        if (
            tensors["trellis"].dtype != torch.int16
            or tensors["suh"].dtype != torch.float16
            or tensors["svh"].dtype != torch.float16
            or tensors["mcg"].dtype != torch.int32
            or int(tensors["mcg"].reshape(-1)[0].item()) != MCG_MARKER_SIGNED_INT32
        ):
            raise RuntimeError("packed projection dtype or MCG marker differs")
        local_in = self.in_features // tp_size if self.role == "row" else self.in_features
        local_out = self.out_features // tp_size if self.role == "col" else self.out_features
        expected = {
            "trellis": (local_in // 16, local_out // 16, self.bits * 16),
            "suh": (local_in,),
            "svh": (local_out,),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise RuntimeError(
                    f"{self.role} TP{tp_size} K{self.bits} {name} shape differs: "
                    f"{tuple(tensors[name].shape)} != {shape}"
                )

    def _ensure_inner(self):
        tensors = self.local_tensors()
        signature = tuple(tensors[name].data_ptr() for name in ("trellis", "suh", "svh", "mcg"))
        if self._inner is None or signature != self._inner_signature:
            cls = importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3
            self._inner = cls(
                config=None,
                in_features=int(tensors["suh"].numel()),
                out_features=int(tensors["svh"].numel()),
                trellis=tensors["trellis"],
                suh=tensors["suh"],
                svh=tensors["svh"],
                mcg=tensors["mcg"],
                out_dtype=_torch().float16,
                transformers_fix=True,
            )
            self._inner_signature = signature
        return self._inner

    def forward(self, hidden_states):
        output = self._ensure_inner().forward(
            hidden_states.half(), {}, out_dtype=_torch().float32
        )
        # Column-parallel gate/up values re-enter the BF16 model immediately.
        # Keep row-parallel down-projection partials in FP32 until the expert
        # container has accumulated them and performed its single TP reduction.
        return output if self.role == "row" else output.to(_torch().bfloat16)


class _PackedExpert(_torch().nn.Module):
    def __init__(self, metadata: Mapping[str, Mapping[str, Any]]):
        super().__init__()
        self.gate_proj = PackedMCGLinear(HIDDEN_SIZE, INTERMEDIATE_SIZE, metadata["gate_proj"], "col")
        self.up_proj = PackedMCGLinear(HIDDEN_SIZE, INTERMEDIATE_SIZE, metadata["up_proj"], "col")
        self.down_proj = PackedMCGLinear(INTERMEDIATE_SIZE, HIDDEN_SIZE, metadata["down_proj"], "row")


class PackedTP2Experts(_torch().nn.ModuleList):
    """Exact eager GLM-5.3 expert dispatch over TP-sharded EXL3 projections.

    The historical class name is retained for checkpoint/runtime compatibility;
    instances are rate-aware and support the sealed K4/TP2 and K6/TP4 plans.
    """

    def __init__(self, storage: Mapping[str, Mapping[str, Any]], module_path: str):
        experts = []
        for expert in range(NUM_EXPERTS):
            experts.append(
                _PackedExpert(
                    {
                        projection: storage[f"{module_path}.{expert}.{projection}"]
                        for projection in PROJECTIONS
                    }
                )
            )
        super().__init__(experts)
        self.num_experts = NUM_EXPERTS
        self.swiglu_limit = SWIGLU_LIMIT
        rates = {
            getattr(expert, projection).bits
            for expert in self
            for projection in PROJECTIONS
        }
        if len(rates) != 1:
            raise ValueError(f"mixed routed rates are forbidden: {sorted(rates)}")
        self.bits = rates.pop()
        self.tp_size = target_tp_size_for_bits(self.bits)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        torch = _torch()
        import torch.nn.functional as F

        if top_k_index.shape[-1] != TOP_K:
            raise RuntimeError("GLM-5.3 packed runtime received non-top8 routing")
        final = torch.zeros(
            hidden_states.shape, dtype=torch.float32, device=hidden_states.device
        )
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert_idx in hit:
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            expert = self[expert_idx]
            gate = expert.gate_proj(hidden_states[token_idx]).clamp(max=self.swiglu_limit)
            up = expert.up_proj(hidden_states[token_idx]).clamp(
                min=-self.swiglu_limit, max=self.swiglu_limit
            )
            current = expert.down_proj(F.silu(gate) * up)
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(final)
        return final.to(hidden_states.dtype)

    def validate_loaded(self) -> int:
        count = 0
        for expert in self:
            for projection in PROJECTIONS:
                getattr(expert, projection).validate_loaded(tp_size=self.tp_size)
                count += 1
        return count


def _install_tp_styles() -> None:
    torch = _torch()
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, TensorParallelLayer

    class _PackedColwise(TensorParallelLayer):
        def should_use_local_tensors(self, module):
            return True

        def shard_param(self, module, param, mesh):
            value = module._parameters.get(param)
            if value is None:
                return
            placement = Shard(1) if param == "trellis" else Shard(0) if param == "svh" else Replicate()
            module._parameters[param] = torch.nn.Parameter(
                distribute_tensor(value, mesh, [placement], src_data_rank=None), requires_grad=False
            )

    class _PackedRowwise(TensorParallelLayer):
        def should_use_local_tensors(self, module):
            return True

        def shard_param(self, module, param, mesh):
            value = module._parameters.get(param)
            if value is None:
                return
            placement = Shard(0) if param in {"trellis", "suh"} else Replicate()
            module._parameters[param] = torch.nn.Parameter(
                distribute_tensor(value, mesh, [placement], src_data_rank=None), requires_grad=False
            )

        def transform_output_post_forward(self, module, output, mesh):
            # PackedTP2Experts combines all selected expert partials in FP32
            # and issues one reduction per layer.  Reducing every expert here
            # is both slower and introduces repeated BF16 rounding.
            return output

    class _LinearAttentionModule(TensorParallelLayer):
        def install_forward(self, module, mesh):
            _localize_linear_attention_attributes(module, int(mesh.size()))
            return module

    class _LinearAttentionConv1d(TensorParallelLayer):
        def should_use_local_tensors(self, module):
            return True

        def validate_param(self, module, param, mesh, parameter_name=None):
            value = module._parameters.get(param)
            if value is not None and param not in {"weight", "bias"}:
                raise ValueError(f"unexpected conv1d parameter: {parameter_name or param}")
            if value is not None:
                _divide_exact(
                    int(value.shape[0]), 3 * int(mesh.size()), name="conv1d QKV parameter channels"
                )

        def shard_param(self, module, param, mesh):
            value = module._parameters.get(param)
            if value is None:
                return
            module._parameters[param] = torch.nn.Parameter(
                # Load the small tensor replicated, then select the matching
                # head slice independently from Q, K, and V after checkpoint
                # loading. A plain Shard(0) would split across QKV regions.
                distribute_tensor(value, mesh, [Replicate()], src_data_rank=None),
                requires_grad=value.requires_grad,
            )

        def install_forward(self, module, mesh):
            _localize_depthwise_conv1d_attributes(module, int(mesh.size()))
            return super().install_forward(module, mesh)

    class _LinearAttentionForgetGate(TensorParallelLayer):
        def should_use_local_tensors(self, module):
            return True

        def validate_param(self, module, param, mesh, parameter_name=None):
            value = module._parameters.get(param)
            if value is not None and param not in {"dt_bias", "A_log"}:
                raise ValueError(f"unexpected forget-gate parameter: {parameter_name or param}")
            if value is not None:
                _divide_exact(int(value.shape[0]), int(mesh.size()), name=f"forget-gate {param}")

        def shard_param(self, module, param, mesh):
            value = module._parameters.get(param)
            if value is None:
                return
            module._parameters[param] = torch.nn.Parameter(
                distribute_tensor(value, mesh, [Shard(0)], src_data_rank=None),
                requires_grad=value.requires_grad,
            )

        def install_forward(self, module, mesh):
            _localize_forget_gate_attributes(module, int(mesh.size()))
            return super().install_forward(module, mesh)

    ALL_PARALLEL_STYLES["glm53_exl3_colwise"] = _PackedColwise()
    ALL_PARALLEL_STYLES["glm53_exl3_rowwise"] = _PackedRowwise()
    ALL_PARALLEL_STYLES["glm53_linear_attention"] = _LinearAttentionModule()
    ALL_PARALLEL_STYLES["glm53_linear_attention_conv1d"] = _LinearAttentionConv1d()
    ALL_PARALLEL_STYLES["glm53_linear_attention_forget_gate"] = _LinearAttentionForgetGate()


def _replace_experts(model, storage: Mapping[str, Mapping[str, Any]]) -> list[str]:
    replaced: list[str] = []
    for name, module in tuple(model.named_modules()):
        if type(module).__name__ != "Glm5NextTextExperts":
            continue
        match = _LAYER_PATH.search(name)
        if match is None or int(match.group(1)) not in MAIN_ROUTED_LAYERS:
            raise ValueError(f"unexpected GLM-5.3 experts module: {name}")
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, PackedTP2Experts(storage, name))
        replaced.append(name)
    expected = [f"model.language_model.layers.{layer}.mlp.experts" for layer in MAIN_ROUTED_LAYERS]
    if replaced != expected:
        raise ValueError(f"GLM-5.3 expert replacement census differs: {replaced}")
    return replaced


def packed_runtime_census(model) -> dict[str, Any]:
    containers = [(name, module) for name, module in model.named_modules() if isinstance(module, PackedTP2Experts)]
    expected_paths = [f"model.language_model.layers.{layer}.mlp.experts" for layer in MAIN_ROUTED_LAYERS]
    if [name for name, _ in containers] != expected_paths:
        raise RuntimeError("loaded packed expert-container census differs")
    matrices = sum(module.validate_loaded() for _, module in containers)
    rates = {module.bits for _, module in containers}
    if len(rates) != 1:
        raise RuntimeError(f"mixed packed expert rates are forbidden: {sorted(rates)}")
    bits = rates.pop()
    forbidden = [
        name
        for name, _ in model.named_parameters()
        if ".mlp.experts." in name and name.endswith(".weight")
    ]
    if forbidden:
        raise RuntimeError(f"routed BF16 fallback parameters exist: {forbidden[:3]}")
    if matrices != EXPECTED_MAIN_MATRIX_COUNT:
        raise RuntimeError("packed routed matrix census differs")
    return {
        "main_routed_layers": list(MAIN_ROUTED_LAYERS),
        "expert_count_per_layer": NUM_EXPERTS,
        "packed_matrix_count": matrices,
        "bits": bits,
        "tp_size": target_tp_size_for_bits(bits),
        "bf16_routed_weight_parameter_count": 0,
        "swiglu_limit": SWIGLU_LIMIT,
        "router": {
            "scoring": "sigmoid",
            "selection": "noaux_tc",
            "top_k": TOP_K,
            "norm_topk_prob": True,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        },
        "mtp_layer_45": "stored_but_not_executed",
    }


def patch_transformers(*, exllamav3_source: str | Path) -> None:
    """Register the custom quantizer and DTensor styles before from_pretrained."""

    root = verify_exllamav3_source(exllamav3_source)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _install_minimal_exllamav3_package(root)
    transformers = importlib.import_module("transformers")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"custom GLM-5.3 runtime requires Transformers {TRANSFORMERS_VERSION}; found {transformers.__version__}"
        )
    _install_tp_styles()
    from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING
    from transformers.quantizers.base import HfQuantizer
    from transformers.utils.quantization_config import QuantizationConfigMixin

    class Glm53Exl3Config(QuantizationConfigMixin):
        quant_method = "exl3"

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            return dict(self.__dict__, quant_method="exl3")

    class Glm53Exl3Quantizer(HfQuantizer):
        requires_calibration = False

        def validate_environment(self, **kwargs):
            torch = _torch()
            if not torch.cuda.is_available():
                raise RuntimeError("GLM-5.3 packed runtime requires CUDA")
            bits = int(getattr(self.quantization_config, "bits", 0))
            tp_size = target_tp_size_for_bits(bits)
            if int(os.environ.get("WORLD_SIZE", "0")) != tp_size:
                raise RuntimeError(
                    f"GLM-5.3 packed K{bits} runtime requires torchrun WORLD_SIZE={tp_size}"
                )

        def update_dtype(self, dtype):
            return _torch().bfloat16

        def update_tp_plan(self, config):
            text = config.get_text_config()
            validate_glm53_text_config(text)
            text.base_model_tp_plan = augment_glm53_tp_plan(text.base_model_tp_plan)
            return config

        def update_ep_plan(self, config):
            distributed = getattr(config, "distributed_config", None)
            if getattr(distributed, "enable_expert_parallel", False):
                raise ValueError("custom packed runtime supports tensor parallelism only; expert parallelism must be disabled")
            return config

        def _process_model_before_weight_loading(self, model, **kwargs):
            validate_glm53_text_config(model.config.get_text_config())
            storage = load_tensor_storage(model.name_or_path)
            self.replaced_paths = _replace_experts(model, storage)
            return model

        def _process_model_after_weight_loading(self, model, **kwargs):
            model._glm53_linear_attention_tp_census = (
                _finalize_linear_attention_local_parameters(model)
            )
            model._glm53_packed_tp2_census = packed_runtime_census(model)
            return model

        @property
        def is_trainable(self):
            return False

        def is_serializable(self, safe_serialization=None):
            return False

    existing_config = AUTO_QUANTIZATION_CONFIG_MAPPING.get("exl3")
    existing_quantizer = AUTO_QUANTIZER_MAPPING.get("exl3")
    if existing_config not in (None, Glm53Exl3Config) or existing_quantizer not in (None, Glm53Exl3Quantizer):
        raise RuntimeError("another EXL3 Transformers integration is already registered")
    AUTO_QUANTIZATION_CONFIG_MAPPING["exl3"] = Glm53Exl3Config
    AUTO_QUANTIZER_MAPPING["exl3"] = Glm53Exl3Quantizer


def build_runtime_receipt(
    *,
    rank_reports: Sequence[Mapping[str, Any]],
    runtime_module: str | Path,
    exllamav3_source: str | Path,
    reference: Mapping[str, Any] | None,
    generation_verified: bool,
    bits: int,
    tp_size: int | None = None,
) -> dict[str, Any]:
    """Seal qualification evidence; incomplete evidence produces qualified=false."""

    expected_tp = target_tp_size_for_bits(bits)
    tp_size = expected_tp if tp_size is None else tp_size
    if tp_size != expected_tp:
        raise ValueError(f"K{bits} qualification requires TP{expected_tp}, not TP{tp_size}")
    reasons: list[str] = []
    if len(rank_reports) != tp_size or [row.get("rank") for row in rank_reports] != list(range(tp_size)):
        reasons.append(f"rank census is not exactly TP{tp_size} ranks 0..{tp_size - 1}")
    for row in rank_reports:
        if (
            row.get("world_size") != tp_size
            or row.get("bits") != bits
            or row.get("packed_matrix_count") != EXPECTED_MAIN_MATRIX_COUNT
            or row.get("bf16_routed_weight_parameter_count") != 0
            or not isinstance(row.get("peak_cuda_bytes"), int)
            or row.get("peak_cuda_bytes", 0) <= 0
            or not isinstance(row.get("steady_cuda_bytes"), int)
            or row.get("steady_cuda_bytes", 0) <= 0
            or not isinstance(row.get("device"), str)
            or row.get("generated_token_count", 0) < 2
        ):
            reasons.append(f"rank {row.get('rank')} load/census/memory evidence is incomplete")
    if not generation_verified:
        reasons.append("prefill plus multi-step generation was not verified")
    reference_complete = (
        isinstance(reference, Mapping)
        and reference.get("passed") is True
        and _HASH.fullmatch(str(reference.get("sha256", ""))) is not None
        and isinstance(reference.get("input_ids_shape"), list)
        and isinstance(reference.get("shape"), list)
        and isinstance(reference.get("max_abs_error"), (int, float))
        and isinstance(reference.get("mean_abs_error"), (int, float))
        and isinstance(reference.get("max_abs_tolerance"), (int, float))
        and isinstance(reference.get("mean_abs_tolerance"), (int, float))
        and reference["max_abs_error"] <= reference["max_abs_tolerance"]
        and reference["mean_abs_error"] <= reference["mean_abs_tolerance"]
    )
    if not reference_complete:
        reasons.append(f"full-logit parity against decoded-K{bits} reference did not pass")
    qualified = not reasons
    verify_exllamav3_source(exllamav3_source)
    body = {
        "schema": runtime_schema_for_bits(bits),
        "runtime_kind": "custom_transformers_glm53_with_exllamav3_ext",
        "stock_vllm_compatible": False,
        "stock_exllamav3_model_compatible": False,
        "transformers_version": TRANSFORMERS_VERSION,
        "exllamav3_version": EXLLAMAV3_VERSION,
        "exllamav3_commit": EXLLAMAV3_COMMIT,
        "runtime_module_sha256": sha256_file(Path(runtime_module)),
        "bits": bits,
        "tp_size": tp_size,
        "expert_parallel": False,
        "native_nonrouted_policy": "official_bf16_tensor_parallel",
        "routed_policy": f"packed_exl3_mcg_k{bits}_no_dense_fallback",
        "packed_projection_numeric_policy": "fp16_input_fp32_row_accumulation_single_reduce_then_bf16",
        "main_packed_matrix_count": EXPECTED_MAIN_MATRIX_COUNT,
        "mtp_layer_45": "stored_but_not_executed",
        "rank_reports": [dict(row) for row in rank_reports],
        "reference_logit_parity": dict(reference) if isinstance(reference, Mapping) else None,
        "generation_verified": bool(generation_verified),
        "qualified": qualified,
        "failure_reasons": reasons,
    }
    return _seal(body)


def build_tp2_runtime_receipt(
    *,
    rank_reports: Sequence[Mapping[str, Any]],
    runtime_module: str | Path,
    exllamav3_source: str | Path,
    reference: Mapping[str, Any] | None,
    generation_verified: bool,
) -> dict[str, Any]:
    """Backward-compatible K4/TP2 receipt entrypoint."""

    normalized = [dict(row, bits=row.get("bits", 4)) for row in rank_reports]
    return build_runtime_receipt(
        rank_reports=normalized,
        runtime_module=runtime_module,
        exllamav3_source=exllamav3_source,
        reference=reference,
        generation_verified=generation_verified,
        bits=4,
        tp_size=2,
    )
