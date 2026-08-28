"""Concrete layer-streaming GLM-MoE-DSA runtime for Round 7.

The runtime constructs one official Transformers ``GlmMoeDsaDecoderLayer`` on
``meta``, supplies only the carrier tensors needed by that layer, advances each
preserved prompt, and releases the layer.  Packed carried linears are decoded
from their byte-audited EXL3 payload; routed experts are supplied only from the
new Round 7 packed-decoded records.  This keeps the 1.4 TB BF16 source and the
full carrier off GPU while preserving the owner-locked sequential state.

Importing this module is inert.  Model/tokenizer/CUDA initialization occurs only
inside the explicit owner-run factory.
"""

from __future__ import annotations

import json
import inspect
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .constants import (
    EXPERTS_IMPLEMENTATION,
    FIRST_MOE_LAYER,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAST_MOE_LAYER,
    MCG_MULT,
    NUM_EXPERTS,
    TOP_K,
    TensorId,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    configure_deterministic_environment,
    derive_seed,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .glm52_backend import (
    GLM52Runtime,
    INSTALL_AUDIT_ROWS,
    INSTALL_MAX_ABS_ERROR,
    INSTALL_MAX_RELATIVE_L2,
)
from .inventory import (
    load_checkpoint_inventory,
    load_numeric_environment,
    load_runtime_code_inventory,
    verify_checkpoint_inventory,
)
from .oracles import audit_v2_layer
from .safetensors_io import (
    SafeTensorReader,
    read_torch_tensor,
    torch_tensor_entry,
    write_safetensors_atomic,
)
from .schema import tensor_name
from .trellis import CodecConfig, Exl3TrellisCodec
from .types import RoutedBatch, StateShard

CORPUS_SEED = 20260711
TARGET_TOKENS = 1_048_576
MAX_SAMPLE_TOKENS = 4096
MIN_SAMPLE_TOKENS = 8
ATTENTION_IMPLEMENTATION = "eager"


def _tensor_sha256(value) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    return sha256_bytes(tensor.view(torch.uint8).numpy().tobytes())


def _read_tensor(reader: SafeTensorReader, name: str):
    import torch

    info = reader.tensors[name]
    dtypes = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
    }
    try:
        dtype = dtypes[info.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported carrier dtype {info.dtype}: {name}") from exc
    raw = bytearray().join(info.payload.chunks())
    return torch.frombuffer(raw, dtype=dtype).reshape(info.shape).clone()


def _projection_from_key(key: str) -> str:
    projection = key.rsplit("/", 1)[-1]
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(f"invalid installed projection key {key!r}")
    return projection


class _CarrierLoader:
    """Inventory-checked direct and full-rank EXL3 tensor reader."""

    def __init__(
        self,
        root: Path,
        inventory: Mapping[str, object],
        codec,
        *,
        verify_payloads: bool = True,
    ) -> None:
        self.root = root
        self.entries: Mapping[str, Mapping[str, object]] = inventory["entries"]  # type: ignore[assignment]
        self.codec = codec
        self.readers: dict[str, SafeTensorReader] = {}
        self.audit_records: dict[str, dict[str, object]] = {}
        self.verify_payloads = bool(verify_payloads)

    def has(self, name: str) -> bool:
        return name in self.entries

    def tensor(self, name: str):
        try:
            record = self.entries[name]
        except KeyError as exc:
            raise KeyError(f"carrier inventory lacks {name}") from exc
        shard_name = str(record["shard"])
        reader = self.readers.setdefault(
            shard_name, SafeTensorReader(self.root / shard_name)
        )
        info = reader.tensors.get(name)
        if info is None or (
            self.verify_payloads
            and info.payload.sha256() != record["payload_sha256"]
        ):
            raise ValueError(f"carrier payload differs from inventory: {name}")
        return _read_tensor(reader, name)

    def weight(self, prefix: str):
        import torch

        direct = f"{prefix}.weight"
        if self.has(direct):
            value = self.tensor(direct)
            self.audit_records[prefix] = {
                "storage": "direct",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "payload_sha256": self.entries[direct]["payload_sha256"],
            }
            return value

        names = {
            suffix: f"{prefix}.{suffix}" for suffix in ("trellis", "suh", "svh", "mcg")
        }
        if not all(self.has(name) for name in names.values()):
            raise KeyError(
                f"carrier has neither direct nor complete EXL3 weight: {prefix}"
            )
        packed = self.tensor(names["trellis"])
        suh = self.tensor(names["suh"])
        svh = self.tensor(names["svh"])
        marker = int(self.tensor(names["mcg"]).item()) & 0xFFFFFFFF
        if marker != MCG_MULT:
            raise ValueError(f"carrier EXL3 codebook marker drift: {prefix}")
        if packed.ndim != 3 or packed.shape[-1] % 16:
            raise ValueError(f"carrier EXL3 trellis geometry drift: {prefix}")
        bits = int(packed.shape[-1] // 16)
        if not 2 <= bits <= 8:
            raise ValueError(f"carrier EXL3 bit width is unsupported: {prefix}={bits}")
        reconstructed_kn = self.codec.decode_to_original(
            packed.to(self.codec.config.device), suh, svh, bits
        ).to(torch.bfloat16)
        expected = (int(suh.numel()), int(svh.numel()))
        if tuple(reconstructed_kn.shape) != expected:
            raise ValueError(f"carrier EXL3 reconstruction shape drift: {prefix}")
        payload_hashes = {
            suffix: self.entries[name]["payload_sha256"]
            for suffix, name in names.items()
        }
        self.audit_records[prefix] = {
            "storage": "exl3-packed-decoded-bf16",
            "bits": bits,
            "payload_sha256": dict(sorted(payload_hashes.items())),
            "reconstruction_bf16_sha256": _tensor_sha256(reconstructed_kn),
            "shape_kn": list(reconstructed_kn.shape),
            "mcg": f"0x{marker:08X}",
        }
        # PyTorch Linear stores [N,K]; TRELLIS reconstructs [K,N].
        return reconstructed_kn.T.contiguous()

    @property
    def audit_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(dict(sorted(self.audit_records.items())))
        )


class TransformersSequentialRuntime(GLM52Runtime):
    """Single-GPU, one-layer-at-a-time reference runtime for GLM-5.2."""

    def __init__(self, config) -> None:
        environment = configure_deterministic_environment()
        self.owner_config = config
        # Verify the supplied closure before importing anything it is meant to
        # authorize. A second verification below detects import-time mutation.
        verify_runtime = bool(getattr(config, "verify_runtime_files", True))
        self.runtime_inventory = load_runtime_code_inventory(
            config.runtime_inventory, verify_files=verify_runtime
        )
        self._fingerprint = str(self.runtime_inventory["inventory_sha256"])
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was initialized before the sealed deterministic environment"
            )
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
        from transformers.integrations import hub_kernels
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
            GlmMoeDsaConfig,
        )
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaDecoderLayer,
            GlmMoeDsaRotaryEmbedding,
        )

        self.device = torch.device(config.device)
        # Optional device pool for the sharded successor forward. Pool of one
        # (the default) is byte-identical to the original sequential flow.
        _pool = tuple(getattr(config, "devices", ()) or ()) or (config.device,)
        if _pool[0] != config.device:
            raise ValueError("forward device pool must start with the primary device")
        self.forward_devices = tuple(torch.device(item) for item in _pool)
        # Same pool, exposed for the one-shot capture pass so its prompt
        # forward can shard across every GPU instead of using only the primary.
        self.capture_devices = self.forward_devices
        if self.device.type != "cuda":
            raise ValueError(
                "the owner-run GLM reference runtime requires one CUDA device"
            )
        inventoried = {
            str(Path(path).resolve()) for path in self.runtime_inventory["files_sha256"]
        }
        package_root = Path(transformers.__file__).resolve().parent
        tokenizer_root = Path(tokenizers.__file__).resolve().parent
        required = {
            path
            for root in (package_root, tokenizer_root)
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".pyo")
        } | {path.resolve() for path in Path(__file__).resolve().parent.glob("*.py")}
        missing = sorted(str(path) for path in required if str(path) not in inventoried)
        if missing:
            raise ValueError(
                "runtime inventory must cover transformers source and the R7 adapter; "
                f"missing {missing[:8]}"
            )
        if (
            load_runtime_code_inventory(
                config.runtime_inventory, verify_files=verify_runtime
            )["inventory_sha256"]
            != self._fingerprint
        ):
            raise RuntimeError("runtime source changed during module import")

        self.carrier_inventory = load_checkpoint_inventory(
            config.carrier_inventory, role="carrier"
        )
        if bool(getattr(config, "verify_carrier_files", True)):
            verify_checkpoint_inventory(config.carrier, self.carrier_inventory)

        self.config = GlmMoeDsaConfig.from_pretrained(
            str(config.carrier.resolve()), local_files_only=True
        )
        self.config._attn_implementation = ATTENTION_IMPLEMENTATION
        self.config._experts_implementation = EXPERTS_IMPLEMENTATION
        geometry = {
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": INTERMEDIATE_SIZE,
            "n_routed_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "first_k_dense_replace": FIRST_MOE_LAYER,
            "num_hidden_layers": 78,
            "n_group": 1,
            "topk_group": 1,
        }
        for name, expected in geometry.items():
            if int(getattr(self.config, name)) != expected:
                raise ValueError(f"GLM configuration drift: {name}")
        if str(getattr(self.config, "model_type")) != "glm_moe_dsa":
            raise ValueError("carrier is not GLM-MoE-DSA")
        if not bool(self.config.norm_topk_prob):
            raise ValueError("Round 7 requires normalized routed weights")

        if environment["USE_HUB_KERNELS"] != "0" or bool(
            getattr(hub_kernels, "_kernels_enabled", True)
        ):
            raise RuntimeError(
                "Transformers hub kernels were not disabled before import"
            )
        if self.config._experts_implementation != EXPERTS_IMPLEMENTATION:
            raise RuntimeError("GLM expert dispatch is not pinned to eager")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_deterministic_debug_mode("error")
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("PyTorch deterministic algorithms are not enabled")
        # The exact reference arithmetic contract is part of the runtime
        # inventory. It avoids TF32 and algorithm selection drift.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.LayerClass = GlmMoeDsaDecoderLayer
        self.rotary = GlmMoeDsaRotaryEmbedding(self.config, device=self.device).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.carrier.resolve()), local_files_only=True, use_fast=True
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Round 7 requires the inventoried fast tokenizer path")
        # Detect any asset change during config/tokenizer construction before a
        # corpus plan or CUDA forward can be accepted.
        if bool(getattr(config, "verify_carrier_files", True)):
            verify_checkpoint_inventory(config.carrier, self.carrier_inventory)
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaNaiveMoe,
        )

        self.ExpertsClass = GlmMoeDsaNaiveMoe

        wrapped_forward = GlmMoeDsaNaiveMoe.forward
        original_forward = getattr(wrapped_forward, "__wrapped__", None)
        callable_paths = {
            "wrapped_experts_forward": Path(
                inspect.getsourcefile(wrapped_forward) or ""
            ).resolve(),
            "original_experts_forward": Path(
                inspect.getsourcefile(original_forward) or ""
            ).resolve(),
        }
        if any(str(path) not in inventoried for path in callable_paths.values()):
            raise RuntimeError(
                "resolved GLM expert callables are outside runtime inventory"
            )
        self.dispatch_audit = {
            "schema": "r7-runtime-dispatch-v1",
            "environment": environment,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "experts_implementation": self.config._experts_implementation,
            "hub_kernels_enabled": False,
            "callable_sha256": {
                name: self.runtime_inventory["files_sha256"][str(path)]
                for name, path in sorted(callable_paths.items())
            },
            "tokenizer_class": (
                f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__qualname__}"
            ),
            "tokenizers_package": str(tokenizer_root),
            "carrier_auxiliary_assets_sha256": sha256_bytes(
                canonical_json_bytes(self.carrier_inventory["auxiliary_files_sha256"])
            ),
        }
        self.dispatch_audit_sha256 = sha256_bytes(
            canonical_json_bytes(self.dispatch_audit)
        )
        self._codec: Exl3TrellisCodec | None = None
        self.loader = _CarrierLoader(
            config.carrier.resolve(),
            self.carrier_inventory,
            self._get_codec(),
            verify_payloads=bool(getattr(config, "verify_carrier_payloads", True)),
        )
        self._capture_layer: int | None = None
        self._capture_module = None
        self._capture_audits: dict[int, dict[str, object]] = {}
        self._installed: dict[int, dict[int, dict[str, Any]]] = {}

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _get_codec(self) -> Exl3TrellisCodec:
        if self._codec is None:
            numeric = load_numeric_environment(
                self.owner_config.numeric_inventory,
                verify_files=bool(
                    getattr(self.owner_config, "verify_numeric_files", True)
                ),
            )
            self._codec = Exl3TrellisCodec(
                CodecConfig(
                    device=str(self.device),
                    sigma_reg=float(self.owner_config.sigma_reg),
                    numeric_core=Path(str(numeric["numeric_core"])),
                    numeric_core_sha256=str(numeric["numeric_core_sha256"]),
                    extension=Path(str(numeric["extension"])),
                    extension_sha256=str(numeric["extension_sha256"]),
                    verify_files=bool(
                        getattr(self.owner_config, "verify_numeric_files", True)
                    ),
                )
            )
        return self._codec

    def _build_corpus_plan_payload(self, corpus: Path) -> dict[str, object]:
        records: list[tuple[int, str]] = []
        with corpus.open("r", encoding="utf-8") as handle:
            for line_index, raw in enumerate(handle):
                raw = raw.strip()
                if raw:
                    payload = json.loads(raw)
                    text = payload.get("text")
                    if not isinstance(text, str):
                        raise ValueError(
                            f"corpus line {line_index} lacks string `text`"
                        )
                    records.append((line_index, text))
        order = list(range(len(records)))
        random.Random(CORPUS_SEED).shuffle(order)
        selected = []
        total = 0
        for position in order:
            line_index, text = records[position]
            input_ids = [int(value) for value in self.tokenizer.encode(text)]
            if len(input_ids) < MIN_SAMPLE_TOKENS:
                continue
            input_ids = input_ids[:MAX_SAMPLE_TOKENS]
            shard_id = f"prompt-{len(selected):06d}-line-{line_index:06d}"
            selected.append((shard_id, line_index, input_ids, total))
            total += len(input_ids)
            if total >= TARGET_TOKENS:
                break
        if total < TARGET_TOKENS:
            raise ValueError(
                f"calibration corpus exhausted at {total} tokens; need {TARGET_TOKENS}"
            )
        payload: dict[str, object] = {
            "schema": "r7-corpus-plan-v1",
            "seed": CORPUS_SEED,
            "target_tokens": TARGET_TOKENS,
            "max_sample_tokens": MAX_SAMPLE_TOKENS,
            "corpus_sha256": sha256_file(corpus),
            "carrier_inventory_sha256": self.carrier_inventory["inventory_sha256"],
            "runtime_inventory_sha256": self.runtime_inventory["inventory_sha256"],
            "dispatch_audit_sha256": self.dispatch_audit_sha256,
            "selected": [
                {
                    "shard_id": shard_id,
                    "line_index": line_index,
                    "input_ids": input_ids,
                    "global_row_start": start,
                    "tokens": len(input_ids),
                }
                for shard_id, line_index, input_ids, start in selected
            ],
        }
        payload["corpus_plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
        return payload

    def prepare_corpus_plan(self, *, corpus: Path) -> Mapping[str, object]:
        plan_path = Path(self.owner_config.work).resolve() / "CORPUS_PLAN.json"
        if plan_path.exists():
            payload = read_json(plan_path)
            digest = payload.pop("corpus_plan_sha256", None)
            if digest != sha256_bytes(canonical_json_bytes(payload)):
                raise ValueError("sealed corpus-plan digest mismatch")
            payload["corpus_plan_sha256"] = digest
            bindings = {
                "corpus_sha256": sha256_file(corpus),
                "carrier_inventory_sha256": self.carrier_inventory["inventory_sha256"],
                "runtime_inventory_sha256": self.runtime_inventory["inventory_sha256"],
                "dispatch_audit_sha256": self.dispatch_audit_sha256,
            }
            if payload.get("schema") != "r7-corpus-plan-v1" or any(
                payload.get(key) != value for key, value in bindings.items()
            ):
                raise ValueError("sealed corpus plan belongs to different inputs")
        else:
            payload = self._build_corpus_plan_payload(corpus)
            atomic_write_json(plan_path, payload)
        selected = payload.get("selected")
        if not isinstance(selected, list) or not selected:
            raise ValueError("sealed corpus plan has no prompts")
        expected: dict[str, int] = {}
        previous_end = 0
        for raw in selected:
            if not isinstance(raw, dict):
                raise ValueError("malformed corpus-plan prompt")
            shard_id = str(raw["shard_id"])
            input_ids = raw["input_ids"]
            tokens = int(raw["tokens"])
            if (
                shard_id in expected
                or not isinstance(input_ids, list)
                or len(input_ids) != tokens
                or tokens < MIN_SAMPLE_TOKENS
                or tokens > MAX_SAMPLE_TOKENS
                or any(type(value) is not int or value < 0 for value in input_ids)
                or type(raw.get("line_index")) is not int
                or int(raw["line_index"]) < 0
                or int(raw["global_row_start"]) != previous_end
            ):
                raise ValueError("corpus-plan prompt domain is not canonical")
            expected[shard_id] = tokens
            previous_end += tokens
        if previous_end < TARGET_TOKENS:
            raise ValueError("sealed corpus plan does not meet the token target")
        self._corpus_plan_payload = payload
        return {
            "corpus_plan_sha256": str(payload["corpus_plan_sha256"]),
            "corpus_plan_artifact_sha256": sha256_file(plan_path),
            "expected_shards": expected,
        }

    def _metadata(
        self,
        *,
        shard_id: str,
        input_ids: list[int],
        global_row_start: int,
        line_index: int,
    ) -> dict[str, object]:
        return {
            "schema": "r7-state-metadata-v2",
            "shard_id": shard_id,
            "tokens": len(input_ids),
            "global_row_start": global_row_start,
            "sequence_lengths": [len(input_ids)],
            "input_ids": input_ids,
            "corpus_line_index": line_index,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "dispatch_audit_sha256": self.dispatch_audit_sha256,
            "auxiliary": "prev_topk_indices-int32",
        }

    def _context(self, hidden, metadata: Mapping[str, object], device=None):
        device = self.device if device is None else device
        import torch

        tokens = int(metadata["tokens"])
        if (
            metadata.get("schema") != "r7-state-metadata-v2"
            or metadata.get("sequence_lengths") != [tokens]
            or metadata.get("attention_implementation") != ATTENTION_IMPLEMENTATION
        ):
            raise ValueError("state metadata is not the pinned one-prompt DSA schema")
        x = (
            torch.as_tensor(hidden, device=device)
            .reshape(1, tokens, HIDDEN_SIZE)
            .to(torch.bfloat16)
        )
        position_ids = torch.arange(tokens, device=device).unsqueeze(0)
        minimum = torch.finfo(x.dtype).min
        causal = torch.full(
            (tokens, tokens), minimum, device=device, dtype=x.dtype
        ).triu(diagonal=1)
        attention_mask = causal.unsqueeze(0).unsqueeze(0)
        position_embeddings = self.rotary(x, position_ids)
        previous = metadata.get("prev_topk_indices")
        if previous is not None:
            previous = torch.as_tensor(
                previous, device=device, dtype=torch.int32
            ).unsqueeze(0)
        return x, attention_mask, position_ids, position_embeddings, previous

    def _layer_state(self, layer: int, *, include_experts: bool):
        import torch

        if not 0 <= layer <= LAST_MOE_LAYER:
            raise ValueError("layer outside main-model streaming range")
        with torch.device("meta"):
            module = self.LayerClass(self.config, layer)
        values = {}
        expert_names = {
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        }
        for local_name in module.state_dict():
            if local_name in expert_names:
                continue
            full_name = f"model.layers.{layer}.{local_name}"
            if local_name.endswith(".weight"):
                values[local_name] = self.loader.weight(full_name[:-7]).to(self.device)
            else:
                values[local_name] = self.loader.tensor(full_name).to(self.device)
        if include_experts and layer >= FIRST_MOE_LAYER:
            installed = self._installed.get(layer)
            if installed is None or set(installed) != set(range(NUM_EXPERTS)):
                raise ValueError(f"layer {layer} lacks all installed Round 7 experts")
            gate_up = torch.empty(
                (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=self.device,
            )
            down = torch.empty(
                (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for expert in range(NUM_EXPERTS):
                gate_up[expert, :INTERMEDIATE_SIZE] = installed[expert][
                    "gate_proj"
                ].T.to(self.device)
                gate_up[expert, INTERMEDIATE_SIZE:] = installed[expert]["up_proj"].T.to(
                    self.device
                )
                down[expert] = installed[expert]["down_proj"].T.to(self.device)
            values["mlp.experts.gate_up_proj"] = gate_up
            values["mlp.experts.down_proj"] = down
        incompatible = module.load_state_dict(values, strict=False, assign=True)
        allowed_missing = set()
        if layer >= FIRST_MOE_LAYER and not include_experts:
            allowed_missing = expert_names
        if (
            set(incompatible.missing_keys) != allowed_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                f"carrier layer state mismatch L{layer}: "
                f"missing={incompatible.missing_keys} extra={incompatible.unexpected_keys}"
            )
        module.eval()
        return module

    def _advance(self, module, hidden, metadata, device=None):
        context = self._context(hidden, metadata, device=device)
        x, mask, positions, rotary, previous = context
        with __import__("torch").inference_mode():
            output, topk = module(
                x,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=rotary,
                prev_topk_indices=previous,
                use_cache=False,
            )
        if topk is None:
            raise ValueError("GLM DSA layer did not return cross-layer top-k state")
        return (
            output.reshape(-1, HIDDEN_SIZE).to(
                "cpu", dtype=__import__("torch").bfloat16
            ),
            topk.squeeze(0).to("cpu", dtype=__import__("torch").int32),
        )

    def _advance_repeat_oracle(self, module, hidden, metadata):
        first_hidden, first_topk = self._advance(module, hidden, metadata)
        second_hidden, second_topk = self._advance(module, hidden, metadata)
        if not __import__("torch").equal(first_hidden, second_hidden) or not __import__(
            "torch"
        ).equal(first_topk, second_topk):
            raise RuntimeError(
                "official GLM successor forward is not byte deterministic"
            )
        return (
            first_hidden,
            first_topk,
            {
                "schema": "r7-official-successor-repeat-v1",
                "hidden_bf16_sha256": _tensor_sha256(first_hidden),
                "prev_topk_i32_sha256": _tensor_sha256(first_topk),
                "dispatch_audit_sha256": self.dispatch_audit_sha256,
                "passed": True,
            },
        )

    def _write_state(
        self,
        *,
        output_partial: Path,
        layer_input: int,
        shard_id: str,
        hidden,
        previous_topk,
        metadata: Mapping[str, object],
        filename_prefix: str = "",
    ) -> tuple[str, Path, Path, int, int]:
        tokens = int(metadata["tokens"])
        hidden_path = output_partial / f"{filename_prefix}hidden-{shard_id}.safetensors"
        metadata_path = output_partial / f"{filename_prefix}metadata-{shard_id}.json"
        write_safetensors_atomic(
            hidden_path,
            (
                torch_tensor_entry("hidden", hidden),
                torch_tensor_entry("prev_topk_indices", previous_topk),
            ),
            metadata={
                "r7_schema": "r7-state-hidden-v2",
                "layer_input": str(layer_input),
                "shard_id": shard_id,
            },
        )
        payload = dict(metadata)
        payload.pop("prev_topk_indices", None)
        payload["prev_topk_shape"] = list(previous_topk.shape)
        atomic_write_json(metadata_path, payload)
        return shard_id, hidden_path, metadata_path, tokens, HIDDEN_SIZE

    def _open_prefix_progress(
        self, output_partial: Path, *, corpus_plan_sha256: str
    ) -> tuple[Path, dict[str, object]]:
        path = output_partial / "CARRIED_PREFIX_PROGRESS.json"
        selected = self._corpus_plan_payload.get("selected")
        if not isinstance(selected, list) or not selected:
            raise ValueError("carried-prefix progress lacks a prepared corpus plan")
        expected_shards = {str(raw["shard_id"]): int(raw["tokens"]) for raw in selected}
        bindings = {
            "corpus_plan_sha256": corpus_plan_sha256,
            "runtime_fingerprint": self.fingerprint,
            "carrier_inventory_sha256": self.carrier_inventory["inventory_sha256"],
            "dispatch_audit_sha256": self.dispatch_audit_sha256,
        }
        if path.exists():
            payload = read_json(path)
            if (
                payload.get("schema") != "r7-carried-prefix-progress-v1"
                or payload.get("bindings") != bindings
                or payload.get("expected_shards") != expected_shards
                or not isinstance(payload.get("records"), dict)
            ):
                raise ValueError("carried-prefix progress binding drift")
            for key, record in payload["records"].items():
                if not isinstance(record, dict):
                    raise ValueError("malformed carried-prefix prompt record")
                hidden = output_partial / str(record["hidden"])
                metadata = output_partial / str(record["metadata"])
                layer_input = int(record.get("layer_input", -1))
                shard_id = str(record.get("shard_id", ""))
                tokens = int(record.get("tokens", -1))
                if (
                    key != f"{layer_input:03d}:{shard_id}"
                    or layer_input not in range(1, FIRST_MOE_LAYER + 1)
                    or expected_shards.get(shard_id) != tokens
                    or not hidden.is_file()
                    or not metadata.is_file()
                    or sha256_file(hidden) != record["sha256_hidden"]
                    or sha256_file(metadata) != record["sha256_metadata"]
                ):
                    raise ValueError("carried-prefix prompt artifact drift")
                metadata_payload = read_json(metadata)
                plan_record = next(
                    raw for raw in selected if str(raw["shard_id"]) == shard_id
                )
                if (
                    metadata_payload.get("corpus_plan_sha256") != corpus_plan_sha256
                    or metadata_payload.get("shard_id") != shard_id
                    or int(metadata_payload.get("tokens", -1)) != tokens
                    or metadata_payload.get("input_ids") != plan_record["input_ids"]
                    or int(metadata_payload.get("global_row_start", -1))
                    != int(plan_record["global_row_start"])
                ):
                    raise ValueError("carried-prefix prompt differs from corpus plan")
        else:
            payload = {
                "schema": "r7-carried-prefix-progress-v1",
                "bindings": bindings,
                "expected_shards": expected_shards,
                "records": {},
            }
            atomic_write_json(path, payload)
        return path, payload

    @staticmethod
    def _prefix_record_tuple(
        output_partial: Path, record: Mapping[str, object]
    ) -> tuple[str, Path, Path, int, int]:
        return (
            str(record["shard_id"]),
            output_partial / str(record["hidden"]),
            output_partial / str(record["metadata"]),
            int(record["tokens"]),
            HIDDEN_SIZE,
        )

    def _seal_prefix_record(
        self,
        *,
        progress_path: Path,
        progress: dict[str, object],
        layer_input: int,
        record: tuple[str, Path, Path, int, int],
    ) -> None:
        shard_id, hidden, metadata, tokens, _ = record
        selected = self._corpus_plan_payload.get("selected")
        if not isinstance(selected, list):
            raise ValueError("carried-prefix seal lacks a corpus plan")
        plan_record = next(
            (raw for raw in selected if str(raw["shard_id"]) == shard_id), None
        )
        metadata_payload = read_json(metadata)
        if (
            plan_record is None
            or int(plan_record["tokens"]) != tokens
            or metadata_payload.get("corpus_plan_sha256")
            != self._corpus_plan_payload["corpus_plan_sha256"]
            or metadata_payload.get("input_ids") != plan_record["input_ids"]
        ):
            raise ValueError("carried-prefix seal differs from corpus plan")
        key = f"{layer_input:03d}:{shard_id}"
        item = {
            "layer_input": layer_input,
            "shard_id": shard_id,
            "hidden": hidden.name,
            "metadata": metadata.name,
            "tokens": tokens,
            "sha256_hidden": sha256_file(hidden),
            "sha256_metadata": sha256_file(metadata),
        }
        records = progress["records"]
        assert isinstance(records, dict)
        incumbent = records.get(key)
        if incumbent is not None and incumbent != item:
            raise ValueError("carried-prefix prompt rewrite drift")
        records[key] = item
        atomic_write_json(progress_path, progress)

    def _prefix_cleanup_checkpoint(self, boundary: str) -> None:
        """Expose every ordered cleanup boundary to synthetic fault injection."""
        hook = getattr(self, "_prefix_cleanup_fault_hook", None)
        if hook is not None:
            hook(boundary)

    def _cleanup_carried_prefix_derivatives(
        self,
        *,
        output_partial: Path,
        progress_path: Path,
        selected: Sequence[Mapping[str, object]],
    ) -> None:
        """Retire prefix-only files after the state partial owns every prompt.

        The state transition validates the complete sealed prompt domain before
        calling this path.  The progress journal and layer-1/2 artifacts are
        therefore derivative.  Removing the journal first prevents a restart
        from trying to validate an already partly collected derivative set.
        """
        self._prefix_cleanup_checkpoint("before-progress-unlink")
        progress_path.unlink(missing_ok=True)
        self._prefix_cleanup_checkpoint("after-progress-unlink")
        for layer_input in range(1, FIRST_MOE_LAYER):
            for raw in selected:
                shard_id = str(raw["shard_id"])
                for kind, suffix in (
                    ("hidden", "safetensors"),
                    ("metadata", "json"),
                ):
                    path = output_partial / (
                        f"prefix-input-{layer_input:03d}-{kind}-{shard_id}.{suffix}"
                    )
                    path.unlink(missing_ok=True)
                    self._prefix_cleanup_checkpoint(
                        f"after-unlink:{layer_input:03d}:{shard_id}:{kind}"
                    )

    def initialize_carried_state(
        self,
        *,
        carrier: Path,
        corpus: Path,
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        import torch
        import torch.nn.functional as functional

        if carrier.resolve() != self.owner_config.carrier.resolve():
            raise ValueError("runtime carrier differs from the sealed configuration")
        plan = self.prepare_corpus_plan(corpus=corpus)
        corpus_plan_sha256 = str(plan["corpus_plan_sha256"])
        selected = tuple(self._corpus_plan_payload["selected"])
        remaining = [
            raw for raw in selected if str(raw["shard_id"]) not in completed_shard_ids
        ]
        if not remaining:
            self._cleanup_carried_prefix_derivatives(
                output_partial=output_partial,
                progress_path=output_partial / "CARRIED_PREFIX_PROGRESS.json",
                selected=selected,
            )
            return
        progress_path, progress = self._open_prefix_progress(
            output_partial, corpus_plan_sha256=corpus_plan_sha256
        )
        progress_records = progress["records"]
        assert isinstance(progress_records, dict)
        for layer in range(FIRST_MOE_LAYER):
            embedding = (
                self.loader.weight("model.embed_tokens").to(self.device)
                if layer == 0
                else None
            )
            module = self._layer_state(layer, include_experts=False)
            try:
                for raw in remaining:
                    shard_id = str(raw["shard_id"])
                    target_layer = layer + 1
                    key = f"{target_layer:03d}:{shard_id}"
                    if key in progress_records:
                        continue
                    metadata = self._metadata(
                        shard_id=shard_id,
                        input_ids=[int(value) for value in raw["input_ids"]],
                        global_row_start=int(raw["global_row_start"]),
                        line_index=int(raw["line_index"]),
                    )
                    metadata["corpus_plan_sha256"] = corpus_plan_sha256
                    if layer == 0:
                        assert embedding is not None
                        hidden = functional.embedding(
                            torch.tensor(
                                metadata["input_ids"],
                                device=self.device,
                                dtype=torch.long,
                            ),
                            embedding,
                        ).to("cpu", dtype=torch.bfloat16)
                    else:
                        previous_key = f"{layer:03d}:{shard_id}"
                        previous_record = progress_records.get(previous_key)
                        if previous_record is None:
                            raise ValueError(
                                "carried-prefix predecessor prompt is missing"
                            )
                        previous_tuple = self._prefix_record_tuple(
                            output_partial, previous_record
                        )
                        previous_reader = SafeTensorReader(previous_tuple[1])
                        hidden = read_torch_tensor(previous_reader, "hidden")
                        metadata["prev_topk_indices"] = read_torch_tensor(
                            previous_reader, "prev_topk_indices"
                        )
                    state, topk = self._advance(module, hidden, metadata)
                    metadata.pop("prev_topk_indices", None)
                    metadata["producer"] = (
                        f"carrier-packed-decoded-layers-0-through-{layer}"
                    )
                    metadata["carrier_decode_audit_sha256"] = self.loader.audit_sha256
                    filename_prefix = (
                        ""
                        if target_layer == FIRST_MOE_LAYER
                        else (f"prefix-input-{target_layer:03d}-")
                    )
                    record = self._write_state(
                        output_partial=output_partial,
                        layer_input=target_layer,
                        shard_id=shard_id,
                        hidden=state,
                        previous_topk=topk,
                        metadata=metadata,
                        filename_prefix=filename_prefix,
                    )
                    self._seal_prefix_record(
                        progress_path=progress_path,
                        progress=progress,
                        layer_input=target_layer,
                        record=record,
                    )
            finally:
                del module
                if embedding is not None:
                    del embedding
                torch.cuda.empty_cache()
        for raw in selected:
            shard_id = str(raw["shard_id"])
            if shard_id in completed_shard_ids:
                continue
            final_record = progress_records.get(f"{FIRST_MOE_LAYER:03d}:{shard_id}")
            if final_record is None:
                raise ValueError("carried-prefix final prompt is missing")
            yield self._prefix_record_tuple(output_partial, final_record)
        # Generator exhaustion means the caller journaled every yielded final
        # prompt.  Retire only deterministic layer-1/2 derivatives; the final
        # layer-3 inputs are owned and hashed by the authoritative state partial.
        self._cleanup_carried_prefix_derivatives(
            output_partial=output_partial,
            progress_path=progress_path,
            selected=selected,
        )

    def begin_capture(self, *, layer: int) -> None:
        if self._capture_layer is not None:
            raise RuntimeError("a capture layer is already loaded")
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("capture outside layers 3..77")
        self._capture_module = self._layer_state(layer, include_experts=False)
        prefix = f"model.layers.{layer}."
        layer_records = {
            key: self.loader.audit_records[key]
            for key in sorted(self.loader.audit_records)
            if key.startswith(prefix)
        }
        self._capture_audits[layer] = {
            "schema": "r7-carried-layer-arithmetic-v1",
            "layer": layer,
            "runtime_fingerprint": self.fingerprint,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "dispatch_audit_sha256": self.dispatch_audit_sha256,
            "tensor_records_sha256": sha256_bytes(canonical_json_bytes(layer_records)),
            "tensor_count": len(layer_records),
            "passed": bool(layer_records),
        }
        self._capture_layer = layer

    def prepare_moe_input(
        self, *, layer: int, hidden: Any, attention_metadata: Mapping[str, object]
    ) -> Any:
        import torch

        if layer != self._capture_layer or self._capture_module is None:
            raise RuntimeError("prepare_moe_input called outside capture lifecycle")
        x, mask, positions, rotary, previous = self._context(hidden, attention_metadata)
        module = self._capture_module
        with torch.inference_mode():
            residual = x
            normalized = module.input_layernorm(x)
            attention, _, _ = module.self_attn(
                hidden_states=normalized,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=rotary,
                prev_topk_indices=previous,
                use_cache=False,
            )
            post_attention = residual + attention
            moe_hidden = module.post_attention_layernorm(post_attention)
        return moe_hidden.reshape(-1, HIDDEN_SIZE)

    def route_exact(
        self,
        *,
        layer: int,
        moe_hidden: Any,
        attention_metadata: Mapping[str, object],
    ) -> RoutedBatch:
        import torch

        if layer != self._capture_layer or self._capture_module is None:
            raise RuntimeError("route_exact called outside capture lifecycle")
        with torch.inference_mode():
            logits = self._capture_module.mlp.gate(
                torch.as_tensor(moe_hidden, device=self.device)
            )
            ids, weights = self._capture_module.mlp.route_tokens_to_experts(logits)
        return RoutedBatch(
            ids.detach().to("cpu", dtype=torch.int64).contiguous(),
            weights.detach().to("cpu", dtype=torch.float32).contiguous(),
            float(self.config.routed_scaling_factor),
        )

    def end_capture(self, *, layer: int) -> None:
        import torch

        if layer != self._capture_layer or self._capture_module is None:
            raise RuntimeError("capture lifecycle mismatch")
        del self._capture_module
        self._capture_module = None
        self._capture_layer = None
        torch.cuda.empty_cache()

    def capture_arithmetic_audit(self, *, layer: int) -> Mapping[str, object]:
        try:
            return self._capture_audits[layer]
        except KeyError as exc:
            raise ValueError("capture arithmetic audit is unavailable") from exc

    @staticmethod
    def _expert_forward(hidden, weights: Mapping[str, Any]):
        import torch
        import torch.nn.functional as functional

        source = torch.as_tensor(hidden)
        device = source.device
        x = source.to(dtype=torch.bfloat16)
        gate = torch.as_tensor(
            weights["gate_proj"], device=device, dtype=torch.bfloat16
        )
        up = torch.as_tensor(weights["up_proj"], device=device, dtype=torch.bfloat16)
        down = torch.as_tensor(
            weights["down_proj"], device=device, dtype=torch.bfloat16
        )
        # Match the official naive-MoE BF16 sequence: both FC1 outputs, SiLU,
        # multiply, and FC2 stay at activation dtype.  Introducing an FP32
        # nonlinear here would make the audit easier to pass but irrelevant to
        # the installed predecessor arithmetic.
        gate_out = torch.matmul(x, gate)
        up_out = torch.matmul(x, up)
        intermediate = functional.silu(gate_out) * up_out
        return torch.matmul(intermediate, down).to(torch.bfloat16)

    @staticmethod
    def _reference_routed_experts(hidden, topk_indices, topk_weights, installed):
        """Independent packed-decoded reference for the official eager MoE.

        The reference does not read the official module parameters or call its
        forward.  It traverses experts in ascending order, reproduces the
        official top-k-position/token ordering, and performs each BF16 output
        accumulation explicitly.  Duplicate experts within a token are
        rejected so every official ``index_add_`` has unique target indices.
        """

        import torch

        source = torch.as_tensor(hidden)
        indices = torch.as_tensor(topk_indices, device=source.device, dtype=torch.int64)
        weights = torch.as_tensor(
            topk_weights, device=source.device, dtype=torch.float32
        )
        if (
            source.ndim != 2
            or indices.ndim != 2
            or weights.shape != indices.shape
            or indices.shape[0] != source.shape[0]
        ):
            raise ValueError("installed-layer reference routing geometry drift")
        if any(len(set(row)) != len(row) for row in indices.detach().cpu().tolist()):
            raise ValueError(
                "installed-layer reference rejects duplicate token experts"
            )
        output = torch.zeros_like(source, dtype=torch.bfloat16)
        hit_experts = sorted(set(indices.detach().cpu().reshape(-1).tolist()))
        for expert in hit_experts:
            if expert not in installed:
                raise ValueError(f"installed-layer reference lacks expert {expert}")
            token_parts = []
            weight_parts = []
            for topk_position in range(indices.shape[1]):
                token_index = torch.nonzero(
                    indices[:, topk_position] == expert, as_tuple=False
                ).flatten()
                if token_index.numel():
                    token_parts.append(token_index)
                    weight_parts.append(weights[token_index, topk_position])
            token_index = torch.cat(token_parts)
            routed_weight = torch.cat(weight_parts)
            expert_output = TransformersSequentialRuntime._expert_forward(
                source[token_index], installed[expert]
            )
            contribution = (expert_output * routed_weight[:, None]).to(
                dtype=torch.bfloat16
            )
            # The official call has unique target indices for this expert, so
            # its index_add is one BF16 addition per token.  Spell that order
            # explicitly instead of sharing its implementation.
            for offset, token in enumerate(token_index.detach().cpu().tolist()):
                output[token] = (output[token] + contribution[offset]).to(
                    torch.bfloat16
                )
        return output

    def install_encoded_expert(
        self, *, layer: int, expert: int, encoded: Mapping[str, Any]
    ) -> Mapping[str, object]:
        import torch

        if (
            not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER
            or not 0 <= expert < NUM_EXPERTS
        ):
            raise ValueError("installed expert identity is outside Round 7")
        codec = self._get_codec()
        installed: dict[str, Any] = {}
        packed_hashes: dict[str, str] = {}
        reconstruction_hashes: dict[str, str] = {}
        installed_shapes: dict[str, list[int]] = {}
        for key, record in sorted(encoded.items()):
            projection = _projection_from_key(key)
            packed = torch.as_tensor(record["trellis"]).detach().contiguous().cpu()
            supplied_raw = record.get("reconstructed_kn")
            supplied = (
                None
                if supplied_raw is None
                else torch.as_tensor(supplied_raw)
                .detach()
                .half()
                .contiguous()
                .cpu()
            )
            if _tensor_sha256(packed) != record["packed_sha256"]:
                raise ValueError(f"installed packed payload hash drift: {key}")
            decoded = (
                codec.decode_to_original(
                    packed.to(codec.config.device),
                    record["suh"],
                    record["svh"],
                    int(record["bits"]),
                )
                .detach()
                .half()
                .contiguous()
                .cpu()
            )
            if _tensor_sha256(decoded) != record["reconstruction_sha256"]:
                raise ValueError(f"packed-decoded reconstruction drift: {key}")
            if supplied is not None and not torch.equal(decoded, supplied):
                raise ValueError(f"supplied reconstruction differs from decode: {key}")
            expected = (
                (HIDDEN_SIZE, INTERMEDIATE_SIZE)
                if projection != "down_proj"
                else (INTERMEDIATE_SIZE, HIDDEN_SIZE)
            )
            if tuple(decoded.shape) != expected:
                raise ValueError(f"installed reconstruction shape drift: {key}")
            packed_hashes[key] = str(record["packed_sha256"])
            reconstruction_hashes[key] = str(record["reconstruction_sha256"])
            installed[projection] = decoded.to(torch.bfloat16)
            installed_shapes[key] = list(decoded.shape)
        self._installed.setdefault(layer, {})[expert] = installed
        return {
            "schema": "r7-packed-install-record-v2",
            "layer": layer,
            "expert": expert,
            "runtime_fingerprint": self.fingerprint,
            "activation_dtype": "BF16",
            "packed_decoded": True,
            "packed_sha256": dict(sorted(packed_hashes.items())),
            "reconstruction_sha256": dict(sorted(reconstruction_hashes.items())),
            "installed_shape_kn": dict(sorted(installed_shapes.items())),
            "passed": True,
        }

    def audit_installed_layer(self, *, layer: int) -> Mapping[str, object]:
        import torch

        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("installed-layer audit is outside Round 7")
        installed = self._installed.get(layer)
        if installed is None or set(installed) != set(range(NUM_EXPERTS)):
            raise ValueError("installed-layer audit requires all 256 experts")

        seed = derive_seed(layer, "official-installed-layer-audit-v1")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed & ((1 << 63) - 1))
        sample = torch.randn(
            (INSTALL_AUDIT_ROWS, HIDDEN_SIZE),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=torch.bfloat16)
        # Eleven experts are hit repeatedly across 32 tokens, while every token
        # still has eight distinct experts.  This exercises aggregate dispatch
        # and BF16 accumulation without duplicate targets inside an index_add.
        rows = torch.arange(INSTALL_AUDIT_ROWS, dtype=torch.int64).unsqueeze(1)
        positions = torch.arange(TOP_K, dtype=torch.int64).unsqueeze(0)
        topk_indices = ((rows % 4) + positions).to(self.device)
        raw_weights = torch.rand(
            (INSTALL_AUDIT_ROWS, TOP_K), generator=generator, dtype=torch.float32
        )
        scaling = float(self.config.routed_scaling_factor)
        topk_weights = (
            raw_weights / raw_weights.sum(dim=1, keepdim=True) * scaling
        ).to(self.device)

        with torch.device("meta"):
            official = self.ExpertsClass(self.config)
        gate_up = torch.empty(
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=self.device,
        )
        down = torch.empty(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for expert in range(NUM_EXPERTS):
            gate_up[expert, :INTERMEDIATE_SIZE] = installed[expert]["gate_proj"].T.to(
                self.device
            )
            gate_up[expert, INTERMEDIATE_SIZE:] = installed[expert]["up_proj"].T.to(
                self.device
            )
            down[expert] = installed[expert]["down_proj"].T.to(self.device)
        incompatible = official.load_state_dict(
            {"gate_up_proj": gate_up, "down_proj": down}, strict=True, assign=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("official installed experts state mismatch")
        official.eval()
        official_module = f"{type(official).__module__}.{type(official).__qualname__}"
        try:
            with torch.inference_mode():
                first = official(sample, topk_indices, topk_weights).to(torch.bfloat16)
                second = official(sample, topk_indices, topk_weights).to(torch.bfloat16)
                reference = self._reference_routed_experts(
                    sample, topk_indices, topk_weights, installed
                )
        finally:
            del official
            torch.cuda.empty_cache()
        repeat_exact = torch.equal(first, second)
        delta = first.float() - reference.float()
        max_abs = float(delta.abs().max().item())
        relative = float(
            torch.linalg.vector_norm(delta).item()
            / max(torch.linalg.vector_norm(reference.float()).item(), 1e-30)
        )
        passed = (
            repeat_exact
            and math.isfinite(max_abs)
            and math.isfinite(relative)
            and max_abs <= INSTALL_MAX_ABS_ERROR
            and relative <= INSTALL_MAX_RELATIVE_L2
        )
        if not passed:
            raise RuntimeError(
                "official installed-layer forward failed repeat/reference audit"
            )
        return {
            "schema": "r7-official-installed-layer-audit-v1",
            "layer": layer,
            "runtime_fingerprint": self.fingerprint,
            "activation_dtype": "BF16",
            "sample_rows": INSTALL_AUDIT_ROWS,
            "sample_seed": seed,
            "sample_input_sha256": _tensor_sha256(sample),
            "top_k": TOP_K,
            "topk_indices_sha256": _tensor_sha256(topk_indices),
            "topk_weights_sha256": _tensor_sha256(topk_weights),
            "routed_scaling_factor": format(scaling, ".17g"),
            "unique_experts_per_token": True,
            "shared_expert_hits": True,
            "official_module": official_module,
            "experts_implementation": EXPERTS_IMPLEMENTATION,
            "dispatch_audit_sha256": self.dispatch_audit_sha256,
            "official_first_output_sha256": _tensor_sha256(first),
            "official_second_output_sha256": _tensor_sha256(second),
            "reference_output_sha256": _tensor_sha256(reference),
            "official_repeat_exact": repeat_exact,
            "max_abs_error": format(max_abs, ".17g"),
            "relative_l2_error": format(relative, ".17g"),
            "max_abs_tolerance": format(INSTALL_MAX_ABS_ERROR, ".17g"),
            "relative_l2_tolerance": format(INSTALL_MAX_RELATIVE_L2, ".17g"),
            "passed": True,
        }

    def restore_encoded_layer(self, *, layer: int, manifest: Path) -> None:
        codec = self._get_codec()
        audit_v2_layer(manifest, codec=codec, tp_sizes=(16,))
        raw = read_json(manifest)
        reader = SafeTensorReader(manifest.parent / str(raw["shard"]))
        restored: dict[int, dict[str, Any]] = {}
        for expert in range(NUM_EXPERTS):
            projections = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensor_id = TensorId(layer, expert, projection)
                prefix = tensor_id.hf_prefix
                refs = raw["vector_refs"][prefix]
                packed = read_torch_tensor(reader, tensor_name(tensor_id, "trellis"))
                reconstructed = (
                    codec.decode_to_original(
                        packed.to(codec.config.device),
                        read_torch_tensor(reader, refs["suh"]),
                        read_torch_tensor(reader, refs["svh"]),
                        int(raw["bit_map"][prefix]),
                    )
                    .half()
                    .cpu()
                )
                if (
                    _tensor_sha256(reconstructed)
                    != raw["roundtrip_hashes"][prefix]["reconstruction_sha256"]
                ):
                    raise ValueError(f"restored reconstruction drift: {tensor_id.key}")
                projections[projection] = reconstructed.to(__import__("torch").bfloat16)
            restored[expert] = projections
        self._installed[layer] = restored

    def forward_installed_layer(
        self,
        *,
        layer: int,
        input_shards: Iterable[StateShard],
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        import torch

        if not FIRST_MOE_LAYER <= layer < LAST_MOE_LAYER:
            raise ValueError("successor forwarding is restricted to layers 3..76")
        input_values = tuple(input_shards)
        if not input_values:
            raise ValueError("installed successor forward has no prompt state")
        if all(shard.shard_id in completed_shard_ids for shard in input_values):
            self._installed.pop(layer, None)
            return
        module = self._layer_state(layer, include_experts=True)
        prefix = f"model.layers.{layer}."
        layer_records = {
            key: self.loader.audit_records[key]
            for key in sorted(self.loader.audit_records)
            if key.startswith(prefix)
        }
        carrier_audit = sha256_bytes(canonical_json_bytes(layer_records))
        first_shard_id = input_values[0].shard_id
        devices = getattr(self, "forward_devices", None) or (
            getattr(self, "device", None),
        )
        pending = [
            shard
            for shard in input_values
            if shard.shard_id not in completed_shard_ids
        ]
        replicas: list = []
        try:
            def _load_shard(shard):
                reader = SafeTensorReader(shard.hidden_path)
                hidden = read_torch_tensor(reader, "hidden")
                previous = read_torch_tensor(reader, "prev_topk_indices")
                metadata = read_json(shard.metadata_path)
                metadata["prev_topk_indices"] = previous
                return hidden, metadata

            def _successor(shard, metadata, repeat_audit, parity_audit):
                successor = dict(metadata)
                successor.pop("prev_topk_indices", None)
                successor["producer"] = f"round7-installed-layer-{layer:03d}"
                successor["predecessor_hidden_sha256"] = shard.sha256_hidden
                successor["carrier_decode_audit_sha256"] = carrier_audit
                if repeat_audit is not None:
                    successor["official_repeat_audit"] = repeat_audit
                if parity_audit is not None:
                    successor["replica_parity_audit"] = parity_audit
                return successor

            sequential = (
                len(devices) <= 1
                or len(pending) <= 1
                # Mid-layer resume: the oracle prompt is already sealed, so
                # the parity check has no reference. Sequential handles it.
                or pending[0].shard_id != first_shard_id
            )
            if sequential:
                # Original sequential flow, byte-identical.
                for shard in pending:
                    hidden, metadata = _load_shard(shard)
                    if shard.shard_id == first_shard_id:
                        output, topk, repeat_audit = self._advance_repeat_oracle(
                            module, hidden, metadata
                        )
                    else:
                        output, topk = self._advance(module, hidden, metadata)
                        repeat_audit = None
                    # Yield immediately: the walk seals this prompt in
                    # PARTIAL.json before the next expensive prompt starts.
                    yield self._write_state(
                        output_partial=output_partial,
                        layer_input=layer + 1,
                        shard_id=shard.shard_id,
                        hidden=output,
                        previous_topk=topk,
                        metadata=_successor(shard, metadata, repeat_audit, None),
                    )
                return

            # Sharded successor forward across the device pool. Compute fans
            # out to worker replicas; every state write and yield stays on
            # THIS thread in canonical shard order, so PARTIAL.json sealing,
            # resume semantics, and state hashing are unchanged.
            import copy
            from concurrent.futures import ThreadPoolExecutor

            first_shard = pending[0]
            first_hidden, first_metadata = _load_shard(first_shard)
            output, topk, repeat_audit = self._advance_repeat_oracle(
                module, first_hidden, first_metadata
            )
            # Replicate only after the primary oracle passes, and prove every
            # replica reproduces the primary bytes before trusting it with
            # calibration state.
            replica_parity = {
                "schema": "r7-replica-parity-v1",
                "devices": [str(item) for item in devices],
            }
            modules = [module]
            for extra in devices[1:]:
                replica = copy.deepcopy(module).to(extra)
                r_out, r_topk = self._advance(
                    replica, first_hidden, first_metadata, device=extra
                )
                if not torch.equal(r_out, output) or not torch.equal(
                    r_topk, topk
                ):
                    raise RuntimeError(
                        f"replica on {extra} diverged from the primary "
                        "successor forward; refusing sharded execution"
                    )
                replicas.append(replica)
                modules.append(replica)
            replica_parity["passed"] = True
            yield self._write_state(
                output_partial=output_partial,
                layer_input=layer + 1,
                shard_id=first_shard.shard_id,
                hidden=output,
                previous_topk=topk,
                metadata=_successor(
                    first_shard, first_metadata, repeat_audit, replica_parity
                ),
            )

            def _compute(worker, shard):
                hidden, metadata = _load_shard(shard)
                out, tk = self._advance(
                    modules[worker], hidden, metadata, device=devices[worker]
                )
                return shard, metadata, out, tk

            # A shared N-thread executor does not bind a Python worker thread to
            # its modulo-selected module: when one device finishes early, that
            # thread may pick up a later task assigned to a module which is still
            # active on another thread. Give every module/device one serial queue
            # instead. Futures are consumed in canonical shard order, preserving
            # state publication and resume semantics.
            executors = []
            futures = []
            try:
                # Construct inside the cleanup scope so a partial allocation
                # cannot leak live threads if a later executor creation fails.
                for _ in modules:
                    executors.append(ThreadPoolExecutor(max_workers=1))
                for index, shard in enumerate(pending[1:]):
                    worker = index % len(modules)
                    futures.append(executors[worker].submit(_compute, worker, shard))
                for future in futures:
                    shard, metadata, out, tk = future.result()
                    yield self._write_state(
                        output_partial=output_partial,
                        layer_input=layer + 1,
                        shard_id=shard.shard_id,
                        hidden=out,
                        previous_topk=tk,
                        metadata=_successor(shard, metadata, None, None),
                    )
            finally:
                for future in futures:
                    future.cancel()
                for executor in executors:
                    executor.shutdown(wait=True, cancel_futures=True)
        finally:
            del module
            for replica in replicas:
                del replica
            replicas.clear()
            self._installed.pop(layer, None)
            torch.cuda.empty_cache()


def factory(config) -> TransformersSequentialRuntime:
    """CLI factory: ``--runtime r7_encoder.transformers_runtime:factory``."""

    return TransformersSequentialRuntime(config)
