"""Offline GLM-5.3 packed K4/K6 reader used for logit/KLD measurement.

The reader independently unpacks native EXL3 K4 trellis words, evaluates the
frozen MCG codebook, applies the persisted FP16 transforms, and installs BF16
reconstructions into the local EP4 expert parameters.  It is intentionally not
the final TP2 packed serving kernel.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import cached_property, lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from ..campaign.glm53_direct_k4 import (
    EXPERT_RECEIPT_SCHEMA,
    LAYER_RECEIPT_SCHEMA,
    MAIN_ROUTED_LAYERS,
    MTP_LAYER,
    NUM_EXPERTS,
    PROJECTIONS,
    contract_bits,
    projection_shape,
    tensor_name,
    verify_contract,
)
from ..campaign.glm53_uniform_k4 import MTP_ADAPTER_RECEIPT_SCHEMA
from ..checkpoint.exact_payload import (
    OBJECT_SCHEMA,
    PACKED_HASH_SCHEMA,
    packed_payload_sha256,
    tensor_sha256,
)
from ..checkpoint.packed_payload import (
    CHECKPOINT_HASH_SCHEMA,
    MCG_MARKER_SIGNED_INT32,
    MCG_MULTIPLIER,
    PACKED_CHOICE_SCHEMA,
    RECONSTRUCTION_CLOSURE_SCHEMA,
    PackedMCGPayloadStore,
    checkpoint_payload_sha256,
)
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file


BITS = 4
SUPPORTED_BITS = (4, 6)
EP_SIZE = 4
EXPERTS_PER_RANK = NUM_EXPERTS // EP_SIZE
MTP_PACKED_LAYER_RECEIPT_SCHEMA = "quant-pipeline.glm53-direct-mcg-mtp-packed-layer-receipt.v1"
READER_BACKEND_SCHEMA = "quant-pipeline.glm53-packed-k4-offline-reader-backend.v1"
MCG_MULT = MCG_MULTIPLIER
MCG_MASK = np.uint32(0x8FFF8FFF)
MCG_XOR = np.uint32(0x3B603B60)
_HASH = re.compile(r"[0-9a-f]{64}")
_LAYER_RECEIPT = re.compile(r"layer-(\d{3})\.json")


def _verify_seal(value: Mapping[str, Any], schema: str, field: str) -> str:
    if value.get("schema") != schema:
        raise ValueError(f"expected sealed {schema}")
    digest = value.get(field)
    if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise ValueError(f"{schema}.{field} is not SHA-256")
    body = copy.deepcopy(dict(value))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        raise ValueError(f"{schema} seal differs")
    return digest


def expert_range(rank: int, world_size: int = EP_SIZE) -> tuple[int, int]:
    if world_size != EP_SIZE or isinstance(rank, bool) or rank not in range(world_size):
        raise ValueError("packed K4 student reader requires exactly four EP ranks")
    start = rank * EXPERTS_PER_RANK
    return start, start + EXPERTS_PER_RANK


def verify_choice_descriptor(
    choice: Mapping[str, Any], *, layer: int, expert: int, projection: str, bits: int = BITS
) -> dict[str, Any]:
    if bits not in SUPPORTED_BITS:
        raise ValueError("packed reader supports only uniform K4/K6")
    row = dict(choice)
    _verify_seal(row, PACKED_CHOICE_SCHEMA, "choice_sha256")
    expected_shape = projection_shape(projection)
    output_features, input_features = expected_shape
    objects = row.get("objects")
    if (
        row.get("layer") != layer
        or row.get("expert") != expert
        or row.get("projection") != projection
        or row.get("bits") != bits
        or row.get("packed_hash_schema") != PACKED_HASH_SCHEMA
        or row.get("param_count") != output_features * input_features
        or not isinstance(objects, Mapping)
        or set(objects) != {"trellis", "suh", "svh", "mcg"}
    ):
        raise ValueError(f"packed K{bits} MCG choice binding differs: L{layer} E{expert} {projection}")
    expected = {
        "trellis": ("int16", [input_features // 16, output_features // 16, bits * 16]),
        "suh": ("float16", [input_features]),
        "svh": ("float16", [output_features]),
        "mcg": ("int32", None),
    }
    for name, (dtype, shape) in expected.items():
        ref = objects[name]
        digest = str(ref.get("sha256", "")) if isinstance(ref, Mapping) else ""
        if (
            not isinstance(ref, Mapping)
            or ref.get("schema") != OBJECT_SCHEMA
            or ref.get("dtype") != dtype
            or (name != "mcg" and ref.get("shape") != shape)
            or (name == "mcg" and ref.get("shape") not in ([], [1]))
            or not isinstance(ref.get("bytes"), int)
            or ref.get("bytes") <= 0
            or _HASH.fullmatch(digest) is None
            or ref.get("path") != f"objects/{digest[:2]}/{digest}.bin"
        ):
            raise ValueError(f"packed object geometry differs: {name} for L{layer} E{expert} {projection}")
    closure = row.get("reconstruction_closure")
    decoder = row.get("decoder")
    if (
        _HASH.fullmatch(str(row.get("packed_sha256", ""))) is None
        or row.get("packed_hash_schema") != PACKED_HASH_SCHEMA
        or row.get("checkpoint_hash_schema") != CHECKPOINT_HASH_SCHEMA
        or _HASH.fullmatch(str(row.get("checkpoint_payload_sha256", ""))) is None
        or row.get("logical_payload_bytes")
        != sum(int(objects[name]["bytes"]) for name in ("trellis", "suh", "svh", "mcg"))
        or not isinstance(closure, Mapping)
        or closure.get("schema") != RECONSTRUCTION_CLOSURE_SCHEMA
        or closure.get("dtype") != "float16"
        or closure.get("shape") != [output_features, input_features]
        or closure.get("orientation") != "huggingface_out_in"
        or closure.get("persisted") is not False
        or closure.get("encoder_full_decode_closure") is not True
        or _HASH.fullmatch(str(closure.get("payload_sha256", ""))) is None
        or not isinstance(decoder, Mapping)
        or decoder.get("codec_family") != "exl3-mcg"
        or decoder.get("mcg_multiplier_hex") != "0xCBAC1FED"
        or decoder.get("mcg_marker_signed_int32") != MCG_MARKER_SIGNED_INT32
        or _HASH.fullmatch(str(decoder.get("reader_abi_sha256", ""))) is None
    ):
        raise ValueError("choice packed/decoder/reconstruction closure differs")
    return row


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("example_only") is True:
        raise ValueError(f"artifact is not an executable receipt: {path}")
    return value


@dataclass(frozen=True)
class PackedK4Surface:
    root: Path
    contract_sha256: str
    mtp_adapter_receipt_sha256: str
    mtp_pack_receipt_sha256: str
    packed_reader_abi_sha256: str
    choices: Mapping[tuple[int, int, str], Mapping[str, Any]]
    main_layer_receipt_sha256: tuple[str, ...]
    bits: int = BITS

    @cached_property
    def store(self) -> PackedMCGPayloadStore:
        return PackedMCGPayloadStore(self.root / "payload-store")

    def choice(self, layer: int, expert: int, projection: str) -> Mapping[str, Any]:
        try:
            return self.choices[(layer, expert, projection)]
        except KeyError as error:
            raise ValueError(f"packed surface omits L{layer} E{expert} {projection}") from error


def _expert_receipt(
    root: Path, *, contract_sha256: str, layer: int, expert: int, bits: int = BITS
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / "experts" / f"layer-{layer:03d}" / f"expert-{expert:03d}.json"
    receipt = _load_json(path)
    _verify_seal(receipt, EXPERT_RECEIPT_SCHEMA, "receipt_sha256")
    choices = receipt.get("choices")
    if (
        receipt.get("contract_sha256") != contract_sha256
        or receipt.get("layer") != layer
        or receipt.get("expert") != expert
        or receipt.get("bits") != bits
        or receipt.get("projections") != list(PROJECTIONS)
        or receipt.get("candidate_rate_grid") is not False
        or receipt.get("global_allocator") is not False
        or not isinstance(choices, Mapping)
        or set(choices) != set(PROJECTIONS)
    ):
        raise ValueError(f"packed expert receipt is incomplete: L{layer} E{expert}")
    verified = {
        projection: verify_choice_descriptor(
            choices[projection], layer=layer, expert=expert, projection=projection, bits=bits
        )
        for projection in PROJECTIONS
    }
    for choice in verified.values():
        choice_path = root / "payload-store/choices" / f"{choice['choice_sha256']}.json"
        if _load_json(choice_path) != choice:
            raise ValueError(f"content-addressed choice file differs: {choice_path}")
        for ref in choice["objects"].values():
            object_path = root / "payload-store" / ref["path"]
            if (
                not object_path.is_file()
                or object_path.is_symlink()
                or object_path.stat().st_size != ref["bytes"]
            ):
                raise ValueError(f"packed object is absent or has wrong size: {object_path}")
    return receipt, verified


def _layer_receipt(
    root: Path,
    *,
    contract_sha256: str,
    layer: int,
    schema: str,
    bits: int = BITS,
) -> tuple[dict[str, Any], dict[tuple[int, int, str], Mapping[str, Any]]]:
    receipt = _load_json(root / "layers" / f"layer-{layer:03d}.json")
    _verify_seal(receipt, schema, "receipt_sha256")
    if (
        receipt.get("contract_sha256") != contract_sha256
        or receipt.get("layer") != layer
        or receipt.get("experts") != NUM_EXPERTS
        or receipt.get("matrix_count") != NUM_EXPERTS * len(PROJECTIONS)
        or receipt.get("bits") != bits
        or receipt.get("complete") is not True
        or not isinstance(receipt.get("expert_receipt_sha256"), list)
        or not isinstance(receipt.get("choice_sha256"), list)
    ):
        raise ValueError(f"packed layer receipt is incomplete: layer {layer}")
    expert_hashes: list[str] = []
    choice_hashes: list[str] = []
    choices: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for expert in range(NUM_EXPERTS):
        expert_receipt, expert_choices = _expert_receipt(
            root, contract_sha256=contract_sha256, layer=layer, expert=expert, bits=bits
        )
        expert_hashes.append(expert_receipt["receipt_sha256"])
        for projection in PROJECTIONS:
            choice = expert_choices[projection]
            choice_hashes.append(choice["choice_sha256"])
            choices[(layer, expert, projection)] = choice
    if receipt["expert_receipt_sha256"] != expert_hashes or receipt["choice_sha256"] != choice_hashes:
        raise ValueError(f"packed layer receipt hash census differs: layer {layer}")
    return receipt, choices


def build_mtp_packed_layer_receipt(
    *, root: str | Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the MTP packed-payload census consumed by its adapter receipt."""

    root = Path(root).resolve()
    contract_sha256 = verify_contract(contract)
    bits = contract_bits(contract)
    expert_hashes: list[str] = []
    choice_hashes: list[str] = []
    for expert in range(NUM_EXPERTS):
        receipt, choices = _expert_receipt(
            root, contract_sha256=contract_sha256, layer=MTP_LAYER, expert=expert, bits=bits
        )
        expert_hashes.append(receipt["receipt_sha256"])
        choice_hashes.extend(choices[projection]["choice_sha256"] for projection in PROJECTIONS)
    body = {
        "schema": MTP_PACKED_LAYER_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "layer": MTP_LAYER,
        "experts": NUM_EXPERTS,
        "matrix_count": NUM_EXPERTS * len(PROJECTIONS),
        "bits": bits,
        "expert_receipt_sha256": expert_hashes,
        "choice_sha256": choice_hashes,
        "complete": True,
    }
    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    return body


def load_complete_surface(
    *,
    root: str | Path,
    contract: Mapping[str, Any],
    mtp_adapter_receipt: Mapping[str, Any],
) -> PackedK4Surface:
    root = Path(root).resolve()
    if (
        not (root / "payload-store/objects").is_dir()
        or not (root / "payload-store/choices").is_dir()
    ):
        raise ValueError("packed root lacks the exact-codec payload store")
    contract_sha256 = verify_contract(contract)
    bits = contract_bits(contract)
    mtp_schema = (
        MTP_ADAPTER_RECEIPT_SCHEMA
        if bits == 4
        else "quant-pipeline.glm53-uniform-k6-mtp-adapter-receipt.v1"
    )
    _verify_seal(mtp_adapter_receipt, mtp_schema, "receipt_sha256")
    if (
        mtp_adapter_receipt.get("layer") != MTP_LAYER
        or mtp_adapter_receipt.get("launch_plan_sha256") != contract.get("launch_plan_sha256")
        or mtp_adapter_receipt.get("inventory_sha256") != contract.get("inventory_sha256")
        or mtp_adapter_receipt.get("expert_count") != NUM_EXPERTS
        or mtp_adapter_receipt.get("matrix_count") != NUM_EXPERTS * len(PROJECTIONS)
        or mtp_adapter_receipt.get("bits") != bits
        or mtp_adapter_receipt.get("qualified") is not True
    ):
        raise ValueError("MTP packed adapter receipt is incomplete")
    all_choices: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    layer_hashes: list[str] = []
    for layer in MAIN_ROUTED_LAYERS:
        receipt, choices = _layer_receipt(
            root, contract_sha256=contract_sha256, layer=layer, schema=LAYER_RECEIPT_SCHEMA, bits=bits
        )
        layer_hashes.append(receipt["receipt_sha256"])
        all_choices.update(choices)
    mtp_receipt, mtp_choices = _layer_receipt(
        root,
        contract_sha256=contract_sha256,
        layer=MTP_LAYER,
        schema=MTP_PACKED_LAYER_RECEIPT_SCHEMA,
        bits=bits,
    )
    if mtp_adapter_receipt.get("packed_payload_receipt_sha256") != mtp_receipt["receipt_sha256"]:
        raise ValueError("MTP adapter does not bind the complete packed MTP layer receipt")
    all_choices.update(mtp_choices)
    expected = (len(MAIN_ROUTED_LAYERS) + 1) * NUM_EXPERTS * len(PROJECTIONS)
    if len(all_choices) != expected:
        raise ValueError("packed main plus MTP choice census is incomplete")
    reader_abis = {str(choice["decoder"]["reader_abi_sha256"]) for choice in all_choices.values()}
    if len(reader_abis) != 1:
        raise ValueError("packed choices do not share one sealed MCG reader ABI")
    return PackedK4Surface(
        root=root,
        contract_sha256=contract_sha256,
        mtp_adapter_receipt_sha256=str(mtp_adapter_receipt["receipt_sha256"]),
        mtp_pack_receipt_sha256=str(mtp_receipt["receipt_sha256"]),
        packed_reader_abi_sha256=reader_abis.pop(),
        choices=all_choices,
        main_layer_receipt_sha256=tuple(layer_hashes),
        bits=bits,
    )


def unpack_trellis_states(packed, bits: int = BITS):
    import torch

    if bits not in SUPPORTED_BITS or packed.dtype != torch.int16 or packed.ndim != 3 or packed.shape[-1] != bits * 16:
        raise ValueError("reader accepts native uniform K4/K6 int16 trellis words only")
    tiles = packed.reshape(-1, bits * 16).to(torch.int64) & 0xFFFF
    tiles = tiles.reshape(tiles.shape[0], -1, 2).flip(-1).reshape(tiles.shape)
    word_shifts = torch.arange(15, -1, -1, device=packed.device)
    bitstream = ((tiles.reshape(-1, 16, bits)[..., None] >> word_shifts) & 1).reshape(
        -1, 16, bits * 16
    )
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=packed.device)
    edges = (bitstream.reshape(-1, 16, 16, bits) << symbol_shifts).sum(dim=-1)
    edges = edges.reshape(-1, 256)
    states = torch.zeros_like(edges)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(edges, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(torch.int16).reshape(*packed.shape[:-1], 256).contiguous()


@lru_cache(maxsize=8)
def mcg_lut(device="cpu"):
    import torch

    indices = np.arange(1 << 16, dtype=np.uint64)
    products = ((indices * np.uint64(MCG_MULT)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    products = ((products & MCG_MASK) ^ MCG_XOR).astype(np.uint32)
    low = (products & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = ((products >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    values = (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)
    if not np.isfinite(values).all():
        raise RuntimeError("frozen MCG lookup table is non-finite")
    return torch.from_numpy(np.ascontiguousarray(values)).to(device)


@lru_cache(maxsize=8)
def _permutation(device):
    import torch

    values = [0] * 256
    for thread in range(32):
        rows = ((thread % 4) * 2, (thread % 4) * 2 + 1, (thread % 4) * 2 + 8, (thread % 4) * 2 + 9)
        columns = (thread // 4, thread // 4 + 8)
        for offset, (row, column) in enumerate(
            ((row, column) for column in columns for row in rows)
        ):
            values[thread * 8 + offset] = row * 16 + column
    return torch.tensor(values, dtype=torch.long, device=device)


@lru_cache(maxsize=16)
def _hadamard(device, dtype):
    import torch

    value = torch.ones((1, 1), dtype=dtype, device=device)
    while value.shape[0] < 128:
        value = torch.cat((torch.cat((value, value), 1), torch.cat((value, -value), 1)), 0)
    return value * (1.0 / math.sqrt(128))


def decode_choice_hf(trellis, suh, svh, *, bits: int = BITS):
    """Decode stored payload to official/HF orientation ``[out_features,in_features]``."""

    import torch

    states = unpack_trellis_states(trellis, bits=bits)
    indices = (states.to(torch.int64) & 0xFFFF).long()
    values = mcg_lut(states.device).index_select(0, indices.flatten()).reshape_as(states).float()
    values = values.index_select(-1, torch.argsort(_permutation(states.device)))
    k_tiles, n_tiles, _ = values.shape
    exl = values.reshape(k_tiles, n_tiles, 16, 16).permute(0, 2, 1, 3).reshape(k_tiles * 16, n_tiles * 16)
    had = _hadamard(exl.device, exl.dtype)
    exl = torch.matmul(had, exl.reshape(-1, 128, exl.shape[1])).reshape_as(exl)
    exl *= suh.to(device=exl.device, dtype=exl.dtype).reshape(-1, 1)
    exl = torch.matmul(exl.reshape(exl.shape[0], -1, 128), had).reshape_as(exl)
    exl *= svh.to(device=exl.device, dtype=exl.dtype).reshape(1, -1)
    return exl.T.contiguous()


def load_decoded_choice(
    surface: PackedK4Surface, *, layer: int, expert: int, projection: str, device
):
    choice = surface.choice(layer, expert, projection)
    store = surface.store
    payload_cpu = {
        name: store.objects.load_tensor(choice["objects"][name])
        for name in ("trellis", "suh", "svh", "mcg")
    }
    if (
        int(payload_cpu["mcg"].reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32
        or packed_payload_sha256({name: payload_cpu[name] for name in ("trellis", "suh", "svh")})
        != choice["packed_sha256"]
        or checkpoint_payload_sha256(payload_cpu) != choice["checkpoint_payload_sha256"]
    ):
        raise ValueError(f"packed payload hash differs: {tensor_name(layer, expert, projection)}")
    payload = {name: value.to(device) for name, value in payload_cpu.items() if name != "mcg"}
    decoded = decode_choice_hf(
        payload["trellis"], payload["suh"], payload["svh"], bits=surface.bits
    )
    if tuple(decoded.shape) != projection_shape(projection):
        raise ValueError("decoded projection orientation/shape differs from official HF tensor")
    return decoded, choice


def stored_encoder_closure(
    surface: PackedK4Surface, *, layer: int, expert: int, projection: str, device="cpu"
) -> dict[str, Any]:
    import torch

    decoded, choice = load_decoded_choice(
        surface, layer=layer, expert=expert, projection=projection, device=device
    )
    decoded_f16 = decoded.to(torch.float16)
    observed_sha256 = tensor_sha256(decoded_f16)
    expected_sha256 = choice["reconstruction_closure"]["payload_sha256"]
    exact = observed_sha256 == expected_sha256
    result = {
        "tensor_name": tensor_name(layer, expert, projection),
        "choice_sha256": choice["choice_sha256"],
        "packed_sha256": choice["packed_sha256"],
        "encoder_reconstruction_payload_sha256": expected_sha256,
        "reader_reconstruction_payload_sha256": observed_sha256,
        "exact_fp16_payload_sha256_match": exact,
        "reconstruction_persisted": False,
    }
    if not exact:
        raise ValueError("independent packed decode differs from encoder FP16 reconstruction closure")
    return result


def fuse_gate_up(gate, up):
    import torch

    if tuple(gate.shape) != projection_shape("gate_proj") or tuple(up.shape) != projection_shape("up_proj"):
        raise ValueError("gate/up projection uses unexpected official orientation")
    return torch.cat((gate, up), dim=0).contiguous()


def _local_parameter(parameter):
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def resolve_main_layers(model):
    try:
        layers = model.model.language_model.layers
    except AttributeError as error:
        raise ValueError("model does not expose official model.language_model.layers ABI") from error
    if len(layers) != 45:
        raise ValueError("standard GLM53 student must expose exactly 45 main layers")
    return layers


def install_local_main_experts(model, surface: PackedK4Surface, *, rank: int, device) -> dict[str, Any]:
    """Install only main routed experts; MTP is receipt-gated but not executed."""

    import torch

    layers = resolve_main_layers(model)
    start, stop = expert_range(rank)
    installed: list[dict[str, Any]] = []
    payload_bytes = 0
    with torch.inference_mode():
        for layer_index in MAIN_ROUTED_LAYERS:
            experts = layers[layer_index].mlp.experts
            gate_up_target = _local_parameter(experts.gate_up_proj)
            down_target = _local_parameter(experts.down_proj)
            if tuple(gate_up_target.shape) != (EXPERTS_PER_RANK, 4096, 4096):
                raise ValueError(f"EP4 fused gate_up local layout differs at layer {layer_index}")
            if tuple(down_target.shape) != (EXPERTS_PER_RANK, 4096, 2048):
                raise ValueError(f"EP4 down local layout differs at layer {layer_index}")
            for global_expert in range(start, stop):
                local_expert = global_expert - start
                gate, gate_choice = load_decoded_choice(
                    surface, layer=layer_index, expert=global_expert, projection="gate_proj", device=device
                )
                up, up_choice = load_decoded_choice(
                    surface, layer=layer_index, expert=global_expert, projection="up_proj", device=device
                )
                down, down_choice = load_decoded_choice(
                    surface, layer=layer_index, expert=global_expert, projection="down_proj", device=device
                )
                gate_up = fuse_gate_up(gate, up)
                gate_up_bf16 = gate_up.to(dtype=torch.bfloat16)
                down_bf16 = down.to(dtype=torch.bfloat16)
                gate_up_target[local_expert].copy_(gate_up_bf16)
                down_target[local_expert].copy_(down_bf16)
                if not torch.equal(gate_up_target[local_expert], gate_up_bf16) or not torch.equal(
                    down_target[local_expert], down_bf16
                ):
                    raise RuntimeError("BF16 local expert installation did not close exactly")
                choices = (gate_choice, up_choice, down_choice)
                payload_bytes += sum(int(row["logical_payload_bytes"]) for row in choices)
                installed.append(
                    {
                        "layer": layer_index,
                        "global_expert": global_expert,
                        "local_expert": local_expert,
                        "choice_sha256": [row["choice_sha256"] for row in choices],
                        "packed_sha256": [row["packed_sha256"] for row in choices],
                    }
                )
                del gate, up, down, gate_up, gate_up_bf16, down_bf16
    return {
        "rank": rank,
        "global_expert_start": start,
        "global_expert_stop": stop,
        "main_layers": list(MAIN_ROUTED_LAYERS),
        "installed_expert_triplets": len(installed),
        "installed_matrix_count": len(installed) * len(PROJECTIONS),
        "verified_packed_payload_bytes": payload_bytes,
        "installed_choice_census_sha256": sha256_bytes(canonical_json(installed)),
        "mutated_parameter_suffixes": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        "nonrouted_parameters_mutated": False,
        "mtp_parameters_mutated": False,
    }


def reader_identity(
    module_path: str | Path, runner_path: str | Path, *, bits: int = BITS
) -> dict[str, Any]:
    if bits not in SUPPORTED_BITS:
        raise ValueError("reader identity supports only uniform K4/K6")
    body = {
        "schema": f"quant-pipeline.glm53-packed-k{bits}-offline-reader-identity.v1",
        "mode": "offline_packed_payload_decode_to_bf16_ep4_for_logit_measurement",
        "serving_kernel": False,
        "final_tp2_kernel": False,
        "bits": bits,
        "codebook": "MCG",
        "mcg_multiplier_hex": "0xCBAC1FED",
        "module_sha256": sha256_file(Path(module_path)),
        "runner_sha256": sha256_file(Path(runner_path)),
    }
    body["runtime_reader_sha256"] = sha256_bytes(canonical_json(body))
    return body
