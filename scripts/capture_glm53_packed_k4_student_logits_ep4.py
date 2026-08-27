#!/usr/bin/env python3
"""Capture final-panel logits from independently decoded packed K4/K6 experts.

Launch with ``torchrun --nproc-per-node 4``.  The official BF16 checkpoint is
loaded with Transformers EP4, then only each rank's local main-model routed
expert parameters are replaced from hash-verified EXL3/MCG K4 payloads.  The
MTP K4 surface must be complete and separately receipt-qualified even though
the standard causal-logit pass does not execute MTP.  This is an offline KLD
reader, not the final TP2 packed serving kernel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from quant_pipeline.core.artifacts import (
    canonical_json,
    prepare_empty_destination,
    sha256_bytes,
    sha256_file,
    write_json,
)
from quant_pipeline.campaign.glm53_direct_k4 import (
    contract_bits,
    contract_schema_for_bits,
)
from quant_pipeline.evaluation.glm53_logits import load_panel_windows
from quant_pipeline.evaluation.glm53_packed_k4_reader import (
    MAIN_ROUTED_LAYERS,
    install_local_main_experts,
    load_complete_surface,
    reader_identity,
    stored_encoder_closure,
)


RELEASED_ARCHITECTURE = "Glm5NextForConditionalGeneration"
RELEASED_MODEL_TYPE = "glm5_next"
RELEASED_TEXT_MODEL_TYPE = "glm5_next_text"
REVISION = re.compile(r"[0-9a-f]{40}")


def _sealed_json(path: Path, schema: str, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    seal = body.pop(field, None)
    if (
        not isinstance(value, dict)
        or value.get("example_only") is True
        or value.get("schema") != schema
        or seal != sha256_bytes(canonical_json(body))
    ):
        raise ValueError(f"invalid sealed {schema}: {path}")
    return value


def _inventory(path: Path, revision: str) -> dict[str, Any]:
    value = _sealed_json(path, "quant-pipeline.glm-release-inventory.v1", "inventory_sha256")
    if (
        value.get("model_revision") != revision
        or value.get("seal_mode") != "full-shard-sha256"
        or not value.get("shard_sha256")
        or any(re.fullmatch(r"[0-9a-f]{64}", str(item)) is None for item in value["shard_sha256"].values())
    ):
        raise ValueError("student capture requires the exact full-hash BF16 inventory")
    return value


def _distributed_environment() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("launch with torchrun --nproc-per-node 4") from error
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise RuntimeError("packed student capture is pinned to exactly four EP ranks")
    return rank, local_rank, world_size


def _tensor_device_type(value: Any) -> str:
    local = value.to_local() if hasattr(value, "to_local") else value
    return str(local.device.type)


def _is_mutated_routed_parameter(name: str) -> bool:
    return any(
        f"language_model.layers.{layer}.mlp.experts.{suffix}" in name
        for layer in MAIN_ROUTED_LAYERS
        for suffix in ("gate_up_proj", "down_proj")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--direct-contract", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--mtp-adapter-receipt", type=Path, required=True)
    parser.add_argument("--token-panel-receipt", type=Path, required=True)
    parser.add_argument("--roles", default="final")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if REVISION.fullmatch(args.model_revision) is None:
        raise ValueError("model revision must be an immutable 40-hex commit")

    model_root = args.model.resolve()
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("official model requires config.json and model.safetensors.index.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    if (
        config.get("architectures") != [RELEASED_ARCHITECTURE]
        or config.get("model_type") != RELEASED_MODEL_TYPE
        or text_config.get("model_type") != RELEASED_TEXT_MODEL_TYPE
        or text_config.get("num_hidden_layers") != 45
        or text_config.get("num_nextn_predict_layers") != 1
        or text_config.get("n_routed_experts") != 288
        or text_config.get("hidden_size") != 4096
        or text_config.get("moe_intermediate_size") != 2048
    ):
        raise ValueError("official GLM5Next main/MTP geometry differs")
    inventory = _inventory(args.inventory.resolve(), args.model_revision)
    if inventory.get("config_sha256") != sha256_file(config_path) or inventory.get("index_sha256") != sha256_file(index_path):
        raise ValueError("BF16 inventory does not bind the local config/index")
    raw_contract = json.loads(args.direct_contract.resolve().read_text(encoding="utf-8"))
    bits = int(raw_contract.get("rate", {}).get("bits", -1))
    contract = _sealed_json(
        args.direct_contract.resolve(), contract_schema_for_bits(bits), "contract_sha256"
    )
    if contract_bits(contract) != bits:
        raise ValueError("packed student contract rate differs")
    if contract.get("inventory_sha256") != inventory["inventory_sha256"]:
        raise ValueError("direct MCG contract targets another BF16 inventory")
    mtp_adapter = _sealed_json(
        args.mtp_adapter_receipt.resolve(),
        f"quant-pipeline.glm53-uniform-k{bits}-mtp-adapter-receipt.v1",
        "receipt_sha256",
    )
    surface = load_complete_surface(
        root=args.packed_root.resolve(), contract=contract, mtp_adapter_receipt=mtp_adapter
    )
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    panel_receipt, _, windows = load_panel_windows(
        args.token_panel_receipt.resolve(), roles=roles, vocab_size=int(text_config["vocab_size"])
    )
    module_path = Path(__file__).parents[1] / "src/quant_pipeline/evaluation/glm53_packed_k4_reader.py"
    identity = reader_identity(module_path, Path(__file__), bits=bits)
    checkpoint_identity = sha256_bytes(
        canonical_json(
            {
                "schema": f"quant-pipeline.glm53-packed-k{bits}-student-identity.v1",
                "inventory_sha256": inventory["inventory_sha256"],
                "contract_sha256": surface.contract_sha256,
                "main_layer_receipt_sha256": list(surface.main_layer_receipt_sha256),
                "mtp_adapter_receipt_sha256": surface.mtp_adapter_receipt_sha256,
                "mtp_pack_receipt_sha256": surface.mtp_pack_receipt_sha256,
                "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
                "bits": bits,
                "codebook": "MCG",
                "nonrouted_policy": "official_source_native",
            }
        )
    )
    plan = {
        "schema": f"quant-pipeline.glm53-packed-k{bits}-student-logit-capture-plan.v1",
        "model": str(model_root),
        "model_revision": args.model_revision,
        "inventory_sha256": inventory["inventory_sha256"],
        "contract_sha256": surface.contract_sha256,
        "checkpoint_identity_sha256": checkpoint_identity,
        "runtime_reader_sha256": identity["runtime_reader_sha256"],
        "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
        "roles": list(roles),
        "windows": len(windows),
        "prediction_positions": sum(window.prediction_positions for window in windows),
        "parallelism": "expert_parallel_world_size_4_contiguous_72_experts_per_rank",
        "reader_mode": identity["mode"],
        "final_tp2_serving_kernel": False,
        "main_routed_policy": f"decode_hash_verified_packed_k{bits}_mcg_to_bf16_local_ep_parameters",
        "mtp_policy": "complete_and_receipt_required_but_not_executed_by_standard_logits",
        "nonrouted_policy": "untouched_official_checkpoint_parameters",
        "stored_logits_dtype": "float32",
        "output": str(args.output.resolve()),
        "dry_run": not args.execute,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    rank, local_rank, world_size = _distributed_environment()
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, __version__ as transformers_version
    from transformers.distributed.configuration_utils import DistributedConfig

    if tuple(int(part) for part in transformers_version.split(".")[:2]) < (5, 16):
        raise RuntimeError("packed GLM5Next EP reader requires transformers>=5.16")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if dist.get_world_size() != 4:
        raise RuntimeError("initialized process group is not world size four")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    output_root = args.output.resolve()
    if rank == 0:
        prepare_empty_destination(output_root)
        (output_root / "logits").mkdir()
        write_json(output_root / "plan.json", plan | {"dry_run": False})
        write_json(output_root / "reader-identity.json", identity)
    dist.barrier()

    load_started = time.monotonic()
    distributed = DistributedConfig(tp_size=world_size, fsdp_size=1, pp_size=1, enable_expert_parallel=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_root,
        dtype=torch.bfloat16,
        distributed_config=distributed,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    non_cuda = sorted(
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if _tensor_device_type(value) != "cuda"
    )
    if non_cuda:
        raise RuntimeError(f"student is not fully GPU resident: {non_cuda[:8]}")
    versions_before = {name: int(value._version) for name, value in model.named_parameters()}
    closure = None
    if rank == 0:
        closure = stored_encoder_closure(
            surface, layer=MAIN_ROUTED_LAYERS[0], expert=0, projection="gate_proj", device=torch.device("cuda", local_rank)
        )
    install_started = time.monotonic()
    install = install_local_main_experts(
        model, surface, rank=rank, device=torch.device("cuda", local_rank)
    )
    torch.cuda.synchronize(local_rank)
    versions_after = {name: int(value._version) for name, value in model.named_parameters()}
    unexpected = sorted(
        name
        for name, version in versions_before.items()
        if versions_after[name] != version and not _is_mutated_routed_parameter(name)
    )
    if unexpected:
        raise RuntimeError(f"packed reader mutated non-routed official parameters: {unexpected[:8]}")
    install.update(
        {
            "gpu": torch.cuda.get_device_name(local_rank),
            "load_seconds": install_started - load_started,
            "install_seconds": time.monotonic() - install_started,
            "allocated_bytes": int(torch.cuda.memory_allocated(local_rank)),
            "reserved_bytes": int(torch.cuda.memory_reserved(local_rank)),
            "parameter_version_nonrouted_unchanged": True,
        }
    )
    rank_installs: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(rank_installs, install)
    closures: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(closures, closure)
    if rank == 0:
        if sum(int(row["installed_matrix_count"]) for row in rank_installs if row) != len(MAIN_ROUTED_LAYERS) * 288 * 3:
            raise RuntimeError("EP4 rank installs do not close the complete main routed matrix census")
        backend = {
            "schema": f"quant-pipeline.glm53-packed-k{bits}-offline-reader-backend.v1",
            "architecture": RELEASED_ARCHITECTURE,
            "model_revision": args.model_revision,
            "inventory_sha256": inventory["inventory_sha256"],
            "checkpoint_identity_sha256": checkpoint_identity,
            "contract_sha256": surface.contract_sha256,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "attention_backend": args.attention_backend,
            "parallelism": plan["parallelism"],
            "reader_mode": identity["mode"],
            "final_tp2_serving_kernel": False,
            "main_routed_runtime_dtype": f"bfloat16 decoded from packed K{bits} payload",
            "nonrouted_runtime_dtype": "official source dtype, untouched",
            "mtp_standard_logits_executed": False,
            "mtp_pack_receipt_sha256": surface.mtp_pack_receipt_sha256,
            "stored_encoder_closure": next(item for item in closures if item is not None),
            "rank_installs": rank_installs,
            "allow_tf32": False,
            "active_tp_plan": getattr(model, "_tp_plan", None),
            "active_ep_plan": getattr(model, "_ep_plan", None),
        }
        backend["backend_identity_sha256"] = sha256_bytes(canonical_json(backend))
        write_json(output_root / "backend.json", backend)
    else:
        backend = None
    backend_rows: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(backend_rows, backend)
    backend = next(item for item in backend_rows if item is not None)

    logit_records = []
    capture_started = time.monotonic()
    input_device = torch.device("cuda", local_rank)
    for index, window in enumerate(windows):
        tokens = np.load(window.token_path, allow_pickle=False)
        mask = np.load(window.attention_mask_path, allow_pickle=False)
        causal_mask = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
        ids = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(0).to(input_device)
        attention_mask = torch.from_numpy(np.asarray(mask, dtype=np.int64)).unsqueeze(0).to(input_device)
        with torch.inference_mode():
            output_logits = model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[:, :-1, :]
        selected = output_logits[0, torch.from_numpy(causal_mask).to(input_device)]
        if selected.shape != (window.prediction_positions, int(text_config["vocab_size"])):
            raise RuntimeError("student logits differ from sealed panel geometry")
        if rank == 0:
            stored = selected.float().cpu().contiguous()
            logit_path = (output_root / "logits" / f"window-{index:04d}.safetensors").resolve()
            save_file(
                {"logits": stored},
                logit_path,
                metadata={
                    "capture_role": "packed_student",
                    "student_label": f"uniform-k{bits}",
                    "window_id": window.window_id,
                    "token_ids_sha256": window.token_sha256,
                    "attention_mask_sha256": window.attention_mask_sha256,
                    "checkpoint_identity_sha256": checkpoint_identity,
                    "runtime_reader_sha256": identity["runtime_reader_sha256"],
                    "backend_identity_sha256": backend["backend_identity_sha256"],
                },
            )
            logit_records.append(
                {
                    "window_id": window.window_id,
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "role": window.role,
                    "token_ids_sha256": window.token_sha256,
                    "attention_mask_sha256": window.attention_mask_sha256,
                    "prediction_positions": window.prediction_positions,
                    "path": str(logit_path),
                    "bytes": logit_path.stat().st_size,
                    "sha256": sha256_file(logit_path),
                }
            )
        del ids, attention_mask, output_logits, selected
        dist.barrier()
    if rank == 0:
        receipt = {
            "schema": "quant-pipeline.glm53-logit-capture.v1",
            "capture_role": "packed_student",
            "model_revision": args.model_revision,
            "checkpoint_identity_sha256": checkpoint_identity,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
            "backend_identity_sha256": backend["backend_identity_sha256"],
            "weight_dtype": f"EXL3/TR3 uniform-k{bits} offline-decoded to BF16",
            "logits_dtype": "float32",
            "kld_direction": "teacher_to_student",
            "prediction_positions": sum(window.prediction_positions for window in windows),
            "vocab_size": int(text_config["vocab_size"]),
            "logit_files": logit_records,
            "elapsed_seconds": time.monotonic() - capture_started,
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
        write_json(output_root / "capture-receipt.json", receipt)
        print(json.dumps({"ok": True, **receipt}, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
