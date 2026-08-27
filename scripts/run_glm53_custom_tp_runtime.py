#!/usr/bin/env python3
"""Run text generation with the selective packed GLM-5.3 K4/TP2 or K6/TP4 runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from quant_pipeline.runtime.glm53_tp2_exl3 import (
    packed_runtime_census,
    patch_transformers,
    target_tp_size_for_bits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--exllamav3-source", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("launch with torchrun --nproc-per-node=2 for K4 or 4 for K6") from error

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    bits = int(config.get("quantization_config", {}).get("bits", 0))
    expected_tp = target_tp_size_for_bits(bits)
    if world_size != expected_tp or rank not in range(world_size) or local_rank not in range(world_size):
        raise RuntimeError(f"packed K{bits} requires exactly TP{expected_tp}")
    patch_transformers(exllamav3_source=args.exllamav3_source)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, Glm5NextForConditionalGeneration
    from transformers.distributed import DistributedConfig

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(
            tp_size=world_size, tp_plan="auto", enable_expert_parallel=False
        ),
        attn_implementation=args.attention_backend,
        local_files_only=True,
    ).eval()
    census = packed_runtime_census(model)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: value.to(torch.device("cuda", local_rank)) for name, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "bits": bits,
                    "tp_size": world_size,
                    "packed_matrix_count": census["packed_matrix_count"],
                    "text": tokenizer.decode(generated[0], skip_special_tokens=True),
                },
                ensure_ascii=False,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
