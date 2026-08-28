"""Layer-parallel encode driver: the owner's B300 architecture over R7 quality.

Two phases, mirroring `encode_b300.py`:

  PHASE A (capture)  one pass over the source model writing ONE flat tmpfs
                     capture per MoE layer (hidden bf16-as-int16, top-8 ids,
                     top-8 float32 router weights). Replaces R7's per-prompt
                     safetensors+sha256 writes, which cost ~4 min/layer.

  PHASE B (encode)   layers are handed to worker PROCESSES, one pinned per
                     GPU, from a shared work QUEUE. Dynamic assignment beats
                     `layers[rank::workers]` striding because it self-balances:
                     nobody sits idle holding 12 layers while a peer grinds 13.

Every R7 quality decision is untouched. Workers call the existing
`LayerProcessor`, which still does probe -> allocate -> final -> emit-v2 with
the joint down encode, Gap 1 round-trip inputs, exact-3.5 mass allocation,
per-expert permutation, shared residual rotations, and per-128 block scales.
Parallelism changes scheduling only.

The one deliberate divergence from R7: calibration comes from a single capture
of the source model rather than from quantized predecessors (Gap 2). That is
the owner's explicit decision and it is what makes layers independent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .constants import FIRST_MOE_LAYER, LAST_MOE_LAYER, MOE_LAYERS
from .determinism import atomic_write_json, read_json
from .parallel_driver import (
    THREADS_PER_WORKER,
    default_log,
    parse_layers,
    worker_environment,
)

QUEUE_SCHEMA = "r7-fast-encode-queue-v1"


# ---------------------------------------------------------------------------
# work queue: dynamic layer assignment
# ---------------------------------------------------------------------------


def claim_next_layer(queue_path: Path, rank: int) -> int | None:
    """Atomically claim the next unclaimed layer.

    A directory rename is the atomic primitive: exactly one worker can create
    `claims/<layer>` and the loser moves on. No lock server, no shared memory,
    and a crashed worker leaves a claim that a resume can audit.
    """
    state = read_json(queue_path)
    claims = Path(str(state["claims_dir"]))
    claims.mkdir(parents=True, exist_ok=True)
    for layer in state["layers"]:
        marker = claims / f"{int(layer):03d}"
        try:
            marker.mkdir()
        except FileExistsError:
            continue
        (marker / "owner").write_text(f"{rank}\n", encoding="utf-8")
        return int(layer)
    return None


def write_queue(queue_path: Path, *, layers, claims_dir: Path) -> None:
    atomic_write_json(
        queue_path,
        {
            "schema": QUEUE_SCHEMA,
            "layers": [int(item) for item in layers],
            "claims_dir": str(claims_dir),
        },
    )


# ---------------------------------------------------------------------------
# phase B worker: encode whatever layers it can claim
# ---------------------------------------------------------------------------


def run_worker(args) -> int:
    """One pinned process. Claims layers until the queue is empty."""

    log_path = Path(args.work) / "logs" / f"worker{args.worker_rank}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"{stamp} rank{args.worker_rank} {message}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    # CUDA_VISIBLE_DEVICES and the thread pins are already in the environment,
    # set by the parent BEFORE this interpreter started, so torch binds to the
    # right device and thread pool on first import.
    import torch

    from .determinism import configure_deterministic_environment

    configure_deterministic_environment()
    log(
        f"up on {torch.cuda.get_device_name(0)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"threads={torch.get_num_threads()}"
    )

    from .fast_layer import encode_one_layer

    queue_path = Path(args.work) / "QUEUE.json"
    done = 0
    while True:
        layer = claim_next_layer(queue_path, args.worker_rank)
        if layer is None:
            break
        started = time.time()
        log(f"layer {layer}: start")
        encode_one_layer(
            layer=layer,
            work=Path(args.work),
            capture_dir=Path(args.capture_dir),
            carrier=Path(args.carrier),
            bf16_source=Path(args.src),
            carrier_inventory=Path(args.carrier_inventory),
            source_inventory=Path(args.source_inventory),
            numeric_inventory=Path(args.numeric_inventory),
            runtime_inventory=Path(args.runtime_inventory),
            fixed_point_iterations=args.fixed_point_iterations,
            draws=args.draws,
            shared_sample_experts=args.shared_sample_experts,
            holdout_rows=args.holdout_rows,
            sigma_reg=args.sigma_reg,
            log=log,
        )
        elapsed = time.time() - started
        done += 1
        log(f"layer {layer}: SEALED in {elapsed/60:.1f} min")
        print(
            f"R9_LAYER_WALL {json.dumps({'layer': layer, 'seconds': round(elapsed, 1), 'rank': args.worker_rank})}",
            flush=True,
        )
    log(f"queue empty; {done} layer(s) encoded")
    return 0


# ---------------------------------------------------------------------------
# parent: spawn the pool, wait, fail closed
# ---------------------------------------------------------------------------


def run_parent(args) -> int:
    import subprocess

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    queue_path = work / "QUEUE.json"
    claims_dir = work / "claims"
    if not queue_path.exists():
        write_queue(queue_path, layers=layers, claims_dir=claims_dir)
    default_log(
        f"fast encode: {len(layers)} layers, {args.workers} workers on "
        f"{args.gpus} GPUs, dynamic queue, fixed_point_iterations="
        f"{args.fixed_point_iterations}"
    )

    processes = []
    for rank in range(args.workers):
        env = worker_environment(rank, args.gpus, base=os.environ.copy())
        cmd = [
            sys.executable,
            "-m",
            "r7_encoder.fast_encode",
            "--worker-rank",
            str(rank),
            "--work",
            str(work),
            "--capture-dir",
            str(args.capture_dir),
            "--carrier",
            str(args.carrier),
            "--src",
            str(args.src),
            "--carrier-inventory",
            str(args.carrier_inventory),
            "--source-inventory",
            str(args.source_inventory),
            "--numeric-inventory",
            str(args.numeric_inventory),
            "--runtime-inventory",
            str(args.runtime_inventory),
            "--fixed-point-iterations",
            str(args.fixed_point_iterations),
            "--draws",
            str(args.draws),
            "--shared-sample-experts",
            str(args.shared_sample_experts),
            "--holdout-rows",
            str(args.holdout_rows),
            "--sigma-reg",
            str(args.sigma_reg),
            "--layers",
            args.layers,
            "--workers",
            str(args.workers),
            "--gpus",
            str(args.gpus),
        ]
        process = subprocess.Popen(cmd, env=env)
        processes.append((rank, process))
        default_log(f"spawned worker {rank} pid {process.pid} on GPU {rank % args.gpus}")

    failures = []
    try:
        for rank, process in processes:
            code = process.wait()
            if code != 0:
                failures.append((rank, code))
    except KeyboardInterrupt:
        for _, process in processes:
            process.terminate()
        raise
    if failures:
        detail = ", ".join(f"rank {rank} exit {code}" for rank, code in failures)
        raise RuntimeError(f"fast encode worker failure: {detail}")
    default_log("all workers finished")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="layer-parallel R7 encode")
    parser.add_argument("--work", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--carrier-inventory", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--numeric-inventory", required=True)
    parser.add_argument("--runtime-inventory", required=True)
    parser.add_argument("--layers", default=f"{FIRST_MOE_LAYER}-{LAST_MOE_LAYER}")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpus", type=int, default=6)
    parser.add_argument("--worker-rank", type=int)
    parser.add_argument("--fixed-point-iterations", type=int, default=1)
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--shared-sample-experts", type=int, default=16)
    parser.add_argument("--holdout-rows", type=int, default=4096)
    parser.add_argument("--sigma-reg", type=float, default=0.025)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_rank is not None:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
