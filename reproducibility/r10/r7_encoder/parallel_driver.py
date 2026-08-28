#!/usr/bin/env python3
"""Layer-parallel driver: shard the 75 routed MoE layers across GPU processes.

Modeled directly on the owner's proven ``encode_b300.py`` orchestrator
(lines 465-551).  The contract it reproduces:

* one OS **process** per rank, never a thread -- each child owns exactly one
  visible GPU;
* ``CUDA_VISIBLE_DEVICES`` is placed in the child's environment **before** the
  child interpreter starts, so it is set long before that child imports Torch;
* rank ``r`` owns the strided slice ``layers[r::workers]``, which is a pure
  function of ``(layers, rank, workers)`` and therefore identical on every
  resume;
* the parent waits on every child, collects every return code, and fails the
  run if any child failed.

Thread pinning follows ``process_pool.py``: the CPU BF16 reductions in this
encoder are thread-count dependent, so every child inherits the same
``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`` value (``THREADS_PER_WORKER``) that
the sequential authoritative path uses.  Process-level parallelism is the only
new scheduling dimension.

This module is pure stdlib and imports no Torch at module scope, so the parent
can plan and print an assignment without ever initializing CUDA.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

__all__ = [
    "THREADS_PER_WORKER",
    "MOE_LAYERS",
    "parse_layers",
    "layer_slice",
    "worker_environment",
    "spawn_workers",
    "wait_workers",
    "assignment_table",
    "main",
]


# Matches r7_encoder.process_pool._THREADS_PER_WORKER / determinism.
# DETERMINISTIC_ENVIRONMENT.  Coordinator and workers must agree exactly or the
# same inputs emit different weights.
THREADS_PER_WORKER = 36

# CPU thread-pool variables pinned in every child before its first Torch import.
CPU_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# The routed MoE layers of GLM-5.2: 3..77 inclusive (75 layers).  Layer 78 is
# the MTP head and is not encoded here.
MOE_LAYERS = tuple(range(3, 78))

# Grace period, in seconds, between terminate() and kill() when reaping.
TERMINATE_GRACE_SECONDS = 10.0

# How long the parent blocks on one outstanding rank before re-sweeping the
# others for an exit.  Bounds the latency of noticing a failed sibling.
POLL_SECONDS = 0.25


def default_log(message: str) -> None:
    """Line-buffered stderr logger; callers may substitute their own."""

    print(message, file=sys.stderr, flush=True)


def parse_layers(spec: str) -> list[int]:
    """Parse ``"3-77"`` or ``"3,4,9-12"`` into a sorted, de-duplicated list."""

    layers: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    out = sorted(set(layers))
    if not out or any(layer not in MOE_LAYERS for layer in out):
        raise ValueError(
            f"layers must be a nonempty subset of {MOE_LAYERS[0]}..{MOE_LAYERS[-1]}, "
            f"got {spec!r}"
        )
    return out


def layer_slice(layers, rank: int, workers: int) -> list[int]:
    """Return rank ``rank``'s strided slice: exactly ``layers[rank::workers]``.

    Strided (not contiguous) assignment is the owner's model
    (``encode_b300.py:476``).  It balances cost across ranks even when per-layer
    cost drifts with depth, and it is stable under a resumed subset because the
    slice depends only on position in the full requested layer list.
    """

    ordered = list(layers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if not 0 <= rank < workers:
        raise ValueError(f"rank must be in [0,{workers}), got {rank}")
    return ordered[rank::workers]


def worker_environment(rank: int, gpus: int, *, base=None) -> dict:
    """Build the child environment: GPU pinning plus the CPU thread contract.

    Returned as a plain dict handed to ``subprocess.Popen(env=...)`` so the
    values are present in the child's environment at interpreter start -- i.e.
    strictly before the child imports Torch.
    """

    if gpus < 1:
        raise ValueError(f"gpus must be >= 1, got {gpus}")
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = str(rank % gpus)
    for name in CPU_ENVIRONMENT_NAMES:
        env[name] = str(THREADS_PER_WORKER)
    return env


def _terminate(processes, log, *, reason: str) -> None:
    """Terminate then kill any still-running child; never leave orphans."""

    survivors = [
        (rank, process)
        for rank, process in enumerate(processes)
        if process.poll() is None
    ]
    if not survivors:
        return
    log(f"terminating {len(survivors)} surviving worker(s): {reason}")
    for rank, process in survivors:
        try:
            process.terminate()
        except OSError as exc:  # already reaped between poll() and terminate()
            log(f"worker {rank} terminate failed: {exc}")
    for rank, process in survivors:
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            log(f"worker {rank} pid {process.pid} ignored SIGTERM; killing")
            try:
                process.kill()
            except OSError as exc:
                log(f"worker {rank} kill failed: {exc}")
            try:
                process.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                log(f"worker {rank} pid {process.pid} is unreapable")


def spawn_workers(*, layers, workers: int, gpus: int, worker_cmd_builder, log=default_log):
    """Spawn one pinned process per rank and return the ``Popen`` handles.

    ``worker_cmd_builder(rank)`` returns the argv list for that rank.  Each
    child is started with ``CUDA_VISIBLE_DEVICES=str(rank % gpus)`` and the
    ``THREADS_PER_WORKER`` CPU thread contract already in its environment.
    If any spawn fails, every already-spawned child is reaped before raising,
    so a partial spawn never leaks GPU-holding orphans.
    """

    # Mirrors encode_b300.py:485 -- one rank per GPU at most.
    if not (1 <= workers <= gpus):
        raise RuntimeError(f"workers must be in [1,gpus], got {workers}/{gpus}")
    ordered = list(layers)
    if not ordered:
        raise RuntimeError("no layers to encode")

    processes: list[subprocess.Popen] = []
    try:
        for rank in range(workers):
            mine = layer_slice(ordered, rank, workers)
            env = worker_environment(rank, gpus)
            command = [str(part) for part in worker_cmd_builder(rank)]
            if not command:
                raise RuntimeError(f"worker_cmd_builder({rank}) returned an empty argv")
            process = subprocess.Popen(command, env=env)
            processes.append(process)
            log(
                f"spawned worker {rank} pid {process.pid} on GPU {rank % gpus} "
                f"with {len(mine)} layer(s)"
            )
    except BaseException:
        _terminate(processes, log, reason="spawn failed")
        raise
    return processes


def wait_workers(processes, log=default_log) -> list[int]:
    """Wait for every worker, collect return codes, and fail on any nonzero.

    A return code is collected for every rank.  The owner's model waits on the
    ranks in order; this driver instead reaps them as they exit, because a
    strictly in-order wait cannot notice that rank 5 died while it is blocked
    on rank 0 -- and if a sibling then hangs, the driver hangs with it holding
    every GPU.  So: as soon as any rank exits nonzero, the survivors are
    terminated (SIGTERM, then SIGKILL after ``TERMINATE_GRACE_SECONDS``) and
    reaped before the failure is raised.  ``KeyboardInterrupt`` takes the same
    path.  Nothing this function spawned outlives it.

    Fail-fast costs at most the in-flight layer on each healthy rank; per-layer
    completion markers make that work resumable, whereas a wedged driver is not.
    """

    handles = list(processes)
    codes: list[int | None] = [None] * len(handles)
    failed: list[tuple[int, int]] = []
    try:
        while True:
            for rank, process in enumerate(handles):
                if codes[rank] is not None:
                    continue
                code = process.poll()
                if code is None:
                    continue
                codes[rank] = code
                log(f"worker {rank} exited {code}")
                if code != 0:
                    failed.append((rank, code))
            pending = [rank for rank, code in enumerate(codes) if code is None]
            if failed or not pending:
                break
            # Block on one outstanding rank rather than busy-spinning; the
            # timeout bounds how long a failure elsewhere can go unnoticed.
            try:
                handles[pending[0]].wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        _terminate(handles, log, reason="driver interrupted")
        raise

    if failed:
        _terminate(handles, log, reason="worker failure")
        terminated: list[tuple[int, int]] = []
        for rank, process in enumerate(handles):
            if codes[rank] is None:
                codes[rank] = process.poll()
                terminated.append((rank, codes[rank]))
                log(f"worker {rank} terminated by driver, exit {codes[rank]}")
        detail = ", ".join(f"rank {rank} exit {code}" for rank, code in failed)
        message = f"{len(failed)} worker(s) failed: {detail}"
        if terminated:
            extra = ", ".join(f"rank {rank} exit {code}" for rank, code in terminated)
            message += f"; driver-terminated survivors: {extra}"
        raise RuntimeError(message)
    return [int(code) for code in codes]


def assignment_table(layers, workers: int, gpus: int) -> str:
    """Render the deterministic per-rank assignment as a printable table."""

    ordered = list(layers)
    rows = [
        (rank, rank % gpus, layer_slice(ordered, rank, workers))
        for rank in range(workers)
    ]
    lines = [
        f"layers={len(ordered)} workers={workers} gpus={gpus}",
        f"{'rank':>4}  {'gpu':>3}  {'count':>5}  layers",
    ]
    for rank, gpu, mine in rows:
        listed = ",".join(str(layer) for layer in mine) if mine else "-"
        lines.append(f"{rank:>4}  {gpu:>3}  {len(mine):>5}  {listed}")
    covered = sorted(layer for _, _, mine in rows for layer in mine)
    lines.append(
        f"coverage: {len(covered)} assigned, "
        f"{len(set(covered))} unique, complete={covered == ordered}"
    )
    return "\n".join(lines)


def _self_worker_command(args, rank: int) -> list[str]:
    """Default builder: re-invoke this module as the worker for ``rank``.

    The worker body is a stub -- it prints its assignment and exits 0.  A
    caller replaces this builder (or the ``--worker-rank`` branch) with the
    real per-layer encode.
    """

    return [
        sys.executable,
        "-m",
        "r7_encoder.parallel_driver",
        "--worker-rank",
        str(rank),
        "--layers",
        args.layers,
        "--workers",
        str(args.workers),
        "--gpus",
        str(args.gpus),
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--layers",
        default="3-77",
        help='layer spec: "3-77" or a comma list such as "3,4,9-12" (default: 3-77)',
    )
    parser.add_argument("--workers", type=int, default=6, help="worker processes")
    parser.add_argument("--gpus", type=int, default=6, help="visible GPUs")
    parser.add_argument(
        "--worker-rank",
        type=int,
        default=None,
        help=(
            "if present this process IS a worker: print its assigned layers and "
            "exit 0 (stub; the caller replaces this body)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the per-rank layer assignment table and exit without spawning",
    )
    args = parser.parse_args(argv)

    layers = parse_layers(args.layers)
    if not (1 <= args.workers <= args.gpus):
        raise RuntimeError(f"workers must be in [1,gpus], got {args.workers}/{args.gpus}")

    if args.worker_rank is not None:
        # Worker body (stub).  CUDA_VISIBLE_DEVICES is already pinned by the
        # parent; a real worker would import Torch only from here down.
        mine = layer_slice(layers, args.worker_rank, args.workers)
        print(
            f"worker {args.worker_rank} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
            f"layers={','.join(str(layer) for layer in mine)}",
            flush=True,
        )
        return 0

    if args.dry_run:
        print(assignment_table(layers, args.workers, args.gpus), flush=True)
        return 0

    default_log(
        f"layer-parallel driver: {len(layers)} layer(s), workers={args.workers}, "
        f"gpus={args.gpus}, threads/worker={THREADS_PER_WORKER}"
    )
    processes = spawn_workers(
        layers=layers,
        workers=args.workers,
        gpus=args.gpus,
        worker_cmd_builder=lambda rank: _self_worker_command(args, rank),
        log=default_log,
    )
    wait_workers(processes, default_log)
    default_log(f"all {args.workers} worker(s) completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
