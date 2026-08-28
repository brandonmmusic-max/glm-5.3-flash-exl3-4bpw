#!/usr/bin/env python3
"""Command line entry points for the Round 7 draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assemble import assemble
from .convert_v2_to_v1 import convert_checkpoint, convert_layer
from .inventory import (
    build_checkpoint_inventory,
    build_numeric_environment_inventory,
    build_runtime_code_inventory,
)
from .oracles import audit_v2_layer
from .preflight import preflight
from .walk import SequentialWalk, WalkConfig


def _walk_config(args) -> WalkConfig:
    return WalkConfig(
        carrier=args.carrier,
        bf16_source=args.src,
        corpus=args.corpus,
        work=args.work,
        backend_factory=args.backend,
        runtime_factory=args.runtime,
        runtime_inventory=args.runtime_inventory,
        carrier_inventory=args.carrier_inventory,
        source_inventory=args.source_inventory,
        numeric_inventory=args.numeric_inventory,
        device=args.device,
        devices=tuple(
            part.strip()
            for part in getattr(args, "devices", "").split(",")
            if part.strip()
        ),
        sigma_reg=args.sigma_reg,
        fixed_point_iterations=args.fixed_point_iterations,
        holdout_rows=args.holdout_rows,
        draws=args.draws,
        shared_sample_experts=args.shared_sample_experts,
        retire_predecessor_state=not args.keep_states,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("preflight", help="read model headers only")
    check.add_argument("--carrier", type=Path, required=True)
    check.add_argument("--src", type=Path, required=True)

    inventory = subparsers.add_parser(
        "inventory-checkpoint", help="content-hash an indexed checkpoint"
    )
    inventory.add_argument("--checkpoint", type=Path, required=True)
    inventory.add_argument("--out", type=Path, required=True)
    inventory.add_argument("--role", choices=("carrier", "bf16-source"), required=True)

    numeric = subparsers.add_parser(
        "inventory-numeric", help="seal the numerical core and TRELLIS extension"
    )
    numeric.add_argument("--numeric-core", type=Path, required=True)
    numeric.add_argument("--extension", type=Path, required=True)
    numeric.add_argument("--device", default="cuda:0")
    numeric.add_argument("--out", type=Path, required=True)

    runtime_inventory = subparsers.add_parser(
        "inventory-runtime", help="seal all GLM runtime/model source files"
    )
    runtime_inventory.add_argument(
        "--file",
        type=Path,
        action="append",
        required=True,
        help="source file or package directory; repeat for disjoint roots",
    )
    runtime_inventory.add_argument("--out", type=Path, required=True)

    audit = subparsers.add_parser("audit-v2", help="hash and shape audit one v2 layer")
    audit.add_argument("--manifest", type=Path, required=True)

    convert = subparsers.add_parser("convert-v1", help="topology conversion only")
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--tp", type=int, required=True)
    convert.add_argument("--assert-unmodified-r13", action="store_true")

    convert_all = subparsers.add_parser(
        "convert-checkpoint-v1", help="convert a complete assembled v2 checkpoint"
    )
    convert_all.add_argument("--checkpoint", type=Path, required=True)
    convert_all.add_argument("--out", type=Path, required=True)
    convert_all.add_argument("--tp", type=int, required=True)
    convert_all.add_argument("--assert-unmodified-r13", action="store_true")

    assembly = subparsers.add_parser("assemble", help="build complete checkpoint")
    assembly.add_argument("--carrier", type=Path, required=True)
    assembly.add_argument("--v2", type=Path, required=True)
    assembly.add_argument("--out", type=Path, required=True)
    assembly.add_argument("--carrier-inventory", type=Path, required=True)
    assembly.add_argument("--walk-manifest", type=Path, required=True)

    run = subparsers.add_parser("run", help="owner-run sequential GPU walk")
    run.add_argument("--carrier", type=Path, required=True)
    run.add_argument("--src", type=Path, required=True)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--backend", required=True, help="module:factory")
    run.add_argument(
        "--runtime", required=True, help="fingerprinted GLM-5.2 runtime module:factory"
    )
    run.add_argument("--runtime-inventory", type=Path, required=True)
    run.add_argument("--carrier-inventory", type=Path, required=True)
    run.add_argument("--source-inventory", type=Path, required=True)
    run.add_argument("--numeric-inventory", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument(
        "--devices",
        default="",
        help=(
            "comma-separated CUDA device pool for the parallel warm passes, "
            "e.g. cuda:0,cuda:1,...; first entry must equal --device; "
            "empty = single-device"
        ),
    )
    run.add_argument("--sigma-reg", type=float, default=0.025)
    run.add_argument("--fixed-point-iterations", type=int, default=4)
    run.add_argument("--holdout-rows", type=int, default=4096)
    run.add_argument("--draws", type=int, default=12)
    run.add_argument("--shared-sample-experts", type=int, default=16)
    run.add_argument("--keep-states", action="store_true")
    run.add_argument(
        "--pilot-stop-after-layer",
        type=int,
        help=(
            "stop after this fully sealed routed layer and write a non-algorithmic "
            "sample-layer timing/storage report; omit for the complete walk"
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.carrier, args.src)
    elif args.command == "inventory-checkpoint":
        result = build_checkpoint_inventory(
            args.checkpoint,
            args.out,
            role=args.role,
            require_routed_bf16=args.role == "bf16-source",
        )
    elif args.command == "inventory-numeric":
        result = build_numeric_environment_inventory(
            args.out,
            numeric_core=args.numeric_core,
            extension=args.extension,
            device=args.device,
        )
    elif args.command == "inventory-runtime":
        result = build_runtime_code_inventory(args.out, files=args.file)
    elif args.command == "audit-v2":
        result = audit_v2_layer(args.manifest)
    elif args.command == "convert-v1":
        result = convert_layer(
            manifest_path=args.manifest,
            output_dir=args.out,
            tp_size=args.tp,
            assert_unmodified_r13=args.assert_unmodified_r13,
        )
    elif args.command == "convert-checkpoint-v1":
        result = convert_checkpoint(
            checkpoint=args.checkpoint,
            output_dir=args.out,
            tp_size=args.tp,
            assert_unmodified_r13=args.assert_unmodified_r13,
        )
    elif args.command == "assemble":
        result = assemble(
            args.carrier,
            args.v2,
            args.out,
            carrier_inventory=args.carrier_inventory,
            walk_manifest=args.walk_manifest,
        )
    elif args.command == "run":
        result = SequentialWalk(_walk_config(args)).run(
            pilot_stop_after_layer=args.pilot_stop_after_layer
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
