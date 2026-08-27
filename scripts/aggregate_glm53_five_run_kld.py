#!/usr/bin/env python3
"""Seal the mean and dispersion of exactly five cold GLM-5.3 K4 KLD runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_pipeline.core.artifacts import sha256_file, write_json
from quant_pipeline.publication.glm53_k4_postmtp import build_five_run_kld_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kld-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(args.kld_report) != 5:
        raise ValueError("pass exactly five --kld-report paths")
    reports = []
    for path in args.kld_report:
        resolved = path.resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"KLD report is not an object: {resolved}")
        reports.append((value, sha256_file(resolved)))
    receipt = build_five_run_kld_receipt(reports)
    print(json.dumps({"dry_run": not args.execute, **receipt}, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json(output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
