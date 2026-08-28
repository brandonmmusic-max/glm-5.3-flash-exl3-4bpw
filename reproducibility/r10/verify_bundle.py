#!/usr/bin/env python3
"""Verify the published R10 encoder source closure without a GPU."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SOURCE_SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.dont_write_bytecode = True
    expected: dict[str, str] = {}
    for number, raw in enumerate(MANIFEST.read_text().splitlines(), 1):
        digest, separator, relative = raw.partition("  ")
        path = Path(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute()
            or ".." in path.parts
            or relative in expected
        ):
            raise SystemExit(f"unsafe manifest row {number}")
        expected[relative] = digest

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "r7_encoder").glob("*.py")
    }
    actual.add("lineage/encode_tr3_v31.py")
    if actual != set(expected):
        raise SystemExit(
            "source inventory mismatch: "
            + json.dumps(
                {
                    "missing": sorted(set(expected) - actual),
                    "unmanifested": sorted(actual - set(expected)),
                },
                sort_keys=True,
            )
        )

    for relative, wanted in sorted(expected.items()):
        path = ROOT / relative
        observed = sha256(path)
        if observed != wanted:
            raise SystemExit(f"SHA-256 mismatch: {relative}")
        ast.parse(path.read_text(), filename=str(path))

    sys.path.insert(0, str(ROOT))
    from r7_encoder.r10_codec import R10TrellisCodec

    if R10TrellisCodec.__module__ != "r7_encoder.r10_codec":
        raise SystemExit("R10TrellisCodec resolved outside the published closure")
    print(
        json.dumps(
            {
                "file_count": len(expected),
                "numeric_core_sha256": expected["lineage/encode_tr3_v31.py"],
                "ok": True,
                "r10_codec_sha256": expected["r7_encoder/r10_codec.py"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
