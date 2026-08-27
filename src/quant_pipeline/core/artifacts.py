from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    if is_dataclass(value):
        value = asdict(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # fsyncing the temporary file makes its contents durable, but it does
        # not make the rename durable.  A power loss can otherwise restore the
        # old directory entry (or no entry at all) after this function returns.
        parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_json(path: str | Path, value: Any) -> str:
    data = canonical_json(value)
    atomic_write(path, data)
    return sha256_bytes(data)


def bind_files(paths: list[str | Path]) -> dict[str, dict[str, int | str]]:
    bound: dict[str, dict[str, int | str]] = {}
    for raw in paths:
        path = Path(raw).resolve()
        bound[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return bound


def prepare_empty_destination(path: str | Path) -> Path:
    """Create a destination or fail closed if it contains any prior artifact."""
    destination = Path(path)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"destination exists and is not a directory: {destination}")
        if next(destination.iterdir(), None) is not None:
            raise FileExistsError(f"destination is not empty; inspect and choose a new path: {destination}")
    else:
        destination.mkdir(parents=True)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return destination


def require_execute(execute: bool, action: str) -> None:
    if not execute:
        raise RuntimeError(f"refusing to {action}: pass --execute after reviewing the sealed plan")
