"""Canonical serialization, hashes, deterministic seeds, and atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CUBLAS_WORKSPACE_POLICY,
    EXPERTS_IMPLEMENTATION,
    HUB_KERNEL_POLICY,
    RECIPE_MARKER,
    RECIPE_VERSION,
)


DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_POLICY,
    # Pin intra-op threads everywhere. CPU reductions (notably the
    # down-projection SwiGLU Hessian) are thread-count dependent, so
    # coordinator and workers must agree or the same inputs emit
    # different weights.
    "OMP_NUM_THREADS": "36",
    "MKL_NUM_THREADS": "36",
    "OPENBLAS_NUM_THREADS": "36",
    "NUMEXPR_NUM_THREADS": "36",
    "USE_HUB_KERNELS": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


def configure_deterministic_environment() -> dict[str, str]:
    """Pin process inputs before Torch CUDA or Transformers imports."""

    for name, expected in DETERMINISTIC_ENVIRONMENT.items():
        incumbent = os.environ.get(name)
        if incumbent is not None and incumbent != expected:
            raise RuntimeError(
                f"deterministic environment mismatch: {name}={incumbent!r}, "
                f"need {expected!r}"
            )
        os.environ[name] = expected
    # Torch fixes its intra-op pool from OMP_NUM_THREADS at import time, so
    # setting the variable alone is too late once torch is loaded. Pin it
    # explicitly: CPU reductions (the down-projection SwiGLU Hessian being the
    # one that bit us) are thread-count dependent, and the worker contract
    # requires coordinator and children to agree exactly.
    try:
        import torch as _torch

        _want = int(DETERMINISTIC_ENVIRONMENT["OMP_NUM_THREADS"])
        if _torch.get_num_threads() != _want:
            _torch.set_num_threads(_want)
        # Inter-op pool too: the worker contract checks both, and it can only
        # be set before any inter-op parallel work has started.
        if _torch.get_num_interop_threads() != _want:
            try:
                _torch.set_num_interop_threads(_want)
            except RuntimeError:
                pass
    except Exception:
        # torch not importable yet in this process; the env vars above still
        # pin it correctly at first import.
        pass
    return {
        **DETERMINISTIC_ENVIRONMENT,
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "hub_kernel_policy": HUB_KERNEL_POLICY,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(*parts: object, bits: int = 63) -> int:
    """Scheduling-independent seed derived from semantic coordinates."""

    if not 1 <= bits <= 64:
        raise ValueError("bits must be in [1,64]")
    payload = canonical_json_bytes(
        [RECIPE_MARKER, RECIPE_VERSION, *[str(part) for part in parts]]
    )
    raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return raw & ((1 << bits) - 1)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write, fsync, replace, then fsync the parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return json.load(handle)


def hash_named_files(root: str | Path, relative_paths: Iterable[str]) -> dict[str, str]:
    base = Path(root)
    return {
        relative: sha256_file(base / relative)
        for relative in sorted(set(relative_paths))
    }
