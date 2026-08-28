"""Immutable covariance and holdout row-plan caches."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .constants import RECIPE_MARKER, RECIPE_VERSION
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .safetensors_io import (
    SafeTensorReader,
    read_torch_tensor,
    torch_tensor_entry,
    write_safetensors_atomic,
)


def _paths(root: Path, expert: int, kind: str) -> tuple[Path, Path]:
    return (
        root / f"expert-{expert:03d}-{kind}.safetensors",
        root / f"expert-{expert:03d}-{kind}.json",
    )


def _discard_derivative_pair(shard: Path, manifest: Path) -> None:
    """Remove only an unsealed derivative cache pair so it can be recomputed."""

    shard.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)


def _write(
    root: Path,
    *,
    expert: int,
    kind: str,
    entries,
    bindings: Mapping[str, str],
    metadata: Mapping[str, object],
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    shard, manifest_path = _paths(root, expert, kind)
    if shard.exists() or manifest_path.exists():
        raise FileExistsError("row-cache writer requires a clean transaction pair")
    payload_hashes, shard_hash = write_safetensors_atomic(
        shard,
        entries,
        metadata={
            "r7_schema": "r7-row-cache-v1",
            "kind": kind,
            "expert": str(expert),
        },
    )
    manifest = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-row-cache-v1",
        "expert": expert,
        "kind": kind,
        "shard": shard.name,
        "shard_sha256": shard_hash,
        "payload_sha256": payload_hashes,
        "bindings": dict(sorted(bindings.items())),
        "metadata": dict(metadata),
    }
    manifest["content_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    atomic_write_json(manifest_path, manifest)
    return sha256_file(manifest_path)


def _load(
    root: Path,
    *,
    expert: int,
    kind: str,
    bindings: Mapping[str, str],
):
    shard, manifest_path = _paths(root, expert, kind)
    if not shard.exists() and not manifest_path.exists():
        return None
    if not shard.is_file() or not manifest_path.is_file():
        _discard_derivative_pair(shard, manifest_path)
        return None
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, TypeError):
        _discard_derivative_pair(shard, manifest_path)
        return None
    if (
        manifest.get("marker") != RECIPE_MARKER
        or manifest.get("schema") != "r7-row-cache-v1"
        or int(manifest.get("expert", -1)) != expert
        or manifest.get("kind") != kind
        or manifest.get("bindings") != dict(sorted(bindings.items()))
    ):
        raise ValueError("row-cache identity/binding drift")
    content = manifest.pop("content_sha256", None)
    if content != sha256_bytes(canonical_json_bytes(manifest)):
        _discard_derivative_pair(shard, manifest_path)
        return None
    manifest["content_sha256"] = content
    if sha256_file(shard) != manifest["shard_sha256"]:
        _discard_derivative_pair(shard, manifest_path)
        return None
    try:
        reader = SafeTensorReader(shard)
    except (OSError, ValueError, TypeError):
        _discard_derivative_pair(shard, manifest_path)
        return None
    if set(reader.tensors) != set(manifest["payload_sha256"]):
        _discard_derivative_pair(shard, manifest_path)
        return None
    for name, expected in manifest["payload_sha256"].items():
        if reader.tensors[name].payload.sha256() != expected:
            _discard_derivative_pair(shard, manifest_path)
            return None
    return manifest, reader


def write_covariance_cache(
    root: str | Path,
    *,
    expert: int,
    matrix,
    row_ids,
    fallback_row_ids,
    bindings: Mapping[str, str],
    metadata: Mapping[str, object],
) -> str:
    import torch

    return _write(
        Path(root),
        expert=expert,
        kind="fit-covariance",
        entries=(
            torch_tensor_entry("covariance", torch.as_tensor(matrix).float()),
            torch_tensor_entry(
                "row_ids", torch.tensor(tuple(row_ids), dtype=torch.int64)
            ),
            torch_tensor_entry(
                "fallback_row_ids",
                torch.tensor(tuple(fallback_row_ids), dtype=torch.int64),
            ),
        ),
        bindings=bindings,
        metadata=metadata,
    )


def load_covariance_cache(
    root: str | Path,
    *,
    expert: int,
    bindings: Mapping[str, str],
    device: str,
):
    value = _load(Path(root), expert=expert, kind="fit-covariance", bindings=bindings)
    if value is None:
        return None
    manifest, reader = value
    return {
        "matrix": read_torch_tensor(reader, "covariance").to(device),
        "row_ids": tuple(
            int(item) for item in read_torch_tensor(reader, "row_ids").tolist()
        ),
        "fallback_row_ids": tuple(
            int(item) for item in read_torch_tensor(reader, "fallback_row_ids").tolist()
        ),
        "metadata": manifest["metadata"],
        "manifest_sha256": sha256_file(
            Path(root) / f"expert-{expert:03d}-fit-covariance.json"
        ),
    }


def write_holdout_cache(
    root: str | Path,
    *,
    expert: int,
    hidden,
    row_ids,
    bindings: Mapping[str, str],
    metadata: Mapping[str, object],
) -> str:
    import torch

    value = torch.as_tensor(hidden).detach().to("cpu")
    return _write(
        Path(root),
        expert=expert,
        kind="holdout",
        entries=(
            torch_tensor_entry("hidden", value),
            torch_tensor_entry(
                "row_ids", torch.tensor(tuple(row_ids), dtype=torch.int64)
            ),
        ),
        bindings=bindings,
        metadata=metadata,
    )


def load_holdout_cache(
    root: str | Path,
    *,
    expert: int,
    bindings: Mapping[str, str],
    device: str,
):
    value = _load(Path(root), expert=expert, kind="holdout", bindings=bindings)
    if value is None:
        return None
    manifest, reader = value
    return {
        "hidden": read_torch_tensor(reader, "hidden").to(device),
        "row_ids": tuple(
            int(item) for item in read_torch_tensor(reader, "row_ids").tolist()
        ),
        "metadata": manifest["metadata"],
        "manifest_sha256": sha256_file(
            Path(root) / f"expert-{expert:03d}-holdout.json"
        ),
    }
