"""Read-only provenance sealing for the GLM-5.3 uniform-K4 campaign.

The prepared process intentionally contains a dirty KQuant checkout.  A clean
checkout is therefore not a valid substitute: production receipts bind the
exact non-``.git`` bytes, Git HEAD, porcelain status, and binary working-tree
patch without modifying or normalizing the checkout.
"""

from __future__ import annotations

import copy
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.artifacts import canonical_json, sha256_bytes, sha256_file


PREPARED_TREE_SCHEMA = "quant-pipeline.glm53-prepared-process-provenance.v1"
KQUANT_SCHEMA = "quant-pipeline.glm53-kquant-working-tree-provenance.v1"
SM100_EXTENSION_SCHEMA = "quant-pipeline.glm53-sm100-extension-provenance.v1"
PREFLIGHT_SCHEMA = "quant-pipeline.glm53-sm100-preflight.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SELF_SEALS = frozenset({"MANIFEST.json", "SHA256SUMS"})


def _seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_seal(
    document: Mapping[str, Any], *, schema: str, field: str, label: str
) -> str:
    if document.get("schema") != schema:
        raise ValueError(f"{label} schema differs")
    digest = document.get(field)
    if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise ValueError(f"{label} seal is absent")
    body = copy.deepcopy(dict(document))
    del body[field]
    if sha256_bytes(canonical_json(body)) != digest:
        raise ValueError(f"{label} seal differs")
    return digest


def _files(root: Path, *, excluded_top_level: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded = set(excluded_top_level)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not relative.parts
            or ".git" in relative.parts
            or relative.parts[0] in excluded
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"provenance tree contains a symlink: {relative}")
        if not path.is_file():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "mode": f"{mode:04o}",
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"provenance tree contains no regular files: {root}")
    return rows


def _tree_identity(root: Path, *, excluded_top_level: Iterable[str] = ()) -> dict[str, Any]:
    rows = _files(root, excluded_top_level=excluded_top_level)
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": sha256_bytes(canonical_json(rows)),
        "files": rows,
    }


def _git(root: Path, *args: str, binary: bool = False) -> bytes | str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    command = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        *args,
    ]
    result = subprocess.run(command, check=True, capture_output=True, env=environment)
    return result.stdout if binary else result.stdout.decode("utf-8", errors="strict")


def seal_prepared_process(
    root: str | Path,
    *,
    kquant_relative: str = "kquant",
    expected_patch: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash the preserved source closure and dirty KQuant tree read-only."""

    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("prepared process root is absent, non-directory, or a symlink")
    kquant = (root / kquant_relative).resolve()
    try:
        kquant.relative_to(root)
    except ValueError as error:
        raise ValueError("KQuant root escapes the prepared process") from error
    if not (kquant / ".git").is_dir():
        raise ValueError("prepared process does not contain the expected KQuant checkout")

    head = str(_git(kquant, "rev-parse", "HEAD")).strip()
    if _REVISION.fullmatch(head) is None:
        raise ValueError("KQuant HEAD is not an immutable Git commit")
    status_text = str(
        _git(kquant, "status", "--porcelain=v1", "--untracked-files=all")
    )
    status = [line for line in status_text.splitlines() if line]
    patch = bytes(_git(kquant, "diff", "--binary", "HEAD", "--", binary=True))
    patch_sha = sha256_bytes(patch)
    dirty = bool(status)
    if dirty and not patch:
        raise ValueError(
            "dirty KQuant contains changes outside the tracked binary patch; provenance is incomplete"
        )
    if not dirty and patch:
        raise ValueError("KQuant patch/status disagree")
    expected_patch_sha: str | None = None
    if expected_patch is not None:
        expected = Path(expected_patch).resolve()
        if not expected.is_file() or expected.is_symlink():
            raise ValueError("expected KQuant patch is absent or is a symlink")
        expected_patch_sha = sha256_file(expected)
        if expected_patch_sha != patch_sha:
            raise ValueError("live KQuant binary patch differs from the preserved patch artifact")

    kquant_tree = _tree_identity(kquant, excluded_top_level={".git"})
    kquant_body = {
        "schema": KQUANT_SCHEMA,
        "source_root": str(kquant),
        "git_head": head,
        "dirty": dirty,
        "status_porcelain_v1": status,
        "status_sha256": sha256_bytes(status_text.encode()),
        "dirty_patch_sha256": patch_sha,
        "expected_patch_sha256": expected_patch_sha,
        "tree_sha256": kquant_tree["tree_sha256"],
        "file_count": kquant_tree["file_count"],
        "total_bytes": kquant_tree["total_bytes"],
        "git_optional_locks": False,
        "checkout_mutation_performed": False,
    }
    kquant_receipt = _seal(kquant_body, "receipt_sha256")

    prepared_tree = _tree_identity(
        root, excluded_top_level={".git", *_SELF_SEALS}
    )
    prepared_body = {
        "schema": PREPARED_TREE_SCHEMA,
        "source_root": str(root),
        "tree_sha256": prepared_tree["tree_sha256"],
        "file_count": prepared_tree["file_count"],
        "total_bytes": prepared_tree["total_bytes"],
        "excluded_top_level": sorted({".git", *_SELF_SEALS}),
        "kquant_relative": kquant_relative,
        "kquant_receipt_sha256": kquant_receipt["receipt_sha256"],
        "checkout_mutation_performed": False,
    }
    return _seal(prepared_body, "receipt_sha256"), kquant_receipt


def seal_sm100_extensions(
    *,
    preflight: Mapping[str, Any],
    exllama_checkout: str | Path,
    exllama_extension: str | Path,
    observed_kquant_extension: str | Path | None = None,
) -> dict[str, Any]:
    """Bind exact SM100 software bytes without claiming codec qualification."""

    preflight_sha = _verify_seal(
        preflight,
        schema=PREFLIGHT_SCHEMA,
        field="preflight_sha256",
        label="SM100 preflight",
    )
    if preflight.get("ready") is not True or preflight.get("workers") != 4:
        raise ValueError("SM100 extension provenance requires a ready four-worker preflight")
    capabilities = sorted(
        {
            str(row.get("compute_capability"))
            for row in preflight.get("gpus", ())
            if isinstance(row, Mapping)
        }
    )
    if not capabilities or any(not capability.startswith("10.") for capability in capabilities):
        raise ValueError("SM100 extension provenance requires four SM100-class devices")

    checkout = Path(exllama_checkout).resolve()
    extension = Path(exllama_extension).resolve()
    if not checkout.is_dir() or not (checkout / ".git").is_dir():
        raise ValueError("ExLlama checkout is absent")
    if not extension.is_file() or extension.is_symlink():
        raise ValueError("ExLlama extension is absent or is a symlink")
    head = str(_git(checkout, "rev-parse", "HEAD")).strip()
    status_text = str(
        _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    )
    observed: dict[str, Any] | None = None
    if observed_kquant_extension is not None:
        path = Path(observed_kquant_extension).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError("observed KQuant extension is absent or is a symlink")
        observed = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "qualification": "observed_only_not_authorized",
        }
    body = {
        "schema": SM100_EXTENSION_SCHEMA,
        "preflight_sha256": preflight_sha,
        "compute_capabilities": capabilities,
        "exllama": {
            "checkout": str(checkout),
            "git_head": head,
            "git_dirty": bool(status_text),
            "git_status_sha256": sha256_bytes(status_text.encode()),
            "extension_path": str(extension),
            "extension_bytes": extension.stat().st_size,
            "extension_sha256": sha256_file(extension),
        },
        "observed_kquant_extension": observed,
        "codec_marker_qualification": "unresolved_external_gate",
        "launch_authorized": False,
    }
    return _seal(body, "receipt_sha256")


def verify_prepared_process(
    prepared: Mapping[str, Any], kquant: Mapping[str, Any]
) -> tuple[str, str]:
    prepared_sha = _verify_seal(
        prepared,
        schema=PREPARED_TREE_SCHEMA,
        field="receipt_sha256",
        label="prepared process",
    )
    kquant_sha = _verify_seal(
        kquant,
        schema=KQUANT_SCHEMA,
        field="receipt_sha256",
        label="KQuant working tree",
    )
    if (
        prepared.get("kquant_receipt_sha256") != kquant_sha
        or prepared.get("checkout_mutation_performed") is not False
        or kquant.get("checkout_mutation_performed") is not False
    ):
        raise ValueError("prepared/KQuant provenance binding differs")
    return prepared_sha, kquant_sha
