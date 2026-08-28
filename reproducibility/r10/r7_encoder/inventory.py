"""Content-addressed checkpoint and numerical-environment inventories.

The owner run is gated on these manifests.  Paths, mtimes, and index hashes are
not substitutes for payload identity: every indexed tensor is hashed from its
raw safetensors byte range before any encoder stage may consume it.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Mapping

from .constants import (
    CUBLAS_WORKSPACE_POLICY,
    EXPERTS_IMPLEMENTATION,
    FIRST_MOE_LAYER,
    HIDDEN_SIZE,
    HUB_KERNEL_POLICY,
    INTERMEDIATE_SIZE,
    LAST_MOE_LAYER,
    LDL_FACTORIZATION_POLICY,
    MCG_MULT,
    NUM_EXPERTS,
    PROJECTIONS,
    RECIPE_MARKER,
    RECIPE_VERSION,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    configure_deterministic_environment,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .safetensors_io import SafeTensorReader

INDEX = "model.safetensors.index.json"
ROUTED = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def _seal_payload(payload: Mapping[str, object]) -> dict[str, object]:
    value = dict(payload)
    value["inventory_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _validate_routed_bf16(entries: Mapping[str, Mapping[str, object]]) -> None:
    required: set[str] = set()
    for layer in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1):
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                required.add(name)
                try:
                    record = entries[name]
                except KeyError as exc:
                    raise ValueError(f"BF16 source is missing {name}") from exc
                expected = (
                    [INTERMEDIATE_SIZE, HIDDEN_SIZE]
                    if projection != "down_proj"
                    else [HIDDEN_SIZE, INTERMEDIATE_SIZE]
                )
                if record.get("dtype") != "BF16" or record.get("shape") != expected:
                    raise ValueError(
                        f"{name}: expected BF16 {expected}, got "
                        f"{record.get('dtype')} {record.get('shape')}"
                    )
    observed = {
        name
        for name in entries
        if ROUTED.match(name)
        and int(ROUTED.match(name).group(1))
        in range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)
    }  # type: ignore[union-attr]
    if observed != required:
        extra = sorted(observed - required)
        raise ValueError(f"unexpected routed-expert source names: {extra[:3]}")


def build_checkpoint_inventory(
    checkpoint: str | Path,
    output: str | Path,
    *,
    role: str,
    require_routed_bf16: bool = False,
) -> dict[str, object]:
    """Hash an indexed checkpoint, checkpointing progress after each shard.

    A resumed shard is trusted only after its whole-file hash is rechecked.
    The final seal covers index/config bytes, shard files, headers, every raw
    tensor payload, dtype, shape, and tensor-to-shard mapping.
    """

    root = Path(checkpoint).resolve()
    destination = Path(output).resolve()
    if destination == root or root in destination.parents:
        raise ValueError(
            "checkpoint inventories must be written outside read-only models"
        )
    index_path = root / INDEX
    config_path = root / "config.json"
    index = json.loads(index_path.read_text())
    config = json.loads(config_path.read_text())
    if require_routed_bf16 and config.get("quantization_config") not in (None, {}):
        raise ValueError(
            "BF16 source config declares quantization metadata; original-source "
            "provenance must be resolved before Round 7"
        )
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    base = {
        "marker": RECIPE_MARKER,
        "recipe_version": RECIPE_VERSION,
        "schema": "r7-checkpoint-inventory-v1",
        "role": role,
        "checkpoint": str(root),
        "index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(config_path),
    }
    progress_path = destination.with_name(f".{destination.name}.progress")
    progress: dict[str, object] = {**base, "shards": {}, "entries": {}}
    if progress_path.exists():
        prior = read_json(progress_path)
        if any(prior.get(key) != value for key, value in base.items()):
            raise ValueError("checkpoint inventory resume binding drift")
        progress = prior
    shards = progress["shards"]
    entries = progress["entries"]
    if not isinstance(shards, dict) or not isinstance(entries, dict):
        raise ValueError("malformed checkpoint inventory progress")

    expected_by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        expected_by_shard.setdefault(shard, set()).add(name)
    for shard_name in sorted(expected_by_shard):
        shard_path = root / shard_name
        incumbent = shards.get(shard_name)
        current_file_hash = sha256_file(shard_path)
        if (
            isinstance(incumbent, dict)
            and incumbent.get("file_sha256") == current_file_hash
        ):
            continue
        # Never retain records from a changed or interrupted shard.
        for name in tuple(entries):
            if (
                isinstance(entries[name], dict)
                and entries[name].get("shard") == shard_name
            ):
                del entries[name]
        reader = SafeTensorReader(shard_path)
        if set(reader.tensors) != expected_by_shard[shard_name]:
            missing = sorted(expected_by_shard[shard_name] - set(reader.tensors))
            extra = sorted(set(reader.tensors) - expected_by_shard[shard_name])
            raise ValueError(
                f"index/shard mismatch in {shard_name}: missing={missing[:3]} extra={extra[:3]}"
            )
        for name in sorted(reader.tensors):
            info = reader.tensors[name]
            entries[name] = {
                "shard": shard_name,
                "dtype": info.dtype,
                "shape": list(info.shape),
                "nbytes": info.nbytes,
                "payload_start": info.payload.start,
                "payload_end": info.payload.end,
                "payload_sha256": info.payload.sha256(),
            }
        shards[shard_name] = {
            "file_sha256": current_file_hash,
            "header_sha256": reader.header_sha256,
            "size": shard_path.stat().st_size,
            "tensor_count": len(reader.tensors),
        }
        atomic_write_json(progress_path, progress)

    if set(entries) != set(weight_map):
        raise ValueError("checkpoint inventory does not cover the exact index")
    if require_routed_bf16:
        _validate_routed_bf16(entries)  # type: ignore[arg-type]
    indexed_shards = set(weight_map.values())
    auxiliary_files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in indexed_shards
    }
    final = _seal_payload(
        {
            **base,
            "shards": {key: shards[key] for key in sorted(shards)},
            "entries": {key: entries[key] for key in sorted(entries)},
            "tensor_count": len(entries),
            "routed_bf16_validated": require_routed_bf16,
            "auxiliary_files_sha256": auxiliary_files,
        }
    )
    atomic_write_json(destination, final)
    progress_path.unlink(missing_ok=True)
    return final


def load_checkpoint_inventory(
    path: str | Path, *, role: str | None = None, require_routed_bf16: bool = False
) -> dict[str, object]:
    payload = read_json(path)
    digest = payload.pop("inventory_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("checkpoint inventory seal mismatch")
    payload["inventory_sha256"] = digest
    if (
        payload.get("marker") != RECIPE_MARKER
        or payload.get("schema") != "r7-checkpoint-inventory-v1"
    ):
        raise ValueError("foreign checkpoint inventory")
    if role is not None and payload.get("role") != role:
        raise ValueError("checkpoint inventory role mismatch")
    if require_routed_bf16:
        if not payload.get("routed_bf16_validated"):
            raise ValueError("BF16 source inventory was not validated")
        _validate_routed_bf16(payload["entries"])  # type: ignore[arg-type]
    return payload


def verify_checkpoint_inventory(
    checkpoint: str | Path, payload: Mapping[str, object]
) -> None:
    """Re-hash every sealed checkpoint file before an owner run."""

    root = Path(checkpoint).resolve()
    if str(root) != payload.get("checkpoint"):
        raise ValueError("checkpoint path differs from sealed inventory")
    if sha256_file(root / INDEX) != payload.get("index_sha256"):
        raise ValueError("checkpoint index changed after inventory")
    if sha256_file(root / "config.json") != payload.get("config_sha256"):
        raise ValueError("checkpoint config changed after inventory")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, dict):
        raise ValueError("checkpoint inventory lacks shard map")
    def _check_shard(item):
        shard_name, record = item
        if not isinstance(record, dict) or sha256_file(root / shard_name) != record.get(
            "file_sha256"
        ):
            raise ValueError(f"checkpoint shard changed after inventory: {shard_name}")

    # Hash shards in parallel; identical failure semantics (any mismatch
    # raises), deterministic because each check is independent and the full
    # set must pass. Turns a ~25-minute single-thread startup verify into a
    # few minutes on every launch and resume.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=16) as _pool:
        for _ in _pool.map(_check_shard, sorted(raw_shards.items())):
            pass
    auxiliary = payload.get("auxiliary_files_sha256")
    if not isinstance(auxiliary, dict) or not auxiliary:
        raise ValueError("checkpoint inventory lacks auxiliary/tokenizer assets")
    indexed_shards = set(raw_shards)
    observed_auxiliary = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in indexed_shards
    }
    if observed_auxiliary != auxiliary:
        raise ValueError(
            "checkpoint auxiliary/tokenizer assets changed after inventory"
        )


def _dynamic_library_closure(binaries: set[Path]) -> dict[str, str]:
    """Hash the resolved ELF dependency closure without initializing CUDA."""

    dependencies: set[Path] = set()
    for binary in sorted(binaries):
        result = subprocess.run(
            ("ldd", str(binary)),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cannot seal dynamic dependencies for {binary}")
        for raw in result.stdout.splitlines():
            line = raw.strip()
            if "not found" in line:
                raise RuntimeError(f"unresolved dynamic dependency: {line}")
            candidate = ""
            if "=>" in line:
                candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
            elif line.startswith("/"):
                candidate = line.split(" ", 1)[0]
            if candidate.startswith("/"):
                path = Path(candidate).resolve()
                if path.is_file():
                    dependencies.add(path)
    return {str(path): sha256_file(path) for path in sorted(dependencies)}


def _torch_binary_closure(torch_module) -> tuple[dict[str, str], set[Path]]:
    package_root = Path(torch_module.__file__).resolve().parent
    candidates: set[Path] = set()
    library_root = package_root / "lib"
    if library_root.is_dir():
        for child in library_root.rglob("*"):
            if child.is_file() and (
                ".so" in child.name
                or child.suffix.lower() in {".dll", ".dylib", ".pyd"}
            ):
                candidates.add(child.resolve())
    torch_c = Path(str(getattr(torch_module._C, "__file__", ""))).resolve()
    if not torch_c.is_file():
        raise RuntimeError("Torch numerical extension path is unavailable")
    candidates.add(torch_c)
    if not candidates:
        raise RuntimeError("Torch numerical binary closure is empty")
    return (
        {str(path): sha256_file(path) for path in sorted(candidates)},
        candidates,
    )


def _nvidia_device_record(device: str) -> dict[str, object]:
    match = re.fullmatch(r"cuda:(\d+)", str(device))
    if match is None:
        raise ValueError("numeric platform device must be an explicit cuda:N")
    logical_index = int(match.group(1))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        selector = str(logical_index)
        visibility_mode = "unset"
        visible_count: int | str = "host-default"
    else:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if logical_index >= len(tokens) or tokens[logical_index] == "-1":
            raise ValueError("logical CUDA device is outside CUDA_VISIBLE_DEVICES")
        selector = tokens[logical_index]
        visibility_mode = "explicit"
        visible_count = len(tokens)
    fields = (
        "name",
        "compute_cap",
        "memory.total",
        "driver_version",
        "vbios_version",
        "pci.bus_id",
    )
    result = subprocess.run(
        (
            "nvidia-smi",
            "-i",
            selector,
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi could not seal the selected CUDA platform")
    rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise RuntimeError("nvidia-smi returned an ambiguous CUDA platform record")
    values = [value.strip() for value in rows[0]]
    if any(not value or value.upper() == "N/A" for value in values):
        raise RuntimeError("selected CUDA platform record is incomplete")
    return {
        "logical_device": str(device),
        "logical_index": logical_index,
        "visibility_mode": visibility_mode,
        "visible_device_count": visible_count,
        "selector_kind": "index" if selector.isdigit() else "stable-token",
        "gpu": dict(zip(fields, values, strict=True)),
    }


def _collect_execution_platform(
    *, device: str, extension: Path
) -> dict[str, object]:
    """Collect exact numerical binaries and GPU/driver identity read-only."""

    configure_deterministic_environment()
    import torch

    torch_binaries, binary_paths = _torch_binary_closure(torch)
    extension_path = extension.resolve()
    dynamic_roots = set(binary_paths)
    dynamic_roots.add(extension_path)
    payload: dict[str, object] = {
        "schema": "r7-execution-platform-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "UNKNOWN")),
        "torch_cuda_build": str(torch.version.cuda),
        # cuDNN build parsed from static build info: torch.backends.cudnn
        # .version() initializes CUDA, which would trip the runtime's
        # CUDA-purity gate 26 minutes later at walk startup.
        "torch_cudnn_build": (
            (re.search(r"CuDNN[^\n]*?([\d.]+)", torch.__config__.show())
             or [None, "UNKNOWN"])[1]
        ),
        "device": _nvidia_device_record(device),
        "torch_numerical_binaries_sha256": torch_binaries,
        "dynamic_dependency_closure_sha256": _dynamic_library_closure(dynamic_roots),
    }
    payload["execution_platform_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def build_numeric_environment_inventory(
    output: str | Path,
    *,
    numeric_core: str | Path,
    extension: str | Path,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Seal numerical code, binaries, dependencies, GPU, and driver identity."""

    configure_deterministic_environment()
    core = Path(numeric_core).resolve()
    binary = Path(extension).resolve()
    packages = {}
    for name in ("torch", "numpy", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "ABSENT"
    try:
        import torch

        torch_cuda_build = str(torch.version.cuda)
    except ImportError:
        torch_cuda_build = "ABSENT"
    payload = _seal_payload(
        {
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "schema": "r7-numeric-environment-v2",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "torch_cuda_build": torch_cuda_build,
            "mcg_codebook": f"0x{MCG_MULT:08X}",
            "ldl_factorization_policy": LDL_FACTORIZATION_POLICY,
            "cublas_workspace_config": CUBLAS_WORKSPACE_POLICY,
            "numeric_core": str(core),
            "numeric_core_sha256": sha256_file(core),
            "extension": str(binary),
            "extension_sha256": sha256_file(binary),
            "execution_platform": _collect_execution_platform(
                device=device, extension=binary
            ),
        }
    )
    atomic_write_json(output, payload)
    return payload


def load_numeric_environment(
    path: str | Path, *, verify_files: bool = True
) -> dict[str, object]:
    configure_deterministic_environment()
    payload = read_json(path)
    digest = payload.pop("inventory_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("numeric environment seal mismatch")
    if verify_files and sha256_file(payload["numeric_core"]) != payload["numeric_core_sha256"]:  # type: ignore[arg-type]
        raise ValueError("numeric core changed after inventory")
    if verify_files and sha256_file(payload["extension"]) != payload["extension_sha256"]:  # type: ignore[arg-type]
        raise ValueError("TRELLIS extension changed after inventory")
    if payload.get("schema") != "r7-numeric-environment-v2":
        raise ValueError("numeric environment schema is not platform-sealed v2")
    if (
        payload.get("python") != platform.python_version()
        or payload.get("platform") != platform.platform()
    ):
        raise ValueError("Python/platform numeric environment drift")
    current_packages = {}
    for name in ("torch", "numpy", "safetensors"):
        try:
            current_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            current_packages[name] = "ABSENT"
    if payload.get("packages") != current_packages:
        raise ValueError("numeric package version drift")
    try:
        import torch

        cuda_build = str(torch.version.cuda)
    except ImportError:
        cuda_build = "ABSENT"
    if payload.get("torch_cuda_build") != cuda_build:
        raise ValueError("Torch CUDA build drift")
    if payload.get("mcg_codebook") != f"0x{MCG_MULT:08X}":
        raise ValueError("numeric codebook marker drift")
    if payload.get("ldl_factorization_policy") != LDL_FACTORIZATION_POLICY:
        raise ValueError("numeric LDL factorization policy drift")
    if payload.get("cublas_workspace_config") != CUBLAS_WORKSPACE_POLICY:
        raise ValueError("numeric cuBLAS workspace policy drift")
    expected_platform = payload.get("execution_platform")
    if not isinstance(expected_platform, dict):
        raise ValueError("numeric environment lacks an execution platform")
    platform_digest = expected_platform.get("execution_platform_sha256")
    unsigned_platform = dict(expected_platform)
    unsigned_platform.pop("execution_platform_sha256", None)
    if platform_digest != sha256_bytes(canonical_json_bytes(unsigned_platform)):
        raise ValueError("execution platform seal mismatch")
    device_record = expected_platform.get("device")
    if not isinstance(device_record, dict) or not isinstance(
        device_record.get("logical_device"), str
    ):
        raise ValueError("execution platform lacks a logical CUDA device")
    # Inventory consumers that explicitly disable file verification still
    # validate the sealed platform record and every static version/policy
    # above, but must not probe a live CUDA device or load the extension.  The
    # R10 parent/worker path uses this mode because its owner run is separately
    # platform-gated before workers are launched.
    if verify_files:
        current_platform = _collect_execution_platform(
            device=str(device_record["logical_device"]),
            extension=Path(str(payload["extension"])),
        )
        if current_platform != expected_platform:
            raise ValueError("numerical execution platform drift")
    payload["inventory_sha256"] = digest
    return payload


def build_runtime_code_inventory(
    output: str | Path, *, files: list[str | Path]
) -> dict[str, object]:
    expanded: set[Path] = set()
    for supplied in files:
        path = Path(supplied).resolve()
        if path.is_dir():
            expanded.update(
                child
                for child in path.rglob("*")
                if child.is_file()
                and ".git" not in child.parts
                and "__pycache__" not in child.parts
                and child.suffix not in (".pyc", ".pyo")
            )
        elif path.is_file():
            expanded.add(path)
        else:
            raise FileNotFoundError(path)
    resolved = sorted(str(path) for path in expanded)
    if not resolved:
        raise ValueError("runtime code inventory requires at least one source file")
    payload = _seal_payload(
        {
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "schema": "r7-runtime-code-inventory-v1",
            "execution_contract": {
                "cublas_workspace_config": CUBLAS_WORKSPACE_POLICY,
                "use_hub_kernels": "0",
                "experts_implementation": EXPERTS_IMPLEMENTATION,
                "hub_kernel_policy": HUB_KERNEL_POLICY,
                "deterministic_algorithms": True,
                "warn_only": False,
            },
            "files_sha256": {path: sha256_file(path) for path in resolved},
        }
    )
    atomic_write_json(output, payload)
    return payload


def load_runtime_code_inventory(
    path: str | Path, *, verify_files: bool = True
) -> dict[str, object]:
    payload = read_json(path)
    digest = payload.pop("inventory_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("runtime code inventory seal mismatch")
    if (
        payload.get("marker") != RECIPE_MARKER
        or payload.get("schema") != "r7-runtime-code-inventory-v1"
    ):
        raise ValueError("foreign runtime code inventory")
    files = payload.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("runtime code inventory is empty")
    if verify_files:
        for source, expected in files.items():
            if sha256_file(source) != expected:
                raise ValueError(f"runtime model code changed after inventory: {source}")
    expected_contract = {
        "cublas_workspace_config": CUBLAS_WORKSPACE_POLICY,
        "use_hub_kernels": "0",
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "hub_kernel_policy": HUB_KERNEL_POLICY,
        "deterministic_algorithms": True,
        "warn_only": False,
    }
    if payload.get("execution_contract") != expected_contract:
        raise ValueError("runtime deterministic execution contract drift")
    payload["inventory_sha256"] = digest
    return payload
