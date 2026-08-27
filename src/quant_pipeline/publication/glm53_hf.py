from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file


PLAN_SCHEMA = "quant-pipeline.glm53-hf-publication-plan.v1"
CONFIG_SCHEMA = "quant-pipeline.glm53-hf-publication-config.v1"
SOURCE_SCHEMA = "quant-pipeline.glm53-source-model.v1"
RECIPE_SCHEMA = "quant-pipeline.glm53-uniform-quant-recipe.v1"
SOURCE_MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
SOURCE_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
UPLOAD_EXCLUDES = (
    ".cache/**",
    ".git/**",
    "**/*.tmp",
    "**/*.part",
    "PUBLICATION_RECEIPT.json",
)


DATASET_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "dataset_card": ("README.md",),
    "source_provenance": ("provenance/source-model-revision.json",),
    "teacher_float32_logits": ("teacher/logits/**",),
    "k4_float32_student_logits": ("students/k4/logits/**",),
    "k6_float32_student_logits": ("students/k6/logits/**",),
    "tokenwise_kld": ("metrics/tokenwise-kld/**", "metrics/tokenwise_kld/**"),
    "hessians": ("statistics/hessians/**",),
    "covariances": ("statistics/covariances/**",),
    "calibration_tensors": ("calibration/tensors/**",),
    "calibration_windows": ("calibration/windows/**",),
    "shapley_artifacts": ("attribution/shapley/**",),
    "route_artifacts": ("routing/**",),
    "inventories": ("inventories/**",),
    "receipts": ("receipts/**",),
    "logs": ("logs/**",),
}
DATASET_GATE_RECEIPTS = {
    "teacher": "receipts/teacher-backend.json",
    "k4": "receipts/k4-packed-kld.json",
    "k6": "receipts/k6-packed-kld.json",
}

MODEL_REQUIRED_EXACT = (
    "README.md",
    "config.json",
    "provenance/source-model-revision.json",
    "quantization/recipe.json",
    "receipts/checkpoint.json",
)
MODEL_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "weights/**")


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or symlinked: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_receipt_seal(value: dict[str, Any], label: str) -> str:
    seal = value.get("receipt_sha256")
    if not isinstance(seal, str) or not HEX64.fullmatch(seal):
        raise ValueError(f"{label} lacks a receipt_sha256 seal")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if seal != _hash_json(body):
        raise ValueError(f"{label} receipt seal mismatch")
    return seal


def _normal_repo_id(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"invalid Hugging Face repository id: {value!r}")
    return value


def _normal_manifest_path(value: str) -> str:
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or ".." in parsed.parts
        or parsed.as_posix() != value
        or "\\" in value
        or value in {"MANIFEST.json", "SHA256SUMS"}
    ):
        raise ValueError(f"invalid artifact-tree path: {value!r}")
    return value


def _validate_sealed_tree(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"publication root is missing or symlinked: {root}")
    manifest_path = root / "MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    manifest = _read_json(manifest_path, "artifact-tree manifest")
    if manifest.get("schema") != "quant-pipeline.artifact-tree-manifest.v1":
        raise ValueError(f"unsupported artifact-tree manifest in {root}")
    rows = list(manifest.get("files", ()))
    if int(manifest.get("file_count", -1)) != len(rows):
        raise ValueError(f"manifest file count mismatch in {root}")
    if int(manifest.get("total_bytes", -1)) != sum(int(row["bytes"]) for row in rows):
        raise ValueError(f"manifest byte count mismatch in {root}")
    seal = manifest.get("manifest_sha256")
    if not isinstance(seal, str) or not HEX64.fullmatch(seal):
        raise ValueError(f"manifest lacks a SHA-256 seal in {root}")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if seal != _hash_json(body):
        raise ValueError(f"manifest seal mismatch in {root}")

    expected_paths: set[str] = set()
    for row in rows:
        relative = _normal_manifest_path(str(row["path"]))
        if relative in expected_paths:
            raise ValueError(f"duplicate sealed path in {root}: {relative}")
        expected_paths.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"sealed file is missing or symlinked: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"sealed file size changed: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"sealed file hash changed: {path}")

    expected_sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in rows
    ).encode()
    if not sums_path.is_file() or sums_path.is_symlink() or sums_path.read_bytes() != expected_sums:
        raise ValueError(f"SHA256SUMS differs from sealed manifest in {root}")

    observed: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        if path.is_symlink():
            raise ValueError(f"publication tree contains symlink: {path}")
        if path.is_file():
            observed.add(relative)
    expected = expected_paths | {"MANIFEST.json", "SHA256SUMS"}
    if observed != expected:
        raise ValueError(
            f"publication tree is not closed: {root}; "
            f"missing={sorted(expected - observed)[:5]}, extra={sorted(observed - expected)[:5]}"
        )
    return {
        "root": str(root),
        "manifest_sha256": seal,
        "manifest_file_sha256": sha256_file(manifest_path),
        "sha256sums_file_sha256": sha256_file(sums_path),
        "file_count": len(rows) + 2,
        "payload_bytes": int(manifest["total_bytes"]),
        "total_bytes": int(manifest["total_bytes"])
        + manifest_path.stat().st_size
        + sums_path.stat().st_size,
        "paths": sorted(expected_paths),
    }


def _matches_any(paths: set[str], patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for path in paths for pattern in patterns)


def _validate_source_receipt(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "provenance/source-model-revision.json"
    receipt = _read_json(path, "source-model receipt")
    if receipt.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unsupported source-model receipt schema in {root}")
    if receipt.get("model_id") != SOURCE_MODEL_ID:
        raise ValueError(f"source-model receipt identifies the wrong model in {root}")
    if receipt.get("revision") != SOURCE_REVISION:
        raise ValueError(f"source-model receipt is not pinned to the approved BF16 revision in {root}")
    if receipt.get("weight_dtype") != "bfloat16":
        raise ValueError(f"source-model receipt must declare bfloat16 weights in {root}")
    if receipt.get("example_only") is not False:
        raise ValueError(f"source-model receipt remains example-only in {root}")
    _require_receipt_seal(receipt, f"source-model receipt in {root}")
    return receipt, sha256_file(path)


def _validate_dataset(root: Path, tree: dict[str, Any]) -> dict[str, Any]:
    paths = set(tree["paths"])
    missing = [
        name for name, patterns in DATASET_REQUIREMENTS.items()
        if not _matches_any(paths, patterns)
    ]
    if missing:
        raise ValueError(f"dataset publication is missing required artifact roles: {missing}")
    teacher_files = [path for path in paths if fnmatch.fnmatchcase(path, "teacher/logits/**")]
    student_files = [
        path for path in paths
        if fnmatch.fnmatchcase(path, "students/k4/logits/**")
        or fnmatch.fnmatchcase(path, "students/k6/logits/**")
    ]
    if not all(path.endswith((".npy", ".npz", ".safetensors")) for path in teacher_files + student_files):
        raise ValueError("teacher and student logit payloads must use a typed tensor format")
    receipt_seals: dict[str, str] = {}
    for profile, relative in DATASET_GATE_RECEIPTS.items():
        if relative not in paths:
            raise ValueError(f"dataset publication is missing mandatory gate receipt: {relative}")
        receipt = _read_json(root / relative, f"{profile} publication gate receipt")
        if receipt.get("example_only") is not False or receipt.get("qualified") is not True:
            raise ValueError(f"{profile} publication gate receipt is not qualified")
        if receipt.get("source_revision") != SOURCE_REVISION:
            raise ValueError(f"{profile} publication gate receipt has the wrong source revision")
        if profile == "teacher":
            if receipt.get("logits_dtype") != "float32":
                raise ValueError("teacher publication gate receipt must declare float32 logits")
        elif receipt.get("profile") != profile:
            raise ValueError(f"{profile} publication gate receipt has the wrong profile")
        receipt_seals[profile] = _require_receipt_seal(
            receipt, f"{profile} publication gate receipt"
        )
    return {
        "artifact_roles": sorted(DATASET_REQUIREMENTS),
        "teacher_logit_files": len(teacher_files),
        "student_logit_files": len(student_files),
        "gate_receipt_seals": receipt_seals,
    }


def _validate_model(root: Path, tree: dict[str, Any], *, profile: str, bits: int, tp: int) -> dict[str, Any]:
    paths = set(tree["paths"])
    missing = [path for path in MODEL_REQUIRED_EXACT if path not in paths]
    if missing:
        raise ValueError(f"{profile} model publication is missing required files: {missing}")
    if not _matches_any(paths, MODEL_WEIGHT_PATTERNS):
        raise ValueError(f"{profile} model publication contains no packed weight payload")
    recipe = _read_json(root / "quantization/recipe.json", f"{profile} quantization recipe")
    expected = {
        "schema": RECIPE_SCHEMA,
        "profile": profile,
        "routed_expert_bits": bits,
        "tensor_policy": "uniform-routed-experts",
        "nonrouted_policy": "native",
        "target_tensor_parallel": tp,
        "source_model_id": SOURCE_MODEL_ID,
        "source_revision": SOURCE_REVISION,
        "example_only": False,
    }
    drift = {key: (recipe.get(key), value) for key, value in expected.items() if recipe.get(key) != value}
    if drift:
        raise ValueError(f"{profile} quantization recipe violates the publication contract: {drift}")
    checkpoint = _read_json(root / "receipts/checkpoint.json", f"{profile} checkpoint receipt")
    if checkpoint.get("example_only") is not False or checkpoint.get("qualified") is not True:
        raise ValueError(f"{profile} checkpoint receipt is not qualified")
    if checkpoint.get("source_revision") != SOURCE_REVISION:
        raise ValueError(f"{profile} checkpoint receipt has the wrong source revision")
    if checkpoint.get("profile") != profile:
        raise ValueError(f"{profile} checkpoint receipt has the wrong profile")
    checkpoint_seal = _require_receipt_seal(checkpoint, f"{profile} checkpoint receipt")
    return {
        "profile": profile,
        "routed_expert_bits": bits,
        "target_tensor_parallel": tp,
        "recipe_sha256": sha256_file(root / "quantization/recipe.json"),
        "checkpoint_receipt_sha256": sha256_file(root / "receipts/checkpoint.json"),
        "checkpoint_receipt_seal": checkpoint_seal,
    }


def _target_plan(
    *,
    name: str,
    config: dict[str, Any],
    repo_type: str,
    tree: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    repo_id = _normal_repo_id(str(config["repo_id"]))
    root = Path(tree["root"])
    receipt = Path(str(config["verification_receipt"])).expanduser().resolve()
    if receipt == root or root in receipt.parents:
        raise ValueError(f"{name} verification receipt must be outside the sealed upload root")
    workers = int(config.get("num_workers", 8))
    if workers < 1 or workers > 64:
        raise ValueError(f"{name} num_workers must be within [1, 64]")
    command = [
        "hf",
        "upload-large-folder",
        repo_id,
        str(root),
        "--repo-type",
        repo_type,
        "--revision",
        "main",
        "--include",
        "*",
    ]
    for pattern in UPLOAD_EXCLUDES:
        command.extend(["--exclude", pattern])
    command.extend(["--num-workers", str(workers)])
    return {
        "name": name,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_root": str(root),
        "verification_receipt": str(receipt),
        "tree": {key: value for key, value in tree.items() if key != "paths"},
        "validation": validation,
        "upload": {
            "strategy": "hf-upload-large-folder-resumable",
            "revision": "main",
            "allow_patterns": ["*"],
            "ignore_patterns": list(UPLOAD_EXCLUDES),
            "command": command,
        },
    }


def build_plan(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _read_json(config_path, "GLM-5.3 publication config")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported GLM-5.3 publication config schema")
    serialized = canonical_json(config).decode()
    if "__REQUIRED_" in serialized or "__ABSOLUTE_" in serialized:
        raise ValueError("publication config contains unresolved placeholders")
    source = config.get("source", {})
    if source.get("model_id") != SOURCE_MODEL_ID or source.get("revision") != SOURCE_REVISION:
        raise ValueError("publication config is not pinned to the approved official BF16 source")
    if not HEX40.fullmatch(str(source.get("revision", ""))):
        raise ValueError("source revision must be an immutable 40-hex commit")
    final_visibility = str(config.get("final_visibility", "private"))
    if final_visibility not in {"private", "public"}:
        raise ValueError("final_visibility must be private or public")

    targets = config.get("targets", {})
    if set(targets) != {"k4", "k6", "dataset"}:
        raise ValueError("publication config must define exactly k4, k6, and dataset targets")
    roots = {name: Path(str(row["local_root"])).expanduser().resolve() for name, row in targets.items()}
    if len(set(roots.values())) != 3:
        raise ValueError("each publication target must have a distinct local root")
    trees = {name: _validate_sealed_tree(root) for name, root in roots.items()}
    source_receipts = {name: _validate_source_receipt(root)[1] for name, root in roots.items()}
    if len(set(source_receipts.values())) != 1:
        raise ValueError("all publication trees must contain byte-identical source provenance")

    validations = {
        "k4": _validate_model(roots["k4"], trees["k4"], profile="k4", bits=4, tp=2),
        "k6": _validate_model(roots["k6"], trees["k6"], profile="k6", bits=6, tp=4),
        "dataset": _validate_dataset(roots["dataset"], trees["dataset"]),
    }
    plan_targets = [
        _target_plan(
            name=name,
            config=targets[name],
            repo_type="dataset" if name == "dataset" else "model",
            tree=trees[name],
            validation=validations[name],
        )
        for name in ("k4", "k6", "dataset")
    ]
    repo_pairs = {(row["repo_type"], row["repo_id"]) for row in plan_targets}
    if len(repo_pairs) != 3:
        raise ValueError("K4, K6, and dataset publication targets must be separate repositories")

    total_bytes = sum(int(row["tree"]["total_bytes"]) for row in plan_targets)
    reserve_bytes = max(50 * 1024**3, (total_bytes + 9) // 10)
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source": {
            "model_id": SOURCE_MODEL_ID,
            "revision": SOURCE_REVISION,
            "weight_dtype": "bfloat16",
            "source_receipt_sha256": next(iter(source_receipts.values())),
        },
        "targets": plan_targets,
        "publication_order": ["k4", "k6", "dataset"],
        "visibility": {
            "upload_initially": "private",
            "release_after_all_remote_verifications": final_visibility,
        },
        "disk_guidance": {
            "sealed_upload_bytes": total_bytes,
            "sealed_upload_gib": total_bytes / 1024**3,
            "minimum_free_reserve_bytes": reserve_bytes,
            "minimum_free_reserve_gib": reserve_bytes / 1024**3,
            "note": "upload-large-folder resumes from local .cache/huggingface state; retain roots and cache until immutable revisions verify",
        },
        "gates": {
            "closed_local_namespaces": True,
            "local_sha256_verified": True,
            "official_source_revision_pinned": True,
            "source_provenance_identical": True,
            "uniform_k4_tp2_recipe": True,
            "uniform_k6_tp4_recipe": True,
            "dataset_role_closure": True,
            "remote_exact_namespace_verification_required": True,
            "credentials_embedded": False,
        },
    }
    plan["plan_sha256"] = _hash_json(plan)
    return plan
