"""Fail-closed contract for release-specific GLM campaign implementations."""

from __future__ import annotations

import importlib
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, runtime_checkable

from ..core.artifacts import canonical_json, sha256_bytes, sha256_file


PROVIDER_SCHEMA = "quant-pipeline.glm53-release-provider.v1"
REQUIRED_METHODS = (
    "discover",
    "capture_teacher_and_routes",
    "fit_streaming_statistics",
    "generate_exact_candidates",
    "measure_causal_attribution",
    "allocate_capacity_frontier",
    "install_and_reanchor",
    "optimize_mtp",
    "emit_checkpoint",
    "audit_runtime_reader",
    "capture_packed_student",
    "qualify_runtime",
    "qualify_behavioral_suite",
)


@runtime_checkable
class GlmReleaseProvider(Protocol):
    def identity(self) -> Mapping[str, Any]: ...
    def discover(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def capture_teacher_and_routes(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def fit_streaming_statistics(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def generate_exact_candidates(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def measure_causal_attribution(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def allocate_capacity_frontier(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def install_and_reanchor(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def optimize_mtp(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def emit_checkpoint(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def audit_runtime_reader(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def capture_packed_student(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def qualify_runtime(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def qualify_behavioral_suite(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


def load_release_provider(reference: str, config: Mapping[str, Any]) -> GlmReleaseProvider:
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("service_factory must use module:attribute")
    module_name, attribute = reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    provider = factory(config) if callable(factory) else factory
    validate_release_provider(provider, str(config.get("model_revision", "")))
    return provider


def validate_release_provider(provider: Any, model_revision: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("release provider requires an immutable model revision")
    if not callable(getattr(provider, "identity", None)):
        raise TypeError("release provider lacks identity()")
    missing = [name for name in REQUIRED_METHODS if not callable(getattr(provider, name, None))]
    if missing:
        raise TypeError(f"release provider lacks required methods: {missing}")
    identity = dict(provider.identity())
    required = {
        "schema",
        "model_revision",
        "source_closure",
        "source_closure_sha256",
        "routing_reference_sha256",
        "codec_identity_sha256",
        "checkpoint_emitter_sha256",
        "runtime_reader_sha256",
        "runtime_source_revision",
        "nvfp4_scale_artifact_sha256",
    }
    if required - set(identity):
        raise ValueError(f"release provider identity lacks fields: {sorted(required - set(identity))}")
    if identity["schema"] != PROVIDER_SCHEMA or identity["model_revision"] != model_revision:
        raise ValueError("release provider schema/model identity mismatch")
    for key in required - {"schema", "model_revision", "runtime_source_revision", "source_closure"}:
        if not isinstance(identity[key], str) or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None:
            raise ValueError(f"release provider {key} must be a 64-hex seal")
    if not isinstance(identity["runtime_source_revision"], str) or not identity["runtime_source_revision"]:
        raise ValueError("release provider runtime source revision is required")
    closure = identity["source_closure"]
    if not isinstance(closure, list) or not closure:
        raise ValueError("release provider source_closure must be a nonempty file list")
    for row in closure:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("release provider source closure row is malformed")
        if not isinstance(row["path"], str) or not row["path"]:
            raise ValueError("release provider source closure path is malformed")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("release provider source closure byte count is malformed")
        if not isinstance(row["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
            raise ValueError("release provider source closure hash is malformed")
    if sha256_bytes(canonical_json(closure)) != identity["source_closure_sha256"]:
        raise ValueError("release provider source closure seal mismatch")
    if identity.get("routing_verified_against_reference") is not True:
        raise ValueError("release provider routing has not been verified against the released reference")
    if identity.get("packed_reader_full_graph_capable") is not True:
        raise ValueError("release provider does not declare a full-graph-capable packed reader")
    return identity | {"identity_sha256": sha256_bytes(canonical_json(identity))}


def provider_source_receipt(provider: Any) -> dict[str, Any]:
    identity = dict(provider.identity())
    rows = []
    for expected in identity.get("source_closure", []):
        path = Path(expected["path"]).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"release provider source is not a regular file: {path}")
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise ValueError(f"release provider source identity drift: {path}")
        rows.append(actual)
    return {"files": rows, "sha256": sha256_bytes(canonical_json(rows))}
