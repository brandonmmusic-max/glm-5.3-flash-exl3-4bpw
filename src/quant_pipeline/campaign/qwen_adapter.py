"""Concrete Qwen campaign adapter for the generic causal runner.

Scientific engines are dependency-injected, but the orchestration, artifact
bindings, fail-closed validation, and production resource checks are concrete.
The default adapter loads a package/local service factory named in the sealed
adapter configuration; it never executes shell placeholders or substitutes a
uniform/reference codec.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import inspect
import json
import os
import platform
import shutil
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ..core.artifacts import atomic_write, canonical_json, sha256_bytes, sha256_file, write_json
from .runner import StageRequest, StageResult


ADAPTER_SCHEMA = "quant-pipeline.qwen-production-adapter.v1"
STAGE_MANIFEST_SCHEMA = "quant-pipeline.qwen-stage-artifacts.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_SCIENTIFIC_CONTRACT = {
    "normalization": "source-derived-absolute-v31",
    "gss": "per-matrix-selected-bit-k3-k4-k5",
    "transform_search": "additive-ablation-against-v31-baseline",
    "allocation_objective": "signed-shapley-fisher-provisional-ratio-v1-with-proxy-control",
    "candidate_payloads": "exact-packed-vectors-reconstruction",
    "checkpoint": "upstream-btx-atoms-v1-pinned",
}


@runtime_checkable
class RouteCaptureProvider(Protocol):
    def capture_teacher(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def capture_routes(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class FitterProvider(Protocol):
    def fit(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class LedgerProvider(Protocol):
    def generate(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class CodecProvider(Protocol):
    def identity(self) -> Mapping[str, Any]: ...
    def install(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class EvaluatorProvider(Protocol):
    def attribute(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def confirm(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def reanchor(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def capture_student(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def final_kld(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class AllocatorProvider(Protocol):
    def allocate(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class CheckpointProvider(Protocol):
    def emit(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def audit(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class QwenCampaignServices:
    capturer: RouteCaptureProvider
    fitter: FitterProvider
    ledger: LedgerProvider
    codec: CodecProvider
    evaluator: EvaluatorProvider
    allocator: AllocatorProvider
    checkpoint: CheckpointProvider


def _load_object(reference: str) -> Any:
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("provider reference must be module:attribute")
    module, attribute = reference.split(":", 1)
    value = getattr(importlib.import_module(module), attribute)
    return value


def _provider_source_closure(reference: str) -> dict[str, Any]:
    """Bind the importable local Python closure selected by sealed config."""

    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("provider reference must be module:attribute")
    module_name, attribute = reference.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("provider reference must name a module and attribute")
    spec = importlib.util.find_spec(module_name)
    module_file = Path(spec.origin or "").resolve() if spec is not None else Path()
    if not module_file.is_file():
        raise ValueError(f"provider reference has no bindable source file: {reference}")
    top_name = module_name.split(".", 1)[0]
    top_spec = importlib.util.find_spec(top_name)
    if top_spec is not None and top_spec.submodule_search_locations:
        root = Path(next(iter(top_spec.submodule_search_locations))).resolve()
    else:
        top_file = Path(top_spec.origin or "").resolve() if top_spec is not None else Path()
        root = top_file.parent if top_file.is_file() else module_file.parent
    files = sorted(root.rglob("*.py"))
    if not files:
        files = [module_file]
    rows = [
        {
            "path": path.relative_to(root.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
        if path.is_file() and not path.is_symlink()
    ]
    return {
        "reference": reference,
        "root": str(root),
        "files": rows,
        "sha256": sha256_bytes(canonical_json(rows)),
    }


def _verify_service_implementation_closure(
    config: Mapping[str, Any], services: "QwenCampaignServices"
) -> dict[str, Any]:
    closure = _provider_source_closure(str(config.get("service_factory")))
    allowed = {(row["bytes"], row["sha256"]) for row in closure["files"]}
    providers: dict[str, Any] = {}
    for role in ("capturer", "fitter", "ledger", "codec", "evaluator", "allocator", "checkpoint"):
        provider = getattr(services, role)
        module = inspect.getmodule(type(provider))
        path = Path(getattr(module, "__file__", "")).resolve()
        if not path.is_file() or (path.stat().st_size, sha256_file(path)) not in allowed:
            raise RuntimeError(f"dynamic provider {role} is outside the sealed service-factory source closure")
        row: dict[str, Any] = {
            "class": f"{type(provider).__module__}:{type(provider).__qualname__}",
            "source_bytes": path.stat().st_size,
            "source_sha256": sha256_file(path),
        }
        identity = getattr(provider, "identity", None)
        if callable(identity):
            declared = dict(identity())
            row["declared"] = declared
            row["declared_sha256"] = sha256_bytes(canonical_json(declared))
        providers[role] = row
    return providers


def _files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {".runner-result.json", "stage-manifest.json"}
    }


def _resolve_result_file(root: Path, result: Mapping[str, Any], key: str) -> Path:
    raw = result.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"provider result must name actual {key}")
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"provider {key} must be a regular file inside the stage output")
    return path


def _stage_manifest(request: StageRequest, provider_result: Mapping[str, Any]) -> dict[str, Any]:
    files = _files(request.output_dir)
    if not files:
        raise ValueError(f"Qwen stage {request.stage_id} produced no artifact files")
    body = {
        "schema": STAGE_MANIFEST_SCHEMA,
        "stage_id": request.stage_id,
        "kind": request.kind,
        "request_sha256": request.request_sha256,
        "predecessor_state_hash": request.predecessor_state_hash,
        "generation": request.generation,
        "generation_context": request.generation_context,
        "installed_layer_prefix": list(request.installed_layer_prefix),
        "layer": request.layer,
        "block_layers": list(request.block_layers),
        "provider_result": dict(provider_result),
        "files": files,
    }
    body["manifest_sha256"] = sha256_bytes(canonical_json(body))
    write_json(request.output_dir / "stage-manifest.json", body)
    return body


def _load_logits(path: Path):
    import numpy as np

    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            key = "logits" if "logits" in archive else keys[0] if len(keys) == 1 else None
            if key is None:
                raise ValueError(f"ambiguous logits archive: {path}")
            value = archive[key]
    elif suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="np") as handle:
            keys = list(handle.keys())
            key = "logits" if "logits" in keys else keys[0] if len(keys) == 1 else None
            if key is None:
                raise ValueError(f"ambiguous logits safetensors: {path}")
            value = handle.get_tensor(key)
    elif suffix == ".json":
        document = json.loads(path.read_text())
        value = document.get("logits") if isinstance(document, dict) else document
    else:
        raise ValueError(f"unsupported bound logits format: {path}")
    value = np.asarray(value)
    if value.ndim < 2 or value.shape[-1] < 2 or not np.isfinite(value).all():
        raise ValueError(f"logits must be finite with shape [..., positions, vocab]: {path}")
    return value.reshape(-1, value.shape[-1])


def _window_binding(request: StageRequest, *, final: bool) -> dict[str, Any]:
    from ..evaluation.kld_window import verify_kld_window

    root = Path(request.static_inputs["kld_window"]["path"])
    document = json.loads((root / "kld-window.json").read_text())
    verify_kld_window(document, root)
    return {
        "role": "historical-control-final" if final else "historical-control-reanchor",
        "artifact_sha256": request.static_inputs["kld_window"]["sha256"],
        "window_count": 1,
        "prediction_positions": int(document["prediction_positions"]),
        "token_sha256": [document["token_sha256"]],
    }


def _bootstrap_mean(values, *, samples: int, seed: int) -> dict[str, float | int]:
    import numpy as np

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = np.mean(generator.choice(values, size=values.size, replace=True))
    return {
        "samples": int(samples),
        "seed": int(seed),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _independent_kld(
    request: StageRequest,
    provider_result: Mapping[str, Any],
    *,
    final: bool,
) -> tuple[dict[str, Any], Path, Path]:
    """Recompute every reported KLD statistic from sealed logit bytes."""

    import numpy as np

    from ..scoring.kld import summarize, token_kld

    result = dict(provider_result)
    if "reference_file" not in result or "capture_file" not in result:
        report_name = result.get("kld_report_file")
        if isinstance(report_name, str):
            report_path = _resolve_result_file(request.output_dir, result, "kld_report_file")
            supplied = json.loads(report_path.read_text())
            for key in ("reference_file", "capture_file"):
                if key not in result and isinstance(supplied.get(key), str):
                    result[key] = supplied[key]
    reference = _resolve_result_file(request.output_dir, result, "reference_file")
    capture = _resolve_result_file(request.output_dir, result, "capture_file")
    teacher_artifact = request.dependency_artifacts.get("teacher_capture")
    if teacher_artifact is None:
        raise ValueError("KLD evaluation lacks the sealed teacher-capture dependency")
    teacher_receipt = json.loads(
        (request.campaign_dir / teacher_artifact["path"] / ".runner-result.json").read_text()
    )
    if teacher_receipt.get("metadata", {}).get("teacher_reference_sha256") != sha256_file(reference):
        raise ValueError("KLD reference logits differ from the sealed teacher capture")
    if final:
        student_artifact = request.dependency_artifacts.get("student_capture")
        if student_artifact is None:
            raise ValueError("final KLD lacks the sealed student-capture dependency")
        student_receipt = json.loads(
            (request.campaign_dir / student_artifact["path"] / ".runner-result.json").read_text()
        )
        if student_receipt.get("metadata", {}).get("student_capture_sha256") != sha256_file(capture):
            raise ValueError("final KLD student logits differ from the sealed student capture")
        checkpoint_binding = {
            "checkpoint_manifest_sha256": student_receipt.get("metadata", {}).get("checkpoint_manifest_sha256"),
            "checkpoint_audit_sha256": student_receipt.get("metadata", {}).get("checkpoint_audit_sha256"),
        }
        if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in checkpoint_binding.values()):
            raise ValueError("final KLD student capture lacks sealed checkpoint/audit identities")
    else:
        checkpoint_binding = {
            "installed_state_hash": request.predecessor_state_hash,
            "installed_layer_count": len(request.installed_layer_prefix),
        }
    teacher = _load_logits(reference)
    student = _load_logits(capture)
    if teacher.shape != student.shape:
        raise ValueError("reference and student logits have different shapes")
    windows = _window_binding(request, final=final)
    if teacher.shape[0] != windows["prediction_positions"]:
        raise ValueError(
            f"logit positions {teacher.shape[0]} do not match sealed window positions {windows['prediction_positions']}"
        )
    values = token_kld(teacher, student)
    summary = summarize(values)
    objective = request.experiment_spec["document"]["objective"]
    bootstrap = _bootstrap_mean(
        values,
        samples=int(objective.get("bootstrap_samples", 2000)),
        seed=int(request.experiment_spec["document"]["corpus"].get("seed", 20260823)),
    )
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    values_path = request.output_dir / "independent-token-kld.npy"
    atomic_write(values_path, buffer.getvalue())
    report = {
        "schema": "quant-pipeline.independent-kld.v1",
        "stage_id": request.stage_id,
        "request_sha256": request.request_sha256,
        "generation": request.generation,
        "predecessor_state_hash": request.predecessor_state_hash,
        "installed_layer_prefix": list(request.installed_layer_prefix),
        "reference_sha256": sha256_file(reference),
        "capture_sha256": sha256_file(capture),
        "token_kld_sha256": sha256_file(values_path),
        "logit_shape": list(teacher.shape),
        "windows": windows,
        "checkpoint_binding": checkpoint_binding,
        "summary": summary,
        "bootstrap_mean": bootstrap,
        "journal_sha256_at_evaluation": sha256_file(request.campaign_dir / "events.jsonl"),
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    write_json(request.output_dir / "independent-kld.json", report)
    return report, reference, capture


def _independent_confirmation_kld(
    request: StageRequest,
    provider_result: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    """Replay a post-freeze KLD gate on the prospective confirmation role."""

    import numpy as np

    from ..calibration.windows import verify_sealed_corpus
    from ..scoring.kld import summarize, token_kld

    reference = _resolve_result_file(request.output_dir, provider_result, "reference_file")
    capture = _resolve_result_file(request.output_dir, provider_result, "capture_file")
    corpus_path = Path(request.static_inputs["sealed_corpus"]["path"])
    corpus = json.loads(corpus_path.read_text())
    verify_sealed_corpus(corpus)
    if corpus.get("schema") != "quant-pipeline.sealed-corpus.v2":
        raise ValueError("confirmation gate requires the five-role sealed-corpus v2 schema")
    windows = corpus["windows"]["confirmation"]
    positions = sum(len(row["token_ids"]) - 1 for row in windows)
    teacher = _load_logits(reference)
    student = _load_logits(capture)
    if teacher.shape != student.shape or teacher.shape[0] != positions:
        raise ValueError("confirmation logits do not match the sealed prospective role")
    values = token_kld(teacher, student)
    values_path = request.output_dir / "independent-confirmation-token-kld.npy"
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    atomic_write(values_path, buffer.getvalue())
    summary = summarize(values)
    objective = request.experiment_spec["document"]["objective"]
    bootstrap = _bootstrap_mean(
        values,
        samples=int(objective.get("bootstrap_samples", 2000)),
        seed=int(request.experiment_spec["document"]["corpus"].get("seed", 20260823)) + 1,
    )
    frozen = {
        key: request.dependency_artifacts[key]["artifact_sha256"]
        for key in ("candidates", "allocation")
    }
    report = {
        "schema": "quant-pipeline.independent-confirmation-kld.v1",
        "stage_id": request.stage_id,
        "role": "confirmation",
        "prospective_only": True,
        "sealed_corpus_sha256": request.static_inputs["sealed_corpus"]["sha256"],
        "frozen_candidate_and_allocation_artifacts": frozen,
        "reference_sha256": sha256_file(reference),
        "capture_sha256": sha256_file(capture),
        "token_kld_sha256": sha256_file(values_path),
        "prediction_positions": positions,
        "summary": summary,
        "bootstrap_mean": bootstrap,
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    write_json(request.output_dir / "independent-confirmation-kld.json", report)
    return report, reference, capture


def _verify_checkpoint_audit(
    request: StageRequest,
    audit_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    document = json.loads(audit_path.read_text())
    required = {
        "checkpoint_manifest_sha256",
        "reader_identity",
        "reader_identity_sha256",
        "reader_result",
        "reader_result_sha256",
    }
    if not required <= set(document):
        raise ValueError(f"checkpoint audit lacks independently verifiable reader fields: {sorted(required - set(document))}")
    reader_identity = document["reader_identity"]
    reader_result = document["reader_result"]
    if sha256_bytes(canonical_json(reader_identity)) != document["reader_identity_sha256"]:
        raise ValueError("checkpoint reader identity seal mismatch")
    if sha256_bytes(canonical_json(reader_result)) != document["reader_result_sha256"]:
        raise ValueError("checkpoint reader result seal mismatch")
    if document["reader_identity_sha256"] != config.get("runtime_reader_identity_sha256"):
        raise ValueError("checkpoint audit did not use the sealed pinned runtime reader")
    emission = request.dependency_artifacts["checkpoint_emission"]
    receipt = json.loads((request.campaign_dir / emission["path"] / ".runner-result.json").read_text())
    expected_checkpoint = receipt.get("metadata", {}).get("checkpoint_manifest_sha256")
    if document["checkpoint_manifest_sha256"] != expected_checkpoint:
        raise ValueError("checkpoint audit targets a different emitted checkpoint")
    if not isinstance(reader_result, dict) or reader_result.get("ok") is not True:
        raise RuntimeError("pinned target runtime reader rejected the checkpoint")
    return document


class QwenCampaignAdapter:
    """StageAdapter whose production behavior is selected by sealed config."""

    def __init__(self, services: QwenCampaignServices | None = None, *, production: bool = True) -> None:
        self._injected = services
        self.production = bool(production)
        self._loaded: QwenCampaignServices | None = None

    def identity(self) -> Mapping[str, Any]:
        closure = [Path(__file__), Path(__file__).with_name("runner.py")]
        package = Path(__file__).resolve().parents[1]
        for relative in (
            "calibration/qwen_capture.py",
            "calibration/fitter.py",
            "candidates/ledger.py",
            "checkpoint/exact_payload.py",
            "checkpoint/btx_qwen.py",
            "checkpoint/official_btx.py",
            "normalization/absolute_v31.py",
            "normalization/streaming_v31.py",
            "codecs/exl3_mcg.py",
            "evaluation/kld_window.py",
        ):
            path = package / relative
            if path.is_file():
                closure.append(path)
        files = [
            {"path": path.relative_to(package.parent).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(set(closure))
        ]
        declared: dict[str, Any] = {
            "schema": ADAPTER_SCHEMA,
            "name": "qwen3-30b-a3b-causal-exact-codec",
            "production": self.production,
            "closure": files,
            "closure_sha256": sha256_bytes(canonical_json(files)),
        }
        if self._injected is not None:
            codec_identity = dict(self._injected.codec.identity())
            declared["injected_codec"] = codec_identity
            declared["injected_codec_sha256"] = sha256_bytes(canonical_json(codec_identity))
        return declared

    def identity_for_inputs(self, static_inputs: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        bound = static_inputs.get("adapter_config")
        if not isinstance(bound, Mapping) or not isinstance(bound.get("path"), str):
            if self.production:
                raise ValueError("production Qwen identity requires a sealed adapter_config")
            return {"schema": "quant-pipeline.qwen-dynamic-closure.v1", "production": False}
        path = Path(bound["path"])
        if not path.is_file() or sha256_file(path) != bound.get("sha256"):
            raise ValueError("adapter_config differs while binding dynamic provider closure")
        config = json.loads(path.read_text())
        reference = config.get("service_factory")
        if not isinstance(reference, str):
            raise RuntimeError("production Qwen adapter has no concrete service_factory to seal")
        closure = _provider_source_closure(reference)
        return {
            "schema": "quant-pipeline.qwen-dynamic-closure.v1",
            "service_factory": reference,
            "provider_source_closure": closure,
        }

    @staticmethod
    def _config_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        bound = plan.get("inputs", {}).get("adapter_config")
        if not isinstance(bound, Mapping) or not isinstance(bound.get("path"), str):
            raise ValueError("production Qwen plan requires a bound adapter_config")
        path = Path(bound["path"])
        if not path.is_file() or sha256_file(path) != bound.get("sha256"):
            raise ValueError("adapter_config differs from the sealed plan")
        config = json.loads(path.read_text())
        if config.get("schema") != ADAPTER_SCHEMA:
            raise ValueError("unsupported Qwen adapter config schema")
        return config

    def _services(self, plan_or_request: Mapping[str, Any] | StageRequest) -> QwenCampaignServices:
        if self._injected is not None:
            return self._injected
        if self._loaded is not None:
            return self._loaded
        if isinstance(plan_or_request, StageRequest):
            bound = plan_or_request.static_inputs.get("adapter_config")
            if not isinstance(bound, Mapping):
                raise ValueError("Qwen request has no sealed adapter_config")
            config_path = Path(bound["path"])
            if not config_path.is_file() or sha256_file(config_path) != bound.get("sha256"):
                raise ValueError("Qwen request adapter_config differs from its sealed identity")
            config = json.loads(config_path.read_text())
        else:
            config = self._config_from_plan(plan_or_request)
        factory_ref = config.get("service_factory")
        if not isinstance(factory_ref, str):
            raise RuntimeError("production Qwen adapter requires a concrete local service_factory")
        factory = _load_object(factory_ref)
        value = factory(config) if callable(factory) else factory
        if not isinstance(value, QwenCampaignServices):
            raise TypeError("service_factory must return QwenCampaignServices")
        self._loaded = value
        return value

    def preflight(self, plan: Mapping[str, Any]) -> Mapping[str, Any]:
        config = self._config_from_plan(plan) if self.production else {
            "required_gpu_count": 0,
            "min_compute_capability": [0, 0],
            "estimated_peak_bytes": 1,
            "safety_margin_bytes": 1,
        }
        # Resolve services during production preflight so a missing corrected
        # codec/fitter/runtime implementation fails before a stage starts.
        services = self._services(plan)
        provider_implementations = (
            _verify_service_implementation_closure(config, services) if self.production else {}
        )
        codec_identity = dict(services.codec.identity())
        if self.production and (
            codec_identity.get("backend") != "corrected-exl3-mcg-r10"
            or codec_identity.get("test_only") is True
        ):
            raise RuntimeError("production preflight requires the attested corrected EXL3/MCG R10 codec")
        scientific = config.get("scientific_contract", {})
        required_contract = REQUIRED_SCIENTIFIC_CONTRACT
        if self.production and scientific != required_contract:
            raise RuntimeError(
                "production preflight requires the additive prior-3.5-bpw scientific contract; "
                f"expected {required_contract}"
            )
        required = int(config.get("required_gpu_count", 0))
        minimum = tuple(int(x) for x in config.get("min_compute_capability", [0, 0]))
        minimum_free_gpu = int(config.get("min_free_gpu_bytes_per_device", 0))
        required_cpus = int(config.get("required_cpu_count", 0))
        minimum_ram = int(config.get("min_available_ram_bytes", 0))
        estimated = int(config.get("estimated_peak_bytes", 0))
        reserve = int(config.get("safety_margin_bytes", 0))
        numeric_required = config.get("required_numeric_environment", {})
        if self.production and (
            required <= 0
            or minimum_free_gpu <= 0
            or required_cpus <= 0
            or minimum_ram <= 0
            or estimated <= 0
            or reserve <= 0
            or not isinstance(numeric_required, dict)
            or not numeric_required
        ):
            raise RuntimeError(
                "production preflight requires non-zero campaign-volume disk, per-GPU memory, CPU, RAM, and numeric-environment estimates"
            )
        gpu_rows = []
        try:
            import torch

            if required:
                if not torch.cuda.is_available() or torch.cuda.device_count() < required:
                    raise RuntimeError(f"Qwen campaign requires {required} CUDA devices")
                for index in range(required):
                    capability = tuple(torch.cuda.get_device_capability(index))
                    if capability < minimum or not torch.cuda.is_bf16_supported():
                        raise RuntimeError(f"CUDA device {index} lacks required SM/BF16 support")
                    free, total = torch.cuda.mem_get_info(index)
                    if int(free) < minimum_free_gpu:
                        raise RuntimeError(f"CUDA device {index} lacks required free campaign memory")
                    gpu_rows.append({"index": index, "name": torch.cuda.get_device_name(index), "capability": list(capability), "free_bytes": int(free), "total_bytes": int(total)})
            torch_version = torch.__version__
            cuda_version = torch.version.cuda
        except ImportError:
            if required:
                raise RuntimeError("production preflight requires torch/CUDA")
            torch_version = None
            cuda_version = None
        campaign_root = Path(plan["campaign_dir"]).resolve()
        available = shutil.disk_usage(campaign_root).free
        cpu_count = os.cpu_count() or 0
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        available_ram = page_size * available_pages
        numeric_observed = {str(key): os.environ.get(str(key)) for key in numeric_required}
        numeric_ok = all(numeric_observed[key] == str(value) for key, value in numeric_required.items())
        ok = (
            len(gpu_rows) >= required
            and available >= estimated + reserve
            and cpu_count >= required_cpus
            and available_ram >= minimum_ram
            and numeric_ok
        )
        return {
            "ok": ok,
            "local_only": True,
            "remote_endpoints": [],
            "gpu": {
                "required_count": required,
                "minimum_capability": list(minimum),
                "minimum_free_bytes_per_device": minimum_free_gpu,
                "devices": gpu_rows,
            },
            "storage": {
                "retention_mode": plan["definition"]["retention_mode"],
                "estimated_peak_bytes": estimated,
                "available_bytes": int(available),
                "safety_margin_bytes": reserve,
            },
            "software": {
                "production": self.production,
                "python": platform.python_version(),
                "torch": torch_version,
                "cuda": cuda_version,
                "codec": codec_identity,
                "provider_implementations": provider_implementations,
                "cpu": {
                    "required_count": required_cpus,
                    "available_count": cpu_count,
                    "minimum_available_ram_bytes": minimum_ram,
                    "available_ram_bytes": available_ram,
                },
                "numeric_environment": {
                    "required": {str(key): str(value) for key, value in numeric_required.items()},
                    "observed": numeric_observed,
                    "matches": numeric_ok,
                },
            },
        }

    def _context(self, request: StageRequest) -> dict[str, Any]:
        dependencies = {
            stage: str((request.campaign_dir / artifact["path"]).resolve())
            for stage, artifact in request.dependency_artifacts.items()
        }
        installed = []
        installed_prefix = []
        for row in request.installed_layer_prefix:
            path = (request.campaign_dir / row["path"]).resolve()
            if not path.is_dir() or path.is_symlink():
                raise ValueError(f"accepted installed-layer artifact is missing: {row['stage_id']}")
            installed.append(str(path))
            installed_prefix.append(dict(row))
        return {
            "request": request,
            "output_dir": str(request.output_dir),
            "stage_id": request.stage_id,
            "kind": request.kind,
            "layer": request.layer,
            "block_layers": list(request.block_layers),
            "predecessor_state_hash": request.predecessor_state_hash,
            "generation": request.generation,
            "generation_context": request.generation_context,
            "inputs": {key: row["path"] for key, row in request.static_inputs.items()},
            "input_identities": {key: row["sha256"] for key, row in request.static_inputs.items()},
            "dependencies": dependencies,
            "dependency_identities": {key: row["artifact_sha256"] for key, row in request.dependency_artifacts.items()},
            "installed_layer_attempts": installed,
            "installed_layer_prefix": installed_prefix,
            "experiment": request.experiment_spec,
            "production": self.production,
        }

    def run(self, request: StageRequest) -> StageResult:
        services = self._services(request)
        context = self._context(request)
        config = json.loads(Path(request.static_inputs["adapter_config"]["path"]).read_text()) if "adapter_config" in request.static_inputs else {}
        kind = request.kind
        metadata: dict[str, Any] = {"kind": kind}
        if kind == "identity":
            result = self._identity_stage(context)
        elif kind == "teacher_capture":
            result = dict(services.capturer.capture_teacher(context))
            reference = _resolve_result_file(request.output_dir, result, "teacher_reference_file")
            metadata["teacher_reference_sha256"] = sha256_file(reference)
        elif kind in {"fit_capture", "causal_fit_capture"}:
            result = dict(services.capturer.capture_routes(context))
            capture = _resolve_result_file(request.output_dir, result, "capture_manifest_file")
            if kind.startswith("causal_"):
                metadata |= self._causal(request, "capture_sha256", sha256_file(capture))
        elif kind in {"fit", "causal_fit"}:
            result = dict(services.fitter.fit(context))
            fit = _resolve_result_file(request.output_dir, result, "fit_manifest_file")
            if kind.startswith("causal_"):
                metadata |= self._causal(request, "fit_sha256", sha256_file(fit))
            transient = result.get("transient_files", [])
            if transient:
                metadata["transient_files"] = list(transient)
        elif kind in {"candidates", "causal_candidates"}:
            result = dict(services.ledger.generate(context))
            ledger = _resolve_result_file(request.output_dir, result, "candidate_ledger_file")
            if kind.startswith("causal_"):
                metadata |= self._causal(request, "candidate_ledger_sha256", sha256_file(ledger))
        elif kind == "attribution":
            result = dict(services.evaluator.attribute(context))
            _resolve_result_file(request.output_dir, result, "attribution_file")
        elif kind == "allocation":
            result = dict(services.allocator.allocate(context))
            _resolve_result_file(request.output_dir, result, "allocation_file")
        elif kind == "confirmation":
            result = dict(services.evaluator.confirm(context))
            independent, reference, capture = _independent_confirmation_kld(request, result)
            threshold = float(config.get("confirmation_kld_threshold", -1))
            if not threshold >= 0.0:
                raise ValueError("confirmation gate requires a sealed non-negative confirmation_kld_threshold")
            value = float(independent["summary"]["mean"])
            metadata["gate"] = {
                "passed": value <= threshold,
                "metric": "kld",
                "value": value,
                "threshold": threshold,
                "reference_sha256": sha256_file(reference),
                "capture_sha256": sha256_file(capture),
            }
            metadata["gate_files"] = {
                "reference": reference.relative_to(request.output_dir).as_posix(),
                "capture": capture.relative_to(request.output_dir).as_posix(),
            }
            metadata["independent_confirmation_report_sha256"] = independent["report_sha256"]
        elif kind == "causal_encode":
            result = dict(services.codec.install(context))
            installed = _resolve_result_file(request.output_dir, result, "installed_manifest_file")
            document = json.loads(installed.read_text())
            identity = document.get("installed_checkpoint_sha256")
            # The scientific identity is a canonical body seal, not a
            # coincidental file hash. Both remain captured by the runner.
            if not isinstance(identity, str) or _SHA256.fullmatch(identity) is None:
                raise ValueError("installed manifest lacks installed_checkpoint_sha256")
            metadata |= self._causal(request, "installed_checkpoint_sha256", identity)
        elif kind == "kld_reanchor":
            result = dict(services.evaluator.reanchor(context))
            independent, reference, capture = _independent_kld(request, result, final=False)
            value = float(independent["summary"]["mean"])
            if self.production:
                threshold = float(config.get("reanchor_kld_threshold", -1))
                if not threshold >= 0:
                    raise ValueError("production re-anchor requires a sealed non-negative reanchor_kld_threshold")
            else:
                threshold = float(result["threshold"])
            metadata["gate"] = {
                "passed": value <= threshold,
                "metric": "kld",
                "value": value,
                "threshold": threshold,
                "reference_sha256": sha256_file(reference),
                "capture_sha256": sha256_file(capture),
            }
            metadata["gate_files"] = {"reference": reference.relative_to(request.output_dir).as_posix(), "capture": capture.relative_to(request.output_dir).as_posix()}
            metadata["independent_kld_report_sha256"] = independent["report_sha256"]
        elif kind == "checkpoint_emission":
            result = dict(services.checkpoint.emit(context))
            checkpoint = _resolve_result_file(request.output_dir, result, "checkpoint_manifest_file")
            metadata["checkpoint_manifest_sha256"] = sha256_file(checkpoint)
        elif kind == "checkpoint_audit":
            result = dict(services.checkpoint.audit(context))
            audit = _resolve_result_file(request.output_dir, result, "audit_file")
            if self.production:
                verified_audit = _verify_checkpoint_audit(request, audit, config)
                metadata["reader_identity_sha256"] = verified_audit["reader_identity_sha256"]
                metadata["reader_result_sha256"] = verified_audit["reader_result_sha256"]
            elif json.loads(audit.read_text()).get("ok") is not True:
                raise RuntimeError("target runtime checkpoint audit failed")
            metadata["ok"] = True
        elif kind == "student_capture":
            result = dict(services.evaluator.capture_student(context))
            student = _resolve_result_file(request.output_dir, result, "student_capture_file")
            metadata["student_capture_sha256"] = sha256_file(student)
            emission_root = request.campaign_dir / request.dependency_artifacts["checkpoint_emission"]["path"]
            audit_root = request.campaign_dir / request.dependency_artifacts["checkpoint_audit"]["path"]
            emission_receipt = json.loads((emission_root / ".runner-result.json").read_text())
            audit_receipt = json.loads((audit_root / ".runner-result.json").read_text())
            metadata["checkpoint_manifest_sha256"] = emission_receipt.get("metadata", {}).get("checkpoint_manifest_sha256")
            metadata["checkpoint_audit_sha256"] = audit_receipt.get("receipt_sha256")
            if any(
                not isinstance(metadata[key], str) or _SHA256.fullmatch(metadata[key]) is None
                for key in ("checkpoint_manifest_sha256", "checkpoint_audit_sha256")
            ):
                raise ValueError("student capture lacks sealed checkpoint and audit identities")
        elif kind == "final_kld":
            result = dict(services.evaluator.final_kld(context))
            independent, _reference, _capture = _independent_kld(request, result, final=True)
            metadata["kld"] = float(independent["summary"]["mean"])
            metadata["kld_summary"] = independent["summary"]
            metadata["kld_bootstrap_mean"] = independent["bootstrap_mean"]
            metadata["independent_kld_report_sha256"] = independent["report_sha256"]
        else:  # pragma: no cover - catches runner expansion
            raise ValueError(f"unsupported Qwen campaign stage: {kind}")
        manifest = _stage_manifest(request, result)
        metadata["stage_manifest_sha256"] = manifest["manifest_sha256"]
        return StageResult(metadata)

    @staticmethod
    def _causal(request: StageRequest, field: str, value: str) -> dict[str, Any]:
        return {"layer": request.layer, "predecessor_state_hash": request.predecessor_state_hash, field: value}

    @staticmethod
    def _identity_stage(context: Mapping[str, Any]) -> dict[str, Any]:
        from safetensors import safe_open

        output = Path(context["output_dir"])
        source = Path(context["inputs"]["source_checkpoint"])
        config_path = source / "config.json"
        config = json.loads(config_path.read_text())
        index_path = source / "model.safetensors.index.json"
        if index_path.is_file():
            mapping = json.loads(index_path.read_text())["weight_map"]
        elif (source / "model.safetensors").is_file():
            with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
                mapping = {key: "model.safetensors" for key in handle.keys()}
        else:
            raise FileNotFoundError("Qwen source checkpoint has no safetensors")
        experts = [name for name in mapping if ".mlp.experts." in name]
        if context["production"]:
            geometry = {
                "layers": int(config["num_hidden_layers"]),
                "experts": int(config["num_experts"]),
                "top_k": int(config["num_experts_per_tok"]),
                "hidden_size": int(config["hidden_size"]),
                "intermediate_size": int(config["moe_intermediate_size"]),
            }
            if geometry != {"layers": 48, "experts": 128, "top_k": 8, "hidden_size": 2048, "intermediate_size": 768}:
                raise ValueError(f"source is not Qwen3-30B-A3B production geometry: {geometry}")
        receipt = {
            "schema": "quant-pipeline.qwen-source-inventory.v1",
            "source_checkpoint_sha256": context["input_identities"]["source_checkpoint"],
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
            "tensor_count": len(mapping),
            "expert_tensor_count": len(experts),
            "expert_tensor_names": sorted(experts),
        }
        receipt["inventory_sha256"] = sha256_bytes(canonical_json(receipt))
        write_json(output / "source-inventory.json", receipt)
        return {"source_inventory_file": "source-inventory.json", "inventory_sha256": receipt["inventory_sha256"]}
