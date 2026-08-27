"""Concrete, local-only services for the Qwen causal quantization campaign.

The campaign adapter intentionally exposes small provider protocols.  This
module is the production composition root: it connects those protocols to the
sealed Qwen capture, route-aware fitter, corrected EXL3/MCG codec, exact
candidate ledger, allocator, causal install, and pinned upstream BTX writer.

No service shells out to a placeholder command and no production branch uses
a uniform quantizer.  Expensive CUDA work is reached only from a campaign
stage; constructing the factory and running preflight merely verifies/pins the
implementation closure.
"""

from __future__ import annotations

import json
import importlib
import inspect
import math
import re
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..calibration.fitter import (
    ACCOUNTING_KINDS,
    CalibrationBatch,
    CalibrationFitter,
    FittedExpertStatistics,
    ROUTE_WEIGHT_POWERS,
    SCHEMA as CALIBRATION_FIT_SCHEMA,
    STORED_ARRAY_FIELDS,
    _oas_style_identity_shrinkage,
    save_fitted_statistics,
    verify_fitted_statistics,
)
from ..calibration.qwen_capture import (
    capture_roles_from_local_bf16,
    qwen_moe_layers,
    verify_capture_chunk,
    verify_capture_manifest,
)
from ..calibration.route_mass import RouteMassRow, build_route_mass_audit
from ..checkpoint.btx_qwen import (
    install_layer_payloads,
    reconcile_installed_allocation,
)
from ..checkpoint.official_btx import (
    UPSTREAM_COMMIT,
    UpstreamBtxRuntimeReader,
    audit_official_btx_checkpoint,
    btx_compatibility_report,
    emit_official_btx_checkpoint,
)
from ..codecs.exl3_mcg import Exl3MCGCodec
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from ..normalization.artifact_v31 import (
    PinnedGSSRequest,
    PinnedGSSResult,
    make_gss_receipt,
)
from ..scoring.attribution import (
    aumann_shapley,
    reconcile_layer_components_for_allocation,
    reconcile_signed_completeness,
    split_layer_damage,
)
from .qwen_adapter import QwenCampaignServices
from .qwen_attribution import (
    persist_provisional_winner_deltas,
    produce_qwen_attribution_inputs_from_local,
    verify_attribution_inputs,
)


SERVICE_SCHEMA = "quant-pipeline.qwen-concrete-services.v1"
CAPTURE_SERVICE_SCHEMA = "quant-pipeline.qwen-service-capture.v1"
FIT_MANIFEST_SCHEMA = "quant-pipeline.qwen-service-fit.v1"
ATTRIBUTION_SCHEMA = "quant-pipeline.qwen-attribution.v2"
ALLOCATION_SCHEMA = "quant-pipeline.qwen-dual-arm-allocation.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _hash_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return sha256_bytes(canonical_json(value))


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_allocation_document(allocation: Mapping[str, Any]) -> None:
    if allocation.get("schema") != ALLOCATION_SCHEMA:
        raise ValueError("allocation schema mismatch")
    seal = allocation.get("allocation_sha256")
    if seal != _hash_json({key: value for key, value in allocation.items() if key != "allocation_sha256"}):
        raise ValueError("allocation seal mismatch")


def _arm_choice_identity(arm: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    rows = arm.get("choices")
    if not isinstance(rows, list) or not rows:
        raise ValueError("allocation arm must contain selected choices")
    identity = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("allocation arm choice must be an object")
        unit_id = row.get("unit_id")
        choice_id = row.get("choice_id")
        record = row.get("candidate_record_sha256")
        if not isinstance(unit_id, str) or not unit_id or not isinstance(choice_id, str) or not choice_id:
            raise ValueError("allocation arm choice lacks unit/choice identity")
        identity.append((unit_id, choice_id, _require_hash(record, "allocation arm candidate record")))
    identity.sort()
    if len({row[0] for row in identity}) != len(identity):
        raise ValueError("allocation arm selects a unit more than once")
    return tuple(identity)


def _requested_allocation_arm(
    allocation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    """Resolve the arm that may be installed, failing closed on rate drift.

    The unconstrained research optimizer and the fused-runtime legality filter
    answer different questions.  A mixed exact-rate research result must never
    be silently replaced by a lower-rate runtime-qualified result merely
    because the latter is the only arm the current fused kernel accepts.
    """

    configured = config.get("requested_allocation_arm")
    declared = allocation.get("requested_allocation_arm")
    if configured is None and declared is None:
        # Compatibility for explicitly non-production fixtures and historical
        # sealed documents.  The shipped production config names its arm.
        name = "serving_arm"
    else:
        name = str(configured if configured is not None else declared)
        if declared is not None and str(declared) != name:
            raise ValueError("sealed allocation requested arm differs from runtime configuration")
    if name not in {"research_arm", "runtime_qualified_arm", "serving_arm"}:
        raise ValueError("requested_allocation_arm must name research_arm or runtime_qualified_arm")
    arm = allocation.get(name)
    if not isinstance(arm, Mapping):
        raise ValueError(f"allocation lacks requested {name}")
    selected_cost = arm.get("selected_cost")
    if not isinstance(selected_cost, Mapping):
        raise ValueError(f"requested {name} lacks reconciled selected_cost")
    allocated = int(selected_cost.get("allocated_payload_bytes", -1))
    if allocated != int(arm.get("stored_bytes", allocated)):
        raise ValueError(f"requested {name} stored bytes differ from selected cost")
    if bool(config.get("require_exact_payload_budget", False)):
        budget_value = config.get("byte_budget", config.get("exact_payload_byte_budget"))
        if isinstance(budget_value, bool) or not isinstance(budget_value, int):
            raise ValueError("exact runtime arm requires an integer exact payload byte budget")
        if allocated != budget_value:
            raise RuntimeError(
                f"requested {name} uses {allocated} bytes, not the exact requested {budget_value} bytes"
            )
    if bool(config.get("require_fused_btx", True)) and name == "research_arm":
        runtime = allocation.get("runtime_qualified_arm")
        if not isinstance(runtime, Mapping):
            raise RuntimeError("fused BTX installation requires an explicit runtime-qualified arm")
        if _arm_choice_identity(arm) != _arm_choice_identity(runtime):
            raise RuntimeError(
                "the requested research allocation is not fused-BTX runtime-qualified; "
                "refusing to substitute the runtime arm or collapse its exact mixed rate"
            )
    return name, arm


def _output(context: Mapping[str, Any]) -> Path:
    root = Path(str(context["output_dir"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dependency(context: Mapping[str, Any], prefix: str) -> Path:
    matches = [Path(path) for stage, path in context["dependencies"].items() if stage == prefix or stage.startswith(prefix + ".")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {prefix} dependency, found {len(matches)}")
    root = matches[0].resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"dependency is not a regular artifact directory: {root}")
    return root


def _provider_result(root: Path) -> Mapping[str, Any]:
    manifest = json.loads((root / "stage-manifest.json").read_text())
    result = manifest.get("provider_result")
    if not isinstance(result, dict):
        raise ValueError(f"dependency has no provider result: {root}")
    return result


def _result_path(root: Path, key: str) -> Path:
    raw = _provider_result(root).get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"dependency provider result lacks {key}")
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"dependency {key} escapes or is missing: {path}")
    return path


def _tensor_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one already-verified safetensors chunk without trusting metadata."""

    from safetensors import safe_open

    receipt = verify_capture_chunk(path)
    tensors: dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return dict(receipt["metadata"]), tensors


def _load_pinned_student_runtime(config: Mapping[str, Any]):
    raw = config.get("student_runtime")
    if not isinstance(raw, Mapping):
        raise ValueError("adapter config requires a pinned student_runtime")
    reference = raw.get("factory")
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("student_runtime.factory must be module:attribute")
    module_name, attribute = reference.split(":", 1)
    module = importlib.import_module(module_name)
    source = Path(inspect.getsourcefile(module) or "").resolve()
    expected_source = _require_hash(raw.get("source_sha256"), "student runtime source")
    if not source.is_file() or source.is_symlink() or sha256_file(source) != expected_source:
        raise RuntimeError("pinned student runtime source is missing or drifted")
    factory = getattr(module, attribute)
    runtime = factory(dict(raw.get("options", {}))) if callable(factory) else factory
    identity_method = getattr(runtime, "identity", None)
    capture_method = getattr(runtime, "capture", None)
    if not callable(identity_method) or not callable(capture_method):
        raise TypeError("student runtime must implement identity() and capture()")
    identity = dict(identity_method())
    identity_sha256 = _hash_json(identity)
    if identity_sha256 != _require_hash(raw.get("identity_sha256"), "student runtime identity"):
        raise RuntimeError("pinned student runtime identity differs from adapter config")
    if identity.get("test_only") is True:
        raise RuntimeError("test-only student runtime cannot be used by the production factory")
    return runtime, {
        "factory": reference,
        "source": str(source),
        "source_sha256": expected_source,
        "identity": identity,
        "identity_sha256": identity_sha256,
    }


def _quantize_route_weight(value: float, denominator: int, tolerance: float) -> int:
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("routed weight must lie in (0, 1]")
    units = int(round(value * denominator))
    units = min(denominator, max(1, units))
    if abs(value - units / denominator) > tolerance:
        raise ValueError(
            f"router weight {value!r} cannot be represented at denominator {denominator} "
            f"within tolerance {tolerance}"
        )
    return units


@dataclass(frozen=True)
class CorrectedPinnedGSSProducer:
    """Real multi-point v31 GSS backed by the pinned corrected R10 core."""

    codec: Exl3MCGCodec
    evaluations: int = 13

    def search(self, request: PinnedGSSRequest) -> PinnedGSSResult:
        if self.evaluations != 13:
            raise ValueError("the pinned v31 core performs exactly 13 golden-section evaluations")
        backend = self.codec._codec()
        quant_args = backend._quant_args(int(request.bits), self.codec.sigma_reg)
        scale, _objective = backend.core.g_scale_gss(request.target, quant_args)
        identity_sha256 = _hash_json(self.codec.identity)
        search_config = {
            "algorithm": "encode_tr3_v31.g_scale_gss",
            "evaluations": self.evaluations,
            "width": 3,
            "bits": int(request.bits),
            "sigma_reg": self.codec.sigma_reg,
        }
        receipt = make_gss_receipt(
            request,
            scale=float(scale),
            evaluator_code_sha256=self.codec.identity["numeric_core_sha256"],
            codec_identity_sha256=identity_sha256,
            search_config_sha256=_hash_json(search_config),
            evaluations=self.evaluations,
        )
        return PinnedGSSResult(scale=float(scale), receipt=receipt)


class QwenRouteCaptureService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def identity(self) -> Mapping[str, Any]:
        return {
            "schema": SERVICE_SCHEMA,
            "provider": "qwen-route-capture",
            "implementation": "sealed-causal-prefix-replay",
            "attention_backend": str(self.config.get("attention_backend", "eager")),
        }

    def _model_revision(self) -> str:
        value = self.config.get("model_revision")
        if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
            raise ValueError("adapter config model_revision must be immutable 40-hex")
        return value

    def _load_logits(self, context: Mapping[str, Any], *, checkpoint: Path | None = None) -> np.ndarray:
        import torch
        from transformers import AutoModelForCausalLM

        from ..checkpoint.btx_qwen import replay_installed_layers
        from ..evaluation.kld_window import verify_kld_window

        source = Path(context["inputs"]["source_checkpoint"]).resolve()
        window_root = Path(context["inputs"]["kld_window"]).resolve()
        window = json.loads((window_root / "kld-window.json").read_text())
        verify_kld_window(window, window_root)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map=self.config.get("device_map", "auto"),
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=str(self.config.get("attention_backend", "eager")),
        ).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        installed = tuple(context.get("installed_layer_attempts", ()))
        if installed:
            replay_installed_layers(
                model,
                installed,
                expected_final_state_hash=context["predecessor_state_hash"],
                expected_prefix=context.get("installed_layer_prefix", ()),
            )
        if checkpoint is not None:
            # The pinned BTX runtime reader is the only accepted path for final
            # checkpoint logits.  Loading it as a Transformers model would
            # silently ignore atom payloads.
            raise RuntimeError("final BTX student logits require the configured pinned runtime evaluator")
        device = model.get_input_embeddings().weight.device
        ids = torch.tensor([window["token_ids"]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
        return logits.float().cpu().numpy().reshape(-1, logits.shape[-1])

    def capture_teacher(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        output = _output(context)
        target = output / "teacher-logits.npy"
        np.save(target, self._load_logits(context), allow_pickle=False)
        return {"teacher_reference_file": target.name, "model_revision": self._model_revision()}

    def capture_routes(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        output = _output(context)
        layers = [int(context["layer"])] if context.get("layer") is not None else [int(value) for value in context["block_layers"]]
        if not layers:
            layers = [int(value) for value in self.config.get("layers", ())]
        if not layers:
            raise ValueError("route capture requires an explicit layer inventory")
        roles = {
            "fit": str(self.config.get("fit_capture_role", "fit")),
            "heldout": str(self.config.get("heldout_capture_role", "selection")),
            "conditional_down": str(self.config.get("conditional_down_capture_role", "conditional_fit")),
        }
        required_roles = {
            "fit": "fit",
            "heldout": "selection",
            "conditional_down": "conditional_fit",
        }
        if any(roles[purpose] != role for purpose, role in required_roles.items()):
            raise ValueError(
                "candidate capture roles must be fit, selection, and conditional_fit; "
                "confirmation is prospective-only"
            )
        supplemental = self.config.get("supplemental_capture_role")
        if supplemental is not None:
            roles["supplemental"] = str(supplemental)
        requests = []
        for index, (purpose, role) in enumerate(roles.items()):
            component_root = output / purpose
            requests.append({
                "purpose": purpose,
                "role": role,
                "output_dir": str(component_root),
                "fisher_rank": int(self.config.get("fisher_rank", 0)) if purpose == "heldout" else 0,
                "seed": int(self.config.get("capture_seed", 20260823)) + index,
            })
        captured = capture_roles_from_local_bf16(
            source_checkpoint=context["inputs"]["source_checkpoint"],
            model_revision=self._model_revision(),
            sealed_corpus=context["inputs"]["sealed_corpus"],
            captures=requests,
            layers=layers,
            predecessor_state_hash=context["predecessor_state_hash"],
            installed_layers=context.get("installed_layer_attempts", ()),
            installed_layer_prefix=context.get("installed_layer_prefix", ()),
            device_map=self.config.get("device_map", "auto"),
            attn_implementation=str(self.config.get("attention_backend", "eager")),
            production_geometry=bool(context["production"]),
        )
        captures: dict[str, Any] = {}
        for purpose, role in roles.items():
            manifest = captured[purpose]
            captures[purpose] = {
                "role": role,
                "manifest": f"{purpose}/capture-manifest.json",
                "capture_sha256": manifest["capture_sha256"],
            }
        service_manifest = {
            "schema": CAPTURE_SERVICE_SCHEMA,
            "predecessor_state_hash": context["predecessor_state_hash"],
            "layers": layers,
            "captures": captures,
            "streaming": "one-window-one-layer-chunk",
            "retention": str(self.config.get("capture_retention", "sealed-chunks")),
        }
        service_manifest["capture_service_sha256"] = _hash_json(service_manifest)
        write_json(output / "capture-service-manifest.json", service_manifest)
        return {
            "capture_manifest_file": "capture-service-manifest.json",
            "capture_sha256": service_manifest["capture_service_sha256"],
            "streaming": "one-window-one-layer-chunk",
            "retention": str(self.config.get("capture_retention", "sealed-chunks")),
        }


def _full_p2_accounting_shell(
    *,
    layer: int,
    expert: int,
    projection: str,
    hidden_size: int,
    predecessor_checkpoint_hash: str,
    source_identities: Mapping[str, str],
    route_weights: Sequence[np.ndarray],
    document_ids: Sequence[Sequence[str]],
    token_offsets: Sequence[np.ndarray],
    regularization_floor: float,
) -> FittedExpertStatistics:
    """Build exact scalar/sample accounting without a redundant covariance pass."""

    weights = np.concatenate(route_weights).astype(np.float64, copy=False)
    documents = [item for batch in document_ids for item in batch]
    offsets = np.concatenate(token_offsets).astype(np.int64, copy=False)
    if weights.ndim != 1 or len(weights) != len(documents) or len(weights) != len(offsets) or not len(weights):
        raise ValueError("full-p2 accounting rows are empty or misaligned")
    sample_keys = sorted(f"{document}\0{int(offset)}" for document, offset in zip(documents, offsets, strict=True))
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("full-p2 accounting contains duplicate routed sample identities")
    sample_sha = sha256_bytes(canonical_json(sample_keys))
    empty_sha = sha256_bytes(canonical_json([]))
    accounting: dict[str, Any] = {}
    for kind in ACCOUNTING_KINDS:
        populated = kind in {"natural", "combined"}
        powers = {}
        for power in ROUTE_WEIGHT_POWERS:
            powered = np.power(weights, power, dtype=np.float64) if populated else np.empty(0, dtype=np.float64)
            weight_sum = float(powered.sum(dtype=np.float64))
            weight_square_sum = float(np.dot(powered, powered))
            powers[str(power)] = {
                "matrix_retained": kind == "combined" and power == 2,
                "count": len(weights) if populated else 0,
                "document_count": len(set(documents)) if populated else 0,
                "sample_count": len(weights) if populated else 0,
                "sample_keys_sha256": sample_sha if populated else empty_sha,
                "weight_sum": weight_sum,
                "weight_square_sum": weight_square_sum,
                "effective_sample_size": weight_sum * weight_sum / weight_square_sum if weight_square_sum else 0.0,
                "shrinkage_coefficient": None,
                "shrinkage_target": "scaled_identity",
                "shrinkage_target_scale": None,
            }
        accounting[kind] = {"powers": powers}
    metadata = {
        "schema": CALIBRATION_FIT_SCHEMA,
        "identity": {
            "layer_id": layer,
            "expert_id": str(expert),
            "projection": projection,
            "hidden_size": hidden_size,
            "predecessor_checkpoint_hash": predecessor_checkpoint_hash,
            "source_identities": dict(sorted(source_identities.items())),
        },
        "estimator": {
            "accumulator_dtype": "float64",
            "artifact_array_dtype": "float32",
            "merge_comparison_tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "covariance": "centered_diagnostic_derived_from_persisted_raw_second_moment",
            "route_weight_powers": list(ROUTE_WEIGHT_POWERS),
            "retained_accounting": ["combined"],
            "retained_powers": [2],
            "covariance_mode": "full",
            "block_size": 128,
            "stored_array_fields": list(STORED_ARRAY_FIELDS),
            "derived_array_fields": ["regularized_covariance", "regularized_second_moment"],
            "supplemental_correction": "inverse_inclusion_probability",
            "combined_accounting": "natural_plus_supplemental_corrected",
            "regularization": "oas_style_heuristic_scaled_identity_for_weighted_routed_moments",
            "regularization_floor": regularization_floor,
        },
        "accounting": accounting,
    }
    return FittedExpertStatistics(metadata=metadata, arrays={})


def _torch_full_p2_statistics(
    accounting: FittedExpertStatistics,
    values: Sequence[np.ndarray],
    route_weights: Sequence[np.ndarray],
    *,
    device: str,
) -> FittedExpertStatistics:
    """Replace a cheap accounting fit with deterministic full-p2 Torch GEMM."""

    import torch

    if not values or len(values) != len(route_weights):
        raise ValueError("Torch full-p2 fitting requires matched routed values and weights")
    x_cpu = np.concatenate(values, axis=0).astype(np.float64, copy=False)
    w_cpu = np.concatenate(route_weights, axis=0).astype(np.float64, copy=False)
    if x_cpu.ndim != 2 or w_cpu.ndim != 1 or len(x_cpu) != len(w_cpu):
        raise ValueError("Torch full-p2 routed rows are malformed")
    x = torch.from_numpy(x_cpu).to(device=device, dtype=torch.float64)
    weights = torch.from_numpy(w_cpu).to(device=device, dtype=torch.float64).square()
    weight_sum = weights.sum()
    if not bool(torch.isfinite(weight_sum)) or float(weight_sum) <= 0.0:
        raise ValueError("Torch full-p2 route mass must be finite and positive")
    mean = torch.einsum("n,nd->d", weights, x) / weight_sum
    second = x.T.matmul(x * weights[:, None]) / weight_sum
    second = (second + second.T) * 0.5
    arrays = {
        "combined.p2.mean": mean.to(torch.float32).cpu().numpy(),
        "combined.p2.second_moment": second.to(torch.float32).cpu().numpy(),
    }
    metadata = copy.deepcopy(accounting.metadata)
    metadata["estimator"]["covariance_mode"] = "full"
    stored_mean = arrays["combined.p2.mean"].astype(np.float64)
    stored_second = arrays["combined.p2.second_moment"].astype(np.float64)
    covariance = (stored_second - np.outer(stored_mean, stored_mean))
    covariance = (covariance + covariance.T) * 0.5
    record = metadata["accounting"]["combined"]["powers"]["2"]
    alpha, target_scale, _regularized = _oas_style_identity_shrinkage(
        covariance,
        float(record["effective_sample_size"]),
        float(metadata["estimator"]["regularization_floor"]),
        dimension=int(metadata["identity"]["hidden_size"]),
        covariance_mode="full",
    )
    record["shrinkage_coefficient"] = alpha
    record["shrinkage_target_scale"] = target_scale
    result = FittedExpertStatistics(metadata=metadata, arrays=arrays)
    verify_fitted_statistics(result)
    del x, weights, mean, second
    return result


class QwenFitterService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def identity(self) -> Mapping[str, Any]:
        route_power = int(self.config.get("route_weight_power", 2))
        retained_powers = tuple(int(value) for value in self.config.get("retained_powers", (route_power,)))
        backend = str(self.config.get("fitter_backend", "numpy_full"))
        return {
            "schema": SERVICE_SCHEMA,
            "provider": "qwen-route-aware-fitter",
            "statistic": "direct-raw-second-moment",
            "accounted_powers": [0, 1, 2],
            "retained_powers": list(retained_powers),
            "retained_accounting": list(self.config.get("retained_accounting", ("combined",))),
            "covariance_mode": str(self.config.get("covariance_mode", "full")),
            "artifact_dtype": str(self.config.get("artifact_dtype", "float32")),
            "fitter_backend": backend,
            "fitter_device": str(self.config.get("fitter_device", "cpu")),
        }

    def fit(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        capture_root = _dependency(context, "causal_fit_capture" if context["kind"] == "causal_fit" else "fit_capture")
        capture_file = _result_path(capture_root, "capture_manifest_file")
        service_capture = json.loads(capture_file.read_text())
        if service_capture.get("schema") != CAPTURE_SERVICE_SCHEMA:
            raise ValueError("fitter requires the concrete multi-role capture manifest")
        service_seal = service_capture.get("capture_service_sha256")
        if service_seal != _hash_json({key: value for key, value in service_capture.items() if key != "capture_service_sha256"}):
            raise ValueError("capture service manifest seal mismatch")
        fit_component = service_capture["captures"]["fit"]
        fit_capture_root = capture_file.parent / fit_component["manifest"].rsplit("/", 1)[0]
        # Every consumed chunk is verified by _tensor_file below. Avoid a
        # campaign-wide readback here so a one-layer streaming fit audits only
        # the bytes it actually consumes rather than rereading all 48 layers.
        manifest = verify_capture_manifest(fit_capture_root, verify_chunks=False)
        output = _output(context)
        denominator = int(self.config.get("route_weight_denominator", 1 << 24))
        tolerance = float(self.config.get("route_weight_tolerance", 0.5 / denominator + 1e-12))
        cold_min = int(self.config.get("cold_expert_min_weight_units", 0))
        seed_hash = str(self.config.get("topup_seed_sha256", sha256_bytes(b"qwen-cold-expert-topup-v1")))
        source_ids = {
            "source_checkpoint": str(context["input_identities"]["source_checkpoint"]),
            "capture": str(manifest["capture_sha256"]),
            "model_revision": str(self.config["model_revision"]),
            "dataset_revision": str(self.config["dataset_revision"]),
        }
        route_power = int(self.config.get("route_weight_power", 2))
        retained_powers = tuple(int(value) for value in self.config.get("retained_powers", (route_power,)))
        if route_power not in retained_powers:
            raise ValueError("retained_powers must include the configured route_weight_power")
        fitter_options = {
            "regularization_floor": float(self.config.get("regularization_floor", 1e-12)),
            "retained_accounting": tuple(self.config.get("retained_accounting", ("combined",))),
            "retained_powers": retained_powers,
            "covariance_mode": str(self.config.get("covariance_mode", "full")),
            "block_size": int(self.config.get("covariance_block_size", 128)),
            "artifact_dtype": str(self.config.get("artifact_dtype", "float32")),
        }
        fitter_backend = str(self.config.get("fitter_backend", "numpy_full"))
        if fitter_backend not in {"numpy_full", "torch_full_p2"}:
            raise ValueError("fitter_backend must be numpy_full or torch_full_p2")
        if fitter_backend == "torch_full_p2" and (
            route_power != 2
            or retained_powers != (2,)
            or tuple(fitter_options["retained_accounting"]) != ("combined",)
            or fitter_options["covariance_mode"] != "full"
            or fitter_options["artifact_dtype"] != "float32"
            or cold_min != 0
        ):
            raise ValueError(
                "torch_full_p2 requires the natural-only competitive combined/full/FP32 p2 estimator"
            )
        fit_inventory: list[dict[str, Any]] = []
        route_audits: list[dict[str, Any]] = []
        experts = list(range(int(manifest["geometry"]["experts"])))
        supplemental_component = service_capture["captures"].get("supplemental")
        supplemental_manifest = None
        supplemental_root = None
        inclusion_numerator = int(self.config.get("supplemental_inclusion_numerator", 1))
        inclusion_denominator = int(self.config.get("supplemental_inclusion_denominator", 1))
        if supplemental_component is not None:
            if fitter_backend == "torch_full_p2":
                raise ValueError("torch_full_p2 does not admit supplemental rows in this experiment")
            supplemental_root = capture_file.parent / supplemental_component["manifest"].rsplit("/", 1)[0]
            supplemental_manifest = verify_capture_manifest(supplemental_root, verify_chunks=False)
            if supplemental_manifest["layers"] != manifest["layers"] or supplemental_manifest["geometry"] != manifest["geometry"]:
                raise ValueError("supplemental and natural capture geometry/layer inventory differ")
            if not 0 < inclusion_numerator <= inclusion_denominator:
                raise ValueError("supplemental inclusion probability must be an exact rational in (0,1]")
        captured_layers = tuple(int(value) for value in manifest["layers"])
        if context.get("layer") is None:
            target_layers = captured_layers
        else:
            requested_layer = int(context["layer"])
            if requested_layer not in captured_layers:
                raise ValueError(f"requested fit layer {requested_layer} is absent from capture")
            target_layers = (requested_layer,)
        for layer in target_layers:
            gate_up = None
            down = None
            if fitter_backend == "numpy_full":
                gate_up = CalibrationFitter(
                    layer_id=layer,
                    projection="gate_up_input",
                    hidden_size=int(manifest["geometry"]["hidden_size"]),
                    predecessor_checkpoint_hash=context["predecessor_state_hash"],
                    source_identities=source_ids,
                    **fitter_options,
                )
                down = CalibrationFitter(
                    layer_id=layer,
                    projection="down_input",
                    hidden_size=int(manifest["geometry"]["intermediate_size"]),
                    predecessor_checkpoint_hash=context["predecessor_state_hash"],
                    source_identities=source_ids,
                    **fitter_options,
                )
            torch_values: dict[str, dict[int, list[np.ndarray]]] = {
                "gate_up": defaultdict(list),
                "down": defaultdict(list),
            }
            torch_weights: dict[int, list[np.ndarray]] = defaultdict(list)
            torch_documents: dict[int, list[list[str]]] = defaultdict(list)
            torch_offsets: dict[int, list[np.ndarray]] = defaultdict(list)
            torch_route_mass: dict[int, dict[str, int]] = defaultdict(
                lambda: {"count": 0, "weight_units": 0, "weight_square_units": 0}
            )
            natural_rows: list[RouteMassRow] = []
            supplemental_rows: list[RouteMassRow] = []
            supplemental_values: dict[str, tuple[np.ndarray, float, str, int]] = {}

            def consume(record: Mapping[str, Any], root: Path, *, origin: str) -> None:
                metadata, tensors = _tensor_file(root / record["file"])
                expert_ids = tensors["assignment_expert_ids"].numpy().astype(np.int64)
                offsets = tensors["assignment_token_offsets"].numpy().astype(np.int64)
                # The fitter contract intentionally requires the captured FP32
                # route weights. It promotes internally for float64 moments;
                # pre-promoting here would both violate the contract and hide
                # accidental capture dtype drift.
                weights = tensors["assignment_router_weights"].numpy().astype(np.float32, copy=False)
                hidden_values = tensors["routed_hidden_states"].float().numpy()
                down_values = tensors["routed_down_inputs"].float().numpy()
                window_document = f"{metadata['document_id']}@{metadata['token_sha256']}"
                documents = [window_document] * len(expert_ids)
                absolute_offsets = offsets + int(metadata["start_token"])
                common = {
                    "expert_ids": expert_ids,
                    "route_weights": weights,
                    "document_ids": documents,
                    "token_offsets": absolute_offsets,
                    "layer_id": layer,
                    "predecessor_checkpoint_hash": context["predecessor_state_hash"],
                    "origins": origin,
                }
                if origin == "natural":
                    if fitter_backend == "torch_full_p2":
                        scaled = weights.astype(np.float64) * denominator
                        units = np.rint(scaled).astype(np.int64)
                        if (
                            np.any(units < 1)
                            or np.any(units > denominator)
                            or np.any(np.abs(weights.astype(np.float64) - units / denominator) > tolerance)
                        ):
                            raise ValueError("captured router weights violate the exact integer audit grid")
                        for expert in np.unique(expert_ids):
                            selected = expert_ids == expert
                            selected_count = int(selected.sum())
                            torch_values["gate_up"][int(expert)].append(np.ascontiguousarray(hidden_values[selected]))
                            torch_values["down"][int(expert)].append(np.ascontiguousarray(down_values[selected]))
                            torch_weights[int(expert)].append(np.ascontiguousarray(weights[selected]))
                            torch_documents[int(expert)].append([window_document] * selected_count)
                            torch_offsets[int(expert)].append(np.ascontiguousarray(absolute_offsets[selected]))
                            expert_units = units[selected]
                            mass = torch_route_mass[int(expert)]
                            mass["count"] += selected_count
                            mass["weight_units"] += int(expert_units.sum(dtype=np.int64))
                            mass["weight_square_units"] += int(
                                np.dot(expert_units.astype(object), expert_units.astype(object))
                            )
                    else:
                        assert gate_up is not None and down is not None
                        gate_up.update(CalibrationBatch(expert_inputs=hidden_values, projection="gate_up_input", **common))
                        down.update(CalibrationBatch(expert_inputs=down_values, projection="down_input", **common))
                if fitter_backend != "torch_full_p2":
                    for index, (expert, offset, weight) in enumerate(
                        zip(expert_ids, absolute_offsets, weights, strict=True)
                    ):
                        units = _quantize_route_weight(float(weight), denominator, tolerance)
                        for role in ("gate_up", "down"):
                            row = RouteMassRow(
                                int(expert), role, window_document, int(offset), units,
                                origin=origin,
                                inclusion_numerator=inclusion_numerator if origin == "supplemental" else 1,
                                inclusion_denominator=inclusion_denominator if origin == "supplemental" else 1,
                            )
                            (natural_rows if origin == "natural" else supplemental_rows).append(row)
                            if origin == "supplemental":
                                value = hidden_values[index] if role == "gate_up" else down_values[index]
                                supplemental_values[row.row_identity] = (
                                    value, float(weight), window_document, int(offset)
                                )

            for record in manifest["records"][str(layer)]:
                consume(record, fit_capture_root, origin="natural")
            if supplemental_manifest is not None and supplemental_root is not None:
                for record in supplemental_manifest["records"][str(layer)]:
                    consume(record, supplemental_root, origin="supplemental")
            if fitter_backend == "torch_full_p2":
                compact_accounting = [
                    {"expert_id": str(expert), **torch_route_mass[expert]}
                    for expert in experts
                ]
                compact_audit = {
                    "schema": "quant-pipeline.route-mass-natural-compact.v1",
                    "identity": {
                        "capture_sha256": manifest["capture_sha256"],
                        "layer": layer,
                        "expert_ids": [str(expert) for expert in experts],
                        "roles": ["gate_up", "down"],
                        "unit_denominator": denominator,
                    },
                    "policy": {
                        "powers": [0, 1, 2],
                        "accounting": "natural-only-exact-integer-power-sums",
                        "cold_expert_min_weight_units": 0,
                        "supplemental_rows": 0,
                    },
                    "accounting": compact_accounting,
                }
                compact_audit["audit_sha256"] = _hash_json(compact_audit)
                selected_by_role = {"gate_up": set(), "down": set()}
            else:
                role_identity = {
                    role: _hash_json(sorted(
                        row.row_identity
                        for row in (*natural_rows, *supplemental_rows)
                        if row.role == role
                    ))
                    for role in ("gate_up", "down")
                }
                audit = build_route_mass_audit(
                    natural_rows=natural_rows,
                    supplemental_pool=supplemental_rows,
                    expert_ids=experts,
                    unit_denominator=denominator,
                    cold_expert_min_weight_units=cold_min,
                    topup_seed_sha256=seed_hash,
                    role_row_identity_sha256=role_identity,
                )
                selected_by_role = {
                    role: {
                        identity
                        for row in audit.metadata["accounting"] if row["role"] == role
                        for identity in row["supplemental_row_identities"]
                    }
                    for role in ("gate_up", "down")
                }
            inclusion = inclusion_numerator / inclusion_denominator
            for role, fitter in (("gate_up", gate_up), ("down", down)):
                selected = sorted(selected_by_role[role])
                if not selected:
                    continue
                if fitter is None:
                    raise AssertionError("supplemental rows reached the torch_full_p2 fitter")
                values = np.stack([supplemental_values[identity][0] for identity in selected])
                weights = np.asarray([supplemental_values[identity][1] for identity in selected], dtype=np.float64)
                documents = [supplemental_values[identity][2] for identity in selected]
                offsets = np.asarray([supplemental_values[identity][3] for identity in selected], dtype=np.int64)
                expert_by_identity = {row.row_identity: int(row.expert_id) for row in supplemental_rows if row.role == role}
                fitter.update(CalibrationBatch(
                    expert_inputs=values,
                    expert_ids=np.asarray([expert_by_identity[identity] for identity in selected], dtype=np.int64),
                    route_weights=weights,
                    document_ids=documents,
                    token_offsets=offsets,
                    layer_id=layer,
                    predecessor_checkpoint_hash=context["predecessor_state_hash"],
                    projection="gate_up_input" if role == "gate_up" else "down_input",
                    origins="supplemental",
                    inclusion_probabilities=np.full(len(selected), inclusion, dtype=np.float64),
                ))
            if fitter_backend == "torch_full_p2":
                route_audits.append({"layer": layer, "audit": compact_audit})
            else:
                route_audits.append({
                    "layer": layer,
                    "audit": audit.metadata,
                    "audit_sha256": audit.content_sha256,
                })
            if fitter_backend == "torch_full_p2":
                observed = set(torch_weights) & set(torch_values["gate_up"]) & set(torch_values["down"])
                missing = [str(expert) for expert in experts if expert not in observed]
            else:
                assert gate_up is not None and down is not None
                observed = set(gate_up.expert_ids) & set(down.expert_ids)
                missing = [str(expert) for expert in experts if str(expert) not in observed]
            if missing:
                raise ValueError(
                    f"capture has no natural or selected supplemental support for layer {layer} experts {missing[:8]}"
                )
            for expert in experts:
                gate_path = output / f"layer-{layer:03d}" / f"expert-{expert:03d}" / "gate-up"
                down_path = output / f"layer-{layer:03d}" / f"expert-{expert:03d}" / "down"
                if fitter_backend == "torch_full_p2":
                    device = str(self.config.get("fitter_device", "cuda:0"))
                    shell_options = {
                        "layer": layer,
                        "expert": expert,
                        "predecessor_checkpoint_hash": context["predecessor_state_hash"],
                        "source_identities": source_ids,
                        "route_weights": torch_weights[expert],
                        "document_ids": torch_documents[expert],
                        "token_offsets": torch_offsets[expert],
                        "regularization_floor": float(fitter_options["regularization_floor"]),
                    }
                    gate_result = _torch_full_p2_statistics(
                        _full_p2_accounting_shell(
                            projection="gate_up_input",
                            hidden_size=int(manifest["geometry"]["hidden_size"]),
                            **shell_options,
                        ),
                        torch_values["gate_up"][expert],
                        torch_weights[expert],
                        device=device,
                    )
                    down_result = _torch_full_p2_statistics(
                        _full_p2_accounting_shell(
                            projection="down_input",
                            hidden_size=int(manifest["geometry"]["intermediate_size"]),
                            **shell_options,
                        ),
                        torch_values["down"][expert],
                        torch_weights[expert],
                        device=device,
                    )
                else:
                    assert gate_up is not None and down is not None
                    gate_result = gate_up.finalize(expert)
                    down_result = down.finalize(expert)
                save_fitted_statistics(gate_path, gate_result)
                save_fitted_statistics(down_path, down_result)
                fit_inventory.append({
                    "layer": layer,
                    "expert": expert,
                    "gate_up": gate_path.relative_to(output).as_posix(),
                    "down": down_path.relative_to(output).as_posix(),
                    "gate_up_manifest_sha256": sha256_file(gate_path / "manifest.json"),
                    "down_manifest_sha256": sha256_file(down_path / "manifest.json"),
                })
        fit_manifest = {
            "schema": FIT_MANIFEST_SCHEMA,
            "predecessor_state_hash": context["predecessor_state_hash"],
            "capture_sha256": manifest["capture_sha256"],
            "capture_service_sha256": service_capture["capture_service_sha256"],
            "source_identities": source_ids,
            "row_identity_policy": "document-id-plus-token-sha256-plus-captured-offset-v1",
            "layers": list(target_layers),
            "estimator": {
                "route_weight_power": route_power,
                "fitter_backend": fitter_backend,
                "fitter_device": str(self.config.get("fitter_device", "cpu")),
                **{key: list(value) if isinstance(value, tuple) else value for key, value in fitter_options.items()},
            },
            "route_mass_audits": route_audits,
            "route_mass_audits_sha256": _hash_json(route_audits),
            "prior_controls": {
                "permutation_policies": ["identity", "ldlq_visit_descending_diag", "stored_descending_diag", "energy_balanced", "energy_balanced_contiguous"],
                "scale_families": ["identity", "per128-grid", "inverse-per128-grid"],
                "candidate_policy": "controls-plus-additive-ablation",
            },
            "statistics": fit_inventory,
        }
        fit_manifest["fit_sha256"] = _hash_json(fit_manifest)
        write_json(output / "fit-manifest.json", fit_manifest)
        transient_files = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file() and path.name != "fit-manifest.json"
        )
        if not transient_files:
            raise RuntimeError("fitter produced no retireable covariance/vector artifacts")
        return {
            "fit_manifest_file": "fit-manifest.json",
            "transient_files": transient_files,
        }


class QwenCodecService:
    """Corrected codec identity plus exact-payload causal installation."""

    def __init__(self, config: Mapping[str, Any], codec: Exl3MCGCodec) -> None:
        self.config = dict(config)
        self.codec = codec

    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "corrected-exl3-mcg-r10",
            "test_only": False,
            "numeric_identity": self.codec.identity,
            "numeric_identity_sha256": _hash_json(self.codec.identity),
        }

    @staticmethod
    def _load_ref(store_root: Path, ref: Mapping[str, Any]):
        import torch

        root = store_root.resolve()
        path = (root / str(ref["path"])).resolve()
        if (
            root not in path.parents
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != ref["sha256"]
            or path.stat().st_size != int(ref["bytes"])
        ):
            raise ValueError("candidate exact-payload object is missing or corrupt")
        dtype = getattr(torch, str(ref["dtype"]))
        raw = bytearray(path.read_bytes())
        return torch.frombuffer(raw, dtype=dtype).clone().reshape(tuple(int(x) for x in ref["shape"]))

    def install(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        from ..candidates.ledger import validate_ledger

        candidate_root = _dependency(context, "causal_candidates")
        ledger_path = _result_path(candidate_root, "candidate_ledger_file")
        ledger = json.loads(ledger_path.read_text())
        competitive = bool(ledger.get("competitive"))
        validate_ledger(ledger, competitive=competitive, allow_test_backend=not competitive)
        allocation_root = _dependency(context, "allocation")
        allocation = json.loads(_result_path(allocation_root, "allocation_file").read_text())
        _validate_allocation_document(allocation)
        layer = int(context["layer"])
        requested_arm_name, serving = _requested_allocation_arm(allocation, self.config)
        selected_rows = [
            dict(row)
            for row in serving.get("choices", ())
            if int(str(row["unit_id"]).split(".", 1)[0][1:]) == layer
        ]
        selected_by_unit = {str(row["unit_id"]): row for row in selected_rows}
        if len(selected_by_unit) != len(selected_rows):
            raise ValueError("serving allocation contains duplicate expert units")
        selected_ids = {unit: str(row["choice_id"]) for unit, row in selected_by_unit.items()}
        records = {row["candidate_id"]: row for row in ledger["candidates"] if int(row["layer"]) == layer}
        selected_records = [records[choice] for _unit, choice in sorted(selected_ids.items())]
        if not selected_records:
            raise ValueError(f"requested {requested_arm_name} has no selected candidates for layer {layer}")
        selected_cost = serving.get("selected_cost")
        if not isinstance(selected_cost, Mapping):
            raise ValueError(f"requested {requested_arm_name} lacks its reconciled selected_cost")
        layer_costs = [row for row in selected_cost.get("selected_layer_costs", ()) if int(row["layer"]) == layer]
        if len(layer_costs) != 1:
            raise ValueError(f"requested {requested_arm_name} selected_cost lacks exactly one layer {layer} row")
        expected_layer_cost = layer_costs[0]
        selected_hashes = sorted(str(row.get("candidate_record_sha256")) for row in selected_rows)
        if selected_hashes != expected_layer_cost["selected_candidate_record_sha256"]:
            raise ValueError("requested arm layer cost hashes differ from the exact selected allocation choices")
        store_root = candidate_root / "journal" / "payloads"
        choices = []
        for record in selected_records:
            for projection in _PROJECTIONS:
                row = record["projections"][projection]
                refs = row["exact_payload_refs"]
                choices.append({
                    "expert": int(record["expert"]),
                    "projection": projection,
                    "choice_id": record["candidate_id"],
                    "bits": int(row["bits"]),
                    "tensors": {
                        "trellis": self._load_ref(store_root, refs["packed_trellis"]),
                        "suh": self._load_ref(store_root, refs["suh"]),
                        "svh": self._load_ref(store_root, refs["svh"]),
                        "reconstruction": self._load_ref(store_root, refs["reconstruction_hf"]),
                    },
                    "vector_topology": {
                        "suh": "layer_shared" if projection != "down_proj" else "expert_private",
                        "svh": "expert_private" if projection != "down_proj" else "layer_shared",
                    },
                    "provenance": {
                        "candidate_record_sha256": record["record_sha256"],
                        "selected_candidate_record_sha256": selected_by_unit[str(record["unit_id"])]["candidate_record_sha256"],
                        "causal_candidate_record_sha256": record["record_sha256"],
                        "codec_identity_sha256": self.identity()["numeric_identity_sha256"],
                    },
                })
        fit_root = _dependency(context, "causal_fit")
        fit_path = _result_path(fit_root, "fit_manifest_file")
        expected_cost = int(expected_layer_cost["allocated_payload_bytes"])
        installed = install_layer_payloads(
            output_dir=_output(context),
            layer=layer,
            predecessor_state_hash=context["predecessor_state_hash"],
            source_checkpoint_sha256=context["input_identities"]["source_checkpoint"],
            fit_sha256=sha256_file(fit_path),
            candidate_ledger_sha256=sha256_file(ledger_path),
            selected_choices=choices,
            production_geometry=bool(context["production"]),
            expected_allocated_payload_bytes=expected_cost,
        )
        observed = installed["cost_breakdown"]
        for key in (
            "semantic_expert_private_bytes",
            "semantic_layer_shared_objects",
            "semantic_layer_shared_bytes",
            "allocated_payload_bytes",
        ):
            if observed[key] != expected_layer_cost[key]:
                raise RuntimeError(f"causal installed layer {layer} {key} differs from selected allocation")
        return {"installed_manifest_file": "manifest.json", "installed_checkpoint_sha256": installed["installed_checkpoint_sha256"]}


class QwenAllocatorService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def identity(self) -> Mapping[str, Any]:
        return {"schema": SERVICE_SCHEMA, "provider": "dual-arm-exact-dp", "official_btx_commit": UPSTREAM_COMMIT}

    def _serving_legal(self, record: Mapping[str, Any]) -> tuple[bool, str | None]:
        gate, up, down = map(int, record["bit_triplet"])
        if gate != up:
            return False, "official BTX master encodes one fc1 rate for gate/up"
        structure = str(self.config.get("btx_rate_structure", "per_expert_pair"))
        if structure == "per_expert_pair" and 5 in (gate, down):
            return False, "official BTX per_expert_pair has no K5 vocabulary"
        if bool(self.config.get("require_fused_btx", True)) and (gate == 4 or down == 4):
            return False, "P44 is schema-declared but not fused on pinned master"
        intermediate = int(self.config.get("moe_intermediate_size", 768))
        slots = intermediate // 32
        alignment = 8 if structure == "per_expert_pair" else 1
        for tp in self.config.get("target_tp_degrees", [1]):
            tp = int(tp)
            if tp < 1 or slots % tp or (slots // tp) % alignment:
                return False, f"TP{tp} cannot partition {slots} slots into {alignment}-slot extents"
        return True, None

    def allocate(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        from ..candidates.ledger import allocate_validated_records, validate_ledger

        ledger_root = _dependency(context, "candidates")
        ledger_path = _result_path(ledger_root, "candidate_ledger_file")
        ledger = json.loads(ledger_path.read_text())
        competitive = bool(ledger.get("competitive"))
        validate_ledger(ledger, competitive=competitive, allow_test_backend=not competitive)
        records = list(ledger["candidates"])
        attribution_root = _dependency(context, "attribution")
        attribution_path = _result_path(attribution_root, "attribution_file")
        attribution = json.loads(attribution_path.read_text())
        if attribution.get("schema") != ATTRIBUTION_SCHEMA:
            raise ValueError("allocator attribution schema mismatch")
        seal = attribution.get("attribution_sha256")
        if seal != _hash_json({key: value for key, value in attribution.items() if key != "attribution_sha256"}):
            raise ValueError("allocator attribution seal mismatch")
        if attribution.get("candidate_ledger_sha256") != sha256_file(ledger_path):
            raise ValueError("allocator attribution belongs to a different candidate ledger")

        provisional = tuple(int(value) for value in self.config.get("attribution_provisional_bit_triplet", [4, 4, 4]))
        direct_by_unit: dict[str, float] = {}
        for layer_row in attribution.get("layers", ()):
            layer = int(layer_row["layer_index"])
            allocation_scores = layer_row.get("expert_allocation_score_reconciled")
            if not isinstance(allocation_scores, list):
                raise ValueError("attribution lacks explicitly redistributed expert allocation scores")
            for expert, direct in enumerate(allocation_scores):
                direct_by_unit[f"L{layer}.E{expert}"] = float(direct)
        by_unit: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_unit.setdefault(str(record["unit_id"]), []).append(record)
        if set(direct_by_unit) != set(by_unit):
            raise ValueError("attribution expert inventory differs from candidate ledger")
        damage_overrides: dict[str, float] = {}
        calibration_rows = []
        for unit_id, unit_records in sorted(by_unit.items()):
            anchors = [row for row in unit_records if tuple(map(int, row["bit_triplet"])) == provisional]
            if len(anchors) != 1:
                raise ValueError(f"{unit_id} lacks one provisional attribution anchor")
            anchor = anchors[0]
            anchor_proxy = float(anchor["predicted_damage"])
            direct = direct_by_unit[unit_id]
            if anchor_proxy <= 0.0:
                raise ValueError(f"{unit_id} provisional proxy damage must be positive")
            scale = direct / anchor_proxy
            raw = {str(row["candidate_id"]): scale * float(row["predicted_damage"]) for row in unit_records}
            offset = max(0.0, -min(raw.values()))
            for candidate_id, value in raw.items():
                damage_overrides[candidate_id] = value + offset
            calibration_rows.append({
                "unit_id": unit_id,
                "expert_allocation_score_reconciled": direct,
                "provisional_candidate_id": anchor["candidate_id"],
                "provisional_proxy_damage": anchor_proxy,
                "signed_scale": scale,
                "nonnegative_unit_offset": offset,
                "provisional_unshifted_damage": raw[str(anchor["candidate_id"])],
                "provisional_shifted_damage": raw[str(anchor["candidate_id"])] + offset,
            })
        budget_value = self.config["byte_budget"] if "byte_budget" in self.config else self.config["exact_payload_byte_budget"]
        budget = int(budget_value)
        allocation_options = {
            "byte_budget": budget,
            "quantum": int(self.config.get("allocation_quantum", 1)),
            "competitive": competitive,
            "allow_test_backend": not competitive,
        }
        proxy_control = allocate_validated_records(records, **allocation_options)
        research = allocate_validated_records(
            records,
            damage_overrides=damage_overrides,
            **allocation_options,
        )
        rejected = []
        legal = []
        for record in records:
            ok, reason = self._serving_legal(record)
            (legal if ok else rejected).append(record if ok else {
                "candidate_id": record["candidate_id"], "unit_id": record["unit_id"], "reason": reason,
            })
        legal_units = {str(row["unit_id"]) for row in legal}
        missing_legal_units = sorted(set(by_unit) - legal_units)
        if missing_legal_units:
            raise ValueError(
                "official BTX filter leaves expert units without a legal candidate: "
                + ", ".join(missing_legal_units)
            )
        legal_overrides = {str(row["candidate_id"]): damage_overrides[str(row["candidate_id"])] for row in legal}
        serving = allocate_validated_records(
            legal,
            damage_overrides=legal_overrides,
            **allocation_options,
        )

        offset_by_unit = {str(row["unit_id"]): float(row["nonnegative_unit_offset"]) for row in calibration_rows}

        records_by_id = {str(row["candidate_id"]): row for row in records}

        def arm(value: Any, *, shifted: bool) -> dict[str, Any]:
            allocation = value.allocation
            choices = [{
                "unit_id": row.unit_id,
                "choice_id": row.choice_id,
                "stored_bytes": row.stored_bytes,
                "predicted_damage": row.predicted_damage,
                "candidate_record_sha256": row.metadata["record_sha256"],
                "bit_triplet": list(records_by_id[row.choice_id]["bit_triplet"]),
            } for row in allocation.choices]
            total_offset = sum(offset_by_unit[row["unit_id"]] for row in choices) if shifted else 0.0
            return {
                "choices": choices,
                "variable_payload_bytes": allocation.variable_payload_bytes,
                "fixed_layer_shared_bytes": allocation.fixed_layer_shared_bytes,
                "stored_bytes": allocation.stored_bytes,
                "predicted_damage": allocation.predicted_damage,
                "selected_unit_offset_total": total_offset,
                "unshifted_predicted_damage": allocation.predicted_damage - total_offset,
                "selected_cost": dict(value.selected_cost),
            }

        research_arm = arm(research, shifted=True)
        runtime_arm = arm(serving, shifted=True)
        requested_arm = str(self.config.get("requested_allocation_arm", "research_arm"))
        if requested_arm not in {"research_arm", "runtime_qualified_arm"}:
            raise ValueError("requested_allocation_arm must name research_arm or runtime_qualified_arm")
        body = {
            "schema": ALLOCATION_SCHEMA,
            "ledger_sha256": ledger["ledger_sha256"],
            "attribution_file_sha256": sha256_file(attribution_path),
            "attribution_sha256": seal,
            "byte_budget": budget,
            "proxy_control_arm": arm(proxy_control, shifted=False),
            "research_arm": research_arm,
            "runtime_qualified_arm": runtime_arm,
            # Historical readers used serving_arm.  Retain the byte-identical
            # alias while making its meaning explicit and never selecting it
            # implicitly in the shipped production configuration.
            "serving_arm": runtime_arm,
            "requested_allocation_arm": requested_arm,
            "arm_contract": {
                "research": "unconstrained-scientific-exact-byte-optimization",
                "runtime_qualified": "official-btx-schema-and-fused-kernel-filtered",
                "runtime_matches_research": _arm_choice_identity(research_arm)
                == _arm_choice_identity(runtime_arm),
                "substitution_policy": "forbidden",
            },
            "shapley_damage_calibration": {
                "method": "signed-provisional-ratio-with-unit-constant-offset-v1",
                "provisional_bit_triplet": list(provisional),
                "unit_rows": calibration_rows,
                "offset_policy": "per-unit constants preserve every within-unit allocation ordering",
            },
            "official_btx_filter": {
                "upstream_commit": UPSTREAM_COMMIT,
                "rejected": rejected,
                "retained_candidate_count": len(legal),
                "require_fused": bool(self.config.get("require_fused_btx", True)),
                "target_tp_degrees": [int(x) for x in self.config.get("target_tp_degrees", [1])],
            },
        }
        body["allocation_sha256"] = _hash_json(body)
        write_json(_output(context) / "allocation.json", body)
        return {"allocation_file": "allocation.json"}


class QwenEvaluatorService:
    def __init__(self, config: Mapping[str, Any], capturer: QwenRouteCaptureService) -> None:
        self.config = dict(config)
        self.capturer = capturer
        self._student_runtime = None
        self._student_runtime_identity = None

    def identity(self) -> Mapping[str, Any]:
        self._runtime()
        assert self._student_runtime_identity is not None
        return {
            "schema": SERVICE_SCHEMA,
            "provider": "aumann-shapley-fisher-kld",
            "remainder": "explicit",
            "student_runtime": self._student_runtime_identity,
        }

    def _runtime(self):
        if self._student_runtime is None:
            self._student_runtime, self._student_runtime_identity = _load_pinned_student_runtime(self.config)
        return self._student_runtime

    def attribute(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        ledger_root = _dependency(context, "candidates")
        provider = _provider_result(ledger_root)
        provisional_name = provider.get("provisional_winner_manifest_file")
        if not isinstance(provisional_name, str):
            raise ValueError("candidate service must emit a sealed provisional winner manifest")
        provisional = _result_path(ledger_root, "provisional_winner_manifest_file")
        provisional_document = json.loads(provisional.read_text())
        ledger_path = _result_path(ledger_root, "candidate_ledger_file")
        ledger = json.loads(ledger_path.read_text())
        if provisional_document.get("candidate_ledger_sha256") != ledger.get("ledger_sha256"):
            raise ValueError("provisional winner manifest belongs to a different candidate ledger")
        triplet = tuple(provisional_document.get("bit_triplet", ()))
        expected_winners = {
            (str(row["candidate_id"]), str(row["record_sha256"]))
            for row in ledger.get("candidates", ())
            if tuple(row.get("bit_triplet", ())) == triplet
        }
        observed_winners = {
            (str(winner["candidate_id"]), str(winner["candidate_record_sha256"]))
            for layer_row in provisional_document.get("layers", ())
            for winner in layer_row.get("winners", ())
        }
        if not expected_winners or observed_winners != expected_winners:
            raise ValueError("provisional winner inventory differs from its exact ledger records")
        teacher_root = _dependency(context, "teacher_capture")
        teacher = _result_path(teacher_root, "teacher_reference_file")
        inputs_path = produce_qwen_attribution_inputs_from_local(
            source_checkpoint=context["inputs"]["source_checkpoint"],
            model_revision=str(self.config["model_revision"]),
            kld_window=context["inputs"]["kld_window"],
            teacher_reference=teacher,
            provisional_manifest=provisional,
            payload_store_root=ledger_root / "journal" / "payloads",
            output_path=_output(context) / "attribution-inputs.npz",
            device_map=self.config.get("attribution_device_map", self.config.get("device_map", "auto")),
            attn_implementation=str(self.config.get("attention_backend", "eager")),
            path_nodes=int(self.config.get("attribution_path_nodes", 5)),
            fisher_rank=int(self.config.get("attribution_fisher_rank", 8)),
            seed=int(self.config.get("attribution_seed", 20260823)),
        )
        archive, receipt = verify_attribution_inputs(inputs_path)
        layer_indices = np.asarray(archive["layer_indices"], dtype=np.int64)
        deltas = np.asarray(archive["layer_deltas"], dtype=np.float64)
        gradients = np.asarray(archive["path_gradients"], dtype=np.float64)
        projected = np.asarray(archive["projected_expert_residuals"], dtype=np.float64)
        projected_routing = np.asarray(archive["projected_routing_residuals"], dtype=np.float64)
        path_weights = np.asarray(archive["path_quadrature_weights"], dtype=np.float64)
        measured = np.asarray(archive["measured_layer_damage"], dtype=np.float64)
        source_kld = float(np.asarray(archive["source_kld"]).reshape(-1)[0])
        candidate_kld = float(np.asarray(archive["candidate_kld"]).reshape(-1)[0])
        measured_end_to_end = float(np.asarray(archive["measured_end_to_end_delta"]).reshape(-1)[0])
        if gradients.ndim < deltas.ndim + 1:
            raise ValueError("attribution path gradients must be [nodes,layers,...]")
        nodes = gradients.shape[0]
        # The producer stores Gauss-Legendre evaluations in canonical order;
        # interpolate only at those exact nodes to keep the receipt auditable.
        expected_nodes, _weights = np.polynomial.legendre.leggauss(nodes)
        expected_nodes = (expected_nodes + 1.0) / 2.0
        def gradient_at(t: float):
            matches = np.flatnonzero(np.isclose(expected_nodes, t, rtol=0.0, atol=1e-14))
            if len(matches) != 1:
                raise ValueError("requested Aumann-Shapley node was not sealed")
            return list(gradients[int(matches[0])])
        layer_raw = aumann_shapley(list(deltas), gradient_at, path_nodes=nodes)
        if not np.allclose(layer_raw, measured, rtol=1e-10, atol=1e-12):
            raise ValueError("sealed measured layer damage differs from recomputed Aumann-Shapley integral")
        layer_accounting = reconcile_signed_completeness(layer_raw, measured_end_to_end)
        layers = []
        for index, total in enumerate(measured):
            split = split_layer_damage(
                float(total),
                projected[index],
                projected_routing_residual=projected_routing[index],
                observation_weights=path_weights,
            )
            reconciled_layer = float(layer_accounting.reconciled[index])
            components = reconcile_layer_components_for_allocation(
                expert_direct=split["expert_direct"],
                routing_state_shift=split["routing_state_shift"],
                explicit_residual=split["unresolved_nonlinear_remainder"],
                raw_layer_damage=float(layer_raw[index]),
                reconciled_layer_damage=reconciled_layer,
            )
            layers.append({
                "layer_index": int(layer_indices[index]),
                "aumann_shapley": float(layer_raw[index]),
                "reconciled_layer_damage": reconciled_layer,
                "expert_direct": split["expert_direct"],
                "routing_state_shift_raw": split["routing_state_shift"],
                "within_layer_unresolved_remainder_raw": split["unresolved_nonlinear_remainder"],
                "raw_joint_fisher_total": split["raw_total"],
                "closed_total": split["closed_total"],
                **components,
            })
        layer_total = float(np.sum(layer_raw))
        path_remainder = float(measured_end_to_end - layer_total)
        body = {
            "schema": ATTRIBUTION_SCHEMA,
            "candidate_ledger_sha256": sha256_file(_result_path(ledger_root, "candidate_ledger_file")),
            "inputs_sha256": sha256_file(inputs_path),
            "inputs_receipt_sha256": receipt["receipt_sha256"],
            "path_nodes": int(nodes),
            "layers": layers,
            "source_kld": source_kld,
            "candidate_kld": candidate_kld,
            "measured_end_to_end_delta": measured_end_to_end,
            "sum_measured_layer_damage": layer_total,
            "unresolved_path_quadrature_and_nonlinear_remainder": path_remainder,
            "closed_end_to_end_delta": layer_total + path_remainder,
            "sum_closed_damage": float(sum(row["closed_total"] for row in layers)),
            "sum_reconciled_expert_damage": float(sum(row["reconciled_expert_total"] for row in layers)),
            "remainder_policy": "explicit-route-and-unresolved-components-with-allocation-only-redistribution-v2",
        }
        body["attribution_sha256"] = _hash_json(body)
        write_json(_output(context) / "attribution.json", body)
        return {"attribution_file": "attribution.json"}

    def confirm(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Evaluate the exact requested allocation arm on the confirmation fold."""

        import torch
        from transformers import AutoModelForCausalLM

        from ..calibration.windows import verify_sealed_corpus
        from ..candidates.ledger import validate_ledger

        candidate_root = _dependency(context, "candidates")
        ledger_path = _result_path(candidate_root, "candidate_ledger_file")
        ledger = json.loads(ledger_path.read_text())
        competitive = bool(ledger.get("competitive"))
        validate_ledger(ledger, competitive=competitive, allow_test_backend=not competitive)
        allocation_root = _dependency(context, "allocation")
        allocation_path = _result_path(allocation_root, "allocation_file")
        allocation = json.loads(allocation_path.read_text())
        _validate_allocation_document(allocation)
        requested_arm_name, requested_arm = _requested_allocation_arm(
            allocation, self.config
        )
        choices = list(requested_arm.get("choices", ()))
        if not choices:
            raise ValueError(
                f"confirmation gate requires frozen {requested_arm_name} allocation choices"
            )
        records = {str(row["candidate_id"]): row for row in ledger["candidates"]}
        selected = []
        for choice in choices:
            candidate_id = str(choice["choice_id"])
            record = records.get(candidate_id)
            if record is None or record.get("record_sha256") != choice.get("candidate_record_sha256"):
                raise ValueError("confirmation allocation choice differs from its exact candidate record")
            selected.append(record)

        corpus_path = Path(context["inputs"]["sealed_corpus"])
        corpus = json.loads(corpus_path.read_text())
        verify_sealed_corpus(corpus)
        if corpus.get("schema") != "quant-pipeline.sealed-corpus.v2":
            raise ValueError("confirmation gate requires sealed-corpus v2 with conditional_fit separated")
        windows = corpus["windows"]["confirmation"]
        source = Path(context["inputs"]["source_checkpoint"]).resolve()
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map=self.config.get("device_map", "auto"),
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=str(self.config.get("attention_backend", "eager")),
        ).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        device = model.get_input_embeddings().weight.device

        def capture() -> np.ndarray:
            rows = []
            with torch.inference_mode():
                for window in windows:
                    ids = torch.tensor([window["token_ids"]], dtype=torch.long, device=device)
                    logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, :-1]
                    rows.append(logits.float().cpu().reshape(-1, logits.shape[-1]).numpy())
            return np.concatenate(rows, axis=0)

        teacher = capture()
        blocks = qwen_moe_layers(model)
        store_root = candidate_root / "journal" / "payloads"
        with torch.no_grad():
            for record in selected:
                layer = int(record["layer"])
                expert = int(record["expert"])
                if layer not in blocks:
                    raise ValueError(f"confirmation choice names absent Qwen MoE layer {layer}")
                module = blocks[layer].experts
                intermediate = module.gate_up_proj.shape[1] // 2
                decoded = {
                    projection: QwenCodecService._load_ref(
                        store_root,
                        record["projections"][projection]["exact_payload_refs"]["reconstruction_hf"],
                    )
                    for projection in _PROJECTIONS
                }
                module.gate_up_proj[expert, :intermediate].copy_(
                    decoded["gate_proj"].to(module.gate_up_proj.device, module.gate_up_proj.dtype)
                )
                module.gate_up_proj[expert, intermediate:].copy_(
                    decoded["up_proj"].to(module.gate_up_proj.device, module.gate_up_proj.dtype)
                )
                module.down_proj[expert].copy_(
                    decoded["down_proj"].to(module.down_proj.device, module.down_proj.dtype)
                )
        student = capture()
        output = _output(context)
        reference = output / "confirmation-teacher-logits.npy"
        candidate = output / "confirmation-student-logits.npy"
        np.save(reference, teacher, allow_pickle=False)
        np.save(candidate, student, allow_pickle=False)
        frozen = {
            "candidate_ledger_sha256": ledger["ledger_sha256"],
            "candidate_ledger_file_sha256": sha256_file(ledger_path),
            "allocation_sha256": allocation["allocation_sha256"],
            "allocation_file_sha256": sha256_file(allocation_path),
            "selected_candidate_record_sha256": sorted(
                str(row["record_sha256"]) for row in selected
            ),
        }
        receipt = {
            "schema": "quant-pipeline.qwen-post-freeze-confirmation-capture.v1",
            "role": "confirmation",
            "prospective_only": True,
            "frozen": frozen,
            "requested_allocation_arm": requested_arm_name,
            "reference_sha256": sha256_file(reference),
            "capture_sha256": sha256_file(candidate),
            "prediction_positions": int(teacher.shape[0]),
            "logit_shape": list(teacher.shape),
        }
        receipt["receipt_sha256"] = _hash_json(receipt)
        write_json(output / "confirmation-capture-receipt.json", receipt)
        return {
            "reference_file": reference.name,
            "capture_file": candidate.name,
            "confirmation_capture_receipt_file": "confirmation-capture-receipt.json",
        }

    def reanchor(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        output = _output(context)
        teacher_root = _dependency(context, "teacher_capture")
        reference = _result_path(teacher_root, "teacher_reference_file")
        target_reference = output / "teacher-logits.npy"
        target_reference.write_bytes(reference.read_bytes())
        capture = output / "student-logits.npy"
        np.save(capture, self.capturer._load_logits(context), allow_pickle=False)
        return {"reference_file": target_reference.name, "capture_file": capture.name}

    def capture_student(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        output = _output(context)
        emission_stage = _dependency(context, "checkpoint_emission")
        manifest = _result_path(emission_stage, "checkpoint_manifest_file")
        target = output / "student-logits.npy"
        result = self._runtime().capture(
            checkpoint_root=manifest.parent,
            source_checkpoint=Path(context["inputs"]["source_checkpoint"]),
            kld_window=Path(context["inputs"]["kld_window"]),
            output_path=target,
            model_revision=str(self.config["model_revision"]),
        )
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise RuntimeError("pinned student runtime did not report a successful capture")
        if not target.is_file() or target.is_symlink():
            raise RuntimeError("pinned student runtime did not write its bound logits")
        logits = np.load(target, allow_pickle=False)
        if logits.ndim < 2 or logits.shape[-1] < 2 or not np.isfinite(logits).all():
            raise ValueError("pinned student runtime produced malformed logits")
        assert self._student_runtime_identity is not None
        receipt = {
            "schema": "quant-pipeline.pinned-btx-student-capture.v1",
            "checkpoint_manifest_sha256": sha256_file(manifest),
            "runtime": self._student_runtime_identity,
            "runtime_result": dict(result),
            "student_logits_sha256": sha256_file(target),
            "student_logits_shape": list(logits.shape),
        }
        receipt["capture_receipt_sha256"] = _hash_json(receipt)
        write_json(output / "student-capture-receipt.json", receipt)
        return {"student_capture_file": target.name, "student_capture_receipt_file": "student-capture-receipt.json"}

    def final_kld(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        teacher = _result_path(_dependency(context, "teacher_capture"), "teacher_reference_file")
        student = _result_path(_dependency(context, "student_capture"), "student_capture_file")
        output = _output(context)
        reference = output / "teacher-logits.npy"
        capture = output / "student-logits.npy"
        reference.write_bytes(teacher.read_bytes())
        capture.write_bytes(student.read_bytes())
        return {"reference_file": reference.name, "capture_file": capture.name}


class QwenCheckpointService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def identity(self) -> Mapping[str, Any]:
        return {"schema": SERVICE_SCHEMA, "provider": "official-btx", "upstream_commit": UPSTREAM_COMMIT}

    def emit(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        installed = list(context["installed_layer_attempts"])
        allocation_root = _dependency(context, "allocation")
        allocation = json.loads(_result_path(allocation_root, "allocation_file").read_text())
        _validate_allocation_document(allocation)
        requested_arm_name, requested_arm = _requested_allocation_arm(allocation, self.config)
        selected_cost = requested_arm["selected_cost"]
        reconciliation = reconcile_installed_allocation(selected_cost, installed)
        reconciliation = dict(reconciliation) | {"requested_allocation_arm": requested_arm_name}
        write_json(_output(context) / "installed-allocation-reconciliation.json", reconciliation)
        compatibility = btx_compatibility_report(
            installed,
            require_fused=bool(self.config.get("require_fused_btx", True)),
            target_tp_degrees=tuple(int(x) for x in self.config.get("target_tp_degrees", [1])),
        )
        if not compatibility["compatible"]:
            raise RuntimeError("official BTX compatibility failed: " + "; ".join(compatibility["failures"]))
        checkpoint_root = _output(context) / "checkpoint"
        emit_official_btx_checkpoint(
            output_dir=checkpoint_root,
            installed_layers=installed,
            expected_allocated_payload_bytes=int(reconciliation["allocated_payload_bytes"]),
            require_fused=bool(self.config.get("require_fused_btx", True)),
            target_tp_degrees=tuple(int(x) for x in self.config.get("target_tp_degrees", [1])),
        )
        structural = audit_official_btx_checkpoint(checkpoint_root, require_runtime_reader=False)
        structural_total = structural.get("accounting", {}).get("source_semantic_allocated_payload_bytes")
        if not structural.get("ok") or int(structural_total if structural_total is not None else -1) != int(reconciliation["allocated_payload_bytes"]):
            raise RuntimeError("official BTX emission failed post-write allocation audit")
        write_json(_output(context) / "emission-accounting-audit.json", structural)
        return {
            "checkpoint_manifest_file": "checkpoint/btx-manifest.json",
            "checkpoint_manifest_sha256": structural["manifest_sha256"],
            "allocation_reconciliation_file": "installed-allocation-reconciliation.json",
            "emission_accounting_audit_file": "emission-accounting-audit.json",
        }

    def audit(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        emission_stage = _dependency(context, "checkpoint_emission")
        emission = _result_path(emission_stage, "checkpoint_manifest_file").parent
        reader = UpstreamBtxRuntimeReader(self.config["b12x_source_root"])
        result = audit_official_btx_checkpoint(
            emission,
            runtime_reader=reader,
            require_runtime_reader=True,
        )
        manifest_path = emission / "btx-manifest.json"
        reader_result = dict(result.get("runtime", {}))
        reader_identity = {
            "reader": "b12x.moe._shared.kernels.w4a16.btx",
            "upstream_commit": UPSTREAM_COMMIT,
            "closure_sha256": reader_result.get("source", {}).get("closure_sha256"),
        }
        report = {
            "schema": "quant-pipeline.pinned-btx-runtime-audit.v1",
            "ok": bool(result.get("ok")),
            "failures": list(result.get("failures", ())),
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "reader_identity": reader_identity,
            "reader_identity_sha256": _hash_json(reader_identity),
            "reader_result": reader_result,
            "reader_result_sha256": _hash_json(reader_result),
            "structural_audit": dict(result),
        }
        report["audit_sha256"] = _hash_json(report)
        write_json(_output(context) / "checkpoint-audit.json", report)
        if not report["ok"]:
            raise RuntimeError("pinned BTX runtime audit failed: " + "; ".join(report["failures"]))
        return {"audit_file": "checkpoint-audit.json"}


class QwenLedgerService:
    """Strict bridge to the canonical exact candidate generator.

    Candidate construction is intentionally delegated to a materializer in
    this same sealed source closure because it is the part that loads source
    checkpoint shards and streams held-out expert rows.  The materializer must
    return canonical ``ExpertCandidateInput`` objects; the generator then owns
    every byte-bearing encode, score, K5 decision, and ledger receipt.
    """

    def __init__(self, config: Mapping[str, Any], codec: Exl3MCGCodec) -> None:
        self.config = dict(config)
        self.codec = codec

    def identity(self) -> Mapping[str, Any]:
        return {"schema": SERVICE_SCHEMA, "provider": "canonical-exact-candidate-ledger", "proposal_domain": "five-policies-x-three-scale-families-plus-additive"}

    def generate(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        # This import boundary is explicit and sealed by QwenCampaignAdapter's
        # provider-source closure.  A clean node therefore cannot smuggle an
        # unpinned notebook object into production.
        from ..candidates.ledger import (
            CandidateJournal,
            CandidateLedgerGenerator,
            attest_corrected_exl3_mcg,
            expected_expert_inventory,
        )
        from .qwen_work_units import materialize_candidate_work_units

        output = _output(context)
        fit_root = _dependency(context, "causal_fit" if context["kind"] == "causal_candidates" else "fit")
        fit_path = _result_path(fit_root, "fit_manifest_file")
        work = materialize_candidate_work_units(context, self.config, self.codec, fit_path)
        units = list(work["experts"])
        inventory = expected_expert_inventory(
            [(int(item.layer), int(item.expert)) for item in units],
            profile="qwen3-30b-a3b-48x128x3" if context["production"] else "tiny-qwen-fixture",
            test_fixture=not bool(context["production"]),
        )
        attestation = attest_corrected_exl3_mcg(self.codec)
        run_identity = dict(work["run_identity"])
        run_identity["fit_artifact_sha256"] = sha256_file(fit_path)
        run_identity["predecessor_state_sha256"] = context["predecessor_state_hash"]
        run_identity["codec_attestation_sha256"] = _hash_json(attestation.as_dict())
        run_identity["expected_inventory"] = inventory
        run_identity["expected_inventory_sha256"] = inventory["inventory_sha256"]
        from dataclasses import replace

        rebound = []
        for item in units:
            fitted = {}
            for projection, value in item.fitted.items():
                fitted[projection] = replace(
                    value,
                    fit_identity=dict(value.fit_identity) | {
                        "fit_artifact_sha256": run_identity["fit_artifact_sha256"],
                        "model_revision": run_identity["model_revision"],
                        "dataset_revision": run_identity["dataset_revision"],
                        "predecessor_state_sha256": run_identity["predecessor_state_sha256"],
                    },
                    transform_identity=dict(value.transform_identity) | {
                        "search_artifact_sha256": run_identity["search_artifact_sha256"],
                    },
                )
            rebound.append(replace(item, fitted=fitted))
        units = rebound
        journal = CandidateJournal(output / "journal", run_identity, resume=(output / "journal" / "journal.json").is_file())
        generator = CandidateLedgerGenerator(
            self.codec,
            attestation,
            objective_arm=str(self.config.get("objective_arm", "energy_normalized_sse")),
            competitive=bool(context["production"]),
            allow_test_backend=False,
            allow_fixed_transform_baseline=False,
        )
        ledger = generator.generate(units, journal=journal, output_path=output / "candidate-ledger.json")
        result = {"candidate_ledger_file": "candidate-ledger.json", "ledger_sha256": ledger["ledger_sha256"]}
        if context["kind"] == "candidates":
            provisional = persist_provisional_winner_deltas(
                output_dir=output / "provisional-winners",
                ledger=ledger,
                payload_store_root=output / "journal" / "payloads",
                checkpoint_sources=work["checkpoint_sources"],
                bit_triplet=self.config.get("attribution_provisional_bit_triplet", [4, 4, 4]),
            )
            result["provisional_winner_manifest_file"] = provisional.relative_to(output).as_posix()
        return result


def build_qwen_campaign_services(config: Mapping[str, Any]) -> QwenCampaignServices:
    """Build every concrete provider from one sealed adapter configuration."""

    codec_config = config.get("codec")
    if not isinstance(codec_config, Mapping):
        raise ValueError("adapter config requires a concrete codec object")
    codec = Exl3MCGCodec(
        source_root=codec_config["source_root"],
        numeric_core=codec_config["numeric_core"],
        extension=codec_config["extension"],
        device=str(codec_config.get("device", "cuda:0")),
        sigma_reg=float(codec_config.get("sigma_reg", 0.025)),
    )
    capturer = QwenRouteCaptureService(config)
    return QwenCampaignServices(
        capturer=capturer,
        fitter=QwenFitterService(config),
        ledger=QwenLedgerService(config, codec),
        codec=QwenCodecService(config, codec),
        evaluator=QwenEvaluatorService(config, capturer),
        allocator=QwenAllocatorService(config),
        checkpoint=QwenCheckpointService(config),
    )
