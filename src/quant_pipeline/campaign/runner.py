from __future__ import annotations

import dataclasses
import datetime as dt
import fcntl
import importlib
import inspect
import json
import math
import os
import re
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..calibration.windows import verify_sealed_corpus
from ..core.artifacts import canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json
from ..evaluation.kld_window import verify_kld_window
from ..spec import ExperimentSpec


PLAN_SCHEMA = "quant-pipeline.campaign-plan.v1"
DEFINITION_SCHEMA = "quant-pipeline.campaign-definition.v1"
JOURNAL_SCHEMA = "quant-pipeline.campaign-event.v1"
RESULT_SCHEMA = "quant-pipeline.campaign-stage-result.v1"
ZERO_HASH = "0" * 64
BASE_KINDS = (
    "identity",
    "teacher_capture",
    "fit_capture",
    "fit",
    "candidates",
    "attribution",
    "allocation",
    "confirmation",
)
GATE_POLICIES = {"continue", "rollback", "request_reallocation"}
GATE_KINDS = {"confirmation", "kld_reanchor"}
DEFAULT_MAX_GENERATIONS = 8


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _require_hash(value: Any, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a lowercase {length}-hex hash")
    return value


def _finite_json(value: Any, label: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string mapping key")
            _finite_json(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_json(item, f"{label}[{index}]")
        return
    raise ValueError(f"{label} contains a non-JSON value of type {type(value).__name__}")


def _seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(document)
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_seal(document: Mapping[str, Any], field: str, label: str) -> str:
    expected = _require_hash(document.get(field), f"{label}.{field}")
    body = dict(document)
    del body[field]
    actual = sha256_bytes(canonical_json(body))
    if actual != expected:
        raise ValueError(f"{label} seal mismatch: expected {expected}, got {actual}")
    return expected


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolved_from(base: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden in bound artifact trees: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported filesystem entry: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def bind_path(path: str | Path) -> dict[str, Any]:
    target = _resolved(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_symlink():
        raise ValueError(f"bound inputs may not be symlinks: {target}")
    if target.is_file():
        return {
            "kind": "file",
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    if target.is_dir():
        entries = _tree_entries(target)
        return {
            "kind": "directory",
            "path": str(target),
            "file_count": len(entries),
            "bytes": sum(int(row["bytes"]) for row in entries),
            "entries": entries,
            "sha256": sha256_bytes(canonical_json(entries)),
        }
    raise ValueError(f"unsupported bound input: {target}")


def _rebind(bound: Mapping[str, Any]) -> dict[str, Any]:
    return bind_path(str(bound["path"]))


def _path_identity(bound: Mapping[str, Any]) -> str:
    return _require_hash(bound.get("sha256"), "bound path sha256")


def code_identity() -> dict[str, Any]:
    campaign_root = Path(__file__).resolve().parent
    package_root = campaign_root.parent
    repository_root = package_root.parents[1]
    # Bind the orchestration closure here. Scientific stage implementations are
    # a separate closure and must be reported by StageAdapter.identity().
    files = sorted(campaign_root.rglob("*.py")) + [
        package_root / "core" / "artifacts.py",
        package_root / "spec.py",
    ]
    pyproject = repository_root / "pyproject.toml"
    if pyproject.exists():
        files.append(pyproject)
    entries = [
        {
            "path": path.relative_to(repository_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(files))
    ]
    return {"root": str(repository_root), "files": entries, "sha256": sha256_bytes(canonical_json(entries))}


@dataclasses.dataclass(frozen=True)
class CampaignDefinition:
    experiment_spec: str
    inputs: dict[str, str]
    layers: tuple[int, ...]
    reanchor_every_layers: int
    reanchor_failure_policy: str
    retention_mode: str
    max_generations: int = DEFAULT_MAX_GENERATIONS
    schema: str = DEFINITION_SCHEMA

    @classmethod
    def load(cls, path: str | Path) -> "CampaignDefinition":
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("campaign definition must be a JSON object")
        allowed = {
            "schema",
            "experiment_spec",
            "inputs",
            "layers",
            "reanchor_every_layers",
            "reanchor_failure_policy",
            "retention_mode",
            "max_generations",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown campaign-definition keys: {sorted(unknown)}")
        definition = cls(
            schema=raw.get("schema", DEFINITION_SCHEMA),
            experiment_spec=raw["experiment_spec"],
            inputs=dict(raw["inputs"]),
            layers=tuple(raw["layers"]),
            reanchor_every_layers=raw["reanchor_every_layers"],
            reanchor_failure_policy=raw["reanchor_failure_policy"],
            retention_mode=raw["retention_mode"],
            max_generations=raw.get("max_generations", DEFAULT_MAX_GENERATIONS),
        )
        definition.validate()
        return definition

    def validate(self) -> None:
        if self.schema != DEFINITION_SCHEMA:
            raise ValueError(f"unsupported campaign definition schema: {self.schema!r}")
        if not isinstance(self.experiment_spec, str) or not self.experiment_spec:
            raise ValueError("experiment_spec is required")
        if not self.inputs or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in self.inputs.items()):
            raise ValueError("inputs must be a non-empty string-to-path mapping")
        if not self.layers or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in self.layers):
            raise ValueError("layers must be a non-empty sequence of non-negative integers")
        if tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("layers must be unique and strictly increasing")
        if isinstance(self.reanchor_every_layers, bool) or not 1 <= self.reanchor_every_layers <= 4:
            raise ValueError("reanchor_every_layers must be in [1, 4]")
        if self.reanchor_failure_policy not in GATE_POLICIES:
            raise ValueError(f"reanchor_failure_policy must be one of {sorted(GATE_POLICIES)}")
        if self.retention_mode not in {"full", "capture-plus-ledger"}:
            raise ValueError("retention_mode must be 'full' or 'capture-plus-ledger'")
        if (
            isinstance(self.max_generations, bool)
            or not isinstance(self.max_generations, int)
            or self.max_generations < 1
        ):
            raise ValueError("max_generations must be a positive integer")


@dataclasses.dataclass(frozen=True)
class StageSpec:
    stage_id: str
    kind: str
    dependencies: tuple[str, ...] = ()
    layer: int | None = None
    block_layers: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StageRequest:
    campaign_dir: Path
    output_dir: Path
    stage_id: str
    kind: str
    attempt: int
    plan_sha256: str
    experiment_spec: dict[str, Any]
    static_inputs: dict[str, dict[str, Any]]
    dependency_artifacts: dict[str, dict[str, Any]]
    predecessor_state_hash: str
    generation: int = 0
    generation_context: dict[str, Any] = dataclasses.field(default_factory=dict)
    installed_layer_prefix: tuple[dict[str, Any], ...] = ()
    layer: int | None = None
    block_layers: tuple[int, ...] = ()

    def binding(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "kind": self.kind,
            "attempt": self.attempt,
            "plan_sha256": self.plan_sha256,
            "experiment_spec_sha256": self.experiment_spec["sha256"],
            "static_inputs": {key: row["sha256"] for key, row in sorted(self.static_inputs.items())},
            "dependency_artifacts": {
                key: row["artifact_sha256"] for key, row in sorted(self.dependency_artifacts.items())
            },
            "predecessor_state_hash": self.predecessor_state_hash,
            "generation": self.generation,
            "generation_context": self.generation_context,
            "installed_layer_prefix": list(self.installed_layer_prefix),
            "layer": self.layer,
            "block_layers": list(self.block_layers),
        }

    @property
    def request_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.binding()))


@dataclasses.dataclass(frozen=True)
class StageResult:
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("stage result metadata must be a dictionary")
        _finite_json(self.metadata, "stage result metadata")


@runtime_checkable
class StageAdapter(Protocol):
    def identity(self) -> Mapping[str, Any]: ...

    def preflight(self, plan: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def run(self, request: StageRequest) -> StageResult: ...


def _adapter_identity(
    adapter: StageAdapter,
    static_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(adapter, StageAdapter):
        raise TypeError("adapter must implement identity(), preflight(plan), and run(request)")
    identity = dict(adapter.identity())
    _finite_json(identity, "adapter identity")
    if not identity:
        raise ValueError("adapter identity may not be empty")
    module = inspect.getmodule(type(adapter))
    module_file = getattr(module, "__file__", None)
    result: dict[str, Any] = {
        "class": f"{type(adapter).__module__}:{type(adapter).__qualname__}",
        "declared": identity,
    }
    if module_file and Path(module_file).is_file():
        result["module"] = bind_path(module_file)
    identity_for_inputs = getattr(adapter, "identity_for_inputs", None)
    if static_inputs is not None and callable(identity_for_inputs):
        dynamic = dict(identity_for_inputs(static_inputs))
        _finite_json(dynamic, "adapter dynamic identity")
        if not dynamic:
            raise ValueError("adapter dynamic identity may not be empty")
        result["dynamic"] = dynamic
    result["sha256"] = sha256_bytes(canonical_json(result))
    return result


def load_adapter(reference: str) -> StageAdapter:
    if ":" not in reference:
        raise ValueError("adapter must be specified as module:attribute")
    module_name, attribute = reference.split(":", 1)
    module: ModuleType = importlib.import_module(module_name)
    value = getattr(module, attribute)
    adapter = value() if inspect.isclass(value) else value
    if not isinstance(adapter, StageAdapter):
        raise TypeError(f"{reference} does not provide a StageAdapter")
    return adapter


def _build_stages(definition: CampaignDefinition) -> list[StageSpec]:
    stages: list[StageSpec] = []
    for kind in BASE_KINDS:
        dependencies: tuple[str, ...]
        if kind == "identity":
            dependencies = ()
        elif kind == "teacher_capture":
            dependencies = ("identity",)
        elif kind == "fit_capture":
            dependencies = ("identity",)
        elif kind == "fit":
            dependencies = ("fit_capture",)
        elif kind == "candidates":
            dependencies = ("fit_capture", "fit")
        elif kind == "attribution":
            dependencies = ("teacher_capture", "candidates")
        elif kind == "confirmation":
            dependencies = ("candidates", "allocation")
        else:
            dependencies = ("candidates", "attribution")
        # Campaign-wide scientific stages operate on the complete sealed layer
        # inventory.  Carry it in the stage request instead of relying on an
        # adapter-specific duplicate setting.  Without this binding the shipped
        # Qwen route-capture service receives neither ``layer`` nor
        # ``block_layers`` and cannot reach the first fit.
        block_layers = definition.layers if kind in {
            "fit_capture",
            "fit",
            "candidates",
            "attribution",
            "allocation",
        } else ()
        stages.append(StageSpec(kind, kind, dependencies, block_layers=block_layers))

    accepted_encode_ids: list[str] = []
    accepted_reanchor_ids: list[str] = []
    block: list[int] = []
    previous_encode = ""
    previous_reanchor = ""
    block_index = 0
    for index, layer in enumerate(definition.layers):
        capture_id = f"causal_fit_capture.layer_{layer:03d}"
        fit_id = f"causal_fit.layer_{layer:03d}"
        candidates_id = f"causal_candidates.layer_{layer:03d}"
        encode_id = f"causal_encode.layer_{layer:03d}"
        predecessor_dependencies = ["allocation", "confirmation"]
        if previous_encode:
            predecessor_dependencies.append(previous_encode)
        if previous_reanchor:
            predecessor_dependencies.append(previous_reanchor)
        stages.extend(
            [
                StageSpec(capture_id, "causal_fit_capture", tuple(predecessor_dependencies), layer=layer),
                StageSpec(fit_id, "causal_fit", (capture_id,) + tuple(predecessor_dependencies), layer=layer),
                StageSpec(
                    candidates_id,
                    "causal_candidates",
                    (capture_id, fit_id) + tuple(predecessor_dependencies),
                    layer=layer,
                ),
                StageSpec(
                    encode_id,
                    "causal_encode",
                    (candidates_id, fit_id) + tuple(predecessor_dependencies),
                    layer=layer,
                ),
            ]
        )
        accepted_encode_ids.append(encode_id)
        previous_encode = encode_id
        block.append(layer)
        if len(block) == definition.reanchor_every_layers or index == len(definition.layers) - 1:
            reanchor_id = f"kld_reanchor.block_{block_index:03d}"
            stages.append(
                StageSpec(
                    reanchor_id,
                    "kld_reanchor",
                    ("teacher_capture",) + tuple(accepted_encode_ids),
                    block_layers=tuple(block),
                )
            )
            accepted_reanchor_ids.append(reanchor_id)
            previous_reanchor = reanchor_id
            block = []
            block_index += 1

    checkpoint_dependencies = ("allocation",) + tuple(accepted_encode_ids) + tuple(accepted_reanchor_ids)
    stages.extend(
        [
            StageSpec("checkpoint_emission", "checkpoint_emission", checkpoint_dependencies),
            StageSpec("checkpoint_audit", "checkpoint_audit", ("checkpoint_emission",)),
            StageSpec("student_capture", "student_capture", ("checkpoint_emission", "checkpoint_audit")),
            StageSpec("final_kld", "final_kld", ("teacher_capture", "student_capture")),
        ]
    )
    return stages


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_control_inputs(
    definition: CampaignDefinition,
    resolved_inputs: Mapping[str, Path],
    experiment: ExperimentSpec,
) -> None:
    required = {"source_checkpoint", "sealed_corpus", "kld_window"}
    missing = required - set(resolved_inputs)
    if missing:
        raise ValueError(f"campaign inputs are missing required identities: {sorted(missing)}")

    corpus_path = resolved_inputs["sealed_corpus"]
    if not corpus_path.is_file():
        raise ValueError("sealed_corpus must name the sealed JSON artifact")
    corpus = json.loads(corpus_path.read_text())
    verify_sealed_corpus(corpus)

    kld_root = resolved_inputs["kld_window"]
    if not kld_root.is_dir():
        raise ValueError("kld_window must name the directory containing kld-window.json and source-prefix.txt")
    kld = json.loads((kld_root / "kld-window.json").read_text())
    verify_kld_window(kld, kld_root)
    if int(kld.get("context_length", 0)) != 2048:
        raise ValueError("the historical control must be the sealed 2,048-token KLD window")
    dataset_revision = kld.get("dataset", {}).get("revision")
    if not isinstance(dataset_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", dataset_revision):
        raise ValueError("the KLD control requires an immutable 40-hex dataset revision")
    model_revision = kld.get("tokenizer", {}).get("expected_model_revision")
    if model_revision != experiment.model.revision:
        raise ValueError("KLD tokenizer revision differs from the experiment model revision")
    kld_tokens = kld["token_sha256"]
    for role, windows in corpus["windows"].items():
        if any(window.get("token_sha256") == kld_tokens for window in windows):
            raise ValueError(f"the historical KLD window is present in calibration role {role!r}")


def create_plan(
    definition_path: str | Path,
    campaign_dir: str | Path,
    adapter: StageAdapter,
) -> dict[str, Any]:
    definition_source = _resolved(definition_path)
    definition = CampaignDefinition.load(definition_source)
    source_base = definition_source.parent
    experiment_source = _resolved_from(source_base, definition.experiment_spec)
    experiment = ExperimentSpec.load(experiment_source)
    if experiment.objective.reanchor_every_layers != definition.reanchor_every_layers:
        raise ValueError(
            "campaign and experiment reanchor_every_layers differ; use one frozen interval in both specifications"
        )
    if experiment.objective.reanchor_every_layers > 4:
        raise ValueError("the experiment requires KLD re-anchoring at least every four accepted layers")
    destination = _resolved(campaign_dir)
    resolved_inputs = {name: _resolved_from(source_base, path) for name, path in sorted(definition.inputs.items())}
    _validate_control_inputs(definition, resolved_inputs, experiment)
    bound_inputs = {name: bind_path(path) for name, path in resolved_inputs.items()}
    bound_definition = bind_path(definition_source)
    bound_experiment = bind_path(experiment_source)
    for bound in [bound_definition, bound_experiment, *bound_inputs.values()]:
        if _overlaps(destination, Path(bound["path"])):
            raise ValueError("campaign_dir may not overlap an immutable input")
    prepare_empty_destination(destination)
    stages = _build_stages(definition)
    plan = _seal(
        {
            "schema": PLAN_SCHEMA,
            "created_at": _utc_now(),
            "campaign_dir": str(destination),
            "definition": dataclasses.asdict(definition),
            "definition_source": bound_definition,
            "experiment_spec": {
                "source": bound_experiment,
                "digest": experiment.digest,
                "sha256": _path_identity(bound_experiment),
                "document": dataclasses.asdict(experiment),
            },
            "inputs": bound_inputs,
            "code": code_identity(),
            "adapter": _adapter_identity(adapter, bound_inputs),
            "stages": [stage.as_dict() for stage in stages],
        },
        "plan_sha256",
    )
    write_json(destination / "plan.json", plan)
    journal = EventJournal(destination / "events.jsonl")
    journal.append(
        {
            "event": "campaign_planned",
            "plan_sha256": plan["plan_sha256"],
            "stage_id": None,
            "kind": None,
            "attempt": 0,
            "input_hashes": {},
            "output_hashes": {},
            "predecessor_state_hash": ZERO_HASH,
            "details": {"stage_count": len(stages)},
        }
    )
    return plan


class EventJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        events: list[dict[str, Any]] = []
        previous = ZERO_HASH
        with self.path.open("rb") as handle:
            for index, raw in enumerate(handle):
                if not raw.endswith(b"\n"):
                    raise ValueError(f"journal line {index} is not newline terminated")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(f"journal line {index} is malformed") from error
                if canonical_json(event) != raw:
                    raise ValueError(f"journal line {index} is not canonical JSON")
                if event.get("schema") != JOURNAL_SCHEMA or event.get("event_index") != index:
                    raise ValueError(f"journal line {index} has invalid schema/index")
                if event.get("previous_event_sha256") != previous:
                    raise ValueError(f"journal chain breaks at line {index}")
                expected = _require_hash(event.get("event_sha256"), f"journal[{index}].event_sha256")
                body = dict(event)
                del body["event_sha256"]
                actual = sha256_bytes(canonical_json(body))
                if actual != expected:
                    raise ValueError(f"journal event hash mismatch at line {index}")
                _finite_json(event, f"journal[{index}]")
                events.append(event)
                previous = expected
        if not events:
            raise ValueError("journal is empty")
        return events

    def append(self, body: Mapping[str, Any]) -> dict[str, Any]:
        _finite_json(body, "journal event")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            events = self.read()
            index = len(events)
            previous = events[-1]["event_sha256"]
        else:
            index = 0
            previous = ZERO_HASH
        event = {
            "schema": JOURNAL_SCHEMA,
            "event_index": index,
            "timestamp": _utc_now(),
            "previous_event_sha256": previous,
            **dict(body),
        }
        event["event_sha256"] = sha256_bytes(canonical_json(event))
        data = canonical_json(event)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        descriptor = os.open(self.path, flags, 0o644)
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while appending the campaign journal")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return event

    def recover_torn_tail(self, recovery_root: Path, plan_sha256: str) -> dict[str, Any] | None:
        """Quarantine and remove only an unterminated final fragment.

        The complete prefix is fully chain-verified before mutation. This method
        must be called while the campaign lock is held.
        """

        data = self.path.read_bytes()
        if data.endswith(b"\n"):
            return None
        boundary = data.rfind(b"\n") + 1
        prefix, fragment = data[:boundary], data[boundary:]
        if not prefix or not fragment:
            raise ValueError("journal has no valid complete prefix to recover")
        # Validate the exact prefix using the normal parser before truncation.
        temporary = self.path.with_name(f".{self.path.name}.recovery-prefix")
        try:
            with temporary.open("wb") as handle:
                handle.write(prefix)
                handle.flush()
                os.fsync(handle.fileno())
            original = self.path
            self.path = temporary
            try:
                self.read()
            finally:
                self.path = original
        finally:
            temporary.unlink(missing_ok=True)
        fragment_sha256 = sha256_bytes(fragment)
        quarantine = recovery_root / "journal-recovery" / f"torn-tail-{fragment_sha256}.bin"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            if sha256_file(quarantine) != fragment_sha256:
                raise ValueError("journal recovery quarantine identity collision")
        else:
            descriptor = os.open(quarantine, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, fragment)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            parent = os.open(quarantine.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        descriptor = os.open(self.path, os.O_WRONLY)
        try:
            os.ftruncate(descriptor, len(prefix))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self.append(
            {
                "event": "journal_tail_recovered",
                "plan_sha256": plan_sha256,
                "stage_id": None,
                "kind": "journal_recovery",
                "attempt": 0,
                "input_hashes": {},
                "output_hashes": {"fragment_sha256": fragment_sha256},
                "predecessor_state_hash": ZERO_HASH,
                "details": {
                    "fragment_bytes": len(fragment),
                    "quarantine_path": quarantine.relative_to(recovery_root).as_posix(),
                },
            }
        )


def _load_plan(campaign_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = _resolved(campaign_dir)
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported campaign plan schema")
    _verify_seal(plan, "plan_sha256", "campaign plan")
    return root, plan


def _current_drift(plan: Mapping[str, Any], adapter: StageAdapter | None) -> list[str]:
    failures: list[str] = []
    checks = [
        ("definition", plan["definition_source"]),
        ("experiment_spec", plan["experiment_spec"]["source"]),
        *((f"input:{name}", bound) for name, bound in sorted(plan["inputs"].items())),
    ]
    for label, expected in checks:
        try:
            actual = _rebind(expected)
        except Exception as error:
            failures.append(f"{label}:unreadable:{type(error).__name__}")
            continue
        if actual != expected:
            failures.append(f"{label}:identity-drift")
    if code_identity() != plan["code"]:
        failures.append("code:identity-drift")
    if adapter is not None:
        try:
            actual_adapter = _adapter_identity(adapter, plan["inputs"])
        except Exception as error:
            failures.append(f"adapter:unreadable:{type(error).__name__}")
        else:
            if actual_adapter != plan["adapter"]:
                failures.append("adapter:identity-drift")
    return failures


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    bound = bind_path(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "artifact_sha256": bound["sha256"],
        "bytes": bound["bytes"],
        "file_count": bound.get("file_count", 1),
    }


def _stage_payload_binding(path: Path) -> dict[str, Any]:
    """Bind adapter output without recursively including the runner receipt."""

    entries = [row for row in _tree_entries(path) if row["path"] != ".runner-result.json"]
    if not entries:
        raise ValueError("stage adapter produced no artifact files")
    return {
        "entries": entries,
        "file_count": len(entries),
        "bytes": sum(int(row["bytes"]) for row in entries),
        "sha256": sha256_bytes(canonical_json(entries)),
    }


def _stage_payload_matches(
    path: Path,
    expected: Mapping[str, Any],
    retired: Sequence[Mapping[str, Any]] = (),
) -> bool:
    current = [row for row in _tree_entries(path) if row["path"] != ".runner-result.json"]
    current_names = {row["path"] for row in current}
    restored: list[dict[str, Any]] = []
    for raw in retired:
        row = {
            "path": _safe_relative_file(raw["path"]),
            "bytes": int(raw["bytes"]),
            "sha256": _require_hash(raw["sha256"], "retired stage-payload hash"),
        }
        if row["path"] in current_names:
            if next(item for item in current if item["path"] == row["path"]) != row:
                return False
        else:
            restored.append(row)
    entries = sorted(current + restored, key=lambda row: row["path"])
    actual = {
        "entries": entries,
        "file_count": len(entries),
        "bytes": sum(int(row["bytes"]) for row in entries),
        "sha256": sha256_bytes(canonical_json(entries)),
    }
    return actual == expected


def _safe_relative_file(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError("transient artifact paths must be non-empty relative paths")
    path = Path(value)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe transient artifact path: {value!r}")
    if path.name.startswith(".runner-"):
        raise ValueError("runner receipt files may not be transient")
    return path.as_posix()


def _confined_path(root: Path, relative: str) -> Path:
    """Return a lexical child path only when no component redirects through a symlink."""
    if root.is_symlink():
        raise ValueError(f"symlinked campaign artifact root: {root}")
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked campaign artifact path component: {relative!r}")
    if not candidate.resolve(strict=False).is_relative_to(root.resolve()):
        raise ValueError(f"campaign artifact path escapes its root: {relative!r}")
    return candidate


def _artifact_matches_with_retirements(
    path: Path,
    root: Path,
    expected: Mapping[str, Any],
    retired: Sequence[Mapping[str, Any]],
) -> bool:
    relative = _safe_relative_file(expected.get("path"))
    confined = _confined_path(root, relative)
    if path != confined:
        return False
    bound = bind_path(path)
    if bound["kind"] != "directory":
        return False
    current_entries = list(bound["entries"])
    current_names = {row["path"] for row in current_entries}
    retired_rows: list[dict[str, Any]] = []
    for raw in retired:
        row = {"path": _safe_relative_file(raw["path"]), "bytes": int(raw["bytes"]), "sha256": raw["sha256"]}
        _require_hash(row["sha256"], "retired artifact file hash")
        if row["bytes"] < 0:
            return False
        if row["path"] in current_names:
            current = next(item for item in current_entries if item["path"] == row["path"])
            if current != row:
                return False
        else:
            retired_rows.append(row)
    entries = sorted(current_entries + retired_rows, key=lambda row: row["path"])
    reconstructed = {
        "path": relative,
        "artifact_sha256": sha256_bytes(canonical_json(entries)),
        "bytes": sum(int(row["bytes"]) for row in entries),
        "file_count": len(entries),
    }
    return reconstructed == expected


def _read_stage_result(
    path: Path,
    expected_request_sha256: str,
    retired: Sequence[Mapping[str, Any]] = (),
) -> StageResult:
    receipt = json.loads((path / ".runner-result.json").read_text())
    if receipt.get("schema") != RESULT_SCHEMA:
        raise ValueError("invalid stage-result schema")
    _verify_seal(receipt, "receipt_sha256", "stage result")
    if receipt.get("request_sha256") != expected_request_sha256:
        raise ValueError("stage result is bound to a different request")
    payload = receipt.get("payload_binding")
    if not isinstance(payload, dict) or not _stage_payload_matches(path, payload, retired):
        raise ValueError("stage result payload bytes differ from the sealed receipt")
    return StageResult(dict(receipt["metadata"]))


def _validate_gate(result: StageResult, request: StageRequest | None = None) -> dict[str, Any]:
    gate = result.metadata.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("KLD re-anchor must return gate metadata")
    allowed = {"passed", "metric", "value", "threshold", "reference_sha256", "capture_sha256"}
    if set(gate) != allowed:
        raise ValueError(f"KLD gate fields must be exactly {sorted(allowed)}")
    if not isinstance(gate["passed"], bool) or gate["metric"] != "kld":
        raise ValueError("KLD gate requires a boolean passed field and metric='kld'")
    for key in ("value", "threshold"):
        if isinstance(gate[key], bool) or not isinstance(gate[key], (int, float)) or not math.isfinite(float(gate[key])) or float(gate[key]) < 0:
            raise ValueError(f"KLD gate {key} must be finite and non-negative")
    _require_hash(gate["reference_sha256"], "KLD gate reference_sha256")
    _require_hash(gate["capture_sha256"], "KLD gate capture_sha256")
    if gate["passed"] != (float(gate["value"]) <= float(gate["threshold"])):
        raise ValueError("KLD gate passed flag contradicts value <= threshold")
    if request is not None:
        files = result.metadata.get("gate_files")
        if not isinstance(files, dict) or set(files) != {"reference", "capture"}:
            raise ValueError("KLD gate must bind exact reference and capture files")
        for role, hash_field in (("reference", "reference_sha256"), ("capture", "capture_sha256")):
            relative = _safe_relative_file(files[role])
            artifact = request.output_dir / relative
            if not artifact.is_file() or artifact.is_symlink():
                raise ValueError(f"KLD gate {role} file is missing or invalid")
            if sha256_file(artifact) != gate[hash_field]:
                raise ValueError(f"KLD gate {role} file hash mismatch")
        if request.kind == "kld_reanchor":
            teacher = request.dependency_artifacts.get("teacher_capture")
            if teacher is None:
                raise ValueError("KLD re-anchor gate lacks teacher_capture dependency")
            teacher_root = request.campaign_dir / teacher["path"]
            teacher_receipt = json.loads((teacher_root / ".runner-result.json").read_text())
            teacher_reference = teacher_receipt.get("metadata", {}).get("teacher_reference_sha256")
            if teacher_reference != gate["reference_sha256"]:
                raise ValueError("KLD gate reference differs from teacher capture identity")
        elif request.kind == "confirmation":
            # The confirmation teacher is captured prospectively inside the
            # gate stage, after the candidate ledger and allocation are
            # frozen.  Its StageRequest hash already binds those exact two
            # dependencies; requiring the historical teacher_capture here
            # would silently turn the gate back into a reused evaluation.
            if set(request.dependency_artifacts) != {"candidates", "allocation"}:
                raise ValueError(
                    "confirmation gate must bind exactly the frozen candidates and allocation"
                )
        else:
            raise ValueError(f"unsupported KLD gate stage kind: {request.kind}")
    return gate


def _validate_causal_result(request: StageRequest, result: StageResult) -> None:
    if not request.kind.startswith("causal_"):
        return
    if result.metadata.get("predecessor_state_hash") != request.predecessor_state_hash:
        raise ValueError(f"{request.kind} result is not bound to its predecessor state")
    if result.metadata.get("layer") != request.layer:
        raise ValueError(f"{request.kind} result is not bound to layer {request.layer}")
    identity_fields = {
        "causal_fit_capture": "capture_sha256",
        "causal_fit": "fit_sha256",
        "causal_candidates": "candidate_ledger_sha256",
        "causal_encode": "installed_checkpoint_sha256",
    }
    field = identity_fields.get(request.kind)
    if field is None:
        raise ValueError(f"unsupported causal stage kind: {request.kind}")
    _require_hash(result.metadata.get(field), f"{request.kind}.{field}")


def _installed_prefix_from_completed(
    root: Path,
    plan: Mapping[str, Any],
    completed: Mapping[str, Mapping[str, Any]],
    state_hash: str,
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return and verify the complete ordered installed-layer predecessor chain."""

    identity = completed.get("identity")
    running = (
        sha256_bytes(canonical_json({"identity": identity["artifact_sha256"]}))
        if identity is not None
        else ZERO_HASH
    )
    rows: list[dict[str, Any]] = []
    seen_gap = False
    for stage in plan["stages"]:
        if stage["kind"] != "causal_encode":
            continue
        stage_id = stage["stage_id"]
        artifact = completed.get(stage_id)
        if artifact is None:
            seen_gap = True
            continue
        if seen_gap:
            raise ValueError("accepted installed layers do not form an ordered plan prefix")
        matching = [
            event
            for event in events
            if event.get("event") == "stage_completed"
            and event.get("stage_id") == stage_id
            and event.get("output_hashes", {}).get("artifact", {}).get("artifact_sha256")
            == artifact["artifact_sha256"]
        ]
        if len(matching) != 1:
            raise ValueError(f"cannot resolve unique accepted completion for {stage_id}")
        completion = matching[0]
        predecessor = _require_hash(completion.get("predecessor_state_hash"), f"{stage_id} predecessor")
        if predecessor != running:
            raise ValueError(f"accepted installed-layer predecessor chain breaks at {stage_id}")
        artifact_root = root / artifact["path"]
        receipt = json.loads((artifact_root / ".runner-result.json").read_text())
        installed_checkpoint = _require_hash(
            receipt.get("metadata", {}).get("installed_checkpoint_sha256"),
            f"{stage_id} installed checkpoint",
        )
        installed_state = sha256_bytes(
            canonical_json(
                {
                    "predecessor_state_hash": running,
                    "stage_id": stage_id,
                    "layer": stage["layer"],
                    "artifact_sha256": artifact["artifact_sha256"],
                }
            )
        )
        if completion.get("details", {}).get("installed_state_hash") != installed_state:
            raise ValueError(f"accepted installed-layer state seal differs at {stage_id}")
        rows.append(
            {
                "stage_id": stage_id,
                "layer": int(stage["layer"]),
                "path": artifact["path"],
                "artifact_sha256": artifact["artifact_sha256"],
                "installed_checkpoint_sha256": installed_checkpoint,
                "predecessor_state_hash": running,
                "installed_state_hash": installed_state,
            }
        )
        running = installed_state
    if running != state_hash:
        raise ValueError("accepted installed-layer prefix does not reproduce current installed state")
    return tuple(rows)


@contextmanager
def _campaign_lock(root: Path):
    path = root / ".campaign.lock"
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another campaign process holds the execution lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CampaignRunner:
    def __init__(self, campaign_dir: str | Path, adapter: StageAdapter):
        self.root, self.plan = _load_plan(campaign_dir)
        self.adapter = adapter
        self.journal = EventJournal(self.root / "events.jsonl")
        self.stages = [
            StageSpec(
                stage_id=row["stage_id"],
                kind=row["kind"],
                dependencies=tuple(row.get("dependencies", ())),
                layer=row.get("layer"),
                block_layers=tuple(row.get("block_layers", ())),
            )
            for row in self.plan["stages"]
        ]

    def execute(self, *, resume: bool = False) -> dict[str, Any]:
        with _campaign_lock(self.root):
            if resume:
                self.journal.recover_torn_tail(self.root, self.plan["plan_sha256"])
            existing_events = self.journal.read()
            if not resume and len(existing_events) > 1:
                raise RuntimeError("campaign has execution history; use resume instead of execute")
            drift = _current_drift(self.plan, self.adapter)
            if drift:
                raise ValueError(f"campaign identity drift: {drift}")
            audit = audit_campaign(self.root, self.adapter, verify_drift=False)
            if not audit["integrity_ok"]:
                raise ValueError(f"campaign integrity failure: {audit['failures']}")
            if audit.get("pending_gate") is not None:
                action = self._recover_pending_gate_decision(audit)
                if action != "continue":
                    return status_campaign(self.root, self.adapter)
                audit = audit_campaign(self.root, self.adapter, verify_drift=False)
                if not audit["integrity_ok"]:
                    raise ValueError(f"campaign integrity failure after gate recovery: {audit['failures']}")
            if audit.get("pending_supersession") is not None:
                self._recover_pending_supersession(audit)
                audit = audit_campaign(self.root, self.adapter, verify_drift=False)
                if not audit["integrity_ok"]:
                    raise ValueError(
                        f"campaign integrity failure after supersession recovery: {audit['failures']}"
                    )
                return status_campaign(self.root, self.adapter)
            if audit["complete"]:
                return status_campaign(self.root, self.adapter)
            self._reconcile_missing_retirement_plans(audit["completed"])
            self._finalize_pending_retirements()
            self._run_preflight()
            completed = dict(audit["completed"])
            state_hash = audit["installed_state_hash"]
            generation = int(audit.get("generation", 0))
            generation_context = dict(audit.get("generation_context", {}))
            for stage in self.stages:
                if stage.stage_id in completed:
                    continue
                dependencies = {key: completed[key] for key in stage.dependencies}
                attempt = self._next_attempt(stage.stage_id)
                attempt_dir = self.root / "attempts" / stage.stage_id / f"attempt-{attempt:04d}"
                installed_prefix = _installed_prefix_from_completed(
                    self.root,
                    self.plan,
                    completed,
                    state_hash,
                    self.journal.read(),
                )
                request = StageRequest(
                    campaign_dir=self.root,
                    output_dir=attempt_dir,
                    stage_id=stage.stage_id,
                    kind=stage.kind,
                    attempt=attempt,
                    plan_sha256=self.plan["plan_sha256"],
                    experiment_spec=self.plan["experiment_spec"],
                    static_inputs=self.plan["inputs"],
                    dependency_artifacts=dependencies,
                    predecessor_state_hash=state_hash,
                    generation=generation,
                    generation_context=generation_context,
                    installed_layer_prefix=installed_prefix,
                    layer=stage.layer,
                    block_layers=stage.block_layers,
                )
                recovered = self._recover_unjournaled_result(request)
                if recovered is None:
                    prepare_empty_destination(attempt_dir)
                    self.journal.append(self._event_body("stage_started", request, {}, {}))
                    try:
                        result = self.adapter.run(request)
                        if not isinstance(result, StageResult):
                            raise TypeError("adapter.run() must return StageResult")
                        if stage.kind in GATE_KINDS:
                            _validate_gate(result, request)
                        _validate_causal_result(request, result)
                        self._validate_retention_result(stage, result)
                        if (attempt_dir / ".runner-result.json").exists():
                            raise ValueError("stage adapter wrote the runner-reserved .runner-result.json path")
                        payload_binding = _stage_payload_binding(attempt_dir)
                        receipt = _seal(
                            {
                                "schema": RESULT_SCHEMA,
                                "request_sha256": request.request_sha256,
                                "metadata": result.metadata,
                                "payload_binding": payload_binding,
                            },
                            "receipt_sha256",
                        )
                        write_json(attempt_dir / ".runner-result.json", receipt)
                    except BaseException as error:
                        self.journal.append(
                            self._event_body(
                                "stage_failed",
                                request,
                                {},
                                {
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "traceback_sha256": sha256_bytes(traceback.format_exc().encode()),
                                },
                            )
                        )
                        raise
                else:
                    result, request = recovered
                    attempt_dir = request.output_dir
                try:
                    artifact = _artifact_binding(attempt_dir, self.root)
                except BaseException as error:
                    self.journal.append(
                        self._event_body(
                            "stage_failed",
                            request,
                            {},
                            {
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback_sha256": sha256_bytes(traceback.format_exc().encode()),
                            },
                        )
                    )
                    raise
                next_state = state_hash
                if stage.kind == "identity":
                    next_state = sha256_bytes(canonical_json({"identity": artifact["artifact_sha256"]}))
                elif stage.kind == "causal_encode":
                    next_state = sha256_bytes(
                        canonical_json(
                            {
                                "predecessor_state_hash": state_hash,
                                "stage_id": stage.stage_id,
                                "layer": stage.layer,
                                "artifact_sha256": artifact["artifact_sha256"],
                            }
                        )
                    )
                details = {"metadata": result.metadata, "installed_state_hash": next_state}
                self.journal.append(
                    self._event_body(
                        "stage_completed",
                        request,
                        {"artifact": artifact},
                        details,
                    )
                )
                completed[stage.stage_id] = artifact
                state_hash = next_state
                self._plan_retirement_after(stage, completed)
                if stage.kind in GATE_KINDS:
                    action = self._record_gate_decision(stage, request, result, completed)
                    if action != "continue":
                        return status_campaign(self.root, self.adapter)
            self.journal.append(
                {
                    "event": "campaign_completed",
                    "plan_sha256": self.plan["plan_sha256"],
                    "stage_id": None,
                    "kind": None,
                    "attempt": 0,
                    "input_hashes": {key: row["sha256"] for key, row in sorted(self.plan["inputs"].items())},
                    "output_hashes": {key: row["artifact_sha256"] for key, row in sorted(completed.items())},
                    "predecessor_state_hash": state_hash,
                    "details": {"completed_stages": len(completed)},
                }
            )
            return status_campaign(self.root, self.adapter)

    def _recover_pending_gate_decision(self, audit: Mapping[str, Any]) -> str:
        stage_id = audit.get("pending_gate")
        stage = next((candidate for candidate in self.stages if candidate.stage_id == stage_id), None)
        if stage is None or stage.kind not in GATE_KINDS:
            raise ValueError("pending gate does not identify a planned KLD gate stage")
        artifact = audit["completed"][stage_id]
        completions = [
            event
            for event in self.journal.read()
            if event.get("event") == "stage_completed"
            and event.get("stage_id") == stage_id
            and event.get("output_hashes", {}).get("artifact", {}).get("artifact_sha256")
            == artifact["artifact_sha256"]
        ]
        if len(completions) != 1:
            raise ValueError("cannot reconstruct a unique sealed re-anchor completion")
        completion = completions[0]
        request = StageRequest(
            campaign_dir=self.root,
            output_dir=self.root / artifact["path"],
            stage_id=stage.stage_id,
            kind=stage.kind,
            attempt=int(completion["attempt"]),
            plan_sha256=self.plan["plan_sha256"],
            experiment_spec=self.plan["experiment_spec"],
            static_inputs=self.plan["inputs"],
            dependency_artifacts={key: audit["completed"][key] for key in stage.dependencies},
            predecessor_state_hash=completion["predecessor_state_hash"],
            generation=int(completion.get("generation", 0)),
            generation_context=dict(completion.get("generation_context", {})),
            installed_layer_prefix=tuple(completion.get("installed_layer_prefix", [])),
            layer=stage.layer,
            block_layers=stage.block_layers,
        )
        result = _read_stage_result(request.output_dir, request.request_sha256)
        _validate_gate(result, request)
        return self._record_gate_decision(stage, request, result, audit["completed"])

    def _record_gate_decision(
        self,
        stage: StageSpec,
        request: StageRequest,
        result: StageResult,
        completed: Mapping[str, Mapping[str, Any]],
    ) -> str:
        gate = _validate_gate(result, request)
        if gate["passed"]:
            action = "continue"
        else:
            action = self.plan["definition"]["reanchor_failure_policy"]
            # Confirmation is an admission gate, not an informational
            # diagnostic.  A legacy `continue` re-anchor policy must not admit
            # a frozen allocation that failed its untouched confirmation fold.
            if stage.kind == "confirmation" and action == "continue":
                action = "request_reallocation"
        rollback_state = (
            request.predecessor_state_hash
            if action != "continue" and stage.kind == "confirmation"
            else self._block_start_state(stage, request.generation)
            if action != "continue"
            else None
        )
        disposition = "running" if action == "continue" else "replan_pending"
        decision = self.journal.append(
            self._event_body(
                "gate_decision",
                request,
                {},
                {
                    "gate": gate,
                    "action": action,
                    "disposition": disposition,
                    "rollback_state_hash": rollback_state,
                },
            )
        )
        if action != "continue":
            self._append_generation_superseded(
                stage=stage,
                generation=request.generation,
                action=action,
                rollback_state=rollback_state,
                completed=completed,
                gate_decision=decision,
            )
        return action

    def _append_generation_superseded(
        self,
        *,
        stage: StageSpec,
        generation: int,
        action: str,
        rollback_state: str,
        completed: Mapping[str, Mapping[str, Any]],
        gate_decision: Mapping[str, Any],
    ) -> None:
        if action == "continue" or action not in GATE_POLICIES:
            raise ValueError("generation supersession requires a non-continue gate action")
        max_generations = int(
            self.plan["definition"].get("max_generations", DEFAULT_MAX_GENERATIONS)
        )
        if generation + 1 >= max_generations:
            raise RuntimeError(
                f"campaign generation cap reached: generation {generation} cannot supersede "
                f"when max_generations={max_generations}"
            )
        if (
            gate_decision.get("event") != "gate_decision"
            or gate_decision.get("stage_id") != stage.stage_id
            or gate_decision.get("details", {}).get("action") != action
            or gate_decision.get("details", {}).get("rollback_state_hash") != rollback_state
            or int(gate_decision.get("generation", -1)) != generation
        ):
            raise ValueError("generation supersession gate-decision binding is invalid")
        decision_sha256 = _require_hash(
            gate_decision.get("event_sha256"), "generation supersession gate decision"
        )
        stage_order = [candidate.stage_id for candidate in self.stages]
        invalidated_ids = [
            stage_id
            for stage_id in stage_order[stage_order.index("allocation") :]
            if stage_id in completed
        ]
        invalidated = [
            {
                "stage_id": stage_id,
                "path": completed[stage_id]["path"],
                "artifact_sha256": completed[stage_id]["artifact_sha256"],
            }
            for stage_id in invalidated_ids
        ]
        failed_hash = completed[stage.stage_id]["artifact_sha256"]
        self.journal.append(
            {
                "event": "generation_superseded",
                "plan_sha256": self.plan["plan_sha256"],
                "stage_id": None,
                "kind": "generation_replan",
                "attempt": generation + 1,
                "generation": generation,
                "generation_context": dict(gate_decision.get("generation_context", {})),
                "input_hashes": {
                    "failed_gate_artifact_sha256": failed_hash,
                    "gate_decision_sha256": decision_sha256,
                },
                "output_hashes": {},
                "predecessor_state_hash": rollback_state,
                "details": {
                    "prior_generation": generation,
                    "new_generation": generation + 1,
                    "failed_gate_stage_id": stage.stage_id,
                    "failed_gate_artifact_sha256": failed_hash,
                    "gate_decision_sha256": decision_sha256,
                    "action": action,
                    "rollback_state_hash": rollback_state,
                    "invalidated_artifacts": invalidated,
                },
            }
        )

    def _recover_pending_supersession(self, audit: Mapping[str, Any]) -> None:
        stage_id = audit.get("pending_supersession")
        stage = next((candidate for candidate in self.stages if candidate.stage_id == stage_id), None)
        if stage is None or stage.kind not in GATE_KINDS:
            raise ValueError("pending supersession does not identify a planned KLD gate stage")
        decisions = [
            event
            for event in self.journal.read()
            if event.get("event") == "gate_decision"
            and event.get("stage_id") == stage_id
            and int(event.get("generation", 0)) == int(audit.get("generation", 0))
        ]
        if len(decisions) != 1:
            raise ValueError("cannot reconstruct a unique pending generation supersession")
        decision = decisions[0]
        details = decision.get("details", {})
        action = details.get("action")
        rollback_state = details.get("rollback_state_hash")
        if action == "continue" or action not in GATE_POLICIES:
            raise ValueError("pending supersession has an invalid gate action")
        _require_hash(rollback_state, "pending supersession rollback state")
        self._append_generation_superseded(
            stage=stage,
            generation=int(decision.get("generation", 0)),
            action=action,
            rollback_state=rollback_state,
            completed=audit["completed"],
            gate_decision=decision,
        )

    def _validate_retention_result(self, stage: StageSpec, result: StageResult) -> None:
        files = result.metadata.get("transient_files")
        if self.plan["definition"]["retention_mode"] == "full":
            if files not in (None, []):
                raise ValueError("full retention mode forbids transient_files")
            return
        if stage.kind not in ("fit", "causal_fit"):
            if files is not None:
                raise ValueError("only fit stages may declare transient covariance files")
            return
        if not isinstance(files, list) or not files:
            raise ValueError("capture-plus-ledger fit stages must declare transient_files")
        normalized = [_safe_relative_file(value) for value in files]
        if len(set(normalized)) != len(normalized):
            raise ValueError("transient_files contains duplicates")

    def _reconcile_missing_retirement_plans(
        self, completed: Mapping[str, Mapping[str, Any]]
    ) -> None:
        if self.plan["definition"]["retention_mode"] != "capture-plus-ledger":
            return
        events = self.journal.read()
        planned_bindings: set[tuple[str, str, str, str]] = set()
        for event in events:
            if event.get("event") != "artifacts_retirement_planned":
                continue
            details = event.get("details")
            inputs = event.get("input_hashes")
            if not isinstance(details, Mapping) or not isinstance(inputs, Mapping):
                continue
            values = (
                event.get("stage_id"),
                details.get("trigger_stage_id"),
                inputs.get("producer_artifact_sha256"),
                inputs.get("trigger_artifact_sha256"),
            )
            if all(isinstance(value, str) for value in values):
                planned_bindings.add(values)  # type: ignore[arg-type]
        by_id = {stage.stage_id: stage for stage in self.stages}
        for trigger_id in completed:
            trigger = by_id.get(trigger_id)
            if trigger is None:
                continue
            if trigger.kind == "candidates":
                producer_id = "fit"
            elif trigger.kind == "causal_candidates":
                assert trigger.layer is not None
                producer_id = f"causal_fit.layer_{trigger.layer:03d}"
            else:
                continue
            binding = (
                producer_id,
                trigger_id,
                completed.get(producer_id, {}).get("artifact_sha256"),
                completed[trigger_id].get("artifact_sha256"),
            )
            if producer_id in completed and binding not in planned_bindings:
                self._plan_retirement_after(trigger, completed)

    def _plan_retirement_after(self, trigger: StageSpec, completed: Mapping[str, Mapping[str, Any]]) -> None:
        if self.plan["definition"]["retention_mode"] != "capture-plus-ledger":
            return
        if trigger.kind == "candidates":
            producer_id = "fit"
        elif trigger.kind == "causal_candidates":
            assert trigger.layer is not None
            producer_id = f"causal_fit.layer_{trigger.layer:03d}"
        else:
            return
        producer = completed[producer_id]
        producer_root = self.root / producer["path"]
        receipt = json.loads((producer_root / ".runner-result.json").read_text())
        files = [_safe_relative_file(value) for value in receipt["metadata"].get("transient_files", [])]
        records = []
        for relative in files:
            path = _confined_path(producer_root, relative)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"declared transient file is missing or invalid: {producer_id}:{relative}")
            records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        event = self.journal.append(
            {
                "event": "artifacts_retirement_planned",
                "plan_sha256": self.plan["plan_sha256"],
                "stage_id": producer_id,
                "kind": "artifact_retirement",
                "attempt": 1,
                "input_hashes": {
                    "producer_artifact_sha256": producer["artifact_sha256"],
                    "trigger_artifact_sha256": completed[trigger.stage_id]["artifact_sha256"],
                },
                "output_hashes": {},
                "predecessor_state_hash": ZERO_HASH,
                "details": {"trigger_stage_id": trigger.stage_id, "files": records},
            }
        )
        self._finalize_retirement(event)

    def _finalize_pending_retirements(self) -> None:
        events = self.journal.read()
        completed = {
            event.get("details", {}).get("planned_event_sha256")
            for event in events
            if event["event"] == "artifacts_retirement_completed"
        }
        for event in events:
            if event["event"] == "artifacts_retirement_planned" and event["event_sha256"] not in completed:
                self._finalize_retirement(event)

    def _finalize_retirement(self, planned: Mapping[str, Any]) -> None:
        # Artifact paths are journaled by the producer completion, not the plan.
        replay = _replay(self.root, self.plan)
        producer = replay["completed"].get(planned["stage_id"])
        if producer is None:
            raise ValueError("cannot retire files for an incomplete producer stage")
        producer_relative = _safe_relative_file(producer["path"])
        producer_root = _confined_path(self.root, producer_relative)
        for row in planned["details"]["files"]:
            relative = _safe_relative_file(row["path"])
            path = _confined_path(producer_root, relative)
            if path.exists():
                if not path.is_file() or path.is_symlink() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                    raise ValueError(f"transient artifact drift before retirement: {planned['stage_id']}:{relative}")
                path.unlink()
        self.journal.append(
            {
                "event": "artifacts_retirement_completed",
                "plan_sha256": self.plan["plan_sha256"],
                "stage_id": planned["stage_id"],
                "kind": "artifact_retirement",
                "attempt": planned["attempt"],
                "input_hashes": planned["input_hashes"],
                "output_hashes": {"retired_file_count": len(planned["details"]["files"])},
                "predecessor_state_hash": ZERO_HASH,
                "details": {
                    "planned_event_sha256": planned["event_sha256"],
                    "trigger_stage_id": planned["details"]["trigger_stage_id"],
                    "files": planned["details"]["files"],
                },
            }
        )

    def _run_preflight(self) -> None:
        attempt = 1 + sum(
            event["event"] in ("resource_preflight_passed", "resource_preflight_failed")
            for event in self.journal.read()
        )
        try:
            report = dict(self.adapter.preflight(self.plan))
            _finite_json(report, "resource preflight")
            required = {"ok", "local_only", "remote_endpoints", "gpu", "storage", "software"}
            if set(report) != required:
                raise ValueError(f"resource preflight fields must be exactly {sorted(required)}")
            if not isinstance(report["ok"], bool) or not isinstance(report["local_only"], bool):
                raise ValueError("resource preflight ok/local_only fields must be booleans")
            if report["local_only"] is not True or report["remote_endpoints"] != []:
                raise ValueError("campaign adapter must be local-only and declare no remote endpoints")
            if not all(isinstance(report[field], dict) and report[field] for field in ("gpu", "storage", "software")):
                raise ValueError("resource preflight requires non-empty gpu, storage, and software reports")
            storage = report["storage"]
            required_storage = {"retention_mode", "estimated_peak_bytes", "available_bytes", "safety_margin_bytes"}
            if not required_storage <= set(storage):
                raise ValueError(f"storage preflight is missing {sorted(required_storage - set(storage))}")
            if storage["retention_mode"] != self.plan["definition"]["retention_mode"]:
                raise ValueError("storage preflight retention mode differs from the sealed plan")
            for field in ("estimated_peak_bytes", "available_bytes", "safety_margin_bytes"):
                minimum = 0 if field == "available_bytes" else 1
                if isinstance(storage[field], bool) or not isinstance(storage[field], int) or storage[field] < minimum:
                    raise ValueError(
                        f"storage preflight {field} must be "
                        f"{'a non-negative' if minimum == 0 else 'a positive'} integer"
                    )
            if storage["available_bytes"] < storage["estimated_peak_bytes"] + storage["safety_margin_bytes"]:
                raise RuntimeError("resource preflight storage estimate exceeds available bytes plus safety margin")
            if not report["ok"]:
                raise RuntimeError("resource preflight reported insufficient resources")
        except BaseException as error:
            self.journal.append(
                {
                    "event": "resource_preflight_failed",
                    "plan_sha256": self.plan["plan_sha256"],
                    "stage_id": None,
                    "kind": "resource_preflight",
                    "attempt": attempt,
                    "input_hashes": {key: row["sha256"] for key, row in sorted(self.plan["inputs"].items())},
                    "output_hashes": {},
                    "predecessor_state_hash": ZERO_HASH,
                    "details": {"error_type": type(error).__name__, "error": str(error)},
                }
            )
            raise
        self.journal.append(
            {
                "event": "resource_preflight_passed",
                "plan_sha256": self.plan["plan_sha256"],
                "stage_id": None,
                "kind": "resource_preflight",
                "attempt": attempt,
                "input_hashes": {key: row["sha256"] for key, row in sorted(self.plan["inputs"].items())},
                "output_hashes": {"report_sha256": sha256_bytes(canonical_json(report))},
                "predecessor_state_hash": ZERO_HASH,
                "details": {"report": report},
            }
        )

    def _next_attempt(self, stage_id: str) -> int:
        starts = [event for event in self.journal.read() if event["event"] == "stage_started" and event["stage_id"] == stage_id]
        return max((int(event["attempt"]) for event in starts), default=0) + 1

    def _block_start_state(self, stage: StageSpec, generation: int) -> str:
        if not stage.block_layers:
            raise ValueError("re-anchor stage does not declare its causal layer block")
        first_id = f"causal_encode.layer_{stage.block_layers[0]:03d}"
        events = self.journal.read()
        completed_attempts = {
            int(event["attempt"])
            for event in events
            if event["event"] == "stage_completed" and event["stage_id"] == first_id
            and int(event.get("generation", 0)) == generation
        }
        starts = [
            event
            for event in events
            if event["event"] == "stage_started"
            and event["stage_id"] == first_id
            and int(event.get("generation", 0)) == generation
            and int(event["attempt"]) in completed_attempts
        ]
        if len(starts) != 1:
            raise ValueError(f"cannot resolve unique predecessor state for rollback block {stage.block_layers}")
        return _require_hash(starts[0]["predecessor_state_hash"], "rollback predecessor state")

    def _recover_unjournaled_result(self, request: StageRequest) -> tuple[StageResult, StageRequest] | None:
        events = self.journal.read()
        starts = [event for event in events if event["event"] == "stage_started" and event["stage_id"] == request.stage_id]
        if not starts:
            return None
        latest = starts[-1]
        prior_attempt = int(latest["attempt"])
        if prior_attempt != request.attempt - 1:
            return None
        terminated = any(
            event["stage_id"] == request.stage_id
            and int(event["attempt"]) == prior_attempt
            and event["event"] in ("stage_completed", "stage_failed")
            for event in events
        )
        if terminated:
            return None
        prior_dir = self.root / "attempts" / request.stage_id / f"attempt-{prior_attempt:04d}"
        receipt_path = prior_dir / ".runner-result.json"
        if not receipt_path.exists():
            return None
        prior_request = dataclasses.replace(request, attempt=prior_attempt, output_dir=prior_dir)
        result = _read_stage_result(prior_dir, prior_request.request_sha256)
        # Reuse the completed attempt rather than creating a new output location.
        return result, prior_request

    @staticmethod
    def _event_body(
        event: str,
        request: StageRequest,
        outputs: Mapping[str, Any],
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "event": event,
            "plan_sha256": request.plan_sha256,
            "stage_id": request.stage_id,
            "kind": request.kind,
            "attempt": request.attempt,
            "generation": request.generation,
            "generation_context": request.generation_context,
            "installed_layer_prefix": list(request.installed_layer_prefix),
            "request_sha256": request.request_sha256,
            "input_hashes": {
                "static": {key: row["sha256"] for key, row in sorted(request.static_inputs.items())},
                "dependencies": {
                    key: row["artifact_sha256"] for key, row in sorted(request.dependency_artifacts.items())
                },
            },
            "output_hashes": dict(outputs),
            "predecessor_state_hash": request.predecessor_state_hash,
            "details": dict(details),
        }


def _replay(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    try:
        events = EventJournal(root / "events.jsonl").read()
    except Exception as error:
        return {
            "failures": [f"journal:{type(error).__name__}:{error}"],
            "completed": {},
            "installed_state_hash": ZERO_HASH,
            "disposition": "invalid",
            "events": [],
            "complete": False,
            "pending_gate": None,
            "pending_supersession": None,
            "generation": 0,
            "generation_context": {},
        }
    if events[0]["event"] != "campaign_planned" or events[0].get("plan_sha256") != plan["plan_sha256"]:
        failures.append("journal:invalid-planned-event")
    stages = {row["stage_id"]: row for row in plan["stages"]}
    stage_order = [row["stage_id"] for row in plan["stages"]]
    retirement_plans = {
        event["event_sha256"]: event
        for event in events
        if event.get("event") == "artifacts_retirement_planned"
    }
    retired_by_artifact: dict[str, list[dict[str, Any]]] = {}
    for event in retirement_plans.values():
        details = event.get("details")
        inputs = event.get("input_hashes")
        files = details.get("files", []) if isinstance(details, Mapping) else []
        producer_sha256 = (
            inputs.get("producer_artifact_sha256") if isinstance(inputs, Mapping) else None
        )
        if isinstance(producer_sha256, str) and isinstance(files, list):
            retired_by_artifact.setdefault(producer_sha256, []).extend(files)
    completed: dict[str, dict[str, Any]] = {}
    started: set[tuple[str, int]] = set()
    start_events: dict[tuple[str, int], dict[str, Any]] = {}
    completion_events: dict[str, dict[str, Any]] = {}
    completed_predecessors: dict[str, str] = {}
    terminated: set[tuple[str, int]] = set()
    state_hash = ZERO_HASH
    disposition = "running"
    completion_seen = False
    gate_decisions_seen: set[tuple[str, int]] = set()
    retirement_completions_seen: set[str] = set()
    pending_gate: str | None = None
    pending_supersession: str | None = None
    generation = 0
    generation_context: dict[str, Any] = {}
    for event in events[1:]:
        if event.get("plan_sha256") != plan["plan_sha256"]:
            failures.append(f"journal[{event['event_index']}]:plan-drift")
            continue
        kind = event["event"]
        stage_id = event.get("stage_id")
        raw_attempt = event.get("attempt", 0)
        if isinstance(raw_attempt, bool) or not isinstance(raw_attempt, int) or raw_attempt < 0:
            failures.append(f"journal[{event['event_index']}]:invalid-attempt")
            attempt = -1
        else:
            attempt = raw_attempt
        is_recovery_event = kind == "journal_tail_recovered" and event.get("kind") == "journal_recovery"
        if pending_gate is not None and not (
            (kind == "gate_decision" and stage_id == pending_gate) or is_recovery_event
        ):
            failures.append(f"journal[{event['event_index']}]:missing-immediate-gate-decision")
            pending_gate = None
        if pending_supersession is not None and not (
            kind == "generation_superseded" or is_recovery_event
        ):
            failures.append(f"journal[{event['event_index']}]:missing-immediate-generation-supersession")
            pending_supersession = None
        if kind == "campaign_completed":
            if completion_seen:
                failures.append("journal:duplicate-campaign-completion")
            completion_seen = True
            disposition = "complete"
            if set(completed) != set(stage_order):
                failures.append("journal:campaign-completed-before-all-stages")
            if event.get("predecessor_state_hash") != state_hash:
                failures.append("journal:completion-state-mismatch")
            expected_outputs = {key: row["artifact_sha256"] for key, row in sorted(completed.items())}
            if event.get("output_hashes") != expected_outputs:
                failures.append("journal:completion-output-mismatch")
            continue
        if kind == "journal_tail_recovered":
            details = event.get("details", {})
            outputs = event.get("output_hashes")
            fragment_hash = outputs.get("fragment_sha256") if isinstance(outputs, Mapping) else None
            try:
                if not isinstance(details, Mapping):
                    raise ValueError("journal recovery details must be a mapping")
                _require_hash(fragment_hash, "journal recovery fragment hash")
                fragment_bytes = int(details["fragment_bytes"])
                relative = _safe_relative_file(details["quarantine_path"])
                quarantine = root / relative
                if (
                    stage_id is not None
                    or event.get("kind") != "journal_recovery"
                    or event.get("input_hashes") != {}
                    or event.get("predecessor_state_hash") != ZERO_HASH
                    or fragment_bytes <= 0
                    or not quarantine.is_file()
                    or quarantine.stat().st_size != fragment_bytes
                    or sha256_file(quarantine) != fragment_hash
                ):
                    raise ValueError("invalid recovery binding")
            except Exception:
                failures.append(f"journal[{event['event_index']}]:invalid-tail-recovery")
            continue
        if kind in ("resource_preflight_passed", "resource_preflight_failed"):
            expected_inputs = {key: row["sha256"] for key, row in sorted(plan["inputs"].items())}
            if stage_id is not None or event.get("kind") != "resource_preflight":
                failures.append(f"journal[{event['event_index']}]:invalid-preflight-identity")
            if event.get("input_hashes") != expected_inputs or event.get("predecessor_state_hash") != ZERO_HASH:
                failures.append(f"journal[{event['event_index']}]:invalid-preflight-binding")
            if kind == "resource_preflight_passed":
                details = event.get("details")
                outputs = event.get("output_hashes")
                report = details.get("report") if isinstance(details, Mapping) else None
                report_sha256 = outputs.get("report_sha256") if isinstance(outputs, Mapping) else None
                if not isinstance(report, dict) or report_sha256 != sha256_bytes(canonical_json(report)):
                    failures.append(f"journal[{event['event_index']}]:invalid-preflight-report")
            continue
        if kind in ("artifacts_retirement_planned", "artifacts_retirement_completed"):
            if plan["definition"].get("retention_mode") != "capture-plus-ledger":
                failures.append(f"journal[{event['event_index']}]:retirement-in-full-mode")
                continue
            producer_id = stage_id
            if (
                not isinstance(producer_id, str)
                or producer_id not in completed
                or event.get("kind") != "artifact_retirement"
            ):
                failures.append(f"journal[{event['event_index']}]:invalid-retirement-producer")
                continue
            details = event.get("details", {})
            if not isinstance(details, Mapping):
                failures.append(f"journal[{event['event_index']}]:invalid-retirement-details")
                continue
            trigger_id = details.get("trigger_stage_id")
            files = details.get("files")
            if (
                not isinstance(trigger_id, str)
                or trigger_id not in completed
                or not isinstance(files, list)
                or not files
            ):
                failures.append(f"journal[{event['event_index']}]:invalid-retirement-trigger-or-files")
                continue
            try:
                normalized = [
                    {"path": _safe_relative_file(row["path"]), "bytes": int(row["bytes"]), "sha256": _require_hash(row["sha256"], "retired hash")}
                    for row in files
                ]
            except Exception:
                failures.append(f"journal[{event['event_index']}]:invalid-retirement-file-record")
                continue
            if len({row["path"] for row in normalized}) != len(normalized) or any(row["bytes"] < 0 for row in normalized):
                failures.append(f"journal[{event['event_index']}]:duplicate-or-invalid-retirement-file")
            expected_inputs = {
                "producer_artifact_sha256": completed[producer_id]["artifact_sha256"],
                "trigger_artifact_sha256": completed[trigger_id]["artifact_sha256"],
            }
            if event.get("input_hashes") != expected_inputs or event.get("predecessor_state_hash") != ZERO_HASH:
                failures.append(f"journal[{event['event_index']}]:retirement-binding-mismatch")
            if kind == "artifacts_retirement_planned" and event.get("output_hashes") != {}:
                failures.append(f"journal[{event['event_index']}]:retirement-plan-output-mismatch")
            if kind == "artifacts_retirement_completed":
                planned_hash = details.get("planned_event_sha256")
                planned = retirement_plans.get(planned_hash) if isinstance(planned_hash, str) else None
                planned_details = planned.get("details") if isinstance(planned, Mapping) else None
                if (
                    planned is None
                    or planned.get("stage_id") != producer_id
                    or not isinstance(planned_details, Mapping)
                    or planned_details.get("files") != files
                ):
                    failures.append(f"journal[{event['event_index']}]:retirement-plan-mismatch")
                if isinstance(planned_hash, str):
                    if planned_hash in retirement_completions_seen:
                        failures.append(f"journal[{event['event_index']}]:duplicate-retirement-completion")
                    retirement_completions_seen.add(planned_hash)
                retired_count = None
                outputs = event.get("output_hashes")
                if isinstance(outputs, Mapping):
                    retired_count = outputs.get("retired_file_count")
                if (
                    not isinstance(retired_count, int)
                    or isinstance(retired_count, bool)
                    or retired_count != len(files)
                    or set(outputs) != {"retired_file_count"}
                    or planned is None
                    or attempt != planned.get("attempt")
                ):
                    failures.append(f"journal[{event['event_index']}]:retirement-completion-mismatch")
                producer_root = root / completed[producer_id]["path"]
                try:
                    paths = [_confined_path(producer_root, row["path"]) for row in normalized]
                    if any(path.exists() for path in paths):
                        failures.append(f"journal[{event['event_index']}]:retired-file-still-present")
                except Exception:
                    failures.append(f"journal[{event['event_index']}]:invalid-retired-file-path")
            continue
        if kind == "generation_superseded":
            details = event.get("details", {})
            if not isinstance(details, dict):
                failures.append(f"journal[{event['event_index']}]:invalid-generation-supersession")
                continue
            prior = details.get("prior_generation")
            new = details.get("new_generation")
            invalidated = details.get("invalidated_artifacts")
            failed_gate = details.get("failed_gate_stage_id")
            max_generations = int(
                plan["definition"].get("max_generations", DEFAULT_MAX_GENERATIONS)
            )
            if (
                stage_id is not None
                or event.get("kind") != "generation_replan"
                or prior != generation
                or new != generation + 1
                or new >= max_generations
                or attempt != new
                or not isinstance(invalidated, list)
                or failed_gate not in completed
                or pending_supersession != failed_gate
                or isinstance(event.get("generation"), bool)
                or event.get("generation") != generation
                or event.get("output_hashes") != {}
            ):
                failures.append(f"journal[{event['event_index']}]:invalid-generation-supersession")
                continue
            gate_event = completion_events.get(failed_gate)
            decision_events = [
                candidate
                for candidate in events[: int(event["event_index"])]
                if candidate.get("event") == "gate_decision"
                and candidate.get("stage_id") == failed_gate
                and not isinstance(candidate.get("generation", 0), bool)
                and candidate.get("generation", 0) == generation
            ]
            if len(decision_events) != 1:
                failures.append(f"journal[{event['event_index']}]:generation-gate-decision-ambiguity")
                continue
            decision_event = decision_events[0]
            decision_details = decision_event.get("details", {})
            expected_decision_sha256 = decision_event.get("event_sha256")
            expected_inputs = {
                "failed_gate_artifact_sha256": completed[failed_gate]["artifact_sha256"],
                "gate_decision_sha256": expected_decision_sha256,
            }
            if (
                set(details) != {
                    "prior_generation", "new_generation", "failed_gate_stage_id",
                    "failed_gate_artifact_sha256", "gate_decision_sha256", "action",
                    "rollback_state_hash", "invalidated_artifacts",
                }
                or details.get("action") != decision_details.get("action")
                or details.get("action") not in {"rollback", "request_reallocation"}
                or details.get("rollback_state_hash") != decision_details.get("rollback_state_hash")
                or event.get("predecessor_state_hash") != decision_details.get("rollback_state_hash")
                or event.get("input_hashes") != expected_inputs
                or details.get("failed_gate_artifact_sha256")
                != completed[failed_gate]["artifact_sha256"]
                or details.get("gate_decision_sha256") != expected_decision_sha256
                or event.get("generation_context") != decision_event.get("generation_context", {})
                or gate_event is None
            ):
                failures.append(f"journal[{event['event_index']}]:generation-gate-binding-mismatch")
                continue
            expected_ids = [
                planned_id
                for planned_id in stage_order[stage_order.index("allocation") :]
                if planned_id in completed
            ]
            if any(
                not isinstance(row, dict)
                or set(row) != {"stage_id", "path", "artifact_sha256"}
                for row in invalidated
            ):
                failures.append(f"journal[{event['event_index']}]:generation-invalidation-record-malformed")
                continue
            if [row.get("stage_id") for row in invalidated] != expected_ids:
                failures.append(f"journal[{event['event_index']}]:generation-invalidation-set-mismatch")
                continue
            valid = True
            for row in invalidated:
                active = completed.get(row["stage_id"])
                if active is None or any(active.get(key) != row.get(key) for key in ("path", "artifact_sha256")):
                    valid = False
            if not valid:
                failures.append(f"journal[{event['event_index']}]:generation-invalidated-artifact-mismatch")
                continue
            for invalidated_id in expected_ids:
                completed.pop(invalidated_id, None)
                completion_events.pop(invalidated_id, None)
                completed_predecessors.pop(invalidated_id, None)
            identity = completed.get("identity")
            state_hash = (
                sha256_bytes(canonical_json({"identity": identity["artifact_sha256"]}))
                if identity is not None
                else ZERO_HASH
            )
            generation = int(new)
            generation_context = {
                "generation_event_sha256": event["event_sha256"],
                **{key: value for key, value in details.items()},
            }
            disposition = "replan_ready"
            completion_seen = False
            pending_supersession = None
            continue
        if not isinstance(stage_id, str):
            failures.append(f"journal[{event['event_index']}]:invalid-stage-id")
            continue
        if stage_id not in stages:
            failures.append(f"journal[{event['event_index']}]:unknown-stage")
            continue
        stage = stages[stage_id]
        key = (stage_id, attempt)
        event_generation = event.get("generation", 0)
        if (
            isinstance(event_generation, bool)
            or not isinstance(event_generation, int)
            or event_generation != generation
            or event.get("generation_context", {}) != generation_context
        ):
            failures.append(f"journal[{event['event_index']}]:generation-binding-mismatch")
            continue
        if event.get("kind") != stage["kind"]:
            failures.append(f"journal[{event['event_index']}]:kind-mismatch")
        expected_static = {key: row["sha256"] for key, row in sorted(plan["inputs"].items())}
        expected_dependencies = {
            dep: completed[dep]["artifact_sha256"]
            for dep in stage.get("dependencies", [])
            if dep in completed
        }
        missing = [dep for dep in stage.get("dependencies", []) if dep not in completed]
        if kind == "stage_started":
            if missing:
                failures.append(f"journal[{event['event_index']}]:dependency-not-complete")
            if event.get("predecessor_state_hash") != state_hash:
                failures.append(f"journal[{event['event_index']}]:predecessor-state-mismatch")
            inputs = event.get("input_hashes", {})
            if (
                not isinstance(inputs, Mapping)
                or inputs.get("static") != expected_static
                or inputs.get("dependencies") != expected_dependencies
            ):
                failures.append(f"journal[{event['event_index']}]:input-hash-mismatch")
            try:
                expected_prefix = _installed_prefix_from_completed(root, plan, completed, state_hash, events)
            except Exception:
                failures.append(f"journal[{event['event_index']}]:invalid-installed-layer-prefix")
                expected_prefix = ()
            if event.get("installed_layer_prefix", []) != list(expected_prefix):
                failures.append(f"journal[{event['event_index']}]:installed-layer-prefix-mismatch")
            expected_request = {
                "stage_id": stage_id,
                "kind": stage["kind"],
                "attempt": attempt,
                "plan_sha256": plan["plan_sha256"],
                "experiment_spec_sha256": plan["experiment_spec"]["sha256"],
                "static_inputs": expected_static,
                "dependency_artifacts": expected_dependencies,
                "predecessor_state_hash": state_hash,
                "generation": generation,
                "generation_context": generation_context,
                "installed_layer_prefix": list(expected_prefix),
                "layer": stage.get("layer"),
                "block_layers": list(stage.get("block_layers", [])),
            }
            if event.get("request_sha256") != sha256_bytes(canonical_json(expected_request)):
                failures.append(f"journal[{event['event_index']}]:request-binding-mismatch")
            if key in started:
                failures.append(f"journal[{event['event_index']}]:duplicate-start")
            started.add(key)
            start_events[key] = event
        elif kind in ("stage_completed", "stage_failed"):
            if key not in started or key in terminated:
                failures.append(f"journal[{event['event_index']}]:invalid-termination")
            start = start_events.get(key)
            if start is not None:
                for field in ("kind", "request_sha256", "input_hashes", "predecessor_state_hash"):
                    if event.get(field) != start.get(field):
                        failures.append(f"journal[{event['event_index']}]:termination-{field}-mismatch")
            terminated.add(key)
            if kind == "stage_completed":
                if stage_id in completed:
                    failures.append(f"journal[{event['event_index']}]:duplicate-stage-completion")
                    continue
                outputs = event.get("output_hashes")
                artifact = outputs.get("artifact") if isinstance(outputs, Mapping) else None
                if not isinstance(artifact, dict):
                    failures.append(f"journal[{event['event_index']}]:missing-artifact")
                    continue
                try:
                    relative = _safe_relative_file(artifact.get("path"))
                    artifact_path = root / relative
                    matches = _artifact_matches_with_retirements(
                        artifact_path,
                        root,
                        artifact,
                        retired_by_artifact.get(artifact.get("artifact_sha256", ""), []),
                    )
                except Exception as error:
                    failures.append(f"artifact:{stage_id}:{type(error).__name__}")
                    continue
                if not matches:
                    failures.append(f"artifact:{stage_id}:identity-drift")
                    continue
                try:
                    replay_request = StageRequest(
                        campaign_dir=root,
                        output_dir=artifact_path,
                        stage_id=stage_id,
                        kind=stage["kind"],
                        attempt=attempt,
                        plan_sha256=plan["plan_sha256"],
                        experiment_spec=plan["experiment_spec"],
                        static_inputs=plan["inputs"],
                        dependency_artifacts={dep: completed[dep] for dep in stage.get("dependencies", [])},
                        predecessor_state_hash=event["predecessor_state_hash"],
                        generation=generation,
                        generation_context=dict(event.get("generation_context", {})),
                        installed_layer_prefix=tuple(event.get("installed_layer_prefix", [])),
                        layer=stage.get("layer"),
                        block_layers=tuple(stage.get("block_layers", [])),
                    )
                    result = _read_stage_result(
                        artifact_path,
                        event.get("request_sha256"),
                        retired_by_artifact.get(artifact.get("artifact_sha256", ""), []),
                    )
                    if stage["kind"] in GATE_KINDS:
                        _validate_gate(result, replay_request)
                    _validate_causal_result(replay_request, result)
                except Exception as error:
                    failures.append(f"artifact:{stage_id}:result:{type(error).__name__}")
                    continue
                completion_details = event.get("details")
                declared_state = (
                    completion_details.get("installed_state_hash")
                    if isinstance(completion_details, Mapping)
                    else None
                )
                if stage["kind"] == "identity":
                    expected_state = sha256_bytes(canonical_json({"identity": artifact["artifact_sha256"]}))
                elif stage["kind"] == "causal_encode":
                    expected_state = sha256_bytes(
                        canonical_json(
                            {
                                "predecessor_state_hash": state_hash,
                                "stage_id": stage_id,
                                "layer": stage.get("layer"),
                                "artifact_sha256": artifact["artifact_sha256"],
                            }
                        )
                    )
                else:
                    expected_state = state_hash
                if declared_state != expected_state:
                    failures.append(f"journal[{event['event_index']}]:installed-state-mismatch")
                else:
                    state_hash = expected_state
                completed[stage_id] = artifact
                completed_predecessors[stage_id] = event["predecessor_state_hash"]
                completion_events[stage_id] = event
                if stage["kind"] in GATE_KINDS:
                    pending_gate = stage_id
        elif kind == "gate_decision":
            decision_key = (stage_id, event_generation)
            if decision_key in gate_decisions_seen:
                failures.append(f"journal[{event['event_index']}]:duplicate-gate-decision")
                continue
            gate_decisions_seen.add(decision_key)
            if stage["kind"] not in GATE_KINDS or stage_id not in completed:
                failures.append(f"journal[{event['event_index']}]:invalid-gate-decision")
                continue
            decision_details = event.get("details", {})
            if not isinstance(decision_details, dict):
                failures.append(f"journal[{event['event_index']}]:gate-details-mismatch")
                continue
            action = decision_details.get("action")
            if action not in GATE_POLICIES:
                failures.append(f"journal[{event['event_index']}]:invalid-gate-action")
                continue
            receipt_path = root / completed[stage_id]["path"] / ".runner-result.json"
            result = StageResult(dict(json.loads(receipt_path.read_text())["metadata"]))
            gate_request = StageRequest(
                campaign_dir=root,
                output_dir=root / completed[stage_id]["path"],
                stage_id=stage_id,
                kind=stage["kind"],
                attempt=attempt,
                plan_sha256=plan["plan_sha256"],
                experiment_spec=plan["experiment_spec"],
                static_inputs=plan["inputs"],
                dependency_artifacts={dep: completed[dep] for dep in stage.get("dependencies", [])},
                predecessor_state_hash=event["predecessor_state_hash"],
                generation=generation,
                generation_context=dict(event.get("generation_context", {})),
                installed_layer_prefix=tuple(event.get("installed_layer_prefix", [])),
                layer=stage.get("layer"),
                block_layers=tuple(stage.get("block_layers", [])),
            )
            gate = _validate_gate(result, gate_request)
            if set(decision_details) != {
                "gate", "action", "disposition", "rollback_state_hash"
            }:
                failures.append(f"journal[{event['event_index']}]:gate-details-mismatch")
            if decision_details.get("gate") != gate or event.get("output_hashes") != {}:
                failures.append(f"journal[{event['event_index']}]:gate-result-binding-mismatch")
            expected_action = "continue" if gate["passed"] else plan["definition"]["reanchor_failure_policy"]
            if action != expected_action:
                failures.append(f"journal[{event['event_index']}]:gate-policy-mismatch")
            expected_disposition = {
                "continue": "running",
                "rollback": "replan_pending",
                "request_reallocation": "replan_pending",
            }.get(action)
            disposition = event.get("details", {}).get("disposition", "running")
            if disposition != expected_disposition:
                failures.append(f"journal[{event['event_index']}]:gate-disposition-mismatch")
            if action == "continue" and decision_details.get("rollback_state_hash") is not None:
                failures.append(f"journal[{event['event_index']}]:unexpected-gate-rollback-state")
            completed_event = completion_events.get(stage_id)
            if completed_event is not None:
                for field in (
                    "attempt", "request_sha256", "input_hashes", "predecessor_state_hash",
                    "generation", "generation_context", "installed_layer_prefix",
                ):
                    if event.get(field) != completed_event.get(field):
                        failures.append(f"journal[{event['event_index']}]:gate-{field}-mismatch")
            if action != "continue":
                rollback = event.get("details", {}).get("rollback_state_hash")
                block_layers = stage.get("block_layers", [])
                first_encode = f"causal_encode.layer_{int(block_layers[0]):03d}" if block_layers else ""
                expected_rollback = completed_predecessors.get(first_encode)
                if rollback != expected_rollback:
                    failures.append(f"journal[{event['event_index']}]:rollback-state-mismatch")
            if action != "continue":
                rollback = event.get("details", {}).get("rollback_state_hash")
                try:
                    _require_hash(rollback, "gate rollback state")
                except Exception:
                    failures.append(f"journal[{event['event_index']}]:invalid-gate-rollback-state")
                pending_supersession = stage_id
            pending_gate = None
        else:
            failures.append(f"journal[{event['event_index']}]:unknown-event:{kind}")
    # Completed stages must form a prefix unless a gate halted the campaign.
    prefix = stage_order[: len(completed)]
    if list(completed) != prefix:
        failures.append("journal:completed-stages-are-not-a-prefix")
    return {
        "failures": failures,
        "completed": completed,
        "installed_state_hash": state_hash,
        "disposition": disposition,
        "events": events,
        "complete": completion_seen,
        "pending_gate": pending_gate,
        "pending_supersession": pending_supersession,
        "generation": generation,
        "generation_context": generation_context,
    }


def audit_campaign(
    campaign_dir: str | Path,
    adapter: StageAdapter | None = None,
    *,
    verify_drift: bool = True,
) -> dict[str, Any]:
    root, plan = _load_plan(campaign_dir)
    replay = _replay(root, plan)
    failures = list(replay["failures"])
    drift = _current_drift(plan, adapter) if verify_drift else []
    failures.extend(drift)
    return {
        "schema": "quant-pipeline.campaign-audit.v1",
        "campaign_dir": str(root),
        "plan_sha256": plan["plan_sha256"],
        "integrity_ok": not failures,
        "complete": bool(replay["complete"] and not failures),
        "disposition": replay["disposition"],
        "completed_stage_count": len(replay["completed"]),
        "total_stage_count": len(plan["stages"]),
        "installed_state_hash": replay["installed_state_hash"],
        "failures": failures,
        "drift": drift,
        "completed": replay["completed"],
        "journal_head_sha256": replay["events"][-1]["event_sha256"] if replay["events"] else None,
        "pending_gate": replay.get("pending_gate"),
        "pending_supersession": replay.get("pending_supersession"),
        "generation": replay.get("generation", 0),
        "generation_context": replay.get("generation_context", {}),
    }


def status_campaign(campaign_dir: str | Path, adapter: StageAdapter | None = None) -> dict[str, Any]:
    audit = audit_campaign(campaign_dir, adapter)
    root, plan = _load_plan(campaign_dir)
    completed = set(audit["completed"])
    next_stage = next((row["stage_id"] for row in plan["stages"] if row["stage_id"] not in completed), None)
    return {
        "schema": "quant-pipeline.campaign-status.v1",
        "campaign_dir": str(root),
        "plan_sha256": plan["plan_sha256"],
        "integrity_ok": audit["integrity_ok"],
        "complete": audit["complete"],
        "disposition": audit["disposition"],
        "completed_stage_count": audit["completed_stage_count"],
        "total_stage_count": audit["total_stage_count"],
        "next_stage": next_stage,
        "installed_state_hash": audit["installed_state_hash"],
        "generation": audit.get("generation", 0),
        "pending_gate": audit.get("pending_gate"),
        "pending_supersession": audit.get("pending_supersession"),
        "drift": audit["drift"],
        "failures": audit["failures"],
    }
