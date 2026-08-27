"""One fail-closed schema for published quantitative result claims.

The row seal proves the exact claim record.  It intentionally does not turn a
missing report or independent replay into evidence: unavailable hashes remain
``None`` and the status must say so.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..core.artifacts import canonical_json, sha256_bytes


ROW_SCHEMA = "shapleymcg.result-ledger-row.v1"
LEDGER_SCHEMA = "shapleymcg.result-ledger.v1"
STATUS_STATES = {
    "pending",
    "source-sealed",
    "independently-verified",
    "reconstructed",
    "superseded",
}
DESIGN_KINDS = {
    "legacy-reported-endpoint",
    "matched-allocation-control",
    "frozen-rate-factory-diagnostic",
    "joint-factory-rate-allocation",
}
HASH_FIELDS = ("evidence_sha256", "report_sha256", "verification_sha256")
DESIGN_HASH_FIELDS = (
    "allocation_plan_sha256",
    "score_calibration_sha256",
    "budget_closure_sha256",
    "untouched_endpoint_sha256",
    "replay_receipt_sha256",
)


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_mapping(value: Any, label: str, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} lacks required fields: {missing}")
    return value


def validate_row(row: Mapping[str, Any]) -> None:
    required = (
        "schema",
        "id",
        "method_profile",
        "design",
        "parent",
        "panel",
        "backend",
        "evaluator",
        "scope",
        "rate",
        "value",
        "evidence",
        "status",
        "row_sha256",
    )
    _required_mapping(row, "result row", required)
    if row["schema"] != ROW_SCHEMA:
        raise ValueError("unsupported result-row schema")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise ValueError("result row id must be non-empty")

    method = _required_mapping(
        row["method_profile"],
        "method_profile",
        ("allocation", "candidate_factory", "calibration", "codec", "encoding"),
    )
    for key, value in method.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"method_profile.{key} must be a string or null")

    design = _required_mapping(
        row["design"], "design", ("kind", "rate_selection", "factory_selection", "bindings")
    )
    if design["kind"] not in DESIGN_KINDS:
        raise ValueError("unsupported experiment design kind")
    if design["rate_selection"] not in {"frozen", "joint", "not-recorded"}:
        raise ValueError("unsupported rate-selection design")
    if design["factory_selection"] not in {"fixed", "selected", "joint", "not-recorded"}:
        raise ValueError("unsupported factory-selection design")
    if design["kind"] == "frozen-rate-factory-diagnostic" and (
        design["rate_selection"] != "frozen" or design["factory_selection"] != "selected"
    ):
        raise ValueError("frozen-rate factory diagnostics must freeze rates and select factories")
    if design["kind"] == "joint-factory-rate-allocation" and (
        design["rate_selection"] != "joint" or design["factory_selection"] != "joint"
    ):
        raise ValueError("joint factory/rate rows must identify both selections as joint")
    bindings = _required_mapping(
        design["bindings"],
        "design.bindings",
        (*DESIGN_HASH_FIELDS, "candidate_inventories"),
    )
    for key in DESIGN_HASH_FIELDS:
        if bindings[key] is not None and not _is_sha256(bindings[key]):
            raise ValueError(f"design.bindings.{key} must be a lowercase SHA-256 or null")
    inventories = bindings["candidate_inventories"]
    if not isinstance(inventories, list):
        raise ValueError("design candidate inventories must be a list")
    roles: list[str] = []
    for inventory in inventories:
        _required_mapping(
            inventory,
            "candidate inventory binding",
            ("role", "repository", "revision", "inventory_sha256", "file_sha256"),
        )
        roles.append(inventory["role"])
        if not all(
            isinstance(inventory[key], str) and inventory[key].strip()
            for key in ("role", "repository", "revision")
        ):
            raise ValueError("candidate inventory role, repository, and revision must be explicit")
        for key in ("inventory_sha256", "file_sha256"):
            if not _is_sha256(inventory[key]):
                raise ValueError(f"candidate inventory {key} must be a lowercase SHA-256")
    if len(roles) != len(set(roles)):
        raise ValueError("candidate inventory roles must be unique")
    if design["kind"] in {"frozen-rate-factory-diagnostic", "joint-factory-rate-allocation"}:
        if any(bindings[key] is None for key in DESIGN_HASH_FIELDS):
            raise ValueError("factory-selection result rows require every design binding hash")
        if len(inventories) < 2:
            raise ValueError("factory-selection result rows require at least two bound candidate inventories")

    parent = _required_mapping(row["parent"], "parent", ("repository", "revision"))
    if not all(isinstance(parent[key], str) and parent[key].strip() for key in parent):
        raise ValueError("parent repository and revision must be explicit strings")

    panel = _required_mapping(row["panel"], "panel", ("id", "sha256", "positions"))
    if not isinstance(panel["id"], str) or not panel["id"].strip():
        raise ValueError("panel id must be explicit")
    if panel["sha256"] is not None and not _is_sha256(panel["sha256"]):
        raise ValueError("panel.sha256 must be a lowercase SHA-256 or null")
    positions = panel["positions"]
    if positions is not None and (not isinstance(positions, int) or isinstance(positions, bool) or positions <= 0):
        raise ValueError("panel.positions must be a positive integer or null")

    backend = _required_mapping(row["backend"], "backend", ("attention", "kv_cache"))
    if not all(isinstance(backend[key], str) and backend[key].strip() for key in backend):
        raise ValueError("backend attention and KV cache must be explicit strings")

    evaluator = _required_mapping(row["evaluator"], "evaluator", ("metric", "implementation"))
    if not all(isinstance(evaluator[key], str) and evaluator[key].strip() for key in evaluator):
        raise ValueError("evaluator metric and implementation must be explicit strings")

    scope = _required_mapping(row["scope"], "scope", ("weights", "nonexpert", "comparison"))
    if not all(isinstance(scope[key], str) and scope[key].strip() for key in scope):
        raise ValueError("result scope must be explicit")

    rate = _required_mapping(row["rate"], "rate", ("value", "unit", "scope"))
    if rate["value"] is not None and (
        not isinstance(rate["value"], (int, float))
        or isinstance(rate["value"], bool)
        or not math.isfinite(float(rate["value"]))
        or float(rate["value"]) < 0.0
    ):
        raise ValueError("rate.value must be finite and non-negative or null")
    if not isinstance(rate["unit"], str) or not rate["unit"].strip():
        raise ValueError("rate.unit must be explicit")
    if not isinstance(rate["scope"], str) or not rate["scope"].strip():
        raise ValueError("rate.scope must be explicit")

    value = _required_mapping(row["value"], "value", ("metric", "value"))
    if not isinstance(value["metric"], str) or not value["metric"].strip():
        raise ValueError("value.metric must be explicit")
    if value["value"] is not None and (
        not isinstance(value["value"], (int, float))
        or isinstance(value["value"], bool)
        or not math.isfinite(float(value["value"]))
    ):
        raise ValueError("value.value must be finite or null")

    evidence = _required_mapping(
        row["evidence"],
        "evidence",
        ("path", "evidence_sha256", "report_sha256", "verification_sha256"),
    )
    if evidence["path"] is not None and (
        not isinstance(evidence["path"], str) or not evidence["path"].strip()
    ):
        raise ValueError("evidence.path must be a non-empty string or null")
    for key in HASH_FIELDS:
        if evidence[key] is not None and not _is_sha256(evidence[key]):
            raise ValueError(f"evidence.{key} must be a lowercase SHA-256 or null")

    status = _required_mapping(row["status"], "status", ("state", "detail"))
    if status["state"] not in STATUS_STATES:
        raise ValueError("unsupported result status")
    if not isinstance(status["detail"], str) or not status["detail"].strip():
        raise ValueError("result status detail must be explicit")
    if status["state"] == "pending":
        if value["value"] is not None:
            raise ValueError("pending rows cannot contain a measured value")
    elif value["value"] is None:
        raise ValueError("non-pending rows require a measured value")
    if status["state"] == "independently-verified" and any(
        evidence[key] is None for key in HASH_FIELDS
    ):
        raise ValueError("independently verified rows require evidence, report, and verification hashes")
    if status["state"] in {"source-sealed", "reconstructed", "independently-verified"} and evidence["evidence_sha256"] is None:
        raise ValueError(f"{status['state']} rows require an evidence hash")

    body = {key: value for key, value in row.items() if key != "row_sha256"}
    if row["row_sha256"] != _hash_json(body):
        raise ValueError(f"result row seal mismatch: {row['id']}")


def seal_row(body: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(body)
    if "row_sha256" in row:
        raise ValueError("seal_row input already contains row_sha256")
    row["row_sha256"] = _hash_json(row)
    validate_row(row)
    return row


def validate_ledger(ledger: Mapping[str, Any]) -> None:
    _required_mapping(ledger, "results ledger", ("schema", "row_count", "rows", "ledger_sha256"))
    if ledger["schema"] != LEDGER_SCHEMA:
        raise ValueError("unsupported results-ledger schema")
    if not isinstance(ledger["rows"], list):
        raise ValueError("results-ledger rows must be a list")
    if ledger["row_count"] != len(ledger["rows"]):
        raise ValueError("results-ledger row count mismatch")
    ids = []
    for row in ledger["rows"]:
        validate_row(row)
        ids.append(row["id"])
    if ids != sorted(ids):
        raise ValueError("results-ledger rows must be sorted by id")
    if len(ids) != len(set(ids)):
        raise ValueError("results-ledger ids must be unique")
    body = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if ledger["ledger_sha256"] != _hash_json(body):
        raise ValueError("results-ledger seal mismatch")


def seal_ledger(rows: Sequence[Mapping[str, Any]], **metadata: Any) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["id"])
    ledger = {"schema": LEDGER_SCHEMA, **metadata, "row_count": len(ordered), "rows": ordered}
    if "ledger_sha256" in ledger:
        raise ValueError("ledger metadata cannot contain ledger_sha256")
    ledger["ledger_sha256"] = _hash_json(ledger)
    validate_ledger(ledger)
    return ledger


def render_markdown(ledger: Mapping[str, Any]) -> str:
    """Render a deterministic compact table; prose can link to richer evidence."""

    validate_ledger(ledger)
    lines = [
        "| Result | Design | Parent revision | Method | Panel | Backend / KV | Rate | Value | Status |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]
    for row in ledger["rows"]:
        rate = row["rate"]
        value = row["value"]
        lines.append(
            "| {id} | {design} | `{revision}` | {method} / {codec} | {panel} | {attention} / {kv} | {rate} {unit} | {value} | {status} |".format(
                id=row["id"],
                design=row["design"]["kind"],
                revision=row["parent"]["revision"],
                method=row["method_profile"]["allocation"] or "unspecified",
                codec=row["method_profile"]["codec"] or "unspecified",
                panel=row["panel"]["id"],
                attention=row["backend"]["attention"],
                kv=row["backend"]["kv_cache"],
                rate="pending" if rate["value"] is None else format(float(rate["value"]), ".12g"),
                unit=rate["unit"],
                value="pending" if value["value"] is None else format(float(value["value"]), ".12g"),
                status=row["status"]["state"],
            )
        )
    return "\n".join(lines) + "\n"
