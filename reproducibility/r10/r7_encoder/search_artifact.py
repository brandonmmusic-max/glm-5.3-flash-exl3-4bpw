"""Validated output of mandatory rotation, G-scale, and permutation pilots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .constants import (
    HAD_K,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MAX_DRAWS,
    MIN_DRAWS,
    NUM_EXPERTS,
    RECIPE_MARKER,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .permutation import validate_permutation


def _finite_nonzero(
    values: Sequence[float], length: int, label: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{label}: length {len(result)} != {length}")
    if any(not math.isfinite(value) or value == 0 for value in result):
        raise ValueError(f"{label}: values must be finite and nonzero")
    import torch

    stored = torch.tensor(result, dtype=torch.float32).half()
    if not torch.isfinite(stored).all() or (stored == 0).any():
        raise ValueError(f"{label}: values are invalid after FP16 storage")
    return result


def _positive_scales(
    values: Sequence[float], blocks: int, label: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != blocks:
        raise ValueError(f"{label}: need {blocks} per-128 scales")
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{label}: scales must be finite and positive")
    return result


@dataclass(frozen=True)
class ExpertSearch:
    permutation: tuple[int, ...]
    permutation_policy: str
    gate_svh: tuple[float, ...]
    up_svh: tuple[float, ...]
    down_suh: tuple[float, ...]
    gate_n_g_scale: tuple[float, ...]
    up_n_g_scale: tuple[float, ...]
    down_k_g_scale: tuple[float, ...]
    draw: int
    selection_score: float
    selection_score_kind: str


@dataclass(frozen=True)
class LayerSearch:
    layer: int
    draws: int
    gate_up_suh: tuple[float, ...]
    down_svh: tuple[float, ...]
    gate_up_k_g_scale: tuple[float, ...]
    down_n_g_scale: tuple[float, ...]
    shared_draw: int
    shared_heldout_score: float
    experts: Mapping[int, ExpertSearch]
    pilot_evidence_sha256: str
    unverified: bool
    bindings: Mapping[str, str]


def load_layer_search(
    path: str | Path, *, require_verified: bool = True
) -> LayerSearch:
    artifact_path = Path(path)
    payload = read_json(artifact_path)
    if (
        payload.get("marker") != RECIPE_MARKER
        or payload.get("schema") != "r7-search-v2"
    ):
        raise ValueError("foreign search artifact")
    layer = int(payload["layer"])
    draws = int(payload["draws"])
    if not MIN_DRAWS <= draws <= MAX_DRAWS:
        raise ValueError("search draw count outside owner-locked range")
    unverified = bool(payload.get("unverified", True))
    if require_verified and unverified:
        raise ValueError("search artifact is UNVERIFIED and cannot drive an owner run")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in bindings.items()
    ):
        raise ValueError("search artifact lacks complete string provenance bindings")
    required_bindings = {
        "capture_sha256",
        "state_sha256",
        "source_inventory_sha256",
        "numeric_environment_sha256",
        "runtime_inventory_sha256",
        "backend_fingerprint",
        "draws",
        "sample_sha256",
    }
    if set(bindings) != required_bindings or bindings["draws"] != str(draws):
        raise ValueError("search artifact provenance binding set drift")
    shared = payload["shared"]
    experts: dict[int, ExpertSearch] = {}
    for key, raw in payload["experts"].items():
        expert = int(key)
        if not 0 <= expert < NUM_EXPERTS or expert in experts:
            raise ValueError(f"invalid expert search key {key}")
        experts[expert] = ExpertSearch(
            permutation=validate_permutation(raw["permutation"], INTERMEDIATE_SIZE),
            permutation_policy=str(raw["permutation_policy"]),
            gate_svh=_finite_nonzero(raw["gate_svh"], INTERMEDIATE_SIZE, "gate_svh"),
            up_svh=_finite_nonzero(raw["up_svh"], INTERMEDIATE_SIZE, "up_svh"),
            down_suh=_finite_nonzero(raw["down_suh"], INTERMEDIATE_SIZE, "down_suh"),
            gate_n_g_scale=_positive_scales(
                raw["gate_n_g_scale"], INTERMEDIATE_SIZE // HAD_K, "gate_n_g_scale"
            ),
            up_n_g_scale=_positive_scales(
                raw["up_n_g_scale"], INTERMEDIATE_SIZE // HAD_K, "up_n_g_scale"
            ),
            down_k_g_scale=_positive_scales(
                raw["down_k_g_scale"], INTERMEDIATE_SIZE // HAD_K, "down_k_g_scale"
            ),
            draw=int(raw["draw"]),
            selection_score=float(raw["selection_score"]),
            selection_score_kind=str(raw["selection_score_kind"]),
        )
        record = experts[expert]
        if (
            record.permutation_policy
            not in {
                "identity",
                "ldlq_visit_descending_diag",
                "stored_descending_diag",
                "energy_balanced",
                "energy_balanced_contiguous",
            }
            or not 0 <= record.draw < draws
            or not math.isfinite(record.selection_score)
            or record.selection_score_kind
            not in {"deterministic-sketch", "heldout-full-rt"}
        ):
            raise ValueError(f"expert {expert}: invalid selected pilot metadata")
    if set(experts) != set(range(NUM_EXPERTS)):
        raise ValueError("search artifact must cover all 256 experts")
    shared_score = float(shared["heldout_score"])
    shared_draw = int(shared["draw"])
    evidence = str(payload["pilot_evidence_sha256"])
    if not 0 <= shared_draw < draws or not math.isfinite(shared_score):
        raise ValueError("invalid shared selected pilot metadata")
    if len(evidence) != 64 or any(
        character not in "0123456789abcdef" for character in evidence
    ):
        raise ValueError("invalid search evidence digest")
    if require_verified:
        progress = artifact_path.with_name(f".{artifact_path.stem}.pilot-progress.json")
        if not progress.is_file() or sha256_file(progress) != evidence:
            raise ValueError("verified search evidence file is missing or changed")
        progress_payload = read_json(progress)
        if progress_payload.get("bindings") != bindings:
            raise ValueError("search progress/artifact provenance drift")
    return LayerSearch(
        layer=layer,
        draws=draws,
        gate_up_suh=_finite_nonzero(shared["gate_up_suh"], HIDDEN_SIZE, "gate_up_suh"),
        down_svh=_finite_nonzero(shared["down_svh"], HIDDEN_SIZE, "down_svh"),
        gate_up_k_g_scale=_positive_scales(
            shared["gate_up_k_g_scale"], HIDDEN_SIZE // HAD_K, "gate_up_k_g_scale"
        ),
        down_n_g_scale=_positive_scales(
            shared["down_n_g_scale"], HIDDEN_SIZE // HAD_K, "down_n_g_scale"
        ),
        shared_draw=shared_draw,
        shared_heldout_score=shared_score,
        experts=experts,
        pilot_evidence_sha256=evidence,
        unverified=unverified,
        bindings=dict(bindings),
    )


def write_layer_search(path: str | Path, payload: Mapping[str, object]) -> str:
    """Write a search artifact incrementally; validation occurs on read."""

    value = dict(payload)
    value.update({"marker": RECIPE_MARKER, "schema": "r7-search-v2"})
    atomic_write_json(path, value)
    return sha256_bytes(canonical_json_bytes(value))
