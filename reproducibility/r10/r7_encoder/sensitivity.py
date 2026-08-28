"""Incremental probe ledger and deterministic gate/up/down fixed point."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from .allocation import build_curves, solve_exact_allocation
from .constants import ALLOWED_BITS, TensorId
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
)
from .types import CandidateLoss, LayerAllocation


class ProbeLedger:
    """Persist every probe result before a later probe is attempted."""

    RECORD_BINDING_FIELDS = (
        "state_sha256",
        "capture_sha256",
        "search_sha256",
        "source_inventory_sha256",
        "numeric_environment_sha256",
        "runtime_inventory_sha256",
        "backend_fingerprint",
    )

    def __init__(
        self,
        path: str | Path,
        layer: int,
        *,
        fixed_point_iteration: int,
        bindings: Mapping[str, str],
    ) -> None:
        self.path = Path(path)
        self.layer = layer
        self.fixed_point_iteration = fixed_point_iteration
        self.bindings = dict(sorted(bindings.items()))
        if set(self.bindings) != {*self.RECORD_BINDING_FIELDS, "context_map_sha256"}:
            raise ValueError("probe ledger binding key-set drift")
        self._records: dict[tuple[str, int], CandidateLoss] = {}
        if self.path.exists():
            payload = read_json(self.path)
            if payload.get("layer") != layer:
                raise ValueError("probe ledger layer mismatch")
            if int(payload.get("fixed_point_iteration", -1)) != fixed_point_iteration:
                raise ValueError("probe ledger iteration mismatch")
            if payload.get("bindings") != self.bindings:
                raise ValueError("probe ledger provenance binding mismatch")
            for raw in payload.get("records", []):
                tensor_id = TensorId(
                    int(raw["layer"]), int(raw["expert"]), str(raw["projection"])
                )
                record = CandidateLoss(
                    tensor_id=tensor_id,
                    bits=int(raw["bits"]),
                    loss=str(raw["loss"]),
                    mass=str(raw["mass"]),
                    fit_rows=int(raw["fit_rows"]),
                    holdout_rows=int(raw["holdout_rows"]),
                    roundtrip_sha256=str(raw["roundtrip_sha256"]),
                    gate_up_roundtrip_sha256=raw.get("gate_up_roundtrip_sha256"),
                    fixed_point_iteration=int(raw.get("fixed_point_iteration", 0)),
                    context_bits_sha256=str(raw.get("context_bits_sha256", "")),
                    expert_roundtrip_sha256=dict(
                        raw.get("expert_roundtrip_sha256", {})
                    ),
                    state_sha256=str(raw.get("state_sha256", "")),
                    capture_sha256=str(raw.get("capture_sha256", "")),
                    search_sha256=str(raw.get("search_sha256", "")),
                    source_inventory_sha256=str(raw.get("source_inventory_sha256", "")),
                    numeric_environment_sha256=str(
                        raw.get("numeric_environment_sha256", "")
                    ),
                    runtime_inventory_sha256=str(
                        raw.get("runtime_inventory_sha256", "")
                    ),
                    backend_fingerprint=str(raw.get("backend_fingerprint", "")),
                    fit_row_ids_sha256=str(raw.get("fit_row_ids_sha256", "")),
                    holdout_row_ids_sha256=str(raw.get("holdout_row_ids_sha256", "")),
                    permutation_sha256=str(raw.get("permutation_sha256", "")),
                    vector_bundle_sha256=str(raw.get("vector_bundle_sha256", "")),
                    used_cold_fallback=bool(raw.get("used_cold_fallback", False)),
                )
                self._validate_record(record)
                key = (tensor_id.key, record.bits)
                if key in self._records:
                    raise ValueError(f"duplicate sealed probe {key}")
                self._records[key] = record

    def _validate_record(self, record: CandidateLoss) -> None:
        if record.tensor_id.layer != self.layer or record.bits not in ALLOWED_BITS:
            raise ValueError("probe record outside ledger scope")
        if record.fixed_point_iteration != self.fixed_point_iteration:
            raise ValueError("probe record iteration drift")
        for field in self.RECORD_BINDING_FIELDS:
            if getattr(record, field) != self.bindings[field]:
                raise ValueError(f"probe record {field} provenance drift")
        required_hashes = (
            record.context_bits_sha256,
            record.fit_row_ids_sha256,
            record.holdout_row_ids_sha256,
            record.permutation_sha256,
            record.vector_bundle_sha256,
        )
        if any(not value for value in required_hashes) or set(
            record.expert_roundtrip_sha256
        ) != {
            "gate_proj",
            "up_proj",
            "down_proj",
        }:
            raise ValueError("probe record lacks complete replay provenance")

    @property
    def records(self) -> tuple[CandidateLoss, ...]:
        return tuple(
            self._records[key]
            for key in sorted(self._records, key=lambda item: (item[0], item[1]))
        )

    def has(self, tensor_id: TensorId, bits: int) -> bool:
        return (tensor_id.key, int(bits)) in self._records

    def add(self, record: CandidateLoss) -> None:
        self._validate_record(record)
        key = (record.tensor_id.key, record.bits)
        incumbent = self._records.get(key)
        if incumbent is not None and incumbent != record:
            raise ValueError(f"attempt to rewrite sealed probe {key}")
        self._records[key] = record
        self.flush()

    def add_many(self, records: Iterable[CandidateLoss]) -> None:
        """Validate and publish one deterministic expert-sized record batch.

        Process workers return all probe widths for one expert together.  A
        single atomic ledger rewrite preserves crash consistency while avoiding
        thousands of progressively larger fsyncs that would otherwise leave the
        GPU workers idle.  Validation is completed for the whole batch before
        any in-memory record is changed.
        """

        ordered = tuple(
            sorted(records, key=lambda item: (item.tensor_id.key, int(item.bits)))
        )
        pending: list[tuple[tuple[str, int], CandidateLoss]] = []
        seen: set[tuple[str, int]] = set()
        for record in ordered:
            self._validate_record(record)
            key = (record.tensor_id.key, int(record.bits))
            if key in seen:
                raise ValueError(f"duplicate probe in batch {key}")
            seen.add(key)
            incumbent = self._records.get(key)
            if incumbent is not None and incumbent != record:
                raise ValueError(f"attempt to rewrite sealed probe {key}")
            pending.append((key, record))
        for key, record in pending:
            self._records[key] = record
        if pending:
            self.flush()

    def flush(self) -> None:
        records = []
        for record in self.records:
            item = asdict(record)
            item.pop("tensor_id")
            item.update(
                {
                    "layer": record.tensor_id.layer,
                    "expert": record.tensor_id.expert,
                    "projection": record.tensor_id.projection,
                }
            )
            records.append(item)
        atomic_write_json(
            self.path,
            {
                "schema": "r7-probe-v2",
                "layer": self.layer,
                "fixed_point_iteration": self.fixed_point_iteration,
                "bindings": self.bindings,
                "records": records,
            },
        )

    def digest(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "layer": self.layer,
                    "fixed_point_iteration": self.fixed_point_iteration,
                    "bindings": self.bindings,
                    "records": [asdict(record) for record in self.records],
                }
            )
        )

    def solve(
        self,
        mass_by_expert: Iterable[float | str],
        *,
        fixed_point_iteration: int,
    ) -> LayerAllocation:
        curves = build_curves(self.layer, self.records, mass_by_expert)
        return solve_exact_allocation(
            self.layer,
            curves,
            fixed_point_iteration=fixed_point_iteration,
            probe_sha256=self.digest(),
        )


def gate_up_map(allocation: LayerAllocation) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (key, bits)
            for key, bits in allocation.bits.items()
            if key.endswith("/gate_proj") or key.endswith("/up_proj")
        )
    )


class FixedPointController:
    def __init__(self, max_iterations: int = 4) -> None:
        if max_iterations < 2:
            raise ValueError("fixed point needs at least two iterations")
        self.max_iterations = max_iterations
        self.history: list[tuple[tuple[str, int], ...]] = []

    def observe(self, allocation: LayerAllocation) -> bool:
        current = tuple(
            sorted((key, int(bits)) for key, bits in allocation.bits.items())
        )
        converged = bool(self.history and current == self.history[-1])
        if current in self.history[:-1]:
            raise RuntimeError(
                "full allocation entered a cycle; final down H must not be emitted"
            )
        self.history.append(current)
        if not converged and len(self.history) >= self.max_iterations:
            raise RuntimeError(
                "gate/up allocation failed to converge; final down H must not be emitted"
            )
        return converged
