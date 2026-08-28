"""Strict model-runtime boundary for the generic Round 7 walk.

An owner-run backend is responsible for model-specific attention, routing, and
layer forwarding. The encoder owns all rate allocation, covariances, TRELLIS
encoding, schema, and audits. This separation prevents a draft from silently
reimplementing a drifting GLM router or attention mask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .constants import HIDDEN_SIZE, INTERMEDIATE_SIZE
from .determinism import canonical_json_bytes, sha256_bytes
from .routing import MassAudit
from .types import RoutedBatch, StateShard


@dataclass(frozen=True)
class CalibrationBatch:
    shard_id: str
    hidden: Any
    row_ids: Any
    attention_metadata: Any
    token_count: int


@dataclass(frozen=True)
class ExpertRows:
    hidden: Any
    router_weight: Any
    row_ids: Any


@dataclass(frozen=True)
class ExpertWeights:
    gate_hf: Any
    up_hf: Any
    down_hf: Any
    dtype: str
    source_names: Mapping[str, str]
    payload_sha256: Mapping[str, str]
    source_records: Mapping[str, Mapping[str, object]]

    def validate_bf16(self, layer: int, expert: int) -> None:
        if self.dtype.lower() not in ("bfloat16", "bf16"):
            raise ValueError(
                f"L{layer} E{expert}: source dtype {self.dtype!r}; quant-of-quant forbidden"
            )
        expected = {
            "gate_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "up_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "down_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        }
        for projection, tensor in (
            ("gate_proj", self.gate_hf),
            ("up_proj", self.up_hf),
            ("down_proj", self.down_hf),
        ):
            shape = tuple(tensor.shape)
            if shape != expected[projection]:
                raise ValueError(
                    f"L{layer} E{expert} {projection}: {shape} != {expected[projection]}"
                )
            if not self.payload_sha256.get(projection):
                raise ValueError(f"missing BF16 payload hash for {projection}")
            record = self.source_records.get(projection)
            if (
                not isinstance(record, Mapping)
                or record.get("payload_sha256") != self.payload_sha256[projection]
                or record.get("dtype") != "BF16"
                or not record.get("shard")
                or type(record.get("payload_start")) is not int
                or type(record.get("payload_end")) is not int
            ):
                raise ValueError(f"missing sealed BF16 source range for {projection}")


@dataclass(frozen=True)
class LayerCapture:
    layer: int
    state_sha256: str
    routing_dir: Path
    mass_audit: MassAudit
    routing_sha256: Mapping[str, str]

    @property
    def digest(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "layer": self.layer,
                    "state_sha256": self.state_sha256,
                    "routing_sha256": dict(sorted(self.routing_sha256.items())),
                    "mass_audit": self.mass_audit.to_json(),
                }
            )
        )


class Round7Backend(ABC):
    """Model-specific backend loaded only by an explicit owner-run CLI."""

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Hash/version covering model code, router, and layer-forward logic."""

    @abstractmethod
    def prepare_corpus_plan(self, *, corpus: Path) -> Mapping[str, object]:
        """Seal and return the ordered domain plus its artifact file hash."""

    @abstractmethod
    def initialize_carried_state(
        self,
        *,
        carrier: Path,
        corpus: Path,
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        """Forward corpus through carried reconstructed layers 0..2.

        Return `(shard_id, hidden_file, metadata_file, tokens, hidden_size)`.
        Files must live under `output_partial`; metadata must preserve sequence
        boundaries, positions, and masks.
        """

    @abstractmethod
    def iter_state(self, shards: Iterable[StateShard]) -> Iterator[CalibrationBatch]:
        """Yield canonically ordered prompt-preserving state batches."""

    @abstractmethod
    def route(self, layer: int, batch: CalibrationBatch) -> RoutedBatch:
        """Return exact runtime top-k IDs and normalized/scaled weights."""

    @abstractmethod
    def capture_layer(
        self,
        *,
        layer: int,
        shards: Iterable[StateShard],
        routing_dir: Path,
    ) -> LayerCapture:
        """Persist routing sidecars and exact mass audit for one state."""

    @abstractmethod
    def open_capture(self, *, layer: int, routing_dir: Path) -> LayerCapture:
        """Open and fully verify a previously sealed capture without rerouting."""

    @abstractmethod
    def iter_expert_rows(
        self,
        *,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        split: str,
    ) -> Iterator[ExpertRows]:
        """Yield deterministic `fit` or `holdout` rows routed to one expert."""

    @abstractmethod
    def iter_cold_fallback_rows(
        self,
        *,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        split: str,
    ) -> Iterator[ExpertRows]:
        """Yield seeded, mass-stratified all-token rows for a cold expert.

        The layer processor still forwards these rows through that expert's
        reconstructed gate/up before constructing a down covariance.
        """

    @abstractmethod
    def load_bf16_expert(self, *, layer: int, expert: int) -> ExpertWeights:
        """Stream one expert from the owner-supplied BF16 source."""

    @abstractmethod
    def install_encoded_expert(
        self, *, layer: int, expert: int, encoded: Mapping[str, Any]
    ) -> Mapping[str, object]:
        """Install from sealed packed/vector records and return a payload receipt.

        The returned record is publication evidence, not optional diagnostics.
        It binds the installed reconstruction to the supplied packed payload.
        """

    @abstractmethod
    def audit_installed_layer(self, *, layer: int) -> Mapping[str, object]:
        """Audit all installed experts through the official aggregate module.

        This is deliberately separate from per-expert installation: the audit
        must exercise the independently inventoried Transformers dispatch path.
        """

    @abstractmethod
    def restore_encoded_layer(self, *, layer: int, manifest: Path) -> None:
        """Reload and reconstruct a sealed schema-v2 layer after preemption."""

    @abstractmethod
    def forward_installed_layer(
        self,
        *,
        layer: int,
        input_shards: Iterable[StateShard],
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        """Forward exactly one installed layer, preserving prompt metadata."""
