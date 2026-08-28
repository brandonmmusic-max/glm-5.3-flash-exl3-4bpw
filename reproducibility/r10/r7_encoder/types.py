"""Small immutable records shared by the walk, codec, and schema code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import ALLOWED_BITS, TensorId


@dataclass(frozen=True)
class StateShard:
    shard_id: str
    hidden_path: Path
    metadata_path: Path
    tokens: int
    hidden_size: int
    sha256_hidden: str
    sha256_metadata: str


@dataclass(frozen=True)
class RoutedBatch:
    """Exact routing result; tensors are backend-owned array-like objects."""

    expert_ids: Any
    expert_weights: Any
    expected_mass_per_token: float


@dataclass(frozen=True)
class EncodedTensor:
    tensor_id: TensorId
    bits: int
    trellis: Any
    suh: Any
    svh: Any
    reconstructed_kn: Any
    proxy_loss: float
    packed_sha256: str
    reconstruction_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bits not in ALLOWED_BITS:
            raise ValueError(f"bits={self.bits} outside {ALLOWED_BITS}")


@dataclass(frozen=True)
class CandidateLoss:
    tensor_id: TensorId
    bits: int
    loss: str
    mass: str
    fit_rows: int
    holdout_rows: int
    roundtrip_sha256: str
    gate_up_roundtrip_sha256: str | None = None
    fixed_point_iteration: int = 0
    context_bits_sha256: str = ""
    expert_roundtrip_sha256: Mapping[str, str] = field(default_factory=dict)
    state_sha256: str = ""
    capture_sha256: str = ""
    search_sha256: str = ""
    source_inventory_sha256: str = ""
    numeric_environment_sha256: str = ""
    runtime_inventory_sha256: str = ""
    backend_fingerprint: str = ""
    fit_row_ids_sha256: str = ""
    holdout_row_ids_sha256: str = ""
    permutation_sha256: str = ""
    vector_bundle_sha256: str = ""
    used_cold_fallback: bool = False


@dataclass(frozen=True)
class LayerAllocation:
    layer: int
    bits: Mapping[str, int]
    masses: Sequence[str]
    score_integer: int
    score_scale: int
    upgrade_units: int
    fixed_point_iteration: int
    probe_sha256: str
