from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class CodecCandidate:
    bits: int
    packed: Any
    reconstructed: Any
    stored_bytes: int
    packed_sha256: str
    reconstruction_sha256: str
    metadata: dict


class CodecAdapter(Protocol):
    name: str

    def encode_candidates(
        self,
        *,
        unit_id: str,
        weight_hf: Any,
        covariance: Any,
        bits: Sequence[int],
        input_vector: Any,
        output_vector: Any,
        provenance: dict | None = None,
    ) -> dict[int, CodecCandidate]: ...

