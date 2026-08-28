"""Exact routed-mass accounting and immutable capture audit records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from .constants import NUM_EXPERTS, TOP_K
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

# Every finite nonnegative float32 is an integer multiple of 2^-149. Summing
# those integer units makes routed mass independent of backend batch boundaries.
MASS_UNIT_POWER = -149


def _float32_bits_to_units(bits: int) -> int:
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if sign:
        raise ValueError("router weights must be nonnegative")
    if exponent == 0xFF:
        raise ValueError("router weights must be finite")
    if exponent == 0:
        return fraction
    return ((1 << 23) | fraction) << (exponent - 1)


def _units_to_decimal(units: int) -> str:
    with localcontext() as context:
        context.prec = 200
        value = Decimal(units) * (Decimal(2) ** MASS_UNIT_POWER)
        # Fixed-point spelling is canonical and can be parsed losslessly by the
        # allocation Decimal path.
        return format(value, "f")


@dataclass(frozen=True)
class MassAudit:
    layer: int
    tokens: int
    assignments: int
    expected_total_mass: str
    observed_total_mass: str
    mass_by_expert: tuple[str, ...]
    mass_units_by_expert: tuple[str, ...]
    count_by_expert: tuple[int, ...]
    max_row_mass_error: str
    route_weight_dtype: str = "float32"
    mass_unit_power: int = MASS_UNIT_POWER

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "MassAudit":
        value = cls(
            layer=int(raw["layer"]),
            tokens=int(raw["tokens"]),
            assignments=int(raw["assignments"]),
            expected_total_mass=str(raw["expected_total_mass"]),
            observed_total_mass=str(raw["observed_total_mass"]),
            mass_by_expert=tuple(str(item) for item in raw["mass_by_expert"]),  # type: ignore[union-attr]
            mass_units_by_expert=tuple(
                str(item)
                for item in raw["mass_units_by_expert"]  # type: ignore[union-attr]
            ),
            count_by_expert=tuple(int(item) for item in raw["count_by_expert"]),  # type: ignore[union-attr]
            max_row_mass_error=str(raw["max_row_mass_error"]),
            route_weight_dtype=str(raw.get("route_weight_dtype", "")),
            mass_unit_power=int(raw.get("mass_unit_power", 0)),
        )
        if raw.get("digest") != value.digest:
            raise ValueError("routed-mass audit digest mismatch")
        if (
            value.route_weight_dtype != "float32"
            or value.mass_unit_power != MASS_UNIT_POWER
            or len(value.mass_by_expert) != NUM_EXPERTS
            or len(value.mass_units_by_expert) != NUM_EXPERTS
            or len(value.count_by_expert) != NUM_EXPERTS
        ):
            raise ValueError("routed-mass audit schema drift")
        return value

    @property
    def digest(self) -> str:
        return self.digest_without_self()

    def to_json(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "tokens": self.tokens,
            "assignments": self.assignments,
            "expected_total_mass": self.expected_total_mass,
            "observed_total_mass": self.observed_total_mass,
            "mass_by_expert": list(self.mass_by_expert),
            "mass_units_by_expert": list(self.mass_units_by_expert),
            "mass_unit_power": self.mass_unit_power,
            "route_weight_dtype": self.route_weight_dtype,
            "count_by_expert": list(self.count_by_expert),
            "max_row_mass_error": self.max_row_mass_error,
            "digest": self.digest_without_self(),
        }

    def digest_without_self(self) -> str:
        payload = {
            "layer": self.layer,
            "tokens": self.tokens,
            "assignments": self.assignments,
            "expected_total_mass": self.expected_total_mass,
            "observed_total_mass": self.observed_total_mass,
            "mass_by_expert": list(self.mass_by_expert),
            "mass_units_by_expert": list(self.mass_units_by_expert),
            "mass_unit_power": self.mass_unit_power,
            "route_weight_dtype": self.route_weight_dtype,
            "count_by_expert": list(self.count_by_expert),
            "max_row_mass_error": self.max_row_mass_error,
        }
        return sha256_bytes(canonical_json_bytes(payload))


class RoutedMassAccumulator:
    """Accumulate exact float32 routed mass, never expert count as a proxy."""

    def __init__(self, layer: int, row_tolerance: float = 2e-5) -> None:
        self.layer = layer
        self.row_tolerance = row_tolerance
        self.tokens = 0
        self.assignments = 0
        self.expected_mass_per_token: float | None = None
        self.mass_units = [0] * NUM_EXPERTS
        self.counts = [0] * NUM_EXPERTS
        self.max_row_mass_error = 0.0

    def add(
        self,
        expert_ids: Any,
        expert_weights: Any,
        expected_mass_per_token: float,
    ) -> None:
        import torch

        ids = (
            torch.as_tensor(expert_ids)
            .detach()
            .to("cpu", dtype=torch.int64)
            .contiguous()
        )
        source_weights = torch.as_tensor(expert_weights).detach().to("cpu").contiguous()
        if source_weights.dtype != torch.float32:
            raise ValueError(
                f"routing weights must be float32 for exact mass accounting, got {source_weights.dtype}"
            )
        weights = source_weights
        if ids.ndim != 2 or ids.shape[1] != TOP_K or ids.shape != weights.shape:
            raise ValueError(
                f"routing must be [tokens,{TOP_K}] IDs+weights; got "
                f"ids={tuple(ids.shape)} weights={tuple(weights.shape)}"
            )
        if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= NUM_EXPERTS):
            raise ValueError("routing contains out-of-range expert ID")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("routing weights must be finite and nonnegative")
        if not math.isfinite(expected_mass_per_token) or expected_mass_per_token <= 0:
            raise ValueError(
                "expected routed mass per token must be finite and positive"
            )
        if self.expected_mass_per_token is None:
            self.expected_mass_per_token = float(expected_mass_per_token)
        elif float(expected_mass_per_token) != self.expected_mass_per_token:
            raise ValueError("expected routed mass per token changed between batches")
        if ids.numel():
            sorted_ids = ids.sort(dim=1).values
            if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
                raise ValueError(
                    "routing contains a duplicate expert ID within one top-k row"
                )

        row_error = (
            weights.to(torch.float64).sum(dim=1) - expected_mass_per_token
        ).abs()
        maximum = float(row_error.max().item()) if row_error.numel() else 0.0
        self.max_row_mass_error = max(self.max_row_mass_error, maximum)
        if maximum > self.row_tolerance:
            raise ValueError(
                f"router mass normalization drift at layer {self.layer}: "
                f"max row error {maximum:.9g} > {self.row_tolerance:.9g}"
            )

        # Preserve original token/slot order. Integer addition is exact and
        # therefore invariant to how callers partition these rows into batches.
        bits = weights.view(torch.int32).tolist()
        id_rows = ids.tolist()
        for id_row, bit_row in zip(id_rows, bits):
            for expert, raw_bits in zip(id_row, bit_row):
                unsigned = int(raw_bits) & 0xFFFFFFFF
                self.mass_units[expert] += _float32_bits_to_units(unsigned)
                self.counts[expert] += 1
        rows = int(ids.shape[0])
        self.tokens += rows
        self.assignments += int(ids.numel())

    def finish(self) -> MassAudit:
        if self.tokens <= 0:
            raise ValueError(f"layer {self.layer} has no routed tokens")
        if self.assignments != self.tokens * TOP_K:
            raise AssertionError("routing assignment accounting drift")
        if self.expected_mass_per_token is None:
            raise AssertionError("routing expected-mass contract was not initialized")
        # One multiplication at finalization makes this spelling invariant to
        # how the same token rows were partitioned into runtime batches.
        expected_total_mass_float = self.tokens * self.expected_mass_per_token
        observed_units = sum(self.mass_units)
        observed_float = math.ldexp(float(observed_units), MASS_UNIT_POWER)
        tolerance = max(
            1e-7 * expected_total_mass_float,
            self.row_tolerance * self.tokens,
        )
        if abs(observed_float - expected_total_mass_float) > tolerance:
            raise ValueError(
                f"layer {self.layer} total routed mass mismatch: "
                f"observed={observed_float:.17g} "
                f"expected={expected_total_mass_float:.17g}"
            )
        return MassAudit(
            layer=self.layer,
            tokens=self.tokens,
            assignments=self.assignments,
            expected_total_mass=format(expected_total_mass_float, ".17g"),
            observed_total_mass=_units_to_decimal(observed_units),
            mass_by_expert=tuple(_units_to_decimal(value) for value in self.mass_units),
            mass_units_by_expert=tuple(str(value) for value in self.mass_units),
            count_by_expert=tuple(self.counts),
            max_row_mass_error=format(self.max_row_mass_error, ".17g"),
        )


def write_mass_audit(path: str | Path, audit: MassAudit) -> str:
    atomic_write_json(path, audit.to_json())
    return sha256_file(path)
