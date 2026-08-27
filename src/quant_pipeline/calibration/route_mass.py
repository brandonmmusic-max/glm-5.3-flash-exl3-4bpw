"""Exact routed-mass receipts and deterministic cold-expert top-up.

All p0/p1/p2 accounting is integer-rational.  Floating router weights must be
quantized by capture into ``weight_units / unit_denominator`` exactly once;
the fitter never reverse-engineers audit totals from rounded floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from ..core.artifacts import canonical_json, sha256_bytes


SCHEMA = "quant-pipeline.route-mass-audit.v2"
ROLES = ("gate_up", "down")
POWERS = (0, 1, 2)


def canonical_expert_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("expert ID cannot be boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("expert ID must be nonnegative")
        return str(value)
    if isinstance(value, str) and value and value.strip() == value:
        return str(int(value)) if value.isdecimal() else value
    raise ValueError("expert ID is not canonical")


def _sort_expert(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdecimal() else (1, value)


@dataclass(frozen=True)
class RouteMassRow:
    expert_id: int | str
    role: str
    document_id: str
    token_offset: int
    weight_units: int
    origin: str = "natural"
    inclusion_numerator: int = 1
    inclusion_denominator: int = 1

    @property
    def canonical_expert_id(self) -> str:
        return canonical_expert_id(self.expert_id)

    @property
    def row_identity(self) -> str:
        return sha256_bytes(canonical_json({
            "document_id": self.document_id,
            "token_offset": self.token_offset,
            "expert_id": self.canonical_expert_id,
            "role": self.role,
        }))


@dataclass(frozen=True)
class RouteMassAudit:
    metadata: dict[str, Any]

    @property
    def content_sha256(self) -> str:
        verify_route_mass_audit(self)
        return sha256_bytes(canonical_json(self.metadata))


def _validate_row(row: RouteMassRow, denominator: int) -> None:
    if not isinstance(row, RouteMassRow):
        raise TypeError("route rows must be RouteMassRow")
    canonical_expert_id(row.expert_id)
    if row.role not in ROLES:
        raise ValueError(f"route role must be one of {ROLES}")
    if not isinstance(row.document_id, str) or not row.document_id or row.document_id.strip() != row.document_id:
        raise ValueError("document_id must be a nonempty canonical string")
    if isinstance(row.token_offset, bool) or not isinstance(row.token_offset, int) or row.token_offset < 0:
        raise ValueError("token_offset must be a nonnegative integer")
    if isinstance(row.weight_units, bool) or not isinstance(row.weight_units, int) or not 0 < row.weight_units <= denominator:
        raise ValueError("weight_units must lie in [1, unit_denominator]")
    if row.origin not in {"natural", "supplemental"}:
        raise ValueError("origin must be natural or supplemental")
    for name, value in (("inclusion_numerator", row.inclusion_numerator), ("inclusion_denominator", row.inclusion_denominator)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if row.inclusion_numerator > row.inclusion_denominator:
        raise ValueError("inclusion probability cannot exceed one")
    if row.origin == "natural" and (row.inclusion_numerator, row.inclusion_denominator) != (1, 1):
        raise ValueError("natural rows must have inclusion probability one")


def _contribution(row: RouteMassRow, power: int, denominator: int, corrected: bool) -> Fraction:
    base = Fraction(row.weight_units, denominator) ** power
    if corrected and row.origin == "supplemental":
        base *= Fraction(row.inclusion_denominator, row.inclusion_numerator)
    return base


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _deterministic_rank(row: RouteMassRow, seed_sha256: str) -> tuple[str, str]:
    return (
        sha256_bytes(canonical_json({"seed_sha256": seed_sha256, "row_identity": row.row_identity})),
        row.row_identity,
    )


def build_route_mass_audit(
    *,
    natural_rows: Sequence[RouteMassRow],
    supplemental_pool: Sequence[RouteMassRow],
    expert_ids: Sequence[int | str],
    unit_denominator: int,
    cold_expert_min_weight_units: int,
    topup_seed_sha256: str,
    role_row_identity_sha256: Mapping[str, str],
) -> RouteMassAudit:
    """Select deterministic supplemental rows and seal exact accounting.

    Coldness is assessed from natural p1 integer units independently for each
    expert and role. Supplemental rows are ranked by a seed-bound row hash and
    selected until their *raw* p1 units fill the deficit.  The receipt exposes
    raw and inverse-inclusion-corrected accounting separately; combined means
    natural plus corrected supplemental, matching ``CalibrationFitter``.
    """

    if isinstance(unit_denominator, bool) or not isinstance(unit_denominator, int) or unit_denominator < 1:
        raise ValueError("unit_denominator must be a positive integer")
    if isinstance(cold_expert_min_weight_units, bool) or not isinstance(cold_expert_min_weight_units, int) or cold_expert_min_weight_units < 0:
        raise ValueError("cold-expert threshold must be a nonnegative integer")
    if not isinstance(topup_seed_sha256, str) or len(topup_seed_sha256) != 64 or any(char not in "0123456789abcdef" for char in topup_seed_sha256):
        raise ValueError("top-up seed must be a lowercase SHA256")
    if set(role_row_identity_sha256) != set(ROLES):
        raise ValueError("role row identities must cover gate_up and down")
    if any(not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in role_row_identity_sha256.values()):
        raise ValueError("role row identities must be SHA256 values")
    experts = tuple(sorted((canonical_expert_id(item) for item in expert_ids), key=_sort_expert))
    if not experts or len(set(experts)) != len(experts):
        raise ValueError("expert inventory must be nonempty and canonically unique")
    for row in (*natural_rows, *supplemental_pool):
        _validate_row(row, unit_denominator)
    if any(row.origin != "natural" for row in natural_rows) or any(row.origin != "supplemental" for row in supplemental_pool):
        raise ValueError("natural and supplemental pools are role-separated")
    if any(row.canonical_expert_id not in experts for row in (*natural_rows, *supplemental_pool)):
        raise ValueError("route row references an undeclared expert")
    identities = [row.row_identity for row in (*natural_rows, *supplemental_pool)]
    if len(set(identities)) != len(identities):
        raise ValueError("route row identities overlap across natural/supplemental roles")

    selected: list[RouteMassRow] = []
    topup_records = []
    for expert in experts:
        for role in ROLES:
            natural_units = sum(row.weight_units for row in natural_rows if row.canonical_expert_id == expert and row.role == role)
            deficit = max(0, cold_expert_min_weight_units - natural_units)
            candidates = sorted(
                (row for row in supplemental_pool if row.canonical_expert_id == expert and row.role == role),
                key=lambda row: _deterministic_rank(row, topup_seed_sha256),
            )
            chosen = []
            supplied = 0
            for row in candidates:
                if supplied >= deficit:
                    break
                chosen.append(row)
                supplied += row.weight_units
            if supplied < deficit:
                raise ValueError(f"supplemental pool cannot fill cold expert {expert} {role}")
            selected.extend(chosen)
            topup_records.append({
                "expert_id": expert, "role": role, "natural_weight_units": natural_units,
                "deficit_weight_units": deficit, "selected_raw_weight_units": supplied,
                "selected_row_identities": [row.row_identity for row in chosen],
                "selected_rows": [
                    {
                        "row_identity": row.row_identity,
                        "expert_id": row.canonical_expert_id,
                        "role": row.role,
                        "document_id": row.document_id,
                        "token_offset": row.token_offset,
                        "origin": row.origin,
                        "weight_units": row.weight_units,
                        "inclusion_numerator": row.inclusion_numerator,
                        "inclusion_denominator": row.inclusion_denominator,
                    }
                    for row in chosen
                ],
                "selected_rows_sha256": sha256_bytes(canonical_json([
                    {
                        "row_identity": row.row_identity,
                        "expert_id": row.canonical_expert_id,
                        "role": row.role,
                        "document_id": row.document_id,
                        "token_offset": row.token_offset,
                        "origin": row.origin,
                        "weight_units": row.weight_units,
                        "inclusion_numerator": row.inclusion_numerator,
                        "inclusion_denominator": row.inclusion_denominator,
                    }
                    for row in chosen
                ])),
            })

    accounting = []
    for expert in experts:
        for role in ROLES:
            natural = [row for row in natural_rows if row.canonical_expert_id == expert and row.role == role]
            supplemental = [row for row in selected if row.canonical_expert_id == expert and row.role == role]
            powers = {}
            for power in POWERS:
                natural_total = sum((_contribution(row, power, unit_denominator, False) for row in natural), Fraction())
                raw_total = sum((_contribution(row, power, unit_denominator, False) for row in supplemental), Fraction())
                corrected_total = sum((_contribution(row, power, unit_denominator, True) for row in supplemental), Fraction())
                powers[str(power)] = {
                    "natural": _fraction_record(natural_total),
                    "supplemental_raw": _fraction_record(raw_total),
                    "supplemental_corrected": _fraction_record(corrected_total),
                    "combined": _fraction_record(natural_total + corrected_total),
                }
            accounting.append({
                "expert_id": expert, "role": role,
                "natural_row_identities": sorted(row.row_identity for row in natural),
                "supplemental_row_identities": sorted(row.row_identity for row in supplemental),
                "powers": powers,
            })
    metadata = {
        "schema": SCHEMA,
        "identity": {
            "expert_ids": list(experts), "roles": list(ROLES),
            "role_row_identity_sha256": dict(sorted(role_row_identity_sha256.items())),
            "unit_denominator": unit_denominator,
        },
        "policy": {
            "powers": list(POWERS),
            "topup": "seeded-row-hash-until-natural-p1-integer-unit-deficit-filled",
            "supplemental_correction": "exact-inverse-inclusion-probability",
            "combined": "natural-plus-supplemental-corrected",
            "cold_expert_min_weight_units": cold_expert_min_weight_units,
            "topup_seed_sha256": topup_seed_sha256,
        },
        "topup": topup_records,
        "accounting": accounting,
    }
    result = RouteMassAudit(metadata)
    verify_route_mass_audit(result)
    return result


def verify_route_mass_audit(value: RouteMassAudit) -> None:
    if not isinstance(value, RouteMassAudit):
        raise TypeError("value must be RouteMassAudit")
    metadata = value.metadata
    if not isinstance(metadata, dict) or set(metadata) != {"schema", "identity", "policy", "topup", "accounting"} or metadata["schema"] != SCHEMA:
        raise ValueError("route-mass audit schema is malformed")
    identity = metadata["identity"]
    if not isinstance(identity, dict) or set(identity) != {"expert_ids", "roles", "role_row_identity_sha256", "unit_denominator"} or identity["roles"] != list(ROLES):
        raise ValueError("route-mass identity is malformed")
    experts = identity["expert_ids"]
    if experts != sorted({canonical_expert_id(item) for item in experts}, key=_sort_expert):
        raise ValueError("route-mass expert inventory is not canonical")
    denominator = identity["unit_denominator"]
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator < 1:
        raise ValueError("route-mass unit denominator is invalid")
    if set(identity["role_row_identity_sha256"]) != set(ROLES) or any(
        not isinstance(item, str) or len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
        for item in identity["role_row_identity_sha256"].values()
    ):
        raise ValueError("route-mass role row identities are invalid")
    policy = metadata["policy"]
    required_policy = {
        "powers", "topup", "supplemental_correction", "combined",
        "cold_expert_min_weight_units", "topup_seed_sha256",
    }
    if (
        not isinstance(policy, dict) or set(policy) != required_policy
        or policy["powers"] != list(POWERS)
        or policy["topup"] != "seeded-row-hash-until-natural-p1-integer-unit-deficit-filled"
        or policy["supplemental_correction"] != "exact-inverse-inclusion-probability"
        or policy["combined"] != "natural-plus-supplemental-corrected"
        or isinstance(policy["cold_expert_min_weight_units"], bool)
        or not isinstance(policy["cold_expert_min_weight_units"], int)
        or policy["cold_expert_min_weight_units"] < 0
        or not isinstance(policy["topup_seed_sha256"], str)
        or len(policy["topup_seed_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in policy["topup_seed_sha256"])
    ):
        raise ValueError("route-mass policy is malformed")
    expected = {(expert, role) for expert in experts for role in ROLES}
    topup = metadata["topup"]
    if not isinstance(topup, list) or {
        (item.get("expert_id"), item.get("role")) for item in topup if isinstance(item, dict)
    } != expected or len(topup) != len(expected):
        raise ValueError("route-mass top-up inventory is incomplete")
    topup_rows: dict[tuple[str, str], set[str]] = {}
    topup_units: dict[tuple[str, str], tuple[int, int]] = {}
    topup_selected_records: dict[tuple[str, str], list[dict[str, int | str]]] = {}
    for record in topup:
        if set(record) != {
            "expert_id", "role", "natural_weight_units", "deficit_weight_units",
            "selected_raw_weight_units", "selected_row_identities", "selected_rows",
            "selected_rows_sha256",
        }:
            raise ValueError("route-mass top-up record is malformed")
        for field in ("natural_weight_units", "deficit_weight_units", "selected_raw_weight_units"):
            if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
                raise ValueError("route-mass top-up integer units are invalid")
        expected_deficit = max(
            0, policy["cold_expert_min_weight_units"] - record["natural_weight_units"]
        )
        if record["deficit_weight_units"] != expected_deficit or record["selected_raw_weight_units"] < expected_deficit:
            raise ValueError("route-mass top-up deficit does not reconcile")
        row_identity_list = record["selected_row_identities"]
        if not isinstance(row_identity_list, list) or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for item in row_identity_list
        ):
            raise ValueError("route-mass top-up row identities are invalid")
        rows = set(row_identity_list)
        if len(rows) != len(record["selected_row_identities"]):
            raise ValueError("duplicate route-mass top-up row")
        selected_rows = record["selected_rows"]
        if not isinstance(selected_rows, list) or len(selected_rows) != len(rows):
            raise ValueError("route-mass selected supplemental row records are incomplete")
        selected_rows_sha256 = record["selected_rows_sha256"]
        if (
            not isinstance(selected_rows_sha256, str)
            or len(selected_rows_sha256) != 64
            or any(char not in "0123456789abcdef" for char in selected_rows_sha256)
            or selected_rows_sha256 != sha256_bytes(canonical_json(selected_rows))
        ):
            raise ValueError("route-mass selected supplemental row payload seal mismatch")
        selected_identities: list[str] = []
        reconstructed_rows: list[RouteMassRow] = []
        selected_units = 0
        for selected in selected_rows:
            if not isinstance(selected, dict) or set(selected) != {
                "row_identity", "expert_id", "role", "document_id", "token_offset", "origin",
                "weight_units", "inclusion_numerator", "inclusion_denominator",
            }:
                raise ValueError("route-mass selected supplemental row record is malformed")
            row_identity = selected["row_identity"]
            if (
                not isinstance(row_identity, str)
                or len(row_identity) != 64
                or any(char not in "0123456789abcdef" for char in row_identity)
            ):
                raise ValueError("route-mass selected supplemental row identity is invalid")
            for field in ("weight_units", "inclusion_numerator", "inclusion_denominator"):
                item = selected[field]
                if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                    raise ValueError("route-mass selected supplemental row units are invalid")
            if selected["weight_units"] > denominator:
                raise ValueError("route-mass selected supplemental weight exceeds denominator")
            if selected["inclusion_numerator"] > selected["inclusion_denominator"]:
                raise ValueError("route-mass selected supplemental inclusion probability exceeds one")
            if (
                selected["expert_id"] != record["expert_id"]
                or selected["role"] != record["role"]
                or selected["origin"] != "supplemental"
            ):
                raise ValueError("route-mass selected supplemental row role/origin binding mismatch")
            reconstructed = RouteMassRow(
                expert_id=selected["expert_id"],
                role=selected["role"],
                document_id=selected["document_id"],
                token_offset=selected["token_offset"],
                weight_units=selected["weight_units"],
                origin=selected["origin"],
                inclusion_numerator=selected["inclusion_numerator"],
                inclusion_denominator=selected["inclusion_denominator"],
            )
            _validate_row(reconstructed, denominator)
            if reconstructed.row_identity != row_identity:
                raise ValueError("route-mass selected supplemental row identity/payload mismatch")
            reconstructed_rows.append(reconstructed)
            selected_identities.append(row_identity)
            selected_units += selected["weight_units"]
        if selected_identities != record["selected_row_identities"] or set(selected_identities) != rows:
            raise ValueError("route-mass selected supplemental row records differ from top-up identities")
        if selected_units != record["selected_raw_weight_units"]:
            raise ValueError("route-mass selected supplemental row units do not reconcile")
        if reconstructed_rows != sorted(
            reconstructed_rows,
            key=lambda row: _deterministic_rank(row, policy["topup_seed_sha256"]),
        ):
            raise ValueError("route-mass selected supplemental rows are not in deterministic rank order")
        deficit = record["deficit_weight_units"]
        if deficit == 0 and reconstructed_rows:
            raise ValueError("route-mass no-op top-up selected supplemental rows")
        if deficit > 0 and (
            not reconstructed_rows
            or selected_units - reconstructed_rows[-1].weight_units >= deficit
        ):
            raise ValueError("route-mass top-up did not stop at the first deficit-filling row")
        topup_rows[(record["expert_id"], record["role"])] = rows
        topup_units[(record["expert_id"], record["role"])] = (
            record["natural_weight_units"], record["selected_raw_weight_units"]
        )
        topup_selected_records[(record["expert_id"], record["role"])] = selected_rows
    accounting = metadata["accounting"]
    if {(item.get("expert_id"), item.get("role")) for item in accounting if isinstance(item, dict)} != expected or len(accounting) != len(expected):
        raise ValueError("route-mass accounting inventory is incomplete")
    natural_rows: set[str] = set()
    supplemental_rows: set[str] = set()
    for record in accounting:
        if set(record) != {"expert_id", "role", "natural_row_identities", "supplemental_row_identities", "powers"} or set(record["powers"]) != {str(power) for power in POWERS}:
            raise ValueError("route-mass accounting record malformed")
        parsed_powers: dict[str, dict[str, Fraction]] = {}
        for power, power_record in record["powers"].items():
            if set(power_record) != {"natural", "supplemental_raw", "supplemental_corrected", "combined"}:
                raise ValueError("route-mass power accounting malformed")
            parsed = {}
            for key, fraction in power_record.items():
                if not isinstance(fraction, dict) or set(fraction) != {"numerator", "denominator"}:
                    raise ValueError("route-mass fraction malformed")
                numerator = fraction["numerator"]
                fraction_denominator = fraction["denominator"]
                if (
                    isinstance(numerator, bool)
                    or not isinstance(numerator, int)
                    or numerator < 0
                    or isinstance(fraction_denominator, bool)
                    or not isinstance(fraction_denominator, int)
                    or fraction_denominator < 1
                ):
                    raise ValueError("route-mass fraction terms are invalid")
                parsed[key] = Fraction(numerator, fraction_denominator)
                if fraction != _fraction_record(parsed[key]):
                    raise ValueError("route-mass fraction is not canonically reduced")
            if parsed["combined"] != parsed["natural"] + parsed["supplemental_corrected"]:
                raise ValueError("route-mass combined accounting does not reconcile")
            parsed_powers[power] = parsed
        natural_units, selected_units = topup_units[(record["expert_id"], record["role"])]
        if parsed_powers["1"]["natural"] != Fraction(natural_units, denominator):
            raise ValueError("route-mass p1 natural accounting differs from top-up integer units")
        if parsed_powers["1"]["supplemental_raw"] != Fraction(selected_units, denominator):
            raise ValueError("route-mass p1 supplemental-raw accounting differs from selected row units")
        for power in POWERS:
            expected_raw = sum(
                (Fraction(int(row["weight_units"]), denominator) ** power
                 for row in topup_selected_records[(record["expert_id"], record["role"])]),
                Fraction(),
            )
            expected_corrected = sum(
                (
                    Fraction(int(row["weight_units"]), denominator) ** power
                    * Fraction(int(row["inclusion_denominator"]), int(row["inclusion_numerator"]))
                    for row in topup_selected_records[(record["expert_id"], record["role"])]
                ),
                Fraction(),
            )
            if parsed_powers[str(power)]["supplemental_raw"] != expected_raw:
                raise ValueError("route-mass supplemental-raw accounting differs from selected row records")
            if parsed_powers[str(power)]["supplemental_corrected"] != expected_corrected:
                raise ValueError("route-mass supplemental-corrected accounting differs from selected row records")
        for field in ("natural_row_identities", "supplemental_row_identities"):
            identities = record[field]
            if not isinstance(identities, list) or any(
                not isinstance(item, str)
                or len(item) != 64
                or any(char not in "0123456789abcdef" for char in item)
                for item in identities
            ):
                raise ValueError("route-mass accounting row identities are invalid")
        current_natural = set(record["natural_row_identities"])
        current_supplemental = set(record["supplemental_row_identities"])
        if parsed_powers["0"]["natural"] != Fraction(len(current_natural), 1):
            raise ValueError("route-mass p0 natural accounting differs from natural row identities")
        if current_supplemental != topup_rows[(record["expert_id"], record["role"])]:
            raise ValueError("top-up and supplemental accounting row identities differ")
        if len(current_natural) != len(record["natural_row_identities"]) or len(current_supplemental) != len(record["supplemental_row_identities"]):
            raise ValueError("duplicate route row identity")
        if natural_rows & current_natural or supplemental_rows & current_supplemental:
            raise ValueError("route row identity is assigned twice")
        natural_rows |= current_natural
        supplemental_rows |= current_supplemental
    if natural_rows & supplemental_rows:
        raise ValueError("natural and supplemental route identities overlap")
    canonical_json(metadata)
