"""Runtime + capture binding for the layer-parallel (flat capture) encode path.

``fast_layer.encode_one_layer`` builds a :class:`~r7_encoder.glm52_backend.GLM52Backend`
whose calibration source is one finalized ``flat_capture`` trio instead of the
sealed per-prompt state / routing-sidecar chain.  That backend still demands a
fingerprinted ``GLM52Runtime``, so this module supplies the smallest object that
satisfies the protocol: it verifies the sealed runtime-code inventory (which is
what the fingerprint contract is actually about) and refuses -- loudly -- every
method that would require a live GLM-5.2 forward pass.

It also owns the two flat-capture helpers that ``GLM52Backend.open_flat_capture``
calls:

``build_flat_mass_audit``
    Reproduces :class:`~r7_encoder.routing.RoutedMassAccumulator` arithmetic
    exactly -- float32 router weights accumulated as integer multiples of
    ``2**-149`` so the per-expert sums are batch-invariant -- but vectorized,
    and cross-checked against the accumulator itself on a deterministic row
    prefix so the two implementations can never silently diverge.

``build_flat_bound_batches``
    Materializes mmap-backed row views in the exact 5-tuple shape
    ``GLM52Backend._iter_bound_batches`` yields, so ``iter_expert_rows`` and
    ``iter_cold_fallback_rows`` run against the flat capture with their bodies
    unchanged: same fit/holdout split, same ``ExpertRows`` fields, same
    ordering, same cold-fallback reservoir.

Importing this module is inert: no torch, no numpy, no CUDA at module scope,
matching the rest of ``r7_encoder``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .backend import CalibrationBatch
from .constants import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TOP_K,
)
from .determinism import canonical_json_bytes, sha256_bytes, sha256_file
from .flat_capture import SCHEMA as FLAT_CAPTURE_SCHEMA
from .flat_capture import X_FILE
from .glm52_backend import GLM52Runtime
from .routing import MASS_UNIT_POWER, MassAudit, RoutedMassAccumulator
from .routing import _units_to_decimal as units_to_decimal
from .types import StateShard

RUNTIME_SCHEMA = "r7-flat-runtime-v1"

# One pseudo state shard per this many capture rows. The flat capture is a
# single flat file, but the row-iteration contract is shard-shaped, so the rows
# are presented in fixed, order-stable blocks. Fixed size == deterministic
# batch boundaries == deterministic holdout/fallback selection.
FLAT_BATCH_TOKENS = 1 << 18
FLAT_SHARD_PREFIX = "flat-"

# Row block used while folding router weights into integer mass units. Integer
# addition is exact, so this is a memory knob only -- never a numeric one.
MASS_CHUNK_ROWS = 1 << 16

# Deterministic prefix on which the vectorized mass folding is proved equal to
# RoutedMassAccumulator itself.
MASS_CROSS_CHECK_ROWS = 4096

# RoutedMassAccumulator's own default; kept in sync deliberately.
ROW_MASS_TOLERANCE = 2e-5

_UNSUPPORTED = (
    "flat runtime does not execute the model: {method} needs a live GLM-5.2 "
    "forward pass. The layer-parallel path calibrates from a sealed flat "
    "capture (phase A) and must not re-enter the sequential walk. Use "
    "r7_encoder.transformers_runtime:factory for that work."
)


# ---------------------------------------------------------------------------
# exact routed mass, folded straight out of the flat capture
# ---------------------------------------------------------------------------


def _hex64(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"flat capture {label} is not a sha256 digest: {value!r}")
    return text


def _exact_mass_units(
    reader: Any,
    *,
    first_row: int = 0,
    last_row: int | None = None,
    chunk: int = MASS_CHUNK_ROWS,
) -> tuple[list[int], list[int]]:
    """Fold router weights into exact integer units of ``2**MASS_UNIT_POWER``.

    Identical arithmetic to ``RoutedMassAccumulator.add``: every finite
    nonnegative float32 ``w`` equals ``mantissa * 2**-149`` for the integer
    ``mantissa = fraction`` (subnormal) or ``(2**23 | fraction) << (exponent-1)``
    (normal), and those integers are summed exactly.  The scalar loop is
    replaced by a per-``(expert, shift)`` histogram and one final shift, which
    is the same sum re-associated -- integer addition is associative, so the
    result is bit-identical and still independent of batch boundaries.

    Returns ``(mass_units_by_expert, count_by_expert)``.
    """

    import numpy as np
    import torch

    total_rows = int(reader.tokens)
    stop = total_rows if last_row is None else int(last_row)
    start_row = int(first_row)
    if not 0 <= start_row <= stop <= total_rows:
        raise ValueError("flat mass row window outside the capture")
    block = int(chunk)
    if block <= 0:
        raise ValueError("mass chunk must be positive")
    num_experts = int(reader.num_experts)
    top_k = int(reader.top_k)
    # Each per-chunk histogram bucket is a float64 sum of integers below 2**24;
    # keep the worst case well inside the exact-integer range of float64.
    if block * top_k * (1 << 24) >= (1 << 53):
        raise ValueError("mass chunk too large for exact float64 histogram folding")

    bins = num_experts * 256
    shifted = np.zeros(bins, dtype=np.int64)
    counts = np.zeros(num_experts, dtype=np.int64)
    for begin in range(start_row, stop, block):
        end = min(begin + block, stop)
        weights = reader.weights[begin:end]
        ids = reader.ids[begin:end]
        bits = weights.contiguous().view(torch.int32).numpy().view(np.uint32)
        if int((bits >> np.uint32(31)).max(initial=0)) != 0:
            raise ValueError("router weights must be nonnegative")
        exponent = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64)
        if int(exponent.max(initial=0)) == 0xFF:
            raise ValueError("router weights must be finite")
        fraction = (bits & np.uint32(0x7FFFFF)).astype(np.int64)
        normal = exponent > 0
        mantissa = np.where(normal, fraction | (1 << 23), fraction)
        shift = np.where(normal, exponent - 1, 0)
        expert_ids = ids.numpy().astype(np.int64)
        if expert_ids.shape != mantissa.shape:
            raise ValueError("flat capture ids/weights shape drift")
        key = (expert_ids * 256 + shift).reshape(-1)
        totals = np.bincount(
            key,
            weights=mantissa.reshape(-1).astype(np.float64),
            minlength=bins,
        )
        if not bool(np.all(totals == np.rint(totals))):
            raise ValueError("mass histogram lost integer exactness")
        shifted += totals.astype(np.int64)
        counts += np.bincount(expert_ids.reshape(-1), minlength=num_experts).astype(
            np.int64
        )
        if int(shifted.max(initial=0)) >= (1 << 62):
            raise ValueError("mass histogram bucket overflow")

    units: list[int] = []
    table = shifted.reshape(num_experts, 256)
    for expert in range(num_experts):
        row = table[expert]
        total = 0
        for position in np.nonzero(row)[0]:
            total += int(row[int(position)]) << int(position)
        units.append(total)
    return units, [int(value) for value in counts.tolist()]


def _row_mass_sums(reader: Any, *, chunk: int = MASS_CHUNK_ROWS):
    """float64 per-token routed-mass sums, computed the way the accumulator does."""

    import torch

    total = torch.empty(int(reader.tokens), dtype=torch.float64)
    for begin in range(0, int(reader.tokens), int(chunk)):
        end = min(begin + int(chunk), int(reader.tokens))
        total[begin:end] = reader.weights[begin:end].to(torch.float64).sum(dim=1)
    return total


def build_flat_mass_audit(
    reader: Any,
    *,
    layer: int,
    row_tolerance: float = ROW_MASS_TOLERANCE,
    chunk: int = MASS_CHUNK_ROWS,
    cross_check_rows: int = MASS_CROSS_CHECK_ROWS,
) -> MassAudit:
    """Build the `MassAudit` `capture_layer` would have produced for this layer.

    The sealed walk gets ``expected_mass_per_token`` from the live router. A
    flat capture stores only the weights, so the contract value is recovered
    from them: the midpoint of the observed float64 row sums. That is
    order-invariant (min and max are), batch-invariant, and it is the choice
    that minimizes the worst-case per-row deviation, which is exactly the
    quantity ``RoutedMassAccumulator`` polices.
    """

    import torch

    value = int(layer)
    tokens = int(reader.tokens)
    top_k = int(reader.top_k)
    num_experts = int(reader.num_experts)
    if tokens <= 0:
        raise ValueError(f"layer {value} flat capture has no routed tokens")
    if top_k != TOP_K or num_experts != NUM_EXPERTS:
        raise ValueError(
            f"flat capture routing geometry drift: top_k={top_k} "
            f"num_experts={num_experts}, expected {TOP_K}/{NUM_EXPERTS}"
        )

    row_sums = _row_mass_sums(reader, chunk=chunk)
    if not bool(torch.isfinite(row_sums).all()):
        raise ValueError("flat capture routed mass contains non-finite row sums")
    low = float(row_sums.min().item())
    high = float(row_sums.max().item())
    expected_mass_per_token = (low + high) / 2.0
    if not math.isfinite(expected_mass_per_token) or expected_mass_per_token <= 0.0:
        raise ValueError(
            "expected routed mass per token must be finite and positive; flat "
            f"capture row sums span [{low:.17g}, {high:.17g}]"
        )
    max_row_mass_error = float((row_sums - expected_mass_per_token).abs().max().item())
    if max_row_mass_error > float(row_tolerance):
        raise ValueError(
            f"router mass normalization drift at layer {value}: max row error "
            f"{max_row_mass_error:.9g} > {float(row_tolerance):.9g}"
        )
    del row_sums

    units, counts = _exact_mass_units(reader, chunk=chunk)
    if sum(counts) != tokens * top_k:
        raise ValueError("flat capture routed-count accounting drift")
    declared = [int(item) for item in reader.routed_counts]
    if declared != counts:
        raise ValueError("flat capture manifest routed counts differ from payload")

    # Prove the vectorized folding equals the sealed accumulator on a
    # deterministic prefix. Cheap, and it makes a future refactor of either
    # implementation fail closed instead of silently reallocating bits.
    sample_rows = min(int(cross_check_rows), tokens)
    if sample_rows > 0:
        reference = RoutedMassAccumulator(value, row_tolerance=float(row_tolerance))
        reference.add(
            reader.ids[:sample_rows].to(torch.int64),
            reader.weights[:sample_rows],
            expected_mass_per_token,
        )
        sample_units, sample_counts = _exact_mass_units(
            reader, last_row=sample_rows, chunk=chunk
        )
        if (
            list(reference.mass_units) != sample_units
            or list(reference.counts) != sample_counts
        ):
            raise ValueError(
                "flat routed-mass folding disagrees with RoutedMassAccumulator"
            )

    expected_total_mass_float = tokens * expected_mass_per_token
    observed_units = sum(units)
    observed_float = math.ldexp(float(observed_units), MASS_UNIT_POWER)
    tolerance = max(
        1e-7 * expected_total_mass_float,
        float(row_tolerance) * tokens,
    )
    if abs(observed_float - expected_total_mass_float) > tolerance:
        raise ValueError(
            f"layer {value} total routed mass mismatch: "
            f"observed={observed_float:.17g} "
            f"expected={expected_total_mass_float:.17g}"
        )
    return MassAudit(
        layer=value,
        tokens=tokens,
        assignments=tokens * top_k,
        expected_total_mass=format(expected_total_mass_float, ".17g"),
        observed_total_mass=units_to_decimal(observed_units),
        mass_by_expert=tuple(units_to_decimal(item) for item in units),
        mass_units_by_expert=tuple(str(item) for item in units),
        count_by_expert=tuple(counts),
        max_row_mass_error=format(max_row_mass_error, ".17g"),
    )


# ---------------------------------------------------------------------------
# row views in the shape GLM52Backend._iter_bound_batches yields
# ---------------------------------------------------------------------------


def build_flat_bound_batches(
    reader: Any,
    *,
    layer: int,
    chunk: int = FLAT_BATCH_TOKENS,
) -> list[tuple[StateShard, CalibrationBatch, Any, Any, Any]]:
    """Present the flat capture as bound `(shard, batch, ids, weights, row_ids)`.

    This is the exact 5-tuple ``GLM52Backend._iter_bound_batches`` yields for a
    sealed capture, so ``iter_expert_rows`` and ``iter_cold_fallback_rows``
    consume it unmodified:

    * ``batch.hidden`` is the captured post-attention MoE input, bf16, a view
      onto the read-only mmap (no copy, no RAM duplication);
    * ``ids`` is int32, matching what ``_read_sidecar`` returns;
    * ``weights`` is the exact float32 router output;
    * ``row_ids`` are the capture's own ascending token indices, which are what
      ``GLM52Backend._split`` hashes for fit/holdout membership.

    Blocks are fixed size and emitted in ascending row order, with shard IDs
    that sort in that same order, so every downstream selection is stable.
    """

    import torch

    tokens = int(reader.tokens)
    block = int(chunk)
    if block <= 0:
        raise ValueError("flat batch chunk must be positive")
    if int(reader.hidden_size) != HIDDEN_SIZE:
        raise ValueError(
            f"flat capture hidden size {reader.hidden_size} != {HIDDEN_SIZE}"
        )
    manifest = reader.manifest
    hidden_sha = _hex64(manifest.get("sha256_x"), "sha256_x")
    manifest_sha = _hex64(manifest.get("content_sha256"), "content_sha256")
    hidden_path = Path(reader.layer_dir) / X_FILE
    # int32 mirrors the sealed routing sidecar dtype exactly; the uint8 storage
    # form is a capture detail and must not leak into row selection.
    ids32 = reader.ids.to(torch.int32)
    batches: list[tuple[StateShard, CalibrationBatch, Any, Any, Any]] = []
    for index, begin in enumerate(range(0, tokens, block)):
        end = min(begin + block, tokens)
        shard_id = f"{FLAT_SHARD_PREFIX}{index:05d}"
        rows = end - begin
        row_ids = torch.arange(begin, end, dtype=torch.int64)
        shard = StateShard(
            shard_id=shard_id,
            hidden_path=hidden_path,
            metadata_path=Path(reader.manifest_path),
            tokens=rows,
            hidden_size=HIDDEN_SIZE,
            sha256_hidden=hidden_sha,
            sha256_metadata=manifest_sha,
        )
        batch = CalibrationBatch(
            shard_id=shard_id,
            hidden=reader.hidden[begin:end],
            row_ids=row_ids,
            attention_metadata={
                "schema": RUNTIME_SCHEMA,
                "source": FLAT_CAPTURE_SCHEMA,
                "layer": int(layer),
                "tokens": rows,
                "global_row_start": begin,
                "hidden_role": "post-attention-post-norm-moe-input",
            },
            token_count=rows,
        )
        batches.append(
            (shard, batch, ids32[begin:end], reader.weights[begin:end], row_ids)
        )
    if not batches:
        raise ValueError("flat capture produced no bound row blocks")
    return batches


# ---------------------------------------------------------------------------
# the runtime object itself
# ---------------------------------------------------------------------------


class FlatRuntime(GLM52Runtime):
    """Non-executing runtime for per-layer encode from a sealed flat capture.

    ``GLM52Backend`` requires ``runtime.fingerprint`` to equal the sealed
    runtime-code inventory digest, exactly as
    ``TransformersSequentialRuntime`` does; that check is what binds an encode
    to a pinned GLM-5.2 source tree, so it is honoured here rather than
    replaced. The module's own content hash is published separately as
    :attr:`code_fingerprint` and carried in :meth:`provenance`.
    """

    def __init__(self, config) -> None:
        self.config = config
        # GLM52Backend.__init__ resolves `config.work` before anything else.
        # The layer-parallel driver's config carries `capture_dir` instead --
        # the flat path never reads work/states or writes routing sidecars --
        # so fill the field in rather than crash on attribute lookup. Setting
        # it here is safe: this factory runs before the backend constructor.
        if getattr(config, "work", None) is None:
            fallback = getattr(config, "capture_dir", None)
            if fallback is None:
                raise ValueError(
                    "flat runtime config must carry `work` or `capture_dir`"
                )
            try:
                config.work = Path(fallback)
            except AttributeError as error:  # frozen/slots config
                raise ValueError(
                    "flat runtime config is immutable and lacks `work`"
                ) from error
        inventory_path = getattr(config, "runtime_inventory", None)
        if inventory_path is None:
            raise ValueError("flat runtime config must carry `runtime_inventory`")
        from .inventory import load_runtime_code_inventory

        self.runtime_inventory = load_runtime_code_inventory(
            inventory_path,
            verify_files=bool(getattr(config, "verify_runtime_files", True)),
        )
        self._fingerprint = str(self.runtime_inventory["inventory_sha256"])
        if not self._fingerprint:
            raise ValueError("sealed runtime code inventory has no digest")
        capture_dir = getattr(config, "capture_dir", None)
        self.capture_dir = None if capture_dir is None else Path(capture_dir)
        self._code_fingerprint = sha256_bytes(
            canonical_json_bytes(
                {
                    "marker": RECIPE_MARKER,
                    "recipe_version": RECIPE_VERSION,
                    "schema": RUNTIME_SCHEMA,
                    "flat_capture_schema": FLAT_CAPTURE_SCHEMA,
                    "module_sha256": sha256_file(Path(__file__)),
                }
            )
        )

    # -- identity --------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def code_fingerprint(self) -> str:
        """Stable hash of this module plus the flat-capture schema string."""

        return self._code_fingerprint

    def provenance(self) -> Mapping[str, object]:
        return {
            "schema": RUNTIME_SCHEMA,
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "flat_capture_schema": FLAT_CAPTURE_SCHEMA,
            "runtime_fingerprint": self._fingerprint,
            "code_fingerprint": self._code_fingerprint,
            "capture_dir": None if self.capture_dir is None else str(self.capture_dir),
            "executes_model": False,
        }

    # -- everything that needs a live model ------------------------------

    def prepare_corpus_plan(self, *, corpus: Path) -> Mapping[str, object]:
        raise NotImplementedError(_UNSUPPORTED.format(method="prepare_corpus_plan"))

    def initialize_carried_state(
        self,
        *,
        carrier: Path,
        corpus: Path,
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        raise NotImplementedError(
            _UNSUPPORTED.format(method="initialize_carried_state")
        )

    def route_exact(
        self, *, layer: int, moe_hidden: Any, attention_metadata: Mapping[str, object]
    ):
        raise NotImplementedError(
            _UNSUPPORTED.format(method="route_exact")
            + " Routing for this layer is already sealed in the flat capture."
        )

    def prepare_moe_input(
        self, *, layer: int, hidden: Any, attention_metadata: Mapping[str, object]
    ) -> Any:
        raise NotImplementedError(
            _UNSUPPORTED.format(method="prepare_moe_input")
            + " The post-attention MoE input is already sealed in x.bin."
        )

    def begin_capture(self, *, layer: int) -> None:
        raise NotImplementedError(
            _UNSUPPORTED.format(method="begin_capture")
            + " Capture is phase A of the layer-parallel driver."
        )

    def end_capture(self, *, layer: int) -> None:
        # Deliberately a no-op: it only ever runs in a `finally` and must not
        # mask the real failure raised by begin_capture.
        return None

    def capture_arithmetic_audit(self, *, layer: int) -> Mapping[str, object]:
        raise NotImplementedError(
            _UNSUPPORTED.format(method="capture_arithmetic_audit")
            + " The flat capture is bound by its own manifest digests."
        )

    def install_encoded_expert(
        self, *, layer: int, expert: int, encoded: Mapping[str, Any]
    ) -> Mapping[str, object]:
        raise NotImplementedError(
            _UNSUPPORTED.format(method="install_encoded_expert")
        )

    def audit_installed_layer(self, *, layer: int) -> Mapping[str, object]:
        raise NotImplementedError(_UNSUPPORTED.format(method="audit_installed_layer"))

    def restore_encoded_layer(self, *, layer: int, manifest: Path) -> None:
        raise NotImplementedError(_UNSUPPORTED.format(method="restore_encoded_layer"))

    def forward_installed_layer(
        self,
        *,
        layer: int,
        input_shards: Iterable[StateShard],
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        raise NotImplementedError(
            _UNSUPPORTED.format(method="forward_installed_layer")
        )


def factory(config) -> FlatRuntime:
    """Driver factory: ``runtime_factory='r7_encoder.flat_runtime:factory'``."""

    return FlatRuntime(config)
