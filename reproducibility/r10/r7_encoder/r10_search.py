"""Fast deterministic search for the layer-parallel Round 10 encoder.

The original :mod:`r7_encoder.search` used complete 4-bit TRELLIS encodes as
search-time validators.  Those encodes do not contribute bytes to the final
checkpoint -- the later 3/4/5 sensitivity pass encodes every tensor again --
and therefore only added wall clock.  This runner keeps the selected search
contract while making every decision from cheap, deterministic proxies:

* all twelve (by default) shared residual-side rotation draws are scored on a
  mass-stratified layer sample;
* all five intermediate permutations and all private rotation draws are
  scored for every expert;
* every 128-coordinate scale is selected by a real grid/coordinate search over
  block energy weighted by the already-required gate/up covariance diagonal;
* no candidate is sent through the TRELLIS codec and no holdout/full-roundtrip
  encode is performed merely to validate search.

``LayerProcessor.run`` remains authoritative for the emitted bytes: it still
does joint full-K down encoding, constructs down inputs through reconstructed
gate/up, probes 3/4/5 bits, allocates exactly 3.5 bpw, and performs the final
encode.  This module changes only how the free permutations, random vectors,
and folded block scales are selected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .constants import HAD_K, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS
from .determinism import derive_seed, sha256_file
from .permutation import (
    descending_diag_permutation,
    energy_balanced_permutation,
    identity_permutation,
    stored_descending_diag_permutation,
    validate_permutation,
)
from .rotations import rademacher_vector
from .search import SearchRunner as _R7SearchRunner
from .search import _base_expert
from .search_artifact import ExpertSearch, LayerSearch, load_layer_search, write_layer_search


ENGINE = "r10-deterministic-block-proxy-v1"
SCALE_GRID = (0.80, 0.90, 1.00, 1.10, 1.25)
SCALE_SWEEPS = 2
SKETCH_COORDINATES = 512
SKETCH_BUCKETS = 32

PERMUTATION_POLICIES = (
    "identity",
    "ldlq_visit_descending_diag",
    "stored_descending_diag",
    "energy_balanced",
    "energy_balanced_contiguous",
)


def _positive(values: Sequence[float]) -> tuple[float, ...]:
    """Return a finite positive proxy vector, preserving coordinate count."""

    result = tuple(max(float(value), 1e-30) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("block proxy must be finite, positive, and nonempty")
    return result


def _block_means(values: Sequence[float], block: int = HAD_K) -> tuple[float, ...]:
    numeric = _positive(values)
    if len(numeric) % block:
        raise ValueError("proxy coordinate count is not block aligned")
    return tuple(
        sum(numeric[start : start + block]) / block
        for start in range(0, len(numeric), block)
    )


def _scale_targets(metrics: Sequence[float]) -> tuple[float, ...]:
    """Quarter-RMS balancing targets, clipped to the owner-locked grid span."""

    values = _positive(metrics)
    geometric = math.exp(sum(math.log(value) for value in values) / len(values))
    low, high = min(SCALE_GRID), max(SCALE_GRID)
    return tuple(
        max(low, min(high, (geometric / value) ** 0.25)) for value in values
    )


def block_scale_proxy_score(
    metrics: Sequence[float], scales: Sequence[float]
) -> float:
    """Energy/covariance-weighted log error from per-block balancing targets.

    Unlike the former three-family pilot, this score changes when *one* block
    changes.  It is intentionally separable and inexpensive: the expensive
    data reduction is performed once before the coordinate walk.
    """

    values = _positive(metrics)
    factors = tuple(float(value) for value in scales)
    if len(factors) != len(values) or any(
        not math.isfinite(value) or value <= 0 for value in factors
    ):
        raise ValueError("scale vector does not match its block proxy")
    targets = _scale_targets(values)
    denominator = sum(values)
    score = sum(
        weight * math.log(factor / target) ** 2
        for weight, factor, target in zip(values, factors, targets)
    ) / denominator
    if not math.isfinite(score):
        raise ValueError("non-finite block-scale proxy score")
    return score


def coordinate_grid_block_scales(
    metrics: Sequence[float],
    *,
    grid: Sequence[float] = SCALE_GRID,
    sweeps: int = SCALE_SWEEPS,
) -> tuple[tuple[float, ...], float]:
    """Deterministically score the complete grid at every 128-block coordinate."""

    values = _positive(metrics)
    candidates = tuple(float(value) for value in grid)
    if sweeps <= 0 or not candidates or any(
        not math.isfinite(value) or value <= 0 for value in candidates
    ):
        raise ValueError("invalid block-scale coordinate grid")
    factors = [1.0] * len(values)
    best = block_scale_proxy_score(values, factors)
    for _ in range(int(sweeps)):
        for index in range(len(factors)):
            scored: list[tuple[float, float]] = []
            incumbent = factors[index]
            for candidate in candidates:
                factors[index] = candidate
                scored.append((block_scale_proxy_score(values, factors), candidate))
            score, selected = min(scored, key=lambda item: (item[0], item[1]))
            factors[index] = selected if score <= best else incumbent
            best = min(best, score)
    return tuple(factors), best


@dataclass(frozen=True)
class _TensorSketch:
    """A fixed sparse view of one matrix; constructing it copies only 512 values."""

    n: int
    k: int
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    values: tuple[float, ...]

    def score(self, suh: Sequence[float], svh: Sequence[float]) -> float:
        if len(suh) != self.k or len(svh) != self.n:
            raise ValueError("rotation vector does not match sketch geometry")
        buckets = [0.0] * SKETCH_BUCKETS
        absolute = 0.0
        for index, (row, column, raw) in enumerate(
            zip(self.rows, self.columns, self.values)
        ):
            value = raw * float(suh[column]) * float(svh[row])
            buckets[index % SKETCH_BUCKETS] += value
            absolute += abs(value)
        return sum(value * value for value in buckets) / max(
            absolute * absolute, 1e-30
        )


@dataclass
class _ExpertProxy:
    weights: object
    identity_sketches: tuple[_TensorSketch, _TensorSketch, _TensorSketch]
    shared_gate_k_blocks: tuple[float, ...]
    shared_down_n_blocks: tuple[float, ...]
    gate_private_channels: tuple[float, ...]
    up_private_channels: tuple[float, ...]
    down_private_channels: tuple[float, ...]
    permutation_importance: tuple[float, ...]


def _progress_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError("cannot persist a non-finite search score")
    return format(float(value), ".17g")


class SearchRunner(_R7SearchRunner):
    """Drop-in ``SearchRunner`` whose search phase performs zero TRELLIS encodes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        incumbent = self.progress.get("engine")
        has_work = bool(self.progress.get("scores")) or bool(
            self.progress.get("expert_results")
        ) or self.progress.get("shared_result") is not None
        if incumbent is None and has_work:
            raise ValueError("refusing to mix R7 full-encode progress with R10 search")
        if incumbent not in (None, ENGINE):
            raise ValueError(f"search progress engine drift: {incumbent!r}")
        self.progress["engine"] = ENGINE
        contract = {
            "scale_grid": [format(value, ".17g") for value in SCALE_GRID],
            "scale_sweeps": SCALE_SWEEPS,
            "permutation_policies": list(PERMUTATION_POLICIES),
        }
        for key, expected in contract.items():
            prior = self.progress.get(key)
            if prior is not None and prior != expected:
                raise ValueError(f"search progress {key} drift")
            self.progress[key] = expected
        self._proxies: dict[int, _ExpertProxy] = {}
        self._flush()

    # ------------------------------------------------------------------
    # one-time proxy preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _row_quadratic_energy(
        weight, input_diagonal, *, device="cpu"
    ) -> tuple[float, ...]:
        import torch

        matrix = torch.as_tensor(
            weight, dtype=torch.float32, device=device
        ).detach()
        diagonal = torch.as_tensor(
            input_diagonal, dtype=torch.float32, device=device
        )
        if matrix.ndim != 2 or diagonal.numel() != matrix.shape[1]:
            raise ValueError("row-energy proxy geometry mismatch")
        return _positive((matrix.square() @ diagonal).detach().cpu().tolist())

    @staticmethod
    def _column_weighted_energy(
        weight, row_weight, *, device="cpu"
    ) -> tuple[float, ...]:
        import torch

        matrix = torch.as_tensor(
            weight, dtype=torch.float32, device=device
        ).detach()
        importance = torch.as_tensor(
            row_weight, dtype=torch.float32, device=device
        )
        if matrix.ndim != 2 or importance.numel() != matrix.shape[0]:
            raise ValueError("column-energy proxy geometry mismatch")
        return _positive(
            (matrix.square().T @ importance).detach().cpu().tolist()
        )

    @staticmethod
    def _make_sketch(
        weight,
        *,
        seed_parts: Sequence[object],
        row_map: Sequence[int] | None = None,
        column_map: Sequence[int] | None = None,
    ) -> _TensorSketch:
        import torch

        matrix = torch.as_tensor(weight).detach().to("cpu")
        if matrix.ndim != 2:
            raise ValueError("rotation sketch requires a matrix")
        n, k = (int(matrix.shape[0]), int(matrix.shape[1]))
        rows: list[int] = []
        columns: list[int] = []
        source_rows: list[int] = []
        source_columns: list[int] = []
        seed = derive_seed("r10-rotation-sketch", *seed_parts, bits=64)
        for index in range(SKETCH_COORDINATES):
            coordinate = derive_seed(seed, index, bits=64)
            row = int(coordinate % n)
            column = int((coordinate // n) % k)
            rows.append(row)
            columns.append(column)
            source_rows.append(row if row_map is None else int(row_map[row]))
            source_columns.append(
                column if column_map is None else int(column_map[column])
            )
        row_index = torch.tensor(source_rows, dtype=torch.long)
        column_index = torch.tensor(source_columns, dtype=torch.long)
        values = matrix[row_index, column_index].to(torch.float32).tolist()
        return _TensorSketch(n, k, tuple(rows), tuple(columns), tuple(values))

    def _prepare_proxy(self, expert: int) -> _ExpertProxy:
        cached = self._proxies.get(expert)
        if cached is not None:
            return cached
        import torch

        weights = self.backend.load_bf16_expert(
            layer=self.capture.layer, expert=expert
        )
        weights.validate_bf16(self.capture.layer, expert)
        remember = getattr(self.processor, "remember_search_weights", None)
        if remember is not None:
            remember(expert, weights)

        # This covariance is required again by the real sensitivity/final pass.
        # LayerProcessor seals it in its row-plan cache, so preparing its
        # diagonal here moves required work earlier; it does not duplicate it.
        covariance = self.processor._covariance(
            self.capture, self.shards, expert, "fit"
        )
        device = self.processor.device
        input_diagonal = covariance.matrix.diagonal().detach().to(
            device, dtype=torch.float32
        )
        input_diagonal = input_diagonal / input_diagonal.mean().clamp_min(1e-30)

        gate_variance = self._row_quadratic_energy(
            weights.gate_hf, input_diagonal, device=device
        )
        up_variance = self._row_quadratic_energy(
            weights.up_hf, input_diagonal, device=device
        )
        down_column_energy = self._column_weighted_energy(
            weights.down_hf, (1.0,) * HIDDEN_SIZE, device=device
        )

        # Diagonal moment proxy for the SwiGLU output and its propagation
        # through down.  It is deliberately computed from BF16 weights only;
        # search must not manufacture a gate/up quantized reconstruction.
        activation = _positive(
            tuple(gate * up for gate, up in zip(gate_variance, up_variance))
        )
        propagated = _positive(
            tuple(value * output for value, output in zip(activation, down_column_energy))
        )
        # Keep gate and up as genuinely separate measured families.  The
        # opposite branch supplies an RMS propagation factor; multiplying by
        # its full variance here would algebraically collapse both families to
        # the same vector and make their nominally independent scale searches
        # cosmetic.
        gate_sensitivity = _positive(
            tuple(
                math.sqrt(up) * output
                for up, output in zip(up_variance, down_column_energy)
            )
        )
        up_sensitivity = _positive(
            tuple(
                math.sqrt(gate) * output
                for gate, output in zip(gate_variance, down_column_energy)
            )
        )

        gate_input = self._column_weighted_energy(
            weights.gate_hf, gate_sensitivity, device=device
        )
        up_input = self._column_weighted_energy(
            weights.up_hf, up_sensitivity, device=device
        )
        shared_gate_channels = _positive(
            tuple(
                float(input_diagonal[index]) * (gate_input[index] + up_input[index])
                for index in range(HIDDEN_SIZE)
            )
        )
        down_output = self._row_quadratic_energy(
            weights.down_hf, activation, device=device
        )

        sketches = (
            self._make_sketch(
                weights.gate_hf,
                seed_parts=(self.capture.layer, expert, "gate_proj"),
            ),
            self._make_sketch(
                weights.up_hf,
                seed_parts=(self.capture.layer, expert, "up_proj"),
            ),
            self._make_sketch(
                weights.down_hf,
                seed_parts=(self.capture.layer, expert, "down_proj"),
            ),
        )
        value = _ExpertProxy(
            weights=weights,
            identity_sketches=sketches,
            shared_gate_k_blocks=_block_means(shared_gate_channels),
            shared_down_n_blocks=_block_means(down_output),
            gate_private_channels=_positive(
                tuple(gate * sensitivity for gate, sensitivity in zip(gate_variance, gate_sensitivity))
            ),
            up_private_channels=_positive(
                tuple(up * sensitivity for up, sensitivity in zip(up_variance, up_sensitivity))
            ),
            down_private_channels=propagated,
            permutation_importance=propagated,
        )
        self._proxies[expert] = value
        return value

    # ------------------------------------------------------------------
    # shared residual-side decision
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_draw_score(
        proxy: _ExpertProxy, base: ExpertSearch, suh, svh
    ) -> float:
        gate, up, down = proxy.identity_sketches
        return (
            gate.score(suh, base.gate_svh)
            + up.score(suh, base.up_svh)
            + down.score(base.down_suh, svh)
        )

    @staticmethod
    def _mass_weighted_blocks(
        records: Sequence[tuple[float, Sequence[float]]]
    ) -> tuple[float, ...]:
        if not records:
            raise ValueError("shared proxy sample is empty")
        width = len(records[0][1])
        total = sum(float(mass) for mass, _ in records)
        if total <= 0 or any(len(values) != width for _, values in records):
            raise ValueError("shared block proxy domain drift")
        return _positive(
            tuple(
                sum(float(mass) * float(values[index]) for mass, values in records)
                / total
                for index in range(width)
            )
        )

    def _choose_shared_proxy(self, experts: Mapping[int, ExpertSearch]):
        masses = tuple(float(value) for value in self.capture.mass_audit.mass_by_expert)
        prepared = {expert: self._prepare_proxy(expert) for expert in self.sample}
        denominator = sum(masses[expert] for expert in self.sample)
        draw_scores: dict[str, str] = {}
        candidates = []
        for draw in range(self.draws):
            suh = rademacher_vector(
                HIDDEN_SIZE, self.capture.layer, "shared", "gate_up_suh", draw
            )
            svh = rademacher_vector(
                HIDDEN_SIZE, self.capture.layer, "shared", "down_svh", draw
            )
            score = sum(
                masses[expert]
                * self._shared_draw_score(prepared[expert], experts[expert], suh, svh)
                for expert in self.sample
            ) / denominator
            draw_scores[str(draw)] = _progress_float(score)
            candidates.append((score, draw, suh, svh))
        rotation_score, draw, suh, svh = min(
            candidates, key=lambda item: (item[0], item[1])
        )

        gate_metrics = self._mass_weighted_blocks(
            [(masses[expert], prepared[expert].shared_gate_k_blocks) for expert in self.sample]
        )
        down_metrics = self._mass_weighted_blocks(
            [(masses[expert], prepared[expert].shared_down_n_blocks) for expert in self.sample]
        )
        gate_scales, gate_score = coordinate_grid_block_scales(gate_metrics)
        down_scales, down_score = coordinate_grid_block_scales(down_metrics)
        result = {
            "schema": ENGINE,
            "draw": draw,
            "draw_scores": draw_scores,
            "gate_up_k_g_scale": list(gate_scales),
            "down_n_g_scale": list(down_scales),
            "rotation_score": _progress_float(rotation_score),
            "gate_scale_score": _progress_float(gate_score),
            "down_scale_score": _progress_float(down_score),
            "selection_score": _progress_float(rotation_score + gate_score + down_score),
        }
        return suh, svh, gate_scales, down_scales, result

    def _restore_shared(self, raw: Mapping[str, object]):
        if raw.get("schema") != ENGINE:
            raise ValueError("shared R10 progress record is malformed")
        scores = raw.get("draw_scores")
        if not isinstance(scores, Mapping) or set(scores) != {
            str(draw) for draw in range(self.draws)
        }:
            raise ValueError("shared progress does not score every rotation draw")
        draw = int(raw["draw"])
        if not 0 <= draw < self.draws:
            raise ValueError("shared progress selected an invalid draw")
        k_scales = tuple(float(value) for value in raw["gate_up_k_g_scale"])
        n_scales = tuple(float(value) for value in raw["down_n_g_scale"])
        if len(k_scales) != HIDDEN_SIZE // HAD_K or len(n_scales) != HIDDEN_SIZE // HAD_K:
            raise ValueError("shared progress block-scale geometry drift")
        return (
            rademacher_vector(
                HIDDEN_SIZE, self.capture.layer, "shared", "gate_up_suh", draw
            ),
            rademacher_vector(
                HIDDEN_SIZE, self.capture.layer, "shared", "down_svh", draw
            ),
            k_scales,
            n_scales,
            float(raw["selection_score"]),
            draw,
        )

    # ------------------------------------------------------------------
    # per-expert intermediate-side decision
    # ------------------------------------------------------------------

    @staticmethod
    def _permutation_score(
        importance: Sequence[float], permutation: Sequence[int]
    ) -> float:
        values = _positive(importance)
        perm = validate_permutation(permutation, len(values))
        visit = tuple(reversed(perm))
        denominator = sum(values)
        order = sum(
            values[old] * (position / max(len(values) - 1, 1))
            for position, old in enumerate(visit)
        ) / denominator
        stored = tuple(values[old] for old in perm)
        blocks = tuple(
            sum(stored[start : start + HAD_K])
            for start in range(0, len(stored), HAD_K)
        )
        mean = sum(blocks) / len(blocks)
        imbalance = sum((value - mean) ** 2 for value in blocks) / max(
            len(blocks) * mean * mean, 1e-30
        )
        return order + 0.01 * imbalance

    def _policy_candidates(self, proxy: _ExpertProxy):
        diagonal = proxy.permutation_importance
        policies = {
            "identity": identity_permutation(),
            "ldlq_visit_descending_diag": descending_diag_permutation(diagonal),
            "stored_descending_diag": stored_descending_diag_permutation(diagonal),
            "energy_balanced": energy_balanced_permutation(diagonal),
            "energy_balanced_contiguous": energy_balanced_permutation(
                diagonal, serpentine=False
            ),
        }
        if tuple(policies) != PERMUTATION_POLICIES:
            raise AssertionError("R10 must evaluate exactly five permutation policies")
        scored = {
            name: self._permutation_score(diagonal, permutation)
            for name, permutation in policies.items()
        }
        name = min(PERMUTATION_POLICIES, key=lambda item: (scored[item], item))
        return name, policies[name], scored

    def _permuted_sketches(
        self, expert: int, proxy: _ExpertProxy, permutation: Sequence[int]
    ) -> tuple[_TensorSketch, _TensorSketch, _TensorSketch]:
        if tuple(permutation) == identity_permutation():
            return proxy.identity_sketches
        weights = proxy.weights
        seed = (self.capture.layer, expert)
        return (
            self._make_sketch(
                weights.gate_hf,
                seed_parts=(*seed, "gate_proj"),
                row_map=permutation,
            ),
            self._make_sketch(
                weights.up_hf,
                seed_parts=(*seed, "up_proj"),
                row_map=permutation,
            ),
            self._make_sketch(
                weights.down_hf,
                seed_parts=(*seed, "down_proj"),
                column_map=permutation,
            ),
        )

    def _choose_expert_proxy(
        self, base: LayerSearch, expert: int, proxy: _ExpertProxy
    ) -> tuple[ExpertSearch, dict[str, object]]:
        policy, permutation, policy_scores = self._policy_candidates(proxy)
        gate_sketch, up_sketch, down_sketch = self._permuted_sketches(
            expert, proxy, permutation
        )

        draw_scores: dict[str, str] = {}
        draw_candidates = []
        for draw in range(self.draws):
            gate_svh = rademacher_vector(
                INTERMEDIATE_SIZE, self.capture.layer, expert, "gate_svh", draw
            )
            up_svh = rademacher_vector(
                INTERMEDIATE_SIZE, self.capture.layer, expert, "up_svh", draw
            )
            down_suh = rademacher_vector(
                INTERMEDIATE_SIZE, self.capture.layer, expert, "down_suh", draw
            )
            score = (
                gate_sketch.score(base.gate_up_suh, gate_svh)
                + up_sketch.score(base.gate_up_suh, up_svh)
                + down_sketch.score(down_suh, base.down_svh)
            )
            draw_scores[str(draw)] = _progress_float(score)
            draw_candidates.append((score, draw, gate_svh, up_svh, down_suh))
        rotation_score, draw, gate_svh, up_svh, down_suh = min(
            draw_candidates, key=lambda item: (item[0], item[1])
        )

        ordered_gate = tuple(proxy.gate_private_channels[index] for index in permutation)
        ordered_up = tuple(proxy.up_private_channels[index] for index in permutation)
        ordered_down = tuple(proxy.down_private_channels[index] for index in permutation)
        gate_scales, gate_scale_score = coordinate_grid_block_scales(
            _block_means(ordered_gate)
        )
        up_scales, up_scale_score = coordinate_grid_block_scales(
            _block_means(ordered_up)
        )
        down_scales, down_scale_score = coordinate_grid_block_scales(
            _block_means(ordered_down)
        )
        selection_score = (
            policy_scores[policy]
            + rotation_score
            + gate_scale_score
            + up_scale_score
            + down_scale_score
        )
        chosen = ExpertSearch(
            permutation=tuple(permutation),
            permutation_policy=policy,
            gate_svh=tuple(gate_svh),
            up_svh=tuple(up_svh),
            down_suh=tuple(down_suh),
            gate_n_g_scale=gate_scales,
            up_n_g_scale=up_scales,
            down_k_g_scale=down_scales,
            draw=draw,
            selection_score=selection_score,
            selection_score_kind="deterministic-sketch",
        )
        progress = {
            "schema": ENGINE,
            "permutation": list(permutation),
            "permutation_policy": policy,
            "permutation_scores": {
                name: _progress_float(policy_scores[name])
                for name in PERMUTATION_POLICIES
            },
            "draw": draw,
            "draw_scores": draw_scores,
            "gate_n_g_scale": list(gate_scales),
            "up_n_g_scale": list(up_scales),
            "down_k_g_scale": list(down_scales),
            "gate_scale_score": _progress_float(gate_scale_score),
            "up_scale_score": _progress_float(up_scale_score),
            "down_scale_score": _progress_float(down_scale_score),
            "selection_score": _progress_float(selection_score),
        }
        return chosen, progress

    def _restore_expert(self, expert: int, raw: Mapping[str, object]) -> ExpertSearch:
        if raw.get("schema") != ENGINE:
            raise ValueError(f"expert {expert}: malformed R10 progress record")
        policy_scores = raw.get("permutation_scores")
        draw_scores = raw.get("draw_scores")
        if not isinstance(policy_scores, Mapping) or set(policy_scores) != set(
            PERMUTATION_POLICIES
        ):
            raise ValueError(f"expert {expert}: five-policy score domain drift")
        if not isinstance(draw_scores, Mapping) or set(draw_scores) != {
            str(draw) for draw in range(self.draws)
        }:
            raise ValueError(f"expert {expert}: rotation draw score domain drift")
        draw = int(raw["draw"])
        if not 0 <= draw < self.draws:
            raise ValueError(f"expert {expert}: invalid selected draw")
        policy = str(raw["permutation_policy"])
        if policy not in PERMUTATION_POLICIES:
            raise ValueError(f"expert {expert}: invalid selected policy")
        return ExpertSearch(
            permutation=validate_permutation(raw["permutation"], INTERMEDIATE_SIZE),
            permutation_policy=policy,
            gate_svh=rademacher_vector(
                INTERMEDIATE_SIZE, self.capture.layer, expert, "gate_svh", draw
            ),
            up_svh=rademacher_vector(
                INTERMEDIATE_SIZE, self.capture.layer, expert, "up_svh", draw
            ),
            down_suh=rademacher_vector(
                INTERMEDIATE_SIZE, self.capture.layer, expert, "down_suh", draw
            ),
            gate_n_g_scale=tuple(float(value) for value in raw["gate_n_g_scale"]),
            up_n_g_scale=tuple(float(value) for value in raw["up_n_g_scale"]),
            down_k_g_scale=tuple(float(value) for value in raw["down_k_g_scale"]),
            draw=draw,
            selection_score=float(raw["selection_score"]),
            selection_score_kind="deterministic-sketch",
        )

    # ------------------------------------------------------------------
    # compatible public runner
    # ------------------------------------------------------------------

    def run(self) -> LayerSearch:
        if self.output.exists():
            value = load_layer_search(self.output, require_verified=True)
            if dict(value.bindings) != self.bindings:
                raise ValueError("sealed R10 search artifact provenance drift")
            return value

        experts = {
            expert: _base_expert(self.capture.layer, expert)
            for expert in range(NUM_EXPERTS)
        }
        shared_raw = self.progress.get("shared_result")
        if shared_raw is None:
            suh, svh, k_scales, n_scales, record = self._choose_shared_proxy(experts)
            self.progress["shared_result"] = record
            self._flush()
            score = float(record["selection_score"])
            draw = int(record["draw"])
        else:
            if not isinstance(shared_raw, Mapping):
                raise ValueError("shared R10 progress record is malformed")
            suh, svh, k_scales, n_scales, score, draw = self._restore_shared(shared_raw)

        base = self._layer_search(
            shared_suh=suh,
            shared_svh=svh,
            shared_k_scales=k_scales,
            shared_n_scales=n_scales,
            experts=experts,
            shared_draw=draw,
            shared_score=score,
        )
        results = self.progress["expert_results"]
        if not isinstance(results, dict):
            raise ValueError("R10 expert progress map is malformed")
        for expert in range(NUM_EXPERTS):
            raw = results.get(str(expert))
            if raw is None:
                chosen, record = self._choose_expert_proxy(
                    base, expert, self._prepare_proxy(expert)
                )
                results[str(expert)] = record
                self._flush()
            else:
                if not isinstance(raw, Mapping):
                    raise ValueError(f"expert {expert}: malformed progress record")
                chosen = self._restore_expert(expert, raw)
            experts[expert] = chosen
            base = replace(base, experts=dict(experts))
            self._proxies.pop(expert, None)
            clear = getattr(self.backend, "clear_expert_row_memo", None)
            if clear is not None:
                clear(self.capture, expert)

        evidence = sha256_file(self.progress_path)
        final = replace(
            base,
            experts=experts,
            pilot_evidence_sha256=evidence,
            # "verified" here means the deterministic search evidence is
            # complete and provenance-bound.  It does not claim a redundant
            # search-only TRELLIS round trip.
            unverified=False,
        )
        write_layer_search(self.output, self._payload(final))
        return load_layer_search(self.output, require_verified=True)
