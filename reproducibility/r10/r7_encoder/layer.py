"""Concrete per-layer fixed-point probe and final encoding logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .allocation import allocation_to_json
from .backend import ExpertWeights, LayerCapture, Round7Backend
from .constants import (
    ALLOWED_BITS,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    TensorId,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .expert_cache import load_cached_expert, write_cached_expert
from .hessian import FullCovarianceAccumulator, down_inputs_from_roundtrip
from .permutation import functional_oracle, permute_expert_hf
from .rotations import FoldedBlockScale, fold_block_g_scale
from .row_cache import (
    load_covariance_cache,
    load_holdout_cache,
    write_covariance_cache,
    write_holdout_cache,
)
from .search_artifact import LayerSearch
from .sensitivity import FixedPointController, ProbeLedger
from .trellis import Exl3TrellisCodec
from .types import CandidateLoss, EncodedTensor, LayerAllocation, StateShard


@dataclass(frozen=True)
class LayerEncodeResult:
    allocation: LayerAllocation
    encoded: tuple[EncodedTensor, ...]
    shared_gate_up_suh: object
    shared_down_svh: object
    fixed_point_iterations: int
    final_gate_up_sha256: Mapping[int, str]
    interaction_audit: Mapping[int, Mapping[str, object]]
    final_expert_artifacts: Mapping[int, str]
    permutation_audit: Mapping[int, Mapping[str, object]]
    probe_artifacts: Mapping[str, str]
    install_audit_sha256: str


@dataclass(frozen=True)
class CovarianceUse:
    matrix: object
    rows: int
    cold_fallback: bool
    routed_rows: int
    fallback_rows: int
    shrinkage_alpha: float
    row_ids: tuple[int, ...]
    row_ids_sha256: str
    fallback_row_ids_sha256: str


@dataclass(frozen=True)
class HoldoutUse:
    hidden: object
    row_ids: tuple[int, ...]
    row_ids_sha256: str
    routed_rows: int
    fallback_rows: int


class LayerProcessor:
    def __init__(
        self,
        *,
        backend: Round7Backend,
        codec: Exl3TrellisCodec,
        work_dir: str | Path,
        device: str,
        codecs: "Sequence[Exl3TrellisCodec] | None" = None,
        sigma_reg: float,
        source_inventory_sha256: str,
        numeric_environment_sha256: str,
        runtime_inventory_sha256: str,
        fixed_point_iterations: int = 4,
        holdout_rows: int = 4096,
        min_fit_rows: int = 1024,
        interaction_relative_tolerance: float = 0.25,
        interaction_absolute_tolerance: float = 1e-4,
        process_pool=None,
    ) -> None:
        self.backend = backend
        self.codec = codec
        self.work_dir = Path(work_dir)
        self.device = device
        # Device pool for the parallel warm passes. codecs[0] must be the
        # primary codec; the sequential authoritative loops below always run
        # against self.codec/self.device, so a pool of size 1 is byte-identical
        # to the original single-device flow.
        self.codecs: tuple[Exl3TrellisCodec, ...] = (
            tuple(codecs) if codecs else (codec,)
        )
        if self.codecs[0] is not codec:
            raise ValueError("codecs[0] must be the primary codec")
        self.sigma_reg = sigma_reg
        self.fixed_point_iterations = fixed_point_iterations
        self.holdout_rows = holdout_rows
        self.min_fit_rows = min_fit_rows
        self.source_inventory_sha256 = source_inventory_sha256
        self.numeric_environment_sha256 = numeric_environment_sha256
        self.runtime_inventory_sha256 = runtime_inventory_sha256
        self.interaction_relative_tolerance = interaction_relative_tolerance
        self.interaction_absolute_tolerance = interaction_absolute_tolerance
        self.process_pool = process_pool
        # Layer-parallel mode: each layer is encoded independently and there is
        # no successor forward, so staging encoded experts back into a live
        # model is dead work. Installing is Gap-2 machinery only.
        self.install_for_successor = True
        self.cold_audit: dict[str, dict[str, object]] = {}
        self._cold_audit_partial_path: Path | None = None
        self._cold_audit_bindings: dict[str, str] | None = None

    @staticmethod
    def _row_hash(row_ids: Iterable[int], role: str) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {"role": role, "row_ids": sorted(int(value) for value in row_ids)}
            )
        )

    @staticmethod
    def _select_new_rows(rows, seen: set[int], *, skip_seen: bool):
        import torch

        row_ids = (
            torch.as_tensor(rows.row_ids)
            .detach()
            .to("cpu", dtype=torch.int64)
            .flatten()
        )
        hidden = torch.as_tensor(rows.hidden)
        router_weight = torch.as_tensor(rows.router_weight).flatten()
        if (
            row_ids.numel() != hidden.shape[0]
            or router_weight.numel() != hidden.shape[0]
        ):
            raise ValueError("ExpertRows hidden/row_ids/router_weight length mismatch")
        local = row_ids.tolist()
        if len(set(local)) != len(local):
            raise ValueError("duplicate row ID within ExpertRows batch")
        keep = []
        accepted = []
        for row_id in local:
            duplicate = row_id in seen
            if duplicate and not skip_seen:
                raise ValueError(
                    f"duplicate row ID across expert-row batches: {row_id}"
                )
            keep.append(not duplicate)
            if not duplicate:
                seen.add(row_id)
                accepted.append(row_id)
        mask = torch.tensor(keep, dtype=torch.bool, device=hidden.device)
        return hidden[mask], tuple(accepted)

    def _flush_cold_audit_partial(self) -> None:
        if self._cold_audit_partial_path is None or self._cold_audit_bindings is None:
            return
        payload: dict[str, object] = {
            "schema": "r8-cold-fallback-partial-v1",
            "layer": int(self._cold_audit_bindings["layer"]),
            "bindings": dict(self._cold_audit_bindings),
            "records": [self.cold_audit[key] for key in sorted(self.cold_audit)],
        }
        payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
        atomic_write_json(self._cold_audit_partial_path, payload)

    def bind_cold_audit(self, capture: LayerCapture) -> None:
        """Open the crash-resumable parent journal for fallback evidence.

        Process workers return records to the coordinator; only the coordinator
        owns this file.  It is committed before any corresponding score/cache
        is published, so a resume cannot skip completed work and silently lose
        the evidence that its Hessian used cold fallback rows.
        """

        bindings = {
            "layer": str(capture.layer),
            "state_sha256": capture.state_sha256,
            "capture_sha256": capture.digest,
            "source_inventory_sha256": self.source_inventory_sha256,
            "numeric_environment_sha256": self.numeric_environment_sha256,
            "runtime_inventory_sha256": self.runtime_inventory_sha256,
            "backend_fingerprint": self.backend.fingerprint,
            "min_fit_rows": str(self.min_fit_rows),
        }
        path = (
            self.work_dir
            / f"layer-{capture.layer:03d}"
            / "cold-fallback-partial.json"
        )
        if self._cold_audit_partial_path == path:
            if self._cold_audit_bindings != bindings:
                raise ValueError("cold-fallback partial binding changed in memory")
            return
        if self._cold_audit_partial_path is not None:
            raise ValueError("LayerProcessor cannot bind cold audits across layers")
        self._cold_audit_partial_path = path
        self._cold_audit_bindings = bindings
        if not path.exists():
            return
        prior = read_json(path)
        content_sha256 = prior.pop("content_sha256", None)
        if content_sha256 != sha256_bytes(canonical_json_bytes(prior)):
            raise ValueError("cold-fallback partial content hash drift")
        if (
            prior.get("schema") != "r8-cold-fallback-partial-v1"
            or int(prior.get("layer", -1)) != capture.layer
            or prior.get("bindings") != bindings
            or not isinstance(prior.get("records"), list)
        ):
            raise ValueError("cold-fallback partial identity/binding drift")
        for raw in prior["records"]:
            if not isinstance(raw, Mapping):
                raise ValueError("cold-fallback partial record is malformed")
            self._record_cold(dict(raw), persist=False)

    def _record_cold(
        self, record: dict[str, object], *, persist: bool = True
    ) -> bool:
        if self._cold_audit_bindings is not None and int(record.get("layer", -1)) != int(
            self._cold_audit_bindings["layer"]
        ):
            raise ValueError("cold-fallback record belongs to another layer")
        key = sha256_bytes(canonical_json_bytes(record))
        incumbent = self.cold_audit.get(key)
        if incumbent is not None and incumbent != record:
            raise ValueError("cold-fallback audit hash collision")
        if incumbent is not None:
            return False
        self.cold_audit[key] = record
        if persist:
            self._flush_cold_audit_partial()
        return True

    def _row_cache_root(self, capture: LayerCapture) -> Path:
        return self.work_dir / f"layer-{capture.layer:03d}" / "row-plans"

    def _covariance_bindings(
        self, capture: LayerCapture, expert: int, split: str
    ) -> dict[str, str]:
        return {
            "capture_sha256": capture.digest,
            "state_sha256": capture.state_sha256,
            "expert": str(expert),
            "split": split,
            "min_fit_rows": str(self.min_fit_rows),
            "row_weighting": "unweighted-routed-membership",
            "fallback_policy": "four-route-weight-strata-seeded-reservoir-v1",
        }

    def _load_covariance_use(
        self, capture: LayerCapture, expert: int, split: str
    ) -> CovarianceUse | None:
        if split != "fit":
            return None
        value = load_covariance_cache(
            self._row_cache_root(capture),
            expert=expert,
            bindings=self._covariance_bindings(capture, expert, split),
            device=self.device,
        )
        if value is None:
            return None
        metadata = value["metadata"]
        row_ids = value["row_ids"]
        fallback_ids = value["fallback_row_ids"]
        if metadata["row_ids_sha256"] != self._row_hash(row_ids, f"{split}:combined"):
            raise ValueError("cached covariance row-ID hash drift")
        if metadata["fallback_row_ids_sha256"] != self._row_hash(
            fallback_ids, f"{split}:fallback"
        ):
            raise ValueError("cached fallback row-ID hash drift")
        result = CovarianceUse(
            value["matrix"],
            int(metadata["rows"]),
            bool(metadata["cold_fallback"]),
            int(metadata["routed_rows"]),
            int(metadata["fallback_rows"]),
            float(metadata["shrinkage_alpha"]),
            row_ids,
            str(metadata["row_ids_sha256"]),
            str(metadata["fallback_row_ids_sha256"]),
        )
        if result.cold_fallback:
            self._record_cold(dict(metadata["cold_audit"]))
        return result

    def _seal_covariance_use(
        self,
        capture: LayerCapture,
        expert: int,
        split: str,
        value: CovarianceUse,
        fallback_row_ids: Iterable[int],
        cold_record: Mapping[str, object] | None,
    ) -> None:
        if split != "fit":
            return
        write_covariance_cache(
            self._row_cache_root(capture),
            expert=expert,
            matrix=value.matrix,
            row_ids=value.row_ids,
            fallback_row_ids=tuple(fallback_row_ids),
            bindings=self._covariance_bindings(capture, expert, split),
            metadata={
                "rows": value.rows,
                "cold_fallback": value.cold_fallback,
                "routed_rows": value.routed_rows,
                "fallback_rows": value.fallback_rows,
                "shrinkage_alpha": format(value.shrinkage_alpha, ".17g"),
                "row_ids_sha256": value.row_ids_sha256,
                "fallback_row_ids_sha256": value.fallback_row_ids_sha256,
                "cold_audit": dict(cold_record or {}),
            },
        )

    def _covariance(
        self,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        split: str,
    ):
        cached = self._load_covariance_use(capture, expert, split)
        if cached is not None:
            return cached
        accumulator = FullCovarianceAccumulator(HIDDEN_SIZE, device=self.device)
        routed_ids: set[int] = set()
        for rows in self.backend.iter_expert_rows(
            capture=capture, shards=shards, expert=expert, split=split
        ):
            hidden, _ = self._select_new_rows(rows, routed_ids, skip_seen=False)
            if hidden.shape[0]:
                # Hessian policy is deliberately unweighted routed membership;
                # router mass enters only the rate allocator.
                accumulator.add(hidden)
        routed_rows = accumulator.rows
        if routed_rows >= self.min_fit_rows:
            result = accumulator.finalize(self.sigma_reg, add_damping=False)
            row_ids = tuple(sorted(routed_ids))
            value = CovarianceUse(
                result.matrix,
                result.rows,
                False,
                routed_rows,
                0,
                0.0,
                row_ids,
                self._row_hash(row_ids, f"{split}:combined"),
                self._row_hash((), f"{split}:fallback"),
            )
            self._seal_covariance_use(capture, expert, split, value, (), None)
            return value

        fallback = FullCovarianceAccumulator(HIDDEN_SIZE, device=self.device)
        fallback_ids: set[int] = set(routed_ids)
        accepted_fallback: list[int] = []
        remaining_fallback = self.min_fit_rows - routed_rows
        for rows in self.backend.iter_cold_fallback_rows(
            capture=capture, shards=shards, expert=expert, split=split
        ):
            if remaining_fallback <= 0:
                break
            hidden, accepted = self._select_new_rows(rows, fallback_ids, skip_seen=True)
            if hidden.shape[0]:
                value = hidden[:remaining_fallback]
                selected = accepted[: int(value.shape[0])]
                fallback.add(value)
                accepted_fallback.extend(selected)
                remaining_fallback -= int(value.shape[0])
        if remaining_fallback > 0:
            raise ValueError(
                f"L{capture.layer} E{expert}: cold fit fallback underfilled by "
                f"{remaining_fallback} rows"
            )
        fallback_result = fallback.finalize(self.sigma_reg, add_damping=False)
        if routed_rows:
            routed_result = accumulator.finalize(self.sigma_reg, add_damping=False)
            alpha = max(
                0.0, min(1.0, (self.min_fit_rows - routed_rows) / self.min_fit_rows)
            )
            matrix = routed_result.matrix.mul(1.0 - alpha).add(
                fallback_result.matrix, alpha=alpha
            )
        else:
            alpha = 1.0
            matrix = fallback_result.matrix
        row_ids = tuple(sorted((*routed_ids, *accepted_fallback)))
        fallback_hash = self._row_hash(accepted_fallback, f"{split}:fallback")
        cold_record = {
            "layer": capture.layer,
            "expert": expert,
            "projection_class": "gate_up",
            "routed_rows": routed_rows,
            "fallback_rows": fallback_result.rows,
            "shrinkage_alpha": format(alpha, ".17g"),
            "through_gate_up_roundtrip": False,
            "row_ids_sha256": self._row_hash(row_ids, f"{split}:combined"),
            "fallback_row_ids_sha256": fallback_hash,
            "hessian_row_weighting": "unweighted_routed_membership",
        }
        self._record_cold(cold_record)
        value = CovarianceUse(
            matrix,
            routed_rows + fallback_result.rows,
            True,
            routed_rows,
            fallback_result.rows,
            alpha,
            row_ids,
            self._row_hash(row_ids, f"{split}:combined"),
            fallback_hash,
        )
        self._seal_covariance_use(
            capture,
            expert,
            split,
            value,
            accepted_fallback,
            cold_record,
        )
        return value

    def _holdout(
        self,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        exclude_ids: Iterable[int] = (),
    ) -> HoldoutUse:
        import torch

        excluded = tuple(sorted(int(value) for value in exclude_ids))
        bindings = {
            "capture_sha256": capture.digest,
            "state_sha256": capture.state_sha256,
            "expert": str(expert),
            "holdout_rows": str(self.holdout_rows),
            "excluded_fit_row_ids_sha256": self._row_hash(excluded, "fit:excluded"),
            "split_policy": "seeded-80-20-with-cold-topup-v1",
        }
        cached = load_holdout_cache(
            self._row_cache_root(capture),
            expert=expert,
            bindings=bindings,
            device=self.device,
        )
        if cached is not None:
            row_ids = cached["row_ids"]
            metadata = cached["metadata"]
            if (
                self._row_hash(row_ids, "holdout:combined")
                != metadata["row_ids_sha256"]
            ):
                raise ValueError("cached holdout row-ID hash drift")
            if set(row_ids) & set(excluded):
                raise ValueError("cached holdout overlaps fit rows")
            return HoldoutUse(
                hidden=cached["hidden"],
                row_ids=row_ids,
                row_ids_sha256=str(metadata["row_ids_sha256"]),
                routed_rows=int(metadata["routed_rows"]),
                fallback_rows=int(metadata["fallback_rows"]),
            )
        xs = []
        seen = set(excluded)
        accepted_ids: list[int] = []
        remaining = self.holdout_rows
        for rows in self.backend.iter_expert_rows(
            capture=capture, shards=shards, expert=expert, split="holdout"
        ):
            if remaining <= 0:
                break
            hidden, accepted = self._select_new_rows(rows, seen, skip_seen=False)
            value = hidden.detach().cpu()[:remaining]
            xs.append(value)
            accepted_ids.extend(accepted[: int(value.shape[0])])
            remaining -= int(value.shape[0])
        routed_rows = len(accepted_ids)
        fallback_rows = 0
        if remaining > 0:
            for rows in self.backend.iter_cold_fallback_rows(
                capture=capture, shards=shards, expert=expert, split="holdout"
            ):
                if remaining <= 0:
                    break
                hidden, accepted = self._select_new_rows(rows, seen, skip_seen=True)
                value = hidden.detach().cpu()[:remaining]
                xs.append(value)
                selected = accepted[: int(value.shape[0])]
                accepted_ids.extend(selected)
                fallback_rows += len(selected)
                remaining -= int(value.shape[0])
        if not xs or remaining > 0:
            raise ValueError(
                f"L{capture.layer} E{expert}: empty routed and fallback holdout"
            )
        row_ids = tuple(accepted_ids)
        result = HoldoutUse(
            hidden=torch.cat(xs, dim=0).to(self.device),
            row_ids=row_ids,
            row_ids_sha256=self._row_hash(row_ids, "holdout:combined"),
            routed_rows=routed_rows,
            fallback_rows=fallback_rows,
        )
        write_holdout_cache(
            self._row_cache_root(capture),
            expert=expert,
            hidden=result.hidden,
            row_ids=result.row_ids,
            bindings=bindings,
            metadata={
                "row_ids_sha256": result.row_ids_sha256,
                "routed_rows": result.routed_rows,
                "fallback_rows": result.fallback_rows,
            },
        )
        return result

    def _vectors(self, search: LayerSearch, expert: int):
        per_expert = search.experts[expert]
        gate = fold_block_g_scale(
            search.gate_up_suh,
            per_expert.gate_svh,
            search.gate_up_k_g_scale,
            per_expert.gate_n_g_scale,
        )
        up = fold_block_g_scale(
            search.gate_up_suh,
            per_expert.up_svh,
            search.gate_up_k_g_scale,
            per_expert.up_n_g_scale,
        )
        down = fold_block_g_scale(
            per_expert.down_suh,
            search.down_svh,
            per_expert.down_k_g_scale,
            search.down_n_g_scale,
        )
        # Keep v31's MCG codebook normalization while replacing its scalar GSS
        # with the selected per-128 separable factors.
        codebook_factor = -self.codec.codebook_scale

        def apply_codebook(value: FoldedBlockScale) -> FoldedBlockScale:
            return FoldedBlockScale(
                suh=tuple(item / codebook_factor for item in value.suh),
                svh=value.svh,
                row_multiplier=value.row_multiplier,
                column_multiplier=value.column_multiplier,
            )

        gate = apply_codebook(gate)
        up = apply_codebook(up)
        down = apply_codebook(down)
        # Shared sides must stay byte-identical after G-scale folding. The
        # artifact therefore owns the layer-shared block factors.
        if gate.suh != up.suh:
            raise AssertionError("shared gate/up folding drift")
        return gate, up, down

    def _encode_gate_up(
        self,
        *,
        layer: int,
        expert: int,
        weights: ExpertWeights,
        covariance,
        search: LayerSearch,
        gate_bits: int,
        up_bits: int,
    ) -> tuple[EncodedTensor, EncodedTensor, object]:
        per_expert = search.experts[expert]
        gate_weight, up_weight, down_weight = permute_expert_hf(
            weights.gate_hf, weights.up_hf, weights.down_hf, per_expert.permutation
        )
        gate_vectors, up_vectors, _ = self._vectors(search, expert)
        gate = self.codec.encode(
            tensor_id=TensorId(layer, expert, "gate_proj"),
            weight_hf=gate_weight,
            covariance=covariance,
            bits=gate_bits,
            suh=gate_vectors.suh,
            svh=gate_vectors.svh,
            sigma_reg=self.sigma_reg,
            provenance={"bf16_sha256": weights.payload_sha256["gate_proj"]},
        )
        up = self.codec.encode(
            tensor_id=TensorId(layer, expert, "up_proj"),
            weight_hf=up_weight,
            covariance=covariance,
            bits=up_bits,
            suh=up_vectors.suh,
            svh=up_vectors.svh,
            sigma_reg=self.sigma_reg,
            provenance={"bf16_sha256": weights.payload_sha256["up_proj"]},
        )
        return gate, up, down_weight

    def _down_covariance(
        self,
        *,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        gate: EncodedTensor,
        up: EncodedTensor,
        expected_row_ids: tuple[int, ...] | None = None,
    ):
        accumulator = FullCovarianceAccumulator(INTERMEDIATE_SIZE, device=self.device)
        routed_ids: set[int] = set()
        # Hoisted once: these are [6144,2048] fp32 (~50 MB each). Moving them
        # per batch cost 2 H2D + 2 D2H copies for every one of ~1,773 sidecars
        # and then ran the SwiGLU on the HOST anyway, because
        # down_inputs_from_roundtrip binds to x.device. Running it on the
        # encode device also removes the CPU intra-op thread-count dependence
        # that made the down Hessian (and therefore the LDLQ trajectory)
        # irreproducible between thread-mismatched workers and the coordinator.
        gate_rt_dev = gate.reconstructed_kn.to(self.device)
        up_rt_dev = up.reconstructed_kn.to(self.device)
        for rows in self.backend.iter_expert_rows(
            capture=capture, shards=shards, expert=expert, split="fit"
        ):
            hidden, _ = self._select_new_rows(rows, routed_ids, skip_seen=False)
            inputs = down_inputs_from_roundtrip(
                hidden.to(self.device), gate_rt_dev, up_rt_dev
            )
            if inputs.shape[0]:
                accumulator.add(inputs)
        routed_rows = accumulator.rows
        fallback_rows = 0
        alpha = 0.0
        if routed_rows >= self.min_fit_rows:
            result = accumulator.finalize(self.sigma_reg, add_damping=False)
            matrix = result.matrix
            total_rows = result.rows
        else:
            fallback = FullCovarianceAccumulator(INTERMEDIATE_SIZE, device=self.device)
            fallback_seen: set[int] = set(routed_ids)
            accepted_fallback: list[int] = []
            remaining_fallback = self.min_fit_rows - routed_rows
            for rows in self.backend.iter_cold_fallback_rows(
                capture=capture, shards=shards, expert=expert, split="fit"
            ):
                if remaining_fallback <= 0:
                    break
                hidden, accepted = self._select_new_rows(
                    rows, fallback_seen, skip_seen=True
                )
                inputs = down_inputs_from_roundtrip(
                    hidden.to(self.device), gate_rt_dev, up_rt_dev
                )
                if inputs.shape[0]:
                    value = inputs[:remaining_fallback]
                    selected = accepted[: int(value.shape[0])]
                    fallback.add(value)
                    accepted_fallback.extend(selected)
                    remaining_fallback -= int(value.shape[0])
            if remaining_fallback > 0:
                raise ValueError(
                    f"L{capture.layer} E{expert}: cold down fallback underfilled by "
                    f"{remaining_fallback} rows"
                )
            fallback_result = fallback.finalize(self.sigma_reg, add_damping=False)
            fallback_rows = fallback_result.rows
            if routed_rows:
                routed_result = accumulator.finalize(self.sigma_reg, add_damping=False)
                alpha = max(
                    0.0,
                    min(1.0, (self.min_fit_rows - routed_rows) / self.min_fit_rows),
                )
                matrix = routed_result.matrix.mul(1.0 - alpha).add(
                    fallback_result.matrix, alpha=alpha
                )
            else:
                alpha = 1.0
                matrix = fallback_result.matrix
            total_rows = routed_rows + fallback_rows
            self._record_cold(
                {
                    "layer": capture.layer,
                    "expert": expert,
                    "projection_class": "down",
                    "routed_rows": routed_rows,
                    "fallback_rows": fallback_rows,
                    "shrinkage_alpha": format(alpha, ".17g"),
                    "through_gate_up_roundtrip": True,
                    "gate_sha256": gate.reconstruction_sha256,
                    "up_sha256": up.reconstruction_sha256,
                    "row_ids_sha256": self._row_hash(
                        (*routed_ids, *accepted_fallback), "down:combined"
                    ),
                    "fallback_row_ids_sha256": self._row_hash(
                        accepted_fallback, "down:fallback"
                    ),
                    "hessian_row_weighting": "unweighted_routed_membership",
                }
            )
        if routed_rows >= self.min_fit_rows:
            accepted_fallback = []
        all_row_ids = tuple(sorted((*routed_ids, *accepted_fallback)))
        if expected_row_ids is not None and all_row_ids != tuple(
            sorted(expected_row_ids)
        ):
            raise ValueError("gate/up and down covariance row plans differ")
        provenance = sha256_bytes(
            canonical_json_bytes(
                {
                    "gate": gate.reconstruction_sha256,
                    "up": up.reconstruction_sha256,
                    "rows": total_rows,
                    "routed_rows": routed_rows,
                    "fallback_rows": fallback_rows,
                    "shrinkage_alpha": format(alpha, ".17g"),
                }
            )
        )
        return CovarianceUse(
            matrix,
            total_rows,
            routed_rows < self.min_fit_rows,
            routed_rows,
            fallback_rows,
            alpha,
            all_row_ids,
            self._row_hash(all_row_ids, "down:combined"),
            self._row_hash(accepted_fallback, "down:fallback"),
        ), provenance

    def _encode_down(
        self,
        *,
        layer: int,
        expert: int,
        weight_hf,
        covariance,
        bits: int,
        search: LayerSearch,
        gate_up_provenance: str,
        bf16_sha256: str,
    ) -> EncodedTensor:
        _, _, vectors = self._vectors(search, expert)
        return self.codec.encode(
            tensor_id=TensorId(layer, expert, "down_proj"),
            weight_hf=weight_hf,
            covariance=covariance,
            bits=bits,
            suh=vectors.suh,
            svh=vectors.svh,
            sigma_reg=self.sigma_reg,
            provenance={
                "bf16_sha256": bf16_sha256,
                "gate_up_roundtrip_sha256": gate_up_provenance,
                "joint_full_k": INTERMEDIATE_SIZE,
            },
        )

    @staticmethod
    def _expert_output(x, gate_kn, up_kn, down_kn):
        import torch.nn.functional as functional

        return (functional.silu(x @ gate_kn) * (x @ up_kn)) @ down_kn

    def _loss(
        self, x, reference_weights: tuple[object, object, object], gate, up, down
    ) -> float:
        import torch

        gate_hf, up_hf, down_hf = reference_weights
        reference = self._expert_output(
            x,
            torch.as_tensor(gate_hf, device=x.device, dtype=x.dtype).T,
            torch.as_tensor(up_hf, device=x.device, dtype=x.dtype).T,
            torch.as_tensor(down_hf, device=x.device, dtype=x.dtype).T,
        )
        candidate = self._expert_output(
            x,
            gate.reconstructed_kn.to(x.device, dtype=x.dtype),
            up.reconstructed_kn.to(x.device, dtype=x.dtype),
            down.reconstructed_kn.to(x.device, dtype=x.dtype),
        )
        denominator = reference.double().square().sum().clamp_min(1e-30)
        return float(
            (
                (reference.double() - candidate.double()).square().sum() / denominator
            ).item()
        )

    def _probe_expert(
        self,
        *,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        search: LayerSearch,
        expert: int,
        context_bits: Mapping[str, int],
        ledger: ProbeLedger,
        fixed_point_iteration: int,
        search_artifact_sha256: str,
    ) -> None:
        layer = capture.layer
        candidate_ids = {
            projection: TensorId(layer, expert, projection)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if all(
            ledger.has(candidate_ids[projection], bits)
            for projection in candidate_ids
            for bits in ALLOWED_BITS
        ):
            return
        weights = self.backend.load_bf16_expert(layer=layer, expert=expert)
        weights.validate_bf16(layer, expert)
        per_expert = search.experts[expert]
        reference_weights = permute_expert_hf(
            weights.gate_hf, weights.up_hf, weights.down_hf, per_expert.permutation
        )
        gu_use = self._covariance(capture, shards, expert, "fit")
        gu_covariance = gu_use.matrix
        holdout = self._holdout(capture, shards, expert, exclude_ids=gu_use.row_ids)
        mass = str(capture.mass_audit.mass_by_expert[expert])
        base_gate_bits = context_bits[TensorId(layer, expert, "gate_proj").key]
        base_up_bits = context_bits[TensorId(layer, expert, "up_proj").key]
        base_down_bits = context_bits[TensorId(layer, expert, "down_proj").key]
        context_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "gate_proj": base_gate_bits,
                    "up_proj": base_up_bits,
                    "down_proj": base_down_bits,
                }
            )
        )
        permutation_hash = sha256_bytes(
            canonical_json_bytes(list(per_expert.permutation))
        )
        gate_vectors, up_vectors, down_vectors = self._vectors(search, expert)
        vector_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "gate_suh": gate_vectors.suh,
                    "gate_svh": gate_vectors.svh,
                    "up_suh": up_vectors.suh,
                    "up_svh": up_vectors.svh,
                    "down_suh": down_vectors.suh,
                    "down_svh": down_vectors.svh,
                }
            )
        )

        gate_weight, up_weight, down_weight = reference_weights
        gate_cache: dict[int, EncodedTensor] = {}
        up_cache: dict[int, EncodedTensor] = {}
        pair_cache = {}

        def gate_at(bits: int) -> EncodedTensor:
            if bits not in gate_cache:
                gate_cache[bits] = self.codec.encode(
                    tensor_id=TensorId(layer, expert, "gate_proj"),
                    weight_hf=gate_weight,
                    covariance=gu_covariance,
                    bits=bits,
                    suh=gate_vectors.suh,
                    svh=gate_vectors.svh,
                    sigma_reg=self.sigma_reg,
                    provenance={"bf16_sha256": weights.payload_sha256["gate_proj"]},
                )
            return gate_cache[bits]

        def up_at(bits: int) -> EncodedTensor:
            if bits not in up_cache:
                up_cache[bits] = self.codec.encode(
                    tensor_id=TensorId(layer, expert, "up_proj"),
                    weight_hf=up_weight,
                    covariance=gu_covariance,
                    bits=bits,
                    suh=up_vectors.suh,
                    svh=up_vectors.svh,
                    sigma_reg=self.sigma_reg,
                    provenance={"bf16_sha256": weights.payload_sha256["up_proj"]},
                )
            return up_cache[bits]

        def pair_at(gate_bits: int, up_bits: int):
            key = (gate_bits, up_bits)
            if key not in pair_cache:
                gate = gate_at(gate_bits)
                up = up_at(up_bits)
                down_covariance, gu_hash = self._down_covariance(
                    capture=capture,
                    shards=shards,
                    expert=expert,
                    gate=gate,
                    up=up,
                    expected_row_ids=gu_use.row_ids,
                )
                down = self._encode_down(
                    layer=layer,
                    expert=expert,
                    weight_hf=down_weight,
                    covariance=down_covariance.matrix,
                    bits=base_down_bits,
                    search=search,
                    gate_up_provenance=gu_hash,
                    bf16_sha256=weights.payload_sha256["down_proj"],
                )
                pair_cache[key] = (gate, up, down_covariance, gu_hash, down)
            return pair_cache[key]

        # Five distinct gate/up contexts cover all six conditional curves; the
        # shared (base_gate, base_up) context is evaluated only once. Gate and
        # up encodes are independently memoized by bit width. Every candidate
        # still receives a real full-expert round-trip loss.
        for projection in ("gate_proj", "up_proj"):
            for bits in ALLOWED_BITS:
                candidate_id = TensorId(layer, expert, projection)
                if ledger.has(candidate_id, bits):
                    continue
                gate_bits = bits if projection == "gate_proj" else base_gate_bits
                up_bits = bits if projection == "up_proj" else base_up_bits
                gate, up, down_covariance, _, down = pair_at(gate_bits, up_bits)
                loss = self._loss(holdout.hidden, reference_weights, gate, up, down)
                selected = gate if projection == "gate_proj" else up
                ledger.add(
                    CandidateLoss(
                        tensor_id=candidate_id,
                        bits=bits,
                        loss=format(loss, ".17g"),
                        mass=mass,
                        fit_rows=gu_use.rows,
                        holdout_rows=int(holdout.hidden.shape[0]),
                        roundtrip_sha256=selected.reconstruction_sha256,
                        fit_row_ids_sha256=gu_use.row_ids_sha256,
                        holdout_row_ids_sha256=holdout.row_ids_sha256,
                        used_cold_fallback=(
                            gu_use.cold_fallback or down_covariance.cold_fallback
                        ),
                        fixed_point_iteration=fixed_point_iteration,
                        context_bits_sha256=context_hash,
                        expert_roundtrip_sha256={
                            "gate_proj": gate.reconstruction_sha256,
                            "up_proj": up.reconstruction_sha256,
                            "down_proj": down.reconstruction_sha256,
                        },
                        state_sha256=capture.state_sha256,
                        capture_sha256=capture.digest,
                        search_sha256=search_artifact_sha256,
                        source_inventory_sha256=self.source_inventory_sha256,
                        numeric_environment_sha256=self.numeric_environment_sha256,
                        runtime_inventory_sha256=self.runtime_inventory_sha256,
                        backend_fingerprint=self.backend.fingerprint,
                        permutation_sha256=permutation_hash,
                        vector_bundle_sha256=vector_hash,
                    )
                )

        # Down candidates all use the one memoized context gate/up round trip.
        gate, up, down_covariance, gu_hash, context_down = pair_at(
            base_gate_bits, base_up_bits
        )
        for bits in ALLOWED_BITS:
            down_id = TensorId(layer, expert, "down_proj")
            if ledger.has(down_id, bits):
                continue
            down = (
                context_down
                if bits == base_down_bits
                else self._encode_down(
                    layer=layer,
                    expert=expert,
                    weight_hf=down_weight,
                    covariance=down_covariance.matrix,
                    bits=bits,
                    search=search,
                    gate_up_provenance=gu_hash,
                    bf16_sha256=weights.payload_sha256["down_proj"],
                )
            )
            loss = self._loss(holdout.hidden, reference_weights, gate, up, down)
            ledger.add(
                CandidateLoss(
                    tensor_id=down_id,
                    bits=bits,
                    loss=format(loss, ".17g"),
                    mass=mass,
                    fit_rows=down_covariance.rows,
                    holdout_rows=int(holdout.hidden.shape[0]),
                    roundtrip_sha256=down.reconstruction_sha256,
                    gate_up_roundtrip_sha256=gu_hash,
                    fit_row_ids_sha256=gu_use.row_ids_sha256,
                    holdout_row_ids_sha256=holdout.row_ids_sha256,
                    used_cold_fallback=(
                        gu_use.cold_fallback or down_covariance.cold_fallback
                    ),
                    fixed_point_iteration=fixed_point_iteration,
                    context_bits_sha256=context_hash,
                    expert_roundtrip_sha256={
                        "gate_proj": gate.reconstruction_sha256,
                        "up_proj": up.reconstruction_sha256,
                        "down_proj": down.reconstruction_sha256,
                    },
                    state_sha256=capture.state_sha256,
                    capture_sha256=capture.digest,
                    search_sha256=search_artifact_sha256,
                    source_inventory_sha256=self.source_inventory_sha256,
                    numeric_environment_sha256=self.numeric_environment_sha256,
                    runtime_inventory_sha256=self.runtime_inventory_sha256,
                    backend_fingerprint=self.backend.fingerprint,
                    permutation_sha256=permutation_hash,
                    vector_bundle_sha256=vector_hash,
                )
            )

    def _final_expert(
        self,
        *,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        search: LayerSearch,
        allocation: LayerAllocation,
        expert: int,
        search_artifact_sha256: str,
    ) -> tuple[
        EncodedTensor,
        EncodedTensor,
        EncodedTensor,
        str,
        float,
        str,
        Mapping[str, object],
    ]:
        layer = capture.layer
        weights = self.backend.load_bf16_expert(layer=layer, expert=expert)
        weights.validate_bf16(layer, expert)
        covariance_use = self._covariance(capture, shards, expert, "fit")
        covariance = covariance_use.matrix
        gate_id = TensorId(layer, expert, "gate_proj")
        up_id = TensorId(layer, expert, "up_proj")
        down_id = TensorId(layer, expert, "down_proj")
        gate, up, down_weight = self._encode_gate_up(
            layer=layer,
            expert=expert,
            weights=weights,
            covariance=covariance,
            search=search,
            gate_bits=allocation.bits[gate_id.key],
            up_bits=allocation.bits[up_id.key],
        )
        down_covariance, gu_hash = self._down_covariance(
            capture=capture,
            shards=shards,
            expert=expert,
            gate=gate,
            up=up,
            expected_row_ids=covariance_use.row_ids,
        )
        down = self._encode_down(
            layer=layer,
            expert=expert,
            weight_hf=down_weight,
            covariance=down_covariance.matrix,
            bits=allocation.bits[down_id.key],
            search=search,
            gate_up_provenance=gu_hash,
            bf16_sha256=weights.payload_sha256["down_proj"],
        )
        if down.provenance["gate_up_roundtrip_sha256"] != gu_hash:
            raise AssertionError("final down provenance drift")
        per_expert = search.experts[expert]
        reference = permute_expert_hf(
            weights.gate_hf,
            weights.up_hf,
            weights.down_hf,
            per_expert.permutation,
        )
        holdout = self._holdout(
            capture,
            shards,
            expert,
            exclude_ids=covariance_use.row_ids,
        )
        loss = self._loss(holdout.hidden, reference, gate, up, down)
        audit = asdict(
            functional_oracle(
                holdout.hidden[:16].detach().cpu().double(),
                weights.gate_hf.detach().cpu().double(),
                weights.up_hf.detach().cpu().double(),
                weights.down_hf.detach().cpu().double(),
                per_expert.permutation,
            )
        )
        audit["passed"] = bool(
            audit["exact_inverse"]
            and audit["exact_weight_roundtrip"]
            and float(audit["relative_function_error"]) <= 1e-12
        )
        if not audit["passed"]:
            raise AssertionError(f"L{layer} E{expert}: permutation oracle failed")
        permutation_hash = sha256_bytes(
            canonical_json_bytes(list(per_expert.permutation))
        )
        vector_hashes = {
            item.tensor_id.projection: sha256_bytes(
                canonical_json_bytes(
                    {
                        "suh": [float(value) for value in item.suh.tolist()],
                        "svh": [float(value) for value in item.svh.tolist()],
                    }
                )
            )
            for item in (gate, up, down)
        }
        enriched = []
        for item in (gate, up, down):
            provenance = dict(item.provenance)
            source_record = weights.source_records[item.tensor_id.projection]
            provenance.update(
                {
                    "source_name": weights.source_names[item.tensor_id.projection],
                    "source_shard": source_record["shard"],
                    "source_payload_start": source_record["payload_start"],
                    "source_payload_end": source_record["payload_end"],
                    "source_inventory_sha256": self.source_inventory_sha256,
                    "numeric_environment_sha256": self.numeric_environment_sha256,
                    "runtime_inventory_sha256": self.runtime_inventory_sha256,
                    "backend_fingerprint": self.backend.fingerprint,
                    "state_sha256": capture.state_sha256,
                    "capture_sha256": capture.digest,
                    "search_sha256": search_artifact_sha256,
                    "allocation_sha256": sha256_bytes(
                        canonical_json_bytes(dict(sorted(allocation.bits.items())))
                    ),
                    "probe_sha256": allocation.probe_sha256,
                    "fit_row_ids_sha256": covariance_use.row_ids_sha256,
                    "down_fit_row_ids_sha256": down_covariance.row_ids_sha256,
                    "holdout_row_ids_sha256": holdout.row_ids_sha256,
                    "permutation_sha256": permutation_hash,
                    "permutation_policy": per_expert.permutation_policy,
                    "vector_sha256": vector_hashes[item.tensor_id.projection],
                    "used_cold_fallback": (
                        covariance_use.cold_fallback or down_covariance.cold_fallback
                    ),
                }
            )
            enriched.append(replace(item, provenance=provenance))
        return (
            enriched[0],
            enriched[1],
            enriched[2],
            gu_hash,
            loss,
            holdout.row_ids_sha256,
            audit,
        )

    def _device_workers(self) -> "list[LayerProcessor]":
        """Shallow worker clones, one per pool codec, for the warm passes.

        Each clone shares every read-only collaborator (backend, caches on
        disk, thresholds) and overrides only the codec and its device, so all
        `self.codec` / `self.device` references inside the per-expert methods
        resolve to that worker's GPU. Workers never touch the ledger, the
        manifest, or any shared mutable state: warm passes only fill
        per-expert disk caches and return records for the caller to merge.
        """
        import copy

        workers = []
        for codec in self.codecs:
            worker = copy.copy(self)
            worker.codec = codec
            worker.device = codec.config.device
            worker.cold_audit = dict(self.cold_audit)
            worker._cold_audit_partial_path = None
            worker._cold_audit_bindings = None
            workers.append(worker)
        return workers

    class _CollectorLedger:
        """Ledger shim for warm probes: collects records, never skips work."""

        def __init__(self) -> None:
            self.records: list = []

        def has(self, *_args, **_kwargs) -> bool:
            return False

        def add(self, record) -> None:
            self.records.append(record)

    def _warm_experts_parallel(self, experts, worker_fn) -> list:
        """Run worker_fn(worker, expert) across the device pool; fail closed.

        Returns the per-expert results in ascending expert order regardless of
        completion order, so every downstream merge is deterministic. Any
        worker exception cancels the pool and propagates unchanged.
        """
        experts = sorted(experts)
        if not experts or len(self.codecs) <= 1:
            return []
        from concurrent.futures import ThreadPoolExecutor

        workers = self._device_workers()
        results: dict[int, object] = {}

        def _run(index_expert):
            index, expert = index_expert
            worker = workers[index % len(workers)]
            return expert, worker_fn(worker, expert)

        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            for expert, value in pool.map(_run, enumerate(experts)):
                results[expert] = value
        return [results[expert] for expert in sorted(results)]

    def _merge_worker_cold_audit(self, raw) -> None:
        """Merge worker-local fallback records without completion-order effects."""

        if not raw:
            return
        values = raw.values() if isinstance(raw, Mapping) else raw
        changed = False
        for item in values:
            record = dict(item)
            changed = self._record_cold(record, persist=False) or changed
        if changed:
            self._flush_cold_audit_partial()

    def run(
        self,
        *,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        search: LayerSearch,
        search_artifact_sha256: str,
    ) -> LayerEncodeResult:
        if search.layer != capture.layer or search.unverified:
            raise ValueError("layer search must be matching and verified")
        self.bind_cold_audit(capture)
        layer_dir = self.work_dir / f"layer-{capture.layer:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        context = {
            tensor_id.key: 4
            for expert in range(NUM_EXPERTS)
            for tensor_id in (
                TensorId(capture.layer, expert, "gate_proj"),
                TensorId(capture.layer, expert, "up_proj"),
                TensorId(capture.layer, expert, "down_proj"),
            )
        }
        controller = FixedPointController(self.fixed_point_iterations)
        allocation = None
        previous_ledger: ProbeLedger | None = None
        previous_context: dict[str, int] | None = None
        for iteration in range(self.fixed_point_iterations):
            context_hash = sha256_bytes(
                canonical_json_bytes(dict(sorted(context.items())))
            )
            ledger = ProbeLedger(
                layer_dir / f"probe-iter-{iteration:02d}.json",
                capture.layer,
                fixed_point_iteration=iteration,
                bindings={
                    "context_map_sha256": context_hash,
                    "state_sha256": capture.state_sha256,
                    "capture_sha256": capture.digest,
                    "search_sha256": search_artifact_sha256,
                    "source_inventory_sha256": self.source_inventory_sha256,
                    "numeric_environment_sha256": self.numeric_environment_sha256,
                    "runtime_inventory_sha256": self.runtime_inventory_sha256,
                    "backend_fingerprint": self.backend.fingerprint,
                },
            )
            # Parallel warm pass: run the probe encodes for every expert the
            # sequential loop below would compute fresh, sharded across the
            # device pool. Records merge into the ledger in canonical order,
            # so the authoritative loop then satisfies its ledger.has() skip
            # and remains byte-identical to the single-device flow.
            if self.process_pool is not None or len(self.codecs) > 1:
                _warm_todo = []
                for expert in range(NUM_EXPERTS):
                    keys = tuple(
                        TensorId(capture.layer, expert, projection).key
                        for projection in ("gate_proj", "up_proj", "down_proj")
                    )
                    reused = (
                        previous_ledger is not None
                        and previous_context is not None
                        and all(
                            context[key] == previous_context[key] for key in keys
                        )
                    )
                    complete = all(
                        ledger.has(TensorId(capture.layer, expert, projection), bits)
                        for projection in ("gate_proj", "up_proj", "down_proj")
                        for bits in ALLOWED_BITS
                    )
                    if not reused and not complete:
                        _warm_todo.append(expert)

                def _warm_probe(worker, expert):
                    collector = LayerProcessor._CollectorLedger()
                    worker._probe_expert(
                        capture=capture,
                        shards=shards,
                        search=search,
                        expert=expert,
                        context_bits=context,
                        ledger=collector,
                        fixed_point_iteration=iteration,
                        search_artifact_sha256=search_artifact_sha256,
                    )
                    return collector.records

                if self.process_pool is not None:
                    warmed = self.process_pool.map(
                        "probe",
                        _warm_todo,
                        {
                            "capture": capture,
                            "shards": shards,
                            "search": search,
                            "context_bits": context,
                            "fixed_point_iteration": iteration,
                            "search_artifact_sha256": search_artifact_sha256,
                        },
                    )
                    if {int(expert) for expert, _ in warmed} != set(_warm_todo):
                        raise RuntimeError("process probe result domain is incomplete")
                    for _, value in warmed:
                        self._merge_worker_cold_audit(value.get("cold_audit"))
                        ledger.add_many(value["records"])
                else:
                    for _records in self._warm_experts_parallel(
                        _warm_todo, _warm_probe
                    ):
                        ledger.add_many(_records)
            for expert in range(NUM_EXPERTS):
                expert_keys = tuple(
                    TensorId(capture.layer, expert, projection).key
                    for projection in ("gate_proj", "up_proj", "down_proj")
                )
                if (
                    previous_ledger is not None
                    and previous_context is not None
                    and all(
                        context[key] == previous_context[key] for key in expert_keys
                    )
                ):
                    for record in previous_ledger.records:
                        if record.tensor_id.expert == expert:
                            ledger.add(
                                replace(
                                    record,
                                    fixed_point_iteration=iteration,
                                )
                            )
                    continue
                self._probe_expert(
                    capture=capture,
                    shards=shards,
                    search=search,
                    expert=expert,
                    context_bits=context,
                    ledger=ledger,
                    fixed_point_iteration=iteration,
                    search_artifact_sha256=search_artifact_sha256,
                )
            allocation = ledger.solve(
                capture.mass_audit.mass_by_expert,
                fixed_point_iteration=iteration,
            )
            atomic_write_json(
                layer_dir / f"allocation-iter-{iteration:02d}.json",
                allocation_to_json(allocation),
            )
            if controller.observe(allocation):
                break
            previous_ledger = ledger
            previous_context = dict(context)
            context = dict(allocation.bits)
        else:
            raise RuntimeError("allocation fixed point did not converge")
        assert allocation is not None
        probe_artifacts = {
            path.name: sha256_file(path)
            for path in sorted(layer_dir.glob("probe-iter-*.json"))
        }
        probe_artifacts.update(
            {
                path.name: sha256_file(path)
                for path in sorted(layer_dir.glob("allocation-iter-*.json"))
            }
        )
        probe_artifacts.update(
            {
                f"row-plans/{path.name}": sha256_file(path)
                for path in sorted((layer_dir / "row-plans").glob("*"))
                if path.is_file()
            }
        )

        encoded: list[EncodedTensor] = []
        gu_hashes: dict[int, str] = {}
        interaction_audit: dict[int, dict[str, object]] = {}
        final_artifacts: dict[int, str] = {}
        permutation_audits: dict[int, Mapping[str, object]] = {}
        final_bindings = {
            "state_sha256": capture.state_sha256,
            "capture_sha256": capture.digest,
            "search_sha256": search_artifact_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "numeric_environment_sha256": self.numeric_environment_sha256,
            "runtime_inventory_sha256": self.runtime_inventory_sha256,
            "backend_fingerprint": self.backend.fingerprint,
            "allocation_sha256": sha256_bytes(
                canonical_json_bytes(dict(sorted(allocation.bits.items())))
            ),
            "probe_sha256": allocation.probe_sha256,
            "fixed_point_iteration": str(allocation.fixed_point_iteration),
        }
        cache_root = layer_dir / "final-experts"
        interaction_partial_path = layer_dir / "interaction-partial.json"
        interaction_partial: dict[str, object] = {
            "schema": "r7-interaction-partial-v1",
            "bindings": final_bindings,
            "experts": {},
        }
        if interaction_partial_path.exists():
            incumbent = read_json(interaction_partial_path)
            if (
                incumbent.get("schema") != "r7-interaction-partial-v1"
                or incumbent.get("bindings") != final_bindings
            ):
                raise ValueError("interaction audit resume binding drift")
            interaction_partial = incumbent
        install_audit_path = layer_dir / "install-audit.json"
        install_audit: dict[str, object] = {
            "schema": "r7-layer-install-audit-v2",
            "layer": capture.layer,
            "bindings": final_bindings,
            "experts": {},
            "official_layer_audit": None,
            "complete": False,
        }
        if install_audit_path.exists():
            incumbent = read_json(install_audit_path)
            if (
                incumbent.get("schema") != "r7-layer-install-audit-v2"
                or incumbent.get("layer") != capture.layer
                or incumbent.get("bindings") != final_bindings
                or not isinstance(incumbent.get("experts"), dict)
            ):
                raise ValueError("install audit resume binding drift")
            install_audit = incumbent
        final_records = {
            (record.tensor_id.key, record.bits): record for record in ledger.records
        }
        # Parallel warm pass for the final encodes: fill the per-expert cache
        # pairs across the device pool, then let the unchanged sequential loop
        # below consume them as cache hits. Per-expert files make concurrent
        # writers collision-free; the pair writer's clean-transaction guard
        # would loudly refuse any accidental overlap.
        if self.process_pool is not None or len(self.codecs) > 1:
            _final_todo = []
            for expert in range(NUM_EXPERTS):
                cached_for_resume = load_cached_expert(
                    cache_root,
                    layer=capture.layer,
                    expert=expert,
                    bits=allocation.bits,
                    bindings=final_bindings,
                    codec=self.codec,
                    reconstruct=False,
                )
                if cached_for_resume is None:
                    _final_todo.append(expert)
                else:
                    # A worker publishes the authenticated cache before it can
                    # return to the coordinator. Its operation-local audit is
                    # part of that same transaction, so a crash in this interval
                    # cannot orphan fallback evidence on resume.
                    self._merge_worker_cold_audit(cached_for_resume.cold_audit)

            def _warm_final(worker, expert):
                cold_before = frozenset(worker.cold_audit)
                (
                    w_gate,
                    w_up,
                    w_down,
                    w_gu_hash,
                    w_final_loss,
                    w_holdout_hash,
                    w_perm_audit,
                ) = worker._final_expert(
                    capture=capture,
                    shards=shards,
                    search=search,
                    allocation=allocation,
                    expert=expert,
                    search_artifact_sha256=search_artifact_sha256,
                )
                cold_records = tuple(
                    worker.cold_audit[key]
                    for key in sorted(worker.cold_audit)
                    if key not in cold_before
                )
                write_cached_expert(
                    cache_root,
                    encoded=(w_gate, w_up, w_down),
                    bindings=final_bindings,
                    gate_up_sha256=w_gu_hash,
                    final_loss=w_final_loss,
                    holdout_row_ids_sha256=w_holdout_hash,
                    permutation_audit=w_perm_audit,
                    cold_audit=cold_records,
                )
                return expert

            if self.process_pool is not None:
                warmed = self.process_pool.map(
                    "final",
                    _final_todo,
                    {
                        "capture": capture,
                        "shards": shards,
                        "search": search,
                        "allocation": allocation,
                        "search_artifact_sha256": search_artifact_sha256,
                        "cache_root": cache_root,
                        "bindings": final_bindings,
                    },
                )
                if {int(expert) for expert, _ in warmed} != set(_final_todo):
                    raise RuntimeError("process final result domain is incomplete")
                for _, value in warmed:
                    self._merge_worker_cold_audit(value.get("cold_audit"))
            else:
                self._warm_experts_parallel(_final_todo, _warm_final)

        # The interaction gate requires a second joint encode with all three
        # tensors at the floor.  It was formerly hidden inside the canonical
        # install loop and serialized 256 full encodes on the primary GPU.
        # Warm only missing audit values in processes, then consume the scalar
        # losses below in canonical expert order.
        partial_experts = interaction_partial["experts"]
        assert isinstance(partial_experts, dict)
        floor_losses: dict[int, float] = {}
        if self.process_pool is not None:
            floor_todo = [
                expert
                for expert in range(NUM_EXPERTS)
                if str(expert) not in partial_experts
            ]
            warmed_floor = self.process_pool.map(
                "floor",
                floor_todo,
                {
                    "capture": capture,
                    "shards": shards,
                    "search": search,
                    "allocation": allocation,
                    "search_artifact_sha256": search_artifact_sha256,
                },
            )
            if {int(expert) for expert, _ in warmed_floor} != set(floor_todo):
                raise RuntimeError("process floor result domain is incomplete")
            for expert, value in warmed_floor:
                floor_losses[int(expert)] = float(value["loss"])
                self._merge_worker_cold_audit(value.get("cold_audit"))
        for expert in range(NUM_EXPERTS):
            cached = load_cached_expert(
                cache_root,
                layer=capture.layer,
                expert=expert,
                bits=allocation.bits,
                bindings=final_bindings,
                codec=self.codec,
                reconstruct=False,
            )
            if cached is not None:
                self._merge_worker_cold_audit(cached.cold_audit)
            if cached is None:
                (
                    gate,
                    up,
                    down,
                    gu_hash,
                    final_loss,
                    holdout_hash,
                    permutation_audit,
                ) = self._final_expert(
                    capture=capture,
                    shards=shards,
                    search=search,
                    allocation=allocation,
                    expert=expert,
                    search_artifact_sha256=search_artifact_sha256,
                )
                write_cached_expert(
                    cache_root,
                    encoded=(gate, up, down),
                    bindings=final_bindings,
                    gate_up_sha256=gu_hash,
                    final_loss=final_loss,
                    holdout_row_ids_sha256=holdout_hash,
                    permutation_audit=permutation_audit,
                )
                # The predecessor walk consumes a reconstruction decoded from
                # the just-published packed mini-shard, not the encoder's
                # transient dense object.
                cached = load_cached_expert(
                    cache_root,
                    layer=capture.layer,
                    expert=expert,
                    bits=allocation.bits,
                    bindings=final_bindings,
                    codec=self.codec,
                    reconstruct=False,
                )
                if cached is None:
                    raise AssertionError("published final expert cache vanished")
                self._merge_worker_cold_audit(cached.cold_audit)
                gate, up, down = cached.encoded
                gu_hash = cached.gate_up_sha256
                final_loss = cached.final_loss
                holdout_hash = cached.holdout_row_ids_sha256
                permutation_audit = cached.permutation_audit
            else:
                gate, up, down = cached.encoded
                gu_hash = cached.gate_up_sha256
                final_loss = cached.final_loss
                holdout_hash = cached.holdout_row_ids_sha256
                permutation_audit = cached.permutation_audit
            final_artifacts[expert] = sha256_file(
                cache_root / f"expert-{expert:03d}.json"
            )
            gu_hashes[expert] = gu_hash
            final_hashes = {
                "gate_proj": gate.reconstruction_sha256,
                "up_proj": up.reconstruction_sha256,
                "down_proj": down.reconstruction_sha256,
            }
            selected_records = [
                final_records[
                    (
                        TensorId(capture.layer, expert, projection).key,
                        allocation.bits[
                            TensorId(capture.layer, expert, projection).key
                        ],
                    )
                ]
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]
            for record in selected_records:
                if dict(record.expert_roundtrip_sha256) != final_hashes:
                    raise AssertionError(
                        "selected conditional probe does not reconstruct the final joint map"
                    )
                if record.holdout_row_ids_sha256 != holdout_hash:
                    raise AssertionError("final and probe holdout row plans differ")
            selected_losses = [float(record.loss) for record in selected_records]
            spread = max(selected_losses + [final_loss]) - min(
                selected_losses + [final_loss]
            )
            if spread > 1e-10:
                raise RuntimeError(
                    f"L{capture.layer} E{expert}: conditional interaction audit drift {spread}"
                )
            floor_records = [
                final_records[(TensorId(capture.layer, expert, projection).key, 3)]
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]
            predicted_gain = sum(
                float(record.loss) - final_loss for record in floor_records
            )
            prior_interaction = partial_experts.get(str(expert))
            if prior_interaction is None:
                if expert in floor_losses:
                    floor_loss = floor_losses[expert]
                else:
                    floor_map = dict(allocation.bits)
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        floor_map[TensorId(capture.layer, expert, projection).key] = 3
                    floor_allocation = replace(allocation, bits=floor_map)
                    *_, floor_loss, _, _ = self._final_expert(
                        capture=capture,
                        shards=shards,
                        search=search,
                        allocation=floor_allocation,
                        expert=expert,
                        search_artifact_sha256=search_artifact_sha256,
                    )
                actual_gain = floor_loss - final_loss
                prior_interaction = {
                    "floor_joint_loss": format(floor_loss, ".17g"),
                    "predicted_conditional_gain": format(predicted_gain, ".17g"),
                    "actual_joint_gain": format(actual_gain, ".17g"),
                }
                partial_experts[str(expert)] = prior_interaction
                atomic_write_json(interaction_partial_path, interaction_partial)
            else:
                if (
                    abs(
                        float(prior_interaction["predicted_conditional_gain"])
                        - predicted_gain
                    )
                    > 1e-15
                ):
                    raise ValueError("interaction prediction drift on resume")
                floor_loss = float(prior_interaction["floor_joint_loss"])
                actual_gain = float(prior_interaction["actual_joint_gain"])
            interaction_audit[expert] = {
                "final_loss": format(final_loss, ".17g"),
                "selected_conditional_losses": [
                    format(value, ".17g") for value in selected_losses
                ],
                "max_absolute_spread": format(spread, ".17g"),
                "roundtrip_sha256": final_hashes,
                "holdout_row_ids_sha256": holdout_hash,
                "floor_joint_loss": format(floor_loss, ".17g"),
                "predicted_conditional_gain": format(predicted_gain, ".17g"),
                "actual_joint_gain": format(actual_gain, ".17g"),
                "passed": True,
            }
            permutation_audits[expert] = permutation_audit
            # Backend may stage this expert immediately and release its source.
            install_record = None if not self.install_for_successor else self.backend.install_encoded_expert(
                layer=capture.layer,
                expert=expert,
                encoded={
                    item.tensor_id.key: {
                        "bits": item.bits,
                        "trellis": item.trellis,
                        "suh": item.suh,
                        "svh": item.svh,
                        "reconstructed_kn": item.reconstructed_kn,
                        "packed_sha256": item.packed_sha256,
                        "reconstruction_sha256": item.reconstruction_sha256,
                    }
                    for item in (gate, up, down)
                },
            )
            install_experts = install_audit["experts"]
            assert isinstance(install_experts, dict)
            incumbent_install = install_experts.get(str(expert))
            if incumbent_install is not None and incumbent_install != install_record:
                raise ValueError(
                    f"L{capture.layer} E{expert}: installed arithmetic audit drift"
                )
            install_experts[str(expert)] = install_record
            install_audit["complete"] = False
            atomic_write_json(install_audit_path, install_audit)
            encoded.extend(
                replace(item, reconstructed_kn=None) for item in (gate, up, down)
            )
        atomic_write_json(
            layer_dir / "cold-fallback-audit.json",
            {
                "layer": capture.layer,
                "min_fit_rows": self.min_fit_rows,
                "records": [self.cold_audit[key] for key in sorted(self.cold_audit)],
            },
        )
        masses = [float(value) for value in capture.mass_audit.mass_by_expert]
        predicted_total = sum(
            masses[expert]
            * float(interaction_audit[expert]["predicted_conditional_gain"])
            for expert in range(NUM_EXPERTS)
        )
        actual_total = sum(
            masses[expert] * float(interaction_audit[expert]["actual_joint_gain"])
            for expert in range(NUM_EXPERTS)
        )
        interaction_error = abs(predicted_total - actual_total)
        interaction_relative = interaction_error / max(abs(actual_total), 1e-30)
        interaction_passed = (
            interaction_error <= self.interaction_absolute_tolerance
            or interaction_relative <= self.interaction_relative_tolerance
        )
        if not interaction_passed:
            raise RuntimeError(
                "conditional DP interaction audit failed; run an interaction-aware "
                "equal-budget local-swap search before publication"
            )
        atomic_write_json(
            layer_dir / "interaction-audit.json",
            {
                "layer": capture.layer,
                "fixed_point_iteration": allocation.fixed_point_iteration,
                "bindings": final_bindings,
                "summary": {
                    "mass_weighted_predicted_gain": format(predicted_total, ".17g"),
                    "mass_weighted_actual_gain": format(actual_total, ".17g"),
                    "absolute_error": format(interaction_error, ".17g"),
                    "relative_error": format(interaction_relative, ".17g"),
                    "absolute_tolerance": format(
                        self.interaction_absolute_tolerance, ".17g"
                    ),
                    "relative_tolerance": format(
                        self.interaction_relative_tolerance, ".17g"
                    ),
                    "passed": True,
                },
                "experts": {
                    str(key): value for key, value in interaction_audit.items()
                },
            },
        )
        install_experts = install_audit["experts"]
        assert isinstance(install_experts, dict)
        if set(install_experts) != {str(expert) for expert in range(NUM_EXPERTS)}:
            raise ValueError("layer install audit does not cover all 256 experts")
        official_layer_audit = (
            self.backend.audit_installed_layer(layer=capture.layer)
            if self.install_for_successor
            else "layer-parallel-no-install"
        )
        incumbent_layer_audit = install_audit.get("official_layer_audit")
        if (
            incumbent_layer_audit is not None
            and incumbent_layer_audit != official_layer_audit
        ):
            raise ValueError(f"L{capture.layer}: official installed-layer audit drift")
        install_audit["official_layer_audit"] = official_layer_audit
        install_audit["complete"] = True
        atomic_write_json(install_audit_path, install_audit)
        return LayerEncodeResult(
            allocation=allocation,
            encoded=tuple(encoded),
            shared_gate_up_suh=encoded[0].suh,
            shared_down_svh=next(
                item.svh for item in encoded if item.tensor_id.projection == "down_proj"
            ),
            fixed_point_iterations=len(controller.history),
            final_gate_up_sha256=gu_hashes,
            interaction_audit=interaction_audit,
            final_expert_artifacts=final_artifacts,
            permutation_audit=permutation_audits,
            probe_artifacts=probe_artifacts,
            install_audit_sha256=sha256_file(install_audit_path),
        )
