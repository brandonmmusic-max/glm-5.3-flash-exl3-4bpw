"""Wall-clock-first, one-pass Round-10 layer encoder.

This is intentionally not a flag on ``LayerProcessor.run``.  The older runner
requires at least two fixed-point iterations and then performs an additional
all-floor interaction encode plus install/oracle replays.  R10 probes each
tensor at 3/4/5 exactly once, allocates once, reuses the selected gate/up probe
bytes, recomputes the full down covariance through those selected round trips,
and emits schema v2 directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

from .allocation import allocation_to_json, build_curves
from .backend import ExpertWeights, LayerCapture
from .constants import ALLOWED_BITS, NUM_EXPERTS, TensorId
from .determinism import atomic_write_json, canonical_json_bytes, sha256_bytes
from .layer import HoldoutUse, LayerProcessor
from .permutation import permute_expert_hf
from .r10_allocation import solve_sequential_allocation
from .schema import emit_layer_v2
from .types import CandidateLoss, EncodedTensor, StateShard


@dataclass
class _ExpertProbe:
    weights: ExpertWeights
    gate: Mapping[int, EncodedTensor]
    up: Mapping[int, EncodedTensor]
    down: Mapping[int, EncodedTensor]
    fit_row_ids: tuple[int, ...]
    fit_row_ids_sha256: str
    holdout_row_ids_sha256: str
    down_fit_row_ids_sha256: str
    down_gate_up_sha256: str
    used_cold_fallback: bool


class R10LayerProcessor(LayerProcessor):
    """Single-pass probing with no encode-time verification replays."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Search needs the same gate/up covariance as probing. Keep it in HBM
        # just long enough for that expert instead of writing/reading 48 GiB of
        # row-plan files per layer.
        self._r10_covariances: dict[tuple[str, int, str], object] = {}
        self._r10_search_weights: dict[int, ExpertWeights] = {}
        self._r10_holdouts: dict[tuple[str, int, str], HoldoutUse] = {}

    def remember_search_weights(self, expert: int, weights: ExpertWeights) -> None:
        self._r10_search_weights[int(expert)] = weights

    def _load_covariance_use(self, capture, expert, split):
        # The old 48 GiB/layer row-plan cache costs more wall time than rebuilding
        # from a flat mmap and is not reusable across the one-pass R10 recipe.
        return self._r10_covariances.get((capture.digest, int(expert), str(split)))

    def _seal_covariance_use(
        self, capture, expert, split, value, fallback_row_ids, cold_record
    ) -> None:
        self._r10_covariances[(capture.digest, int(expert), str(split))] = value

    def _holdout(
        self,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        expert: int,
        exclude_ids=(),
    ) -> HoldoutUse:
        """Collect held-out rows in memory; never write the 48 GiB row cache."""

        import torch

        excluded = set(int(value) for value in exclude_ids)
        cache_key = (
            capture.digest,
            int(expert),
            self._row_hash(tuple(sorted(excluded)), "holdout:exclude"),
        )
        cached = self._r10_holdouts.get(cache_key)
        if cached is not None:
            return cached
        seen = set(excluded)
        xs = []
        accepted_ids: list[int] = []
        remaining = self.holdout_rows
        for rows in self.backend.iter_expert_rows(
            capture=capture, shards=shards, expert=expert, split="holdout"
        ):
            if remaining <= 0:
                break
            hidden, accepted = self._select_new_rows(rows, seen, skip_seen=False)
            value = hidden[:remaining].detach().cpu()
            if value.shape[0]:
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
                value = hidden[:remaining].detach().cpu()
                if value.shape[0]:
                    xs.append(value)
                    selected = accepted[: int(value.shape[0])]
                    accepted_ids.extend(selected)
                    fallback_rows += len(selected)
                    remaining -= int(value.shape[0])
        if remaining > 0 or not xs:
            raise ValueError(
                f"L{capture.layer} E{expert}: held-out rows underfilled by {remaining}"
            )
        row_ids = tuple(accepted_ids)
        result = HoldoutUse(
            hidden=torch.cat(xs, dim=0).to(self.device),
            row_ids=row_ids,
            row_ids_sha256=self._row_hash(row_ids, "holdout:combined"),
            routed_rows=routed_rows,
            fallback_rows=fallback_rows,
        )
        self._r10_holdouts[cache_key] = result
        return result

    @staticmethod
    def _packed_only(item: EncodedTensor) -> EncodedTensor:
        """Retain emitted bytes and hashes, not a 48 MiB FP32 reconstruction."""

        return replace(item, reconstructed_kn=None)

    def _decode_selected(self, item: EncodedTensor) -> EncodedTensor:
        """Decode one selected candidate for honest final down calibration."""

        if item.reconstructed_kn is not None:
            return item
        packed = item.trellis.to(self.device)
        reconstructed = self.codec.decode_to_original(
            packed, item.suh, item.svh, item.bits
        )
        return replace(item, reconstructed_kn=reconstructed)

    def _encode_bits(self, *, tensor_id, weight_hf, covariance, suh, svh, provenance):
        method = getattr(self.codec, "encode_bits", None)
        if method is not None:
            values = method(
                tensor_id=tensor_id,
                weight_hf=weight_hf,
                covariance=covariance,
                bits=ALLOWED_BITS,
                suh=suh,
                svh=svh,
                sigma_reg=self.sigma_reg,
                provenance=provenance,
            )
            if isinstance(values, Mapping):
                result = {int(key): value for key, value in values.items()}
            else:
                result = {value.bits: value for value in values}
            if set(result) != set(ALLOWED_BITS):
                raise ValueError("R10 codec did not return all 3/4/5 candidates")
            return result
        return {
            bits: self.codec.encode(
                tensor_id=tensor_id,
                weight_hf=weight_hf,
                covariance=covariance,
                bits=bits,
                suh=suh,
                svh=svh,
                sigma_reg=self.sigma_reg,
                provenance=provenance,
            )
            for bits in ALLOWED_BITS
        }

    def _encode_gate_up_group(
        self,
        *,
        layer,
        expert,
        gate_weight,
        up_weight,
        covariance,
        gate_vectors,
        up_vectors,
        gate_bits,
        up_bits,
        gate_sha256,
        up_sha256,
    ):
        """Use Brandon's equal-K lockstep path for the gate/up pair."""

        grouped = getattr(self.codec, "encode_group", None)
        if callable(grouped):
            gate, up = grouped(
                (
                    {
                        "tensor_id": TensorId(layer, expert, "gate_proj"),
                        "weight_hf": gate_weight,
                        "covariance": covariance,
                        "bits": tuple(gate_bits),
                        "suh": gate_vectors.suh,
                        "svh": gate_vectors.svh,
                        "sigma_reg": self.sigma_reg,
                        "provenance": {"bf16_sha256": gate_sha256},
                    },
                    {
                        "tensor_id": TensorId(layer, expert, "up_proj"),
                        "weight_hf": up_weight,
                        "covariance": covariance,
                        "bits": tuple(up_bits),
                        "suh": up_vectors.suh,
                        "svh": up_vectors.svh,
                        "sigma_reg": self.sigma_reg,
                        "provenance": {"bf16_sha256": up_sha256},
                    },
                )
            )
            return gate, up
        gate = self._encode_bits(
            tensor_id=TensorId(layer, expert, "gate_proj"),
            weight_hf=gate_weight,
            covariance=covariance,
            suh=gate_vectors.suh,
            svh=gate_vectors.svh,
            provenance={"bf16_sha256": gate_sha256},
        )
        up = self._encode_bits(
            tensor_id=TensorId(layer, expert, "up_proj"),
            weight_hf=up_weight,
            covariance=covariance,
            suh=up_vectors.suh,
            svh=up_vectors.svh,
            provenance={"bf16_sha256": up_sha256},
        )
        return (
            {bits: gate[bits] for bits in gate_bits},
            {bits: up[bits] for bits in up_bits},
        )

    def _encode_gate_up(
        self,
        *,
        layer,
        expert,
        weights,
        covariance,
        search,
        gate_bits,
        up_bits,
    ):
        """Search-time gate/up encode, co-stepped at their common K."""

        per_expert = search.experts[expert]
        gate_weight, up_weight, down_weight = permute_expert_hf(
            weights.gate_hf,
            weights.up_hf,
            weights.down_hf,
            per_expert.permutation,
        )
        gate_vectors, up_vectors, _ = self._vectors(search, expert)
        gate, up = self._encode_gate_up_group(
            layer=layer,
            expert=expert,
            gate_weight=gate_weight,
            up_weight=up_weight,
            covariance=covariance,
            gate_vectors=gate_vectors,
            up_vectors=up_vectors,
            gate_bits=(gate_bits,),
            up_bits=(up_bits,),
            gate_sha256=weights.payload_sha256["gate_proj"],
            up_sha256=weights.payload_sha256["up_proj"],
        )
        return gate[gate_bits], up[up_bits], down_weight

    def _probe_one(
        self,
        *,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        search,
        expert: int,
        search_sha256: str,
    ) -> tuple[_ExpertProbe, tuple[CandidateLoss, ...]]:
        layer = capture.layer
        weights: ExpertWeights = self._r10_search_weights.pop(
            expert,
            None,
        ) or self.backend.load_bf16_expert(layer=layer, expert=expert)
        weights.validate_bf16(layer, expert)
        per_expert = search.experts[expert]
        gate_weight, up_weight, down_weight = permute_expert_hf(
            weights.gate_hf,
            weights.up_hf,
            weights.down_hf,
            per_expert.permutation,
        )
        gu_use = self._covariance(capture, shards, expert, "fit")
        holdout = self._holdout(
            capture, shards, expert, exclude_ids=gu_use.row_ids
        )
        gate_vectors, up_vectors, down_vectors = self._vectors(search, expert)
        gate, up = self._encode_gate_up_group(
            layer=layer,
            expert=expert,
            gate_weight=gate_weight,
            up_weight=up_weight,
            covariance=gu_use.matrix,
            gate_vectors=gate_vectors,
            up_vectors=up_vectors,
            gate_bits=ALLOWED_BITS,
            up_bits=ALLOWED_BITS,
            gate_sha256=weights.payload_sha256["gate_proj"],
            up_sha256=weights.payload_sha256["up_proj"],
        )

        # The 4/4 pair is the one-pass base context.  All nine tensor candidates
        # are real encodes, and every sensitivity is a complete
        # gate+up+SwiGLU+down held-out loss.  Gate/up candidates therefore get
        # the same conditional down recalibration as the settled R7 design;
        # isolated linear-projection NMSE would change the allocation bytes.
        down_use, gu_hash = self._down_covariance(
            capture=capture,
            shards=shards,
            expert=expert,
            gate=gate[4],
            up=up[4],
            expected_row_ids=gu_use.row_ids,
        )
        down = self._encode_bits(
            tensor_id=TensorId(layer, expert, "down_proj"),
            weight_hf=down_weight,
            covariance=down_use.matrix,
            suh=down_vectors.suh,
            svh=down_vectors.svh,
            provenance={
                "bf16_sha256": weights.payload_sha256["down_proj"],
                "gate_up_roundtrip_sha256": gu_hash,
                "joint_full_k": 2048,
            },
        )
        pair_cache = {(4, 4): (down_use, gu_hash, down[4])}
        conditional_keys = ((3, 4), (5, 4), (4, 3), (4, 5))
        pending = []
        for gate_bits, up_bits in conditional_keys:
            candidate_use, candidate_hash = self._down_covariance(
                capture=capture,
                shards=shards,
                expert=expert,
                gate=gate[gate_bits],
                up=up[up_bits],
                expected_row_ids=gu_use.row_ids,
            )
            pending.append((gate_bits, up_bits, candidate_use, candidate_hash))
        grouped = getattr(self.codec, "encode_group", None)
        if callable(grouped):
            conditional_down = grouped(
                tuple(
                    {
                        "tensor_id": TensorId(layer, expert, "down_proj"),
                        "weight_hf": down_weight,
                        "covariance": candidate_use.matrix,
                        "bits": (4,),
                        "suh": down_vectors.suh,
                        "svh": down_vectors.svh,
                        "sigma_reg": self.sigma_reg,
                        "provenance": {
                            "bf16_sha256": weights.payload_sha256["down_proj"],
                            "gate_up_roundtrip_sha256": candidate_hash,
                            "joint_full_k": 2048,
                        },
                    }
                    for _, _, candidate_use, candidate_hash in pending
                )
            )
            down_values = [value[4] for value in conditional_down]
        else:
            down_values = [
                self.codec.encode(
                    tensor_id=TensorId(layer, expert, "down_proj"),
                    weight_hf=down_weight,
                    covariance=candidate_use.matrix,
                    bits=4,
                    suh=down_vectors.suh,
                    svh=down_vectors.svh,
                    sigma_reg=self.sigma_reg,
                    provenance={
                        "bf16_sha256": weights.payload_sha256["down_proj"],
                        "gate_up_roundtrip_sha256": candidate_hash,
                        "joint_full_k": 2048,
                    },
                )
                for _, _, candidate_use, candidate_hash in pending
            ]
        for record, candidate_down in zip(pending, down_values):
            gate_bits, up_bits, candidate_use, candidate_hash = record
            pair_cache[(gate_bits, up_bits)] = (
                candidate_use,
                candidate_hash,
                candidate_down,
            )

        measured: dict[tuple[str, int], tuple[float, object, str, EncodedTensor]] = {}
        for projection in ("gate_proj", "up_proj"):
            for bits in ALLOWED_BITS:
                gate_bits = bits if projection == "gate_proj" else 4
                up_bits = bits if projection == "up_proj" else 4
                candidate_use, candidate_hash, candidate_down = pair_cache[
                    (gate_bits, up_bits)
                ]
                loss = self._loss(
                    holdout.hidden,
                    (gate_weight, up_weight, down_weight),
                    gate[gate_bits],
                    up[up_bits],
                    candidate_down,
                )
                measured[(projection, bits)] = (
                    loss,
                    candidate_use,
                    candidate_hash,
                    candidate_down,
                )
        for bits in ALLOWED_BITS:
            loss = self._loss(
                holdout.hidden,
                (gate_weight, up_weight, down_weight),
                gate[4],
                up[4],
                down[bits],
            )
            measured[("down_proj", bits)] = (loss, down_use, gu_hash, down[bits])
        mass = str(capture.mass_audit.mass_by_expert[expert])
        context_hash = sha256_bytes(
            canonical_json_bytes({"gate_proj": 4, "up_proj": 4})
        )
        permutation_hash = sha256_bytes(
            canonical_json_bytes(list(per_expert.permutation))
        )
        vector_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "gate_suh": list(gate_vectors.suh),
                    "gate_svh": list(gate_vectors.svh),
                    "up_suh": list(up_vectors.suh),
                    "up_svh": list(up_vectors.svh),
                    "down_suh": list(down_vectors.suh),
                    "down_svh": list(down_vectors.svh),
                }
            )
        )
        records = []
        all_candidates = {"gate_proj": gate, "up_proj": up, "down_proj": down}
        for projection, candidates in all_candidates.items():
            for bits in ALLOWED_BITS:
                item = candidates[bits]
                loss, candidate_use, candidate_hash, candidate_down = measured[
                    (projection, bits)
                ]
                gate_bits = bits if projection == "gate_proj" else 4
                up_bits = bits if projection == "up_proj" else 4
                records.append(
                    CandidateLoss(
                        tensor_id=item.tensor_id,
                        bits=bits,
                        loss=format(loss, ".17g"),
                        mass=mass,
                        fit_rows=(
                            candidate_use.rows
                            if projection == "down_proj"
                            else gu_use.rows
                        ),
                        holdout_rows=int(holdout.hidden.shape[0]),
                        roundtrip_sha256=item.reconstruction_sha256,
                        gate_up_roundtrip_sha256=(
                            candidate_hash if projection == "down_proj" else None
                        ),
                        fixed_point_iteration=0,
                        context_bits_sha256=context_hash,
                        expert_roundtrip_sha256={
                            "gate_proj": gate[gate_bits].reconstruction_sha256,
                            "up_proj": up[up_bits].reconstruction_sha256,
                            "down_proj": candidate_down.reconstruction_sha256,
                        },
                        state_sha256=capture.state_sha256,
                        capture_sha256=capture.digest,
                        search_sha256=search_sha256,
                        source_inventory_sha256=self.source_inventory_sha256,
                        numeric_environment_sha256=self.numeric_environment_sha256,
                        runtime_inventory_sha256=self.runtime_inventory_sha256,
                        backend_fingerprint=self.backend.fingerprint,
                        fit_row_ids_sha256=gu_use.row_ids_sha256,
                        holdout_row_ids_sha256=holdout.row_ids_sha256,
                        permutation_sha256=permutation_hash,
                        vector_bundle_sha256=vector_hash,
                        used_cold_fallback=(
                            gu_use.cold_fallback or candidate_use.cold_fallback
                        ),
                    )
                )
        return (
            _ExpertProbe(
                weights=weights,
                gate={bits: self._packed_only(gate[bits]) for bits in ALLOWED_BITS},
                up={bits: self._packed_only(up[bits]) for bits in ALLOWED_BITS},
                down={bits: self._packed_only(down[bits]) for bits in ALLOWED_BITS},
                fit_row_ids=gu_use.row_ids,
                fit_row_ids_sha256=gu_use.row_ids_sha256,
                holdout_row_ids_sha256=holdout.row_ids_sha256,
                down_fit_row_ids_sha256=down_use.row_ids_sha256,
                down_gate_up_sha256=gu_hash,
                used_cold_fallback=gu_use.cold_fallback or down_use.cold_fallback,
            ),
            tuple(records),
        )

    def _enrich(
        self,
        *,
        item: EncodedTensor,
        weights: ExpertWeights,
        capture: LayerCapture,
        search,
        search_sha256: str,
        allocation,
        probe: _ExpertProbe,
        down_fit_sha256: str,
    ) -> EncodedTensor:
        per_expert = search.experts[item.tensor_id.expert]
        source = weights.source_records[item.tensor_id.projection]
        permutation_hash = sha256_bytes(
            canonical_json_bytes(list(per_expert.permutation))
        )
        vector_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "suh": item.suh.tolist(),
                    "svh": item.svh.tolist(),
                }
            )
        )
        provenance = dict(item.provenance)
        provenance.update(
            {
                "source_name": weights.source_names[item.tensor_id.projection],
                "source_shard": source["shard"],
                "source_payload_start": source["payload_start"],
                "source_payload_end": source["payload_end"],
                "source_inventory_sha256": self.source_inventory_sha256,
                "numeric_environment_sha256": self.numeric_environment_sha256,
                "runtime_inventory_sha256": self.runtime_inventory_sha256,
                "backend_fingerprint": self.backend.fingerprint,
                "state_sha256": capture.state_sha256,
                "capture_sha256": capture.digest,
                "search_sha256": search_sha256,
                "allocation_sha256": sha256_bytes(
                    canonical_json_bytes(dict(sorted(allocation.bits.items())))
                ),
                "probe_sha256": allocation.probe_sha256,
                "fit_row_ids_sha256": probe.fit_row_ids_sha256,
                "down_fit_row_ids_sha256": down_fit_sha256,
                "holdout_row_ids_sha256": probe.holdout_row_ids_sha256,
                "permutation_sha256": permutation_hash,
                "permutation_policy": per_expert.permutation_policy,
                "vector_sha256": vector_hash,
                "used_cold_fallback": probe.used_cold_fallback,
            }
        )
        return replace(item, provenance=provenance, reconstructed_kn=None)

    def run_r10(
        self,
        *,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        search,
        search_artifact_sha256: str,
        log=print,
    ) -> dict[str, object]:
        if search.layer != capture.layer:
            raise ValueError("search/capture layer mismatch")
        layer = capture.layer
        layer_dir = self.work_dir / f"layer-{layer:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        probes: dict[int, _ExpertProbe] = {}
        records: list[CandidateLoss] = []
        for expert in range(NUM_EXPERTS):
            probe, expert_records = self._probe_one(
                capture=capture,
                shards=shards,
                search=search,
                expert=expert,
                search_sha256=search_artifact_sha256,
            )
            probes[expert] = probe
            records.extend(expert_records)
            self._r10_covariances.pop((capture.digest, expert, "fit"), None)
            if expert % 8 == 7:
                log(f"layer {layer}: probed {expert + 1}/{NUM_EXPERTS} experts")

        probe_payload = {
            "schema": "r10-one-pass-probe-v1",
            "layer": layer,
            "records": [
                {
                    **asdict(record),
                    "tensor_id": record.tensor_id.key,
                }
                for record in records
            ],
        }
        probe_sha256 = sha256_bytes(canonical_json_bytes(probe_payload))
        probe_payload["content_sha256"] = probe_sha256
        atomic_write_json(layer_dir / "probe.json", probe_payload)
        curves = build_curves(
            layer, records, capture.mass_audit.mass_by_expert
        )
        allocation = solve_sequential_allocation(
            layer, curves, probe_sha256=probe_sha256
        )
        atomic_write_json(layer_dir / "allocation.json", allocation_to_json(allocation))

        encoded: list[EncodedTensor] = []
        final_artifacts: dict[int, str] = {}
        for expert in range(NUM_EXPERTS):
            probe = probes.pop(expert)
            weights = probe.weights
            weights.validate_bf16(layer, expert)
            per_expert = search.experts[expert]
            _, _, down_weight = permute_expert_hf(
                weights.gate_hf,
                weights.up_hf,
                weights.down_hf,
                per_expert.permutation,
            )
            gate = self._decode_selected(
                probe.gate[
                    allocation.bits[TensorId(layer, expert, "gate_proj").key]
                ]
            )
            up = self._decode_selected(
                probe.up[allocation.bits[TensorId(layer, expert, "up_proj").key]]
            )
            down_bits = allocation.bits[TensorId(layer, expert, "down_proj").key]
            down_use, gu_hash = self._down_covariance(
                capture=capture,
                shards=shards,
                expert=expert,
                gate=gate,
                up=up,
                expected_row_ids=probe.fit_row_ids,
            )
            if (
                gate.bits == 4
                and up.bits == 4
                and gu_hash == probe.down_gate_up_sha256
            ):
                down = probe.down[down_bits]
            else:
                _, _, down_vectors = self._vectors(search, expert)
                down = self.codec.encode(
                    tensor_id=TensorId(layer, expert, "down_proj"),
                    weight_hf=down_weight,
                    covariance=down_use.matrix,
                    bits=down_bits,
                    suh=down_vectors.suh,
                    svh=down_vectors.svh,
                    sigma_reg=self.sigma_reg,
                    provenance={
                        "bf16_sha256": weights.payload_sha256["down_proj"],
                        "gate_up_roundtrip_sha256": gu_hash,
                        "joint_full_k": 2048,
                    },
                )
            if down.provenance.get("gate_up_roundtrip_sha256") != gu_hash:
                raise AssertionError("final down is not bound to selected gate/up")
            final = tuple(
                self._enrich(
                    item=item,
                    weights=weights,
                    capture=capture,
                    search=search,
                    search_sha256=search_artifact_sha256,
                    allocation=allocation,
                    probe=probe,
                    down_fit_sha256=down_use.row_ids_sha256,
                )
                for item in (gate, up, down)
            )
            final_artifacts[expert] = sha256_bytes(
                canonical_json_bytes(
                    {
                        item.tensor_id.projection: {
                            "bits": item.bits,
                            "packed_sha256": item.packed_sha256,
                            "reconstruction_sha256": item.reconstruction_sha256,
                        }
                        for item in final
                    }
                )
            )
            encoded.extend(final)
            self.backend.clear_expert_row_memo(capture, expert)
            if expert % 8 == 7:
                log(f"layer {layer}: finalized {expert + 1}/{NUM_EXPERTS} experts")

        output = self.work_dir / "v2"
        manifest = emit_layer_v2(
            output,
            layer=layer,
            encoded_tensors=encoded,
            shared_gate_up_suh=encoded[0].suh,
            shared_down_svh=next(
                item.svh
                for item in encoded
                if item.tensor_id.projection == "down_proj"
            ),
            allocation_bits=allocation.bits,
            layer_provenance={
                "mode": "r10-one-pass-no-encode-time-verification",
                "source_inventory_sha256": self.source_inventory_sha256,
                "numeric_environment_sha256": self.numeric_environment_sha256,
                "runtime_inventory_sha256": self.runtime_inventory_sha256,
                "backend_fingerprint": self.backend.fingerprint,
                "capture_sha256": capture.digest,
                "search_sha256": search_artifact_sha256,
                "probe_sha256": probe_sha256,
            },
            permutations={
                expert: search.experts[expert].permutation
                for expert in range(NUM_EXPERTS)
            },
            permutation_policies={
                expert: search.experts[expert].permutation_policy
                for expert in range(NUM_EXPERTS)
            },
            final_expert_artifacts=final_artifacts,
        )
        return manifest
