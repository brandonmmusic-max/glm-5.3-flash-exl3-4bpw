"""Concrete GLM-5.2 capture/source backend for the Round 7 walk.

Only transformer execution is supplied by a fingerprinted runtime adapter.
This module owns the encoder-critical semantics: canonical state validation,
exact runtime-route capture, float32 mass accounting, immutable sidecars,
fit/holdout membership, cold fallback sampling, and inventory-checked BF16
expert streaming.  No classifier or reimplemented approximate router exists.
"""

from __future__ import annotations

import heapq
import importlib
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .backend import (
    CalibrationBatch,
    ExpertRows,
    ExpertWeights,
    LayerCapture,
    Round7Backend,
)
from .constants import (
    FIRST_MOE_LAYER,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAST_MOE_LAYER,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TOP_K,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    derive_seed,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .inventory import load_checkpoint_inventory, load_runtime_code_inventory
from .routing import MassAudit, RoutedMassAccumulator
from .safetensors_io import (
    SafeTensorReader,
    read_torch_tensor_mmap,
    torch_tensor_entry,
    write_safetensors_atomic,
)
from .types import RoutedBatch, StateShard

SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
INSTALL_AUDIT_ROWS = 32
INSTALL_MAX_ABS_ERROR = 0.03125
INSTALL_MAX_RELATIVE_L2 = 0.001953125
DSA_INDEX_TOPK = 2048


class GLM52Runtime(ABC):
    """Narrow, fingerprinted bridge to the pinned model implementation.

    The adapter must invoke the model's own carried EXL3 layers, attention,
    router, shared experts, residuals, and layer norms.  It does not decide
    row membership, routed mass, bits, Hessians, or checkpoint schema.
    """

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Content hash covering every imported model/runtime source file."""

    @abstractmethod
    def prepare_corpus_plan(self, *, corpus: Path) -> Mapping[str, object]:
        """Seal the tokenizer-bound ordered prompt plan before forwarding."""

    @abstractmethod
    def initialize_carried_state(
        self,
        *,
        carrier: Path,
        corpus: Path,
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        """Write canonical state after reconstructed carried layers 0..2."""

    @abstractmethod
    def route_exact(
        self, *, layer: int, moe_hidden: Any, attention_metadata: Mapping[str, object]
    ) -> RoutedBatch:
        """Call the installed GLM-5.2 router and return its actual top-8 result."""

    @abstractmethod
    def prepare_moe_input(
        self, *, layer: int, hidden: Any, attention_metadata: Mapping[str, object]
    ) -> Any:
        """Run this layer's carried attention/residual/norm to the MoE input.

        ``hidden`` is the sealed transformer-block input.  Returning it
        unchanged is not permitted: gate/up Hessians and routing are defined on
        the post-attention, post-normalization tensor consumed by the routed
        experts.
        """

    def begin_capture(self, *, layer: int) -> None:
        """Optional lifecycle hook used to retain one carried block while capturing."""

    def end_capture(self, *, layer: int) -> None:
        """Optional lifecycle hook paired with :meth:`begin_capture`."""

    @abstractmethod
    def capture_arithmetic_audit(self, *, layer: int) -> Mapping[str, object]:
        """Return the sealed carried-layer decode/arithmetic contract.

        The record is required for both a fresh capture and a capture rebuilt
        entirely from already sealed routing sidecars.  It binds the official
        layer implementation, its inventoried carrier payloads, and the pinned
        arithmetic mode to the capture manifest.
        """

    @abstractmethod
    def install_encoded_expert(
        self, *, layer: int, expert: int, encoded: Mapping[str, Any]
    ) -> Mapping[str, object]:
        """Install the three sealed packed records in the active layer.

        Each record includes packed TRELLIS, stored vectors, bits, and its sealed
        reconstruction hash.  The runtime must decode and hash-check that payload
        before installation.  A caller may additionally supply ``reconstructed_kn``
        for an exact equality check against the runtime decode.  Aggregate
        arithmetic is checked separately through :meth:`audit_installed_layer`.
        """

    @abstractmethod
    def audit_installed_layer(self, *, layer: int) -> Mapping[str, object]:
        """Call the official installed MoE and an independent routed reference."""

    @abstractmethod
    def restore_encoded_layer(self, *, layer: int, manifest: Path) -> None:
        """Restore a sealed layer using the schema-v2 reconstruction path."""

    @abstractmethod
    def forward_installed_layer(
        self,
        *,
        layer: int,
        input_shards: Iterable[StateShard],
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        """Forward one full layer and write canonical successor state shards."""


def _load_runtime(factory_path: str, config) -> GLM52Runtime:
    if ":" not in factory_path:
        raise ValueError("runtime factory must be `module:callable`")
    module_name, attribute = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    runtime = factory(config)
    if not isinstance(runtime, GLM52Runtime) or not runtime.fingerprint:
        raise TypeError("runtime factory must return a fingerprinted GLM52Runtime")
    return runtime


def _read_tensor(reader: SafeTensorReader, name: str):
    return read_torch_tensor_mmap(reader, name)


class GLM52Backend(Round7Backend):
    def __init__(self, *, config, runtime: GLM52Runtime) -> None:
        self.config = config
        self.runtime = runtime
        self.runtime_inventory = load_runtime_code_inventory(
            config.runtime_inventory,
            verify_files=bool(getattr(config, "verify_runtime_files", True)),
        )
        if runtime.fingerprint != self.runtime_inventory["inventory_sha256"]:
            raise ValueError(
                "GLM runtime fingerprint must equal the sealed model-code inventory"
            )
        self.work = Path(config.work).resolve()
        self.source = Path(config.bf16_source).resolve()
        self.source_inventory = load_checkpoint_inventory(
            config.source_inventory,
            role="bf16-source",
            require_routed_bf16=True,
        )
        self.source_entries: Mapping[str, Mapping[str, object]] = self.source_inventory[
            "entries"
        ]  # type: ignore[assignment]
        self._fingerprint = sha256_bytes(
            canonical_json_bytes(
                {
                    "backend_sha256": sha256_file(Path(__file__)),
                    "runtime_fingerprint": runtime.fingerprint,
                    "source_inventory_sha256": self.source_inventory[
                        "inventory_sha256"
                    ],
                    "state_schema": "r7-state-shard-v2",
                    "routing_schema": "r7-routing-sidecar-v2",
                }
            )
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def prepare_corpus_plan(self, *, corpus: Path) -> Mapping[str, object]:
        plan = dict(self.runtime.prepare_corpus_plan(corpus=corpus))
        if set(plan) != {
            "corpus_plan_sha256",
            "corpus_plan_artifact_sha256",
            "expected_shards",
        }:
            raise ValueError("runtime corpus-plan contract key drift")
        digest = plan["corpus_plan_sha256"]
        artifact_digest = plan["corpus_plan_artifact_sha256"]
        expected = plan["expected_shards"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(artifact_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None
            or not isinstance(expected, Mapping)
            or not expected
            or any(
                not SHARD_ID.fullmatch(str(shard_id))
                or type(tokens) is not int
                or tokens <= 0
                for shard_id, tokens in expected.items()
            )
        ):
            raise ValueError("runtime emitted an invalid corpus plan")
        return {
            "corpus_plan_sha256": digest,
            "corpus_plan_artifact_sha256": artifact_digest,
            "expected_shards": {
                str(key): int(value) for key, value in sorted(expected.items())
            },
        }

    @staticmethod
    def _validate_record(
        record: tuple[str, Path, Path, int, int], partial: Path
    ) -> None:
        shard_id, hidden, metadata, tokens, hidden_size = record
        if not SHARD_ID.fullmatch(shard_id):
            raise ValueError(f"unsafe state shard ID {shard_id!r}")
        if Path(hidden).resolve().parent != partial.resolve():
            raise ValueError("runtime hidden shard escaped the transition directory")
        if Path(metadata).resolve().parent != partial.resolve():
            raise ValueError("runtime metadata shard escaped the transition directory")
        if tokens <= 0 or hidden_size != HIDDEN_SIZE:
            raise ValueError("runtime emitted invalid state geometry")
        reader = SafeTensorReader(hidden)
        if set(reader.tensors) != {"hidden", "prev_topk_indices"}:
            raise ValueError(
                "canonical state shard must contain exactly hidden and "
                "prev_topk_indices"
            )
        info = reader.tensors["hidden"]
        if info.dtype != "BF16" or info.shape != (tokens, HIDDEN_SIZE):
            raise ValueError("canonical state hidden tensor must be BF16 [tokens,6144]")
        previous = reader.tensors["prev_topk_indices"]
        expected_topk = min(tokens, DSA_INDEX_TOPK)
        if previous.dtype != "I32" or previous.shape != (tokens, expected_topk):
            raise ValueError(
                "canonical DSA state must be I32 [tokens,min(tokens,2048)]"
            )
        meta = read_json(metadata)
        required = {
            "schema",
            "shard_id",
            "tokens",
            "global_row_start",
            "sequence_lengths",
            "input_ids",
            "prev_topk_shape",
            "attention_implementation",
            "dispatch_audit_sha256",
            "auxiliary",
            "corpus_plan_sha256",
            "producer",
            "carrier_decode_audit_sha256",
        }
        if not required <= set(meta) or meta["schema"] != "r7-state-metadata-v2":
            raise ValueError("canonical state metadata is incomplete")
        if meta["shard_id"] != shard_id or int(meta["tokens"]) != tokens:
            raise ValueError("canonical state metadata binding drift")
        lengths = [int(value) for value in meta["sequence_lengths"]]
        if lengths != [tokens]:
            raise ValueError("canonical DSA state must contain one preserved prompt")
        input_ids = meta["input_ids"]
        if (
            not isinstance(input_ids, list)
            or len(input_ids) != tokens
            or any(not isinstance(value, int) or value < 0 for value in input_ids)
        ):
            raise ValueError("state input IDs do not cover the preserved prompt")
        if meta["prev_topk_shape"] != [tokens, expected_topk]:
            raise ValueError("state DSA auxiliary shape metadata drift")
        if (
            meta["attention_implementation"] != "eager"
            or meta["auxiliary"] != "prev_topk_indices-int32"
            or int(meta["global_row_start"]) < 0
        ):
            raise ValueError("state attention/arithmetic contract drift")
        for key in (
            "corpus_plan_sha256",
            "carrier_decode_audit_sha256",
            "dispatch_audit_sha256",
        ):
            if (
                not isinstance(meta[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", meta[key]) is None
            ):
                raise ValueError(f"state metadata has invalid {key}")

    def initialize_carried_state(
        self,
        *,
        carrier: Path,
        corpus: Path,
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        for record in self.runtime.initialize_carried_state(
            carrier=carrier,
            corpus=corpus,
            output_partial=output_partial,
            completed_shard_ids=completed_shard_ids,
        ):
            self._validate_record(record, output_partial)
            yield record

    def _read_state(self, shard: StateShard) -> CalibrationBatch:
        self._validate_record(
            (
                shard.shard_id,
                shard.hidden_path,
                shard.metadata_path,
                shard.tokens,
                shard.hidden_size,
            ),
            shard.hidden_path.parent,
        )
        reader = SafeTensorReader(shard.hidden_path)
        hidden = _read_tensor(reader, "hidden")
        previous = _read_tensor(reader, "prev_topk_indices")
        metadata = dict(read_json(shard.metadata_path))
        metadata["prev_topk_indices"] = previous
        start = int(metadata["global_row_start"])
        import torch

        row_ids = torch.arange(start, start + shard.tokens, dtype=torch.int64)
        return CalibrationBatch(
            shard_id=shard.shard_id,
            hidden=hidden,
            row_ids=row_ids,
            attention_metadata=metadata,
            token_count=shard.tokens,
        )

    def iter_state(self, shards: Iterable[StateShard]) -> Iterator[CalibrationBatch]:
        values = tuple(shards)
        if [item.shard_id for item in values] != sorted(
            item.shard_id for item in values
        ):
            raise ValueError("state iteration must be canonical")
        previous_end = -1
        for shard in values:
            batch = self._read_state(shard)
            first = int(batch.row_ids[0].item())
            if first <= previous_end:
                raise ValueError("global row-ID ranges overlap or are unordered")
            previous_end = int(batch.row_ids[-1].item())
            yield batch

    def route(self, layer: int, batch: CalibrationBatch) -> RoutedBatch:
        import torch

        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("routing is restricted to replaced layers 3..77")

        result = self.runtime.route_exact(
            layer=layer,
            moe_hidden=batch.hidden,
            attention_metadata=batch.attention_metadata,
        )
        ids = torch.as_tensor(result.expert_ids)
        weights = torch.as_tensor(result.expert_weights)
        if ids.shape != (batch.token_count, TOP_K) or weights.shape != ids.shape:
            raise ValueError("runtime router did not return [tokens,8]")
        if weights.dtype != torch.float32:
            raise ValueError("runtime router weights must be materialized as float32")
        return RoutedBatch(ids, weights, float(result.expected_mass_per_token))

    def _state_sha(self, layer: int) -> str:
        return sha256_file(
            self.work / "states" / f"input-layer-{layer:03d}" / "STATE.json"
        )

    def _sidecar_path(self, routing_dir: Path, shard_id: str) -> Path:
        if not SHARD_ID.fullmatch(shard_id):
            raise ValueError(f"unsafe routing shard ID {shard_id!r}")
        return routing_dir / f"routes-{shard_id}.safetensors"

    def _read_sidecar(self, path: Path):
        reader = SafeTensorReader(path)
        if set(reader.tensors) != {
            "expert_ids",
            "expert_weights",
            "moe_hidden",
            "row_ids",
        }:
            raise ValueError(f"invalid routing sidecar tensor set: {path}")
        ids = _read_tensor(reader, "expert_ids")
        weights = _read_tensor(reader, "expert_weights")
        moe_hidden = _read_tensor(reader, "moe_hidden")
        row_ids = _read_tensor(reader, "row_ids")
        if ids.ndim != 2 or ids.shape[1] != TOP_K or weights.shape != ids.shape:
            raise ValueError(f"invalid routing sidecar shape: {path}")
        if row_ids.shape != (ids.shape[0],):
            raise ValueError(f"invalid routing row-ID shape: {path}")
        if moe_hidden.dtype != __import__("torch").bfloat16 or moe_hidden.shape != (
            ids.shape[0],
            HIDDEN_SIZE,
        ):
            raise ValueError(f"invalid captured MoE input: {path}")
        if not __import__("torch").isfinite(moe_hidden).all():
            raise ValueError(f"non-finite captured MoE input: {path}")
        return ids, weights, moe_hidden, row_ids

    def capture_layer(
        self,
        *,
        layer: int,
        shards: Iterable[StateShard],
        routing_dir: Path,
    ) -> LayerCapture:
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("capture is restricted to replaced layers 3..77")
        routing_dir.mkdir(parents=True, exist_ok=True)
        state_sha = self._state_sha(layer)
        progress_path = routing_dir / "CAPTURE-PARTIAL.json"
        records: dict[str, object] = {}
        if progress_path.exists():
            partial = read_json(progress_path)
            if (
                partial.get("layer") != layer
                or partial.get("state_sha256") != state_sha
            ):
                raise ValueError("routing capture resume binding drift")
            records = dict(partial.get("sidecars", {}))
        accumulator = RoutedMassAccumulator(layer)
        self.runtime.begin_capture(layer=layer)
        try:
            for shard in tuple(shards):
                batch = self._read_state(shard)
                sidecar = self._sidecar_path(routing_dir, shard.shard_id)
                prior = records.get(shard.shard_id)
                if (
                    isinstance(prior, dict)
                    and prior.get("state_hidden_sha256") == shard.sha256_hidden
                    and prior.get("state_metadata_sha256") == shard.sha256_metadata
                    and sidecar.is_file()
                    and sha256_file(sidecar) == prior.get("sha256")
                ):
                    ids, weights, _, row_ids = self._read_sidecar(sidecar)
                    expected_mass = float(prior["expected_mass_per_token"])
                else:
                    exact_moe_hidden = self.runtime.prepare_moe_input(
                        layer=layer,
                        hidden=batch.hidden,
                        attention_metadata=batch.attention_metadata,
                    )
                    import torch

                    exact_moe_hidden = torch.as_tensor(exact_moe_hidden)
                    if exact_moe_hidden.shape != (
                        batch.token_count,
                        HIDDEN_SIZE,
                    ):
                        raise ValueError(
                            "runtime MoE input must have shape [tokens,6144]"
                        )
                    if exact_moe_hidden.dtype not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                    ):
                        raise ValueError("runtime MoE input has unsupported dtype")
                    if not torch.isfinite(exact_moe_hidden).all():
                        raise ValueError("runtime emitted non-finite MoE input")
                    routed_batch = CalibrationBatch(
                        shard_id=batch.shard_id,
                        hidden=exact_moe_hidden,
                        row_ids=batch.row_ids,
                        attention_metadata=batch.attention_metadata,
                        token_count=batch.token_count,
                    )
                    result = self.route(layer, routed_batch)
                    moe_hidden = exact_moe_hidden.to(torch.bfloat16)
                    ids, weights, row_ids = (
                        result.expert_ids,
                        result.expert_weights,
                        batch.row_ids,
                    )
                    _, sidecar_hash = write_safetensors_atomic(
                        sidecar,
                        (
                            torch_tensor_entry("expert_ids", ids.to(dtype=torch.int32)),
                            torch_tensor_entry("expert_weights", weights),
                            torch_tensor_entry("moe_hidden", moe_hidden),
                            torch_tensor_entry("row_ids", row_ids),
                        ),
                        metadata={
                            "r7_schema": "r7-routing-sidecar-v2",
                            "layer": str(layer),
                            "shard_id": shard.shard_id,
                            "hidden_role": "post-attention-post-norm-moe-input",
                        },
                    )
                    expected_mass = float(result.expected_mass_per_token)
                    records[shard.shard_id] = {
                        "file": sidecar.name,
                        "sha256": sidecar_hash,
                        "state_hidden_sha256": shard.sha256_hidden,
                        "state_metadata_sha256": shard.sha256_metadata,
                        "expected_mass_per_token": format(expected_mass, ".17g"),
                        "tokens": shard.tokens,
                    }
                    atomic_write_json(
                        progress_path,
                        {
                            "marker": RECIPE_MARKER,
                            "recipe_version": RECIPE_VERSION,
                            "layer": layer,
                            "state_sha256": state_sha,
                            "sidecars": records,
                        },
                    )
                if not __import__("torch").equal(row_ids, batch.row_ids):
                    raise ValueError("routing sidecar row IDs differ from sealed state")
                accumulator.add(ids, weights, expected_mass)
        finally:
            self.runtime.end_capture(layer=layer)
        runtime_audit = dict(self.runtime.capture_arithmetic_audit(layer=layer))
        self._validate_capture_arithmetic_audit(runtime_audit, layer=layer)
        audit = accumulator.finish()
        capture_payload = {
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "schema": "r7-routing-capture-v2",
            "layer": layer,
            "state_sha256": state_sha,
            "backend_fingerprint": self.fingerprint,
            "sidecars": {key: records[key] for key in sorted(records)},
            "mass_audit": audit.to_json(),
            "runtime_arithmetic_audit": runtime_audit,
        }
        capture_path = routing_dir / "CAPTURE.json"
        atomic_write_json(capture_path, capture_payload)
        progress_path.unlink(missing_ok=True)
        return self.open_capture(layer=layer, routing_dir=routing_dir)

    def open_capture(self, *, layer: int, routing_dir: Path) -> LayerCapture:
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("capture is restricted to replaced layers 3..77")
        path = routing_dir / "CAPTURE.json"
        raw = read_json(path)
        if (
            raw.get("marker") != RECIPE_MARKER
            or raw.get("schema") != "r7-routing-capture-v2"
            or int(raw.get("layer", -1)) != layer
            or raw.get("backend_fingerprint") != self.fingerprint
        ):
            raise ValueError("foreign routing capture")
        if raw.get("state_sha256") != self._state_sha(layer):
            raise ValueError("routing capture state binding drift")
        runtime_audit = raw.get("runtime_arithmetic_audit")
        if not isinstance(runtime_audit, dict):
            raise ValueError("routing capture lacks carried-layer arithmetic audit")
        self._validate_capture_arithmetic_audit(runtime_audit, layer=layer)
        hashes: dict[str, str] = {}
        accumulator = RoutedMassAccumulator(layer)
        sidecars = raw.get("sidecars")
        if not isinstance(sidecars, dict) or not sidecars:
            raise ValueError("routing capture has no sidecars")
        for shard_id, record in sorted(sidecars.items()):
            if not isinstance(record, dict):
                raise ValueError("malformed routing sidecar record")
            sidecar = routing_dir / str(record["file"])
            digest = sha256_file(sidecar)
            if digest != record.get("sha256"):
                raise ValueError(f"routing sidecar hash drift: {shard_id}")
            ids, weights, _, _ = self._read_sidecar(sidecar)
            accumulator.add(ids, weights, float(record["expected_mass_per_token"]))
            hashes[str(sidecar.relative_to(self.work))] = digest
        hashes[str(path.relative_to(self.work))] = sha256_file(path)
        audit = MassAudit.from_json(raw["mass_audit"])
        if accumulator.finish() != audit:
            raise ValueError("fresh sidecar mass audit differs from capture manifest")
        return LayerCapture(
            layer=layer,
            state_sha256=str(raw["state_sha256"]),
            routing_dir=routing_dir,
            mass_audit=audit,
            routing_sha256=hashes,
        )

    def open_flat_capture(self, *, layer: int, capture_dir: Path) -> LayerCapture:
        """Bind one layer's flat capture as a `LayerCapture`, without sealed state.

        The layer-parallel encoder calibrates from a single `flat_capture` trio
        (`x.bin` post-attention MoE input, `ids.bin` top-8 routed experts,
        `weights.bin` exact float32 router weights) instead of the per-prompt
        state/sidecar chain, which is what makes layers independent.  This is
        the only substitution: the returned record carries the same fields
        `capture_layer` produces, so `capture.digest`, the mass allocator, and
        every downstream binding behave identically.

        `state_sha256` is the sealed hidden payload digest and `routing_sha256`
        maps each capture payload to its manifest digest, so the capture digest
        still covers exactly what the encode consumed.

        Row iteration is wired here rather than in `iter_expert_rows`: the flat
        rows are installed into the same `(capture.digest, shard-ids)`-keyed
        bound-batch cache that `_iter_bound_batches` reads, so the sealed-state
        path is untouched while `iter_expert_rows` and `iter_cold_fallback_rows`
        run against the capture with identical semantics (same `_split`
        membership, same `ExpertRows` fields, same order, same reservoir).

        Fails closed on any manifest, geometry, digest, or routed-mass drift.
        """

        from .flat_capture import (
            IDS_FILE,
            MANIFEST_FILE,
            WEIGHTS_FILE,
            X_FILE,
            FlatCaptureReader,
            layer_capture_dir,
        )
        from .flat_runtime import build_flat_bound_batches, build_flat_mass_audit

        value = int(layer)
        if not FIRST_MOE_LAYER <= value <= LAST_MOE_LAYER:
            raise ValueError("capture is restricted to replaced layers 3..77")
        layer_dir = layer_capture_dir(capture_dir, value)
        manifest_path = layer_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            raise ValueError(f"flat capture manifest not found: {manifest_path}")
        reader = FlatCaptureReader(
            manifest_path,
            expected_layer=value,
            expected_bindings=getattr(self.config, "expected_flat_bindings", None),
            expected_verification=getattr(
                self.config, "expected_flat_verification", None
            ),
            verify_payloads=bool(getattr(self.config, "verify_flat_payloads", True)),
            verify_structure=bool(
                getattr(self.config, "verify_flat_structure", True)
            ),
        )
        try:
            manifest = reader.manifest
            if manifest.get("recipe_version") != RECIPE_VERSION:
                raise ValueError(
                    "flat capture recipe drift: "
                    f"{manifest.get('recipe_version')!r} != {RECIPE_VERSION!r}"
                )
            if (
                int(reader.hidden_size) != HIDDEN_SIZE
                or int(reader.top_k) != TOP_K
                or int(reader.num_experts) != 256
            ):
                raise ValueError(
                    "flat capture geometry drift: "
                    f"hidden={reader.hidden_size} top_k={reader.top_k} "
                    f"experts={reader.num_experts}"
                )
            audit = build_flat_mass_audit(
                reader,
                layer=value,
                cross_check_rows=(
                    4096
                    if bool(getattr(self.config, "verify_flat_mass", True))
                    else 0
                ),
            )
            if audit.layer != value or audit.tokens != int(reader.tokens):
                raise ValueError("flat capture mass audit binding drift")
            prefix = layer_dir.name
            hashes = {
                f"{prefix}/{X_FILE}": str(manifest["sha256_x"]),
                f"{prefix}/{IDS_FILE}": str(manifest["sha256_ids"]),
                f"{prefix}/{WEIGHTS_FILE}": str(manifest["sha256_weights"]),
                f"{prefix}/{MANIFEST_FILE}": sha256_file(manifest_path),
            }
            for name, digest in hashes.items():
                if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError(f"flat capture digest is malformed: {name}")
            capture = LayerCapture(
                layer=value,
                state_sha256=hashes[f"{prefix}/{X_FILE}"],
                routing_dir=layer_dir,
                mass_audit=audit,
                routing_sha256=hashes,
            )
            registry = getattr(self, "_flat_captures", None)
            if registry is None:
                registry = self._flat_captures = {}
            registry[capture.digest] = reader
            if not bool(getattr(self.config, "flat_direct_rows", False)):
                batches = build_flat_bound_batches(reader, layer=value)
                # `_bound_batches_cached` keys on (capture digest, bound shard IDs).
                # The flat path binds no state shards, so pre-seeding that exact key
                # makes the existing row iterators consume these blocks and never
                # reach `_iter_bound_batches_raw`, which reads sealed state files.
                self._bound_cache = ((capture.digest, ()), batches)
            else:
                # R10 indexes routed rows once in FlatCaptureReader instead of
                # scanning every token block once per expert.
                self._bound_cache = None
                # Preserve the exact fit/holdout index tensors for this reader.
                # Search, probes, and final encode ask for the same expert split
                # repeatedly; recomputing the SHA-derived Python membership loop
                # each time was both expensive and unnecessary.
                reader._r10_split_index_cache = {}
                reader._r10_fallback_index_cache = {}
            self._expert_rows_memo = {}
        except BaseException:
            reader.close()
            raise
        return capture

    def _validate_capture_arithmetic_audit(
        self, audit: Mapping[str, object], *, layer: int
    ) -> None:
        expected_keys = {
            "schema",
            "layer",
            "runtime_fingerprint",
            "attention_implementation",
            "dispatch_audit_sha256",
            "tensor_records_sha256",
            "tensor_count",
            "passed",
        }
        if set(audit) != expected_keys:
            raise ValueError("carried-layer arithmetic audit key-set drift")
        if (
            audit["schema"] != "r7-carried-layer-arithmetic-v1"
            or int(audit["layer"]) != layer
            or audit["runtime_fingerprint"] != self.runtime.fingerprint
            or audit["attention_implementation"] != "eager"
            or not isinstance(audit["dispatch_audit_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", audit["dispatch_audit_sha256"]) is None
            or int(audit["tensor_count"]) <= 0
            or audit["passed"] is not True
            or not isinstance(audit["tensor_records_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", audit["tensor_records_sha256"]) is None
        ):
            raise ValueError("carried-layer arithmetic audit failed")

    @staticmethod
    def _split(layer: int, expert: int, row_id: int) -> str:
        return (
            "holdout"
            if derive_seed(layer, expert, row_id, "fit-holdout") % 5 == 0
            else "fit"
        )

    def _bound_batches_cached(self, capture: LayerCapture, shards):
        """Load the layer's bound batches ONCE into RAM and reuse.

        The prior design re-read 1,773 state files plus 1,773 routing
        sidecars from disk on every row request; the search phase issues
        dozens of such requests per layer. One resident copy (~13 GB
        against 2 TB of RAM) serves them all. Keyed by capture digest so
        layer N+1 replaces layer N; contents are treated as read-only.
        """
        key = (capture.digest, tuple(shard.shard_id for shard in shards))
        cached = getattr(self, "_bound_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        batches = list(self._iter_bound_batches_raw(capture, shards))
        self._bound_cache = (key, batches)
        # a new layer invalidates every per-expert row memo as well
        self._expert_rows_memo = {}
        return batches

    def _iter_bound_batches(self, capture: LayerCapture, shards: Iterable[StateShard]):
        yield from self._bound_batches_cached(capture, tuple(shards))

    def _iter_bound_batches_raw(self, capture: LayerCapture, shards: Iterable[StateShard]):
        import torch

        manifest = read_json(capture.routing_dir / "CAPTURE.json")
        if (
            manifest.get("layer") != capture.layer
            or manifest.get("state_sha256") != capture.state_sha256
        ):
            raise ValueError("routing capture/state binding drift")
        records = manifest.get("sidecars")
        if not isinstance(records, dict):
            raise ValueError("routing capture has no sidecar map")
        values = tuple(shards)
        if [shard.shard_id for shard in values] != sorted(
            shard.shard_id for shard in values
        ) or len({shard.shard_id for shard in values}) != len(values):
            raise ValueError("bound state shards are not a canonical unique domain")
        if set(records) != {shard.shard_id for shard in values}:
            raise ValueError("routing/state shard domains differ")
        previous_end = -1
        for shard in values:
            record = records.get(shard.shard_id)
            if (
                not isinstance(record, dict)
                or record.get("state_hidden_sha256") != shard.sha256_hidden
                or record.get("state_metadata_sha256") != shard.sha256_metadata
                or int(record.get("tokens", -1)) != shard.tokens
            ):
                raise ValueError("routing sidecar/state shard binding drift")
            # Validate the sealed state header and metadata contract without
            # materializing either large tensor. Row planning consumes only the
            # post-attention MoE input already sealed in the routing sidecar.
            self._validate_record(
                (
                    shard.shard_id,
                    shard.hidden_path,
                    shard.metadata_path,
                    shard.tokens,
                    shard.hidden_size,
                ),
                shard.hidden_path.parent,
            )
            metadata = read_json(shard.metadata_path)
            start = int(metadata["global_row_start"])
            end = start + shard.tokens
            if start <= previous_end:
                raise ValueError("bound state row-ID ranges overlap or are unordered")
            previous_end = end - 1
            expected_row_ids = torch.arange(start, end, dtype=torch.int64)
            sidecar_name = str(record.get("file", ""))
            if not sidecar_name or Path(sidecar_name).name != sidecar_name:
                raise ValueError("routing sidecar path escapes its capture directory")
            sidecar = capture.routing_dir / sidecar_name
            ids, weights, moe_hidden, row_ids = self._read_sidecar(sidecar)
            if not torch.equal(row_ids, expected_row_ids):
                raise ValueError("state/routing row-ID drift")
            yield (
                shard,
                CalibrationBatch(
                    shard_id=shard.shard_id,
                    hidden=moe_hidden,
                    row_ids=row_ids,
                    attention_metadata=metadata,
                    token_count=shard.tokens,
                ),
                ids,
                weights,
                row_ids,
            )

    def clear_expert_row_memo(self, capture: LayerCapture, expert: int) -> None:
        """Release copied selected rows for one completed worker task.

        The layer-wide bound batches remain mmap-backed and reusable. Only the
        expert-specific advanced-index results are evicted, bounding each worker
        to the experts it is actively processing.
        """

        if not 0 <= int(expert) < 256:
            raise ValueError("expert is outside [0,256)")
        memo = getattr(self, "_expert_rows_memo", None)
        if memo is None:
            return
        for split in ("fit", "holdout"):
            memo.pop((capture.digest, int(expert), split), None)

    def iter_expert_rows(
        self,
        *,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        split: str,
    ) -> Iterator[ExpertRows]:
        import torch

        if split not in ("fit", "holdout"):
            raise ValueError("row split must be fit or holdout")
        registry = getattr(self, "_flat_captures", {})
        reader = registry.get(capture.digest)
        if reader is not None and bool(getattr(self.config, "flat_direct_rows", False)):
            split_cache = getattr(reader, "_r10_split_index_cache", None)
            if split_cache is None:
                split_cache = reader._r10_split_index_cache = {}
            cache_key = (int(expert), split)
            selected = split_cache.get(cache_key)
            if selected is None:
                rows = reader.expert_rows(expert)
                # Preserve the exact historical SHA-derived membership split.
                # Populate both halves in one pass so each reader/expert hashes
                # its row IDs once across search, probes, and final encode.
                holdout = torch.tensor(
                    [
                        self._split(capture.layer, expert, int(row_id))
                        == "holdout"
                        for row_id in rows.tolist()
                    ],
                    dtype=torch.bool,
                )
                split_cache[(int(expert), "holdout")] = rows[holdout]
                split_cache[(int(expert), "fit")] = rows[~holdout]
                selected = split_cache[cache_key]
            chunk = int(getattr(self.config, "flat_gather_rows", 4096))
            for start in range(0, int(selected.numel()), chunk):
                index = selected[start : start + chunk]
                ids = reader.ids.index_select(0, index)
                weights = reader.weights.index_select(0, index)
                route_weight = (weights * (ids == expert)).sum(dim=1)
                hidden = reader.hidden.index_select(0, index)
                yield ExpertRows(hidden, route_weight, index)
            return
        memo = getattr(self, "_expert_rows_memo", None)
        if memo is None:
            memo = self._expert_rows_memo = {}
        memo_key = (capture.digest, expert, split)
        cached_rows = memo.get(memo_key)
        if cached_rows is not None:
            yield from cached_rows
            return
        collected = []
        for _, batch, ids, weights, row_ids in self._iter_bound_batches(
            capture, shards
        ):
            chosen = ids == expert
            selected = chosen.any(dim=1)
            if not selected.any():
                continue
            # The split only matters where the expert selected the row, so
            # compute the per-row hash on the ~selected subset instead of the
            # full batch. Identical outcome: mask == selected & split_mask.
            sel_index = selected.nonzero(as_tuple=True)[0]
            keep = torch.tensor(
                [
                    self._split(capture.layer, expert, int(row_ids[int(i)]))
                    == split
                    for i in sel_index
                ],
                dtype=torch.bool,
            )
            final_index = sel_index[keep]
            if final_index.numel():
                route_weight = (weights * chosen).sum(dim=1)
                rows = ExpertRows(
                    batch.hidden[final_index],
                    route_weight[final_index],
                    row_ids[final_index],
                )
                collected.append(rows)
                yield rows
        memo[memo_key] = tuple(collected)

    def _fallback_locations(
        self,
        capture: LayerCapture,
        shards: tuple[StateShard, ...],
        expert: int,
        split: str,
        limit: int = 8192,
    ) -> set[tuple[str, int]]:
        # Four equal deterministic reservoirs stratified by maximum selected
        # route-weight share; this is fallback calibration, never fake expert
        # mass.  A global reservoir tops up sparse bins so a one-bin routing
        # distribution cannot silently return only one quarter of `limit`.
        per_bin = limit // 4
        heaps: list[list[tuple[int, str, int]]] = [[] for _ in range(4)]
        global_heap: list[tuple[int, str, int]] = []
        for shard, _, _, weights, row_ids in self._iter_bound_batches(capture, shards):
            maxima = weights.max(dim=1).values.tolist()
            totals = weights.sum(dim=1).tolist()
            for offset, (row_id, maximum, total) in enumerate(
                zip(row_ids.tolist(), maxima, totals)
            ):
                if self._split(capture.layer, expert, int(row_id)) != split:
                    continue
                share = float(maximum) / max(float(total), 1e-30)
                bucket = min(3, int(share * 4.0))
                priority = derive_seed(capture.layer, expert, row_id, split, "fallback")
                item = (-priority, shard.shard_id, offset)
                heap = heaps[bucket]
                if len(heap) < per_bin:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
                if len(global_heap) < limit:
                    heapq.heappush(global_heap, item)
                elif item > global_heap[0]:
                    heapq.heapreplace(global_heap, item)
        selected = {
            (shard_id, offset) for heap in heaps for _, shard_id, offset in heap
        }
        for _, shard_id, offset in sorted(global_heap, reverse=True):
            if len(selected) >= limit:
                break
            selected.add((shard_id, offset))
        if not selected:
            raise ValueError("cold fallback reservoir is empty")
        return selected

    def iter_cold_fallback_rows(
        self,
        *,
        capture: LayerCapture,
        shards: Iterable[StateShard],
        expert: int,
        split: str,
    ) -> Iterator[ExpertRows]:
        import torch

        values = tuple(shards)
        registry = getattr(self, "_flat_captures", {})
        reader = registry.get(capture.digest)
        if reader is not None and bool(getattr(self.config, "flat_direct_rows", False)):
            # Sparse experts are topped up from a seeded global row permutation.
            # Generate it in C++ and inspect only enough chunks to collect the
            # requested split, avoiding the legacy all-token Python reservoir.
            limit = int(getattr(self.config, "flat_fallback_rows", 8192))
            cache = getattr(reader, "_r10_fallback_index_cache", None)
            if cache is None:
                cache = reader._r10_fallback_index_cache = {}
            cache_key = (int(expert), split, limit)
            rows = cache.get(cache_key)
            if rows is None:
                generator = torch.Generator().manual_seed(
                    derive_seed(capture.layer, expert, split, "r10-flat-fallback")
                )
                order = torch.randperm(int(reader.tokens), generator=generator)
                selected: list[int] = []
                scan = 1 << 16
                for start in range(0, int(order.numel()), scan):
                    for row_id in order[start : start + scan].tolist():
                        if self._split(capture.layer, expert, int(row_id)) == split:
                            selected.append(int(row_id))
                            if len(selected) >= limit:
                                break
                    if len(selected) >= limit:
                        break
                if not selected:
                    raise ValueError("flat cold-fallback reservoir is empty")
                rows = torch.tensor(sorted(selected), dtype=torch.int64)
                cache[cache_key] = rows
                del order
            chunk = int(getattr(self.config, "flat_gather_rows", 4096))
            for start in range(0, int(rows.numel()), chunk):
                index = rows[start : start + chunk]
                weights = reader.weights.index_select(0, index)
                yield ExpertRows(
                    reader.hidden.index_select(0, index),
                    weights.max(dim=1).values,
                    index,
                )
            return
        locations = self._fallback_locations(capture, values, expert, split)
        for shard, batch, _, weights, row_ids in self._iter_bound_batches(
            capture, values
        ):
            offsets = [
                offset
                for offset in range(shard.tokens)
                if (shard.shard_id, offset) in locations
            ]
            if offsets:
                index = torch.tensor(offsets, dtype=torch.int64)
                yield ExpertRows(
                    batch.hidden[index],
                    weights[index].max(dim=1).values,
                    row_ids[index],
                )

    def load_bf16_expert(self, *, layer: int, expert: int) -> ExpertWeights:
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("BF16 expert loads are restricted to layers 3..77")
        if not 0 <= expert < 256:
            raise ValueError("expert is outside [0,256)")
        tensors = {}
        names = {}
        hashes = {}
        records = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            record = self.source_entries[name]
            shard = self.source / str(record["shard"])
            reader = SafeTensorReader(shard)
            info = reader.tensors[name]
            if bool(getattr(self.config, "verify_source_payloads", True)) and (
                info.payload.sha256() != record["payload_sha256"]
            ):
                raise ValueError(f"BF16 source payload changed after inventory: {name}")
            tensors[projection] = _read_tensor(reader, name)
            names[projection] = name
            hashes[projection] = str(record["payload_sha256"])
            records[projection] = dict(record)
        return ExpertWeights(
            gate_hf=tensors["gate_proj"],
            up_hf=tensors["up_proj"],
            down_hf=tensors["down_proj"],
            dtype="BF16",
            source_names=names,
            payload_sha256=hashes,
            source_records=records,
        )

    def install_encoded_expert(
        self, *, layer: int, expert: int, encoded: Mapping[str, Any]
    ) -> Mapping[str, object]:
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("expert replacement is restricted to layers 3..77")
        if not 0 <= expert < 256:
            raise ValueError("expert is outside [0,256)")
        expected_keys = {
            f"L{layer:02d}/E{expert:03d}/{projection}"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if set(encoded) != expected_keys:
            raise ValueError(
                "installed expert must contain exactly gate/up/down records"
            )
        packed = {}
        reconstructed = {}
        for key, raw in encoded.items():
            if not isinstance(raw, Mapping):
                raise TypeError(f"malformed installed tensor record: {key}")
            packed[key] = str(raw["packed_sha256"])
            reconstructed[key] = str(raw["reconstruction_sha256"])
            if not packed[key] or not reconstructed[key]:
                raise ValueError(f"incomplete installed tensor record: {key}")
        raw_audit = self.runtime.install_encoded_expert(
            layer=layer, expert=expert, encoded=encoded
        )
        if not isinstance(raw_audit, Mapping):
            raise TypeError("runtime install must return a packed-decoded receipt")
        audit = dict(raw_audit)
        required = {
            "schema",
            "layer",
            "expert",
            "runtime_fingerprint",
            "activation_dtype",
            "packed_decoded",
            "packed_sha256",
            "reconstruction_sha256",
            "installed_shape_kn",
            "passed",
        }
        if set(audit) != required:
            raise ValueError(
                "runtime install audit key drift: "
                f"missing={sorted(required - set(audit))} "
                f"extra={sorted(set(audit) - required)}"
            )
        expected_shapes = {
            key: [
                HIDDEN_SIZE
                if key.endswith(("gate_proj", "up_proj"))
                else INTERMEDIATE_SIZE,
                INTERMEDIATE_SIZE
                if key.endswith(("gate_proj", "up_proj"))
                else HIDDEN_SIZE,
            ]
            for key in sorted(expected_keys)
        }
        bindings_ok = (
            audit["schema"] == "r7-packed-install-record-v2"
            and int(audit["layer"]) == layer
            and int(audit["expert"]) == expert
            and audit["runtime_fingerprint"] == self.runtime.fingerprint
            and str(audit["activation_dtype"]).upper() == "BF16"
            and audit["packed_decoded"] is True
            and dict(audit["packed_sha256"]) == dict(sorted(packed.items()))
            and dict(audit["reconstruction_sha256"])
            == dict(sorted(reconstructed.items()))
            and dict(audit["installed_shape_kn"]) == expected_shapes
        )
        if not bindings_ok or audit["passed"] is not True:
            raise ValueError("packed-decoded install receipt failed")
        audit["backend_fingerprint"] = self.fingerprint
        return audit

    def audit_installed_layer(self, *, layer: int) -> Mapping[str, object]:
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("installed-layer audit is restricted to layers 3..77")
        raw_audit = self.runtime.audit_installed_layer(layer=layer)
        if not isinstance(raw_audit, Mapping):
            raise TypeError("runtime installed-layer audit is malformed")
        audit = dict(raw_audit)
        required = {
            "schema",
            "layer",
            "runtime_fingerprint",
            "activation_dtype",
            "sample_rows",
            "sample_seed",
            "sample_input_sha256",
            "top_k",
            "topk_indices_sha256",
            "topk_weights_sha256",
            "routed_scaling_factor",
            "unique_experts_per_token",
            "shared_expert_hits",
            "official_module",
            "experts_implementation",
            "dispatch_audit_sha256",
            "official_first_output_sha256",
            "official_second_output_sha256",
            "reference_output_sha256",
            "official_repeat_exact",
            "max_abs_error",
            "relative_l2_error",
            "max_abs_tolerance",
            "relative_l2_tolerance",
            "passed",
        }
        if set(audit) != required:
            raise ValueError(
                "installed-layer audit key drift: "
                f"missing={sorted(required - set(audit))} "
                f"extra={sorted(set(audit) - required)}"
            )
        hashes = (
            "sample_input_sha256",
            "topk_indices_sha256",
            "topk_weights_sha256",
            "dispatch_audit_sha256",
            "official_first_output_sha256",
            "official_second_output_sha256",
            "reference_output_sha256",
        )
        valid_hashes = all(
            isinstance(audit[key], str)
            and re.fullmatch(r"[0-9a-f]{64}", str(audit[key])) is not None
            for key in hashes
        )
        max_abs = float(audit["max_abs_error"])
        relative = float(audit["relative_l2_error"])
        if not (
            audit["schema"] == "r7-official-installed-layer-audit-v1"
            and int(audit["layer"]) == layer
            and audit["runtime_fingerprint"] == self.runtime.fingerprint
            and str(audit["activation_dtype"]).upper() == "BF16"
            and int(audit["sample_rows"]) == INSTALL_AUDIT_ROWS
            and int(audit["sample_seed"])
            == derive_seed(layer, "official-installed-layer-audit-v1")
            and int(audit["top_k"]) == TOP_K
            and audit["unique_experts_per_token"] is True
            and audit["shared_expert_hits"] is True
            and str(audit["official_module"]).endswith(".GlmMoeDsaNaiveMoe")
            and audit["experts_implementation"] == "eager"
            and math.isfinite(float(audit["routed_scaling_factor"]))
            and float(audit["routed_scaling_factor"]) > 0.0
            and audit["official_repeat_exact"] is True
            and audit["official_first_output_sha256"]
            == audit["official_second_output_sha256"]
            and float(audit["max_abs_tolerance"]) == INSTALL_MAX_ABS_ERROR
            and float(audit["relative_l2_tolerance"]) == INSTALL_MAX_RELATIVE_L2
            and 0.0 <= max_abs <= INSTALL_MAX_ABS_ERROR
            and 0.0 <= relative <= INSTALL_MAX_RELATIVE_L2
            and valid_hashes
            and audit["passed"] is True
        ):
            raise ValueError("official installed-layer arithmetic audit failed")
        audit["backend_fingerprint"] = self.fingerprint
        return audit

    def restore_encoded_layer(self, *, layer: int, manifest: Path) -> None:
        if not FIRST_MOE_LAYER <= layer <= LAST_MOE_LAYER:
            raise ValueError("expert restoration is restricted to layers 3..77")
        self.runtime.restore_encoded_layer(layer=layer, manifest=manifest)

    def forward_installed_layer(
        self,
        *,
        layer: int,
        input_shards: Iterable[StateShard],
        output_partial: Path,
        completed_shard_ids: frozenset[str],
    ) -> Iterable[tuple[str, Path, Path, int, int]]:
        if not FIRST_MOE_LAYER <= layer < LAST_MOE_LAYER:
            raise ValueError("successor forwarding is restricted to layers 3..76")
        for record in self.runtime.forward_installed_layer(
            layer=layer,
            input_shards=input_shards,
            output_partial=output_partial,
            completed_shard_ids=completed_shard_ids,
        ):
            self._validate_record(record, output_partial)
            yield record


def factory(config) -> GLM52Backend:
    """CLI factory: `--backend r7_encoder.glm52_backend:factory`."""

    return GLM52Backend(
        config=config, runtime=_load_runtime(config.runtime_factory, config)
    )
