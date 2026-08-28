"""Integrated sequential layer walk with crash-consistent stage boundaries."""

from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import Round7Backend
from .constants import (
    DEFAULT_SIGMA_REG,
    FIRST_MOE_LAYER,
    LAST_MOE_LAYER,
    MOE_LAYERS,
    NUM_EXPERTS,
    RECIPE_MARKER,
    RECIPE_VERSION,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    configure_deterministic_environment,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .layer import LayerProcessor
from .inventory import (
    load_checkpoint_inventory,
    load_numeric_environment,
    load_runtime_code_inventory,
    verify_checkpoint_inventory,
)
from .oracles import audit_v2_layer, shared_vector_oracle
from .schema import emit_layer_v2
from .search import SearchRunner
from .state import Journal, StageSeal, StateStore
from .trellis import CodecConfig, Exl3TrellisCodec


@dataclass(frozen=True)
class WalkConfig:
    carrier: Path
    bf16_source: Path
    corpus: Path
    work: Path
    backend_factory: str
    runtime_factory: str
    runtime_inventory: Path
    carrier_inventory: Path
    source_inventory: Path
    numeric_inventory: Path
    device: str = "cuda:0"
    # Optional device pool for the parallel warm passes (probe/final encodes).
    # When set, devices[0] must equal `device`. Empty tuple = single-device
    # behavior, byte-identical to the original flow.
    devices: tuple[str, ...] = ()
    sigma_reg: float = DEFAULT_SIGMA_REG
    fixed_point_iterations: int = 4
    holdout_rows: int = 4096
    draws: int = 12
    shared_sample_experts: int = 16
    retire_predecessor_state: bool = True

    def canonical(self) -> dict[str, object]:
        value = asdict(self)
        # The warm-pass device pool is an execution detail: it never changes
        # emitted bytes (pool of 1 == original flow; pool of N fills the same
        # per-expert caches). Excluding it keeps resume-with-a-different-pool
        # legal under the same recipe identity.
        value.pop("devices", None)
        for key in (
            "carrier",
            "bf16_source",
            "corpus",
            "work",
            "carrier_inventory",
            "source_inventory",
            "numeric_inventory",
            "runtime_inventory",
        ):
            value[key] = str(value[key])
        value.update(
            {
                "marker": RECIPE_MARKER,
                "recipe_version": RECIPE_VERSION,
                "moe_layers": [FIRST_MOE_LAYER, LAST_MOE_LAYER],
                "mtp_layer_carried": 78,
            }
        )
        return value


def _read_only_fingerprint(model_dir: Path) -> str:
    files = []
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required model metadata missing: {path}")
        files.append((name, sha256_file(path)))
    return sha256_bytes(canonical_json_bytes(files))


def _load_backend(factory_path: str, config: WalkConfig) -> Round7Backend:
    if ":" not in factory_path:
        raise ValueError("backend factory must be `module:callable`")
    module_name, attribute = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    backend = factory(config)
    if not isinstance(backend, Round7Backend):
        raise TypeError("backend factory did not return Round7Backend")
    if not backend.fingerprint:
        raise ValueError("backend fingerprint is empty")
    return backend


class SequentialWalk:
    def __init__(
        self, config: WalkConfig, backend: Round7Backend | None = None
    ) -> None:
        configure_deterministic_environment()
        self.config = config
        self._validate_paths()
        self.work = config.work.resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.carrier_inventory = load_checkpoint_inventory(
            config.carrier_inventory, role="carrier"
        )
        self.source_inventory = load_checkpoint_inventory(
            config.source_inventory,
            role="bf16-source",
            require_routed_bf16=True,
        )
        self.numeric_inventory = load_numeric_environment(config.numeric_inventory)
        self.runtime_inventory = load_runtime_code_inventory(config.runtime_inventory)
        verify_checkpoint_inventory(config.carrier, self.carrier_inventory)
        verify_checkpoint_inventory(config.bf16_source, self.source_inventory)
        self.backend = backend or _load_backend(config.backend_factory, config)
        recipe = config.canonical()
        recipe.update(
            {
                "carrier_fingerprint": _read_only_fingerprint(config.carrier.resolve()),
                "bf16_source_fingerprint": _read_only_fingerprint(
                    config.bf16_source.resolve()
                ),
                "corpus_sha256": sha256_file(config.corpus.resolve()),
                "backend_fingerprint": self.backend.fingerprint,
                "carrier_inventory_sha256": self.carrier_inventory["inventory_sha256"],
                "source_inventory_sha256": self.source_inventory["inventory_sha256"],
                "numeric_environment_sha256": self.numeric_inventory[
                    "inventory_sha256"
                ],
                "runtime_inventory_sha256": self.runtime_inventory["inventory_sha256"],
            }
        )
        self.journal = Journal(self.work / "JOURNAL.json", recipe_config=recipe)
        self.states = StateStore(self.work / "states")

        def _codec_for(device: str) -> Exl3TrellisCodec:
            return Exl3TrellisCodec(
                CodecConfig(
                    device=device,
                    sigma_reg=config.sigma_reg,
                    numeric_core=Path(str(self.numeric_inventory["numeric_core"])),
                    numeric_core_sha256=str(
                        self.numeric_inventory["numeric_core_sha256"]
                    ),
                    extension=Path(str(self.numeric_inventory["extension"])),
                    extension_sha256=str(self.numeric_inventory["extension_sha256"]),
                )
            )

        pool_devices = tuple(config.devices) or (config.device,)
        if pool_devices[0] != config.device:
            raise ValueError("devices[0] must equal the primary --device")
        if len(set(pool_devices)) != len(pool_devices):
            raise ValueError("devices must be unique")
        self.codec = _codec_for(pool_devices[0])
        self.codecs = (self.codec,) + tuple(
            _codec_for(device) for device in pool_devices[1:]
        )
        self.process_pool = None
        # Fail closed on a stale/corrupt resume before creating new work.
        self.journal.audit_outputs(self.work)

    def _validate_paths(self) -> None:
        carrier = self.config.carrier.resolve()
        source = self.config.bf16_source.resolve()
        work = self.config.work.resolve()
        if carrier == source:
            raise ValueError(
                "carrier and BF16 expert source must be explicit separate roles"
            )
        for protected in (carrier, source):
            if (
                work == protected
                or protected in work.parents
                or work in protected.parents
            ):
                raise ValueError(
                    f"work directory must be disjoint from protected model {protected}"
                )
        if not self.config.corpus.resolve().is_file():
            raise FileNotFoundError(self.config.corpus)

    def initialize(self) -> None:
        stage = "state-input-003"
        if self.journal.has(stage):
            self.states.load(FIRST_MOE_LAYER)
            return
        corpus_plan = self.backend.prepare_corpus_plan(
            corpus=self.config.corpus.resolve()
        )
        predecessor = sha256_bytes(
            canonical_json_bytes(
                {
                    "carrier_inventory_sha256": self.carrier_inventory[
                        "inventory_sha256"
                    ],
                    "corpus_sha256": sha256_file(self.config.corpus.resolve()),
                    "backend_fingerprint": self.backend.fingerprint,
                    "corpus_plan_sha256": corpus_plan["corpus_plan_sha256"],
                    "corpus_plan_artifact_sha256": corpus_plan[
                        "corpus_plan_artifact_sha256"
                    ],
                    "carried_reconstruction_boundary": [0, 2],
                }
            )
        )
        adopted = self.states.adopt_existing(
            FIRST_MOE_LAYER,
            predecessor_sha256=predecessor,
            backend_fingerprint=self.backend.fingerprint,
        )
        if adopted is not None:
            relative = self.states.seal_archive_path(FIRST_MOE_LAYER).relative_to(
                self.work
            )
            plan_relative = Path("CORPUS_PLAN.json")
            self.journal.seal(
                StageSeal(
                    name=stage,
                    input_sha256={
                        "carrier": _read_only_fingerprint(
                            self.config.carrier.resolve()
                        ),
                        "corpus": sha256_file(self.config.corpus.resolve()),
                    },
                    output_sha256={
                        str(relative): adopted,
                        str(plan_relative): str(
                            corpus_plan["corpus_plan_artifact_sha256"]
                        ),
                    },
                    metadata={
                        "through_carried_layer": 2,
                        "next_layer": 3,
                        "adopted": True,
                    },
                )
            )
            return
        transition = self.states.begin_transition(
            FIRST_MOE_LAYER,
            corpus_plan_sha256=str(corpus_plan["corpus_plan_sha256"]),
            expected_shards=corpus_plan["expected_shards"],
        )
        records = self.backend.initialize_carried_state(
            carrier=self.config.carrier.resolve(),
            corpus=self.config.corpus.resolve(),
            output_partial=transition.temporary,
            completed_shard_ids=transition.completed_shard_ids,
        )
        for shard_id, hidden, metadata, tokens, hidden_size in records:
            transition.add_existing_shard(
                shard_id=shard_id,
                hidden_path=hidden,
                metadata_path=metadata,
                tokens=tokens,
                hidden_size=hidden_size,
            )
        digest = transition.commit(
            predecessor_sha256=predecessor,
            backend_fingerprint=self.backend.fingerprint,
        )
        relative = self.states.seal_archive_path(FIRST_MOE_LAYER).relative_to(self.work)
        plan_relative = Path("CORPUS_PLAN.json")
        self.journal.seal(
            StageSeal(
                name=stage,
                input_sha256={
                    "carrier": _read_only_fingerprint(self.config.carrier.resolve()),
                    "corpus": sha256_file(self.config.corpus.resolve()),
                },
                output_sha256={
                    str(relative): digest,
                    str(plan_relative): str(corpus_plan["corpus_plan_artifact_sha256"]),
                },
                metadata={"through_carried_layer": 2, "next_layer": 3},
            )
        )

    def _run_layer(self, layer: int) -> None:
        if layer == LAST_MOE_LAYER and self.journal.has("walk-complete-at-layer-077"):
            return
        if layer < LAST_MOE_LAYER and self.journal.has(f"state-input-{layer + 1:03d}"):
            # The successor seal proves the layer was encoded and forwarded;
            # its predecessor state may already have been retired.
            self.states.load(layer + 1)
            return
        input_stage = f"state-input-{layer:03d}"
        self.journal.require(input_stage)
        state_shards = self.states.load(layer)
        state_manifest = self.states.manifest_path(layer)
        state_sha = sha256_file(state_manifest)

        encode_stage = f"layer-{layer:03d}-encoded"
        layer_output = self.work / "v2"
        manifest_path = layer_output / f"r7-experts-layer-{layer:03d}.json"
        if self.journal.has(encode_stage):
            self.backend.restore_encoded_layer(layer=layer, manifest=manifest_path)
        else:
            routing_dir = self.work / f"layer-{layer:03d}" / "routing"
            capture_stage = f"layer-{layer:03d}-capture"
            capture_stage_sealed = self.journal.has(capture_stage)
            if not capture_stage_sealed:
                routing_dir.mkdir(parents=True, exist_ok=True)
                if (routing_dir / "CAPTURE.json").is_file():
                    capture = self.backend.open_capture(
                        layer=layer, routing_dir=routing_dir
                    )
                else:
                    capture = self.backend.capture_layer(
                        layer=layer,
                        shards=state_shards,
                        routing_dir=routing_dir,
                    )
                if capture.layer != layer or capture.state_sha256 != state_sha:
                    raise ValueError(
                        "backend capture provenance does not match sealed state"
                    )
                self.journal.seal(
                    StageSeal(
                        name=capture_stage,
                        input_sha256={
                            str(
                                self.states.seal_archive_path(layer).relative_to(
                                    self.work
                                )
                            ): state_sha
                        },
                        output_sha256=dict(capture.routing_sha256),
                        metadata=capture.mass_audit.to_json(),
                    )
                )
            else:
                # A journal-sealed resume must re-open and verify the immutable
                # routing sidecars. Fresh captures are already fully verified
                # by capture_layer/open_capture above and need no second pass.
                capture = self.backend.open_capture(
                    layer=layer, routing_dir=routing_dir
                )
            sealed_capture = self.journal.require(capture_stage)
            if dict(capture.routing_sha256) != dict(sealed_capture.output_sha256):
                raise ValueError("routing sidecar drift on resume")

            processor = LayerProcessor(
                backend=self.backend,
                codec=self.codec,
                codecs=self.codecs,
                work_dir=self.work,
                device=self.config.device,
                sigma_reg=self.config.sigma_reg,
                fixed_point_iterations=self.config.fixed_point_iterations,
                holdout_rows=self.config.holdout_rows,
                source_inventory_sha256=str(self.source_inventory["inventory_sha256"]),
                numeric_environment_sha256=str(
                    self.numeric_inventory["inventory_sha256"]
                ),
                runtime_inventory_sha256=str(
                    self.runtime_inventory["inventory_sha256"]
                ),
                process_pool=self.process_pool,
            )
            search_path = self.work / "search" / f"layer-{layer:03d}.json"
            search = SearchRunner(
                processor=processor,
                backend=self.backend,
                capture=capture,
                shards=state_shards,
                output=search_path,
                draws=self.config.draws,
                shared_sample_experts=self.config.shared_sample_experts,
                process_pool=self.process_pool,
            ).run()
            search_sha = sha256_file(search_path)
            search_stage = f"layer-{layer:03d}-search"
            self.journal.seal(
                StageSeal(
                    name=search_stage,
                    input_sha256={
                        str(
                            self.states.seal_archive_path(layer).relative_to(self.work)
                        ): state_sha,
                        **dict(capture.routing_sha256),
                    },
                    output_sha256={str(search_path.relative_to(self.work)): search_sha},
                    metadata={
                        "draws": search.draws,
                        "shared_sample_experts": self.config.shared_sample_experts,
                        "pilot_evidence_sha256": search.pilot_evidence_sha256,
                    },
                )
            )
            result = processor.run(
                capture=capture,
                shards=state_shards,
                search=search,
                search_artifact_sha256=search_sha,
            )
            shared_vector_oracle(result.encoded)
            manifest = emit_layer_v2(
                layer_output,
                layer=layer,
                encoded_tensors=result.encoded,
                shared_gate_up_suh=result.shared_gate_up_suh,
                shared_down_svh=result.shared_down_svh,
                allocation_bits=result.allocation.bits,
                layer_provenance={
                    "state_sha256": state_sha,
                    "search_sha256": search_sha,
                    "routing_sha256": dict(capture.routing_sha256),
                    "fixed_point_iterations": result.fixed_point_iterations,
                    "allocation_sha256": sha256_bytes(
                        canonical_json_bytes(
                            dict(sorted(result.allocation.bits.items()))
                        )
                    ),
                    "probe_sha256": result.allocation.probe_sha256,
                    "gate_up_roundtrip_sha256": dict(result.final_gate_up_sha256),
                    "source_inventory_sha256": self.source_inventory[
                        "inventory_sha256"
                    ],
                    "carrier_inventory_sha256": self.carrier_inventory[
                        "inventory_sha256"
                    ],
                    "numeric_environment_sha256": self.numeric_inventory[
                        "inventory_sha256"
                    ],
                    "runtime_inventory_sha256": self.runtime_inventory[
                        "inventory_sha256"
                    ],
                    "backend_fingerprint": self.backend.fingerprint,
                    "capture_sha256": capture.digest,
                    "interaction_audit": dict(result.interaction_audit),
                    "permutation_audit": dict(result.permutation_audit),
                    "probe_artifact_sha256": dict(result.probe_artifacts),
                    "install_audit_sha256": result.install_audit_sha256,
                },
                permutations={
                    expert: search.experts[expert].permutation
                    for expert in range(NUM_EXPERTS)
                },
                permutation_policies={
                    expert: search.experts[expert].permutation_policy
                    for expert in range(NUM_EXPERTS)
                },
                final_expert_artifacts=result.final_expert_artifacts,
            )
            # TP=16 is the most fragmented legal common topology; equality at
            # those 128-aligned pieces also covers every coarser divisor.
            oracle_result = audit_v2_layer(
                manifest_path,
                codec=self.codec,
                tp_sizes=(16,),
                process_pool=self.process_pool,
            )
            oracle_path = self.work / f"layer-{layer:03d}" / "oracle-report.json"
            from .oracles import write_oracle_report

            write_oracle_report(oracle_path, [oracle_result])
            shard_path = layer_output / manifest["shard"]
            allocation_path = (
                self.work
                / f"layer-{layer:03d}"
                / (
                    f"allocation-iter-{result.allocation.fixed_point_iteration:02d}.json"
                )
            )
            cold_audit_path = (
                self.work / f"layer-{layer:03d}" / "cold-fallback-audit.json"
            )
            interaction_path = (
                self.work / f"layer-{layer:03d}" / "interaction-audit.json"
            )
            interaction_partial_path = (
                self.work / f"layer-{layer:03d}" / "interaction-partial.json"
            )
            install_audit_path = self.work / f"layer-{layer:03d}" / "install-audit.json"
            final_cache_outputs = {
                str(path.relative_to(self.work)): sha256_file(path)
                for path in sorted(
                    (self.work / f"layer-{layer:03d}" / "final-experts").iterdir()
                )
                if path.is_file()
            }
            probe_outputs = {
                str(
                    (self.work / f"layer-{layer:03d}" / name).relative_to(self.work)
                ): digest
                for name, digest in result.probe_artifacts.items()
            }
            self.journal.seal(
                StageSeal(
                    name=encode_stage,
                    input_sha256={
                        str(
                            self.states.seal_archive_path(layer).relative_to(self.work)
                        ): state_sha,
                        str(search_path.relative_to(self.work)): search_sha,
                        **dict(capture.routing_sha256),
                    },
                    output_sha256={
                        str(shard_path.relative_to(self.work)): sha256_file(shard_path),
                        str(manifest_path.relative_to(self.work)): sha256_file(
                            manifest_path
                        ),
                        str(allocation_path.relative_to(self.work)): sha256_file(
                            allocation_path
                        ),
                        str(cold_audit_path.relative_to(self.work)): sha256_file(
                            cold_audit_path
                        ),
                        str(interaction_path.relative_to(self.work)): sha256_file(
                            interaction_path
                        ),
                        str(
                            interaction_partial_path.relative_to(self.work)
                        ): sha256_file(interaction_partial_path),
                        str(install_audit_path.relative_to(self.work)): sha256_file(
                            install_audit_path
                        ),
                        str(oracle_path.relative_to(self.work)): sha256_file(
                            oracle_path
                        ),
                        **final_cache_outputs,
                        **probe_outputs,
                    },
                    metadata={
                        "layer": layer,
                        "bit_units": sum(result.allocation.bits.values()),
                        "target_bpw": "3.5",
                        "joint_down_k": 2048,
                    },
                )
            )

        if layer == LAST_MOE_LAYER:
            # Owner-locked boundary: never forward into carried MTP layer 78.
            self.journal.seal(
                StageSeal(
                    name="walk-complete-at-layer-077",
                    input_sha256={
                        str(manifest_path.relative_to(self.work)): sha256_file(
                            manifest_path
                        )
                    },
                    output_sha256={},
                    metadata={"mtp_layer_78": "carried-not-forwarded"},
                )
            )
            return

        successor = layer + 1
        predecessor = sha256_file(manifest_path)
        adopted = self.states.adopt_existing(
            successor,
            predecessor_sha256=predecessor,
            backend_fingerprint=self.backend.fingerprint,
        )
        next_archive = self.states.seal_archive_path(successor)
        if adopted is not None:
            self.journal.seal(
                StageSeal(
                    name=f"state-input-{successor:03d}",
                    input_sha256={
                        str(manifest_path.relative_to(self.work)): predecessor,
                        str(
                            self.states.seal_archive_path(layer).relative_to(self.work)
                        ): state_sha,
                    },
                    output_sha256={str(next_archive.relative_to(self.work)): adopted},
                    metadata={
                        "quantized_predecessor_layer": layer,
                        "next_layer": successor,
                        "adopted": True,
                    },
                )
            )
            if self.config.retire_predecessor_state:
                self.states.retire(layer)
            return
        corpus_plan_sha256, expected_shards = self.states.transition_domain(layer)
        transition = self.states.begin_transition(
            successor,
            corpus_plan_sha256=corpus_plan_sha256,
            expected_shards=expected_shards,
        )
        records = self.backend.forward_installed_layer(
            layer=layer,
            input_shards=state_shards,
            output_partial=transition.temporary,
            completed_shard_ids=transition.completed_shard_ids,
        )
        for shard_id, hidden, metadata, tokens, hidden_size in records:
            transition.add_existing_shard(
                shard_id=shard_id,
                hidden_path=hidden,
                metadata_path=metadata,
                tokens=tokens,
                hidden_size=hidden_size,
            )
        next_digest = transition.commit(
            predecessor_sha256=predecessor,
            backend_fingerprint=self.backend.fingerprint,
        )
        next_manifest = self.states.seal_archive_path(successor)
        self.journal.seal(
            StageSeal(
                name=f"state-input-{successor:03d}",
                input_sha256={
                    str(manifest_path.relative_to(self.work)): sha256_file(
                        manifest_path
                    ),
                    str(
                        self.states.seal_archive_path(layer).relative_to(self.work)
                    ): state_sha,
                },
                output_sha256={str(next_manifest.relative_to(self.work)): next_digest},
                metadata={
                    "quantized_predecessor_layer": layer,
                    "next_layer": successor,
                },
            )
        )
        if self.config.retire_predecessor_state:
            self.states.retire(layer)

    def run(self, *, pilot_stop_after_layer: int | None = None) -> dict[str, object]:
        if (
            pilot_stop_after_layer is not None
            and pilot_stop_after_layer not in MOE_LAYERS
        ):
            raise ValueError("pilot stop layer must be one of the routed layers 3..77")
        started_ns = time.perf_counter_ns()
        initial_seals = tuple(sorted(self.journal.seals))
        self.initialize()
        initialization_ns = time.perf_counter_ns() - started_ns
        layer_elapsed_ns: dict[int, int] = {}
        pool_devices = tuple(self.config.devices)
        if len(pool_devices) > 1:
            from .process_pool import PinnedDeviceProcessPool

            self.process_pool = PinnedDeviceProcessPool(self.config, pool_devices)
            print(
                "R8_PROCESS_POOL "
                + canonical_json_bytes(self.process_pool.worker_info).decode("utf-8"),
                flush=True,
            )
        try:
            for layer in MOE_LAYERS:
                layer_started_ns = time.perf_counter_ns()
                self._run_layer(layer)
                layer_elapsed_ns[layer] = time.perf_counter_ns() - layer_started_ns
                timing_payload = {
                    "schema": "r8-layer-wall-v1",
                    "initialization_seconds": format(
                        initialization_ns / 1_000_000_000, ".6f"
                    ),
                    "layers": {
                        str(index): format(value / 1_000_000_000, ".6f")
                        for index, value in sorted(layer_elapsed_ns.items())
                    },
                }
                atomic_write_json(self.work / "R8_LAYER_TIMINGS.json", timing_payload)
                print(
                    f"R8_LAYER_WALL layer={layer} "
                    f"seconds={layer_elapsed_ns[layer] / 1_000_000_000:.6f}",
                    flush=True,
                )
                if layer == pilot_stop_after_layer:
                    return self._write_pilot_report(
                        layer=layer,
                        elapsed_ns=time.perf_counter_ns() - started_ns,
                        initialization_ns=initialization_ns,
                        layer_elapsed_ns=layer_elapsed_ns,
                        initial_seals=initial_seals,
                    )
            self._finalize_walk()
            return {
                "passed": True,
                "complete": True,
                "last_layer": LAST_MOE_LAYER,
                "walk_manifest": str(self.work / "WALK_COMPLETE.json"),
            }
        finally:
            if self.process_pool is not None:
                self.process_pool.close()
                self.process_pool = None

    def _write_pilot_report(
        self,
        *,
        layer: int,
        elapsed_ns: int,
        initialization_ns: int,
        layer_elapsed_ns: dict[int, int],
        initial_seals: tuple[str, ...],
    ) -> dict[str, object]:
        """Write non-algorithmic timing evidence after a fully sealed layer.

        This report is intentionally outside the deterministic journal and output
        manifests.  It records wall time and storage for rental planning; it can
        never stand in for a complete walk, a quality evaluation, or a serving
        validation.
        """

        encoded = self.journal.require(f"layer-{layer:03d}-encoded")
        manifest = self.work / "v2" / f"r7-experts-layer-{layer:03d}.json"
        oracle = self.work / f"layer-{layer:03d}" / "oracle-report.json"
        search = self.work / "search" / f"layer-{layer:03d}.json"
        total_bytes = sum(
            path.stat().st_size
            for path in self.work.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        payload: dict[str, object] = {
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "schema": "r7-sample-layer-feasibility-v1",
            "measurement_scope": "initialization-through-one-or-more-sealed-routed-layers",
            "first_layer": FIRST_MOE_LAYER,
            "last_measured_layer": layer,
            "completed_layer_count": layer - FIRST_MOE_LAYER + 1,
            "elapsed_seconds": format(elapsed_ns / 1_000_000_000, ".6f"),
            "initialization_seconds": format(initialization_ns / 1_000_000_000, ".6f"),
            "layer_seconds": {
                str(index): format(value / 1_000_000_000, ".6f")
                for index, value in sorted(layer_elapsed_ns.items())
            },
            "work_bytes_after_measurement": total_bytes,
            "device": self.config.device,
            "started_with_existing_seals": list(initial_seals),
            "timing_valid_for_projection": not initial_seals,
            "projection_status": "UNVERIFIED",
            "quality_status": "UNVERIFIED-not-an-evaluation",
            "assembly_conversion_evaluation_included": False,
            "bindings": {
                "recipe_sha256": self.journal.recipe_sha256,
                "journal_sha256": sha256_file(self.journal.path),
                "encoded_stage_sha256": encoded.digest,
                "layer_manifest_sha256": sha256_file(manifest),
                "oracle_report_sha256": sha256_file(oracle),
                "search_sha256": sha256_file(search),
                "carrier_inventory_sha256": self.carrier_inventory["inventory_sha256"],
                "source_inventory_sha256": self.source_inventory["inventory_sha256"],
                "numeric_environment_sha256": self.numeric_inventory[
                    "inventory_sha256"
                ],
                "runtime_inventory_sha256": self.runtime_inventory["inventory_sha256"],
            },
            "owner_inputs_still_required": [
                "rental_hourly_rate",
                "storage_and_egress_price",
                "external_GPU_utilization_and_VRAM_telemetry",
            ],
        }
        report_path = self.work / f"FEASIBILITY_LAYER_{layer:03d}.json"
        payload["report"] = str(report_path)
        payload["passed"] = True
        payload["complete"] = False
        atomic_write_json(report_path, payload)
        return payload

    def _finalize_walk(self) -> None:
        stage = "round7-walk-complete"
        manifest_path = self.work / "WALK_COMPLETE.json"
        layer_records = {}
        for layer in MOE_LAYERS:
            self.journal.require(f"layer-{layer:03d}-capture")
            self.journal.require(f"layer-{layer:03d}-search")
            encoded = self.journal.require(f"layer-{layer:03d}-encoded")
            manifest = self.work / "v2" / f"r7-experts-layer-{layer:03d}.json"
            oracle = self.work / f"layer-{layer:03d}" / "oracle-report.json"
            interaction = self.work / f"layer-{layer:03d}" / "interaction-audit.json"
            install = self.work / f"layer-{layer:03d}" / "install-audit.json"
            layer_records[str(layer)] = {
                "manifest": str(manifest.relative_to(self.work)),
                "manifest_sha256": sha256_file(manifest),
                "oracle": str(oracle.relative_to(self.work)),
                "oracle_sha256": sha256_file(oracle),
                "interaction_audit": str(interaction.relative_to(self.work)),
                "interaction_audit_sha256": sha256_file(interaction),
                "install_audit": str(install.relative_to(self.work)),
                "install_audit_sha256": sha256_file(install),
                "encoded_stage_sha256": encoded.digest,
            }
        payload = {
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "schema": "r7-walk-complete-v1",
            "layers": layer_records,
            "carrier_inventory_sha256": self.carrier_inventory["inventory_sha256"],
            "source_inventory_sha256": self.source_inventory["inventory_sha256"],
            "numeric_environment_sha256": self.numeric_inventory["inventory_sha256"],
            "runtime_inventory_sha256": self.runtime_inventory["inventory_sha256"],
            "backend_fingerprint": self.backend.fingerprint,
            "boundary": "layer-077-encoded; layer-078-carried-and-not-forwarded",
        }
        if manifest_path.exists():
            if read_json(manifest_path) != payload:
                raise ValueError("walk completion manifest drift")
        else:
            atomic_write_json(manifest_path, payload)
        digest = sha256_file(manifest_path)
        self.journal.seal(
            StageSeal(
                name=stage,
                input_sha256={
                    record["manifest"]: record["manifest_sha256"]
                    for record in layer_records.values()
                },
                output_sha256={str(manifest_path.relative_to(self.work)): digest},
                metadata={"layers": len(layer_records), "last_encoded_layer": 77},
            )
        )
