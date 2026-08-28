"""Spawned, rank-pinned intra-layer workers for the Round 7 encoder.

The coordinator owns every canonical artifact and every cross-expert decision.
Workers own exactly one visible GPU and execute a static rank slice serially, so
one codec is never entered concurrently and completion order cannot affect the
result order.  This module intentionally imports no Torch/model code before a
child has been pinned with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from .determinism import DETERMINISTIC_ENVIRONMENT


_CUDA_DEVICE = re.compile(r"^cuda:(\d+)$")
# The pre-process sequential encoder runs these CPU BF16 reductions with the
# host's sealed 128-thread Torch intra-op contract. In particular, down-input
# covariance changes at the bit level if this count changes. Workers must match
# the authoritative path exactly; process-level parallelism is the only new
# scheduling dimension.
_THREADS_PER_WORKER = "36"
_INTEROP_THREADS_PER_WORKER = 36
_CPU_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _stable_expert_partitions(
    experts: Iterable[int],
    worker_count: int,
    *,
    assignment_domain: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Assign experts to stable owners independently of a resumed task subset.

    Ordinary expert work is owned by ``expert % worker_count``. Shared-search
    samples instead provide the full immutable sample domain: its position owns
    the worker, so balanced full-sample slicing survives a partial resume.
    """

    if worker_count < 1:
        raise ValueError("process task requires at least one worker")
    domain = tuple(sorted(int(expert) for expert in experts))
    if len(set(domain)) != len(domain):
        raise ValueError("process task expert domain contains duplicates")
    if assignment_domain is None:
        owner = {expert: expert % worker_count for expert in domain}
    else:
        assignment = tuple(int(expert) for expert in assignment_domain)
        if len(set(assignment)) != len(assignment):
            raise ValueError("process assignment domain contains duplicates")
        positions = {expert: index for index, expert in enumerate(assignment)}
        if set(domain).difference(positions):
            raise ValueError(
                "process task contains experts outside its assignment domain"
            )
        owner = {expert: positions[expert] % worker_count for expert in domain}
    return tuple(
        tuple(expert for expert in domain if owner[expert] == rank)
        for rank in range(worker_count)
    )


class _FingerprintRuntime:
    """Row/source workers never execute the Transformers model runtime."""

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint


def _load_worker_numeric(path):
    """Verify the sealed numeric payload without rebinding it to one GPU UUID.

    The coordinator performs the full execution-platform check before it creates
    this pool.  Worker devices are an execution detail excluded from the recipe;
    their UUID and visible-device count necessarily differ after rank pinning.
    Each child still independently verifies the signed payload and exact numeric
    core/extension bytes it will load.
    """

    from .determinism import canonical_json_bytes, read_json, sha256_bytes, sha256_file

    payload = read_json(path)
    digest = payload.pop("inventory_sha256", None)
    if digest != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("numeric environment seal mismatch in process worker")
    if payload.get("schema") != "r7-numeric-environment-v2":
        raise ValueError("process worker requires platform-sealed numeric inventory")
    if sha256_file(payload["numeric_core"]) != payload["numeric_core_sha256"]:
        raise ValueError("numeric core changed before process worker startup")
    if sha256_file(payload["extension"]) != payload["extension_sha256"]:
        raise ValueError("TRELLIS extension changed before process worker startup")
    payload["inventory_sha256"] = digest
    return payload


def _physical_token(device: str, ambient: str | None) -> str:
    match = _CUDA_DEVICE.fullmatch(device)
    if match is None:
        raise ValueError(f"process worker device must be cuda:N, got {device!r}")
    index = int(match.group(1))
    if ambient:
        visible = [item.strip() for item in ambient.split(",") if item.strip()]
        if index >= len(visible):
            raise ValueError(f"{device} is outside ambient CUDA_VISIBLE_DEVICES")
        return visible[index]
    return str(index)


def _worker_environment(visible_token: str) -> dict[str, str | None]:
    # Reproduce the original coordinator exactly: its process environment is
    # sealed at 32 for non-Torch consumers, while Torch was imported before
    # that seal and has an observed 128-thread intra/inter-op contract. A
    # spawned child sees the sealed environment first, then _WorkerService
    # explicitly restores Torch's observed thread counts.
    updates = {"CUDA_VISIBLE_DEVICES": visible_token}
    updates.update(
        {name: DETERMINISTIC_ENVIRONMENT[name] for name in _CPU_ENVIRONMENT_NAMES}
    )
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    return previous


def _restore_environment(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class _WorkerService:
    """GPU-local numerical service, constructed only after rank pinning."""

    def __init__(self, config) -> None:
        import torch

        torch.set_num_threads(int(_THREADS_PER_WORKER))
        try:
            torch.set_num_interop_threads(_INTEROP_THREADS_PER_WORKER)
        except RuntimeError:
            if torch.get_num_interop_threads() != _INTEROP_THREADS_PER_WORKER:
                raise
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_deterministic_debug_mode("error")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("worker deterministic algorithms are not enabled")
        if torch.get_deterministic_debug_mode() != 2:
            raise RuntimeError("worker deterministic debug mode is not error")
        warn_only = getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", None
        )
        if warn_only is not None and warn_only():
            raise RuntimeError("worker deterministic algorithms are warn-only")
        if (
            torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"
        ):
            raise RuntimeError("worker floating-point arithmetic contract drift")
        if torch.get_num_threads() != int(_THREADS_PER_WORKER):
            raise RuntimeError("worker Torch intra-op thread contract drift")
        if torch.get_num_interop_threads() != _INTEROP_THREADS_PER_WORKER:
            raise RuntimeError("worker Torch inter-op thread contract drift")
        self.cpu_threads = torch.get_num_threads()
        self.interop_threads = torch.get_num_interop_threads()
        from .glm52_backend import GLM52Backend
        from .inventory import load_checkpoint_inventory, load_runtime_code_inventory
        from .layer import LayerProcessor
        from .trellis import CodecConfig, Exl3TrellisCodec

        self.config = config
        if config.backend_factory != "r7_encoder.glm52_backend:factory":
            raise ValueError(
                "process workers require the audited GLM52 row/source backend"
            )
        runtime = load_runtime_code_inventory(config.runtime_inventory)
        self.backend = GLM52Backend(
            config=config,
            runtime=_FingerprintRuntime(str(runtime["inventory_sha256"])),
        )
        numeric = _load_worker_numeric(config.numeric_inventory)
        source = load_checkpoint_inventory(
            config.source_inventory,
            role="bf16-source",
            require_routed_bf16=True,
        )
        codec = Exl3TrellisCodec(
            CodecConfig(
                device="cuda:0",
                sigma_reg=config.sigma_reg,
                numeric_core=Path(str(numeric["numeric_core"])),
                numeric_core_sha256=str(numeric["numeric_core_sha256"]),
                extension=Path(str(numeric["extension"])),
                extension_sha256=str(numeric["extension_sha256"]),
            )
        )
        self.processor = LayerProcessor(
            backend=self.backend,
            codec=codec,
            codecs=(codec,),
            work_dir=config.work,
            device="cuda:0",
            sigma_reg=config.sigma_reg,
            fixed_point_iterations=config.fixed_point_iterations,
            holdout_rows=config.holdout_rows,
            source_inventory_sha256=str(source["inventory_sha256"]),
            numeric_environment_sha256=str(numeric["inventory_sha256"]),
            runtime_inventory_sha256=str(runtime["inventory_sha256"]),
            process_pool=None,
        )
        self.codec = codec
        # V2 reconstruction audits revisit one immutable layer shard for a
        # static expert slice.  Parse its (large) safetensors header and load
        # the two shared vectors once per worker/layer, while every unique
        # expert payload is still read and checked by the expert operation.
        self._oracle_cache_key: tuple[str, str, str] | None = None
        self._oracle_manifest = None
        self._oracle_reader = None
        self._oracle_shared_vectors: dict[str, object] = {}
        # The 16 shared-search experts recur across 12 proxy barriers, four
        # full-score barriers, and the scale-family pass.  Keep their immutable
        # BF16 weights/covariance/holdout objects resident on the rank that owns
        # them instead of re-reading and re-uploading the same sealed bytes for
        # every barrier.  A capture digest change drops the whole layer-local
        # cache; per-expert search drops each entry after its final use.
        self._search_capture_digest: str | None = None
        self._search_prepared: dict[int, tuple[object, object, object]] = {}
        # Query CUDA only after the runtime's fail-closed pre-initialization
        # check has run. Each worker must see its one inherited physical token
        # as logical cuda:0, matching the v31 numeric core's hardcoded device.
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "each spawned encoder worker must see exactly one CUDA device"
            )
        self.device_name = torch.cuda.get_device_name(0)

    def _runner(self, processor, backend, payload):
        """Build an in-memory SearchRunner that never writes shared progress."""

        from .search import SearchRunner, mass_stratified_experts

        runner = SearchRunner.__new__(SearchRunner)
        runner.processor = processor
        runner.backend = backend
        runner.capture = payload["capture"]
        runner.shards = tuple(payload["shards"])
        runner.output = Path(payload["output"])
        runner.draws = int(payload["draws"])
        runner.sample = mass_stratified_experts(
            runner.capture.mass_audit.mass_by_expert,
            int(payload["shared_sample_experts"]),
        )
        runner.progress_path = runner.output.with_name(
            f".{runner.output.stem}.worker-progress-unused.json"
        )
        runner.progress = {"scores": {}, "down_diagonal": {}, "expert_results": {}}
        runner._prepared = self._search_prepared
        runner.process_pool = None
        runner._flush = lambda: None
        return runner

    def _bind_capture(self, payload: Mapping[str, object]) -> None:
        """Invalidate every layer-local cache at an operation capture boundary."""

        capture = payload.get("capture")
        if capture is None:
            return
        capture_digest = str(getattr(capture, "digest"))
        if not capture_digest:
            raise ValueError("process task capture has an empty digest")
        if self._search_capture_digest != capture_digest:
            self._search_prepared.clear()
            self.processor.cold_audit.clear()
            self._search_capture_digest = capture_digest

    def _cold_records(
        self, prior_keys: Iterable[str] = ()
    ) -> list[dict[str, object]]:
        excluded = set(prior_keys)
        return [
            dict(self.processor.cold_audit[key])
            for key in sorted(self.processor.cold_audit)
            if key not in excluded
        ]

    def _clear(self, capture, expert: int) -> None:
        clear = getattr(self.backend, "clear_expert_row_memo", None)
        if clear is not None:
            clear(capture, expert)

    def _oracle_context(self, payload: Mapping[str, object]):
        """Open a coordinator-validated V2 manifest under an exact binding."""

        from .constants import RECIPE_MARKER, RECIPE_VERSION
        from .determinism import read_json, sha256_file
        from .safetensors_io import SafeTensorReader, read_torch_tensor
        from .schema import shared_down_svh_name, shared_gate_up_suh_name

        manifest_path = Path(str(payload["manifest_path"])).resolve()
        manifest_sha256 = str(payload["manifest_sha256"])
        shard_sha256 = str(payload["shard_sha256"])
        key = (str(manifest_path), manifest_sha256, shard_sha256)
        if self._oracle_cache_key == key:
            return (
                self._oracle_manifest,
                self._oracle_reader,
                self._oracle_shared_vectors,
            )

        if sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("V2 manifest changed between coordinator and worker")
        manifest = read_json(manifest_path)
        layer = int(payload["layer"])
        if (
            manifest.get("marker") != RECIPE_MARKER
            or manifest.get("recipe_version") != RECIPE_VERSION
            or int(manifest.get("schema_version", -1)) != 2
            or int(manifest.get("layer", -1)) != layer
            or str(manifest.get("shard_sha256")) != shard_sha256
        ):
            raise ValueError("worker V2 manifest binding drift")
        shard = (manifest_path.parent / str(manifest["shard"])).resolve()
        if shard != Path(str(payload["shard_path"])).resolve():
            raise ValueError("worker V2 shard path differs from coordinator")
        reader = SafeTensorReader(shard)
        if reader.metadata != {
            "format": "pt",
            "r7_layer": str(layer),
            "r7_marker": RECIPE_MARKER,
            "r7_schema": "2",
        }:
            raise ValueError("worker V2 shard metadata mismatch")
        shared_names = (
            shared_gate_up_suh_name(layer),
            shared_down_svh_name(layer),
        )
        shared = {name: read_torch_tensor(reader, name) for name in shared_names}
        self._oracle_cache_key = key
        self._oracle_manifest = manifest
        self._oracle_reader = reader
        self._oracle_shared_vectors = shared
        return manifest, reader, shared

    def run(self, operation: str, expert: int, payload: Mapping[str, object]):
        self._bind_capture(payload)
        cold_before = frozenset(self.processor.cold_audit)
        if operation == "audit_v2":
            import torch

            from .constants import MCG_MULT, PROJECTIONS, TensorId
            from .determinism import sha256_bytes
            from .oracles import sliced_reconstruction_oracle
            from .safetensors_io import read_torch_tensor
            from .schema import tensor_name
            from .types import EncodedTensor

            manifest, reader, shared = self._oracle_context(payload)
            layer = int(payload["layer"])
            tp_sizes = tuple(int(value) for value in payload["tp_sizes"])
            projection_results: dict[str, dict[str, object]] = {}
            for projection in PROJECTIONS:
                tensor_id = TensorId(layer, expert, projection)
                prefix = tensor_id.hf_prefix
                packed_name = tensor_name(tensor_id, "trellis")
                refs = manifest["vector_refs"][prefix]
                bits = int(manifest["bit_map"][prefix])
                if bits not in (3, 4, 5):
                    raise ValueError(f"{tensor_id.key}: worker bit-width drift")
                marker_name = tensor_name(tensor_id, "mcg")
                marker = int(read_torch_tensor(reader, marker_name).item()) & 0xFFFFFFFF
                if marker != MCG_MULT:
                    raise ValueError(f"{tensor_id.key}: worker MCG marker drift")
                packed = read_torch_tensor(reader, packed_name)
                suh = shared.get(refs["suh"])
                if suh is None:
                    suh = read_torch_tensor(reader, refs["suh"])
                svh = shared.get(refs["svh"])
                if svh is None:
                    svh = read_torch_tensor(reader, refs["svh"])
                packed_hash = sha256_bytes(
                    packed.detach()
                    .contiguous()
                    .cpu()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                expected_packed = manifest["payload_sha256"][packed_name]
                if (
                    packed_hash != expected_packed
                    or manifest["roundtrip_hashes"][prefix]["packed_sha256"]
                    != expected_packed
                ):
                    raise ValueError(f"{tensor_id.key}: worker packed hash drift")
                reconstructed = self.codec.decode_to_original(
                    packed.to(self.codec.config.device), suh, svh, bits
                ).half()
                reconstructed_hash = sha256_bytes(
                    reconstructed.detach()
                    .contiguous()
                    .cpu()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                if (
                    reconstructed_hash
                    != manifest["roundtrip_hashes"][prefix]["reconstruction_sha256"]
                ):
                    raise ValueError(
                        f"{tensor_id.key}: worker reconstruction hash drift"
                    )
                encoded = EncodedTensor(
                    tensor_id=tensor_id,
                    bits=bits,
                    trellis=packed,
                    suh=suh,
                    svh=svh,
                    reconstructed_kn=None,
                    proxy_loss=0.0,
                    packed_sha256=packed_hash,
                    reconstruction_sha256=reconstructed_hash,
                    provenance=manifest["tensor_provenance"][prefix],
                )
                for tp_size in tp_sizes:
                    sliced_reconstruction_oracle(encoded, self.codec, tp_size)
                projection_results[projection] = {
                    "packed_sha256": packed_hash,
                    "reconstruction_sha256": reconstructed_hash,
                    "tp_sizes": list(tp_sizes),
                    "passed": True,
                }
                del reconstructed, encoded, packed
            return {
                "layer": layer,
                "expert": expert,
                "projections": projection_results,
                "passed": True,
            }

        if operation == "proxy_search":
            runner = self._runner(self.processor, self.backend, payload)
            value = runner._proxy_expert(payload["search"], expert)
            # Keep the selected-row memo warm for the later shared full scores
            # and scale-family preparation, along with the immutable prepared
            # tensors themselves. Static rank slicing sends this expert back to
            # the same process at every shared-search barrier.
            return {
                "value": float(value),
                "cold_audit": self._cold_records(cold_before),
            }

        if operation == "shared_scale_search":
            runner = self._runner(self.processor, self.backend, payload)
            value = runner._shared_scale_expert(expert)
            # As above, the shared sample is deliberately retained in the
            # prepared cache and backend row memo until per-expert search.
            return {"value": value, "cold_audit": self._cold_records(cold_before)}

        if operation == "score_search":
            runner = self._runner(self.processor, self.backend, payload)
            value = runner._score_expert(
                payload["search"],
                expert,
                return_diag=bool(payload.get("return_diag", False)),
            )
            # Retain the small 16-expert prepared/selected-row domain across the
            # four shared-score barriers. Static rank slicing returns it to this
            # same worker, while a new layer invalidates both caches.
            return {"value": value, "cold_audit": self._cold_records(cold_before)}

        if operation == "choose_search":
            runner = self._runner(self.processor, self.backend, payload)
            parent = payload["progress"]
            prefix = f"expert/{expert:03d}/"
            parent_scores = parent.get("scores", {})
            scores = {
                str(key): dict(value)
                for key, value in parent_scores.items()
                if str(key).startswith(prefix)
            }
            diagonal_map = parent.get("down_diagonal", {})
            diagonals = {}
            if str(expert) in diagonal_map:
                diagonals[str(expert)] = list(diagonal_map[str(expert)])
            runner.progress = {
                "scores": scores,
                "down_diagonal": diagonals,
                "expert_results": {},
            }
            base = payload["base"]
            chosen = runner._choose_expert(base, expert, base.experts)
            result = {
                "chosen": chosen,
                "scores": dict(runner.progress["scores"]),
                "diagonal": list(runner.progress["down_diagonal"][str(expert)]),
                "cold_audit": self._cold_records(cold_before),
            }
            runner._prepared.pop(expert, None)
            self._clear(payload["capture"], expert)
            return result

        if operation == "probe":
            from .layer import LayerProcessor

            collector = LayerProcessor._CollectorLedger()
            self.processor._probe_expert(
                capture=payload["capture"],
                shards=tuple(payload["shards"]),
                search=payload["search"],
                expert=expert,
                context_bits=payload["context_bits"],
                ledger=collector,
                fixed_point_iteration=int(payload["fixed_point_iteration"]),
                search_artifact_sha256=str(payload["search_artifact_sha256"]),
            )
            result = {
                "records": list(collector.records),
                "cold_audit": self._cold_records(cold_before),
            }
            self._clear(payload["capture"], expert)
            return result

        if operation == "final":
            from .expert_cache import write_cached_expert

            gate, up, down, gu_hash, loss, holdout_hash, permutation = (
                self.processor._final_expert(
                    capture=payload["capture"],
                    shards=tuple(payload["shards"]),
                    search=payload["search"],
                    allocation=payload["allocation"],
                    expert=expert,
                    search_artifact_sha256=str(payload["search_artifact_sha256"]),
                )
            )
            cold_records = self._cold_records(cold_before)
            write_cached_expert(
                payload["cache_root"],
                encoded=(gate, up, down),
                bindings=payload["bindings"],
                gate_up_sha256=gu_hash,
                final_loss=loss,
                holdout_row_ids_sha256=holdout_hash,
                permutation_audit=permutation,
                cold_audit=cold_records,
            )
            result = {"cold_audit": cold_records}
            del gate, up, down
            self._clear(payload["capture"], expert)
            return result

        if operation == "floor":
            from .constants import TensorId

            allocation = payload["allocation"]
            bits = dict(allocation.bits)
            layer = payload["capture"].layer
            for projection in ("gate_proj", "up_proj", "down_proj"):
                bits[TensorId(layer, expert, projection).key] = 3
            floor = replace(allocation, bits=bits)
            *_, loss, _, _ = self.processor._final_expert(
                capture=payload["capture"],
                shards=tuple(payload["shards"]),
                search=payload["search"],
                allocation=floor,
                expert=expert,
                search_artifact_sha256=str(payload["search_artifact_sha256"]),
            )
            result = {
                "loss": float(loss),
                "cold_audit": self._cold_records(cold_before),
            }
            self._clear(payload["capture"], expert)
            return result

        raise ValueError(f"unknown process-pool operation {operation!r}")


def _worker_main(
    rank, physical_device, visible_token, config, inbound, outbound
) -> None:
    # The parent starts this spawned interpreter with the same values already in
    # its inherited environment. Repeat them before importing any task modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_token
    for name in _CPU_ENVIRONMENT_NAMES:
        os.environ[name] = DETERMINISTIC_ENVIRONMENT[name]
    try:
        # This must precede the first Torch import in the spawned interpreter.
        from .determinism import configure_deterministic_environment

        configure_deterministic_environment()
        service = _WorkerService(config)
        outbound.put(
            {
                "type": "ready",
                "rank": rank,
                "pid": os.getpid(),
                "physical_device": physical_device,
                "visible_token": visible_token,
                "logical_device": "cuda:0",
                "device_name": service.device_name,
                "cpu_threads": service.cpu_threads,
                "interop_threads": service.interop_threads,
            }
        )
        while True:
            message = inbound.get()
            if message is None:
                return
            job = int(message["job"])
            operation = str(message["operation"])
            experts = tuple(int(item) for item in message["experts"])
            payload = message["payload"]
            try:
                values = [
                    (expert, service.run(operation, expert, payload))
                    for expert in experts
                ]
            except BaseException as exc:
                outbound.put(
                    {
                        "type": "error",
                        "rank": rank,
                        "job": job,
                        "operation": operation,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                return
            outbound.put(
                {
                    "type": "result",
                    "rank": rank,
                    "job": job,
                    "operation": operation,
                    "values": values,
                }
            )
    except BaseException as exc:
        outbound.put(
            {
                "type": "fatal",
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


class PinnedDeviceProcessPool:
    """One spawned interpreter and one serial queue per requested CUDA device."""

    def __init__(self, config, devices: Iterable[str]) -> None:
        import torch

        if torch.get_num_threads() != int(_THREADS_PER_WORKER):
            raise RuntimeError(
                "coordinator Torch intra-op threads differ from the exact "
                f"worker contract: {torch.get_num_threads()} != "
                f"{_THREADS_PER_WORKER}"
            )
        if torch.get_num_interop_threads() != _INTEROP_THREADS_PER_WORKER:
            raise RuntimeError(
                "coordinator Torch inter-op threads differ from the exact "
                f"worker contract: {torch.get_num_interop_threads()} != "
                f"{_INTEROP_THREADS_PER_WORKER}"
            )
        self.devices = tuple(str(item) for item in devices)
        if len(self.devices) < 2 or len(set(self.devices)) != len(self.devices):
            raise ValueError("process pool requires at least two unique devices")
        self._context = multiprocessing.get_context("spawn")
        self._outbound = self._context.Queue()
        self._inbound = []
        self._processes = []
        self._closed = False
        self._job = 0
        ambient = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for rank, device in enumerate(self.devices):
                visible_token = _physical_token(device, ambient)
                inbound = self._context.Queue()
                child_config = replace(config, device="cuda:0", devices=())
                previous = _worker_environment(visible_token)
                try:
                    process = self._context.Process(
                        target=_worker_main,
                        args=(
                            rank,
                            device,
                            visible_token,
                            child_config,
                            inbound,
                            self._outbound,
                        ),
                        name=f"r8-gpu-worker-{rank}",
                        daemon=False,
                    )
                    process.start()
                finally:
                    _restore_environment(previous)
                self._inbound.append(inbound)
                self._processes.append(process)
            ready: dict[int, dict[str, object]] = {}
            deadline = time.monotonic() + 180.0
            while len(ready) != len(self._processes):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for GPU worker initialization"
                    )
                try:
                    message = self._outbound.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    dead = [
                        (rank, process.exitcode)
                        for rank, process in enumerate(self._processes)
                        if not process.is_alive()
                    ]
                    if dead:
                        raise RuntimeError(f"GPU workers exited before ready: {dead}")
                    continue
                if message.get("type") == "fatal":
                    raise RuntimeError(
                        "GPU worker initialization failed:\n" + message["traceback"]
                    )
                if message.get("type") != "ready":
                    raise RuntimeError("unexpected GPU worker initialization response")
                rank = int(message["rank"])
                if rank in ready or not 0 <= rank < len(self._processes):
                    raise RuntimeError("duplicate or foreign GPU worker rank")
                ready[rank] = dict(message)
            self.worker_info = [ready[rank] for rank in sorted(ready)]
        except BaseException:
            self.close(force=True)
            raise

    def map(
        self,
        operation: str,
        experts: Iterable[int],
        payload: Mapping[str, object],
    ) -> list[tuple[int, object]]:
        if self._closed:
            raise RuntimeError("process pool is closed")
        domain = tuple(sorted(int(expert) for expert in experts))
        if not domain:
            return []
        self._job += 1
        job = self._job
        expected: dict[int, tuple[int, ...]] = {}
        assignment_domain = payload.get("assignment_domain")
        partitions = _stable_expert_partitions(
            domain,
            len(self._inbound),
            assignment_domain=(
                None
                if assignment_domain is None
                else tuple(int(expert) for expert in assignment_domain)
            ),
        )
        for rank, inbound in enumerate(self._inbound):
            partition = partitions[rank]
            if not partition:
                continue
            expected[rank] = partition
            inbound.put(
                {
                    "job": job,
                    "operation": str(operation),
                    "experts": partition,
                    "payload": dict(payload),
                }
            )
        received: dict[int, object] = {}
        waiting = set(expected)
        while waiting:
            try:
                message = self._outbound.get(timeout=1.0)
            except queue.Empty:
                dead = [
                    (rank, process.exitcode)
                    for rank, process in enumerate(self._processes)
                    if rank in waiting and not process.is_alive()
                ]
                if dead:
                    self.close(force=True)
                    raise RuntimeError(f"GPU workers exited during {operation}: {dead}")
                continue
            kind = message.get("type")
            if kind in {"fatal", "error"}:
                self.close(force=True)
                raise RuntimeError(
                    f"GPU worker failed during {operation}:\n{message.get('traceback')}"
                )
            if kind != "result" or int(message.get("job", -1)) != job:
                self.close(force=True)
                raise RuntimeError("foreign or out-of-order GPU worker result")
            rank = int(message["rank"])
            if rank not in waiting or message.get("operation") != operation:
                self.close(force=True)
                raise RuntimeError("duplicate or mismatched GPU worker result")
            values = message.get("values")
            if not isinstance(values, list):
                self.close(force=True)
                raise RuntimeError("malformed GPU worker result payload")
            returned = tuple(int(item[0]) for item in values)
            if returned != expected[rank]:
                self.close(force=True)
                raise RuntimeError("GPU worker result domain/order drift")
            for expert, value in values:
                if int(expert) in received:
                    self.close(force=True)
                    raise RuntimeError("duplicate expert returned by GPU workers")
                received[int(expert)] = value
            waiting.remove(rank)
        if set(received) != set(domain):
            self.close(force=True)
            raise RuntimeError("GPU worker result domain is incomplete")
        return [(expert, received[expert]) for expert in domain]

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if not force:
            for inbound, process in zip(self._inbound, self._processes):
                if process.is_alive():
                    inbound.put(None)
        deadline = time.monotonic() + (15.0 if not force else 2.0)
        for process in self._processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2.0)
        for inbound in self._inbound:
            inbound.close()
        self._outbound.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(force=exc_type is not None)
        return False
