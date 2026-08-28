"""Encode ONE layer end to end, independently of every other layer.

This is the unit of work the layer-parallel driver hands to a pinned worker.
It reuses R7's `SearchRunner` and `LayerProcessor` verbatim, so every quality
decision survives: joint full-K down encode, Gap 1 round-trip inputs, exact
3.5-bpw mass allocation, per-expert permutation, layer-shared residual
rotations, per-128 block scales, and schema-v2 emission.

What it does NOT do is the sequential state chain. The capture for this layer
comes from the one-shot flat capture rather than from quantized predecessors,
which is what makes layers independent and lets six GPUs work at once.
"""

from __future__ import annotations

from pathlib import Path

from .determinism import sha256_file


def _build_backend(
    *,
    carrier: Path,
    bf16_source: Path,
    capture_dir: Path,
    carrier_inventory: Path,
    source_inventory: Path,
    numeric_inventory: Path,
    runtime_inventory: Path,
    device: str,
    sigma_reg: float,
):
    """Construct the GLM backend bound to a flat capture instead of states."""

    from types import SimpleNamespace

    from .glm52_backend import factory as backend_factory

    config = SimpleNamespace(
        carrier=carrier,
        bf16_source=bf16_source,
        capture_dir=capture_dir,
        carrier_inventory=carrier_inventory,
        source_inventory=source_inventory,
        numeric_inventory=numeric_inventory,
        runtime_inventory=runtime_inventory,
        device=device,
        devices=(device,),
        sigma_reg=sigma_reg,
        runtime_factory="r7_encoder.flat_runtime:factory",
    )
    return backend_factory(config)


def encode_one_layer(
    *,
    layer: int,
    work: Path,
    capture_dir: Path,
    carrier: Path,
    bf16_source: Path,
    carrier_inventory: Path,
    source_inventory: Path,
    numeric_inventory: Path,
    runtime_inventory: Path,
    fixed_point_iterations: int = 1,
    draws: int = 12,
    shared_sample_experts: int = 16,
    holdout_rows: int = 4096,
    sigma_reg: float = 0.025,
    device: str = "cuda:0",
    log=print,
) -> dict:
    """Probe -> allocate -> final encode -> emit schema v2, for one layer."""

    from .inventory import load_checkpoint_inventory, load_numeric_environment
    from .layer import LayerProcessor
    from .oracles import audit_v2_layer, shared_vector_oracle
    from .schema import emit_layer_v2
    from .search import SearchRunner
    from .trellis import CodecConfig, Exl3TrellisCodec

    numeric = load_numeric_environment(numeric_inventory)
    src_inv = load_checkpoint_inventory(
        source_inventory, role="bf16-source", require_routed_bf16=True
    )
    car_inv = load_checkpoint_inventory(carrier_inventory, role="carrier")
    from .inventory import load_runtime_code_inventory

    run_inv = load_runtime_code_inventory(runtime_inventory)

    codec = Exl3TrellisCodec(
        CodecConfig(
            device=device,
            sigma_reg=sigma_reg,
            numeric_core=Path(str(numeric["numeric_core"])),
            numeric_core_sha256=str(numeric["numeric_core_sha256"]),
            extension=Path(str(numeric["extension"])),
            extension_sha256=str(numeric["extension_sha256"]),
        )
    )

    backend = _build_backend(
        carrier=carrier,
        bf16_source=bf16_source,
        capture_dir=capture_dir,
        carrier_inventory=carrier_inventory,
        source_inventory=source_inventory,
        numeric_inventory=numeric_inventory,
        runtime_inventory=runtime_inventory,
        device=device,
        sigma_reg=sigma_reg,
    )

    # The flat capture IS the calibration source. `open_flat_capture` builds the
    # same LayerCapture record the walk produced, so everything downstream is
    # byte-for-byte the R7 path.
    capture = backend.open_flat_capture(layer=layer, capture_dir=capture_dir)
    shards = ()

    processor = LayerProcessor(
        backend=backend,
        codec=codec,
        codecs=(codec,),
        work_dir=work,
        device=device,
        sigma_reg=sigma_reg,
        fixed_point_iterations=fixed_point_iterations,
        holdout_rows=holdout_rows,
        source_inventory_sha256=str(src_inv["inventory_sha256"]),
        numeric_environment_sha256=str(numeric["inventory_sha256"]),
        runtime_inventory_sha256=str(run_inv["inventory_sha256"]),
    )
    # No successor forward in layer-parallel mode: skip staging encoded experts
    # back into a live model. Encoding, allocation, and emission are unaffected.
    processor.install_for_successor = False

    search_dir = work / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    search_path = search_dir / f"layer-{layer:03d}.json"
    log(f"layer {layer}: search ({draws} draws, {shared_sample_experts} sample experts)")
    search = SearchRunner(
        processor=processor,
        backend=backend,
        capture=capture,
        shards=shards,
        output=search_path,
        draws=draws,
        shared_sample_experts=shared_sample_experts,
    ).run()

    log(f"layer {layer}: probe + allocate + final encode")
    result = processor.run(
        capture=capture,
        shards=shards,
        search=search,
        search_artifact_sha256=sha256_file(search_path),
    )

    layer_output = work / "v2"
    layer_output.mkdir(parents=True, exist_ok=True)
    manifest_path = emit_layer_v2(
        layer=layer,
        result=result,
        search=search,
        output_dir=layer_output,
        codec=codec,
    )
    shared_vector_oracle(manifest_path)
    audit_v2_layer(manifest_path, codec=codec, tp_sizes=(16,))
    log(f"layer {layer}: v2 emitted + audited -> {manifest_path.name}")
    return {
        "layer": layer,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
