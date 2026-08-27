# SPDX-License-Identifier: Apache-2.0
# Derived in part from Local Inference Lab B12X btx_synth.py at UPSTREAM_COMMIT.
"""Writer/auditor for upstream B12X ``btx-atoms-v1``.

The implementation is intentionally pinned to upstream commit
``36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8``.  It ports the atom assembly
from ``btx_synth.py`` and validates the same strict manifest/tensor contract as
``btx_schema.py`` and ``btx.py``.  The content-addressed installed-layer store
remains the richer causal source of truth; this module is only the runtime
container transform.
"""

from __future__ import annotations

import json
import os
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core.artifacts import canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json
from .btx_qwen import verify_installed_layer
from .exact_payload import ExactCodecPayloadStore


UPSTREAM_COMMIT = "36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8"
UPSTREAM_CLOSURE_SHA256 = {
    "docs/btx-checkpoint-format.md": "62ed1996ba54d4f2ab63ccb14ba9dc7e22e15d4443bec330226696600368aebd",
    "b12x/moe/_shared/btx_schema.py": "282190602b38c70a2085b40a9a2ef895ba6925c38725671cc99b0204d8d1bbdd",
    "b12x/moe/_shared/kernels/w4a16/btx.py": "d6b241b59b29265235914e1be955607b3e9b0cd83a40f72a40350455adf927bd",
    "b12x/moe/_shared/kernels/w4a16/btx_synth.py": "f94b5e50c02551a041660d194266849f656e8a08da179d1e36fa64119d04f54e",
}
SCHEMA = "btx-atoms-v1"
MANIFEST = "btx-manifest.json"
ACCOUNTING = "btx-accounting.json"
MCG_MULTIPLIER = 0xCBAC1FED
ATOM_CHANNELS = 32
ATOMS_PER_PAIR = 8
PAIR_CODES = {"P22": 0x22, "P33": 0x33, "P24": 0x24, "P43": 0x43, "P44": 0x44}
CODE_KINDS = {value: key for key, value in PAIR_CODES.items()}


def _atomic_save(tensors: Mapping[str, Any], path: Path, metadata: Mapping[str, str]) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(dict(tensors), temporary, metadata=dict(metadata))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _bits(code: int) -> tuple[int, int]:
    return ((int(code) >> 4) & 0xF, int(code) & 0xF)


def _matrix_atom_bytes(hidden: int, low: int, high: int) -> int:
    return (hidden // 16) * 32 * (low + high)


def _layer_choices(root: Path) -> tuple[dict[str, Any], dict[tuple[int, str], tuple[dict[str, Any], ExactCodecPayloadStore]]]:
    manifest = verify_installed_layer(root)
    store = ExactCodecPayloadStore(root / "payload-store")
    choices = {(int(row["expert"]), row["projection"]): (row, store) for row in manifest["choices"]}
    return manifest, choices


def btx_compatibility_report(
    installed_layers: Sequence[str | Path],
    *,
    require_fused: bool = True,
    target_tp_degrees: Sequence[int] = (1,),
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    layers = []
    geometry: tuple[int, int, int] | None = None
    pair_kinds: set[str] = set()
    structures: set[str] = set()
    observed_bits: set[int] = set()
    observed_rate_pairs: set[tuple[int, int]] = set()
    for raw in installed_layers:
        root = Path(raw)
        manifest, choices = _layer_choices(root)
        experts = sorted({key[0] for key in choices})
        expected = {(expert, projection) for expert in experts for projection in ("gate_proj", "up_proj", "down_proj")}
        if set(choices) != expected or experts != list(range(len(experts))):
            failures.append(f"layer {manifest['layer']}: choices are not a dense expert x 3 projection inventory")
            continue
        layer_rates = []
        layer_geometry = None
        for expert in experts:
            gate, gate_store = choices[(expert, "gate_proj")]
            up, _ = choices[(expert, "up_proj")]
            down, _ = choices[(expert, "down_proj")]
            gate_shape = gate_store.load_tensor(gate["objects"]["reconstruction"]).shape
            up_shape = gate_store.load_tensor(up["objects"]["reconstruction"]).shape
            down_shape = gate_store.load_tensor(down["objects"]["reconstruction"]).shape
            hidden, intermediate = int(gate_shape[1]), int(gate_shape[0])
            if tuple(up_shape) != (intermediate, hidden) or tuple(down_shape) != (hidden, intermediate):
                failures.append(f"layer {manifest['layer']} expert {expert}: Qwen projection geometry mismatch")
            if int(gate["bits"]) != int(up["bits"]):
                failures.append(
                    f"layer {manifest['layer']} expert {expert}: BTX fc1 shares one pair rate for gate/up; "
                    f"independent K{gate['bits']}/K{up['bits']} is unexpressible on master"
                )
            layer_geometry = (len(experts), hidden, intermediate)
            layer_rates.append((int(gate["bits"]), int(down["bits"])))
            observed_bits.update((int(gate["bits"]), int(up["bits"]), int(down["bits"])))
            observed_rate_pairs.add((int(gate["bits"]), int(down["bits"])))
        if layer_geometry and (layer_geometry[1] % 16 or layer_geometry[2] % 256):
            failures.append(f"layer {manifest['layer']}: hidden must divide 16 and intermediate must divide 256")
        if geometry is None:
            geometry = layer_geometry
        elif geometry != layer_geometry:
            failures.append("installed layers disagree on Qwen geometry")
        all_bits = {bit for pair in layer_rates for bit in pair}
        structures.add("uniform" if len(all_bits) == 1 else "per_expert_pair")
        layers.append(int(manifest["layer"]))
    if len(structures) > 1:
        failures.append("BTX declares one model-level rate structure; layers cannot mix uniform and per_expert_pair")
    structure = next(iter(structures), None)
    uniform_bits = next(iter(observed_bits)) if structure == "uniform" and len(observed_bits) == 1 else None
    if structure == "per_expert_pair" and any(bit not in {3, 4} for bit in observed_bits):
        failures.append("master per_expert_pair has no K5/K6 pair kind")
    if structure == "per_expert_pair":
        for fc1, fc2 in observed_rate_pairs:
            if fc1 in {3, 4}:
                pair_kinds.add(f"P{fc1}{fc1}")
            if fc2 in {3, 4}:
                pair_kinds.add(f"P{fc2}{fc2}")
    if structure == "uniform" and uniform_bits not in {3, 4, 5, 6}:
        failures.append(f"master uniform rate does not support K{uniform_bits}")
    if "P44" in pair_kinds:
        warnings.append("P44 is schema-declared but unfused on upstream master")
        if require_fused:
            failures.append("P44 requires the unfused/two-launch path; production fused emission was requested")
    if pair_kinds not in ({"P33"}, {"P33", "P44"}, {"P44"}, set()):
        failures.append(f"allocation requires unsupported master pair kinds: {sorted(pair_kinds)}")
    tp_compatibility: dict[str, Any] = {}
    if geometry is not None:
        slots = geometry[2] // ATOM_CHANNELS
        extent_alignment = ATOMS_PER_PAIR if structure == "per_expert_pair" else 1
        for raw_tp in target_tp_degrees:
            tp = int(raw_tp)
            legal = tp > 0 and slots % tp == 0 and (slots // tp) % extent_alignment == 0
            reason = None if legal else (
                f"{slots} slots / TP{tp} does not produce {extent_alignment}-slot-aligned contiguous rank extents"
            )
            tp_compatibility[str(tp)] = {"legal": legal, "slots_per_rank": slots // tp if tp > 0 and slots % tp == 0 else None, "reason": reason}
            if not legal:
                failures.append(f"TP{tp}: {reason}")
    body = {
        "schema": "quant-pipeline.btx-master-compatibility.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_closure_sha256": UPSTREAM_CLOSURE_SHA256,
        "compatible": not failures,
        "require_fused": require_fused,
        "rate_structure": structure,
        "uniform_bits": uniform_bits,
        "pair_kinds": sorted(pair_kinds),
        "geometry": None if geometry is None else {"num_experts": geometry[0], "hidden_size": geometry[1], "intermediate_size": geometry[2]},
        "layers": sorted(layers),
        "failures": failures,
        "warnings": warnings,
        "target_tp_degrees": [int(tp) for tp in target_tp_degrees],
        "tp_compatibility": tp_compatibility,
    }
    body["report_sha256"] = sha256_bytes(canonical_json(body))
    return body


def _trellis_planes(
    choice: Mapping[str, Any],
    store: ExactCodecPayloadStore,
    *,
    projection: str,
    slot: int,
    hidden: int,
    intermediate: int,
):
    trellis = store.load_tensor(choice["objects"]["trellis"])
    bits = int(choice["bits"])
    expected = (
        (hidden // 16, intermediate // 16, 16 * bits)
        if projection in {"gate_proj", "up_proj"}
        else (intermediate // 16, hidden // 16, 16 * bits)
    )
    if tuple(trellis.shape) != expected:
        raise ValueError(f"{projection} trellis shape {tuple(trellis.shape)} != {expected}")
    if projection in {"gate_proj", "up_proj"}:
        low, high = trellis[:, 2 * slot, :], trellis[:, 2 * slot + 1, :]
    else:
        low, high = trellis[2 * slot, :, :], trellis[2 * slot + 1, :, :]
    return low.contiguous(), high.contiguous()


def _assemble_layer(
    root: Path,
    choices: Mapping[tuple[int, str], tuple[Mapping[str, Any], ExactCodecPayloadStore]],
    *,
    rate_structure: str,
    row_alignment: int,
) -> tuple[dict[str, Any], set[str]]:
    import torch

    experts = sorted({key[0] for key in choices})
    first, store = choices[(experts[0], "gate_proj")]
    reconstruction = store.load_tensor(first["objects"]["reconstruction"])
    intermediate, hidden = map(int, reconstruction.shape)
    slots, pairs = intermediate // ATOM_CHANNELS, intermediate // 256
    rates_fc1 = torch.empty((pairs, len(experts)), dtype=torch.uint8)
    rates_fc2 = torch.empty_like(rates_fc1)
    kinds: set[str] = set()
    for expert in experts:
        gate = choices[(expert, "gate_proj")][0]
        down = choices[(expert, "down_proj")][0]
        fc1 = (int(gate["bits"]) << 4) | int(gate["bits"])
        fc2 = (int(down["bits"]) << 4) | int(down["bits"])
        rates_fc1[:, expert] = fc1
        rates_fc2[:, expert] = fc2
        if rate_structure == "per_expert_pair":
            kinds |= {CODE_KINDS[fc1], CODE_KINDS[fc2]}
    row_sizes = []
    for slot in range(slots):
        pair = slot // ATOMS_PER_PAIR
        total = 0
        for expert in experts:
            f1 = _bits(int(rates_fc1[pair, expert]))
            f2 = _bits(int(rates_fc2[pair, expert]))
            total += 2 * _matrix_atom_bytes(hidden, *f1) + _matrix_atom_bytes(hidden, *f2)
        row_sizes.append(total)
    stride = (max(row_sizes) + row_alignment - 1) // row_alignment * row_alignment
    atoms = torch.zeros((slots, stride), dtype=torch.uint8)
    rotations = torch.empty((slots, len(experts), 3, ATOM_CHANNELS), dtype=torch.float16)
    gate_hidden, up_hidden, down_hidden = [], [], []
    for expert in experts:
        gate, gate_store = choices[(expert, "gate_proj")]
        up, up_store = choices[(expert, "up_proj")]
        down, down_store = choices[(expert, "down_proj")]
        gate_hidden.append(gate_store.load_tensor(gate["objects"]["suh"]).half())
        up_hidden.append(up_store.load_tensor(up["objects"]["suh"]).half())
        down_hidden.append(down_store.load_tensor(down["objects"]["svh"]).half())
        rotations[:, expert, 0] = gate_store.load_tensor(gate["objects"]["svh"]).half().reshape(slots, ATOM_CHANNELS)
        rotations[:, expert, 1] = up_store.load_tensor(up["objects"]["svh"]).half().reshape(slots, ATOM_CHANNELS)
        rotations[:, expert, 2] = down_store.load_tensor(down["objects"]["suh"]).half().reshape(slots, ATOM_CHANNELS)
    hidden_tables = [torch.stack(rows) for rows in (gate_hidden, up_hidden, down_hidden)]
    per_expert = any(not torch.equal(table, table[0:1].expand_as(table)) for table in hidden_tables)
    if not per_expert:
        hidden_tables = [table[0] for table in hidden_tables]
    for slot in range(slots):
        cursor = 0
        for expert in experts:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                choice, choice_store = choices[(expert, projection)]
                for plane in _trellis_planes(choice, choice_store, projection=projection, slot=slot, hidden=hidden, intermediate=intermediate):
                    raw = plane.view(torch.uint8).reshape(-1)
                    atoms[slot, cursor : cursor + raw.numel()] = raw
                    cursor += raw.numel()
        if cursor != row_sizes[slot] or bool(torch.any(atoms[slot, cursor:] != 0)):
            raise AssertionError("BTX atom row packing/padding invariant failed")
    tensors = {
        "atoms": atoms,
        "rotations": rotations,
        "gate_suh": hidden_tables[0],
        "up_suh": hidden_tables[1],
        "down_svh": hidden_tables[2],
    }
    if rate_structure == "per_expert_pair":
        tensors["rates_fc1"] = rates_fc1
        tensors["rates_fc2"] = rates_fc2
    return tensors, kinds


def emit_official_btx_checkpoint(
    *,
    installed_layers: Sequence[str | Path],
    output_dir: str | Path,
    require_fused: bool = True,
    row_alignment: int = 4096,
    target_tp_degrees: Sequence[int] = (1,),
    expected_allocated_payload_bytes: int | None = None,
) -> dict[str, Any]:
    report = btx_compatibility_report(
        installed_layers,
        require_fused=require_fused,
        target_tp_degrees=target_tp_degrees,
    )
    if not report["compatible"]:
        raise ValueError("allocation is not representable by upstream BTX master: " + "; ".join(report["failures"]))
    verified_installed = [verify_installed_layer(path) for path in installed_layers]
    allocated_payload_bytes = sum(int(row["allocated_payload_bytes"]) for row in verified_installed)
    if expected_allocated_payload_bytes is not None and allocated_payload_bytes != expected_allocated_payload_bytes:
        raise ValueError("allocator cost differs from official BTX source payload cost")
    destination = prepare_empty_destination(output_dir)
    geometry = report["geometry"]
    assert geometry is not None
    layers: dict[str, Any] = {}
    observed_kinds: set[str] = set()
    per_expert_hidden = None
    emitted_tensor_payload_bytes = 0
    emitted_container_bytes = 0
    for raw in installed_layers:
        installed_root = Path(raw)
        installed, choices = _layer_choices(installed_root)
        layer = int(installed["layer"])
        tensors, kinds = _assemble_layer(installed_root, choices, rate_structure=report["rate_structure"], row_alignment=row_alignment)
        observed_kinds |= kinds
        this_per_expert = tensors["gate_suh"].ndim == 2
        per_expert_hidden = this_per_expert if per_expert_hidden is None else per_expert_hidden
        if per_expert_hidden != this_per_expert:
            raise ValueError("BTX model-level hidden rotation topology differs across layers")
        filename = f"btx-layer-{layer:05d}.safetensors"
        metadata = {
            "schema": SCHEMA,
            "codebook": "mcg",
            "layer": str(layer),
            "num_experts": str(geometry["num_experts"]),
            "hidden_size": str(geometry["hidden_size"]),
            "intermediate_size": str(geometry["intermediate_size"]),
            "atom_channels": str(ATOM_CHANNELS),
        }
        _atomic_save(tensors, destination / filename, metadata)
        emitted_tensor_payload_bytes += sum(int(value.numel() * value.element_size()) for value in tensors.values())
        emitted_container_bytes += (destination / filename).stat().st_size
        layers[str(layer)] = {"file": filename, "sha256": sha256_file(destination / filename)}
    rates = (
        {"structure": "uniform", "bits": int(report["uniform_bits"])}
        if report["rate_structure"] == "uniform"
        else {"structure": "per_expert_pair", "pair_kinds": sorted(observed_kinds)}
    )
    manifest = {
        "kind": "btx-manifest",
        "schema": SCHEMA,
        "codebook": "mcg",
        "codebook_seed": MCG_MULTIPLIER,
        "geometry": {
            **geometry,
            "atom_channels": ATOM_CHANNELS,
            "atom_slots": geometry["intermediate_size"] // ATOM_CHANNELS,
            "moe_layer_indices": sorted(int(layer) for layer in layers),
        },
        "rates": rates,
        "hadamard": {"coupled": False, "per_expert_input_rotations": bool(per_expert_hidden)},
        "layout": {
            "atom_row_alignment": row_alignment,
            "extent_alignment_slots": ATOMS_PER_PAIR if report["rate_structure"] == "per_expert_pair" else 1,
            "extent_barriers": [],
        },
        "layers": layers,
    }
    write_json(destination / MANIFEST, manifest)
    accounting = {
        "schema": "quant-pipeline.official-btx-accounting.v1",
        "byte_semantics": "source-codec-payload-excludes-container-and-btx-padding",
        "installed_checkpoint_sha256": sorted(row["installed_checkpoint_sha256"] for row in verified_installed),
        "source_codec_logical_bytes": sum(int(row["logical_payload_bytes"]) for row in verified_installed),
        "source_semantic_allocated_payload_bytes": allocated_payload_bytes,
        "official_emitted_tensor_payload_bytes_including_padding_and_rate_tables": emitted_tensor_payload_bytes,
        "official_emitted_layer_container_bytes": emitted_container_bytes,
        "official_manifest_sha256": sha256_file(destination / MANIFEST),
    }
    accounting["accounting_sha256"] = sha256_bytes(canonical_json(accounting))
    write_json(destination / ACCOUNTING, accounting)
    return manifest


def _strict_manifest(data: Mapping[str, Any]) -> None:
    required = {"kind", "schema", "codebook", "codebook_seed", "geometry", "rates", "hadamard", "layout", "layers"}
    if set(data) != required or data["kind"] != "btx-manifest" or data["schema"] != SCHEMA or data["codebook"] != "mcg" or data["codebook_seed"] != MCG_MULTIPLIER:
        raise ValueError("BTX manifest top-level contract mismatch")
    geometry = data["geometry"]
    if set(geometry) != {"num_experts", "hidden_size", "intermediate_size", "atom_channels", "atom_slots", "moe_layer_indices"}:
        raise ValueError("BTX geometry keys mismatch")
    if geometry["atom_channels"] != 32 or geometry["atom_slots"] * 32 != geometry["intermediate_size"] or geometry["hidden_size"] % 16:
        raise ValueError("BTX geometry arithmetic mismatch")
    rates = data["rates"]
    if rates.get("structure") == "uniform":
        if set(rates) != {"structure", "bits"} or rates["bits"] not in {3, 4, 5, 6}:
            raise ValueError("BTX uniform rates mismatch")
    elif rates.get("structure") == "per_expert_pair":
        if set(rates) != {"structure", "pair_kinds"} or any(kind not in PAIR_CODES for kind in rates["pair_kinds"]):
            raise ValueError("BTX pair-rate vocabulary mismatch")
    else:
        raise ValueError("unknown BTX rate structure")
    if set(data["layers"]) != {str(layer) for layer in geometry["moe_layer_indices"]}:
        raise ValueError("BTX layer coverage mismatch")


def audit_official_btx_checkpoint(
    root: str | Path,
    *,
    runtime_reader: Callable[[Path], Mapping[str, Any]] | None = None,
    require_runtime_reader: bool = True,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    root = Path(root)
    failures: list[str] = []
    try:
        manifest = json.loads((root / MANIFEST).read_text())
        _strict_manifest(manifest)
    except Exception as error:
        return {"ok": False, "failures": [f"manifest:{error}"]}
    expected_files = {MANIFEST, ACCOUNTING} | {row["file"] for row in manifest["layers"].values()}
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        failures.append("file-set")
    geometry = manifest["geometry"]
    per_expert = manifest["rates"]["structure"] == "per_expert_pair"
    hidden_shape = ((geometry["num_experts"], geometry["hidden_size"]) if manifest["hadamard"]["per_expert_input_rotations"] else (geometry["hidden_size"],))
    observed_tensor_payload_bytes = 0
    observed_container_bytes = 0
    for layer_text, ref in manifest["layers"].items():
        path = root / ref["file"]
        if not path.is_file() or sha256_file(path) != ref["sha256"]:
            failures.append(f"layer-sha:{layer_text}")
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            observed_container_bytes += path.stat().st_size
            metadata = handle.metadata() or {}
            expected_metadata = {"schema": SCHEMA, "codebook": "mcg", "layer": layer_text, "num_experts": str(geometry["num_experts"]), "hidden_size": str(geometry["hidden_size"]), "intermediate_size": str(geometry["intermediate_size"]), "atom_channels": "32"}
            if metadata != expected_metadata:
                failures.append(f"metadata:{layer_text}")
            required = {"atoms", "rotations", "gate_suh", "up_suh", "down_svh"} | ({"rates_fc1", "rates_fc2"} if per_expert else set())
            if set(handle.keys()) != required:
                failures.append(f"tensor-set:{layer_text}")
                continue
            observed_tensor_payload_bytes += sum(
                int(handle.get_tensor(name).numel() * handle.get_tensor(name).element_size())
                for name in handle.keys()
            )
            atoms = handle.get_tensor("atoms")
            rotations = handle.get_tensor("rotations")
            if atoms.dtype != torch.uint8 or atoms.shape[0] != geometry["atom_slots"] or atoms.shape[1] % manifest["layout"]["atom_row_alignment"]:
                failures.append(f"atoms:{layer_text}")
            if rotations.dtype != torch.float16 or tuple(rotations.shape) != (geometry["atom_slots"], geometry["num_experts"], 3, 32) or not torch.isfinite(rotations).all():
                failures.append(f"rotations:{layer_text}")
            for name in ("gate_suh", "up_suh", "down_svh"):
                value = handle.get_tensor(name)
                if value.dtype != torch.float16 or tuple(value.shape) != hidden_shape or not torch.isfinite(value).all():
                    failures.append(f"hidden-vector:{layer_text}:{name}")
            if per_expert:
                tables = [handle.get_tensor(name) for name in ("rates_fc1", "rates_fc2")]
                pair_shape = (geometry["atom_slots"] // 8, geometry["num_experts"])
                for table in tables:
                    if table.dtype != torch.uint8 or tuple(table.shape) != pair_shape or any(int(code) not in CODE_KINDS for code in table.unique().tolist()):
                        failures.append(f"rate-table:{layer_text}")
            else:
                code = (manifest["rates"]["bits"] << 4) | manifest["rates"]["bits"]
                tables = [torch.full((geometry["atom_slots"] // 8, geometry["num_experts"]), code, dtype=torch.uint8)] * 2
            # Derive every row's used prefix and prove padding is zero.
            for slot in range(geometry["atom_slots"]):
                pair = slot // 8
                used = 0
                for expert in range(geometry["num_experts"]):
                    used += 2 * _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(tables[0][pair, expert])))
                    used += _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(tables[1][pair, expert])))
                if used > atoms.shape[1] or bool(torch.any(atoms[slot, used:] != 0)):
                    failures.append(f"atom-padding:{layer_text}:{slot}")
                    break
    accounting = None
    try:
        accounting = json.loads((root / ACCOUNTING).read_text())
        if accounting.get("schema") != "quant-pipeline.official-btx-accounting.v1":
            raise ValueError("schema")
        seal = accounting.get("accounting_sha256")
        if sha256_bytes(canonical_json({key: value for key, value in accounting.items() if key != "accounting_sha256"})) != seal:
            raise ValueError("seal")
        if accounting.get("official_manifest_sha256") != sha256_file(root / MANIFEST):
            raise ValueError("manifest")
        if accounting.get("official_emitted_tensor_payload_bytes_including_padding_and_rate_tables") != observed_tensor_payload_bytes:
            raise ValueError("tensor-bytes")
        if accounting.get("official_emitted_layer_container_bytes") != observed_container_bytes:
            raise ValueError("container-bytes")
    except Exception as error:
        failures.append(f"accounting:{error}")
    runtime = None
    if runtime_reader is None:
        if require_runtime_reader:
            failures.append("runtime-reader-unavailable")
    else:
        runtime = runtime_reader(root)
        if not isinstance(runtime, Mapping) or runtime.get("ok") is not True:
            failures.append("runtime-load")
    return {"ok": not failures, "failures": failures, "runtime": dict(runtime or {}), "upstream_commit": UPSTREAM_COMMIT, "manifest_sha256": sha256_file(root / MANIFEST), "accounting": dict(accounting or {})}


class UpstreamBtxRuntimeReader:
    """Load emitted files through the pinned upstream B12X reader.

    The local checkout may contain unrelated work, so validation binds the
    exact four-file reader/schema/writer closure to ``UPSTREAM_COMMIT`` rather
    than assuming that the checkout's current branch is clean.  Any already
    imported ``b12x`` module from another checkout is rejected; silently
    reusing a different runtime would invalidate the audit.
    """

    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).resolve()

    def _verify_source(self) -> dict[str, Any]:
        if not (self.source_root / ".git").exists():
            raise ValueError("B12X runtime source_root is not a Git checkout")
        observed = {}
        for relative, expected in UPSTREAM_CLOSURE_SHA256.items():
            path = self.source_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"pinned upstream BTX closure drifted: {relative}")
            observed[relative] = actual
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"],
                cwd=self.source_root,
                check=True,
                capture_output=True,
                text=True,
            )
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.source_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError("cannot verify pinned upstream BTX Git identity") from error
        return {"commit": UPSTREAM_COMMIT, "checkout_head": head, "closure_sha256": observed}

    def __call__(self, checkpoint_root: Path) -> Mapping[str, Any]:
        source = self._verify_source()
        incumbent = {
            name: Path(str(getattr(module, "__file__", ""))).resolve()
            for name, module in sys.modules.items()
            if name == "b12x" or name.startswith("b12x.")
        }
        foreign = {
            name: path
            for name, path in incumbent.items()
            if path != self.source_root / "b12x" / "__init__.py" and self.source_root not in path.parents
        }
        if foreign:
            raise RuntimeError(f"refusing already-imported foreign B12X runtime: {sorted(foreign)}")
        source_text = str(self.source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        module = importlib.import_module("b12x.moe._shared.kernels.w4a16.btx")
        loaded = Path(str(getattr(module, "__file__", ""))).resolve()
        if self.source_root not in loaded.parents:
            raise RuntimeError(f"upstream BTX reader resolved outside pinned source_root: {loaded}")
        manifest = module.read_btx_manifest(checkpoint_root)
        slots = int(manifest.geometry.atom_slots)
        layer_rows = []
        for layer in manifest.geometry.moe_layer_indices:
            loaded_layer = module.read_btx_layer(
                checkpoint_root,
                manifest,
                int(layer),
                first_slot=0,
                slot_count=slots,
                verify_sha=True,
            )
            layer_rows.append(
                {
                    "layer": int(layer),
                    "first_slot": int(loaded_layer.first_slot),
                    "slot_count": int(loaded_layer.slot_count),
                    "atom_shape": list(loaded_layer.atoms.shape),
                }
            )
        return {
            "ok": True,
            "reader": "b12x.moe._shared.kernels.w4a16.btx",
            "source": source,
            "layers": layer_rows,
        }


def unpack_official_btx_plane(
    root: str | Path,
    *,
    layer: int,
    slot: int,
    expert: int,
    projection: str,
    plane: str,
):
    """Recover one logical int16 plane from an official atom row.

    This independent address derivation is used as a packer round-trip oracle;
    it follows the upstream expert-major gate/up/down, low/high ordering.
    """

    import torch
    from safetensors import safe_open

    if projection not in {"gate_proj", "up_proj", "down_proj"} or plane not in {"low", "high"}:
        raise ValueError("invalid BTX projection/plane")
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text())
    _strict_manifest(manifest)
    geometry = manifest["geometry"]
    ref = manifest["layers"][str(layer)]
    with safe_open(root / ref["file"], framework="pt", device="cpu") as handle:
        atoms = handle.get_tensor("atoms")
        if manifest["rates"]["structure"] == "uniform":
            code = (manifest["rates"]["bits"] << 4) | manifest["rates"]["bits"]
            rates_fc1 = rates_fc2 = torch.full(
                (geometry["atom_slots"] // 8, geometry["num_experts"]), code, dtype=torch.uint8
            )
        else:
            rates_fc1 = handle.get_tensor("rates_fc1")
            rates_fc2 = handle.get_tensor("rates_fc2")
    pair = slot // 8
    cursor = 0
    for preceding in range(expert):
        cursor += 2 * _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(rates_fc1[pair, preceding])))
        cursor += _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(rates_fc2[pair, preceding])))
    matrix_index = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}[projection]
    code = int(rates_fc1[pair, expert]) if matrix_index < 2 else int(rates_fc2[pair, expert])
    low_bits, high_bits = _bits(code)
    matrix_sizes = [
        _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(rates_fc1[pair, expert]))),
        _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(rates_fc1[pair, expert]))),
        _matrix_atom_bytes(geometry["hidden_size"], *_bits(int(rates_fc2[pair, expert]))),
    ]
    cursor += sum(matrix_sizes[:matrix_index])
    low_bytes = (geometry["hidden_size"] // 16) * 16 * low_bits * 2
    bits = low_bits if plane == "low" else high_bits
    if plane == "high":
        cursor += low_bytes
    byte_count = (geometry["hidden_size"] // 16) * 16 * bits * 2
    raw = atoms[slot, cursor : cursor + byte_count].contiguous()
    return raw.view(torch.int16).reshape(geometry["hidden_size"] // 16, 16 * bits)
