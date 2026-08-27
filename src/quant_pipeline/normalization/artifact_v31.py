"""Canonical, byte-sealed absolute-v31/GSS production artifact.

Raw transform-search vectors are proposals, never production vectors.  The
only constructor in this module runs each proposal through the proven layer
absolute-v31 fit and a receipt-bound, injected GSS producer for K3/K4/K5.
"""

from __future__ import annotations

import json
import io
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..core.artifacts import atomic_write, canonical_json, prepare_empty_destination, sha256_bytes, sha256_file, write_json
from .absolute_v31 import (
    MatrixInput,
    fit_layer_absolute_normalization,
    tensor_identity_sha256,
    tensor_sha256,
)


SCHEMA = "quant-pipeline.absolute-v31-gss.v2"
GSS_RECEIPT_SCHEMA = "quant-pipeline.pinned-gss-receipt.v1"
BITS = (3, 4, 5)
_HASH_RE = re.compile(r"[0-9a-f]{64}")


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _identities(value: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty mapping")
    result = {str(key): _hash(item, f"{label}.{key}") for key, item in value.items()}
    if any(not key or key.strip() != key for key in result) or len(result) != len(value):
        raise ValueError(f"{label} has noncanonical keys")
    return dict(sorted(result.items()))


def _expert(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("expert ID cannot be boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("expert ID must be nonnegative")
        return str(value)
    if isinstance(value, str) and value and value.strip() == value:
        return str(int(value)) if value.isdecimal() else value
    raise ValueError("expert ID is not canonical")


def _matrix_identity(key: str) -> tuple[str, str]:
    match = re.fullmatch(r"E([^.]*)\.(gate_proj|up_proj|down_proj)", key)
    if match is None or _expert(match.group(1)) != match.group(1):
        raise ValueError(f"matrix key is not canonical: {key!r}")
    return match.group(1), match.group(2)


def _array_record(array: np.ndarray, name: str) -> dict[str, Any]:
    return {
        "array": name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256_bytes(array.tobytes(order="C")),
    }


@dataclass(frozen=True)
class PinnedGSSRequest:
    matrix_key: str
    bits: int
    target: Any
    target_sha256: str
    source_weight_identity_sha256: str
    predecessor_checkpoint_hash: str


@dataclass(frozen=True)
class PinnedGSSResult:
    scale: float
    receipt: Mapping[str, Any]


class PinnedGSSProducer(Protocol):
    """Injected exact evaluator.  Returning a bare scalar is insufficient."""

    def search(self, request: PinnedGSSRequest) -> PinnedGSSResult: ...


@dataclass(frozen=True)
class AbsoluteV31Artifact:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def content_sha256(self) -> str:
        verify_absolute_v31_artifact(self)
        inventory = {
            name: _array_record(np.asarray(array), name)
            for name, array in sorted(self.arrays.items())
        }
        return sha256_bytes(canonical_json({"metadata": self.metadata, "arrays": inventory}))


# Compatibility name, but there is only one schema and validator.
AbsoluteV31BaselineArtifact = AbsoluteV31Artifact


@dataclass(frozen=True)
class CandidateEvaluation:
    method: str
    score: float
    artifact_sha256: str
    evaluator_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class AdditiveV31CandidateResult:
    artifact: AbsoluteV31Artifact
    exact_codec_proxy: CandidateEvaluation
    heldout_full_expert: CandidateEvaluation


def make_candidate_evaluation(
    artifact: AbsoluteV31Artifact, *, method: str, score: float, evaluator_sha256: str
) -> CandidateEvaluation:
    if method not in {"exact_codec_proxy", "heldout_full_expert_roundtrip"}:
        raise ValueError("unknown additive-candidate evaluation method")
    if not math.isfinite(float(score)) or float(score) < 0.0:
        raise ValueError("candidate evaluation score must be finite and nonnegative")
    _hash(evaluator_sha256, "candidate evaluator identity")
    artifact_sha256 = artifact.content_sha256
    body = {
        "method": method, "score": float(score), "artifact_sha256": artifact_sha256,
        "evaluator_sha256": evaluator_sha256,
    }
    return CandidateEvaluation(
        method, float(score), artifact_sha256, evaluator_sha256,
        sha256_bytes(canonical_json(body)),
    )


def _validate_candidate_evaluation(
    value: Any, *, method: str, artifact: AbsoluteV31Artifact
) -> CandidateEvaluation:
    if not isinstance(value, CandidateEvaluation):
        raise TypeError("candidate evaluator must return CandidateEvaluation")
    expected = make_candidate_evaluation(
        artifact, method=method, score=value.score, evaluator_sha256=value.evaluator_sha256
    )
    if value != expected:
        raise ValueError("candidate evaluation receipt/artifact binding mismatch")
    return value


def evaluate_additive_v31_candidate(
    *, exact_codec_evaluator: Any, heldout_evaluator: Any, **production_kwargs: Any
) -> AdditiveV31CandidateResult:
    """Mandatory order: exact v31/GSS, exact codec proxy, held-out roundtrip."""

    if not callable(exact_codec_evaluator) or not callable(heldout_evaluator):
        raise TypeError("both exact-codec and held-out evaluators are required")
    artifact = produce_absolute_v31_artifact(**production_kwargs)
    proxy = _validate_candidate_evaluation(
        exact_codec_evaluator(artifact), method="exact_codec_proxy", artifact=artifact
    )
    heldout = _validate_candidate_evaluation(
        heldout_evaluator(artifact), method="heldout_full_expert_roundtrip", artifact=artifact
    )
    return AdditiveV31CandidateResult(artifact, proxy, heldout)


def make_gss_receipt(
    request: PinnedGSSRequest,
    *,
    scale: float,
    evaluator_code_sha256: str,
    codec_identity_sha256: str,
    search_config_sha256: str,
    evaluations: int,
) -> dict[str, Any]:
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("GSS scale must be finite and positive")
    if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations < 2:
        raise ValueError("pinned GSS must attest a real multi-point search")
    body = {
        "schema": GSS_RECEIPT_SCHEMA,
        "matrix_key": request.matrix_key,
        "bits": request.bits,
        "target_sha256": request.target_sha256,
        "source_weight_identity_sha256": request.source_weight_identity_sha256,
        "predecessor_checkpoint_hash": request.predecessor_checkpoint_hash,
        "scale": float(scale),
        "evaluator_code_sha256": _hash(evaluator_code_sha256, "GSS evaluator identity"),
        "codec_identity_sha256": _hash(codec_identity_sha256, "GSS codec identity"),
        "search_config_sha256": _hash(search_config_sha256, "GSS search identity"),
        "evaluations": evaluations,
    }
    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    return body


def _validate_gss_result(request: PinnedGSSRequest, value: Any) -> PinnedGSSResult:
    if not isinstance(value, PinnedGSSResult):
        raise TypeError("pinned GSS producer must return PinnedGSSResult, not a scalar")
    if isinstance(value.scale, bool):
        raise ValueError("GSS result scale must be finite and positive, not boolean")
    scale = float(value.scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("GSS result scale must be finite and positive")
    receipt = dict(value.receipt)
    expected = {
        "schema", "matrix_key", "bits", "target_sha256", "source_weight_identity_sha256",
        "predecessor_checkpoint_hash", "scale", "evaluator_code_sha256", "codec_identity_sha256",
        "search_config_sha256", "evaluations", "receipt_sha256",
    }
    if set(receipt) != expected or receipt.get("schema") != GSS_RECEIPT_SCHEMA:
        raise ValueError("GSS receipt is incomplete or unknown")
    if isinstance(receipt.get("scale"), bool) or not isinstance(receipt.get("scale"), (int, float)):
        raise ValueError("GSS receipt scale must be numeric and non-boolean")
    bindings = {
        "matrix_key": request.matrix_key,
        "bits": request.bits,
        "target_sha256": request.target_sha256,
        "source_weight_identity_sha256": request.source_weight_identity_sha256,
        "predecessor_checkpoint_hash": request.predecessor_checkpoint_hash,
        "scale": scale,
    }
    if any(receipt.get(key) != expected_value for key, expected_value in bindings.items()):
        raise ValueError("GSS receipt request/scale binding mismatch")
    for field in ("evaluator_code_sha256", "codec_identity_sha256", "search_config_sha256"):
        _hash(receipt[field], field)
    if isinstance(receipt["evaluations"], bool) or not isinstance(receipt["evaluations"], int) or receipt["evaluations"] < 2:
        raise ValueError("GSS receipt does not prove a multi-point search")
    seal = receipt["receipt_sha256"]
    _hash(seal, "GSS receipt seal")
    body = dict(receipt)
    del body["receipt_sha256"]
    if sha256_bytes(canonical_json(body)) != seal:
        raise ValueError("GSS receipt content hash mismatch")
    return PinnedGSSResult(scale=scale, receipt=receipt)


def produce_absolute_v31_artifact(
    *,
    core: Any,
    matrices: Sequence[MatrixInput],
    producer: PinnedGSSProducer,
    selected_bits: Mapping[str, int],
    selection_decision_sha256: Mapping[str, str],
    layer_id: int,
    predecessor_checkpoint_hash: str,
    source_identities: Mapping[str, str],
    core_identities: Mapping[str, str],
    codebook_scale: float,
    block: int = 128,
) -> AbsoluteV31Artifact:
    """Run proposal signs/factors through exact v31 and pinned K3/K4/K5 GSS."""

    if not hasattr(producer, "search") or not callable(producer.search):
        raise TypeError("producer must implement the pinned GSS interface")
    _hash(predecessor_checkpoint_hash, "predecessor checkpoint")
    if isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0:
        raise ValueError("layer_id must be a nonnegative integer")
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError("block must be a positive integer")
    if not matrices:
        raise ValueError("absolute-v31 production requires a complete layer matrix set")
    keys = [item.key for item in matrices]
    if len(set(keys)) != len(keys):
        raise ValueError("matrix keys must be unique")
    identities = {item.key: _matrix_identity(item.key) for item in matrices}
    experts = sorted({expert for expert, _ in identities.values()}, key=lambda item: (not item.isdecimal(), int(item) if item.isdecimal() else item))
    expected = {f"E{expert}.{projection}" for expert in experts for projection in ("gate_proj", "up_proj", "down_proj")}
    if set(keys) != expected:
        raise ValueError("absolute-v31 production requires every expert gate/up/down matrix")
    if set(selected_bits) != expected or set(selection_decision_sha256) != expected:
        raise ValueError("selected-bit decisions must cover the exact matrix inventory")

    arrays: dict[str, np.ndarray] = {}
    candidate_records: dict[str, dict[str, Any]] = {key: {} for key in sorted(expected)}
    shared_records: dict[str, dict[str, Any]] | None = None
    source_hashes: dict[str, str] = {}
    for bits in BITS:
        bit_inputs = [replace(item, bits=bits) for item in matrices]
        fit = fit_layer_absolute_normalization(core, bit_inputs, codebook_scale=codebook_scale, block=block)
        targets = fit.gss_targets()
        scales: dict[str, float] = {}
        receipts: dict[str, Mapping[str, Any]] = {}
        for key in sorted(expected):
            target_hash = tensor_sha256(targets[key])
            source_hash = fit.matrices[key].source.weight_kn
            source_identity = tensor_identity_sha256(source_hash)
            request = PinnedGSSRequest(
                matrix_key=key, bits=bits, target=targets[key], target_sha256=target_hash,
                source_weight_identity_sha256=source_identity,
                predecessor_checkpoint_hash=predecessor_checkpoint_hash,
            )
            result = _validate_gss_result(request, producer.search(request))
            scales[key] = result.scale
            receipts[key] = result.receipt
            source_hashes.setdefault(key, source_identity)
            if source_hashes[key] != source_identity:
                raise ValueError("source weights drifted across bit-specific v31 fits")
        finalized = fit.finalize(scales)
        current_shared = {
            "gate_up_suh": np.ascontiguousarray(fit.shared_gate_up_suh.detach().cpu().numpy()),
            "down_svh": np.ascontiguousarray(fit.shared_down_svh.detach().cpu().numpy()),
        }
        if shared_records is None:
            shared_records = {}
            for name, array in current_shared.items():
                array_name = f"shared.{name}"
                arrays[array_name] = array.copy()
                shared_records[name] = _array_record(arrays[array_name], array_name)
        else:
            for name, array in current_shared.items():
                if not np.array_equal(array, arrays[f"shared.{name}"]):
                    raise ValueError("v31 shared vectors drifted across K3/K4/K5")
        for key, value in finalized.matrices.items():
            private_side = value.gss_fold_side
            tensor = value.stored_svh if private_side == "svh" else value.stored_suh
            array = np.ascontiguousarray(tensor.detach().cpu().numpy())
            array_name = f"{key}.K{bits}.private_{private_side}"
            arrays[array_name] = array.copy()
            receipt = dict(receipts[key])
            candidate_records[key][str(bits)] = {
                "bits": bits,
                "g_scale": value.g_scale,
                "gss_target_sha256": receipt["target_sha256"],
                "gss_receipt": receipt,
                "private_side": private_side,
                "private_vector": _array_record(arrays[array_name], array_name),
            }

    assert shared_records is not None
    records = []
    by_key = {item.key: item for item in matrices}
    for key in sorted(expected):
        expert_id, projection = identities[key]
        bits = selected_bits[key]
        if isinstance(bits, bool) or bits not in BITS:
            raise ValueError(f"{key} has invalid selected bit")
        records.append({
            "key": key,
            "expert_id": expert_id,
            "projection": projection,
            "mass": float(by_key[key].mass),
            "source_weight_identity_sha256": source_hashes[key],
            "shared_side": "suh" if projection != "down_proj" else "svh",
            "shared_vector": shared_records["gate_up_suh" if projection != "down_proj" else "down_svh"],
            "candidates": candidate_records[key],
            "selection": {
                "bits": bits,
                "decision_sha256": _hash(selection_decision_sha256[key], f"{key} selection decision"),
            },
        })
    metadata = {
        "schema": SCHEMA,
        "identity": {
            "layer_id": layer_id,
            "expert_ids": experts,
            "block": block,
            "predecessor_checkpoint_hash": predecessor_checkpoint_hash,
            "source_identities": _identities(source_identities, "source identities"),
            "core_identities": _identities(core_identities, "core identities"),
        },
        "policy": {
            "control": "source-derived-absolute-v31",
            "candidate_relationship": "additive-unless-matched-exact-codec-and-kld-ablation-wins",
            "gss": "injected-pinned-per-matrix-per-bit-receipt-bound",
            "bits": list(BITS),
            "vector_boundary": "fp16-checkpoint-bytes",
        },
        "matrices": records,
    }
    artifact = AbsoluteV31Artifact(metadata=metadata, arrays=arrays)
    verify_absolute_v31_artifact(artifact)
    return artifact


def verify_absolute_v31_artifact(value: AbsoluteV31Artifact) -> None:
    if not isinstance(value, AbsoluteV31Artifact):
        raise TypeError("value must be AbsoluteV31Artifact")
    metadata = value.metadata
    if not isinstance(metadata, dict) or set(metadata) != {"schema", "identity", "policy", "matrices"} or metadata["schema"] != SCHEMA:
        raise ValueError("unsupported or malformed absolute-v31 artifact")
    identity = metadata["identity"]
    if not isinstance(identity, dict) or set(identity) != {"layer_id", "expert_ids", "block", "predecessor_checkpoint_hash", "source_identities", "core_identities"}:
        raise ValueError("absolute-v31 identity is incomplete or unknown")
    if (
        isinstance(identity["layer_id"], bool)
        or not isinstance(identity["layer_id"], int)
        or identity["layer_id"] < 0
    ):
        raise ValueError("absolute-v31 layer_id must be a nonnegative integer")
    block = identity["block"]
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError("absolute-v31 block must be a positive integer")
    _hash(identity["predecessor_checkpoint_hash"], "predecessor checkpoint")
    _identities(identity["source_identities"], "source identities")
    _identities(identity["core_identities"], "core identities")
    expected_policy = {
        "control": "source-derived-absolute-v31",
        "candidate_relationship": "additive-unless-matched-exact-codec-and-kld-ablation-wins",
        "gss": "injected-pinned-per-matrix-per-bit-receipt-bound",
        "bits": list(BITS),
        "vector_boundary": "fp16-checkpoint-bytes",
    }
    if metadata["policy"] != expected_policy:
        raise ValueError("absolute-v31 policy mismatch")
    experts = identity["expert_ids"]
    if not isinstance(experts, list) or not experts or experts != sorted({_expert(item) for item in experts}, key=lambda item: (not item.isdecimal(), int(item) if item.isdecimal() else item)):
        raise ValueError("absolute-v31 expert inventory is not canonical")
    expected_keys = {f"E{expert}.{projection}" for expert in experts for projection in ("gate_proj", "up_proj", "down_proj")}
    records = metadata["matrices"]
    if not isinstance(records, list) or {item.get("key") for item in records if isinstance(item, dict)} != expected_keys or len(records) != len(expected_keys):
        raise ValueError("absolute-v31 matrix inventory is incomplete")
    expected_arrays: set[str] = set()
    shared_hashes = {"gate_up": set(), "down": set()}
    geometry: dict[str, dict[str, tuple[int, tuple[int, ...]]]] = {}
    for record in records:
        required = {"key", "expert_id", "projection", "mass", "source_weight_identity_sha256", "shared_side", "shared_vector", "candidates", "selection"}
        if set(record) != required:
            raise ValueError("absolute-v31 matrix record is incomplete or unknown")
        expert, projection = _matrix_identity(record["key"])
        if record["expert_id"] != expert or record["projection"] != projection:
            raise ValueError("absolute-v31 matrix identity mismatch")
        mass = record["mass"]
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(float(mass)) or float(mass) <= 0.0:
            raise ValueError("absolute-v31 matrix mass must be finite and positive")
        _hash(record["source_weight_identity_sha256"], "source weight identity")
        shared_side = "suh" if projection != "down_proj" else "svh"
        if record["shared_side"] != shared_side:
            raise ValueError("absolute-v31 shared-side topology mismatch")
        expected_shared_name = "shared.gate_up_suh" if projection != "down_proj" else "shared.down_svh"
        if not isinstance(record["shared_vector"], dict) or record["shared_vector"].get("array") != expected_shared_name:
            raise ValueError("absolute-v31 shared vector name/topology mismatch")
        shared_array = _verify_array_reference(value, record["shared_vector"], expected_arrays)
        shared_hashes["gate_up" if projection != "down_proj" else "down"].add(record["shared_vector"]["sha256"])
        candidates = record["candidates"]
        if not isinstance(candidates, dict) or set(candidates) != {str(bit) for bit in BITS}:
            raise ValueError("absolute-v31 must retain exact K3/K4/K5 candidates")
        for bit in BITS:
            candidate = candidates[str(bit)]
            required_candidate = {"bits", "g_scale", "gss_target_sha256", "gss_receipt", "private_side", "private_vector"}
            if not isinstance(candidate, dict) or set(candidate) != required_candidate or candidate["bits"] != bit:
                raise ValueError("absolute-v31 bit candidate malformed")
            if candidate["private_side"] != ("svh" if projection != "down_proj" else "suh"):
                raise ValueError("absolute-v31 GSS was folded onto shared side")
            expected_private_name = f"{record['key']}.K{bit}.private_{candidate['private_side']}"
            if not isinstance(candidate["private_vector"], dict) or candidate["private_vector"].get("array") != expected_private_name:
                raise ValueError("absolute-v31 private vector name/topology mismatch")
            private_array = _verify_array_reference(value, candidate["private_vector"], expected_arrays)
            geometry.setdefault(expert, {}).setdefault(projection, (shared_array.size, tuple()))
            shared_size, private_sizes = geometry[expert][projection]
            if shared_size != shared_array.size:
                raise ValueError("absolute-v31 candidate shared-vector geometry drifted")
            geometry[expert][projection] = (shared_size, (*private_sizes, private_array.size))
            request = PinnedGSSRequest(
                matrix_key=record["key"], bits=bit, target=None,
                target_sha256=candidate["gss_target_sha256"],
                source_weight_identity_sha256=record["source_weight_identity_sha256"],
                predecessor_checkpoint_hash=identity["predecessor_checkpoint_hash"],
            )
            _validate_gss_result(request, PinnedGSSResult(candidate["g_scale"], candidate["gss_receipt"]))
        selection = record["selection"]
        if not isinstance(selection, dict) or set(selection) != {"bits", "decision_sha256"} or selection["bits"] not in BITS:
            raise ValueError("absolute-v31 selected decision malformed")
        if str(selection["bits"]) not in candidates or candidates[str(selection["bits"])]["bits"] != selection["bits"]:
            raise ValueError("absolute-v31 selected bit has no matching candidate")
        _hash(selection["decision_sha256"], "selection decision")
    dimensions: set[tuple[int, int]] = set()
    for expert, projections in geometry.items():
        if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"absolute-v31 expert {expert} projection geometry is incomplete")
        gate_shared, gate_private = projections["gate_proj"]
        up_shared, up_private = projections["up_proj"]
        down_shared, down_private = projections["down_proj"]
        if (
            gate_shared != up_shared
            or gate_shared != down_shared
            or len(set(gate_private)) != 1
            or len(set(up_private)) != 1
            or len(set(down_private)) != 1
            or gate_private[0] != up_private[0]
            or gate_private[0] != down_private[0]
        ):
            raise ValueError("absolute-v31 candidate/shared vector shapes are incompatible")
        dimensions.add((gate_shared, gate_private[0]))
    if len(dimensions) != 1:
        raise ValueError("absolute-v31 expert geometry differs across the layer")
    hidden_size, intermediate_size = next(iter(dimensions))
    if hidden_size % block or intermediate_size % block:
        raise ValueError("absolute-v31 block is incompatible with vector geometry")
    if any(len(hashes) != 1 for hashes in shared_hashes.values()) or set(value.arrays) != expected_arrays:
        raise ValueError("absolute-v31 shared vectors or array inventory drifted")


def _verify_array_reference(
    value: AbsoluteV31Artifact, record: Any, expected_arrays: set[str]
) -> np.ndarray:
    if not isinstance(record, dict) or set(record) != {"array", "dtype", "shape", "sha256"}:
        raise ValueError("absolute-v31 vector record malformed")
    name = record["array"]
    if not isinstance(name, str) or name not in value.arrays:
        raise ValueError("absolute-v31 vector array missing")
    array = np.asarray(value.arrays[name])
    if (
        array.dtype != np.float16
        or array.ndim != 1
        or array.size < 1
        or record != _array_record(array, name)
        or not np.isfinite(array).all()
        or np.any(array == 0)
    ):
        raise ValueError("absolute-v31 vector bytes or boundary invalid")
    expected_arrays.add(name)
    return array


def save_absolute_v31_artifact(path: str | Path, value: AbsoluteV31Artifact) -> None:
    verify_absolute_v31_artifact(value)
    destination = prepare_empty_destination(path)
    inventory = {}
    for index, (name, array) in enumerate(sorted(value.arrays.items())):
        filename = f"array-{index:05d}.npy"
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(array), allow_pickle=False)
        atomic_write(destination / filename, buffer.getvalue())
        inventory[name] = {**_array_record(np.asarray(array), name), "file": filename, "file_sha256": sha256_file(destination / filename)}
    manifest = {"metadata": value.metadata, "arrays": inventory, "content_sha256": value.content_sha256}
    manifest["seal_sha256"] = sha256_bytes(canonical_json(manifest))
    write_json(destination / "manifest.json", manifest)


def load_absolute_v31_artifact(path: str | Path) -> AbsoluteV31Artifact:
    source = Path(path)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("absolute-v31 artifact root must be a regular local directory")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("absolute-v31 manifest must be a regular local file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = manifest.pop("seal_sha256", None)
    if not isinstance(seal, str) or sha256_bytes(canonical_json(manifest)) != seal:
        raise ValueError("absolute-v31 manifest seal mismatch")
    if set(manifest) != {"metadata", "arrays", "content_sha256"}:
        raise ValueError("absolute-v31 manifest fields are incomplete or unknown")
    _hash(manifest["content_sha256"], "absolute-v31 content identity")
    inventory = manifest["arrays"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("absolute-v31 array inventory is missing")
    arrays = {}
    files = {"manifest.json"}
    filenames: set[str] = set()
    for name, record in inventory.items():
        if not isinstance(record, dict) or set(record) != {
            "array", "dtype", "shape", "sha256", "file", "file_sha256"
        }:
            raise ValueError("absolute-v31 array file record is malformed")
        filename = record["file"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in filenames
            or re.fullmatch(r"array-[0-9]{5}\.npy", filename) is None
        ):
            raise ValueError("absolute-v31 array filename is not local/canonical")
        filenames.add(filename)
        _hash(record["file_sha256"], "absolute-v31 array file identity")
        file = source / filename
        files.add(filename)
        if not file.is_file() or file.is_symlink() or sha256_file(file) != record["file_sha256"]:
            raise ValueError("absolute-v31 array file hash mismatch")
        array = np.load(file, allow_pickle=False, mmap_mode="r")
        expected = {key: record[key] for key in ("array", "dtype", "shape", "sha256")}
        if expected != _array_record(array, name):
            raise ValueError("absolute-v31 array identity mismatch")
        arrays[name] = array
    if {item.name for item in source.iterdir()} != files:
        raise ValueError("absolute-v31 directory contains unbound files")
    result = AbsoluteV31Artifact(manifest["metadata"], arrays)
    verify_absolute_v31_artifact(result)
    if result.content_sha256 != manifest["content_sha256"]:
        raise ValueError("absolute-v31 content identity mismatch")
    return result
