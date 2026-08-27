"""Materialize exact Qwen candidate work units from sealed campaign artifacts.

This is deliberately separate from orchestration but remains inside the
service factory's sealed Python closure.  It loads local immutable Qwen shard
bytes, verifies all capture/fitter artifacts, evaluates the five historical
permutation controls crossed with the three scale families through canonical
absolute-v31 + K3/K4/K5 GSS, and returns the winning exact-codec work units.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..calibration.fitter import FittedExpertStatistics, load_fitted_statistics
from ..calibration.qwen_capture import verify_capture_chunk, verify_capture_manifest
from ..candidates.ledger import (
    ConditionalDownFitBatch,
    K5Decision,
    ProjectionTensors,
    RoutedExpertBatch,
    all_k5_triplets,
    build_expert_candidate_input,
    conditional_down_fit_batch_sha256,
    routed_batch_sha256,
    _codec_candidates,
    _conditional_down_hessian,
    _tensor_record,
)
from ..core.artifacts import canonical_json, sha256_bytes, sha256_file, write_json
from ..normalization.absolute_v31 import MatrixInput
from ..normalization.artifact_v31 import (
    AbsoluteV31Artifact,
    evaluate_additive_v31_candidate,
    make_candidate_evaluation,
    save_absolute_v31_artifact,
)
from ..normalization.prior_search import (
    PERMUTATION_POLICIES,
    SCALE_FAMILIES,
    permute_expert_hf,
    policy_permutations,
    scale_family_candidates,
)
from .qwen_services import CAPTURE_SERVICE_SCHEMA, CorrectedPinnedGSSProducer


def _hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _dependency(context: Mapping[str, Any], prefix: str) -> Path:
    matches = [Path(value) for key, value in context["dependencies"].items() if key == prefix or key.startswith(prefix + ".")]
    if len(matches) != 1:
        raise ValueError(f"expected one {prefix} dependency")
    candidate = matches[0]
    root = candidate.resolve()
    if candidate.is_symlink() or not root.is_dir():
        raise ValueError(f"dependency is not a regular artifact directory: {candidate}")
    return root


def _provider_result(root: Path) -> Mapping[str, Any]:
    value = json.loads((root / "stage-manifest.json").read_text()).get("provider_result")
    if not isinstance(value, dict):
        raise ValueError("stage dependency lacks provider result")
    return value


def _result_path(root: Path, key: str) -> Path:
    raw = _provider_result(root).get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"stage dependency lacks {key}")
    path = Path(raw)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"stage dependency {key} escapes or is missing")
    return path


def _load_capture_component(context: Mapping[str, Any], purpose: str) -> tuple[Path, dict[str, Any], str]:
    prefix = "causal_fit_capture" if context["kind"] == "causal_candidates" else "fit_capture"
    stage = _dependency(context, prefix)
    service_path = _result_path(stage, "capture_manifest_file")
    service = json.loads(service_path.read_text())
    if service.get("schema") != CAPTURE_SERVICE_SCHEMA:
        raise ValueError("candidate materializer requires the concrete multi-role capture")
    seal = service.get("capture_service_sha256")
    if seal != _hash_json({key: value for key, value in service.items() if key != "capture_service_sha256"}):
        raise ValueError("capture service seal mismatch")
    component = service["captures"][purpose]
    expected_role = {"fit": "fit", "heldout": "selection", "conditional_down": "conditional_fit"}.get(purpose)
    if expected_role is not None and component.get("role") != expected_role:
        raise ValueError(
            f"capture component {purpose} must use role {expected_role}; confirmation is prospective-only"
        )
    manifest_path = (service_path.parent / component["manifest"]).resolve()
    if service_path.parent.resolve() not in manifest_path.parents or manifest_path.is_symlink():
        raise ValueError("capture component manifest escapes its stage artifact")
    manifest = verify_capture_manifest(manifest_path.parent)
    return manifest_path.parent, manifest, service["capture_service_sha256"]


def _read_chunk(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    receipt = verify_capture_chunk(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return dict(receipt["metadata"]), tensors


class _CheckpointWeights:
    def __init__(self, root: str | Path) -> None:
        from safetensors import safe_open

        self.root = Path(root).resolve()
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.mapping = json.loads(index.read_text())["weight_map"]
        elif (self.root / "model.safetensors").is_file():
            with safe_open(self.root / "model.safetensors", framework="pt", device="cpu") as handle:
                self.mapping = {name: "model.safetensors" for name in handle.keys()}
        else:
            raise FileNotFoundError("source checkpoint has no safetensors index or monolith")
        self._cache: dict[str, Any] = {}

    def tensor(self, name: str):
        from safetensors import safe_open

        if name in self._cache:
            return self._cache[name]
        filename = self.mapping.get(name)
        if not isinstance(filename, str):
            raise KeyError(f"source checkpoint lacks {name}")
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name).contiguous()
        self._cache[name] = value
        return value

    def expert(self, layer: int, expert: int) -> ProjectionTensors:
        prefix = f"model.layers.{layer}.mlp.experts"
        stacked_gate_up = prefix + ".gate_up_proj"
        stacked_down = prefix + ".down_proj"
        if stacked_gate_up in self.mapping and stacked_down in self.mapping:
            stacked = self.tensor(stacked_gate_up)[expert]
            gate, up = stacked.chunk(2, dim=0)
            down = self.tensor(stacked_down)[expert]
        else:
            expert_prefix = f"{prefix}.{expert}"
            gate = self.tensor(expert_prefix + ".gate_proj.weight")
            up = self.tensor(expert_prefix + ".up_proj.weight")
            down = self.tensor(expert_prefix + ".down_proj.weight")
        return ProjectionTensors(gate.contiguous(), up.contiguous(), down.contiguous())


def _fit_rows(fit_path: Path, layer: int) -> dict[int, tuple[FittedExpertStatistics, FittedExpertStatistics]]:
    root = fit_path.parent
    manifest = json.loads(fit_path.read_text())
    seal = manifest.get("fit_sha256")
    if seal != _hash_json({key: value for key, value in manifest.items() if key != "fit_sha256"}):
        raise ValueError("fit manifest seal mismatch")
    result = {}
    for row in manifest["statistics"]:
        if int(row["layer"]) != layer:
            continue
        gate_path, down_path = root / row["gate_up"], root / row["down"]
        if sha256_file(gate_path / "manifest.json") != row["gate_up_manifest_sha256"]:
            raise ValueError("gate/up fit manifest drift")
        if sha256_file(down_path / "manifest.json") != row["down_manifest_sha256"]:
            raise ValueError("down fit manifest drift")
        result[int(row["expert"])] = (load_fitted_statistics(gate_path), load_fitted_statistics(down_path))
    if not result:
        raise ValueError(f"fit artifact contains no layer {layer}")
    return result


def _document_hash(metadata: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json({
        "document_id": str(metadata["document_id"]),
        "token_sha256": str(metadata["token_sha256"]),
    }))


def _all_expert_batches(
    root: Path,
    manifest: Mapping[str, Any],
    layer: int,
    experts: Sequence[int],
    *,
    conditional: bool,
):
    import torch

    requested = set(map(int, experts))
    batches: dict[int, list[Any]] = {expert: [] for expert in sorted(requested)}
    artifact_sha = str(manifest["capture_sha256"])
    for record in manifest["records"][str(layer)]:
        metadata, tensors = _read_chunk(root / record["file"])
        assignment_expert = tensors["assignment_expert_ids"].long()
        present = sorted(requested.intersection(map(int, torch.unique(assignment_expert).tolist())))
        for expert in present:
            selected = torch.nonzero(assignment_expert == expert, as_tuple=False).flatten()
            offsets = tensors["assignment_token_offsets"].long()[selected]
            hidden = tensors["routed_hidden_states"][selected].contiguous()
            route_weights = tensors["assignment_router_weights"][selected].contiguous()
            source_indices = tensors["expert_ids"].long()[offsets].contiguous()
            source_weights = tensors["router_weights"][offsets].contiguous()
            rows = [
                f"{metadata['document_id']}@{metadata['token_sha256']}:"
                f"{int(metadata['start_token']) + int(offset)}:L{layer}:E{expert}"
                for offset in offsets
            ]
            document_sha = _document_hash(metadata)
            batch_id = f"{metadata['role']}:{metadata['window_index']}:L{layer}:E{expert}"
            if conditional:
                identity = {
                    "role": str(metadata["role"]),
                    "conditional_down_fit_artifact_sha256": artifact_sha,
                    "row_identity_sha256": _hash_json(rows),
                    "document_sha256": document_sha,
                    "layer": layer,
                    "expert": expert,
                }
                batch = ConditionalDownFitBatch(
                    batch_id=batch_id,
                    hidden_states=hidden,
                    route_weights=route_weights,
                    sampling_weights=torch.ones_like(route_weights),
                    source_route_indices=source_indices,
                    source_route_weights=source_weights,
                    identity=identity,
                    row_keys=rows,
                )
                batch = replace(
                    batch,
                    identity=identity | {"batch_payload_sha256": conditional_down_fit_batch_sha256(batch)},
                )
            else:
                fisher = None
                if "fisher_gradients" in tensors:
                    raw = tensors["fisher_gradients"]
                    if raw.ndim != 4 or raw.shape[1] != 1:
                        raise ValueError("Qwen Fisher capture must have one batch dimension per sealed window")
                    fisher = raw[:, 0, offsets].contiguous()
                identity = {
                    "role": str(metadata["role"]),
                    "document_sha256": document_sha,
                    "heldout_artifact_sha256": artifact_sha,
                    "capture_artifact_sha256": artifact_sha,
                    "fisher_probe_sha256": _hash_json({
                        "capture": artifact_sha, "rank": int(manifest["fisher_rank"]),
                    }),
                    "fisher_window_sha256": _hash_json({
                        "capture": artifact_sha, "window_set": "all",
                    }),
                    "layer": layer,
                    "expert": expert,
                }
                batch = RoutedExpertBatch(
                    batch_id=batch_id,
                    hidden_states=hidden,
                    route_weights=route_weights,
                    source_route_indices=source_indices,
                    source_route_weights=source_weights,
                    candidate_route_indices=source_indices.clone(),
                    candidate_route_weights=source_weights.clone(),
                    fisher_gradients=fisher,
                    identity=identity,
                    row_keys=rows,
                )
                batch = replace(
                    batch,
                    identity=identity | {"batch_payload_sha256": routed_batch_sha256(batch)},
                )
            batches[expert].append(batch)
    missing = [expert for expert, rows in batches.items() if not rows]
    if missing:
        raise ValueError(f"capture has no rows for layer {layer} experts {missing[:8]}")
    return {expert: tuple(rows) for expert, rows in batches.items()}


def _expert_batches(root: Path, manifest: Mapping[str, Any], layer: int, expert: int, *, conditional: bool):
    """Compatibility wrapper around the one-pass layer partitioner."""

    return _all_expert_batches(root, manifest, layer, (expert,), conditional=conditional)[expert]


def _permuted_statistics(value: FittedExpertStatistics, permutation: Sequence[int]) -> FittedExpertStatistics:
    arrays = {}
    index = np.asarray(permutation, dtype=np.int64)
    for name, array in value.arrays.items():
        raw = np.asarray(array)
        if name.endswith(".mean"):
            arrays[name] = np.ascontiguousarray(raw[index])
        elif name.endswith(".second_moment"):
            arrays[name] = np.ascontiguousarray(raw[np.ix_(index, index)])
        else:
            arrays[name] = np.ascontiguousarray(raw)
    return FittedExpertStatistics(metadata=value.metadata, arrays=arrays)


def _sign(length: int, seed: str, *domain: Any):
    import torch

    value = int(sha256_bytes(canonical_json([seed, *domain]))[:16], 16)
    generator = torch.Generator(device="cpu").manual_seed(value)
    return (torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8).float() * 2.0 - 1.0).contiguous()


def _block_values(value: np.ndarray, block: int) -> tuple[float, ...]:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size % block:
        raise ValueError("proposal statistic is not block aligned")
    return tuple(float(max(np.mean(value[index:index + block]), 1e-30)) for index in range(0, len(value), block))


def _proposal(
    *, layer: int, policy: str, family: str, source: Mapping[int, ProjectionTensors],
    fits: Mapping[int, tuple[FittedExpertStatistics, FittedExpertStatistics]], seed: str,
) -> tuple[list[MatrixInput], dict[int, ProjectionTensors], dict[int, tuple[FittedExpertStatistics, FittedExpertStatistics]]]:
    matrices = []
    proposal_source = {}
    proposal_fits = {}
    block = 128
    hidden = int(next(iter(source.values())).gate_proj.shape[1])
    intermediate = int(next(iter(source.values())).gate_proj.shape[0])
    shared_gate_sign = _sign(hidden, seed, layer, policy, family, "gate-up-suh")
    shared_down_sign = _sign(hidden, seed, layer, policy, family, "down-svh")
    gate_mass = 0.0
    shared_gate_diagonal = np.zeros(hidden, dtype=np.float64)
    down_output_mass = 0.0
    shared_down_output_energy = np.zeros(hidden, dtype=np.float64)
    gate_diagonals: dict[int, np.ndarray] = {}
    down_diagonals: dict[int, np.ndarray] = {}
    for expert in sorted(source):
        gate_fit, down_fit = fits[expert]
        gate_weight = float(
            gate_fit.metadata["accounting"]["combined"]["powers"]["2"]["weight_sum"]
        )
        down_weight = float(
            down_fit.metadata["accounting"]["combined"]["powers"]["2"]["weight_sum"]
        )
        gate_diagonals[expert] = np.diag(
            gate_fit.dense_hessian("combined", 2, regularized=False)
        ).copy()
        down_diagonals[expert] = np.diag(
            down_fit.dense_hessian("combined", 2, regularized=False)
        ).copy()
        shared_gate_diagonal += gate_weight * gate_diagonals[expert]
        shared_down_output_energy += down_weight * (
            source[expert].down_proj.float().pow(2).mean(dim=1).numpy()
        )
        gate_mass += gate_weight
        down_output_mass += down_weight
    if gate_mass <= 0.0 or down_output_mass <= 0.0:
        raise ValueError("proposal layer has non-positive shared transform mass")
    shared_gate_k_scales = scale_family_candidates(
        _block_values(shared_gate_diagonal / gate_mass, block)
    )[family]
    shared_down_n_scales = scale_family_candidates(
        _block_values(shared_down_output_energy / down_output_mass, block)
    )[family]
    for expert in sorted(source):
        gate_fit, down_fit = fits[expert]
        diagonal = down_diagonals[expert]
        permutation = policy_permutations(diagonal, block=block)[policy]
        gate, up, down = permute_expert_hf(
            source[expert].gate_proj, source[expert].up_proj, source[expert].down_proj, permutation
        )
        transformed_down_fit = _permuted_statistics(down_fit, permutation)
        proposal_source[expert] = ProjectionTensors(gate, up, down)
        proposal_fits[expert] = (gate_fit, transformed_down_fit)
        gate_h = gate_diagonals[expert]
        down_h = down_diagonals[expert][np.asarray(permutation, dtype=np.int64)]
        for projection, weight, hdiag, suh, svh in (
            ("gate_proj", gate, gate_h, shared_gate_sign, _sign(intermediate, seed, layer, expert, policy, family, "gate-svh")),
            ("up_proj", up, gate_h, shared_gate_sign, _sign(intermediate, seed, layer, expert, policy, family, "up-svh")),
            ("down_proj", down, down_h, _sign(intermediate, seed, layer, expert, policy, family, "down-suh"), shared_down_sign),
        ):
            k_scales = (
                shared_gate_k_scales
                if projection != "down_proj"
                else scale_family_candidates(_block_values(hdiag, block))[family]
            )
            n_scales = (
                shared_down_n_scales
                if projection == "down_proj"
                else scale_family_candidates(
                    _block_values(weight.float().pow(2).mean(dim=1).numpy(), block)
                )[family]
            )
            matrices.append(MatrixInput(
                key=f"E{expert}.{projection}", projection=projection, bits=4,
                weight_kn=weight.T.contiguous(), suh_sign=suh, svh_sign=svh,
                k_block_scales=k_scales,
                n_block_scales=n_scales,
                mass=float((gate_fit if projection != "down_proj" else transformed_down_fit).metadata["accounting"]["combined"]["powers"]["2"]["weight_sum"]),
            ))
    return matrices, proposal_source, proposal_fits


def _make_units(
    *, layer: int, source: Mapping[int, ProjectionTensors], fits: Mapping[int, tuple[FittedExpertStatistics, FittedExpertStatistics]],
    artifact: AbsoluteV31Artifact, heldout: Mapping[int, Sequence[RoutedExpertBatch]],
    conditional: Mapping[int, Sequence[ConditionalDownFitBatch]], codec: Any, config: Mapping[str, Any],
):
    result = []
    reject = {triplet: K5Decision(False, "K5 retained only after an explicit confirmation-tail rescue gate") for triplet in all_k5_triplets()}
    for expert in sorted(source):
        gate_fit, down_fit = fits[expert]
        result.append(build_expert_candidate_input(
            layer=layer, expert=expert, source=source[expert], gate_up_statistics=gate_fit,
            down_statistics=down_fit, heldout_batches=heldout[expert], k5_screen=reject,
            route_power=int(config.get("route_weight_power", 2)), accounting="combined",
            transform_seed_sha256=str(config["transform_seed_sha256"]),
            codebook_scale=float(codec._codec().codebook_scale), searched_transform=artifact,
            conditional_down_fit_batches=conditional[expert],
        ))
    return result


def _score_units(units: Sequence[Any], codec: Any) -> tuple[float, float, dict[str, dict[str, Any]]]:
    import torch
    import torch.nn.functional as functional

    proxy = 0.0
    heldout = 0.0
    candidate_identities: dict[str, dict[str, Any]] = {}
    for item in units:
        generated = _codec_candidates(
            codec,
            item,
            [(4, 4, 4)],
            {"proposal_score": True, "layer": item.layer, "expert": item.expert},
        )
        candidates = {
            "gate_proj": generated["gate_proj"][4],
            "up_proj": generated["up_proj"][4],
            "down_proj": generated["down_proj"][(4, 4)][4],
        }
        encoded = {name: candidate.reconstructed for name, candidate in candidates.items()}
        conditional_down = _conditional_down_hessian(
            item, encoded["gate_proj"], encoded["up_proj"]
        )
        identity = {}
        for score_name, projection in (
            ("gate_proj", "gate_proj"),
            ("up_proj", "up_proj"),
            ("down_proj.g4u4", "down_proj"),
        ):
            candidate = candidates[projection]
            identity[score_name] = {
                "bits": 4,
                "packed_sha256": candidate.packed_sha256,
                "reconstruction_deployed_fp16_sha256": candidate.reconstruction_sha256,
                "reconstruction_hf_sha256": _tensor_record(candidate.reconstructed)["sha256"],
            }
        candidate_identities[item.unit_id] = identity
        for projection in ("gate_proj", "up_proj", "down_proj"):
            fit = item.fitted[projection]
            candidate = candidates[projection]
            residual = getattr(item.source, projection).float() - candidate.reconstructed.float()
            covariance = conditional_down if projection == "down_proj" else fit.covariance
            proxy += float(torch.einsum("nk,kl,nl->", residual, torch.as_tensor(covariance).float(), residual).item())
        for batch in item.heldout_batches:
            x = torch.as_tensor(batch.hidden_states).to(torch.bfloat16)
            gate0 = functional.linear(x, item.source.gate_proj.to(torch.bfloat16))
            up0 = functional.linear(x, item.source.up_proj.to(torch.bfloat16))
            y0 = functional.linear(functional.silu(gate0) * up0, item.source.down_proj.to(torch.bfloat16))
            gate1 = functional.linear(x, encoded["gate_proj"].to(torch.bfloat16))
            up1 = functional.linear(x, encoded["up_proj"].to(torch.bfloat16))
            y1 = functional.linear(functional.silu(gate1) * up1, encoded["down_proj"].to(torch.bfloat16))
            weights = torch.as_tensor(batch.route_weights).float().unsqueeze(1)
            heldout += float(((y0.float() - y1.float()).pow(2) * weights.pow(2)).sum().item())
    return proxy, heldout, candidate_identities


def materialize_candidate_work_units(
    context: Mapping[str, Any], config: Mapping[str, Any], codec: Any, fit_path: Path
) -> dict[str, Any]:
    """Return canonical work units and complete run identities for one stage."""

    layer_values = [int(context["layer"])] if context.get("layer") is not None else sorted({int(row["layer"]) for row in json.loads(fit_path.read_text())["statistics"]})
    if len(layer_values) > 1:
        parts = []
        for layer_value in layer_values:
            child_context = dict(context)
            child_context["layer"] = layer_value
            parts.append(materialize_candidate_work_units(child_context, config, codec, fit_path))
        experts = [item for part in parts for item in part["experts"]]
        checkpoint_sources = {
            identity: value
            for part in parts
            for identity, value in part["checkpoint_sources"].items()
        }
        identities = [part["run_identity"] for part in parts]
        invariant = ("model_revision", "dataset_revision", "heldout_artifact_sha256", "capture_artifact_sha256", "conditional_down_fit_artifact_sha256", "fisher_probe_sha256", "fisher_window_sha256")
        for key in invariant:
            if len({row[key] for row in identities}) != 1:
                raise ValueError(f"layer proposal searches disagree on {key}")
        aggregate = {
            "schema": "quant-pipeline.qwen-v31-proposal-search-index.v1",
            "layers": layer_values,
            "search_artifact_sha256": [row["search_artifact_sha256"] for row in identities],
        }
        aggregate["index_sha256"] = _hash_json(aggregate)
        write_json(Path(context["output_dir"]) / "proposal-search-index.json", aggregate)
        run_identity = {key: identities[0][key] for key in invariant}
        run_identity["search_artifact_sha256"] = aggregate["index_sha256"]
        return {"experts": experts, "checkpoint_sources": checkpoint_sources, "run_identity": run_identity}
    if not layer_values:
        raise ValueError("fit artifact has no candidate layers")
    layer = layer_values[0]
    heldout_root, heldout_manifest, capture_service_sha = _load_capture_component(context, "heldout")
    conditional_root, conditional_manifest, _ = _load_capture_component(context, "conditional_down")
    fits = _fit_rows(fit_path, layer)
    weights = _CheckpointWeights(context["inputs"]["source_checkpoint"])
    source = {expert: weights.expert(layer, expert) for expert in sorted(fits)}
    heldout = _all_expert_batches(
        heldout_root, heldout_manifest, layer, tuple(source), conditional=False,
    )
    conditional = _all_expert_batches(
        conditional_root, conditional_manifest, layer, tuple(source), conditional=True,
    )
    producer = CorrectedPinnedGSSProducer(codec)
    seed = str(config["transform_seed_sha256"])
    proposal_rows = []
    best = None
    output = Path(context["output_dir"])
    for policy in PERMUTATION_POLICIES:
        for family in SCALE_FAMILIES:
            matrices, proposal_source, proposal_fits = _proposal(
                layer=layer, policy=policy, family=family, source=source, fits=fits, seed=seed,
            )
            selected = {item.key: 4 for item in matrices}
            decisions = {key: _hash_json({"policy": policy, "family": family, "key": key, "bits": 4}) for key in selected}
            kwargs = {
                "core": codec._codec().core, "matrices": matrices, "producer": producer,
                "selected_bits": selected, "selection_decision_sha256": decisions,
                "layer_id": layer, "predecessor_checkpoint_hash": context["predecessor_state_hash"],
                "source_identities": {"source_checkpoint": context["input_identities"]["source_checkpoint"]},
                "core_identities": {
                    "numeric_core": codec.identity["numeric_core_sha256"],
                    "extension": codec.identity["extension_sha256"],
                    "codec": _hash_json(codec.identity),
                },
                "codebook_scale": float(codec._codec().codebook_scale),
            }
            cache = {}
            def score(artifact: AbsoluteV31Artifact):
                units = _make_units(
                    layer=layer, source=proposal_source, fits=proposal_fits, artifact=artifact,
                    heldout=heldout, conditional=conditional, codec=codec, config=config,
                )
                proxy, held, candidate_identities = _score_units(units, codec)
                cache["units"] = [
                    replace(
                        unit,
                        proposal_score_candidate_identity=candidate_identities[unit.unit_id],
                    )
                    for unit in units
                ]
                cache["proxy"] = proxy
                cache["heldout"] = held
                cache["candidate_identities"] = candidate_identities
                return proxy, held
            evaluator_sha = _hash_json({"implementation": "exact-r10-k4-proxy-and-bf16-full-expert", "codec": _hash_json(codec.identity)})
            evaluated = evaluate_additive_v31_candidate(
                exact_codec_evaluator=lambda artifact: make_candidate_evaluation(
                    artifact, method="exact_codec_proxy", score=score(artifact)[0], evaluator_sha256=evaluator_sha,
                ),
                heldout_evaluator=lambda artifact: make_candidate_evaluation(
                    artifact, method="heldout_full_expert_roundtrip", score=float(cache["heldout"]), evaluator_sha256=evaluator_sha,
                ),
                **kwargs,
            )
            proposal_id = f"{policy}__{family}"
            artifact_dir = output / "proposal-artifacts" / f"layer-{layer:03d}" / proposal_id
            save_absolute_v31_artifact(artifact_dir, evaluated.artifact)
            row = {
                "proposal_id": proposal_id, "permutation_policy": policy, "scale_family": family,
                "artifact": artifact_dir.relative_to(output).as_posix(),
                "artifact_sha256": evaluated.artifact.content_sha256,
                "exact_codec_proxy": evaluated.exact_codec_proxy.__dict__,
                "heldout_full_expert": evaluated.heldout_full_expert.__dict__,
                "proposal_score_candidate_identity_sha256": _hash_json(cache["candidate_identities"]),
            }
            row["proposal_sha256"] = _hash_json(row)
            proposal_rows.append(row)
            rank = (evaluated.heldout_full_expert.score, evaluated.exact_codec_proxy.score, proposal_id)
            if best is None or rank < best[0]:
                best = (rank, evaluated.artifact, cache["units"], row)
    assert best is not None
    search = {
        "schema": "quant-pipeline.qwen-v31-proposal-search.v1",
        "layer": layer,
        "control_domain": {"permutation_policies": list(PERMUTATION_POLICIES), "scale_families": list(SCALE_FAMILIES)},
        "proposal_count": len(proposal_rows),
        "proposals": proposal_rows,
        "winner": best[3]["proposal_id"],
        "winner_artifact_sha256": best[1].content_sha256,
    }
    search["search_sha256"] = _hash_json(search)
    write_json(output / f"proposal-search-layer-{layer:03d}.json", search)
    model_revision = str(config["model_revision"])
    dataset_revision = str(config["dataset_revision"])
    run_identity = {
        "model_revision": model_revision,
        "dataset_revision": dataset_revision,
        "heldout_artifact_sha256": heldout_manifest["capture_sha256"],
        "search_artifact_sha256": search["search_sha256"],
        "capture_artifact_sha256": heldout_manifest["capture_sha256"],
        "conditional_down_fit_artifact_sha256": conditional_manifest["capture_sha256"],
        "fisher_probe_sha256": _hash_json({"capture": heldout_manifest["capture_sha256"], "rank": heldout_manifest["fisher_rank"]}),
        "fisher_window_sha256": _hash_json({"capture": heldout_manifest["capture_sha256"], "window_set": "all"}),
    }
    return {
        "experts": best[2],
        "checkpoint_sources": {(layer, expert): value for expert, value in source.items()},
        "run_identity": run_identity,
    }
