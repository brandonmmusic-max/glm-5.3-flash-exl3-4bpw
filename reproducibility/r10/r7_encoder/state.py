"""Crash-consistent journal and rolling calibration-state seals."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .constants import HIDDEN_SIZE, RECIPE_MARKER, RECIPE_VERSION
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)
from .types import StateShard

SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class StageSeal:
    name: str
    input_sha256: Mapping[str, str]
    output_sha256: Mapping[str, str]
    metadata: Mapping[str, object]

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))


class Journal:
    def __init__(
        self, path: str | Path, *, recipe_config: Mapping[str, object]
    ) -> None:
        self.path = Path(path)
        self.recipe_config = dict(recipe_config)
        self.recipe_sha256 = sha256_bytes(canonical_json_bytes(self.recipe_config))
        self.seals: dict[str, StageSeal] = {}
        if self.path.exists():
            payload = read_json(self.path)
            if payload.get("marker") != RECIPE_MARKER:
                raise ValueError("foreign journal marker")
            if payload.get("recipe_version") != RECIPE_VERSION:
                raise ValueError("journal recipe version drift")
            if payload.get("recipe_sha256") != self.recipe_sha256:
                raise ValueError("resume recipe differs from sealed recipe")
            for raw in payload.get("seals", []):
                seal = StageSeal(
                    name=str(raw["name"]),
                    input_sha256=dict(raw["input_sha256"]),
                    output_sha256=dict(raw["output_sha256"]),
                    metadata=dict(raw["metadata"]),
                )
                if seal.name in self.seals:
                    raise ValueError(f"duplicate journal stage {seal.name}")
                self.seals[seal.name] = seal
        else:
            self.flush()

    def flush(self) -> None:
        atomic_write_json(
            self.path,
            {
                "marker": RECIPE_MARKER,
                "recipe_version": RECIPE_VERSION,
                "recipe_sha256": self.recipe_sha256,
                "recipe_config": self.recipe_config,
                "seals": [asdict(self.seals[name]) for name in sorted(self.seals)],
            },
        )

    def has(self, name: str) -> bool:
        return name in self.seals

    def require(self, name: str) -> StageSeal:
        try:
            return self.seals[name]
        except KeyError as exc:
            raise ValueError(
                f"required predecessor stage is not sealed: {name}"
            ) from exc

    def seal(self, seal: StageSeal) -> None:
        incumbent = self.seals.get(seal.name)
        if incumbent is not None:
            if incumbent != seal:
                raise ValueError(f"attempt to rewrite immutable stage {seal.name}")
            return
        self.seals[seal.name] = seal
        self.flush()

    def audit_outputs(self, root: str | Path) -> None:
        base = Path(root)
        for name in sorted(self.seals):
            seal = self.seals[name]
            artifacts = dict(seal.output_sha256)
            artifacts.update(
                {
                    relative: digest
                    for relative, digest in seal.input_sha256.items()
                    if "/" in relative or relative.endswith((".json", ".safetensors"))
                }
            )
            for relative, expected in artifacts.items():
                path = base / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise ValueError(f"journal artifact drift: {name}:{relative}")


class StateStore:
    """Own rolling state directories, never model/checkpoint directories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def state_dir(self, layer_input: int) -> Path:
        return self.root / f"input-layer-{layer_input:03d}"

    def manifest_path(self, layer_input: int) -> Path:
        return self.state_dir(layer_input) / "STATE.json"

    def seal_archive_path(self, layer_input: int) -> Path:
        return self.root / "seals" / f"input-layer-{layer_input:03d}.json"

    def load(self, layer_input: int) -> tuple[StateShard, ...]:
        path = self.manifest_path(layer_input)
        shards = self._load_manifest(
            path, layer_input=layer_input, require_payloads=True
        )
        archive = self.seal_archive_path(layer_input)
        if not archive.is_file() or sha256_file(archive) != sha256_file(path):
            raise ValueError(
                f"live/archive state seal mismatch for layer {layer_input}"
            )
        return shards

    def _load_manifest(
        self, path: Path, *, layer_input: int, require_payloads: bool
    ) -> tuple[StateShard, ...]:
        payload = read_json(path)
        if (
            payload.get("marker") != RECIPE_MARKER
            or payload.get("recipe_version") != RECIPE_VERSION
            or payload.get("layer_input") != layer_input
        ):
            raise ValueError(f"invalid state seal {path}")
        corpus_plan_sha256 = payload.get("corpus_plan_sha256")
        expected_shards = payload.get("expected_shards")
        if (
            not isinstance(corpus_plan_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", corpus_plan_sha256) is None
            or not isinstance(expected_shards, dict)
            or not expected_shards
            or any(
                SHARD_ID.fullmatch(str(shard_id)) is None
                or type(tokens) is not int
                or tokens <= 0
                for shard_id, tokens in expected_shards.items()
            )
        ):
            raise ValueError("state lacks a sealed corpus-plan domain")
        shards: list[StateShard] = []
        hidden_names: set[str] = set()
        metadata_names: set[str] = set()
        for raw in payload["shards"]:
            hidden_name = str(raw["hidden"])
            metadata_name = str(raw["metadata"])
            if (
                Path(hidden_name).name != hidden_name
                or Path(metadata_name).name != metadata_name
            ):
                raise ValueError("state shard path escapes its sealed directory")
            hidden = path.parent / hidden_name
            metadata = path.parent / metadata_name
            shard = StateShard(
                shard_id=str(raw["shard_id"]),
                hidden_path=hidden,
                metadata_path=metadata,
                tokens=int(raw["tokens"]),
                hidden_size=int(raw["hidden_size"]),
                sha256_hidden=str(raw["sha256_hidden"]),
                sha256_metadata=str(raw["sha256_metadata"]),
            )
            if shard.tokens <= 0 or shard.hidden_size != HIDDEN_SIZE:
                raise ValueError(f"invalid state geometry in {path}: {shard.shard_id}")
            if (
                SHARD_ID.fullmatch(shard.shard_id) is None
                or expected_shards.get(shard.shard_id) != shard.tokens
            ):
                raise ValueError("state shard is outside its sealed corpus domain")
            if hidden.name in hidden_names or metadata.name in metadata_names:
                raise ValueError(f"state shard filenames are not unique in {path}")
            hidden_names.add(hidden.name)
            metadata_names.add(metadata.name)
            if require_payloads and sha256_file(hidden) != shard.sha256_hidden:
                raise ValueError(f"state hidden hash mismatch: {hidden}")
            if require_payloads and sha256_file(metadata) != shard.sha256_metadata:
                raise ValueError(f"state metadata hash mismatch: {metadata}")
            if require_payloads:
                metadata_payload = read_json(metadata)
                if (
                    metadata_payload.get("corpus_plan_sha256") != corpus_plan_sha256
                    or metadata_payload.get("shard_id") != shard.shard_id
                    or int(metadata_payload.get("tokens", -1)) != shard.tokens
                ):
                    raise ValueError("state shard differs from sealed corpus plan")
            shards.append(shard)
        if [shard.shard_id for shard in shards] != sorted(
            shard.shard_id for shard in shards
        ):
            raise ValueError("state shards are not canonically ordered")
        if len({shard.shard_id for shard in shards}) != len(shards):
            raise ValueError("state shard IDs are not unique")
        if not shards or int(payload.get("tokens", -1)) != sum(
            shard.tokens for shard in shards
        ):
            raise ValueError("state token arithmetic mismatch")
        observed_domain = {shard.shard_id: shard.tokens for shard in shards}
        if observed_domain != {
            str(key): int(value) for key, value in expected_shards.items()
        }:
            raise ValueError("state shard domain differs from the sealed corpus plan")
        without_transaction = dict(payload)
        transaction = without_transaction.pop("transaction_id", None)
        if transaction != sha256_bytes(canonical_json_bytes(without_transaction)):
            raise ValueError("state transaction digest mismatch")
        return tuple(shards)

    def adopt_existing(
        self,
        layer_input: int,
        *,
        predecessor_sha256: str,
        backend_fingerprint: str,
    ) -> str | None:
        """Adopt publication that completed just before its journal seal."""

        destination_manifest = self.manifest_path(layer_input)
        if not destination_manifest.exists():
            return None
        self.load(layer_input)
        payload = read_json(destination_manifest)
        if payload.get("predecessor_sha256") != predecessor_sha256:
            raise ValueError("published state predecessor binding drift")
        if payload.get("backend_fingerprint") != backend_fingerprint:
            raise ValueError("published state backend binding drift")
        archive = self.seal_archive_path(layer_input)
        if not archive.is_file() or sha256_file(archive) != sha256_file(
            destination_manifest
        ):
            raise ValueError("published state lacks matching durable archive seal")
        return sha256_file(destination_manifest)

    def transition_domain(self, layer_input: int) -> tuple[str, dict[str, int]]:
        manifest = read_json(self.manifest_path(layer_input))
        self.load(layer_input)
        return str(manifest["corpus_plan_sha256"]), {
            str(key): int(value)
            for key, value in sorted(manifest["expected_shards"].items())
        }

    def begin_transition(
        self,
        layer_input: int,
        *,
        corpus_plan_sha256: str,
        expected_shards: Mapping[str, int],
    ) -> "StateTransition":
        destination = self.state_dir(layer_input)
        temporary = self.root / f".input-layer-{layer_input:03d}.partial"
        if destination.exists():
            raise FileExistsError(
                f"state {layer_input} is already sealed; adopt it first"
            )
        temporary.mkdir(parents=True, exist_ok=True)
        return StateTransition(
            self,
            layer_input,
            temporary,
            destination,
            corpus_plan_sha256=corpus_plan_sha256,
            expected_shards=expected_shards,
        )

    def retire(self, layer_input: int) -> None:
        """Remove a predecessor only after its successor has passed `load()`."""

        self.load(layer_input + 1)
        source = self.state_dir(layer_input)
        retired = self.root / f".retired-input-layer-{layer_input:03d}"
        if retired.exists():
            shutil.rmtree(retired)
        os.replace(source, retired)
        shutil.rmtree(retired)


class StateTransition:
    def __init__(
        self,
        store: StateStore,
        layer_input: int,
        temporary: Path,
        destination: Path,
        *,
        corpus_plan_sha256: str,
        expected_shards: Mapping[str, int],
    ) -> None:
        self.store = store
        self.layer_input = layer_input
        self.temporary = temporary
        self.destination = destination
        self.corpus_plan_sha256 = str(corpus_plan_sha256)
        self.expected_shards = {
            str(key): int(value) for key, value in sorted(expected_shards.items())
        }
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.corpus_plan_sha256) is None
            or not self.expected_shards
            or any(tokens <= 0 for tokens in self.expected_shards.values())
        ):
            raise ValueError(
                "state transition requires a sealed nonempty corpus domain"
            )
        self._shards: list[dict[str, object]] = []
        self._committed = False
        self.partial_manifest = temporary / "PARTIAL.json"
        if self.partial_manifest.exists():
            payload = read_json(self.partial_manifest)
            if payload.get("layer_input") != layer_input:
                raise ValueError("partial state layer drift")
            if (
                payload.get("corpus_plan_sha256") != self.corpus_plan_sha256
                or payload.get("expected_shards") != self.expected_shards
            ):
                raise ValueError("partial state corpus-plan domain drift")
            self._shards = list(payload.get("shards", []))
            for raw in self._shards:
                shard_id = str(raw.get("shard_id", ""))
                hidden_name = str(raw.get("hidden", ""))
                metadata_name = str(raw.get("metadata", ""))
                tokens = int(raw.get("tokens", -1))
                if (
                    SHARD_ID.fullmatch(shard_id) is None
                    or self.expected_shards.get(shard_id) != tokens
                    or Path(hidden_name).name != hidden_name
                    or Path(metadata_name).name != metadata_name
                ):
                    raise ValueError(
                        "partial shard is outside the sealed corpus domain"
                    )
                hidden = temporary / hidden_name
                metadata = temporary / metadata_name
                if (
                    not hidden.is_file()
                    or not metadata.is_file()
                    or sha256_file(hidden) != raw["sha256_hidden"]
                    or sha256_file(metadata) != raw["sha256_metadata"]
                ):
                    raise ValueError("partial state shard hash mismatch")
                metadata_payload = read_json(metadata)
                if (
                    metadata_payload.get("corpus_plan_sha256")
                    != self.corpus_plan_sha256
                    or metadata_payload.get("shard_id") != shard_id
                    or int(metadata_payload.get("tokens", -1)) != tokens
                ):
                    raise ValueError("partial state metadata differs from corpus plan")
            if len({raw["shard_id"] for raw in self._shards}) != len(self._shards):
                raise ValueError("duplicate shard IDs in partial state")
        else:
            # Bind the complete expected domain before the runtime writes prompt 0.
            self._flush_partial()

    @property
    def completed_shard_ids(self) -> frozenset[str]:
        return frozenset(str(raw["shard_id"]) for raw in self._shards)

    def _flush_partial(self) -> None:
        atomic_write_json(
            self.partial_manifest,
            {
                "marker": RECIPE_MARKER,
                "recipe_version": RECIPE_VERSION,
                "layer_input": self.layer_input,
                "corpus_plan_sha256": self.corpus_plan_sha256,
                "expected_shards": self.expected_shards,
                "shards": sorted(self._shards, key=lambda raw: str(raw["shard_id"])),
            },
        )

    def add_existing_shard(
        self,
        *,
        shard_id: str,
        hidden_path: str | Path,
        metadata_path: str | Path,
        tokens: int,
        hidden_size: int,
    ) -> None:
        if self._committed:
            raise RuntimeError("transition is already committed")
        hidden = Path(hidden_path).resolve()
        metadata = Path(metadata_path).resolve()
        if hidden.parent != self.temporary or metadata.parent != self.temporary:
            raise ValueError(
                "backend must write transition shards inside partial directory"
            )
        if not hidden.is_file() or not metadata.is_file():
            raise FileNotFoundError("transition shard is incomplete")
        if tokens <= 0 or hidden_size != HIDDEN_SIZE:
            raise ValueError("transition shard has invalid token/hidden geometry")
        if SHARD_ID.fullmatch(str(shard_id)) is None or self.expected_shards.get(
            str(shard_id)
        ) != int(tokens):
            raise ValueError("transition shard is outside the sealed corpus domain")
        metadata_payload = read_json(metadata)
        if (
            metadata_payload.get("corpus_plan_sha256") != self.corpus_plan_sha256
            or metadata_payload.get("shard_id") != str(shard_id)
            or int(metadata_payload.get("tokens", -1)) != int(tokens)
        ):
            raise ValueError("transition metadata differs from sealed corpus domain")
        record = {
            "shard_id": str(shard_id),
            "hidden": hidden.name,
            "metadata": metadata.name,
            "tokens": int(tokens),
            "hidden_size": int(hidden_size),
            "sha256_hidden": sha256_file(hidden),
            "sha256_metadata": sha256_file(metadata),
        }
        incumbent = next(
            (raw for raw in self._shards if raw["shard_id"] == str(shard_id)), None
        )
        if incumbent is not None:
            if incumbent != record:
                raise ValueError(f"partial shard rewrite drift: {shard_id}")
            return
        if any(
            raw["hidden"] == hidden.name or raw["metadata"] == metadata.name
            for raw in self._shards
        ):
            raise ValueError("transition shard filename collision")
        self._shards.append(record)
        self._flush_partial()

    def commit(self, *, predecessor_sha256: str, backend_fingerprint: str) -> str:
        if self._committed:
            raise RuntimeError("transition is already committed")
        self._shards.sort(key=lambda raw: str(raw["shard_id"]))
        observed_domain = {
            str(raw["shard_id"]): int(raw["tokens"]) for raw in self._shards
        }
        if observed_domain != self.expected_shards:
            raise ValueError("transition has not completed the sealed corpus domain")
        if self.layer_input > 3:
            first = min(self._shards, key=lambda raw: str(raw["shard_id"]))
            first_metadata = read_json(self.temporary / str(first["metadata"]))
            repeat = first_metadata.get("official_repeat_audit")
            if (
                not isinstance(repeat, dict)
                or set(repeat)
                != {
                    "schema",
                    "hidden_bf16_sha256",
                    "prev_topk_i32_sha256",
                    "dispatch_audit_sha256",
                    "passed",
                }
                or repeat.get("schema") != "r7-official-successor-repeat-v1"
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", str(repeat.get(key, ""))) is None
                    for key in (
                        "hidden_bf16_sha256",
                        "prev_topk_i32_sha256",
                        "dispatch_audit_sha256",
                    )
                )
                or repeat.get("passed") is not True
            ):
                raise ValueError("successor state lacks the official repeat oracle")
        payload = {
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "layer_input": self.layer_input,
            "predecessor_sha256": predecessor_sha256,
            "backend_fingerprint": backend_fingerprint,
            "tokens": sum(int(raw["tokens"]) for raw in self._shards),
            "shards": self._shards,
            "corpus_plan_sha256": self.corpus_plan_sha256,
            "expected_shards": self.expected_shards,
        }
        payload["transaction_id"] = sha256_bytes(canonical_json_bytes(payload))
        manifest = self.temporary / "STATE.json"
        atomic_write_json(manifest, payload)
        digest = sha256_file(manifest)
        archive = self.store.seal_archive_path(self.layer_input)
        atomic_write_json(archive, payload)
        if sha256_file(archive) != digest:
            raise AssertionError("state seal archive differs from live manifest")
        self.partial_manifest.unlink(missing_ok=True)
        os.replace(self.temporary, self.destination)
        parent_fd = os.open(self.store.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self._committed = True
        # Re-read all hashes after directory publication.
        self.store.load(self.layer_input)
        return digest
