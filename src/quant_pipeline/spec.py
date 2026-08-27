from __future__ import annotations

import dataclasses
import re
import tomllib
from pathlib import Path
from typing import Any

from .core.artifacts import canonical_json, sha256_bytes


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    family: str
    local_path: str = ""
    trust_remote_code: bool = False


@dataclasses.dataclass(frozen=True)
class CorpusSpec:
    input_jsonl: str
    tokenizer_id: str
    window_tokens: int = 2048
    seed: int = 20260823
    fit_windows: int = 32
    conditional_fit_windows: int = 16
    selection_windows: int = 16
    confirmation_windows: int = 16
    final_windows: int = 25
    minimum_domains: int = 4


@dataclasses.dataclass(frozen=True)
class CodecSpec:
    bits: tuple[int, ...] = (2, 3, 4, 5)
    group_size: int = 128
    symmetric: bool = True


@dataclasses.dataclass(frozen=True)
class ObjectiveSpec:
    path_nodes: int = 5
    fisher_rank: int = 32
    bootstrap_samples: int = 2000
    reanchor_every_layers: int = 4


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
    name: str
    output_dir: str
    model: ModelSpec
    corpus: CorpusSpec
    codec: CodecSpec = CodecSpec()
    objective: ObjectiveSpec = ObjectiveSpec()

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(self))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        with Path(path).open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        allowed = {"name", "output_dir", "model", "corpus", "codec", "objective"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
        codec = raw.get("codec", {})
        if "bits" in codec:
            codec["bits"] = tuple(int(x) for x in codec["bits"])
        spec = cls(
            name=raw["name"],
            output_dir=raw["output_dir"],
            model=ModelSpec(**raw["model"]),
            corpus=CorpusSpec(**raw["corpus"]),
            codec=CodecSpec(**codec),
            objective=ObjectiveSpec(**raw.get("objective", {})),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.name or not re.fullmatch(r"[0-9a-f]{40}", self.model.revision):
            raise ValueError("name and a 40-hex immutable model commit are required")
        roles = (
            self.corpus.fit_windows,
            self.corpus.conditional_fit_windows,
            self.corpus.selection_windows,
            self.corpus.confirmation_windows,
            self.corpus.final_windows,
        )
        if min(roles) < 1 or self.corpus.window_tokens < 16 or self.corpus.minimum_domains < 1:
            raise ValueError("all corpus roles and window_tokens must be positive")
        if sorted(set(self.codec.bits)) != list(self.codec.bits):
            raise ValueError("codec bits must be unique and sorted")
        if min(self.codec.bits) < 2 or max(self.codec.bits) > 8:
            raise ValueError("the reference codec supports 2..8 bits")
