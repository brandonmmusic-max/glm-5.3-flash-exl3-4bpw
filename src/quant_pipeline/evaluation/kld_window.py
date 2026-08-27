from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

from ..core.artifacts import canonical_json, prepare_empty_destination, require_execute, sha256_bytes, sha256_file, write_json


SCHEMA = "quant-pipeline.kld-window.v1"


def _require_revision(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be an immutable 40-hex revision")


def _tokenizer_files(model_path: Path) -> dict[str, dict[str, int | str]]:
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    }
    files = sorted(path for path in model_path.iterdir() if path.is_file() and path.name in names)
    if not files:
        raise FileNotFoundError(f"no tokenizer identity files found in {model_path}")
    return {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files}


def seal_kld_window_from_texts(
    texts: Iterable[str],
    output_dir: str | Path,
    encode: Callable[[str, int], list[int]],
    tokenizer_identity: dict,
    dataset_identity: dict,
    context_length: int = 2048,
) -> dict:
    """Seal the historical GLM WikiText construction using the target tokenizer."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    rows = [str(text) for text in texts]
    nonempty = [text for text in rows if text.strip()]
    if not nonempty:
        raise ValueError("WikiText split has no nonempty rows")
    joined = "\n\n".join(nonempty)
    prefix = joined[: context_length * 5]
    token_ids = list(map(int, encode(prefix, context_length)))
    if len(token_ids) != context_length:
        raise ValueError(f"target tokenizer produced {len(token_ids)} tokens; expected exactly {context_length}")

    destination = prepare_empty_destination(output_dir)
    prefix_path = destination / "source-prefix.txt"
    prefix_path.write_text(prefix, encoding="utf-8")
    artifact = {
        "schema": SCHEMA,
        "method": "glm-wikitext-2-raw-test-prefix-v1",
        "dataset": dataset_identity,
        "construction": {
            "filter": "text.strip() is nonempty",
            "join_separator": "\\n\\n",
            "prefix_characters_requested": context_length * 5,
            "source_rows": len(rows),
            "nonempty_rows": len(nonempty),
            "joined_characters": len(joined),
            "prefix_characters": len(prefix),
            "tokenizer_add_special_tokens": False,
            "tokenizer_truncation": True,
            "tokenizer_max_length": context_length,
        },
        "source_prefix": {
            "file": prefix_path.name,
            "bytes": prefix_path.stat().st_size,
            "sha256": sha256_file(prefix_path),
        },
        "tokenizer": tokenizer_identity,
        "context_length": context_length,
        "prediction_positions": context_length - 1,
        "token_ids": token_ids,
        "token_sha256": sha256_bytes(canonical_json(token_ids)),
        "first_16_token_ids": token_ids[:16],
    }
    artifact["seal_sha256"] = sha256_bytes(canonical_json(artifact))
    write_json(destination / "kld-window.json", artifact)
    return artifact


def seal_kld_window(
    model_path: str | Path,
    model_revision: str,
    dataset_revision: str,
    output_dir: str | Path,
    execute: bool,
    context_length: int = 2048,
) -> dict:
    require_execute(execute, "load the pinned WikiText split and seal the KLD window")
    _require_revision(model_revision, "model_revision")
    _require_revision(dataset_revision, "dataset_revision")
    local_model = Path(model_path).resolve()
    if not local_model.is_dir():
        raise ValueError("KLD sealing requires a local immutable model/tokenizer directory")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except Exception as error:  # pragma: no cover
        raise RuntimeError("install quant-pipeline[hf] to seal a KLD window") from error

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)

    def encode(text: str, maximum: int) -> list[int]:
        return tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=maximum,
        )

    return seal_kld_window_from_texts(
        dataset["text"],
        output_dir,
        encode,
        {
            "model_path": str(local_model),
            "expected_model_revision": model_revision,
            "class": type(tokenizer).__name__,
            "files": _tokenizer_files(local_model),
        },
        {
            "repo": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "revision": dataset_revision,
            "fingerprint": getattr(dataset, "_fingerprint", None),
        },
        context_length,
    )


def verify_kld_window(value: dict, artifact_dir: str | Path | None = None) -> None:
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported KLD window schema")
    expected_seal = value.get("seal_sha256")
    body = {key: item for key, item in value.items() if key != "seal_sha256"}
    if not expected_seal or sha256_bytes(canonical_json(body)) != expected_seal:
        raise ValueError("KLD window body hash mismatch")
    context_length = int(value.get("context_length", 0))
    token_ids = list(map(int, value.get("token_ids", [])))
    if len(token_ids) != context_length or context_length < 2:
        raise ValueError("KLD window token count mismatch")
    if sha256_bytes(canonical_json(token_ids)) != value.get("token_sha256"):
        raise ValueError("KLD window token hash mismatch")
    if int(value.get("prediction_positions", -1)) != context_length - 1:
        raise ValueError("KLD prediction-position count mismatch")
    if artifact_dir is not None:
        source = value.get("source_prefix", {})
        path = Path(artifact_dir) / str(source.get("file", ""))
        if not path.is_file() or path.stat().st_size != int(source.get("bytes", -1)) or sha256_file(path) != source.get("sha256"):
            raise ValueError("KLD source-prefix identity mismatch")

