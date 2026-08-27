from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from ..core.artifacts import canonical_json, sha256_bytes, write_json


ROLES = ("fit", "conditional_fit", "selection", "confirmation", "final")
LEGACY_ROLES = ("fit", "selection", "confirmation", "final")


def read_documents(path: str | Path) -> list[dict]:
    documents: list[dict] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(key in row for key in ("id", "domain", "text")):
                raise ValueError(f"line {line_number}: expected id, domain, and text")
            document_id = str(row["id"])
            if document_id in seen:
                raise ValueError(f"duplicate document id {document_id!r}")
            seen.add(document_id)
            documents.append({"id": document_id, "domain": str(row["domain"]), "text": str(row["text"])})
    return documents


def document_split(documents: Iterable[dict], seed: int) -> dict[str, list[dict]]:
    """Assign whole documents to roles, stratified by domain and deterministic."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        by_domain[document["domain"]].append(document)
    result = {role: [] for role in ROLES}
    for domain, rows in sorted(by_domain.items()):
        keyed = []
        for row in rows:
            key = hashlib.sha256(f"{seed}\0{domain}\0{row['id']}".encode()).digest()
            keyed.append((key, row))
        keyed.sort(key=lambda item: item[0])
        for index, (_, row) in enumerate(keyed):
            result[ROLES[index % len(ROLES)]].append(row)
    return result


def build_windows(
    documents: Iterable[dict],
    encode: Callable[[str], list[int]],
    window_tokens: int,
    limit: int,
    seed: int,
) -> list[dict]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        token_ids = list(map(int, encode(document["text"])))
        for offset in range(0, max(0, len(token_ids) - window_tokens + 1), window_tokens):
            window = token_ids[offset : offset + window_tokens]
            if len(window) != window_tokens:
                continue
            record = {
                "document_id": document["id"],
                "domain": document["domain"],
                "offset": offset,
                "token_ids": window,
            }
            record["token_sha256"] = sha256_bytes(canonical_json(window))
            by_domain[document["domain"]].append(record)
    generator = random.Random(seed)
    for candidates in by_domain.values():
        generator.shuffle(candidates)
    selected: list[dict] = []
    domains = sorted(by_domain)
    while len(selected) < limit:
        advanced = False
        for domain in domains:
            if by_domain[domain]:
                selected.append(by_domain[domain].pop())
                advanced = True
                if len(selected) == limit:
                    break
        if not advanced:
            break
    return selected


def seal_corpus(
    input_jsonl: str | Path,
    output_json: str | Path,
    encode: Callable[[str], list[int]],
    window_tokens: int,
    role_limits: dict[str, int],
    seed: int,
    tokenizer_identity: dict,
    minimum_domains: int = 4,
) -> dict:
    if set(role_limits) != set(ROLES):
        raise ValueError(f"role limits must be exactly {ROLES}")
    documents = read_documents(input_jsonl)
    split = document_split(documents, seed)
    windows = {
        role: build_windows(split[role], encode, window_tokens, role_limits[role], seed + index)
        for index, role in enumerate(ROLES)
    }
    for role in ROLES:
        if len(windows[role]) != role_limits[role]:
            raise ValueError(f"insufficient full windows for {role}: {len(windows[role])}/{role_limits[role]}")
        domains = {window["domain"] for window in windows[role]}
        if len(domains) < minimum_domains:
            raise ValueError(f"insufficient domain coverage for {role}: {len(domains)}/{minimum_domains}")
    ids = [{row["document_id"] for row in windows[role]} for role in ROLES]
    if any(
        ids[i] & ids[j]
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ):
        raise AssertionError("document leakage across corpus roles")
    artifact = {
        "schema": "quant-pipeline.sealed-corpus.v2",
        "seed": seed,
        "window_tokens": window_tokens,
        "minimum_domains": minimum_domains,
        "tokenizer": tokenizer_identity,
        "source": {"path": str(Path(input_jsonl).resolve()), "sha256": hashlib.sha256(Path(input_jsonl).read_bytes()).hexdigest()},
        "role_counts": {role: len(windows[role]) for role in ROLES},
        "windows": windows,
    }
    artifact["seal_sha256"] = sha256_bytes(canonical_json(artifact))
    write_json(output_json, artifact)
    return artifact


def verify_sealed_corpus(value: dict) -> None:
    expected_seal = value.get("seal_sha256")
    body = {key: item for key, item in value.items() if key != "seal_sha256"}
    if not expected_seal or sha256_bytes(canonical_json(body)) != expected_seal:
        raise ValueError("sealed corpus body hash mismatch")
    schema = value.get("schema")
    roles = ROLES if schema == "quant-pipeline.sealed-corpus.v2" else LEGACY_ROLES
    if schema not in {"quant-pipeline.sealed-corpus.v1", "quant-pipeline.sealed-corpus.v2"}:
        raise ValueError("unsupported sealed corpus schema")
    if set(value.get("windows", {})) != set(roles):
        raise ValueError("sealed corpus roles are incomplete")
    minimum_domains = int(value.get("minimum_domains", 1))
    role_counts = value.get("role_counts", {})
    document_sets = []
    for role in roles:
        windows = value["windows"][role]
        if int(role_counts.get(role, -1)) != len(windows) or not windows:
            raise ValueError(f"sealed corpus count mismatch for {role}")
        domains = {window["domain"] for window in windows}
        if len(domains) < minimum_domains:
            raise ValueError(f"sealed corpus domain coverage mismatch for {role}")
        documents = set()
        for window in windows:
            if sha256_bytes(canonical_json(list(map(int, window["token_ids"])))) != window.get("token_sha256"):
                raise ValueError(f"sealed corpus token hash mismatch for {role}")
            documents.add(window["document_id"])
        document_sets.append(documents)
    if any(
        document_sets[i] & document_sets[j]
        for i in range(len(document_sets))
        for j in range(i + 1, len(document_sets))
    ):
        raise ValueError("sealed corpus document leakage across roles")
