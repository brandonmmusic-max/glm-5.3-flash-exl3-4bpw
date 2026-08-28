"""Flat per-layer routed-expert capture: one memmap trio instead of file storms.

The prior capture path wrote one safetensors payload plus one JSON sidecar per
prompt (~1,773 file pairs per layer, each individually sha256'd) and the search
phase then re-read all of them.  This module replaces that with the owner's
proven `LayerCalibRAM` layout from ``encode_b300.py``: three growable flat files
per layer, hashed once, mmapped read-only afterwards.

    x.bin        bf16 hidden states stored as int16   shape (tokens, 6144)
    ids.bin      top-8 routed expert ids, uint8       shape (tokens, 8)
    weights.bin  top-8 router weights, float32        shape (tokens, 8)

``weights.bin`` is a deliberate addition to the owner's ids-only layout: this
project's bit allocator consumes the exact float32 router weights, so they are
captured rather than recomputed.

Importing this module is inert - it touches no model, no CUDA, and no tensor
library.  numpy and torch are imported inside the functions that need them,
matching the rest of ``r7_encoder``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from .constants import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    RECIPE_MARKER,
    RECIPE_VERSION,
    TOP_K,
)
from .determinism import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
)

SCHEMA = "r7-flat-capture-v2"
LEGACY_SCHEMA = "r7-flat-capture-v1"
X_FILE = "x.bin"
IDS_FILE = "ids.bin"
WEIGHTS_FILE = "weights.bin"
MANIFEST_FILE = "layer_manifest.json"
DEFAULT_CHUNK = 65536

# storage dtype of each payload, and how many columns it carries
_X_STORAGE_DTYPE = "int16"
_IDS_DTYPE = "uint8"
_WEIGHTS_DTYPE = "float32"
_ITEMSIZE = {"int16": 2, "uint8": 1, "float32": 4}


def layer_capture_dir(capture_dir: str | Path, layer: int | None = None) -> Path:
    """Resolve the directory that holds one layer's flat capture trio."""

    base = Path(capture_dir)
    if layer is None:
        return base
    value = int(layer)
    if not 0 <= value < 1000:
        raise ValueError(f"layer {value} outside the capturable range [0,1000)")
    return base / f"layer_{value:03d}"


def _safe_child(directory: Path, name: str) -> Path:
    """Resolve a manifest-declared file name without escaping its directory."""

    if not name or Path(name).name != name:
        raise ValueError(f"capture payload name escapes its directory: {name!r}")
    return directory / name


class FlatCaptureWriter:
    """Append-only writer for one layer's flat capture trio.

    Payload digests are computed incrementally while writing, so ``finalize``
    never re-reads the (potentially multi-GB) payloads.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        layer: int,
        hidden_size: int = HIDDEN_SIZE,
        top_k: int = TOP_K,
        num_experts: int = NUM_EXPERTS,
        bindings: Mapping[str, object],
        verification: Mapping[str, object],
        overwrite: bool = False,
        buffer_bytes: int = 1 << 22,
    ) -> None:
        if int(hidden_size) <= 0 or int(top_k) <= 0 or int(num_experts) <= 0:
            raise ValueError("flat capture geometry must be positive")
        if int(num_experts) > 256:
            raise ValueError("uint8 expert ids cannot address more than 256 experts")
        self.layer = int(layer)
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.num_experts = int(num_experts)
        self.bindings = dict(bindings)
        self.verification = dict(verification)
        if not self.bindings or any(
            not isinstance(key, str) or not key for key in self.bindings
        ):
            raise ValueError("flat capture requires nonempty string-keyed bindings")
        if not self.verification or any(
            not isinstance(key, str) or not key for key in self.verification
        ):
            raise ValueError(
                "flat capture requires nonempty string-keyed verification provenance"
            )
        # Fail before opening/truncating any payload if the supplied provenance
        # is not canonically JSON serializable.
        canonical_json_bytes(self.bindings)
        canonical_json_bytes(self.verification)
        self.layer_dir = layer_capture_dir(capture_dir, self.layer)
        self.layer_dir.mkdir(parents=True, exist_ok=True)
        self.tokens = 0
        self.manifest_path: Path | None = None
        self._appends = 0
        self._closed = False
        self._finalized = False

        self._paths = {
            X_FILE: self.layer_dir / X_FILE,
            IDS_FILE: self.layer_dir / IDS_FILE,
            WEIGHTS_FILE: self.layer_dir / WEIGHTS_FILE,
        }
        existing = [str(path) for path in self._paths.values() if path.exists()]
        manifest_default = self.layer_dir / MANIFEST_FILE
        if manifest_default.exists():
            existing.append(str(manifest_default))
        if existing and not overwrite:
            raise FileExistsError(
                "flat capture requires a clean layer directory; found "
                + ", ".join(sorted(existing))
            )
        for path in self._paths.values():
            path.unlink(missing_ok=True)
        manifest_default.unlink(missing_ok=True)

        self._handles = {
            name: open(path, "wb", buffering=int(buffer_bytes))
            for name, path in self._paths.items()
        }
        self._digests = {name: hashlib.sha256() for name in self._paths}
        self._bytes = {name: 0 for name in self._paths}
        self._counts: Any = None  # np.ndarray, allocated on first append

    # -- writing ---------------------------------------------------------

    def _emit(self, name: str, array: Any) -> None:
        import numpy as np

        payload = np.ascontiguousarray(array)
        view = memoryview(payload).cast("B")
        self._handles[name].write(view)
        self._digests[name].update(view)
        self._bytes[name] += payload.nbytes

    def append(self, hidden_bf16, topk_ids, topk_weights) -> int:
        """Append ``n`` token rows; returns the number of rows written."""

        import numpy as np
        import torch

        if self._closed:
            raise ValueError("flat capture writer is already closed")

        hidden = torch.as_tensor(hidden_bf16).detach()
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                f"hidden states must be bfloat16, got {hidden.dtype}"
            )
        if hidden.ndim != 2 or int(hidden.shape[1]) != self.hidden_size:
            raise ValueError(
                f"hidden states must be [tokens,{self.hidden_size}], got "
                f"{tuple(hidden.shape)}"
            )
        rows = int(hidden.shape[0])

        ids = torch.as_tensor(topk_ids).detach()
        weights = torch.as_tensor(topk_weights).detach()
        if tuple(ids.shape) != (rows, self.top_k):
            raise ValueError(
                f"expert ids must be [{rows},{self.top_k}], got {tuple(ids.shape)}"
            )
        if tuple(weights.shape) != (rows, self.top_k):
            raise ValueError(
                f"router weights must be [{rows},{self.top_k}], got "
                f"{tuple(weights.shape)}"
            )
        if rows == 0:
            return 0

        hidden = hidden.to("cpu").contiguous()
        if not torch.isfinite(hidden).all():
            raise ValueError("captured hidden states contain non-finite values")

        ids_cpu = ids.to("cpu")
        if ids_cpu.dtype not in (
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"expert ids must be an integer dtype, got {ids_cpu.dtype}")
        ids_long = ids_cpu.to(torch.int64)
        if int(ids_long.min()) < 0 or int(ids_long.max()) >= self.num_experts:
            raise ValueError(f"routed expert id outside [0,{self.num_experts})")
        sorted_ids = ids_long.sort(dim=1).values
        if rows and not bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()):
            raise ValueError("duplicate routed expert within a token")

        weights_cpu = weights.to("cpu")
        if weights_cpu.dtype != torch.float32:
            raise ValueError(
                "router weights must be materialized as float32, got "
                f"{weights_cpu.dtype}"
            )
        if not torch.isfinite(weights_cpu).all():
            raise ValueError("captured router weights contain non-finite values")

        x_np = hidden.view(torch.int16).numpy()
        ids_np = ids_long.to(torch.uint8).contiguous().numpy()
        weights_np = weights_cpu.contiguous().numpy()

        self._emit(X_FILE, x_np)
        self._emit(IDS_FILE, ids_np)
        self._emit(WEIGHTS_FILE, weights_np)

        counts = np.bincount(ids_np.reshape(-1), minlength=self.num_experts)
        if self._counts is None:
            self._counts = counts.astype(np.int64)
        else:
            self._counts += counts.astype(np.int64)
        self.tokens += rows
        self._appends += 1
        return rows

    # -- sealing ---------------------------------------------------------

    def _close_handles(self) -> None:
        if self._closed:
            return
        for handle in self._handles.values():
            try:
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
        self._closed = True

    def _expected_bytes(self, name: str) -> int:
        if name == X_FILE:
            return self.tokens * self.hidden_size * _ITEMSIZE[_X_STORAGE_DTYPE]
        if name == IDS_FILE:
            return self.tokens * self.top_k * _ITEMSIZE[_IDS_DTYPE]
        return self.tokens * self.top_k * _ITEMSIZE[_WEIGHTS_DTYPE]

    def _shape(self, name: str) -> list[int]:
        columns = self.hidden_size if name == X_FILE else self.top_k
        return [self.tokens, columns]

    def finalize(
        self, manifest_path: str | Path | None = None, *, verify: bool = False
    ) -> dict[str, Any]:
        """Seal the payloads and write the JSON manifest; returns the manifest."""

        if self._finalized:
            raise ValueError("flat capture writer is already finalized")
        if self.tokens <= 0:
            raise ValueError("refusing to finalize an empty flat capture")
        self._close_handles()

        target = (
            self.layer_dir / MANIFEST_FILE
            if manifest_path is None
            else Path(manifest_path)
        )

        files: dict[str, Any] = {}
        for name in (X_FILE, IDS_FILE, WEIGHTS_FILE):
            path = self._paths[name]
            expected = self._expected_bytes(name)
            actual = path.stat().st_size
            if actual != expected or self._bytes[name] != expected:
                raise ValueError(
                    f"{name}: payload size mismatch (wrote {self._bytes[name]}, "
                    f"on disk {actual}, expected {expected})"
                )
            digest = self._digests[name].hexdigest()
            if verify and sha256_file(path) != digest:
                raise ValueError(f"{name}: payload digest mismatch after fsync")
            dtype = (
                _X_STORAGE_DTYPE
                if name == X_FILE
                else (_IDS_DTYPE if name == IDS_FILE else _WEIGHTS_DTYPE)
            )
            files[name] = {
                "name": name,
                "dtype": dtype,
                "shape": self._shape(name),
                "bytes": expected,
                "sha256": digest,
            }

        counts = [0] * self.num_experts
        if self._counts is not None:
            counts = [int(value) for value in self._counts.tolist()]
        if sum(counts) != self.tokens * self.top_k:
            raise ValueError("routed-count accumulator disagrees with token count")

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "marker": RECIPE_MARKER,
            "recipe_version": RECIPE_VERSION,
            "layer": self.layer,
            "bindings": self.bindings,
            "verification": self.verification,
            "tokens": self.tokens,
            "hidden": self.hidden_size,
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "appends": self._appends,
            "x_dtype": "bfloat16",
            "x_storage_dtype": _X_STORAGE_DTYPE,
            "ids_dtype": _IDS_DTYPE,
            "weights_dtype": _WEIGHTS_DTYPE,
            "files": files,
            # convenience aliases, mirroring the owner's capture manifest
            "sha256_x": files[X_FILE]["sha256"],
            "sha256_ids": files[IDS_FILE]["sha256"],
            "sha256_weights": files[WEIGHTS_FILE]["sha256"],
            "routed_counts": counts,
        }
        manifest["content_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        atomic_write_json(target, manifest)
        self.manifest_path = target
        self._finalized = True
        return manifest

    def abort(self) -> None:
        """Close handles and remove partial payloads (nothing is sealed)."""

        self._close_handles()
        if not self._finalized:
            for path in self._paths.values():
                path.unlink(missing_ok=True)

    def close(self) -> None:
        self._close_handles()

    def __enter__(self) -> "FlatCaptureWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self._close_handles()


class FlatCaptureReader:
    """Read-only mmap view over one finalized flat capture layer."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_layer: int | None = None,
        expected_bindings: Mapping[str, object] | None = None,
        expected_verification: Mapping[str, object] | None = None,
        verify_payloads: bool = True,
        verify_structure: bool = True,
    ) -> None:
        import numpy as np
        import torch

        path = Path(manifest_path)
        if path.is_dir():
            path = path / MANIFEST_FILE
        if not path.is_file():
            raise ValueError(f"flat capture manifest not found: {path}")
        self.manifest_path = path
        self.layer_dir = path.parent

        manifest = read_json(path)
        if not isinstance(manifest, dict):
            raise ValueError(f"malformed flat capture manifest: {path}")
        legacy_v1 = manifest.get("schema") == LEGACY_SCHEMA
        if manifest.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
            raise ValueError(
                f"flat capture schema drift: {manifest.get('schema')!r} != {SCHEMA!r}"
            )
        if manifest.get("marker") != RECIPE_MARKER:
            raise ValueError("flat capture recipe marker drift")
        content = manifest.pop("content_sha256", None)
        if content != sha256_bytes(canonical_json_bytes(manifest)):
            raise ValueError(f"flat capture manifest content digest mismatch: {path}")
        manifest["content_sha256"] = content
        self.manifest = manifest

        bindings = manifest.get("bindings")
        if legacy_v1 and bindings is None:
            if expected_bindings is not None:
                raise ValueError("legacy flat capture cannot satisfy expected bindings")
            bindings = {}
        verification = manifest.get("verification")
        if legacy_v1 and verification is None:
            if expected_verification is not None:
                raise ValueError(
                    "legacy flat capture cannot satisfy expected verification provenance"
                )
            verification = {}
        if not isinstance(bindings, dict) or (not bindings and not legacy_v1) or any(
            not isinstance(key, str) or not key for key in bindings
        ):
            raise ValueError("flat capture manifest lacks sealed input bindings")
        if expected_bindings is not None and bindings != dict(expected_bindings):
            raise ValueError("flat capture input bindings differ from the requested run")
        if not isinstance(verification, dict) or (not verification and not legacy_v1) or any(
            not isinstance(key, str) or not key for key in verification
        ):
            raise ValueError("flat capture manifest lacks verification provenance")
        if expected_verification is not None and verification != dict(
            expected_verification
        ):
            raise ValueError("flat capture verification provenance differs from run")
        self.bindings = dict(bindings)
        self.verification = dict(verification)

        self.layer = int(manifest["layer"])
        if expected_layer is not None and self.layer != int(expected_layer):
            raise ValueError(
                f"flat capture layer drift: manifest {self.layer} != "
                f"requested {int(expected_layer)}"
            )
        self.tokens = int(manifest["tokens"])
        self.hidden_size = int(manifest["hidden"])
        self.top_k = int(manifest["top_k"])
        self.num_experts = int(manifest["num_experts"])
        if self.tokens <= 0 or self.hidden_size <= 0 or self.top_k <= 0:
            raise ValueError("flat capture manifest declares an empty capture")
        if manifest.get("x_dtype") != "bfloat16":
            raise ValueError("flat capture hidden dtype must be bfloat16")
        if (
            manifest.get("x_storage_dtype") != _X_STORAGE_DTYPE
            or manifest.get("ids_dtype") != _IDS_DTYPE
            or manifest.get("weights_dtype") != _WEIGHTS_DTYPE
        ):
            raise ValueError("flat capture payload dtype drift")

        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {
            X_FILE,
            IDS_FILE,
            WEIGHTS_FILE,
        }:
            raise ValueError("flat capture manifest payload set drift")
        expected = {
            X_FILE: (_X_STORAGE_DTYPE, [self.tokens, self.hidden_size]),
            IDS_FILE: (_IDS_DTYPE, [self.tokens, self.top_k]),
            WEIGHTS_FILE: (_WEIGHTS_DTYPE, [self.tokens, self.top_k]),
        }
        aliases = {
            X_FILE: "sha256_x",
            IDS_FILE: "sha256_ids",
            WEIGHTS_FILE: "sha256_weights",
        }
        resolved: dict[str, Path] = {}
        for name, (dtype, shape) in expected.items():
            record = files[name]
            if not isinstance(record, dict):
                raise ValueError(f"{name}: malformed manifest record")
            if record.get("dtype") != dtype or list(record.get("shape", [])) != shape:
                raise ValueError(
                    f"{name}: dtype/shape drift (manifest {record.get('dtype')} "
                    f"{record.get('shape')}, expected {dtype} {shape})"
                )
            payload = _safe_child(self.layer_dir, str(record.get("name", name)))
            if not payload.is_file():
                raise ValueError(f"{name}: payload missing at {payload}")
            size = shape[0] * shape[1] * _ITEMSIZE[dtype]
            if int(record.get("bytes", -1)) != size:
                raise ValueError(f"{name}: manifest byte count disagrees with shape")
            actual = payload.stat().st_size
            if actual != size:
                raise ValueError(
                    f"{name}: payload size mismatch (on disk {actual}, expected {size})"
                )
            digest = str(record.get("sha256", ""))
            if manifest.get(aliases[name]) != digest:
                raise ValueError(f"{name}: manifest digest alias drift")
            if verify_payloads and sha256_file(payload) != digest:
                raise ValueError(f"{name}: payload SHA mismatch at {payload}")
            resolved[name] = payload

        # copy-on-write mmaps: reads never fault a copy, and torch accepts them
        self._x_np = np.memmap(
            resolved[X_FILE],
            dtype=np.int16,
            mode="c",
            shape=(self.tokens, self.hidden_size),
        )
        self.hidden = torch.from_numpy(self._x_np).view(torch.bfloat16)
        self._ids_np = np.memmap(
            resolved[IDS_FILE],
            dtype=np.uint8,
            mode="c",
            shape=(self.tokens, self.top_k),
        )
        self.ids = torch.from_numpy(self._ids_np)
        self._weights_np = np.memmap(
            resolved[WEIGHTS_FILE],
            dtype=np.float32,
            mode="c",
            shape=(self.tokens, self.top_k),
        )
        self.weights = torch.from_numpy(self._weights_np)

        declared = manifest.get("routed_counts")
        if not isinstance(declared, list) or len(declared) != self.num_experts:
            raise ValueError("flat capture routed-count manifest is malformed")
        counts = np.asarray([int(value) for value in declared], dtype=np.int64)
        if int(counts.sum()) != self.tokens * self.top_k:
            raise ValueError("flat capture routed-count total mismatch")
        if verify_structure:
            if (
                int(self._ids_np.min()) < 0
                or int(self._ids_np.max()) >= self.num_experts
            ):
                raise ValueError(f"routed expert id outside [0,{self.num_experts})")
            sorted_ids = np.sort(self._ids_np, axis=1)
            if not bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()):
                raise ValueError("duplicate routed expert within a token")
            observed = np.bincount(
                self._ids_np.reshape(-1), minlength=self.num_experts
            ).astype(np.int64)
            if observed.tolist() != counts.tolist():
                raise ValueError("flat capture routed-count manifest mismatch")
        # R10's encode path intentionally trusts Phase A's just-written manifest.
        # Avoiding the all-token sort/bincount does not affect any encoded byte;
        # the default remains the fully verified historical behavior.
        self.routed_counts = counts.tolist()
        self._starts: Any = None
        self._token_of: Any = None

    # -- row planning ----------------------------------------------------

    def _build_index(self) -> None:
        if self._starts is not None:
            return
        import numpy as np

        flat = self._ids_np.reshape(-1)
        order = np.argsort(flat, kind="stable")
        counts = np.asarray(self.routed_counts, dtype=np.int64)
        self._starts = np.concatenate([[0], np.cumsum(counts)])
        self._token_of = (order // self.top_k).astype(np.int64)

    def expert_rows(self, expert: int):
        """LongTensor of ascending row indices where ``expert`` was routed."""

        import torch

        value = int(expert)
        if not 0 <= value < self.num_experts:
            raise ValueError(f"expert {value} outside [0,{self.num_experts})")
        self._build_index()
        start = int(self._starts[value])
        end = int(self._starts[value + 1])
        return torch.from_numpy(self._token_of[start:end].copy())

    def all_rows(self):
        import torch

        return torch.arange(self.tokens, dtype=torch.int64)

    def gather_chunks(
        self, rows, device, chunk: int = DEFAULT_CHUNK, dtype=None
    ) -> Iterator[Any]:
        """Yield ``chunk``-row blocks of hidden states moved to ``device``.

        The rows keep their stored bfloat16 dtype unless ``dtype`` is given;
        pass ``torch.float32`` for covariance/Hessian accumulation.
        """

        import torch

        if int(chunk) <= 0:
            raise ValueError("gather chunk must be positive")
        selection = torch.as_tensor(rows)
        if selection.ndim != 1:
            raise ValueError("gather rows must be a 1-D index tensor")
        selection = selection.to("cpu").to(torch.int64)
        if selection.numel():
            if int(selection.min()) < 0 or int(selection.max()) >= self.tokens:
                raise ValueError("gather row index outside the capture")
        for start in range(0, selection.numel(), int(chunk)):
            index = selection[start : start + int(chunk)]
            block = torch.index_select(self.hidden, 0, index).to(device)
            yield block if dtype is None else block.to(dtype)

    def ids_long(self):
        """int64 copy of the routed ids, for consumers that index with them."""

        import torch

        return self.ids.to(torch.int64)

    def close(self) -> None:
        self.hidden = None
        self.ids = None
        self.weights = None
        for name in ("_x_np", "_ids_np", "_weights_np"):
            array = getattr(self, name, None)
            handle = getattr(array, "_mmap", None)
            if handle is not None:
                try:
                    handle.close()
                except (BufferError, ValueError):
                    pass
            setattr(self, name, None)
        self._starts = None
        self._token_of = None

    def __enter__(self) -> "FlatCaptureReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
