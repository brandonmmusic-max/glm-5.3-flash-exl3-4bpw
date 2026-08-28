"""Dtype-agnostic, streaming safetensors I/O with atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
CHUNK_BYTES = 32 << 20


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys instead of accepting last-key-wins headers."""

    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate safetensors JSON key {key!r}")
        value[key] = item
    return value


def tensor_nbytes(dtype: str, shape: Iterable[int]) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    size = DTYPE_BYTES[dtype]
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError(f"invalid tensor dimension {dimension!r}")
        size *= dimension
    return size


@dataclass(frozen=True)
class TensorRange:
    path: Path
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    def chunks(self, chunk_bytes: int = CHUNK_BYTES) -> Iterator[bytes]:
        with self.path.open("rb") as handle:
            handle.seek(self.start)
            remaining = self.nbytes
            while remaining:
                payload = handle.read(min(chunk_bytes, remaining))
                if not payload:
                    raise IOError(f"short read from {self.path} at {handle.tell()}")
                remaining -= len(payload)
                yield payload

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for payload in self.chunks():
            digest.update(payload)
        return digest.hexdigest()


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: TensorRange

    @property
    def nbytes(self) -> int:
        return self.payload.nbytes


class SafeTensorReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors prefix: {self.path}")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length > 1 << 30:
                raise ValueError(f"implausible safetensors header: {header_length}")
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise ValueError(f"truncated safetensors header: {self.path}")
        try:
            header = json.loads(raw_header, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors JSON header: {self.path}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header is not an object: {self.path}")
        raw_metadata = header.get("__metadata__", {})
        if not isinstance(raw_metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_metadata.items()
        ):
            raise ValueError(f"invalid safetensors metadata: {self.path}")
        self.metadata = dict(raw_metadata)
        self.header_sha256 = hashlib.sha256(raw_header).hexdigest()
        self.header_length = header_length
        self.data_start = 8 + header_length
        self.file_size = opened_stat.st_size
        self._file_identity = (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
        )
        self._mapped_bytes = None
        self.tensors: dict[str, TensorInfo] = {}
        file_size = self.file_size
        relative_ranges: list[tuple[int, int, str]] = []
        for name, raw in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"invalid tensor record {name!r} in {self.path}")
            dtype = str(raw["dtype"])
            shape = tuple(int(value) for value in raw["shape"])
            start, end = (int(value) for value in raw["data_offsets"])
            expected = tensor_nbytes(dtype, shape)
            if start < 0 or end < start or end - start != expected:
                raise ValueError(f"invalid byte range for {name} in {self.path}")
            absolute_start = self.data_start + start
            absolute_end = self.data_start + end
            if absolute_end > file_size:
                raise ValueError(f"truncated payload for {name} in {self.path}")
            if name in self.tensors:
                raise ValueError(f"duplicate tensor {name} in {self.path}")
            self.tensors[name] = TensorInfo(
                name,
                dtype,
                shape,
                TensorRange(self.path, absolute_start, absolute_end),
            )
            relative_ranges.append((start, end, name))

        # The format requires one dense, non-overlapping data buffer.  This
        # rejects aliases, holes, and unindexed trailing bytes before any
        # payload is trusted or copied into an output checkpoint.
        cursor = 0
        for start, end, name in sorted(relative_ranges):
            if start != cursor:
                relation = "overlap" if start < cursor else "hole"
                raise ValueError(
                    f"safetensors data {relation} before {name!r} in {self.path}"
                )
            cursor = end
        if self.data_start + cursor != file_size:
            raise ValueError(f"unindexed trailing bytes in {self.path}")

    def _file_backed_bytes(self):
        """Return one private, file-backed byte mapping for all tensor views.

        ``shared=False`` gives the mapping copy-on-write semantics: clean pages
        remain backed by the file/page cache and are shareable across worker
        processes, while an accidental tensor mutation cannot alter the sealed
        source file.  Tensor views retain the mapping's storage after this
        reader itself goes out of scope.
        """

        import torch

        current = self.path.stat()
        identity = (current.st_dev, current.st_ino, current.st_size)
        if identity != self._file_identity:
            raise ValueError(f"safetensors file changed before mapping: {self.path}")
        if self._mapped_bytes is None:
            mapped = torch.from_file(
                str(self.path),
                shared=False,
                size=self.file_size,
                dtype=torch.uint8,
            )
            if mapped.numel() != self.file_size:
                raise IOError(f"short safetensors mapping: {self.path}")
            self._mapped_bytes = mapped
        return self._mapped_bytes


PayloadFactory = Callable[[], bytes | bytearray | memoryview | TensorRange]


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes | bytearray | memoryview | TensorRange | PayloadFactory

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.dtype, self.shape)


def _payload_chunks(
    payload: bytes | bytearray | memoryview | TensorRange | PayloadFactory,
) -> tuple[Iterator[bytes], int]:
    value = payload() if callable(payload) else payload
    if isinstance(value, TensorRange):
        return value.chunks(), value.nbytes
    raw = memoryview(value)

    def iterator() -> Iterator[bytes]:
        for offset in range(0, len(raw), CHUNK_BYTES):
            yield raw[offset : offset + CHUNK_BYTES].tobytes()

    return iterator(), len(raw)


def write_safetensors_atomic(
    path: str | Path,
    entries: Iterable[TensorEntry],
    *,
    metadata: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = tuple(entries)
    names = [entry.name for entry in ordered]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor name in output shard")
    header: dict[str, object] = {}
    if metadata is not None:
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("safetensors metadata must be string-to-string")
        header["__metadata__"] = dict(sorted(metadata.items()))
    offset = 0
    for entry in ordered:
        size = entry.nbytes
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)

    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    file_digest = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    try:
        with os.fdopen(fd, "wb") as handle:
            prefix = struct.pack("<Q", len(raw_header)) + raw_header
            handle.write(prefix)
            file_digest.update(prefix)
            for entry in ordered:
                chunks, actual_size = _payload_chunks(entry.payload)
                if actual_size != entry.nbytes:
                    raise ValueError(
                        f"{entry.name}: payload has {actual_size} bytes, expected {entry.nbytes}"
                    )
                tensor_digest = hashlib.sha256()
                written = 0
                for chunk in chunks:
                    handle.write(chunk)
                    file_digest.update(chunk)
                    tensor_digest.update(chunk)
                    written += len(chunk)
                if written != entry.nbytes:
                    raise IOError(
                        f"short payload for {entry.name}: {written}/{entry.nbytes}"
                    )
                tensor_digests[entry.name] = tensor_digest.hexdigest()
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return tensor_digests, file_digest.hexdigest()


def torch_tensor_entry(name: str, tensor) -> TensorEntry:
    import torch

    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    dtype_map = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.float32: "F32",
        torch.int64: "I64",
        torch.float64: "F64",
    }
    if value.dtype not in dtype_map:
        raise ValueError(f"unsupported torch dtype {value.dtype}")
    # PyTorch rejects a dtype-changing view of a zero-dimensional tensor.
    # Flatten only the byte-extraction view; keep ``value.shape`` below so a
    # scalar remains a scalar in the safetensors header.
    payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return TensorEntry(name, dtype_map[value.dtype], tuple(value.shape), payload)


def read_torch_tensor(reader: SafeTensorReader, name: str):
    """Materialize one validated raw range without trusting pickle metadata."""

    import torch

    info = reader.tensors[name]
    dtype_map = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I32": torch.int32,
        "F32": torch.float32,
        "I64": torch.int64,
        "F64": torch.float64,
    }
    try:
        dtype = dtype_map[info.dtype]
    except KeyError as exc:
        raise ValueError(
            f"unsupported torch materialization dtype {info.dtype}"
        ) from exc
    # ``torch.frombuffer`` rejects an empty buffer on the live Torch 2.12
    # runtime.  Safetensors legitimately permits zero-length tensors (the row
    # cache uses one for an empty fallback-row domain), so construct that exact
    # empty shape directly instead of asking Torch to infer it from no bytes.
    if info.nbytes == 0:
        return torch.empty(info.shape, dtype=dtype)
    raw = bytearray().join(info.payload.chunks())
    return torch.frombuffer(raw, dtype=dtype).reshape(info.shape).clone()


def read_torch_tensor_mmap(reader: SafeTensorReader, name: str):
    """Return a validated tensor view backed by a private file mapping.

    Unlike :func:`read_torch_tensor`, this performs no payload copy.  The
    returned tensor must be treated as immutable.  Its mapping is private
    (copy-on-write), so even an accidental in-place operation cannot modify the
    source checkpoint or sidecar.  The ``SafeTensorReader`` has already
    validated the dense ranges, dtype byte sizes, shapes, and file boundary.
    """

    import torch

    info = reader.tensors[name]
    dtype_map = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I32": torch.int32,
        "F32": torch.float32,
        "I64": torch.int64,
        "F64": torch.float64,
    }
    try:
        dtype = dtype_map[info.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported torch mmap dtype {info.dtype}") from exc
    if info.nbytes == 0:
        return torch.empty(info.shape, dtype=dtype)
    element_size = torch.empty((), dtype=dtype).element_size()
    if info.payload.start % element_size:
        # Safetensors permits densely packed tensor ranges without per-tensor
        # dtype alignment. A byte mapping cannot be re-viewed at such an offset
        # on Torch 2.12, so retain format compatibility by materializing only
        # this uncommon unaligned tensor through the validated copy path.
        return read_torch_tensor(reader, name)
    raw = reader._file_backed_bytes()[info.payload.start : info.payload.end]
    if raw.numel() != info.nbytes:
        raise IOError(f"short mapped tensor payload: {name} in {reader.path}")
    try:
        return raw.view(dtype).reshape(info.shape)
    except RuntimeError as exc:
        raise ValueError(f"unaligned mapped tensor payload: {name} in {reader.path}") from exc
