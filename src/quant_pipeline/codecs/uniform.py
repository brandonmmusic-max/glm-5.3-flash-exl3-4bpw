from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedArray:
    packed: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    bits: int
    group_size: int
    count: int

    @property
    def stored_bytes(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes)


def pack_unsigned(values: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64).ravel()
    if bits < 1 or bits > 8 or np.any(values >= (1 << bits)):
        raise ValueError("values do not fit requested bit width")
    output = np.zeros((values.size * bits + 7) // 8, dtype=np.uint8)
    bit_position = 0
    for value in values:
        byte = bit_position >> 3
        shift = bit_position & 7
        word = int(value) << shift
        output[byte] |= word & 0xFF
        if shift + bits > 8:
            output[byte + 1] |= (word >> 8) & 0xFF
        bit_position += bits
    return output


def unpack_unsigned(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8).ravel()
    output = np.empty(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    bit_position = 0
    for index in range(count):
        byte = bit_position >> 3
        shift = bit_position & 7
        word = int(packed[byte])
        if shift + bits > 8:
            word |= int(packed[byte + 1]) << 8
        output[index] = (word >> shift) & mask
        bit_position += bits
    return output


def quantize(array: np.ndarray, bits: int, group_size: int = 128) -> QuantizedArray:
    if bits < 2 or bits > 8 or group_size < 1:
        raise ValueError("bits must be 2..8 and group_size positive")
    source = np.asarray(array, dtype=np.float32)
    flat = source.ravel()
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    codes = np.empty(flat.size, dtype=np.uint8)
    scales = np.empty((flat.size + group_size - 1) // group_size, dtype=np.float32)
    for group, start in enumerate(range(0, flat.size, group_size)):
        block = flat[start : start + group_size]
        scale = float(np.max(np.abs(block)) / qmax) if np.any(block) else 1.0
        scales[group] = scale
        signed = np.clip(np.rint(block / scale), qmin, qmax).astype(np.int16)
        codes[start : start + block.size] = (signed - qmin).astype(np.uint8)
    return QuantizedArray(pack_unsigned(codes, bits), scales, source.shape, bits, group_size, flat.size)


def dequantize(value: QuantizedArray) -> np.ndarray:
    qmax = (1 << (value.bits - 1)) - 1
    qmin = -qmax
    codes = unpack_unsigned(value.packed, value.count, value.bits).astype(np.int16) + qmin
    output = np.empty(value.count, dtype=np.float32)
    for group, start in enumerate(range(0, value.count, value.group_size)):
        stop = min(start + value.group_size, value.count)
        output[start:stop] = codes[start:stop] * value.scales[group]
    return output.reshape(value.shape)

