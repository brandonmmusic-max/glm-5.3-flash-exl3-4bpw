#!/usr/bin/env python3
"""
encode_tr3_v31.py — GLM-5.2 NVFP4 -> EXL3-trellis-3.0 hybrid production encoder.
v2: CALIBRATED LDLQ (owner mandate — the v1 identity-Hessian shortcut is gone).
v3: CROSS-SLICE LOCKSTEP LDLQ — same math, same bits, ~2-3x the speed.
v3.1: v3 + CROSS-SLICE LOCKSTEP g_scale GOLDEN-SECTION SEARCH — same bits.

  v2's LDLQ main pass launches ext.quantize_tiles on one 16-row block of ONE
  slice at a time (n=512 slices -> 32 tiles/call) against a kernel wave of
  2*SM tile-slots (296 on B200), idling ~89% of the wave.  v3 co-processes
  many independent slice-walks in LOCKSTEP: at each LDLQ step it gathers the
  current 16-row block from every active walk into ONE quantize_tiles call,
  scatters the results back, and runs each walk's error-feedback GEMMs on
  that walk's own tensors.  Per-slice operand values and op order are exactly
  v2's, and each tile's trellis DP is independent of every other tile in the
  batch (quantize.cu: tile_idx = blockIdx.x; all pointers offset purely by
  tile_idx; per-tile scratch slots), so v3 output is BIT-IDENTICAL to v2 —
  asserted once per worker (lockstep_self_check) and proven on real slices by
  --oracle (hard deployment gate).

  New CLI (encode CLI otherwise unchanged): --lockstep N (default 'auto' =
  12 = all 3*TP slice-walks of one expert; N>12 pools ceil(N/12) experts).
  Validation extras: --oracle / --oracle-experts / --oracle-log, --bench /
  --bench-experts (both read-only wrt --work).  Everything else — prologue
  math (finalize_capture_H, regularize, g_scale_gss per slice, seeds, su/sv
  draws), resumability, bitmap selection, bit-exactness asserts, logging,
  assemble/upload — is v2 verbatim.  Done-JSON gains a "lockstep" field.

  v3.1 extends the same lockstep idea to the remaining serial hot spot,
  regularize()'s per-slice g_scale golden-section search: the 12 slice-walks'
  searches are independent, so at each golden-section step ALL active
  searches' (sampled tiles x candidate scale) evaluations are gathered into
  ONE quantize_tiles call, the per-search MSEs scattered back, and each
  search's bracket advanced independently (searches whose bracket closes drop
  out of the pool; survivors keep pooling).  Tile sampling, candidate scales,
  iteration sequence, bracket arithmetic and tie behavior are VERBATIM
  v3 (= v2 = reference), so output stays bit-identical — asserted once per
  worker (pooled-vs-sequential g_scale equality on the first expert pool,
  g_scale_gss_lockstep self_check) and proven on real slices by --oracle
  (v3.1 vs v3 vs v2 byte equality, hard deployment gate).  CLI unchanged;
  done-JSON additionally gains "gss_lockstep": true.

Builds the KLC hybrid checkpoint from lukealonso/GLM-5.2-NVFP4 (ModelOpt-style
NVFP4, 87 shards):

  * per MoE layer (3..77): encode ALL 256 routed experts with EXL3 trellis at
    exactly 3.0 bpw via the exllamav3 v0.0.43 CALIBRATED path (finalize_capture_H
    -> block LDL -> ldlq blocked quantization with error feedback), TP4 per-slice
    (gate/up N-sliced 4x[512,6144], down K-sliced 4x[6144,512], each slice its
    own Hadamard/su/sv/g_scale/trellis encode);
  * Hessians come from the OWNER'S calibration corpus via capture_hessians.py
    (Pass A) — stored MoE-input activations x + top-8 routing ids per layer:
      - gate/up slice (k=6144): H_e = X_e^T X_e over the tokens ROUTED to
        expert e.  N-slicing does not change the input side, so all 8 gate/up
        slices of an expert share ONE 6144x6144 H (and, per reference
        semantics, one su drawn at first finalize; sv per slice).
      - down slice (k=512): I_e = silu(X_e Wg^T) * (X_e Wu^T) computed offline
        from the stored routed x with the DEQUANTIZED gate/up weights
        (GLM-5.2 config: hidden_act=silu, no swiglu clamp), H_down_e = I_e^T I_e
        [2048x2048]; K-slice r uses the [512r,512r+512) DIAGONAL block —
        slice-local LDLQ, consistent with slice-before-encode (each rank's
        down slice is an independent linear with input I_e[:,512r:512r+512),
        so its H is exactly that diagonal block; cross-slice H terms have no
        consumer because slices are encoded and Hadamard-transformed
        independently).
      - cold experts: if an expert saw < --min-routed (1024) routed tokens,
        its gate/up H falls back to the LAYER-level H (all tokens) and its
        down H to I over a seeded subsample of all tokens through that
        expert's own gate/up — recorded in the done-JSON, never a failure.
  * keep the 64 experts with the HIGHEST trellis round-trip error as NVFP4
    (copied byte-exact from the source; their trellis encodes are discarded);
    RT-error is now measured under the calibrated encode;
  * carry every non-(routed-expert) tensor byte-exact (verified by sha256);
  * assemble a complete checkpoint dir + index + tier_bitmap + MANIFEST;
  * upload to a private HF repo (token from HF_TOKEN env, never logged).

BIT-EXACTNESS GUARANTEES (owner mandate — these are correctness checks, not
quality gates; there is NO quality evaluation anywhere in this driver):
  (a) every carried tensor: sha256(source payload bytes) is asserted equal to
      sha256(bytes re-read from the written output shard);
  (b) every trellis tensor: pack->unpack index equality (torch.equal) AND
      ext.reconstruct(packed) is asserted bit-equal to the encoder's own
      reconstructed values (the values the home exl3 kernel will decode);
  (c) --assemble writes MANIFEST.sha256 over every output file and audits the
      index (every weight_map entry -> existing shard containing that tensor,
      and every shard tensor -> weight_map).

Per-expert RT-error is computed ONLY to choose the 64 NVFP4 keeps per layer.
It never gates, never aborts.

CALIBRATION IS MANDATORY: --encode refuses to start unless the capture dir
(Pass A, capture_hessians.py) is complete for every requested layer.  There is
no identity-Hessian escape hatch.  The reference q_fallback branch (count==0 /
degenerate diagonal) is vendored for faithfulness but cannot trigger under the
>= min-routed floor; if it ever does, it is recorded loudly in the done-JSON.

DEPENDENCIES on the box (see RUNBOOK_TR3.md):
    torch==2.11.0+cu130, numpy, safetensors (unused at runtime but harmless),
    huggingface_hub, and the exllamav3 v0.0.43 cu132/torch2.11/cp312 wheel
    installed with --no-deps.  Only the compiled extension `exllamav3_ext` is
    imported; the pure-Python quantization orchestration below is vendored
    VERBATIM from exllamav3 v0.0.43 (MIT, (c) turboderp), file
    exllamav3/modules/quant/exl3_lib/quantize.py, so the heavy deps of the
    exllamav3 python package (flash_attn, xformers, formatron, fla) are never
    imported.

OUTPUT EXL3 TENSOR SCHEMA (native exllamav3 Linear storage; loadable by
exllamav3.modules.quant.exl3.LinearEXL3 unmodified):
    model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.trellis  int16 [K/16, N/16, 48]
    model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.suh      float16 [K]
    model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.svh      float16 [N]
    model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.mcg      int32 scalar (0xCBAC1FED)
  where for proj in {gate_proj, up_proj}: K=6144 (model hidden), N=512
  (=2048/4, rank r holds output rows [512r, 512r+512) of the full [2048,6144]
  weight); for down_proj: K=512 (rank r holds input cols [512r, 512r+512) of
  the full [6144,2048] weight), N=6144.  K/N are exl3 conventions
  (in_features/out_features); trellis last dim = 16*bits = 48 at 3.0 bpw.

Usage (all commands also in RUNBOOK_TR3.md; capture_hessians.py --capture must
have completed first):
    python3 encode_tr3.py --smoke                       # kernel + calibrated-path check (synthetic H)
    python3 encode_tr3.py --encode [--layers 3-77]      # multi-GPU calibrated encode (resumable)
    python3 encode_tr3.py --assemble                    # build final checkpoint dir
    python3 encode_tr3.py --upload [--repo NAME]        # HF upload (HF_TOKEN env)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
import traceback
from functools import lru_cache

# ----------------------------------------------------------------------------
# defaults / constants
# ----------------------------------------------------------------------------

DEF_SRC = "/root/models/luke-nvfp4"
DEF_WORK = "/root/tr3_work"
DEF_OUT = "/root/out/GLM-5.2-NVFP4-TR3-Hybrid"
DEF_REPO = "brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid"
DEF_CAPTURE = "/root/tr3_capture"   # Pass A output (capture_hessians.py)

BITS = 3                      # owner-pinned: EXACTLY 3.0 bpw
KEEP_NVFP4 = 64               # experts kept NVFP4 per MoE layer
TP = 4                        # TP4 per-slice encode, decided before encode
SEED_BASE = 20260711
SIGMA_REG = 0.025             # exllamav3 v0.0.43 default Hessian damping
MIN_ROUTED = 1024             # per-expert routed-token floor -> layer-H fallback
FALLBACK_ROWS = 131072        # token subsample for cold-expert down-proj I
HCHUNK = 65536                # row chunk for H accumulation matmuls

FIRST_MOE_LAYER = 3           # config.first_k_dense_replace
NUM_LAYERS = 78               # config.num_hidden_layers (layer 78 = MTP, carried)
NUM_EXPERTS = 256             # config.n_routed_experts
HIDDEN = 6144                 # config.hidden_size
MOE_INTER = 2048              # config.moe_intermediate_size
SLICE = MOE_INTER // TP       # 512

PROJS = ("gate_proj", "up_proj", "down_proj")

# exllamav3 v0.0.43 constants (exl3_lib/quantize.py)
HAD_K, HAD_N = 128, 128
CODEBOOK_SCALE = 1.24371088
MCG_MULT = 0xCBAC1FED         # default codebook ("mcg") — matches convert_model.py default

ST_DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

CHUNK = 64 << 20  # 64 MiB io chunk


def log(msg: str, logfile: str | None = None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------
# raw safetensors IO (byte-exact, dtype-agnostic)
# ----------------------------------------------------------------------------

class STReader:
    """Minimal safetensors reader working on raw byte ranges."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        self.data_start = 8 + hlen
        self.tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            self.tensors[name] = (info["dtype"], tuple(info["shape"]),
                                  self.data_start + s, self.data_start + e)

    def nbytes(self, name: str) -> int:
        _, _, s, e = self.tensors[name]
        return e - s

    def read_bytes(self, name: str) -> bytes:
        _, _, s, e = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(s)
            return f.read(e - s)

    def sha256_stream(self, name: str) -> str:
        _, _, s, e = self.tensors[name]
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            f.seek(s)
            left = e - s
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    raise IOError(f"short read on {self.path}:{name}")
                h.update(b)
                left -= len(b)
        return h.hexdigest()


def st_file_complete(path: str) -> bool:
    """A shard is complete iff its header parses and the file covers all data.
    (hf downloads rename atomically, so existence normally suffices; this
    also guards against partial copies.)"""
    try:
        size = os.path.getsize(path)
        if size < 16:
            return False
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            if hlen > 100 << 20:
                return False
            header = json.loads(f.read(hlen))
        end = max((v["data_offsets"][1] for k, v in header.items()
                   if k != "__metadata__"), default=0)
        return size >= 8 + hlen + end
    except Exception:
        return False


def wait_for_shard(path: str, logfile: str | None = None):
    waited = False
    while not (os.path.exists(path) and st_file_complete(path)):
        if not waited:
            log(f"waiting for shard {os.path.basename(path)} (download in progress?)", logfile)
            waited = True
        time.sleep(30)
    if waited:
        log(f"shard ready: {os.path.basename(path)}", logfile)


def write_safetensors(path: str, entries: list, metadata: dict | None = None):
    """entries: list of (name, dtype_str, shape:tuple, payload) where payload is
    bytes or a zero-arg callable returning bytes.  Streams to disk; returns
    (per_tensor_payload_sha256: dict, file_sha256: str)."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    off = 0
    sizes = []
    for name, dt, shape, _ in entries:
        n = ST_DTYPE_SIZE[dt]
        for d in shape:
            n *= d
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        sizes.append(n)
        off += n
    assert len(header) == len(entries) + (1 if metadata else 0), "duplicate tensor name in shard"
    hjson = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(hjson) % 8)) % 8      # 8-byte align like the reference impl
    hjson += b" " * pad

    tensor_sha = {}
    fh = hashlib.sha256()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        head = struct.pack("<Q", len(hjson)) + hjson
        f.write(head)
        fh.update(head)
        for (name, dt, shape, payload), n in zip(entries, sizes):
            b = payload() if callable(payload) else payload
            if len(b) != n:
                raise ValueError(f"{name}: payload {len(b)} bytes != header {n}")
            f.write(b)
            fh.update(b)
            tensor_sha[name] = hashlib.sha256(b).hexdigest()
    os.replace(tmp, path)
    return tensor_sha, fh.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_range(path: str, start: int, end: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise IOError(f"short read {path} [{start},{end})")
            h.update(b)
            left -= len(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# source model map
# ----------------------------------------------------------------------------

class SourceModel:
    def __init__(self, src: str):
        self.src = src
        with open(os.path.join(src, "config.json")) as f:
            self.config = json.load(f)
        c = self.config
        assert c["num_hidden_layers"] == NUM_LAYERS, c["num_hidden_layers"]
        assert c["first_k_dense_replace"] == FIRST_MOE_LAYER
        assert c["n_routed_experts"] == NUM_EXPERTS
        assert c["hidden_size"] == HIDDEN
        assert c["moe_intermediate_size"] == MOE_INTER
        with open(os.path.join(src, "model.safetensors.index.json")) as f:
            self.index = json.load(f)
        self.weight_map = self.index["weight_map"]
        self._readers = {}

    def shard_path(self, tensor_name: str) -> str:
        return os.path.join(self.src, self.weight_map[tensor_name])

    def reader_for(self, tensor_name: str, logfile=None) -> STReader:
        p = self.shard_path(tensor_name)
        if p not in self._readers:
            wait_for_shard(p, logfile)
            self._readers[p] = STReader(p)
        return self._readers[p]

    def moe_layers(self):
        return list(range(FIRST_MOE_LAYER, NUM_LAYERS))  # 3..77 (78 = MTP, carried)


def expert_key(L, E, proj, suffix):
    return f"model.layers.{L}.mlp.experts.{E}.{proj}.{suffix}"


# ----------------------------------------------------------------------------
# VENDORED from exllamav3 v0.0.43 (MIT (c) turboderp-org), quantize.py —
# verbatim math, single-device, ProgressBar/stream scaffolding removed.
# Verified byte-identical between tag v0.0.43 and master @ 2026-07-10.
# ----------------------------------------------------------------------------

def _lazy_torch():
    import torch  # noqa
    import exllamav3_ext as ext  # noqa  (torch must be imported first)
    return torch, ext


@lru_cache
def tensor_core_perm(device):
    torch, _ = _lazy_torch()
    perm_a = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm_a[t * 8 + 0] = r0 * 16 + c0
        perm_a[t * 8 + 1] = r1 * 16 + c0
        perm_a[t * 8 + 2] = r2 * 16 + c0
        perm_a[t * 8 + 3] = r3 * 16 + c0
        perm_a[t * 8 + 4] = r0 * 16 + c1
        perm_a[t * 8 + 5] = r1 * 16 + c1
        perm_a[t * 8 + 6] = r2 * 16 + c1
        perm_a[t * 8 + 7] = r3 * 16 + c1
    return torch.tensor(perm_a, dtype=torch.int, device=device)


@lru_cache
def tensor_core_perm_i(device):
    torch, _ = _lazy_torch()
    return torch.argsort(tensor_core_perm(device))


@lru_cache
def _hadamard_f16(n: int):
    """Sylvester construction, identical to exllamav3 util/hadamard.py for
    power-of-two sizes (base case = hadamard_1.txt = [[1]], built in fp16)."""
    torch, _ = _lazy_torch()
    assert n & (n - 1) == 0 and n >= 1
    h = torch.ones((1, 1), dtype=torch.float16)
    while h.shape[0] < n:
        d = h.shape[0]
        s = torch.empty((d * 2, d * 2), dtype=h.dtype)
        s[:d, :d] = h
        s[:d, d:] = h
        s[d:, :d] = h
        s[d:, d:] = -h
        h = s
    return h


@lru_cache
def get_hadamard_dt(n, device, dtype, scale=1.0):
    had = _hadamard_f16(n).to(device=device, dtype=dtype, copy=True)
    had *= scale
    return had


def preapply_had_l(x, had_dim):
    torch, _ = _lazy_torch()
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (had @ x.view(-1, had_dim, n)).view(k, n)
    return x.to(x_dtype)


def preapply_had_r(x, had_dim):
    torch, _ = _lazy_torch()
    k, n = x.shape
    x_dtype = x.dtype
    x = x.to(torch.float)
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    x = (x.view(k, -1, had_dim) @ had).view(k, n)
    return x.to(x_dtype)


def blockwise_preapply_had_l_(x, had_dim):
    torch, _ = _lazy_torch()
    k, n = x.shape
    assert k % had_dim == 0 and x.dtype == torch.float
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    for i in range(k // had_dim):
        s = i * had_dim
        x[s:s + had_dim, :] = had @ x[s:s + had_dim, :].view(had_dim, n)


def blockwise_preapply_had_r_(x, had_dim):
    torch, _ = _lazy_torch()
    k, n = x.shape
    assert n % had_dim == 0 and x.dtype == torch.float
    had = get_hadamard_dt(had_dim, x.device, x.dtype, 1 / math.sqrt(had_dim))
    for i in range(n // had_dim):
        s = i * had_dim
        x[:, s:s + had_dim] = x[:, s:s + had_dim] @ had


def block_rms(x, dim, keepdim=False, blocksize=32):
    torch, _ = _lazy_torch()
    n = x.size(dim)
    sq = None
    for block in torch.split(x, blocksize, dim=dim):
        bsq = block.square().sum(dim=dim, keepdim=keepdim)
        sq = bsq if sq is None else sq + bsq
    return (sq / n).sqrt()


@lru_cache
def _quant_temp_buffers(K: int):
    """Same layout as exl3_lib get_temp_buffers, but sized 2*SM-count so the
    kernel's internal wave (MIN(temp.size(0), 2*num_sms)) is fully utilized."""
    torch, _ = _lazy_torch()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    b = 2 * sms
    temp_costs = torch.zeros((b, 2, 65536 >> K), dtype=torch.half, device="cuda:0")
    temp_edges = torch.zeros((b, 256, 65536 >> K), dtype=torch.short, device="cuda:0")
    return temp_costs, temp_edges


def quantize_tiles_batched(tiles, quant_args):
    """ext.quantize_tiles over an arbitrary batch of 256-element tiles.
    The kernel loops internally in waves, so one call suffices."""
    torch, ext = _lazy_torch()
    tiles = tiles.contiguous()
    assert tiles.shape[1] == 256 and tiles.dtype == torch.float
    K = quant_args["K"]
    q_tiles = torch.zeros_like(tiles)
    q_idx = torch.zeros_like(tiles, dtype=torch.short)
    temp_costs, temp_edges = _quant_temp_buffers(K)
    ext.quantize_tiles(tiles, q_tiles, q_idx, temp_costs, temp_edges,
                       K, "mcg" in quant_args, "mul1" in quant_args)
    return q_tiles, q_idx


def g_scale_gss(weight_r, quant_args, width=3):
    """Verbatim golden-section global-scale search (quantize.py g_scale_gss),
    minus streams/ProgressBar; identical sampling and iteration math."""
    torch, _ = _lazy_torch()
    tiles = []
    tiles_k = weight_r.shape[0] // 16
    tiles_n = weight_r.shape[1] // 16
    perm = tensor_core_perm(weight_r.device)
    for i in range(max(tiles_k, tiles_n)):
        for w in range(width):
            k = (i % tiles_k) * 16
            n = ((i + w) % tiles_n) * 16
            tile = weight_r[k:k + 16, n:n + 16].clone().view(256)
            tiles.append(tile[perm])
    tiles = torch.stack(tiles)

    def test_scale(scale):
        quant_w, _ = quantize_tiles_batched(tiles * scale, quant_args)
        return ((quant_w / scale - tiles) ** 2).mean()

    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    a, b = 0.1, 1.9
    tol = 0.01
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = test_scale(x1)
    f2 = test_scale(x2)
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = test_scale(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = test_scale(x2)
    return (a + b) / 2, (f1 + f2) / 2


# ----------------------------------------------------------------------------
# v3.1 CROSS-SLICE LOCKSTEP g_scale GOLDEN-SECTION SEARCH
#
# v3 pools the LDLQ main pass; the remaining serial hot spot is regularize()'s
# g_scale_gss: 13 quantize_tiles evaluations (2 bracket-init + 11 iterations
# for the fixed [0.1,1.9]/tol=0.01 bracket) on a ~1,152-tile sampled batch,
# PER SLICE, each eval with its own launch + device-sync round trip.  The 12
# slice-walks' searches are independent, so v3.1 runs them in LOCKSTEP: at
# each golden-section step it gathers every active search's (tiles x candidate
# scale) evaluation into ONE quantize_tiles call, scatters the per-search MSEs
# back, and advances each search's bracket independently.
#
# Bit-exactness argument (same shape as the v3 lockstep-LDLQ argument above):
#   * each search's sampled tile batch is built by the VERBATIM v3 loop;
#   * each search's bracket walk is the VERBATIM v3 iteration sequence —
#     _gss_bracket_gen() is a line-for-line copy of g_scale_gss()'s arithmetic
#     with test_scale(x) replaced by `yield x` (x1, x2, then exactly one new
#     candidate per iteration, evaluated even on the final iteration, exactly
#     like the reference);
#   * the pooled evaluation computes each search's input as its own verbatim
#     `tiles * scale` (torch.cat copies bits; quantize_tiles is per-tile
#     independent — see the v3 block comment) and each search's MSE with the
#     verbatim test_scale expression on that search's own contiguous rows of
#     the batched output (row-slices of a contiguous 2D tensor are contiguous
#     and 1024B-aligned, so the elementwise/mean kernels see byte-identical
#     standalone-shaped inputs);
#   * f-value comparisons move from 0-dim fp32 CUDA tensors to python floats:
#     fp32->fp64 conversion is exact and order-preserving, so every branch
#     (ties and NaN included) is decided identically; bracket arithmetic is
#     pure python fp64 in both versions.
# Searches whose bracket closes drop out of the pool; survivors keep pooling.
# (In practice all brackets close after the same 11 iterations — both golden-
# section branches shrink |b-a| by the identical factor phi-1 — but nothing
# below assumes that.)  Proven once per worker by the pooled-vs-sequential
# self_check on the first expert pool, and on real slices by --oracle
# (v3.1 == v3 == v2, hard deployment gate).
# ----------------------------------------------------------------------------

def _gss_bracket_gen():
    """Verbatim g_scale_gss() bracket walk as a coroutine: yields candidate
    scales in the reference evaluation order, receives each candidate's MSE,
    and returns ((a+b)/2, (f1+f2)/2) exactly like the reference."""
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    a, b = 0.1, 1.9
    tol = 0.01
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = yield x1
    f2 = yield x2
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = yield x1
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = yield x2
    return (a + b) / 2, (f1 + f2) / 2


_self_checked_gss = False


def g_scale_gss_lockstep(pre_ctxs, logfile=None, self_check=False, width=3):
    """Pooled golden-section g_scale search over many slices' regularized
    weights (see block comment above).  Sets ctx["g_scale"] on every entry,
    bit-identical to sequential g_scale_gss(ctx["weight_r"], ...).  One-time
    self_check (first pool of each worker): re-runs the UNTOUCHED v3
    sequential g_scale_gss per slice and asserts exact g_scale equality
    before any result is used."""
    global _self_checked_gss
    torch, _ = _lazy_torch()
    if not pre_ctxs:
        return
    qa = pre_ctxs[0]["quant_args"]
    for c in pre_ctxs:
        cq = c["quant_args"]
        assert cq["K"] == qa["K"] and ("mcg" in cq) == ("mcg" in qa) \
            and ("mul1" in cq) == ("mul1" in qa), \
            "gss lockstep pool requires uniform quantizer args"
    searches = []
    for c in pre_ctxs:
        weight_r = c["weight_r"]
        # --- sampled tile batch: VERBATIM v3 g_scale_gss construction --------
        tiles = []
        tiles_k = weight_r.shape[0] // 16
        tiles_n = weight_r.shape[1] // 16
        perm = tensor_core_perm(weight_r.device)
        for i in range(max(tiles_k, tiles_n)):
            for w in range(width):
                k = (i % tiles_k) * 16
                n = ((i + w) % tiles_n) * 16
                tile = weight_r[k:k + 16, n:n + 16].clone().view(256)
                tiles.append(tile[perm])
        tiles = torch.stack(tiles)
        gen = _gss_bracket_gen()
        searches.append({"ctx": c, "tiles": tiles, "gen": gen,
                         "scale": gen.send(None)})  # primes to x1 (cannot stop here)
    active = list(searches)
    while active:
        # one pooled evaluation wave: every active search contributes its own
        # verbatim test_scale input (tiles * candidate_scale)
        parts = [s["tiles"] * s["scale"] for s in active]
        batch = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        q_w, _ = quantize_tiles_batched(batch, qa)
        off = 0
        mses = []
        for s, p in zip(active, parts):
            m = p.shape[0]
            # verbatim test_scale MSE on this search's own rows
            mses.append(((q_w[off:off + m] / s["scale"] - s["tiles"]) ** 2).mean())
            off += m
        assert off == batch.shape[0]
        vals = torch.stack(mses).cpu().tolist()   # ONE device sync per wave
        nxt = []
        for s, v in zip(active, vals):
            try:
                s["scale"] = s["gen"].send(v)
                nxt.append(s)              # bracket still open — keep pooling
            except StopIteration as e:     # bracket closed — drop out of pool
                s["ctx"]["g_scale"] = e.value[0]
        active = nxt
    if self_check and not _self_checked_gss:
        for s in searches:
            g_seq, _ = g_scale_gss(s["ctx"]["weight_r"], s["ctx"]["quant_args"])
            assert g_seq == s["ctx"]["g_scale"], \
                f"pooled GSS g_scale {s['ctx']['g_scale']!r} != sequential {g_seq!r}"
        _self_checked_gss = True
        log(f"self-check PASSED: pooled GSS ({len(searches)} searches) == sequential "
            f"g_scale_gss, exact g_scale equality per slice", logfile)
    for s in searches:
        s["tiles"] = None
        s["gen"] = None


def regularize(weight, su, sv, quant_args, H_diag=None, q_fallback=False):
    """Verbatim quantize.py regularize() (v0.0.43, full calibrated version:
    H_diag skew decides out-channel scaling), ProgressBar/verbose/save-image
    scaffolding removed.  Mutates/returns weight."""
    torch, _ = _lazy_torch()
    force_out_scales = quant_args["apply_out_scales"]

    # From experiments, it seems the deciding factor in when scaling output
    # channels is beneficial is when the input to the linear layer is very
    # irregular.  Cutoff: 15% of the RMS sum on 2% of the channels.
    if not q_fallback and H_diag is not None:
        diag = H_diag.sqrt()
        diag, _ = torch.sort(diag, descending=True)
        cutoff = diag.shape[0] // 50
        skew_factor = diag[:cutoff].sum() / diag.sum()
        if force_out_scales is None:
            apply_out_scales = skew_factor.item() < 0.15
        else:
            apply_out_scales = force_out_scales
    else:
        apply_out_scales = True if force_out_scales is None else force_out_scales

    if q_fallback:
        apply_out_scales = force_out_scales

    out_channel_scales = block_rms(weight, dim=0, keepdim=True)
    mean = out_channel_scales.mean().item()
    if mean > 1e-30:
        out_channel_scales /= mean
        quant_args["zeros"] = False
    else:
        quant_args["zeros"] = True
        if force_out_scales is not None:
            apply_out_scales = True
    zero_out_scales = out_channel_scales.abs() < 1e-30

    if apply_out_scales:
        out_channel_scales[zero_out_scales] = 0.1
        sv = (sv * out_channel_scales + 1e-10).float()

    weight /= sv
    sv[zero_out_scales] = 0.0
    blockwise_preapply_had_r_(weight, HAD_N)

    in_channel_scales = block_rms(weight, dim=1, keepdim=True)
    in_channel_scales[in_channel_scales.abs() < 1e-30] = 0.1
    su = (su * in_channel_scales / (-CODEBOOK_SCALE) + 1e-10).float()
    weight /= su
    blockwise_preapply_had_l_(weight, HAD_K)

    g_scale, _ = g_scale_gss(weight, quant_args)
    weight *= g_scale
    su /= g_scale
    return apply_out_scales, weight, g_scale, su, sv


def regularize_pre(weight, su, sv, quant_args, H_diag=None, q_fallback=False):
    """v3.1: VERBATIM regularize() head — everything up to (excluding) the
    g_scale_gss call, so the search itself can run pooled across slices
    (g_scale_gss_lockstep) followed by regularize_post().  regularize() above
    stays byte-untouched (v2 verbatim; used by encode_slice, i.e. --smoke and
    the --oracle reference passes, and by the v3-path encode_slice_prologue)."""
    torch, _ = _lazy_torch()
    force_out_scales = quant_args["apply_out_scales"]

    # From experiments, it seems the deciding factor in when scaling output
    # channels is beneficial is when the input to the linear layer is very
    # irregular.  Cutoff: 15% of the RMS sum on 2% of the channels.
    if not q_fallback and H_diag is not None:
        diag = H_diag.sqrt()
        diag, _ = torch.sort(diag, descending=True)
        cutoff = diag.shape[0] // 50
        skew_factor = diag[:cutoff].sum() / diag.sum()
        if force_out_scales is None:
            apply_out_scales = skew_factor.item() < 0.15
        else:
            apply_out_scales = force_out_scales
    else:
        apply_out_scales = True if force_out_scales is None else force_out_scales

    if q_fallback:
        apply_out_scales = force_out_scales

    out_channel_scales = block_rms(weight, dim=0, keepdim=True)
    mean = out_channel_scales.mean().item()
    if mean > 1e-30:
        out_channel_scales /= mean
        quant_args["zeros"] = False
    else:
        quant_args["zeros"] = True
        if force_out_scales is not None:
            apply_out_scales = True
    zero_out_scales = out_channel_scales.abs() < 1e-30

    if apply_out_scales:
        out_channel_scales[zero_out_scales] = 0.1
        sv = (sv * out_channel_scales + 1e-10).float()

    weight /= sv
    sv[zero_out_scales] = 0.0
    blockwise_preapply_had_r_(weight, HAD_N)

    in_channel_scales = block_rms(weight, dim=1, keepdim=True)
    in_channel_scales[in_channel_scales.abs() < 1e-30] = 0.1
    su = (su * in_channel_scales / (-CODEBOOK_SCALE) + 1e-10).float()
    weight /= su
    blockwise_preapply_had_l_(weight, HAD_K)

    return apply_out_scales, weight, su, sv


def regularize_post(weight, su, g_scale):
    """v3.1: VERBATIM regularize() tail (g_scale application)."""
    weight *= g_scale
    su /= g_scale
    return weight, su


@lru_cache
def get_quant_stream(device):
    torch, _ = _lazy_torch()
    return torch.cuda.Stream(device=device)


def block_ldl(H, b, quant_args, verbose=False):
    """Verbatim quantize.py block_ldl() (v0.0.43): Cholesky with diagonal-damping
    retries, blocked column normalization, identity diagonal blocks.  Returns
    (L, H_cpu); H's GPU storage is reused for L (reference memory trick)."""
    torch, _ = _lazy_torch()
    n, _ = H.shape
    assert (n % b == 0)
    m = n // b

    num_cholesky_retries = 0
    retry_cpu = False
    while True:
        try:
            L = torch.linalg.cholesky(H)
            H_cpu = H.cpu()
            H.copy_(L)
            L = H
            H = H_cpu
            break
        except torch._C._LinAlgError as e:
            num_cholesky_retries += 1
            if num_cholesky_retries > 10:
                print(" ## Cholesky decomp. failed, number of retries exceeded")
                raise e
            print(f" !! Cholesky decomp. failed, increasing diagonal damping, "
                  f"attempt {num_cholesky_retries}/10")
            H.diagonal().add_(2.0 * quant_args.get("sigma_reg", 0.025) * H.diagonal().mean())
            continue
        except Exception as e:
            if e.__class__.__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e) \
                    or "HIP out of memory" in str(e):
                retry_cpu = True
                break
            else:
                raise e

    if retry_cpu:
        print(f" !! Out of memory on {str(H.device)}, trying CPU fallback")
        torch.cuda.empty_cache()
        H_cpu = H.cpu()
        L_cpu = torch.linalg.cholesky(H_cpu)
        H.copy_(L_cpu)
        del L_cpu
        L = H
        H = H_cpu

    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]
    L = L.reshape(n, n).contiguous()

    L_block = L.view(m, b, m, b).permute(0, 2, 1, 3)
    dr = torch.arange(m)
    L_block[dr, dr] = torch.stack([torch.eye(b, device=L.device, dtype=H.dtype)] * m)
    return L, H


def ldlq(weight, L, quant_args):
    """Verbatim quantize.py ldlq() (v0.0.43): blocked (buf_size_k=128) LDLQ with
    error feedback — 16-row blocks quantized bottom-up, compensation from the
    LDL factor, cross-span error carried in prod_cache.  ProgressBar removed;
    quantize_tiles_multigpu -> quantize_tiles_batched (single device; asserted
    bit-identical to the reference kernel wrapper by the worker self-checks)."""
    torch, _ = _lazy_torch()
    devices = quant_args["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        device = L.device
        assert device == torch.device(devices[0])

        buffer_device = weight.device
        size_k, size_n = weight.shape  # Row-major
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        tiles_k = size_k // 16
        tiles_n = size_n // 16

        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0
        assert size_n % buf_size_k == 0

        prod_cache = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
        weight_q = torch.zeros((size_k, size_n), dtype=torch.float, device=buffer_device)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.short, device=buffer_device)

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k

            b_weight = weight[i:j].to(device)
            b_weight_q = weight_q[i:j] if device == buffer_device else \
                torch.zeros_like(weight_q[i:j], device=device)
            b_encoded = encoded[i // 16: j // 16] if device == buffer_device else \
                torch.zeros_like(encoded[i // 16: j // 16], device=device)
            b_prod_cache = prod_cache[i:j]
            b_L = L[i:j]

            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16

                # Error so far for the current span
                bb_err = b_weight[bj:] - b_weight_q[bj:]
                # Corresponding slice of LDL decomposition of H
                bb_L = b_L[bj:, i + bi:i + bj]
                # Input tiles for quantization
                compensation_term = b_prod_cache[bi:bj]
                compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
                rows = b_weight[bi:bj] + compensation_term

                tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
                tiles = tiles[:, tensor_core_perm(device)]
                quant_w, quant_i = quantize_tiles_batched(tiles, quant_args)
                quant_w = quant_w[:, tensor_core_perm_i(device)]
                quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)
                b_weight_q[bi:bj] = quant_w
                b_encoded[bi // 16: bj // 16] = quant_i.unsqueeze(0)

            if device != buffer_device:
                weight_q[i:j] = b_weight_q.to(buffer_device)
                encoded[i // 16: j // 16] = b_encoded.to(buffer_device)

            # Cache error term for the rest of the matrix
            b_err = b_weight - b_weight_q
            prod_cache.addmm_(b_L.T, b_err, alpha=1.0, beta=1.0)

        for device in devices:
            torch.cuda.synchronize(device)

    return weight_q, encoded


# ----------------------------------------------------------------------------
# v3 CROSS-SLICE LOCKSTEP LDLQ
#
# quantize_tiles_kernel (exllamav3 v0.0.43 quantize.cu) assigns one thread
# block per tile (tile_idx = blockIdx.x); the input/output/index/temp pointers
# are all offset purely by tile_idx (temp_edges is a per-tile-slot scratch;
# for K>=2 the DP costs live in the block's own shared memory), so a tile's
# result is a pure function of its own 256 values — batch composition cannot
# change per-tile results.  The host wrapper loops internally in waves of
# MIN(temp_costs.size(0), 2*num_sms) blocks, and _quant_temp_buffers already
# allocates exactly 2*SM slots, so one call handles any batch size with full
# waves and correct per-slot scratch (no temp resize needed — the underfill
# was purely caller-side: 32 tiles/call vs 296 slots).
#
# LDLQWalk steps ONE slice's ldlq() 16-row block at a time — prepare() and
# commit() are the v2 ldlq() inner-loop math, verbatim, on the walk's own
# tensors — and lockstep_ldlq() co-processes many independent walks, gathering
# every active walk's current tiles into ONE quantize_tiles call per step.
# Each walk's error feedback uses only that walk's own outputs, and per-walk
# op order/operands are exactly v2's, so lockstep(K walks) is bit-identical
# to sequential per-slice ldlq() — asserted once per worker by
# lockstep_self_check() and proven on real slices by --oracle.
# ----------------------------------------------------------------------------

class LDLQWalk:
    """One slice's ldlq() as a stepped state machine (verbatim v2 math)."""

    __slots__ = ("ctx", "weight", "L", "weight_q", "encoded", "prod_cache",
                 "size_k", "size_n", "tiles_k", "tiles_n", "buf", "j", "bj",
                 "perm", "perm_i", "done")

    def __init__(self, ctx):
        torch, _ = _lazy_torch()
        weight, L, quant_args = ctx["weight_r"], ctx["L"], ctx["quant_args"]
        device = L.device
        assert device == torch.device(quant_args["devices"][0])
        assert weight.device == device, "lockstep requires weight and L on one device"
        size_k, size_n = weight.shape          # Row-major
        assert size_k % 16 == 0
        assert size_n % 128 == 0
        buf = max(quant_args.get("buf_size_k", 128), 16)
        assert buf % 16 == 0
        assert size_n % buf == 0 and size_k % buf == 0
        self.ctx = ctx
        self.weight = weight                   # read-only here (as in v2 ldlq)
        self.L = L                             # read-only here (diag pre-zeroed)
        self.size_k, self.size_n = size_k, size_n
        self.tiles_k, self.tiles_n = size_k // 16, size_n // 16
        self.buf = buf
        self.prod_cache = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
        self.weight_q = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
        self.encoded = torch.zeros((self.tiles_k, self.tiles_n, 256),
                                   dtype=torch.short, device=device)
        self.perm = tensor_core_perm(device)
        self.perm_i = tensor_core_perm_i(device)
        self.j = size_k                        # span end (spans walk bottom-up)
        self.bj = buf                          # in-span block end
        self.done = False

    def prepare(self):
        """Verbatim v2 ldlq() inner-iteration head for block [bi,bj) of span
        [i,j): in-place compensation addmm_ + tile gather/permute.  Returns
        this walk's [tiles_n, 256] fp32 tile rows for the batched call."""
        j = self.j
        i = j - self.buf
        bj = self.bj
        bi = bj - 16
        b_weight = self.weight[i:j]
        b_weight_q = self.weight_q[i:j]
        b_prod_cache = self.prod_cache[i:j]
        b_L = self.L[i:j]
        # Error so far for the current span
        bb_err = b_weight[bj:] - b_weight_q[bj:]
        # Corresponding slice of LDL decomposition of H
        bb_L = b_L[bj:, i + bi:i + bj]
        # Input tiles for quantization
        compensation_term = b_prod_cache[bi:bj]
        compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
        rows = b_weight[bi:bj] + compensation_term
        tiles = rows.reshape(16, self.tiles_n, 16).permute(1, 0, 2).reshape(self.tiles_n, 256)
        tiles = tiles[:, self.perm]
        return tiles

    def commit(self, quant_w, quant_i):
        """Verbatim v2 ldlq() inner-iteration tail: un-permute, store, and at
        span end run the cross-span error-feedback addmm_ and advance."""
        j = self.j
        i = j - self.buf
        bj = self.bj
        bi = bj - 16
        quant_w = quant_w[:, self.perm_i]
        quant_w = quant_w.reshape(self.tiles_n, 16, 16).permute(1, 0, 2).reshape(16, self.size_n)
        self.weight_q[i + bi:i + bj] = quant_w
        self.encoded[(i + bi) // 16:(i + bj) // 16] = quant_i.unsqueeze(0)
        self.bj -= 16
        if self.bj == 0:
            # Cache error term for the rest of the matrix
            b_err = self.weight[i:j] - self.weight_q[i:j]
            self.prod_cache.addmm_(self.L[i:j].T, b_err, alpha=1.0, beta=1.0)
            self.j -= self.buf
            self.bj = self.buf
            if self.j == 0:
                self.done = True
                self.ctx["wq_reg"] = self.weight_q
                self.ctx["encoded"] = self.encoded
        return self.done


def lockstep_ldlq(walks, n_max, logfile=None):
    """Advance up to n_max LDLQ walks in lockstep with ONE quantize_tiles call
    per step (see block comment above for the bit-exactness argument).  Same
    stream/sync discipline as v2 ldlq(): sync all devices, run every op on the
    cached quant stream, sync again before returning."""
    torch, _ = _lazy_torch()
    if not walks:
        return
    qa = walks[0].ctx["quant_args"]
    for w in walks:
        wq = w.ctx["quant_args"]
        assert wq["K"] == qa["K"] and ("mcg" in wq) == ("mcg" in qa) \
            and ("mul1" in wq) == ("mul1" in qa) and wq["devices"] == qa["devices"], \
            "lockstep pool requires uniform quantizer args"
    devices = qa["devices"]
    for device in devices:
        torch.cuda.synchronize(device)
    main_stream = get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        pending = list(walks)
        active = []
        while pending or active:
            while pending and len(active) < n_max:
                active.append(pending.pop(0))
            parts = [w.prepare() for w in active]
            tiles = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            q_w, q_i = quantize_tiles_batched(tiles, qa)
            off = 0
            nxt = []
            for w, p in zip(active, parts):
                m = p.shape[0]
                w.commit(q_w[off:off + m], q_i[off:off + m])
                off += m
                if not w.done:
                    nxt.append(w)
            assert off == tiles.shape[0]
            active = nxt
    for device in devices:
        torch.cuda.synchronize(device)


def resolve_lockstep(val) -> int:
    """--lockstep: 'auto' -> 3*TP = 12, i.e. all slice-walks of one expert
    co-stepped (8 gate/up walks x 32 tiles + 4 down walks x 384 tiles; the
    steady-state gate/up phase gathers 256 tiles vs the 296-slot wave).
    Integer N > 12 pools ceil(N/12) experts' walks together."""
    if isinstance(val, int):
        n = val
    elif str(val).strip().lower() == "auto":
        return 3 * TP
    else:
        n = int(val)
    assert n >= 1, f"--lockstep must be >= 1 or 'auto' (got {val!r})"
    return n


import threading
finalize_capture_H_mutex = threading.Lock()


def finalize_capture_H(H_data, quant_args, verbose=False):
    """Verbatim quantize.py finalize_capture_H() (v0.0.43): mean over captured
    rows, sigma_reg diagonal damping, su sign draw (shared by all linears using
    this H_data), su/128-Hadamard transform of H, block LDL with zeroed
    diagonal.  Returns (q_fallback, H, L, su, H_diag)."""
    torch, _ = _lazy_torch()
    with finalize_capture_H_mutex:

        if H_data["H"].is_meta:
            H_data["L"] = None
            H_data["finalized"] = True
            H_data["diag"] = None
            H_data["q_fallback"] = True
            H = H_data["H"]
            k = H.shape[0]
            su = (torch.randn(k, device=H_data["device"]).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
            H_data["su"] = su
            return True, None, None, su, None

        if "H_swap_device" in H_data:
            H_data["H"] = H_data["H"].to(H_data["H_swap_device"])
            del H_data["H_swap_device"]

        H = H_data["H"]
        if H_data["finalized"]:
            return H_data["q_fallback"], H, H_data["L"], H_data["su"], H_data["diag"]

        # Mean of samples summed up during forward pass; uncalibrated fallback
        # if no input activations or diagonal too small
        count = H_data["count"]
        if count == 0:
            q_fallback = True
            diag_mean = 0.0
        else:
            H /= count
            diag_mean = torch.diag(H).mean()
            q_fallback = diag_mean.item() < 1e-20

        # Regularize diagonal
        H.diagonal().add_(quant_args.get("sigma_reg", 0.025) * diag_mean)
        diag = H.diagonal().clone()

        # Random sign flips for input channel, fixed for the first linear
        # layer to quantize with this H
        k = H.shape[0]
        su = (torch.randn(k, device=H.device).sign() + 1e-5).sign().to(torch.float).unsqueeze(1)
        H_data["su"] = su

        # Input had
        H *= su.T
        blockwise_preapply_had_r_(H, HAD_K)
        H *= su
        blockwise_preapply_had_l_(H, HAD_K)

        # Get block LDL decomposition of H, zero diagonal
        if q_fallback:
            L = None
        else:
            L, H = block_ldl(H, 16, quant_args, verbose)
            dr = torch.arange(k)
            L[dr, dr] = 0

        H_data["L"] = L
        H = H.cpu()
        H_data["H"] = H.cpu()
        H_data["finalized"] = True
        H_data["diag"] = diag
        H_data["q_fallback"] = q_fallback
        return q_fallback, H, L, su, diag


def block_trace(A, B, block_size=1024):
    """Verbatim inner helper of quantize_exl3() metrics: trace(A^T B A) summed
    in blocks (A: [k, n], B: [k, k])."""
    torch, _ = _lazy_torch()
    total = 0.0
    for j_start in range(0, B.shape[1], block_size):
        j_end = min(j_start + block_size, B.shape[1])
        B_block = B[:, j_start:j_end]
        A_j_block = A[j_start:j_end, :]
        partial = torch.einsum("ik,ij,jk->", A, B_block, A_j_block)
        total += partial.item()
    return total


def pack_trellis(encoded, quant_args):
    torch, ext = _lazy_torch()
    K = quant_args["K"]
    shape = encoded.shape
    assert len(shape) == 3 and shape[2] == 256 and encoded.dtype == torch.int16
    packed = torch.zeros((shape[0], shape[1], 256 * K // 16),
                         dtype=torch.int16, device=encoded.device)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed


# ----------------------------------------------------------------------------
# trellis quantization of one (k, n) tensor — LDLQ-calibrated (fallback path
# retained verbatim for the reference q_fallback branch + self-checks)
# ----------------------------------------------------------------------------

def trellis_quant_all_tiles(weight_r, quant_args):
    """Identity-Hessian quantization of the full tensor in one batch.
    With H=I there is no LDLQ error feedback, so every 16-row block is
    independent — bit-identical to the reference fallback_quant() loop
    (asserted by self_check_reference once per worker)."""
    torch, _ = _lazy_torch()
    k, n = weight_r.shape
    tk, tn = k // 16, n // 16
    perm = tensor_core_perm(weight_r.device)
    perm_i = tensor_core_perm_i(weight_r.device)
    tiles = (weight_r.view(tk, 16, tn, 16).permute(0, 2, 1, 3)
             .reshape(tk * tn, 256))[:, perm].contiguous()
    q_tiles, q_idx = quantize_tiles_batched(tiles, quant_args)
    wq = (q_tiles[:, perm_i].view(tk, tn, 16, 16).permute(0, 2, 1, 3)
          .reshape(k, n).contiguous())
    encoded = q_idx.view(tk, tn, 256).contiguous()
    return wq, encoded


def reference_fallback_quant(weight_r, quant_args):
    """Verbatim inner loop of quantize.py fallback_quant() (streams removed):
    used once per worker to assert trellis_quant_all_tiles is bit-identical."""
    torch, _ = _lazy_torch()
    device = weight_r.device
    size_k, size_n = weight_r.shape
    tiles_k, tiles_n = size_k // 16, size_n // 16
    buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
    assert buf_size_k % 16 == 0 and size_n % buf_size_k == 0
    perm = tensor_core_perm(device)
    perm_i = tensor_core_perm_i(device)
    weight_q = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
    encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.short, device=device)
    for j in range(size_k, 0, -buf_size_k):
        i = j - buf_size_k
        b_weight = weight_r[i:j]
        for bj in range(buf_size_k, 0, -16):
            bi = bj - 16
            rows = b_weight[bi:bj]
            tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
            tiles = tiles[:, perm]
            quant_w, quant_i = quantize_tiles_batched(tiles, quant_args)
            quant_w = quant_w[:, perm_i]
            quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)
            weight_q[i + bi:i + bj] = quant_w
            encoded[(i + bi) // 16:(i + bj) // 16] = quant_i.unsqueeze(0)
    return weight_q, encoded


_self_checked = False
_self_checked_ldlq0 = False
_self_checked_lockstep = False


def encode_slice(w_slice, H_data, seed, out_scales, logfile=None, self_check=False):
    """Encode one (k, n) float32 CUDA tensor (exl3 convention: k=in_features,
    n=out_features) at BITS bpw with the CALIBRATED LDLQ path, mirroring
    exllamav3 v0.0.43 quantize_exl3() step for step.

    H_data: dict as built by make_H_data() — H = SUM of x x^T over calibration
    rows (fp32 [k,k] CUDA), count = rows.  May be SHARED between slices with
    identical inputs (all 8 gate/up slices of an expert): the first encode
    finalizes it (draws su, transforms H, computes the block LDL); subsequent
    encodes reuse the cached factorization and su, drawing only their own sv —
    exactly the reference behavior for linears sharing a qmap.

    Returns (out: dict of cpu tensors {suh, svh, trellis}, stats dict).
    Raises AssertionError on any exactness failure (exact-or-bug)."""
    global _self_checked, _self_checked_ldlq0
    torch, ext = _lazy_torch()
    k, n = w_slice.shape
    assert w_slice.dtype == torch.float32 and w_slice.is_cuda
    assert k % HAD_K == 0 and n % HAD_N == 0 and n % 128 == 0
    assert H_data["H"].is_meta or tuple(H_data["H"].shape) == (k, k), \
        f"H shape {tuple(H_data['H'].shape)} != ({k},{k})"
    device = w_slice.device

    quant_args = {"K": BITS, "devices": [0], "apply_out_scales": out_scales,
                  "mcg": True, "sigma_reg": SIGMA_REG}

    # --- reference order: seed -> finalize (su on first use) -> sv ----------
    torch.manual_seed(seed)
    q_fallback, H, L, su, H_diag = finalize_capture_H(H_data, quant_args, verbose=False)
    if H is not None and not H.is_cuda:
        H = H.to(device)                 # working copy for the proxy metric
    if L is not None and not L.is_cuda:
        L = L.to(device)
    if su.device != device:
        su = su.to(device)
    if H_diag is not None and H_diag.device != device:
        H_diag = H_diag.to(device)
    sv = (torch.randn(n, device=device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
    # reference keeps the stored L on CPU between linears sharing this H_data
    if H_data.get("L") is not None and H_data["L"].is_cuda:
        H_data["L"] = H_data["L"].cpu()

    w_orig = w_slice
    apply_out_scales, weight_r, g_scale, su, sv = regularize(
        w_slice.clone(), su, sv, quant_args, H_diag=H_diag, q_fallback=q_fallback)

    if not q_fallback:
        # one-time wiring proof: ldlq with L=0 (no feedback) must equal the
        # independent batched tile quant (which itself is asserted equal to the
        # reference fallback loop below)
        if self_check and not _self_checked_ldlq0:
            wq_a, enc_a = ldlq(weight_r, torch.zeros_like(L), quant_args)
            wq_b, enc_b = trellis_quant_all_tiles(weight_r, quant_args)
            assert torch.equal(enc_a, enc_b), "ldlq(L=0) vs batched: encoded indices differ"
            assert torch.equal(wq_a, wq_b), "ldlq(L=0) vs batched: reconstructed values differ"
            wq_c, enc_c = reference_fallback_quant(weight_r, quant_args)
            assert torch.equal(enc_a, enc_c) and torch.equal(wq_a, wq_c), \
                "batched vs reference fallback loop differ"
            _self_checked_ldlq0 = True
            _self_checked = True
            log("self-check PASSED: ldlq(L=0) == batched tile quant == reference fallback loop", logfile)
        wq_reg, encoded = ldlq(weight_r, L, quant_args)
    else:
        # reference q_fallback branch (uncalibrated) — cannot trigger under the
        # min-routed floor; kept verbatim, recorded loudly by the caller
        if self_check and not _self_checked:
            wq_a, enc_a = trellis_quant_all_tiles(weight_r, quant_args)
            wq_b, enc_b = reference_fallback_quant(weight_r, quant_args)
            assert torch.equal(enc_a, enc_b), "batched vs reference: encoded indices differ"
            assert torch.equal(wq_a, wq_b), "batched vs reference: reconstructed values differ"
            _self_checked = True
            log("self-check PASSED: batched tile quant is bit-identical to reference fallback_quant", logfile)
            wq_reg, encoded = wq_a, enc_a
        else:
            wq_reg, encoded = trellis_quant_all_tiles(weight_r, quant_args)

    # --- reference proxy error: trace(E H E^T) / trace(W H W^T) -------------
    if not q_fallback:
        E = weight_r - wq_reg
        num = block_trace(E, H)
        den = block_trace(weight_r, H)
        proxy_err = num / max(den, 1e-8)
        del E
    else:
        proxy_err = (weight_r - wq_reg).square().mean().item()

    # --- mandatory encode->decode round-trip exactness (owner mandate (b)) ---
    packed = pack_trellis(encoded, quant_args)
    unpacked = torch.zeros_like(encoded)
    ext.unpack_trellis(unpacked, packed, BITS)
    assert torch.equal(unpacked, encoded), "pack/unpack round-trip mismatch (format bug)"
    w_dec = torch.empty((k, n), dtype=torch.half, device=packed.device)
    ext.reconstruct(w_dec, packed, BITS, True, False)  # mcg codebook
    assert torch.equal(w_dec, wq_reg.half()), \
        "ext.reconstruct != encoder values (codebook/packing bug)"
    expected_bytes = k * n * BITS // 8
    assert packed.numel() * 2 == expected_bytes, "trellis size != exact 3.0 bpw"

    # relative round-trip error in weight domain (selection metric ONLY)
    wq = preapply_had_l(wq_reg, HAD_K)
    wq *= su
    wq = preapply_had_r(wq, HAD_N)
    wq *= sv
    sse = (wq - w_orig).double().pow(2).sum().item()
    ss = w_orig.double().pow(2).sum().item()

    out = {
        "suh": su.flatten().contiguous().to(dtype=torch.half).cpu(),
        "svh": sv.flatten().contiguous().to(dtype=torch.half).cpu(),
        "trellis": packed.cpu(),
    }
    stats = {"sse": sse, "ss": ss, "nmse": sse / max(ss, 1e-30), "g_scale": g_scale,
             "proxy_err": proxy_err, "q_fallback": bool(q_fallback),
             "apply_out_scales": bool(apply_out_scales) if apply_out_scales is not None else None}
    return out, stats


# ----------------------------------------------------------------------------
# v3: encode_slice split into prologue / (lockstep LDLQ) / epilogue.
# encode_slice() above is kept VERBATIM v2 (used by --smoke, --oracle and the
# one-time self-checks); the split below is the same code with the ldlq() call
# lifted out so many slices' main passes can run in lockstep.  RNG contract is
# identical to v2: everything a slice draws (su on first finalize of a shared
# H, sv always) happens in its prologue under torch.manual_seed(seed) with
# seeds derived from (L, E, proj, r); neither the LDLQ walk nor the epilogue
# consumes RNG — so results are independent of walk scheduling order.
# ----------------------------------------------------------------------------

def encode_slice_prologue(w_slice, H_data, seed, out_scales, logfile=None, self_check=False):
    """First half of encode_slice() (v2-verbatim math): seed -> finalize ->
    sv draw -> regularize (incl. per-slice g_scale_gss, UNCHANGED per owner
    mandate).  Returns a ctx dict; when ctx["wq_reg"] is None the slice still
    needs its LDLQ main pass (run it as an LDLQWalk inside lockstep_ldlq),
    otherwise it was completed inline (reference q_fallback branch)."""
    global _self_checked, _self_checked_ldlq0
    torch, ext = _lazy_torch()
    k, n = w_slice.shape
    assert w_slice.dtype == torch.float32 and w_slice.is_cuda
    assert k % HAD_K == 0 and n % HAD_N == 0 and n % 128 == 0
    assert H_data["H"].is_meta or tuple(H_data["H"].shape) == (k, k), \
        f"H shape {tuple(H_data['H'].shape)} != ({k},{k})"
    device = w_slice.device

    quant_args = {"K": BITS, "devices": [0], "apply_out_scales": out_scales,
                  "mcg": True, "sigma_reg": SIGMA_REG}

    # --- reference order: seed -> finalize (su on first use) -> sv ----------
    torch.manual_seed(seed)
    q_fallback, H, L, su, H_diag = finalize_capture_H(H_data, quant_args, verbose=False)
    if H is not None and not H.is_cuda:
        H = H.to(device)                 # working copy for the proxy metric
    if L is not None and not L.is_cuda:
        L = L.to(device)
    if su.device != device:
        su = su.to(device)
    if H_diag is not None and H_diag.device != device:
        H_diag = H_diag.to(device)
    sv = (torch.randn(n, device=device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
    # reference keeps the stored L on CPU between linears sharing this H_data
    if H_data.get("L") is not None and H_data["L"].is_cuda:
        H_data["L"] = H_data["L"].cpu()

    w_orig = w_slice
    apply_out_scales, weight_r, g_scale, su, sv = regularize(
        w_slice.clone(), su, sv, quant_args, H_diag=H_diag, q_fallback=q_fallback)

    wq_reg, encoded = None, None
    if not q_fallback:
        # one-time wiring proof (v2): ldlq with L=0 (no feedback) must equal the
        # independent batched tile quant and the reference fallback loop
        if self_check and not _self_checked_ldlq0:
            wq_a, enc_a = ldlq(weight_r, torch.zeros_like(L), quant_args)
            wq_b, enc_b = trellis_quant_all_tiles(weight_r, quant_args)
            assert torch.equal(enc_a, enc_b), "ldlq(L=0) vs batched: encoded indices differ"
            assert torch.equal(wq_a, wq_b), "ldlq(L=0) vs batched: reconstructed values differ"
            wq_c, enc_c = reference_fallback_quant(weight_r, quant_args)
            assert torch.equal(enc_a, enc_c) and torch.equal(wq_a, wq_c), \
                "batched vs reference fallback loop differ"
            _self_checked_ldlq0 = True
            _self_checked = True
            log("self-check PASSED: ldlq(L=0) == batched tile quant == reference fallback loop", logfile)
        # calibrated main pass deferred to the lockstep pool (LDLQWalk)
    else:
        # reference q_fallback branch (uncalibrated) — cannot trigger under the
        # min-routed floor; kept verbatim, recorded loudly by the caller
        if self_check and not _self_checked:
            wq_a, enc_a = trellis_quant_all_tiles(weight_r, quant_args)
            wq_b, enc_b = reference_fallback_quant(weight_r, quant_args)
            assert torch.equal(enc_a, enc_b), "batched vs reference: encoded indices differ"
            assert torch.equal(wq_a, wq_b), "batched vs reference: reconstructed values differ"
            _self_checked = True
            log("self-check PASSED: batched tile quant is bit-identical to reference fallback_quant", logfile)
            wq_reg, encoded = wq_a, enc_a
        else:
            wq_reg, encoded = trellis_quant_all_tiles(weight_r, quant_args)

    return {"k": k, "n": n, "device": device, "quant_args": quant_args,
            "q_fallback": q_fallback, "H": H, "L": L, "H_diag": H_diag,
            "su": su, "sv": sv, "g_scale": g_scale,
            "apply_out_scales": apply_out_scales, "w_orig": w_orig,
            "weight_r": weight_r, "wq_reg": wq_reg, "encoded": encoded}


def encode_slice_epilogue(ctx):
    """Second half of encode_slice() — verbatim v2 math from the proxy metric
    onward (block_trace, pack/unpack/reconstruct exactness asserts, exact-bpw
    check, weight-domain RT error, output tensors)."""
    torch, ext = _lazy_torch()
    k, n = ctx["k"], ctx["n"]
    weight_r, wq_reg, encoded = ctx["weight_r"], ctx["wq_reg"], ctx["encoded"]
    assert wq_reg is not None and encoded is not None, \
        "epilogue called before the slice's LDLQ walk completed"
    H, su, sv = ctx["H"], ctx["su"], ctx["sv"]
    q_fallback, quant_args = ctx["q_fallback"], ctx["quant_args"]
    w_orig, g_scale = ctx["w_orig"], ctx["g_scale"]
    apply_out_scales = ctx["apply_out_scales"]

    # --- reference proxy error: trace(E H E^T) / trace(W H W^T) -------------
    if not q_fallback:
        E = weight_r - wq_reg
        num = block_trace(E, H)
        den = block_trace(weight_r, H)
        proxy_err = num / max(den, 1e-8)
        del E
    else:
        proxy_err = (weight_r - wq_reg).square().mean().item()

    # --- mandatory encode->decode round-trip exactness (owner mandate (b)) ---
    packed = pack_trellis(encoded, quant_args)
    unpacked = torch.zeros_like(encoded)
    ext.unpack_trellis(unpacked, packed, BITS)
    assert torch.equal(unpacked, encoded), "pack/unpack round-trip mismatch (format bug)"
    w_dec = torch.empty((k, n), dtype=torch.half, device=packed.device)
    ext.reconstruct(w_dec, packed, BITS, True, False)  # mcg codebook
    assert torch.equal(w_dec, wq_reg.half()), \
        "ext.reconstruct != encoder values (codebook/packing bug)"
    expected_bytes = k * n * BITS // 8
    assert packed.numel() * 2 == expected_bytes, "trellis size != exact 3.0 bpw"

    # relative round-trip error in weight domain (selection metric ONLY)
    wq = preapply_had_l(wq_reg, HAD_K)
    wq *= su
    wq = preapply_had_r(wq, HAD_N)
    wq *= sv
    sse = (wq - w_orig).double().pow(2).sum().item()
    ss = w_orig.double().pow(2).sum().item()

    out = {
        "suh": su.flatten().contiguous().to(dtype=torch.half).cpu(),
        "svh": sv.flatten().contiguous().to(dtype=torch.half).cpu(),
        "trellis": packed.cpu(),
    }
    stats = {"sse": sse, "ss": ss, "nmse": sse / max(ss, 1e-30), "g_scale": g_scale,
             "proxy_err": proxy_err, "q_fallback": bool(q_fallback),
             "apply_out_scales": bool(apply_out_scales) if apply_out_scales is not None else None}
    return out, stats


def encode_slice_prologue_pre(w_slice, H_data, seed, out_scales, logfile=None):
    """v3.1: first part of encode_slice_prologue() (v2-verbatim math): seed ->
    finalize (su on first use of a shared H) -> sv draw -> regularize_pre
    (out-scale decision, su/sv scaling, Hadamard transforms; g_scale NOT yet
    applied).  The per-slice g_scale search then runs POOLED across slices
    (g_scale_gss_lockstep) before encode_slice_prologue_post().  RNG contract
    unchanged from v2/v3: every draw happens HERE under torch.manual_seed(seed)
    with seeds derived from (L, E, proj, r); the pooled search and _post
    consume no RNG, so results are independent of pool scheduling."""
    torch, ext = _lazy_torch()
    k, n = w_slice.shape
    assert w_slice.dtype == torch.float32 and w_slice.is_cuda
    assert k % HAD_K == 0 and n % HAD_N == 0 and n % 128 == 0
    assert H_data["H"].is_meta or tuple(H_data["H"].shape) == (k, k), \
        f"H shape {tuple(H_data['H'].shape)} != ({k},{k})"
    device = w_slice.device

    quant_args = {"K": BITS, "devices": [0], "apply_out_scales": out_scales,
                  "mcg": True, "sigma_reg": SIGMA_REG}

    # --- reference order: seed -> finalize (su on first use) -> sv ----------
    torch.manual_seed(seed)
    q_fallback, H, L, su, H_diag = finalize_capture_H(H_data, quant_args, verbose=False)
    if H is not None and not H.is_cuda:
        H = H.to(device)                 # working copy for the proxy metric
    if L is not None and not L.is_cuda:
        L = L.to(device)
    if su.device != device:
        su = su.to(device)
    if H_diag is not None and H_diag.device != device:
        H_diag = H_diag.to(device)
    sv = (torch.randn(n, device=device).sign() + 1e-5).sign().to(torch.float).unsqueeze(0)
    # reference keeps the stored L on CPU between linears sharing this H_data
    if H_data.get("L") is not None and H_data["L"].is_cuda:
        H_data["L"] = H_data["L"].cpu()

    w_orig = w_slice
    apply_out_scales, weight_r, su, sv = regularize_pre(
        w_slice.clone(), su, sv, quant_args, H_diag=H_diag, q_fallback=q_fallback)

    return {"k": k, "n": n, "device": device, "quant_args": quant_args,
            "q_fallback": q_fallback, "H": H, "L": L, "H_diag": H_diag,
            "su": su, "sv": sv, "g_scale": None,
            "apply_out_scales": apply_out_scales, "w_orig": w_orig,
            "weight_r": weight_r, "wq_reg": None, "encoded": None}


def encode_slice_prologue_post(ctx, logfile=None, self_check=False):
    """v3.1: rest of encode_slice_prologue() after the pooled g_scale search:
    apply g_scale (verbatim regularize tail), then the verbatim self-check /
    q_fallback branch.  Returns the same ctx shape v3's prologue returned;
    when ctx["wq_reg"] is None the slice still needs its LDLQ main pass."""
    global _self_checked, _self_checked_ldlq0
    torch, ext = _lazy_torch()
    assert ctx["g_scale"] is not None, \
        "prologue_post called before the pooled g_scale search set ctx['g_scale']"
    weight_r, su = regularize_post(ctx["weight_r"], ctx["su"], ctx["g_scale"])
    ctx["weight_r"], ctx["su"] = weight_r, su
    L, q_fallback, quant_args = ctx["L"], ctx["q_fallback"], ctx["quant_args"]

    wq_reg, encoded = None, None
    if not q_fallback:
        # one-time wiring proof (v2): ldlq with L=0 (no feedback) must equal the
        # independent batched tile quant and the reference fallback loop
        if self_check and not _self_checked_ldlq0:
            wq_a, enc_a = ldlq(weight_r, torch.zeros_like(L), quant_args)
            wq_b, enc_b = trellis_quant_all_tiles(weight_r, quant_args)
            assert torch.equal(enc_a, enc_b), "ldlq(L=0) vs batched: encoded indices differ"
            assert torch.equal(wq_a, wq_b), "ldlq(L=0) vs batched: reconstructed values differ"
            wq_c, enc_c = reference_fallback_quant(weight_r, quant_args)
            assert torch.equal(enc_a, enc_c) and torch.equal(wq_a, wq_c), \
                "batched vs reference fallback loop differ"
            _self_checked_ldlq0 = True
            _self_checked = True
            log("self-check PASSED: ldlq(L=0) == batched tile quant == reference fallback loop", logfile)
        # calibrated main pass deferred to the lockstep pool (LDLQWalk)
    else:
        # reference q_fallback branch (uncalibrated) — cannot trigger under the
        # min-routed floor; kept verbatim, recorded loudly by the caller
        if self_check and not _self_checked:
            wq_a, enc_a = trellis_quant_all_tiles(weight_r, quant_args)
            wq_b, enc_b = reference_fallback_quant(weight_r, quant_args)
            assert torch.equal(enc_a, enc_b), "batched vs reference: encoded indices differ"
            assert torch.equal(wq_a, wq_b), "batched vs reference: reconstructed values differ"
            _self_checked = True
            log("self-check PASSED: batched tile quant is bit-identical to reference fallback_quant", logfile)
            wq_reg, encoded = wq_a, enc_a
        else:
            wq_reg, encoded = trellis_quant_all_tiles(weight_r, quant_args)

    ctx["wq_reg"], ctx["encoded"] = wq_reg, encoded
    return ctx


def lockstep_self_check(walk_ctxs, logfile=None):
    """One-time per worker (mirrors the v2 self-check philosophy): re-run the
    UNTOUCHED v2 ldlq() on every walk's (weight_r, L) — both read-only to the
    walk — and assert the lockstep pool produced bit-identical encodings."""
    global _self_checked_lockstep
    torch, _ = _lazy_torch()
    for ctx in walk_ctxs:
        wq2, enc2 = ldlq(ctx["weight_r"], ctx["L"], ctx["quant_args"])
        assert torch.equal(enc2, ctx["encoded"]), \
            "lockstep vs sequential ldlq: encoded indices differ"
        assert torch.equal(wq2, ctx["wq_reg"]), \
            "lockstep vs sequential ldlq: reconstructed values differ"
    _self_checked_lockstep = True
    log(f"self-check PASSED: lockstep pool ({len(walk_ctxs)} walks) == sequential "
        f"v2 ldlq(), bit-identical", logfile)


def slice_seed(L, E, proj_idx, r):
    return (SEED_BASE * 1000003 + L * 65536 + E * 16 + proj_idx * 4 + r) & 0x7FFFFFFF


# ----------------------------------------------------------------------------
# calibration artifacts (Pass A, capture_hessians.py) -> per-expert Hessians
# ----------------------------------------------------------------------------

def make_H_data(H_sum, count, device):
    """Mimics exllamav3 Linear.init_H_data(True) + capture_H accumulation:
    H = SUM over calibration rows of x x^T (fp32 CUDA [k,k]), count = rows.
    finalize_capture_H() divides by count and mutates H in place, so callers
    must pass an OWNED tensor (clone shared/layer-level Hessians)."""
    return {"H": H_sum, "first_key": "slice", "count": count,
            "finalized": False, "num_total": count, "inf_nan": None,
            "device": device}


class LayerCalib:
    """Loads one layer's capture artifacts (x.bin bf16 [N,6144], ids.bin uint8
    [N,8]) into RAM and serves per-expert routed-token views + fallbacks."""

    def __init__(self, capture_dir, L, logfile=None):
        torch, _ = _lazy_torch()
        import numpy as np
        d = os.path.join(capture_dir, f"layer_{L:03d}")
        man_path = os.path.join(d, "layer_manifest.json")
        assert os.path.exists(man_path), \
            f"layer {L}: capture manifest missing ({man_path}) — run capture_hessians.py --capture first"
        with open(man_path) as f:
            self.manifest = json.load(f)
        self.L = L
        self.tokens = self.manifest["tokens"]
        assert self.manifest["hidden"] == HIDDEN and self.manifest["x_dtype"] == "bfloat16"
        xp = os.path.join(d, "x.bin")
        ip = os.path.join(d, "ids.bin")
        assert os.path.getsize(xp) == self.tokens * HIDDEN * 2, f"{xp}: size mismatch vs manifest"
        assert os.path.getsize(ip) == self.tokens * 8, f"{ip}: size mismatch vs manifest"
        t0 = time.time()
        x_np = np.fromfile(xp, dtype=np.int16).reshape(self.tokens, HIDDEN)
        self.x = torch.from_numpy(x_np).view(torch.bfloat16)          # CPU RAM, ~12.3 GB
        ids = np.fromfile(ip, dtype=np.uint8).reshape(self.tokens, 8)
        # routed-token index: token lists per expert via stable flat argsort
        flat = ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=NUM_EXPERTS)
        self._starts = np.concatenate([[0], np.cumsum(counts)])
        self._token_of = (order // 8).astype(np.int64)
        self.routed_counts = counts.tolist()
        self._H_layer_cpu = None       # lazy (sum, count) for cold-expert fallback
        self._fallback_rows = None
        if logfile is not None:
            log(f"layer {L}: calib loaded — {self.tokens} tokens, routed min/max="
                f"{counts.min()}/{counts.max()}, cold(<{MIN_ROUTED})="
                f"{(counts < MIN_ROUTED).sum()} ({time.time()-t0:.1f}s)", logfile)

    def expert_rows(self, E):
        torch, _ = _lazy_torch()
        return torch.from_numpy(self._token_of[self._starts[E]:self._starts[E + 1]].copy())

    def gather_chunks(self, rows, device, chunk=HCHUNK):
        """Yield fp32 CUDA row blocks of x[rows]."""
        torch, _ = _lazy_torch()
        for s in range(0, rows.numel(), chunk):
            sel = rows[s:s + chunk]
            yield torch.index_select(self.x, 0, sel).to(device).float()

    def all_rows(self):
        torch, _ = _lazy_torch()
        return torch.arange(self.tokens, dtype=torch.int64)

    def fallback_rows(self):
        """Deterministic token subsample for cold-expert down-proj Hessians."""
        torch, _ = _lazy_torch()
        if self._fallback_rows is None:
            g = torch.Generator().manual_seed(SEED_BASE ^ (self.L * 2654435761 & 0x7FFFFFFF))
            n = min(self.tokens, FALLBACK_ROWS)
            self._fallback_rows = torch.randperm(self.tokens, generator=g)[:n].sort().values
        return self._fallback_rows

    def layer_H(self, device):
        """Layer-level H (ALL tokens): SUM x x^T fp32 [6144,6144] + count.
        Cached on CPU; returns a fresh owned CUDA clone per call (finalize
        mutates its input)."""
        torch, _ = _lazy_torch()
        if self._H_layer_cpu is None:
            H = torch.zeros(HIDDEN, HIDDEN, dtype=torch.float32, device=device)
            for xc in self.gather_chunks(self.all_rows(), device):
                H.addmm_(xc.T, xc)
            assert torch.isfinite(H).all(), f"layer {self.L}: non-finite layer H"
            self._H_layer_cpu = H.cpu()
            del H
        return self._H_layer_cpu.to(device), self.tokens


def build_expert_hessians(calib: "LayerCalib", E, Wg, Wu, device, min_routed):
    """Per-expert Hessian data for one MoE expert.

    Returns (hd_gateup, hd_down[4], meta):
      hd_gateup: ONE shared H_data (H = X_e^T X_e, [6144,6144]) for all 8
                 gate/up slices (identical inputs -> shared H/su, reference
                 semantics).  Cold expert -> layer-level H over all tokens.
      hd_down:   4 per-slice H_datas; slice r holds the [512r,512r+512)
                 diagonal block of H_down_e = I_e^T I_e, where
                 I_e = silu(X_e Wg^T) * (X_e Wu^T)  (fp32, dequantized bf16
                 gate/up weights, GLM-5.2 swiglu: silu, no clamp).  Cold
                 expert -> I over the seeded layer token subsample.
    """
    torch, _ = _lazy_torch()
    import torch.nn.functional as Fn
    rows = calib.expert_rows(E)
    n_e = int(rows.numel())
    fallback = n_e < min_routed

    WgT = Wg.float().T.contiguous()      # [6144, 2048]
    WuT = Wu.float().T.contiguous()

    H2048 = torch.zeros(MOE_INTER, MOE_INTER, dtype=torch.float32, device=device)
    if fallback:
        H6144, cnt_gu = calib.layer_H(device)
        i_rows = calib.fallback_rows()
    else:
        H6144 = torch.zeros(HIDDEN, HIDDEN, dtype=torch.float32, device=device)
        cnt_gu = n_e
        i_rows = rows

    cnt_down = int(i_rows.numel())
    for xc in calib.gather_chunks(i_rows, device):
        if not fallback:
            H6144.addmm_(xc.T, xc)
        inter = Fn.silu(xc @ WgT) * (xc @ WuT)      # [m, 2048] fp32
        H2048.addmm_(inter.T, inter)
        del inter, xc
    del WgT, WuT

    assert torch.isfinite(H6144).all() and torch.isfinite(H2048).all(), \
        f"expert {E}: non-finite Hessian accumulation"
    hd_gateup = make_H_data(H6144, cnt_gu, device)
    hd_down = [make_H_data(H2048[r * SLICE:(r + 1) * SLICE,
                                 r * SLICE:(r + 1) * SLICE].clone(), cnt_down, device)
               for r in range(TP)]
    del H2048
    meta = {"routed": n_e, "h_fallback": fallback, "count_gu": cnt_gu,
            "count_down": cnt_down}
    return hd_gateup, hd_down, meta


def free_H_data(*hds):
    for hd in hds:
        for k in ("H", "L", "su", "diag"):
            hd.pop(k, None)


# ----------------------------------------------------------------------------
# NVFP4 dequantization (ModelOpt convention, verified against Luke's shards +
# vLLM modelopt.py: weight_scale_2 = amax/(448*6), dequant = fp4 * fp8scale * ws2,
# packed uint8 = two e2m1 nibbles, LOW nibble first)
# ----------------------------------------------------------------------------

@lru_cache
def _fp4_lut(device):
    torch, _ = _lazy_torch()
    v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return torch.tensor(v + [-x for x in v], dtype=torch.float32, device=device)


def dequant_nvfp4(packed_u8, scale_fp8_u8, ws2: float, device):
    """packed_u8: torch.uint8 [N, K//2] (cpu); scale u8 view of fp8e4m3
    [N, K//16]; ws2 float.  Returns bf16 [N, K] on device."""
    torch, _ = _lazy_torch()
    p = packed_u8.to(device)
    N, Kh = p.shape
    K = Kh * 2
    lo = (p & 0xF).long()
    hi = (p >> 4).long()
    idx = torch.stack((lo, hi), dim=-1).view(N, K)   # low nibble = element 2i
    w = _fp4_lut(device)[idx]                        # f32 [N, K]
    sc = scale_fp8_u8.to(device).view(torch.float8_e4m3fn).float() * ws2
    w = (w.view(N, K // 16, 16) * sc.unsqueeze(-1)).view(N, K)
    # convention guard: catches multiply-vs-divide errors immediately
    rms = w.square().mean().sqrt().item()
    assert 1e-6 < rms < 1.0, f"NVFP4 dequant RMS implausible ({rms}) — scale convention bug"
    return w.to(torch.bfloat16)


# ----------------------------------------------------------------------------
# per-layer encode
# ----------------------------------------------------------------------------

def load_expert_bf16(src: SourceModel, L, E, proj, device, logfile):
    torch, _ = _lazy_torch()
    import numpy as np
    kw = expert_key(L, E, proj, "weight")
    ks = expert_key(L, E, proj, "weight_scale")
    k2 = expert_key(L, E, proj, "weight_scale_2")
    rd_w = src.reader_for(kw, logfile)
    rd_s = src.reader_for(ks, logfile)
    rd_2 = src.reader_for(k2, logfile)
    dt, shape, _, _ = rd_w.tensors[kw]
    assert dt == "U8", (kw, dt)
    exp_shape = (MOE_INTER, HIDDEN // 2) if proj != "down_proj" else (HIDDEN, MOE_INTER // 2)
    assert tuple(shape) == exp_shape, (kw, shape)
    dts, sshape, _, _ = rd_s.tensors[ks]
    assert dts == "F8_E4M3" and tuple(sshape) == (exp_shape[0], exp_shape[1] // 8), (ks, dts, sshape)
    packed = torch.from_numpy(np.frombuffer(rd_w.read_bytes(kw), dtype=np.uint8).copy()).view(*shape)
    scale = torch.from_numpy(np.frombuffer(rd_s.read_bytes(ks), dtype=np.uint8).copy()).view(*sshape)
    ws2 = struct.unpack("<f", rd_2.read_bytes(k2))[0]
    return dequant_nvfp4(packed, scale, ws2, device)


def expert_slices(W_bf16, proj):
    """Yield (rank, slice_f32 (k,n) exl3-convention).  gate/up: [2048,6144]
    N-sliced by output rows; down: [6144,2048] K-sliced by input cols."""
    torch, _ = _lazy_torch()
    if proj != "down_proj":
        assert W_bf16.shape == (MOE_INTER, HIDDEN)
        for r in range(TP):
            sl = W_bf16[r * SLICE:(r + 1) * SLICE, :]        # [512, 6144] (out,in)
            yield r, sl.transpose(0, 1).contiguous().float()  # (k=6144, n=512)
    else:
        assert W_bf16.shape == (HIDDEN, MOE_INTER)
        for r in range(TP):
            sl = W_bf16[:, r * SLICE:(r + 1) * SLICE]        # [6144, 512] (out,in-slice)
            yield r, sl.transpose(0, 1).contiguous().float()  # (k=512, n=6144)


def layer_paths(work, L):
    return (os.path.join(work, f"tr3-layer-{L:03d}.safetensors"),
            os.path.join(work, f"layer-{L:03d}.done.json"))


def layer_done(work, L):
    st_path, done_path = layer_paths(work, L)
    if not (os.path.exists(st_path) and os.path.exists(done_path)):
        return False
    try:
        with open(done_path) as f:
            d = json.load(f)
        return d.get("file_sha256") == sha256_file(st_path)
    except Exception:
        return False


def process_layer(src: SourceModel, work: str, L: int, out_scales, capture_dir: str,
                  min_routed: int, logfile: str, lockstep_n: int = None):
    torch, _ = _lazy_torch()
    t0 = time.time()
    device = "cuda:0"
    st_path, done_path = layer_paths(work, L)
    if lockstep_n is None:
        lockstep_n = 3 * TP

    calib = LayerCalib(capture_dir, L, logfile)
    calib_tokens = calib.tokens
    calib_sha_x = calib.manifest.get("sha256_x")

    stash = {}       # (E, proj, r) -> tensors dict
    err_num = [0.0] * NUM_EXPERTS
    err_den = [0.0] * NUM_EXPERTS
    slice_nmse = {}
    slice_proxy = {}
    routed_count = [0] * NUM_EXPERTS
    h_fallback_experts = []
    q_fallback_slices = []
    out_scales_on = 0

    group_size = max(1, (lockstep_n + 3 * TP - 1) // (3 * TP))  # experts per pool
    log(f"layer {L}: lockstep={lockstep_n} walks in flight "
        f"({group_size} expert(s) per pool), gss_lockstep=on", logfile)

    for E0 in range(0, NUM_EXPERTS, group_size):
        Es = list(range(E0, min(E0 + group_size, NUM_EXPERTS)))
        te = time.time()
        group_hd = []
        pre_list = []    # [(E, proj, r, ctx)] in the v2 slice order
        walk_ctxs = []
        for E in Es:
            W = {proj: load_expert_bf16(src, L, E, proj, device, logfile) for proj in PROJS}
            hd_gu, hd_down, hmeta = build_expert_hessians(
                calib, E, W["gate_proj"], W["up_proj"], device, min_routed)
            routed_count[E] = hmeta["routed"]
            if hmeta["h_fallback"]:
                h_fallback_experts.append(E)

            # fixed slice order: gate r0..3 (finalizes the shared 6144-H, drawing
            # su under gate.rank0's seed), up r0..3 (reuse), down r0..3 (own H);
            # _pre prologues stay strictly sequential per slice (seed/su/sv/
            # regularize_pre = v2 verbatim); the per-slice g_scale searches then
            # run POOLED (v3.1) and the LDLQ main passes run POOLED (v3)
            for pi, proj in enumerate(PROJS):
                for r, sl in expert_slices(W[proj], proj):
                    hd = hd_gu if proj != "down_proj" else hd_down[r]
                    ctx = encode_slice_prologue_pre(
                        sl, hd, slice_seed(L, E, pi, r), out_scales, logfile=logfile)
                    pre_list.append((E, proj, r, ctx))
            del W
            group_hd.append((hd_gu, hd_down))

        # v3.1 pooled golden-section g_scale search across the group's slices
        # (one-time pooled-vs-sequential equality assert on the first pool)
        g_scale_gss_lockstep([c for _, _, _, c in pre_list], logfile, self_check=True)

        ordered = []     # [(E, proj, r, ctx)] in the v2 slice order
        for E, proj, r, ctx in pre_list:
            ctx = encode_slice_prologue_post(ctx, logfile=logfile, self_check=True)
            ordered.append((E, proj, r, ctx))
            if ctx["wq_reg"] is None:
                walk_ctxs.append(ctx)
        del pre_list

        # lockstep LDLQ main pass over every pending walk in the group
        walks = [LDLQWalk(c) for c in walk_ctxs]
        lockstep_ldlq(walks, lockstep_n, logfile)
        if walks and not _self_checked_lockstep:
            lockstep_self_check(walk_ctxs, logfile)
        del walks

        for E, proj, r, ctx in ordered:
            out, stats = encode_slice_epilogue(ctx)
            stash[(E, proj, r)] = out
            err_num[E] += stats["sse"]
            err_den[E] += stats["ss"]
            key = f"{E}.{proj}.rank{r}"
            slice_nmse[key] = stats["nmse"]
            slice_proxy[key] = stats["proxy_err"]
            if stats["q_fallback"]:
                q_fallback_slices.append(key)   # should stay empty (loud record)
            if stats["apply_out_scales"]:
                out_scales_on += 1
            ctx.clear()
        for hd_gu, hd_down in group_hd:
            free_H_data(hd_gu, *hd_down)
        del ordered, walk_ctxs, group_hd

        dt_e = (time.time() - te) / len(Es)
        for E in Es:
            if E % 16 == 15:
                rel = err_num[E] / max(err_den[E], 1e-30)
                log(f"layer {L} expert {E:3d} done ({dt_e:.1f}s/expert, "
                    f"routed {routed_count[E]}, last nmse {rel:.3e})", logfile)

    del calib
    if q_fallback_slices:
        log(f"layer {L} WARNING: {len(q_fallback_slices)} slices hit the reference "
            f"q_fallback (uncalibrated) branch: {q_fallback_slices[:4]}...", logfile)

    rel_err = [err_num[E] / max(err_den[E], 1e-30) for E in range(NUM_EXPERTS)]
    order = sorted(range(NUM_EXPERTS), key=lambda E: rel_err[E], reverse=True)
    keep = sorted(order[:KEEP_NVFP4])          # highest trellis RT-error stay NVFP4
    tail = [E for E in range(NUM_EXPERTS) if E not in set(keep)]
    assert len(tail) == NUM_EXPERTS - KEEP_NVFP4

    entries = []
    mcg_bytes = struct.pack("<I", MCG_MULT)    # stored as I32 scalar (uint32 bit pattern)
    for E in tail:
        for proj in PROJS:
            for r in range(TP):
                t = stash[(E, proj, r)]
                base = f"model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}"
                tr = t["trellis"]
                entries.append((f"{base}.suh", "F16", tuple(t["suh"].shape),
                                t["suh"].numpy().tobytes()))
                entries.append((f"{base}.svh", "F16", tuple(t["svh"].shape),
                                t["svh"].numpy().tobytes()))
                entries.append((f"{base}.trellis", "I16", tuple(tr.shape),
                                tr.numpy().tobytes()))
                entries.append((f"{base}.mcg", "I32", (), mcg_bytes))

    _, file_sha = write_safetensors(st_path, entries, metadata={"format": "pt"})
    done = {
        "layer": L,
        "bits": BITS,
        "codebook": "mcg",
        "hessian": "ldlq-calibrated",
        "sigma_reg": SIGMA_REG,
        "min_routed": min_routed,
        "capture": {
            "dir": capture_dir,
            "tokens": calib_tokens,
            "sha256_x": calib_sha_x,
        },
        "expert_routed_count": routed_count,
        "experts_layer_h_fallback": h_fallback_experts,
        "q_fallback_slices": q_fallback_slices,
        "slices_with_out_scales": out_scales_on,
        "su_scheme": "gate/up slices share one H+su per expert (su drawn under "
                     "gate.rank0 seed at first finalize, reference semantics); "
                     "down slices each own H+su under their slice seed; sv per slice",
        "tp": TP,
        "keep_nvfp4": keep,
        "tail_tr3": tail,
        "expert_rel_rt_mse": rel_err,
        "slice_nmse": slice_nmse,
        "slice_proxy_err": slice_proxy,
        "seed_base": SEED_BASE,
        "lockstep": lockstep_n,
        "gss_lockstep": True,
        "exllamav3": "0.0.43",
        "file": os.path.basename(st_path),
        "file_sha256": file_sha,
        "encode_seconds": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = done_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(done, f, indent=1)
    os.replace(tmp, done_path)
    log(f"layer {L} COMPLETE in {done['encode_seconds']}s; keep_nvfp4={keep[:8]}...", logfile)


# ----------------------------------------------------------------------------
# worker + orchestrator
# ----------------------------------------------------------------------------

def parse_layers(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    for L in out:
        assert FIRST_MOE_LAYER <= L < NUM_LAYERS, f"layer {L} is not a main MoE layer"
    return sorted(set(out))


def check_capture(capture_dir: str, layers) -> None:
    """CALIBRATION IS MANDATORY: refuse to encode without a complete capture."""
    missing = []
    for L in layers:
        d = os.path.join(capture_dir, f"layer_{L:03d}")
        if not (os.path.exists(os.path.join(d, "layer_manifest.json"))
                and os.path.exists(os.path.join(d, "x.bin"))
                and os.path.exists(os.path.join(d, "ids.bin"))):
            missing.append(L)
    if missing:
        raise SystemExit(
            f"capture incomplete in {capture_dir}: layers {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''} — run capture_hessians.py --capture "
            f"first (owner mandate: no identity-Hessian encode)")


def worker_main(args):
    logfile = os.path.join(args.work, "logs", f"worker{args.worker_rank}.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    torch, ext = _lazy_torch()
    lockstep_n = resolve_lockstep(args.lockstep)
    log(f"worker {args.worker_rank} up on {torch.cuda.get_device_name(0)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"lockstep={lockstep_n})", logfile)
    src = SourceModel(args.src)
    layers = parse_layers(args.layers)
    mine = layers[args.worker_rank::args.workers]
    check_capture(args.capture_dir, mine)
    out_scales = {"auto": None, "always": True, "never": False}[args.out_scales]
    for L in mine:
        if layer_done(args.work, L):
            log(f"layer {L} already done, skipping", logfile)
            continue
        process_layer(src, args.work, L, out_scales, args.capture_dir,
                      args.min_routed, logfile, lockstep_n)
    log(f"worker {args.worker_rank} finished {len(mine)} layers", logfile)


def orchestrate_encode(args):
    os.makedirs(os.path.join(args.work, "logs"), exist_ok=True)
    logfile = os.path.join(args.work, "logs", "driver.log")
    layers = parse_layers(args.layers)
    check_capture(args.capture_dir, layers)
    todo = [L for L in layers if not layer_done(args.work, L)]
    log(f"encode: {len(layers)} layers requested, {len(todo)} to do, "
        f"{args.workers} workers, bits={BITS}, keep={KEEP_NVFP4}, tp={TP}, "
        f"hessian=ldlq-calibrated (capture {args.capture_dir}, "
        f"min_routed={args.min_routed}), lockstep={resolve_lockstep(args.lockstep)}", logfile)
    if not todo:
        log("nothing to do", logfile)
        return
    procs = []
    for r in range(args.workers):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(r % args.gpus)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--worker-rank", str(r),
               "--workers", str(args.workers),
               "--src", args.src, "--work", args.work,
               "--layers", args.layers, "--out-scales", args.out_scales,
               "--capture-dir", args.capture_dir,
               "--min-routed", str(args.min_routed),
               "--lockstep", str(args.lockstep)]
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
        log(f"spawned worker {r} pid {p.pid} on GPU {r}", logfile)
    rc = 0
    try:
        for r, p in enumerate(procs):
            code = p.wait()
            log(f"worker {r} exited with code {code}", logfile)
            rc |= code
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        raise
    done = [L for L in layers if layer_done(args.work, L)]
    log(f"encode pass done: {len(done)}/{len(layers)} layers complete", logfile)
    sys.exit(1 if rc else 0)


# ----------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------

def aux_files(src_dir: str):
    """Every non-weight file in the source repo is carried byte-exact —
    everything except the shards, the index, and config.json (rewritten
    with the hybrid addendum)."""
    skip = {"model.safetensors.index.json", "config.json", "MANIFEST.sha256"}
    out = []
    for fn in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, fn)
        if not os.path.isfile(p) or fn in skip or fn.endswith(".safetensors"):
            continue
        out.append(fn)
    return out


def assemble(args):
    os.makedirs(args.out, exist_ok=True)
    logfile = os.path.join(args.work, "logs", "assemble.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    src = SourceModel(args.src)

    layers = src.moe_layers()
    dones = {}
    for L in layers:
        st_path, done_path = layer_paths(args.work, L)
        assert layer_done(args.work, L), f"layer {L} not encoded/verified; run --encode first"
        with open(done_path) as f:
            dones[L] = json.load(f)
    log(f"assemble: all {len(layers)} MoE layers present", logfile)

    # ---- shard plan: one output shard per layer + embed + head -------------
    all_src = sorted(src.weight_map.keys())
    dropped = set()
    for L in layers:
        for E in dones[L]["tail_tr3"]:
            for proj in PROJS:
                for sfx in ("weight", "weight_scale", "weight_scale_2"):
                    dropped.add(expert_key(L, E, proj, sfx))

    def carried_for_prefix(pfx):
        return [k for k in all_src if k.startswith(pfx) and k not in dropped]

    shard_jobs = []   # (out_name, [("src", key) | ("tr3", L, key)])
    shard_jobs.append(("model-embed.safetensors",
                       [("src", "model.embed_tokens.weight")]))
    head = [k for k in ("lm_head.weight", "model.norm.weight") if k in src.weight_map]
    shard_jobs.append(("model-head.safetensors", [("src", k) for k in head]))
    for L in range(NUM_LAYERS + 1):  # 0..78 incl. MTP layer 78
        items = [("src", k) for k in carried_for_prefix(f"model.layers.{L}.")]
        if L in dones:
            st_path, _ = layer_paths(args.work, L)
            rd = STReader(st_path)
            items += [("tr3", L, k) for k in sorted(rd.tensors.keys())]
        shard_jobs.append((f"model-layer-{L:03d}.safetensors", items))

    accounted = set()
    for _, items in shard_jobs:
        for it in items:
            if it[0] == "src":
                accounted.add(it[1])
    missing = [k for k in all_src if k not in dropped and k not in accounted]
    assert not missing, f"unaccounted source tensors: {missing[:5]} (+{len(missing)-5 if len(missing)>5 else 0})"

    # ---- write shards with per-tensor byte-exact verification --------------
    weight_map = {}
    total_size = 0
    from multiprocessing import Pool

    job_args = [(args.src, args.work, args.out, name, items) for name, items in shard_jobs]
    with Pool(processes=min(args.io_workers, len(job_args))) as pool:
        results = pool.map(_assemble_shard, job_args)

    for (name, _), (wm_part, nbytes) in zip(shard_jobs, results):
        weight_map.update({k: name for k in wm_part})
        total_size += nbytes
    log(f"assemble: wrote {len(shard_jobs)} shards, {total_size/2**30:.1f} GiB tensor payload, "
        f"all carried tensors verified byte-exact", logfile)

    # ---- config.json with hybrid addendum (quantization_config untouched) --
    hset = {d.get("hessian") for d in dones.values()}
    assert hset == {"ldlq-calibrated"}, f"mixed/legacy hessian modes in done JSONs: {hset}"
    cap_infos = {json.dumps(d["capture"], sort_keys=True) for d in dones.values()}
    assert len(cap_infos) == 1, "layers were encoded against different captures"
    cap = dones[layers[0]]["capture"]
    calibration = {
        "capture_dir": cap["dir"],
        "tokens_per_layer": cap["tokens"],
        "min_routed_floor": dones[layers[0]]["min_routed"],
        "sigma_reg": dones[layers[0]]["sigma_reg"],
        "layer_h_fallback_experts_total": sum(len(d["experts_layer_h_fallback"])
                                              for d in dones.values()),
        "q_fallback_slices_total": sum(len(d["q_fallback_slices"])
                                       for d in dones.values()),
    }
    cap_manifest = os.path.join(cap["dir"], "capture_manifest.json")
    if os.path.exists(cap_manifest):
        with open(cap_manifest) as f:
            cm = json.load(f)
        calibration.update({
            "corpus_path": cm.get("corpus_path"),
            "corpus_sha256": cm.get("corpus_sha256"),
            "corpus_seed": cm.get("seed"),
            "num_samples": cm.get("num_samples"),
            "total_tokens": cm.get("total_tokens"),
        })
        import shutil
        shutil.copyfile(cap_manifest, os.path.join(args.out, "calibration_manifest.json"))
        log("copied calibration_manifest.json into checkpoint", logfile)

    cfg = dict(src.config)
    cfg["hybrid_tr3_tail"] = {
        "producer": "encode_tr3.py",
        "source_repo": "lukealonso/GLM-5.2-NVFP4",
        "format": "exl3-trellis",
        "bits": float(BITS),
        "codebook": "mcg",
        "mcg_multiplier": MCG_MULT,
        "hessian": "ldlq-calibrated",
        "calibration": calibration,
        "exllamav3_version": "0.0.43",
        "moe_layers": [layers[0], layers[-1]],
        "experts_per_layer": NUM_EXPERTS,
        "nvfp4_keep_per_layer": KEEP_NVFP4,
        "tr3_tail_per_layer": NUM_EXPERTS - KEEP_NVFP4,
        "tp": TP,
        "slicing": {
            "gate_proj": "N-sliced: rank r = output rows [512r, 512r+512) of [2048,6144]; "
                         "H = expert-routed X_e^T X_e [6144,6144], shared with up_proj",
            "up_proj": "N-sliced: rank r = output rows [512r, 512r+512) of [2048,6144]; "
                       "H shared with gate_proj (same input)",
            "down_proj": "K-sliced: rank r = input cols [512r, 512r+512) of [6144,2048]; "
                         "H = [512r,512r+512) diagonal block of I_e^T I_e "
                         "(I_e = silu(X_e Wg^T) * (X_e Wu^T), dequantized gate/up) — "
                         "slice-local LDLQ",
        },
        "tensor_schema": "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}",
        "selection": "per layer, 64 experts with highest pooled relative trellis RT-MSE "
                     "(measured under the calibrated LDLQ encode) kept NVFP4",
        "tier_bitmap": "tier_bitmap.json",
        "seed_base": SEED_BASE,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- index + audit ------------------------------------------------------
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=1, sort_keys=True)
    audit_index(args.out, logfile)

    # ---- tier bitmap ---------------------------------------------------------
    tb = {str(L): {"keep_nvfp4": dones[L]["keep_nvfp4"],
                   "expert_rel_rt_mse": dones[L]["expert_rel_rt_mse"]} for L in layers}
    with open(os.path.join(args.out, "tier_bitmap.json"), "w") as f:
        json.dump(tb, f)

    # ---- aux files byte-exact ------------------------------------------------
    for fn in aux_files(args.src):
        sp = os.path.join(args.src, fn)
        dp = os.path.join(args.out, fn)
        with open(sp, "rb") as fi, open(dp, "wb") as fo:
            while True:
                b = fi.read(CHUNK)
                if not b:
                    break
                fo.write(b)
        assert sha256_file(sp) == sha256_file(dp), f"aux copy mismatch: {fn}"
        log(f"copied {fn} (byte-exact verified)", logfile)

    # ---- manifest -------------------------------------------------------------
    names = sorted(fn for fn in os.listdir(args.out)
                   if os.path.isfile(os.path.join(args.out, fn)) and fn != "MANIFEST.sha256")
    with open(os.path.join(args.out, "MANIFEST.sha256"), "w") as f:
        for fn in names:
            f.write(f"{sha256_file(os.path.join(args.out, fn))}  {fn}\n")
    log(f"assemble COMPLETE -> {args.out} ({len(names)} files + MANIFEST.sha256)", logfile)


def _assemble_shard(job):
    """Pool worker: write one output shard, then re-read it and assert every
    tensor's payload sha256 equals the source payload sha256."""
    src_dir, work, out, name, items = job
    src = SourceModel(src_dir)
    readers = {}

    def rd(path):
        if path not in readers:
            readers[path] = STReader(path)
        return readers[path]

    entries = []
    src_sha = {}
    for it in items:
        if it[0] == "src":
            key = it[1]
            r = rd(os.path.join(src_dir, src.weight_map[key]))
            dt, shape, _, _ = r.tensors[key]
            entries.append((key, dt, shape, (lambda r=r, key=key: r.read_bytes(key))))
        else:
            _, L, key = it
            st_path, _ = layer_paths(work, L)
            r = rd(st_path)
            dt, shape, _, _ = r.tensors[key]
            entries.append((key, dt, shape, (lambda r=r, key=key: r.read_bytes(key))))

    out_path = os.path.join(out, name)
    written_sha, _ = write_safetensors(out_path, entries, metadata={"format": "pt"})

    # source-side hashes (streamed straight from source files)
    for it in items:
        if it[0] == "src":
            key = it[1]
            src_sha[key] = rd(os.path.join(src_dir, src.weight_map[key])).sha256_stream(key)
        else:
            _, L, key = it
            st_path, _ = layer_paths(work, L)
            src_sha[key] = rd(st_path).sha256_stream(key)

    # re-read the written file and verify byte-exactness per tensor
    check = STReader(out_path)
    nbytes = 0
    for key, sha in src_sha.items():
        dt, shape, s, e = check.tensors[key]
        got = sha256_range(out_path, s, e)
        assert got == sha, f"BYTE-EXACT VERIFY FAILED: {name}:{key}"
        assert written_sha[key] == sha, f"write-path hash mismatch: {name}:{key}"
        nbytes += e - s
    return list(src_sha.keys()), nbytes


def audit_index(out: str, logfile):
    with open(os.path.join(out, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wm = index["weight_map"]
    by_file = {}
    for k, fn in wm.items():
        by_file.setdefault(fn, set()).add(k)
    for fn, keys in by_file.items():
        p = os.path.join(out, fn)
        assert os.path.exists(p), f"index maps to missing shard {fn}"
        rd = STReader(p)
        in_file = set(rd.tensors.keys())
        assert keys == in_file, (f"index/shard mismatch in {fn}: "
                                 f"missing={list(keys-in_file)[:3]} extra={list(in_file-keys)[:3]}")
    log(f"index audit OK: {len(wm)} tensors across {len(by_file)} shards, bijective", logfile)


# ----------------------------------------------------------------------------
# upload
# ----------------------------------------------------------------------------

def upload(args):
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN env var not set (never pass it on the command line)"
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    for attempt in range(1, 6):
        try:
            api.upload_large_folder(repo_id=args.repo, folder_path=args.out,
                                    repo_type="model")
            print(f"upload complete: https://huggingface.co/{args.repo} (private)")
            return
        except Exception as e:
            print(f"upload attempt {attempt}/5 failed: {type(e).__name__}: {e}")
            time.sleep(min(60 * attempt, 300))
    raise SystemExit("upload failed after 5 attempts")


# ----------------------------------------------------------------------------
# smoke test (random tensors + synthetic-activation Hessians, no source model
# or capture needed — exercises finalize_capture_H/block_ldl/ldlq end to end
# plus both bit-exactness self-checks)
# ----------------------------------------------------------------------------

def smoke(args):
    torch, ext = _lazy_torch()
    log(f"smoke: torch {torch.__version__}, gpu {torch.cuda.get_device_name(0)}")
    torch.manual_seed(99)
    for (k, n) in ((HIDDEN, SLICE), (SLICE, HIDDEN)):
        w = torch.randn(k, n, device="cuda:0", dtype=torch.float32) * 0.02
        # synthetic calibration activations (mildly anisotropic, like real x)
        X = torch.randn(8192, k, device="cuda:0", dtype=torch.float32)
        X *= (1.0 + torch.rand(k, device="cuda:0")).unsqueeze(0)
        H_data = make_H_data(X.T @ X, X.shape[0], "cuda:0")
        del X
        t0 = time.time()
        out, stats = encode_slice(w, H_data, 1234, None, self_check=True)
        dt = time.time() - t0
        assert not stats["q_fallback"], "smoke unexpectedly hit the uncalibrated branch"
        assert out["trellis"].numel() * 2 == k * n * BITS // 8
        free_H_data(H_data)
        log(f"smoke ({k}x{n}): OK (ldlq), nmse={stats['nmse']:.4e}, "
            f"proxy_err={stats['proxy_err']:.4e}, g_scale={stats['g_scale']:.3f}, "
            f"{dt:.2f}s (round-trip exactness asserted)")
    # shared-H reuse path (gate/up semantics): second encode on a finalized H
    X = torch.randn(4096, HIDDEN, device="cuda:0", dtype=torch.float32)
    hd = make_H_data(X.T @ X, X.shape[0], "cuda:0")
    del X
    w1 = torch.randn(HIDDEN, SLICE, device="cuda:0", dtype=torch.float32) * 0.02
    w2 = torch.randn(HIDDEN, SLICE, device="cuda:0", dtype=torch.float32) * 0.02
    _, s1 = encode_slice(w1, hd, 555, None)
    assert hd["finalized"]
    _, s2 = encode_slice(w2, hd, 556, None)
    free_H_data(hd)
    log(f"smoke shared-H reuse: OK (nmse {s1['nmse']:.3e} / {s2['nmse']:.3e})")

    # --- v3: lockstep-vs-sequential byte-equality on synthetic H -------------
    # (a) mixed geometries co-stepped (384-step gate/up-like walk + 32-step
    #     down-like walk, uneven completion); (b) shared-H pair (su drawn once
    #     under the first slice's seed, reference shared-qmap semantics).
    torch.manual_seed(101)
    cases = []           # (w, H_sum clone A, clone B, seed)
    for i, (k, n) in enumerate(((HIDDEN, SLICE), (SLICE, HIDDEN))):
        w = torch.randn(k, n, device="cuda:0", dtype=torch.float32) * 0.02
        X = torch.randn(8192, k, device="cuda:0", dtype=torch.float32)
        X *= (1.0 + torch.rand(k, device="cuda:0")).unsqueeze(0)
        Hs = X.T @ X
        del X
        cases.append((w, Hs, 4321 + i))
    ws = torch.randn(HIDDEN, SLICE, device="cuda:0", dtype=torch.float32) * 0.02
    ws2 = torch.randn(HIDDEN, SLICE, device="cuda:0", dtype=torch.float32) * 0.02
    Xs = torch.randn(4096, HIDDEN, device="cuda:0", dtype=torch.float32)
    Hshared = Xs.T @ Xs
    del Xs
    # sequential pass (verbatim v2 encode_slice)
    hds = [make_H_data(Hs.clone(), 8192, "cuda:0") for _, Hs, _ in cases]
    hd_sh = make_H_data(Hshared.clone(), 4096, "cuda:0")
    outs_a = [encode_slice(w, hd, seed, None)[0]
              for (w, _, seed), hd in zip(cases, hds)]
    outs_a.append(encode_slice(ws, hd_sh, 777, None)[0])
    outs_a.append(encode_slice(ws2, hd_sh, 778, None)[0])
    for hd in hds:
        free_H_data(hd)
    free_H_data(hd_sh)
    # lockstep pass (v3 pipeline) on identical inputs
    hds = [make_H_data(Hs.clone(), 8192, "cuda:0") for _, Hs, _ in cases]
    hd_sh = make_H_data(Hshared.clone(), 4096, "cuda:0")
    ctxs = [encode_slice_prologue(w, hd, seed, None)
            for (w, _, seed), hd in zip(cases, hds)]
    ctxs.append(encode_slice_prologue(ws, hd_sh, 777, None))
    ctxs.append(encode_slice_prologue(ws2, hd_sh, 778, None))
    walks = [LDLQWalk(c) for c in ctxs if c["wq_reg"] is None]
    lockstep_ldlq(walks, 12)
    outs_b = [encode_slice_epilogue(c)[0] for c in ctxs]
    for hd in hds:
        free_H_data(hd)
    free_H_data(hd_sh)
    names = ["6144x512", "512x6144", "sharedH-1", "sharedH-2"]
    for name, oa, ob in zip(names, outs_a, outs_b):
        for tname in ("trellis", "suh", "svh"):
            assert oa[tname].numpy().tobytes() == ob[tname].numpy().tobytes(), \
                f"smoke lockstep mismatch: {name} {tname}"
    log(f"smoke lockstep: OK — {len(names)} walks (mixed geometry + shared-H) "
        f"lockstep == sequential encode_slice, byte-identical")

    # --- v3.1: pooled-GSS pipeline byte-equality on the same inputs ----------
    # (pre-prologues -> g_scale_gss_lockstep [with the one-time pooled-vs-
    # sequential g_scale self-check] -> post-prologues -> lockstep LDLQ ->
    # epilogues) vs the sequential v2 encode_slice outputs above.
    hds = [make_H_data(Hs.clone(), 8192, "cuda:0") for _, Hs, _ in cases]
    hd_sh = make_H_data(Hshared.clone(), 4096, "cuda:0")
    pres = [encode_slice_prologue_pre(w, hd, seed, None)
            for (w, _, seed), hd in zip(cases, hds)]
    pres.append(encode_slice_prologue_pre(ws, hd_sh, 777, None))
    pres.append(encode_slice_prologue_pre(ws2, hd_sh, 778, None))
    g_scale_gss_lockstep(pres, None, self_check=True)
    ctxs = [encode_slice_prologue_post(c) for c in pres]
    walks = [LDLQWalk(c) for c in ctxs if c["wq_reg"] is None]
    lockstep_ldlq(walks, 12)
    outs_c = [encode_slice_epilogue(c)[0] for c in ctxs]
    for hd in hds:
        free_H_data(hd)
    free_H_data(hd_sh)
    for name, oa, oc in zip(names, outs_a, outs_c):
        for tname in ("trellis", "suh", "svh"):
            assert oa[tname].numpy().tobytes() == oc[tname].numpy().tobytes(), \
                f"smoke v3.1 pooled-GSS mismatch: {name} {tname}"
    log(f"smoke gss lockstep: OK — {len(names)} walks (pooled GSS pipeline) == "
        f"sequential encode_slice, byte-identical")
    log("smoke PASSED")


# ----------------------------------------------------------------------------
# v3 oracle (HARD deployment gate) + bench — both read-only wrt --work
# ----------------------------------------------------------------------------

def oracle(args):
    """HARD GATE: byte-equality of the deployed v2 sequential encode_slice()
    vs the v3 lockstep-LDLQ pipeline vs the v3.1 pooled-GSS pipeline on REAL
    slices — dequantized source experts + real captured Hessians of ONE layer
    (pick a layer the running encode has not reached), spanning both
    geometries and the shared-H gate/up semantics.  The expert's raw Hessian
    sums are built once and CLONED per pass so all three passes see
    bit-identical inputs.  PASS requires v3.1 == v3 == v2 on trellis+suh+svh
    bytes AND exact g_scale equality.  Writes nothing except --oracle-log."""
    torch, _ = _lazy_torch()
    olog = args.oracle_log
    layers = parse_layers(args.layers)
    assert len(layers) == 1, "--oracle expects exactly one layer, e.g. --layers 72"
    L = layers[0]
    check_capture(args.capture_dir, [L])
    lockstep_n = resolve_lockstep(args.lockstep)
    out_scales = {"auto": None, "always": True, "never": False}[args.out_scales]
    experts = [int(x) for x in args.oracle_experts.split(",")]
    src = SourceModel(args.src)
    log(f"ORACLE start (v3.1): layer {L}, experts {experts}, lockstep={lockstep_n}, "
        f"gpu {torch.cuda.get_device_name(0)}, torch {torch.__version__}", olog)
    calib = LayerCalib(args.capture_dir, L, olog)
    device = "cuda:0"
    total = passed = 0
    t_v2 = t_v3 = t_v31 = 0.0
    for E in experts:
        W = {proj: load_expert_bf16(src, L, E, proj, device, olog) for proj in PROJS}
        hd0_gu, hd0_down, hmeta = build_expert_hessians(
            calib, E, W["gate_proj"], W["up_proj"], device, args.min_routed)
        log(f"ORACLE E{E}: routed={hmeta['routed']} h_fallback={hmeta['h_fallback']} "
            f"(raw H sums built once, cloned per pass)", olog)
        snap_gu = (hd0_gu["H"].clone(), hd0_gu["count"])
        snap_down = [(hd["H"].clone(), hd["count"]) for hd in hd0_down]
        free_H_data(hd0_gu, *hd0_down)
        slices = []
        for pi, proj in enumerate(PROJS):
            for r, sl in expert_slices(W[proj], proj):
                slices.append((pi, proj, r, sl))
        del W

        # ---- pass 1: v2 sequential encode_slice (the deployed math) --------
        hd_gu = make_H_data(snap_gu[0].clone(), snap_gu[1], device)
        hd_down = [make_H_data(h.clone(), c, device) for h, c in snap_down]
        v2_out = {}
        torch.cuda.synchronize()
        t1 = time.time()
        for pi, proj, r, sl in slices:
            hd = hd_gu if proj != "down_proj" else hd_down[r]
            v2_out[(proj, r)] = encode_slice(sl, hd, slice_seed(L, E, pi, r),
                                             out_scales, logfile=None, self_check=False)
        torch.cuda.synchronize()
        t_v2 += time.time() - t1
        free_H_data(hd_gu, *hd_down)

        # ---- pass 2: v3 lockstep-LDLQ pipeline (the RUNNING encode's path) ---
        hd_gu = make_H_data(snap_gu[0].clone(), snap_gu[1], device)
        hd_down = [make_H_data(h.clone(), c, device) for h, c in snap_down]
        torch.cuda.synchronize()
        t1 = time.time()
        ctxs = []
        for pi, proj, r, sl in slices:
            hd = hd_gu if proj != "down_proj" else hd_down[r]
            ctxs.append((proj, r, encode_slice_prologue(
                sl, hd, slice_seed(L, E, pi, r), out_scales,
                logfile=None, self_check=False)))
        walks = [LDLQWalk(c) for _, _, c in ctxs if c["wq_reg"] is None]
        lockstep_ldlq(walks, lockstep_n, None)
        v3_out = {}
        for proj, r, ctx in ctxs:
            v3_out[(proj, r)] = encode_slice_epilogue(ctx)
        torch.cuda.synchronize()
        t_v3 += time.time() - t1
        free_H_data(hd_gu, *hd_down)
        del walks, ctxs

        # ---- pass 3: v3.1 pooled-GSS pipeline on identical inputs -----------
        hd_gu = make_H_data(snap_gu[0].clone(), snap_gu[1], device)
        hd_down = [make_H_data(h.clone(), c, device) for h, c in snap_down]
        torch.cuda.synchronize()
        t1 = time.time()
        pres = []
        for pi, proj, r, sl in slices:
            hd = hd_gu if proj != "down_proj" else hd_down[r]
            pres.append((proj, r, encode_slice_prologue_pre(
                sl, hd, slice_seed(L, E, pi, r), out_scales, logfile=None)))
        g_scale_gss_lockstep([c for _, _, c in pres], None, self_check=False)
        ctxs31 = [(proj, r, encode_slice_prologue_post(c)) for proj, r, c in pres]
        walks = [LDLQWalk(c) for _, _, c in ctxs31 if c["wq_reg"] is None]
        lockstep_ldlq(walks, lockstep_n, None)
        v31_out = {}
        for proj, r, ctx in ctxs31:
            v31_out[(proj, r)] = encode_slice_epilogue(ctx)
        torch.cuda.synchronize()
        t_v31 += time.time() - t1
        free_H_data(hd_gu, *hd_down)
        del walks, ctxs31, pres

        # ---- byte comparison: v3.1 vs v3 vs v2 -------------------------------
        for pi, proj, r, sl in slices:
            (o2, s2) = v2_out[(proj, r)]
            (o3, s3) = v3_out[(proj, r)]
            (o31, s31) = v31_out[(proj, r)]
            eq_31_2 = all(o31[t].numpy().tobytes() == o2[t].numpy().tobytes()
                          for t in ("trellis", "suh", "svh"))
            eq_31_3 = all(o31[t].numpy().tobytes() == o3[t].numpy().tobytes()
                          for t in ("trellis", "suh", "svh"))
            g_eq = (s31["g_scale"] == s3["g_scale"] == s2["g_scale"])
            ok = eq_31_2 and eq_31_3 and g_eq
            sha = hashlib.sha256(o31["trellis"].numpy().tobytes()).hexdigest()[:16]
            total += 1
            passed += int(ok)
            log(f"ORACLE layer {L} E{E} {proj}.rank{r} k={sl.shape[0]} n={sl.shape[1]}: "
                f"{'BYTE-IDENTICAL v3.1==v3==v2' if ok else '*** MISMATCH ***'} "
                f"[trellis sha256[:16]={sha}] "
                f"g v2={s2['g_scale']:.6f} v3={s3['g_scale']:.6f} v3.1={s31['g_scale']:.6f} "
                f"g-exact={g_eq} nmse v2={s2['nmse']:.9e} v3.1={s31['nmse']:.9e} "
                f"proxy v2={s2['proxy_err']:.9e} v3.1={s31['proxy_err']:.9e}", olog)
        del slices, v2_out, v3_out, v31_out, snap_gu, snap_down
        torch.cuda.empty_cache()
    verdict = "PASS" if passed == total else "FAIL"
    log(f"ORACLE {verdict}: {passed}/{total} slices byte-identical v3.1==v3==v2 "
        f"(trellis+suh+svh, g_scale exact-equal), both geometries + shared-H "
        f"semantics; identical inputs took v2 sequential {t_v2:.1f}s vs v3 "
        f"lockstep {t_v3:.1f}s vs v3.1 pooled-GSS {t_v31:.1f}s (contended timings)", olog)
    if passed != total:
        sys.exit(2)


def bench(args):
    """Short v3 throughput probe (no --work writes): run --bench-experts
    experts of one layer through the FULL v3 per-expert path (shard load +
    H build + 12 prologues + lockstep LDLQ + epilogues incl. every exactness
    assert) and report s/expert.  Safe to run beside a live encode; timings
    are then worst-case contended."""
    torch, _ = _lazy_torch()
    layers = parse_layers(args.layers)
    assert len(layers) == 1, "--bench expects exactly one layer, e.g. --layers 72"
    L = layers[0]
    check_capture(args.capture_dir, [L])
    lockstep_n = resolve_lockstep(args.lockstep)
    out_scales = {"auto": None, "always": True, "never": False}[args.out_scales]
    src = SourceModel(args.src)
    log(f"BENCH start (v3.1 pooled GSS): layer {L}, {args.bench_experts} experts, "
        f"lockstep={lockstep_n}, gpu {torch.cuda.get_device_name(0)}")
    calib = LayerCalib(args.capture_dir, L, None)
    device = "cuda:0"
    group_size = max(1, (lockstep_n + 3 * TP - 1) // (3 * TP))
    times = []
    E0 = 0
    while E0 < args.bench_experts:
        Es = list(range(E0, min(E0 + group_size, args.bench_experts)))
        torch.cuda.synchronize()
        te = time.time()
        pre_list, walk_ctxs, group_hd = [], [], []
        for E in Es:
            W = {proj: load_expert_bf16(src, L, E, proj, device, None) for proj in PROJS}
            hd_gu, hd_down, _hm = build_expert_hessians(
                calib, E, W["gate_proj"], W["up_proj"], device, args.min_routed)
            for pi, proj in enumerate(PROJS):
                for r, sl in expert_slices(W[proj], proj):
                    hd = hd_gu if proj != "down_proj" else hd_down[r]
                    pre_list.append(encode_slice_prologue_pre(
                        sl, hd, slice_seed(L, E, pi, r), out_scales))
            del W
            group_hd.append((hd_gu, hd_down))
        tg = time.time()
        g_scale_gss_lockstep(pre_list)
        torch.cuda.synchronize()
        tg = time.time() - tg
        ordered = []
        for ctx in pre_list:
            ctx = encode_slice_prologue_post(ctx)
            ordered.append(ctx)
            if ctx["wq_reg"] is None:
                walk_ctxs.append(ctx)
        walks = [LDLQWalk(c) for c in walk_ctxs]
        tq = time.time()
        lockstep_ldlq(walks, lockstep_n)
        tq = time.time() - tq
        for ctx in ordered:
            encode_slice_epilogue(ctx)
            ctx.clear()
        for hd_gu, hd_down in group_hd:
            free_H_data(hd_gu, *hd_down)
        torch.cuda.synchronize()
        dt = (time.time() - te) / len(Es)
        times.append(dt)
        log(f"BENCH experts {Es[0]}-{Es[-1]}: {dt:.2f}s/expert "
            f"(lockstep main pass {tq/len(Es):.2f}s/expert, "
            f"pooled gss {tg/len(Es):.2f}s/expert)")
        E0 += group_size
    times_sorted = sorted(times)
    mean = sum(times) / len(times)
    log(f"BENCH RESULT (v3.1 pooled GSS): layer {L}, {len(times)} pool(s), "
        f"lockstep={lockstep_n} — mean {mean:.2f}s/expert, best {times_sorted[0]:.2f}, "
        f"worst {times_sorted[-1]:.2f} "
        f"(v3 baselines: 19.92s bench under 3-contention, ~5.1s solo in enc3)")
    log(f"BENCH projection at --workers 4 (1/GPU): {mean:.1f}s/expert -> "
        f"{mean*256/60:.1f} min/layer/worker -> {mean*256*75/4/3600:.1f} h for 75 layers")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 NVFP4 -> EXL3-tr3 hybrid encoder")
    ap.add_argument("--encode", action="store_true", help="run multi-GPU encode (default action)")
    ap.add_argument("--assemble", action="store_true", help="assemble final checkpoint dir")
    ap.add_argument("--upload", action="store_true", help="upload --out to private HF repo")
    ap.add_argument("--smoke", action="store_true", help="kernel/roundtrip smoke test on random data")
    ap.add_argument("--src", default=DEF_SRC)
    ap.add_argument("--work", default=DEF_WORK)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--repo", default=DEF_REPO)
    ap.add_argument("--layers", default=f"{FIRST_MOE_LAYER}-{NUM_LAYERS-1}",
                    help="e.g. '3-77' or '3,5,10-12'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=4,
                    help="physical GPU count; workers are assigned round-robin (worker r -> GPU r%%gpus)")
    ap.add_argument("--io-workers", type=int, default=12)
    ap.add_argument("--capture-dir", default=DEF_CAPTURE,
                    help="Pass A output (capture_hessians.py) — REQUIRED for --encode")
    ap.add_argument("--min-routed", type=int, default=MIN_ROUTED,
                    help="per-expert routed-token floor; below it the expert uses the "
                         "layer-level H (recorded in done JSON, never a failure)")
    ap.add_argument("--out-scales", choices=["auto", "always", "never"], default="auto",
                    help="exl3 out-channel scales; 'auto' matches convert_model.py default "
                         "(calibrated: per-slice H-diag skew decision, reference math)")
    ap.add_argument("--lockstep", default="auto",
                    help="v3: max LDLQ slice-walks co-stepped per quantize_tiles call; "
                         "'auto' = 12 (all walks of one expert); N>12 pools "
                         "ceil(N/12) experts per lockstep group")
    ap.add_argument("--oracle", action="store_true",
                    help="HARD GATE: byte-equality of v2 sequential vs v3 lockstep vs "
                         "v3.1 pooled-GSS on real slices of ONE --layers layer; "
                         "logs to --oracle-log")
    ap.add_argument("--oracle-experts", default="0,1",
                    help="comma-separated expert ids for --oracle (12 slices each)")
    ap.add_argument("--oracle-log", default="/root/tr3_v31_oracle.log")
    ap.add_argument("--bench", action="store_true",
                    help="v3 throughput probe on ONE --layers layer (no --work writes)")
    ap.add_argument("--bench-experts", type=int, default=8)
    ap.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        if args.worker_rank is not None:
            worker_main(args)
        elif args.smoke:
            smoke(args)
        elif args.oracle:
            oracle(args)
        elif args.bench:
            bench(args)
        elif args.assemble:
            assemble(args)
        elif args.upload:
            upload(args)
        else:
            orchestrate_encode(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
