# GLM-5.3-Flash-EXL3-4bpw

Source: `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43`. All routed experts including MTP45 are uniform four-bit EXL3/TR3 MCG; non-routed tensors retain their official native dtype. The original custom Transformers TP2 runtime and the dedicated SM120 vLLM image below are qualified separately against the same BF16 teacher evidence.

Five-cold-run mean teacher-to-student KLD: `0.024554564250` over 51,175 sealed causal positions per run. Actual TP2 runtime qualification-window KLD: `0.022750847878` over 2,047 positions (both gates: mean KLD < 0.06). This checkpoint requires the included custom Transformers TP2 adapter and is not a stock vLLM/ExLlamaV3 compatibility claim.

## Five cold KLD runs

| Run | Mean teacher-to-student KLD | Positions | Report receipt | Capture receipt |
|---:|---:|---:|---|---|
| 1 | 0.024554564249958 | 51,175 | `ef6a8dedc20f11e582658f94923da3e66c2b6cea4ff62d936abb790e376e2461` | `013759025d8414f8811fa140250e2c79097c1082926edd4ae2cfc6751722fc8d` |
| 2 | 0.024554564249958 | 51,175 | `b7d1cac829f6b21471da4ea724aac479f9db250d4286edd412e099fa747f8257` | `eae08903737bde9f31bf6f8632d2de7b6539f4b1efd8113c5f81461d92aaf671` |
| 3 | 0.024554564249958 | 51,175 | `663629ccd2bda08a4c299d767b7e6e6d622a81ad6830ad1acf08d0eb8ca1a196` | `000896721ea7116322eb31d8e75718985d29240fabd6a921627bb02c03516bec` |
| 4 | 0.024554564249958 | 51,175 | `cdb2d8ee4ce795f695f335f0bb3ce7bd135dcf6df4f48c6e3862b40cd1340586` | `7ece4defa651c3693bffd624ad7d07ff85c0dceb7674a7752ec136dea6370c3f` |
| 5 | 0.024554564249958 | 51,175 | `ac4d6d94aef27b09ca9b2dd513516e793cf5f4afe3d1f2b008a3fb4ed64ae243` | `5b59145332206b4c0fb82f791e2c09be8fadb16d18e6e58818b78e919294cb65` |

All five accepted executions used the same sealed 25-window panel, so each has 51,175 causal prediction positions. They produced the same tokenwise-KLD SHA-256 and a population standard deviation of zero. The first attempt at the fifth capture received an external SIGTERM before it wrote any logits; it retained only its plan and reader identity and is excluded. The table's run 5 is the clean `run5b` retry, with a distinct cold-execution backend/capture receipt and the same measured KLD as runs 1-4.

The direct packed TP2 serving result (`0.022750847878`) is a separate one-window runtime qualification measurement, not a replacement for the five full-panel runs. The raw decoded-logit absolute-error diagnostic remains failed and is disclosed in the receipts; qualification is based on teacher-to-runtime KLD, rank-identical output, complete packed-tensor census, and multi-token generation.

Code and the five-run receipts: [brandonmmusic-max/glm-5.3-flash-exl3-4bpw](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw).

## BF16 teacher logits and replay calibration

The complete teacher dataset is published at
[`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits).
It contains 640 rolling calibration windows plus the 25 qualification-only final
windows: 665 windows total, each with 2,048 input tokens, 2,047 scored positions,
and the full 154,880-token vocabulary. The 1,361,255 scored positions occupy
843,324,965,136 raw logits bytes.

The teacher is the released BF16 checkpoint with its native FP32 tensors
preserved; the logits are stored as float32 to avoid an additional
storage-precision loss. Payload revision
[`7c378d5f17dba158c4c803eff27c346dd0615660`](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/tree/7c378d5f17dba158c4c803eff27c346dd0615660)
is bound by the
[`16e16e90078bc0b54bd1cd37b08ba7dad03819726d0258443a0e30b68b354472` aggregate audit](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/blob/267ccf27ca92575529e0a1ef80e7eed8d209a8f4/logits/full-panel/receipts/full-dataset-audit.json),
which records every payload path, size, and SHA-256. The 25 final windows remain
qualification-only and are excluded from fitting and expert selection.

## Minimal TP2 launch

The historical Transformers runtime below is an evidence/qualification path,
not the optimized daily-driver launch. Use the SM120 container in the next
section for serving.

Use Transformers 5.16.1, clone ExLlamaV3 at commit `c5d9c657966ffeeaa9353f0cc899f18629da4a13`, compile its CUDA extension, then run:

```bash
PYTHONPATH=runtime/src torchrun --standalone --nproc-per-node=2 runtime/scripts/run_glm53_custom_tp_runtime.py --model . --exllamav3-source /path/to/exllamav3 --prompt 'Hello'
```

## SM120 TP2 daily-driver image

Docker Hub: [`verdictai/glm53-flash-exl3-k4`](https://hub.docker.com/r/verdictai/glm53-flash-exl3-k4)

- Version: `r19-sm120-tp2-v44`
- Immutable OCI index digest: `sha256:15192e3930b4ae5558271ebe7d1a5a02da6dcc5a6c292c44e79a3fb8c883b5e1`
- Linux/amd64 manifest: `sha256:0c35421c2c773743ee74b17592d5bd32143546a2bbfe32fc8f32b97ca74167bf`
- Hardware qualified: 2x RTX PRO 6000 Blackwell (SM120), TP2
- Daily-driver mode: NVFP4 MLA KV, DCP2, CUDA graphs, MTP3, 499,968-token maximum context
- Accuracy mode: FP8 MLA KV, DCP1 or DCP2; the published serve script uses a measured-safe 262,144-token FP8 ceiling
- Alternate mode: DCP1; DCP2 CUDA graphs with MTP3 are fixed and qualified in v44
- Sampling defaults from the model generation config: temperature `1.0`, top-p `0.95`

The image contains a dedicated vLLM/B12X overlay for GLM-5.3-Flash's hybrid
linear/sparse-attention architecture, NoPE MLA, EXL3 K4 routed experts, and the
MTP layer. The official 45-layer pattern is honored directly: 34 linear layers
and 11 DeepSeek sparse-attention layers, alternating three linear layers and one
sparse layer. Sparse layers use IndexPool-4 and top-k 2,048. This is not a claim
that the checkpoint runs in upstream stock vLLM.

### Current local optimum: v71 validation profile (2026-08-27)

`v71` names the latest validated benchmark/profile revision, not a Docker tag.
It uses the local `r19-sm120-tp2-ep2-v47-candidate` image with NVFP4 MLA KV,
DCP2, MTP3 probabilistic rejection sampling, TP2/EP2, CUDA graphs, and the
route128 SMEM/register fast path. The 46-entry power-of-two calibration bank
covers all 45 backbone layers plus MTP45. No TMEM path is used on SM120.

The v71 measurements used physical GPUs 1 and 3, both RTX PRO 6000 Blackwell
Workstation Edition cards, with a +6000 MHz memory VF offset and 600 W power
limit. These are workstation-pair/OC results and are not a controlled claim
that overclocking alone caused the change.

| Context | Warm prefill tok/s | C1 sustained decode tok/s | MTP draft acceptance |
|---:|---:|---:|---:|
| 0 | — | **147.79** | 42.86% |
| 8K | **5,723** | — | — |
| 16K | **6,234** | **148.55** | 50.88% |
| 32K | **6,219** | **149.58** | 51.72% |

The exact v71 receipts and the later C1-C16/128K stress matrix are under
`runtime-results/v71/benchmarks/`. The stress matrix reached 564.8 aggregate
tok/s at C8/0K and 481.7 tok/s at C8/32K, but those cells admitted only 7/8 and
6/8 requests. C8/C16 at 64K was severely capacity/thermal limited. GPU 3
reached 94 C and accumulated hardware thermal slowdown. The nominal 128K cells
submitted a 131,072-token prompt plus requested output against a 131,072-token
server ceiling; those request errors are disclosed and are not reported as zero
model throughput. C16/128K was skipped by the harness because it did not fit.

### Current actual-runtime KLD by MLA KV-cache type

Both current results are independent five-run averages over the complete
2,048-token `final-0000` qualification window (2,047 causal positions per run),
compared against the sealed BF16 teacher logits. The matched correctness regime
is TP2/EP2, DCP2, eager, no MTP, route128 SMEM. MTP is disabled only for this
teacher-logit comparison so draft-token sampling cannot alter the scored
runtime logits.

| MLA KV cache | Five-run mean KLD | Population stddev | Mean top-1 agreement | Gate |
|---|---:|---:|---:|---:|
| FP8 | **0.024581652920** | 0.000159556478 | 0.936297020029 | pass |
| NVFP4, calibrated power-of-two scales | **0.054757372223** | 0.000000000000 | 0.914997557401 | pass |

The FP8 receipt SHA-256 is
`da072d243fbdb231388bfc23b84bdb0cee2cb26c1885d3ec407c4164525b6b6b`;
the NVFP4 receipt SHA-256 is
`b52b6d7abbcbf1f0bc81f713e4513bc8a376235e2f44cc7f4ba7d368f62e69ca`.
The NVFP4 no-MTP KLD exercises the 45 backbone cache entries; the published
46th calibrated entry is the MTP layer used by the daily MTP3 profile.

### Historical v44 actual-runtime KLD

The exact 2,048-token `final-0000` qualification window was captured with TP2,
DCP1, eager execution, `fp8_ds_mla`, no MTP, and full-vocabulary float32
runtime logits, then compared to the sealed BF16 teacher in float64 chunks.

| Metric | SM120 FP8 KV five-run mean | Rented B200 custom TP2 | Offline K4 |
|---|---:|---:|---:|
| Mean teacher KLD | **0.024628576596** | 0.022750847878 | 0.031831601179 |
| Top-1 agreement | **0.937957987298** | 0.9384 | — |

The earlier local KLD near `0.10` was a runtime scale-decoding defect, not a
routing-quality result. The cache writer stores GLM's four calibrated
per-token, per-128-channel scales as arbitrary FP32 values (`amax / 448`). The
SM120 FlashInfer reader was left at `kv_scale_format="auto"`, which interprets
inline scales using the DeepSeek-v3.2 power-of-two convention. v34 explicitly
selects `arbitrary_fp32` in the GLM NoPE adapter. The corrected first-64-row KLD
is `0.1918669499`; rows 64 onward are `0.0194743407`, and the whole-window
result reproduces the independently observed server range.

The current five-run receipt is published at
`runtime-results/v44/kld/fp8-five-run-kld-receipt.json`.

An independent five-run repetition was then executed on physical GPUs 2 and 3
with the same TP2/DCP1 eager/no-MTP FP8-cache regime and the complete 2,048-token
window (2,047 causal prediction positions per run):

| Run | Mean teacher KLD | Top-1 agreement |
|---:|---:|---:|
| 1 | 0.024566116964 | 0.939423546654 |
| 2 | 0.024849557477 | 0.939423546654 |
| 3 | 0.024882931269 | 0.936492427943 |
| 4 | 0.024016412384 | 0.938935026869 |
| 5 | 0.024827864889 | 0.935515388373 |

The five-run mean is **0.024628576596**, population standard deviation is
`0.000326156681`, and mean top-1 agreement is `0.937957987298`. All five runs
pass the preregistered mean-KLD `< 0.06` gate.

The matched v44 TP2/DCP1/eager/no-MTP test was repeated five times with
`nvfp4_ds_mla` on physical GPUs 2 and 3. Every run covered the same complete
2,047 causal positions and produced the same tokenwise result:

- five-run mean KLD: **`0.06053485053836315`**;
- population standard deviation: **`0.0`**;
- mean top-1 agreement: **`0.9154860771861261`**;
- tokenwise KLD SHA-256 (all five runs):
  `03dc42308d83b9f64e04c101253a5e316dd21f1e55332a9d63c36fabac7b156e`.

This narrowly misses the preregistered `<0.06` gate by
`0.00053485053836315` and is disclosed as a failure, not rounded into a pass.
The newer per-token dynamic-scale control was also tested once and was worse:
mean KLD `0.068229579401`, top-1 agreement `0.919882755252`, with tail outliers
up to `7.1683`. Dynamic scaling is therefore disabled in the published daily
profile. Use FP8 KV when KLD fidelity is the priority; use NVFP4 KV when the
499,968-token capacity is required.

### MTP3 DCP2 measured decode and prefill

MTP3 remains enabled by default with probabilistic rejection sampling. The
current v71 workstation-pair figures are reported above; the older v44
measurements below remain useful as a non-OC historical baseline. On that v44
NVFP4 DCP2 CUDA-graph path, concurrency-1 decode measured 98.9, 106.5, 101.2,
107.2, and 112.4 tokens/s at 0, 16K, 32K, 64K, and 128K context. Repeated warm
prefill measured 3,819, 4,112, 4,149, 4,174, and 4,145 tokens/s at 8K, 16K,
32K, 64K, and 128K. The selected attention backend is `B12X_MLA_SPARSE`; this
is the SM120 sparse fast path, not eager fallback.

The DCP2 NVFP4 launch retained **608,656 logical KV tokens** (1.58 GiB cache
memory per GPU), or 1.22x the configured 499,968-token maximum. The separately
qualified FP8 DCP2 launch retained 356,352 logical tokens at 95% utilization.

### DCP2 CUDA-graph MTP3 fix

v44 includes the DCP2 fix for the failure that previously appeared only when CUDA graphs,
MTP depth greater than one, and graph-padded multi-request batches were used
together. The sparse KPool cache writer had used the padded scoring width for
cache writes even when the live per-request MTP widths summed to fewer tokens;
GDN metadata also retained zero-token graph rows. The v44 runtime separates the live write
count from the padded scoring count and compacts those zero-token rows.

The exact formerly failing TP2/DCP2/MTP3/max-seqs-4 graph geometry was
validated on two RTX PRO 6000 Blackwell GPUs with both cache paths:

| Cache | Attention path | Concurrent coherence | Runtime errors |
|---|---|---:|---:|
| `fp8_ds_mla` | `FLASHINFER_MLA_SPARSE_SM120` | 4/4 arithmetic requests correct | 0 |
| `nvfp4_ds_mla` | `B12X_MLA_SPARSE` | 4/4 arithmetic requests correct and clean stop | 0 |

The final NVFP4 validation exposed 348 MTP draft steps and 629 accepted
speculative tokens, or 60.25% of the 1,044 drafted-token opportunities. This
is the fast CUDA-graph path, not eager fallback.

Repeated warm prefill on the production 2,048-token scheduler setting measured
4,565 tok/s at 8K, 4,746 at 16K, 4,823 at 32K, 4,788 at 64K, and 4,685 at
128K. The live route is the B12X SM120 unified MG sparse-prefill kernel, not a
generic fallback. A 2,304-token batch was no faster and retained only 1.04x KV
headroom for a 499,968-token request; 4,096 could not retain the full context
budget. The published 2,048-token scheduler setting is therefore intentional.

The extreme-context qualification found a separate transient-workspace limit.
With the 2,048-token chunk, a 499K request asked KPool for a 758 MiB logits
matrix when only 754 MiB was free. The memory-safe `long500k` profile uses a
1,024-token chunk, one active sequence, and 98.5% memory utilization. On the
published v37 digest it retained 678,968 KV tokens (1.36x the configured
499,968-token maximum) and recovered a middle-depth needle from a 498,365-token
prompt in 141.5 seconds. The same profile on v34 recovered all three insertion
depths at both 384K and 499K. The current v43 qualification root filesystem is
byte-identical to published v44 and extends this evidence with the 17/18 raw
matrix plus the exact 498,368-token retry above. The long profile trades
prefill throughput for transient
workspace safety; it does not select a generic attention backend.

### Current quality gates

Using the official generation defaults (temperature 1.0, top-p 0.95), the
NVFP4/DCP2/CUDA-graph/MTP3 profile scored Estonia **10/10** with 127.87
aggregate generated tokens/s and no 40,000-token cap hits. The raw needle
matrix scored **17/18** through 499K. Its only apparent miss consumed all 1,600
remaining output tokens in reasoning; replaying the exact 498,368-token prompt
with low reasoning returned the exact needle in seven completion tokens. The
raw 17/18 receipt is preserved rather than rewritten.

LAVD is not yet a passing quality gate. With the same official sampling and
normal reasoning, the harness reported 1/10 exact; the conservative response
audit recovered two additional near answers (3/10 accepted), while nine runs
hit the 40,000-token ceiling after an average of 39,752 completion tokens.
Constraining the generic API reasoning mode to `low` removed every cap hit and
reduced the average to 3,647 completion tokens, but did not solve accuracy: the
original scorer found 1/10 near, and the response audit recovered 1 exact plus
3 near answers (4/10 accepted). Generation throughput was effectively
unchanged at 136.48 versus 136.59 aggregate tokens/s. These raw and audited
receipts are published under `runtime-results/v44/quality/`; the result is
reported as a reasoning/scoring-harness diagnostic, not silently counted as a
model pass.

### Docker Compose

Download `runtime/compose.sm120-tp2.yaml` from this repo, set the model path if
needed, and run:

```bash
GLM53_MODEL_PATH=/absolute/path/to/GLM-5.3-Flash-EXL3-4bpw \
  docker compose -f compose.sm120-tp2.yaml up -d
```

### Serve script

The published `runtime/serve-glm53-sm120-tp2.sh` defaults to GPUs 0,1, port
8012, NVFP4/DCP2/MTP3, prefix caching disabled, and the immutable v44 digest:

```bash
chmod +x serve-glm53-sm120-tp2.sh
MODEL=/absolute/path/to/GLM-5.3-Flash-EXL3-4bpw ./serve-glm53-sm120-tp2.sh
```

Use `CACHE=fp8_ds_mla`, `DCP=1`, or `MTP_TOKENS=0` for controlled variants.
For a qualified single-request prompt through 500K, use:

```bash
PROFILE=long500k MODEL=/absolute/path/to/GLM-5.3-Flash-EXL3-4bpw \
  ./serve-glm53-sm120-tp2.sh
```

The default `daily` profile keeps the faster 2,048-token scheduler chunk for
ordinary serving. `PROFILE=long500k` is NVFP4/MTP3 and intentionally sets
`MAX_NUM_BATCHED_TOKENS=1024`, `MAX_NUM_SEQS=1`, and
`GPU_MEMORY_UTILIZATION=0.985`.

Credit goes to turboderp for the EXL3 quantization format. Local Inference Lab,
Martin Vit, and Luke Alonzo contributed or helped test components of the base
image.
