---
base_model: zai-org/GLM-5.3-Flash-BF16
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: shapleymcg-1.0
license_link: https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/main/LICENSE
tags:
  - glm
  - exl3
  - tr3
  - vllm
  - sm120
  - nvfp4
  - fp8
---

# GLM-5.3-Flash TR3 4bpw

Uniform-K4 EXL3/TR3 routed experts for GLM-5.3-Flash, paired with a dedicated
two-GPU SM120 vLLM/B12X runtime. The checkpoint is hosted at
[`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw).
The runtime is not compatible with stock upstream vLLM.

## Run v75

Qualified hardware: two RTX PRO 6000 Blackwell Workstation Edition GPUs
(SM120, 96 GB each), TP2/EP2. Model weights are mounted from the host and are
not baked into the runtime image.

```text
verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v75
OCI index:   sha256:4605c420cc589be9fd15fc759c7f7c2a6035dab48f885c9466eb2233527bca64
linux/amd64: sha256:e75a00b5e1ce4debd568d029db2868b1ba01e9d7e87ddf034e6b644935213558
```

### Docker Compose

```bash
curl -L -o compose.sm120-tp2.yaml \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/compose.sm120-tp2.yaml

GLM53_MODEL_PATH=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
  docker compose -f compose.sm120-tp2.yaml up -d

curl http://127.0.0.1:8012/v1/models
```

### Serve script

```bash
curl -L -o serve-glm53-sm120-tp2.sh \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/serve-glm53-sm120-tp2.sh
chmod +x serve-glm53-sm120-tp2.sh

MODEL=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
GPU_DEVICES=0,1 \
  ./serve-glm53-sm120-tp2.sh
```

The daily defaults are NVFP4 MLA KV, TP2/EP2, DCP2, CUDA graphs,
probabilistic MTP3, route-128 SMEM/register kernels, and prefix caching off.
The image contains the calibrated 46-layer NVFP4 scale bank. SM120 does not
use TMEM or TCGEN in this path.

Controlled alternatives:

```bash
# FP8 MLA KV; the measured default maximum context is 262,144 tokens.
CACHE=fp8_ds_mla ./serve-glm53-sm120-tp2.sh

# Correctness/KLD-style eager run without speculative decoding.
MTP_TOKENS=0 ENFORCE_EAGER=1 ./serve-glm53-sm120-tp2.sh

# Explicit long-context NVFP4 profile.
PROFILE=long500k ./serve-glm53-sm120-tp2.sh
```

The full launch artifacts are
[`runtime/compose.sm120-tp2.yaml`](runtime/compose.sm120-tp2.yaml) and
[`runtime/serve-glm53-sm120-tp2.sh`](runtime/serve-glm53-sm120-tp2.sh).
Both explicitly enable expert parallelism; omitting `--enable-expert-parallel`
does not reproduce the qualified TP2/EP2 regime.

## v75 performance

The speed profile is NVFP4 MLA KV, TP2/EP2, DCP2, CUDA graphs, MTP3,
route-128 SMEM/register, and prefix caching disabled. It used workstation GPUs
1 and 3 at a +6000 MHz memory offset and 600 W power limits. These are OC
measurements, not a claim that overclocking alone produced the result.

### Standalone cold prefill

| Target | Actual prompt | TTFT | Prefill tok/s | Samples |
|---:|---:|---:|---:|---:|
| 8K | 8,201 | 1.427 s | **5,748** | 7 |
| 16K | 16,230 | 2.598 s | **6,246** | 4 |
| 32K | 32,323 | 5.172 s | **6,250** | 2 |
| 64K | 64,515 | 10.366 s | **6,224** | 1 |
| 127.9K | 127,888 | 27.944 s | **4,577** | 1 |

Receipt: [prefill JSON](runtime-results/v75/benchmarks/prefill-8k-127k.json)
and [benchmark log](runtime-results/v75/benchmarks/prefill-8k-127k.log).

### Sustained C1 decode

| Context | Aggregate tok/s | MTP draft acceptance | Errors/capacity limit |
|---:|---:|---:|---:|
| 0 | **141.94** | 57.06% | 0 / no |
| 16K | **147.46** | 61.49% | 0 / no |
| 32K | **143.49** | 29.89% | 0 / no |
| 64K | **146.29** | 48.85% | 0 / no |
| 124K | **148.35** | 55.93% | 0 / no |

Receipt: [decode JSON](runtime-results/v75/benchmarks/decode-c1-through-124k.json)
and [benchmark log](runtime-results/v75/benchmarks/decode-c1-through-124k.log).

v75 is not a material short-context speed increase over v71: the 8K-32K
prefill gain is only about 0.2-0.5%, and short-context C1 decode is slightly
lower. Its release value is the corrected, stable long-prefill path: valid 64K
and near-128K prefill receipts plus stable C1 decode through 124K.

The route-128 kernel itself is materially faster than the generic kernel in
its isolated 128-row test: about 37.3% on spread routes and 39.2% on random
routes, with cosine similarity above `0.99999996`. That microbenchmark should
not be read as a 37-39% end-to-end serving gain. See the
[numerical/timing receipt](runtime-results/v75/validation/route128-vs-generic.json).

## MLA KV-cache KLD

These are final v75 five-run means over the complete 2,048-token `final-0000`
window: 2,047 causal prediction positions per run against sealed BF16 teacher
logits. The matched regime is TP2/EP2, DCP2, eager, no MTP, and route-128
SMEM/register. MTP is intentionally disabled so draft sampling cannot alter
the runtime logits being compared.

| MLA KV cache | Five-run mean KLD | Population stddev | Mean top-1 agreement | Gate |
|---|---:|---:|---:|---:|
| FP8 | **0.024610591221** | 0.000256852524 | 0.937274059599 | pass |
| NVFP4 calibrated power-of-two | **0.054757372223** | 0.000000000000 | 0.914997557401 | pass |

Receipts: [FP8](runtime-results/v75/kld/fp8-five-run-kld.json),
[NVFP4](runtime-results/v75/kld/nvfp4-five-run-kld.json), and the
[46-layer scale bank](runtime-results/v71/calibration/glm53-nvfp4-mla-46-layer-power2-scales.json).
Only compact aggregate receipts are published; the multi-gigabyte raw captures
are intentionally excluded.

## Quality and long context

The v75 quality runs use the production NVFP4/DCP2/CUDA-graph/MTP3 path and
the model's generation defaults, temperature `1.0` and top-p `0.95`.

| Test | v75 result | Interpretation |
|---|---:|---|
| Estonia 10x | **10/10** | No errors or token-cap hits; 183.77 aggregate generation tok/s |
| LAVD-low 10x | **3/10 accepted** | 1 exact + 2 near after response audit; failed quality gate |
| Needle through 500K | **17/18 raw** | Final 498,368-token/depth-0.9 cell exhausted its 256-token reasoning budget |
| Exact-limit retry | **1/1** | Same final cell passed with a 1,600-token output allowance |

Receipts: [Estonia](runtime-results/v75/quality/estonia-10x.json),
[LAVD raw](runtime-results/v75/quality/lavd-low-10x.json),
[LAVD audit](runtime-results/v75/quality/lavd-low-10x-rescored.json),
[needle matrix](runtime-results/v75/quality/needle-through-500k.json), and
[final-cell retry](runtime-results/v75/quality/needle-499k-depth-0.9-retry.json).

Hotel Lights was explicitly stopped and was not rerun on v75. The available
historical checkpoint-lineage result is 7/10 exact on v30; it is preserved as
[historical evidence](runtime-results/v30/quality/hotel-10x.json), not presented
as a v75 measurement. LAVD remains prominently disclosed as a failed gate.

Both cache paths produced coherent generation. The NVFP4 receipt is
[`coherence-smoke.json`](runtime-results/v75/validation/coherence-smoke.json);
the FP8 receipt was captured live from the exact published digest with
TP2/EP2, DCP2, CUDA graphs, and MTP3 in
[`coherence-smoke-fp8.json`](runtime-results/v75/validation/coherence-smoke-fp8.json).

## Architecture and implementation

- BF16 source: `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43`
- Routed experts, including MTP45: uniform four-bit EXL3/TR3 MCG
- Non-routed tensors: official native dtype
- Expert layout: global E288 namespace, rank-local E144 slabs under EP2
- 45-layer pattern: 34 linear-attention layers and 11 sparse-attention layers
- Sparse attention: IndexPool-4, top-k 2,048
- Route-128 kernel: physical M128/N256/K64, 256 threads, SMEM/register only
- Production speculation: probabilistic MTP3
- Generation defaults: temperature `1.0`, top-p `0.95`

The current implementation has reached a practical flag/block-size tuning
plateau on this workstation pair, not a fundamental EXL3 or SM120 limit.
Further large gains would require new kernel, attention, or scheduler work.

## Provenance and license

The image embeds `/usr/share/glm53/provenance.json`, carries standard OCI
source/revision/documentation/license labels, and includes a transparent
runtime-bundle fingerprint. It performs no telemetry, callback, hidden output
watermark, or inference modification. Verify the immutable image with:

```bash
curl -L -o verify-provenance.sh \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/verify-provenance.sh
chmod +x verify-provenance.sh
./verify-provenance.sh \
  verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v75@sha256:4605c420cc589be9fd15fc759c7f7c2a6035dab48f885c9466eb2233527bca64
```

See [PROVENANCE.md](PROVENANCE.md) and the
[Docker release receipt](runtime-results/v75/validation/docker-release.json).
This repository is distributed under the ShapleyMCG License 1.0 in
[LICENSE](LICENSE). It is source-available and is not described here as an
OSI-approved open-source license.

## Evidence index

- [v75 release evidence](runtime-results/v75/)
- [v71 benchmark and calibration archive](runtime-results/v71/)
- [v44 qualification archive](runtime-results/v44/)
- [historical long-form model card](docs/HISTORICAL_MODEL_CARD_2026-08-27.md)

Credit goes to turboderp for EXL3. Local Inference Lab, Martin Vit, and Luke
Alonzo contributed or helped test components of the base runtime.
