---
base_model: zai-org/GLM-5.3-Flash-BF16
library_name: transformers
pipeline_tag: text-generation
tags:
  - glm
  - exl3
  - vllm
  - sm120
  - nvfp4
  - fp8
---

# GLM-5.3-Flash EXL3 4bpw

Uniform-K4 EXL3 routed experts for GLM-5.3-Flash, with a dedicated two-GPU
SM120 vLLM/B12X runtime. The recommended release is the evidence-bound
`v71` image below. It is not compatible with upstream stock vLLM.

## Run the current best image

**Hardware qualified:** two RTX PRO 6000 Blackwell Workstation Edition GPUs
(SM120, 96 GB each), TP2/EP2.

**Docker Hub:** [`verdictai/glm53-flash-exl3-k4`](https://hub.docker.com/r/verdictai/glm53-flash-exl3-k4)

```text
verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v71
OCI index: sha256:bb6d2516f88d963a0c8c10d85582c4580adc8754d506f62ce4915b84c095faca
linux/amd64: sha256:77850e030d07df2e2907e6741b69883e82287cf731d1aaef8ff3f2070aedf351
```

```bash
docker pull verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v71@sha256:bb6d2516f88d963a0c8c10d85582c4580adc8754d506f62ce4915b84c095faca
```

The image contains the exact runtime overlay and 46-layer NVFP4 scale bank used
by the v71/v74 measurements. It uses rank-local E144 expert slabs from the
global E288 route namespace, uniform EXL3 K4 experts, route-128
SMEM/register kernels, DCP2, and MTP3. SM120 does not use a TMEM path here.

### Option A: Docker Compose

The complete pinned Compose file is
[`runtime/compose.sm120-tp2.yaml`](runtime/compose.sm120-tp2.yaml).

```bash
curl -L -o compose.sm120-tp2.yaml \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/compose.sm120-tp2.yaml

GLM53_MODEL_PATH=/absolute/path/to/GLM-5.3-Flash-EXL3-4bpw \
  docker compose -f compose.sm120-tp2.yaml up -d
```

The service listens on port `8012` by default:

```bash
curl http://127.0.0.1:8012/v1/models
```

### Option B: serve script

The complete script is
[`runtime/serve-glm53-sm120-tp2.sh`](runtime/serve-glm53-sm120-tp2.sh).

```bash
curl -L -o serve-glm53-sm120-tp2.sh \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/serve-glm53-sm120-tp2.sh
chmod +x serve-glm53-sm120-tp2.sh

MODEL=/absolute/path/to/GLM-5.3-Flash-EXL3-4bpw \
GPU_DEVICES=0,1 \
  ./serve-glm53-sm120-tp2.sh
```

Defaults are NVFP4 MLA KV, TP2/EP2, DCP2, CUDA graphs, probabilistic MTP3,
prefix caching off, and the immutable v71 digest. Controlled variants:

```bash
CACHE=fp8_ds_mla ./serve-glm53-sm120-tp2.sh
DCP=1 ./serve-glm53-sm120-tp2.sh
MTP_TOKENS=0 ENFORCE_EAGER=1 ./serve-glm53-sm120-tp2.sh
```

The inherited `PROFILE=long500k` launcher is available for NVFP4, but its
500K qualification is historical v44 evidence; 500K was not rebenchmarked
after the v71 overlay.

## Current measured performance

The current performance profile is NVFP4 MLA KV, TP2/EP2, DCP2, CUDA graphs,
probabilistic MTP3, route-128 SMEM/register, and prefix caching disabled.
Measurements used GPUs 1 and 3, both full-power workstation cards, at a +6000
MHz memory offset and 600 W limits. These are OC results, not a claim that the
overclock alone caused the improvement.

### Standalone cold prefill

| Context | Actual prompt tokens | TTFT | Prefill tok/s | Samples |
|---:|---:|---:|---:|---:|
| 8K | 8,201 | 1.433 s | **5,723** | 7 |
| 16K | 16,230 | 2.604 s | **6,234** | 4 |
| 32K | 32,323 | 5.198 s | **6,219** | 2 |

The later attempt to extend standalone prefill through 64K and 127K ended with
an incomplete HTTP chunked read before it wrote valid measurements. Those
contexts are not assigned prefill numbers.

Receipt:
[`nvfp4-dcp2-mtp3-ws13-oc6000-prefill.json`](runtime-results/v71/benchmarks/nvfp4-dcp2-mtp3-ws13-oc6000-prefill.json).

### Sustained decode

The clean dedicated C1 run measured:

| Context | C1 tok/s | MTP draft acceptance |
|---:|---:|---:|
| 0 | **147.79** | 42.86% |
| 16K | **148.55** | 50.88% |
| 32K | **149.58** | 51.72% |

The later concurrency matrix measured:

| Context | C1 | C2 | C4 | C8 | C16 |
|---:|---:|---:|---:|---:|---:|
| 0 | 146.7 | 258.2 | 393.9 | 564.8 (7/8)* | 563.1 (7/16)* |
| 16K | 143.1 | 260.3 | 395.0 | 315.3 (6/8)* | 360.9 (6/16)* |
| 32K | 144.6 | 242.4 | 385.6 | 481.7 (6/8)* | 445.1 (6/16)* |
| 64K | 143.2 | 238.5 | 379.2 | 17.1 (5/8)* | 26.5 (5/16)* |

`(X/Y)` is average running requests versus requested concurrency. `*` means
the cell was underfilled, queued, or hit its admission warmup timeout; it is
not automatically proof of KV exhaustion. The 64K C8/C16 rows were also
thermally/capacity limited. The nominal 128K requests combined a 131,072-token
prompt with requested output against a 131,072-token server ceiling, so those
cells errored and are not reported as throughput.

Receipts:

- [dedicated C1 JSON](runtime-results/v71/benchmarks/nvfp4-dcp2-mtp3-ws13-oc6000-decode-c1.json)
- [C1-C16 matrix JSON](runtime-results/v71/benchmarks/nvfp4-dcp2-mtp3-ws13-oc6000-c1-c16-through128k.json)
- [GitHub-renderable benchmark TUI](runtime-results/v71/benchmarks/nvfp4-dcp2-mtp3-ws13-oc6000-c1-c16-through128k-tui.txt)

## Current MLA KV-cache KLD

These are independent five-run means over the complete 2,048-token
`final-0000` qualification window: 2,047 causal positions per run against
the sealed BF16 teacher logits. The matched regime is TP2/EP2, DCP2, eager,
no MTP, route-128 SMEM. MTP is disabled so draft sampling cannot change the
runtime logits being compared.

| MLA KV cache | Five-run mean KLD | Population stddev | Mean top-1 agreement | Gate |
|---|---:|---:|---:|---:|
| FP8 | **0.024581652920** | 0.000159556478 | 0.936297020029 | pass |
| NVFP4 calibrated power-of-two | **0.054757372223** | 0.000000000000 | 0.914997557401 | pass |

Receipts:

- [FP8 five-run KLD](runtime-results/v71/kld/fp8-dcp2-route128-five-run-kld.json)
- [NVFP4 five-run KLD](runtime-results/v71/kld/nvfp4-dcp2-route128-power2-five-run-kld.json)
- [46-layer NVFP4 scale bank](runtime-results/v71/calibration/glm53-nvfp4-mla-46-layer-power2-scales.json)

## Quality results

All tests below used the official generation defaults, temperature `1.0` and
top-p `0.95`. The weights are unchanged in v71. Estonia and LAVD were
collected on the v44 NVFP4/DCP2/CUDA-graph/MTP3 runtime line; Hotel was
collected on an older NVFP4/DCP1 runtime. The v71 image packages the later
measured runtime bytes and passed current FP8/NVFP4 KLD, but these three task
suites were not rerun after pulling the final registry digest. They are
checkpoint/runtime-line evidence, not a claim of a digest-specific rerun.

| Test | Result | Runtime interpretation |
|---|---:|---|
| Estonia 10x | **10/10 correct** | No errors or 40K-token cap hits; 127.87 aggregate generation tok/s |
| Hotel Lights 10x | **7/10 exact** | No errors or cap hits; answers were 48 seven times, then 45, 32, and 47 |
| LAVD normal 10x | **1/10 exact raw; 3/10 accepted after audit** | Nine runs hit the 40K-token ceiling; primarily a reasoning/harness failure |
| LAVD low-reasoning 10x | **1/10 near raw; 4/10 accepted after audit** | No cap hits; still not a passing quality gate |
| Needle matrix through 499K | **17/18 raw** | Exact 498,368-token retry recovered the final needle in seven output tokens |

LAVD remains disclosed as a failed gate. The audit only recovers exact/near
answers already present in the responses; it does not rewrite wrong answers
as passes.

Receipts:

- [Estonia 10x](runtime-results/v44/quality/estonia-10x.json)
- [Hotel Lights 10x](runtime-results/v30/quality/hotel-10x.json)
- [LAVD normal raw](runtime-results/v44/quality/lavd-10x-normal.json) and [response audit](runtime-results/v44/quality/lavd-10x-normal-rescored.json)
- [LAVD low-reasoning raw](runtime-results/v44/quality/lavd-low-10x.json) and [response audit](runtime-results/v44/quality/lavd-low-10x-rescored.json)
- [needle matrix](runtime-results/v44/quality/needle-through-500k.json) and [final-cell retry](runtime-results/v44/quality/needle-499k-depth-0.9-retry.json)

## Optimization status

It is fair to call v71 the current validated tuning ceiling for this
implementation on this two-GPU workstation pair. Rank-local uniform-K loading,
route-128 SMEM/register kernels, DCP2, MTP3, and calibrated NVFP4 MLA KV have
all been exercised extensively, and ordinary flag/block-size tuning has
plateaued.

That is not a fundamental EXL3 or hardware ceiling. Prefill remains around
6.2K tok/s rather than the hoped-for 10K+, and the underfilled high-concurrency
cells show scheduler/admission headroom. A material next gain would likely
require new kernel, attention, or scheduler work rather than another launch
flag sweep.

## Architecture and checkpoint

- BF16 source: `zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43`
- All routed experts, including MTP45: uniform four-bit EXL3/TR3 MCG
- Non-routed tensors: official native dtype
- 45-layer pattern: 34 linear layers and 11 sparse-attention layers
- Sparse attention: IndexPool-4, top-k 2,048
- Sampling defaults: temperature `1.0`, top-p `0.95`

The separate offline checkpoint evidence measured a five-cold-run
teacher-to-student mean KLD of `0.024554564250` over 51,175 causal positions
per run. Teacher-logit provenance and the former long-form model card are
preserved in [the historical evidence page](docs/HISTORICAL_MODEL_CARD_2026-08-27.md).

## Artifact index

- [v71 Docker release receipt](runtime-results/v71/validation/docker-release.json)
- [v71 benchmarks](runtime-results/v71/benchmarks/)
- [v71 KLD](runtime-results/v71/kld/)
- [v71 NVFP4 calibration](runtime-results/v71/calibration/)
- [v44 qualification archive](runtime-results/v44/)
- [historical model card](docs/HISTORICAL_MODEL_CARD_2026-08-27.md)

Credit goes to turboderp for EXL3. Local Inference Lab, Martin Vit, and Luke
Alonzo contributed or helped test components of the base runtime.
