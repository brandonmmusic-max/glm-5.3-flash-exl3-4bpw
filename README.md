---
base_model: zai-org/GLM-5.3-Flash-BF16
library_name: transformers
pipeline_tag: image-text-to-text
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
  - dflash2
  - multimodal
---

# GLM-5.3-Flash TR3 4bpw — current SM120 runtime

This is the uniform-K4 EXL3/TR3 routed-expert checkpoint for GLM-5.3-Flash.
The current v84 runtime supports three explicit TP2/EP2/DCP2 profiles on two
SM120 GPUs: multimodal DFlash2, language-only DFlash2, and language-only MTP3.
All use calibrated NVFP4 MLA KV and CUDA graphs. This is a custom vLLM/B12X
build and is not compatible with stock upstream vLLM.

## Pick a serving profile

| Goal | Launcher | Extra checkpoint | Measured KV tokens |
|---|---|---|---:|
| Images plus fastest measured C1 decode | `compose.sm120-tp2.yaml` | DFlash2-7 | 129,473 |
| Text-only DFlash2 decode | `compose.sm120-tp2-language-only-dflash2.yaml` | DFlash2-7 | 184,619 |
| **Text-only capacity/default** | **`compose.sm120-tp2-language-only.yaml`** | **none; built-in MTP3** | **1,376,256** |

The MTP3 option means the model's built-in MTP head only: it does not load or
mount the external DFlash checkpoint. Choose DFlash2 when its modest C1 decode
gain matters more than resident context/concurrency; choose MTP3 for the normal
text-only daily driver.

## Run the current image

```text
verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2
OCI digest: sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
```

The runtime image does not contain either checkpoint. Download/mount this
EXL3 model and `incoai/GLM-5.3-Flash-DFlash2` separately. The DFlash2
checkpoint is distributed under CC-BY-NC-ND-4.0; review its license before use.

Docker Compose:

```bash
curl -L -o compose.sm120-tp2.yaml \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/compose.sm120-tp2.yaml

GLM53_MODEL_PATH=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
GLM53_DFLASH_PATH=/absolute/path/to/GLM-5.3-Flash-DFlash2 \
docker compose -f compose.sm120-tp2.yaml up -d

curl http://127.0.0.1:8012/v1/models
```

Standalone serve script:

```bash
curl -L -o serve-glm53-sm120-tp2.sh \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/serve-glm53-sm120-tp2.sh
chmod +x serve-glm53-sm120-tp2.sh

MODEL=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
DFLASH_MODEL=/absolute/path/to/GLM-5.3-Flash-DFlash2 \
GPU_DEVICES=0,1 \
./serve-glm53-sm120-tp2.sh
```

The published profile has a 98,304-token request ceiling and allocated 129,473
KV tokens on the qualified pair. Its hybrid Mamba/DFlash rollback layout has
room for one full resident request; additional requests queue. C2/C4 rows in
the raw benchmark are therefore capacity-limited and are not throughput claims.

## Language-only profile

For text serving, use the language-only profile. It disables the vision tower
and uses the built-in MTP3 head by default, avoiding a second external
checkpoint and leaving substantially more room for KV cache:

```bash
curl -L -o compose.sm120-tp2-language-only.yaml \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/compose.sm120-tp2-language-only.yaml

GLM53_MODEL_PATH=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
docker compose -f compose.sm120-tp2-language-only.yaml up -d
```

Standalone:

```bash
curl -L -o serve-glm53-sm120-tp2-language-only.sh \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/serve-glm53-sm120-tp2-language-only.sh
chmod +x serve-glm53-sm120-tp2-language-only.sh

MODEL=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
GPU_DEVICES=0,1 \
./serve-glm53-sm120-tp2-language-only.sh
```

To keep DFlash2 while disabling vision, use the separate speed-first launcher:

```bash
curl -L -o compose.sm120-tp2-language-only-dflash2.yaml \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/compose.sm120-tp2-language-only-dflash2.yaml

GLM53_MODEL_PATH=/absolute/path/to/GLM-5.3-Flash-tr3-4bpw \
GLM53_DFLASH_PATH=/absolute/path/to/GLM-5.3-Flash-DFlash2 \
docker compose -f compose.sm120-tp2-language-only-dflash2.yaml up -d
```

Its standalone equivalent is
[`serve-glm53-sm120-tp2-language-only-dflash2.sh`](runtime/serve-glm53-sm120-tp2-language-only-dflash2.sh).

The language-only alias points to the same tested v84 code digest:

```text
verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-language-only
OCI digest: sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
```

Measured capacity on the same two 96 GB GPUs at 300 W each:

| Runtime profile | Vision | Speculator | KV tokens | Concurrency at tested ceiling |
|---|:---:|---|---:|---:|
| Multimodal DFlash2-7, 98,304 max | on | external 7-layer draft | 129,473 | 1.32x |
| Language-only DFlash2-7, 98,304 max | off | external 7-layer draft | 184,619 | 1.88x |
| **Language-only MTP3, 131,072 max** | **off** | **built-in head** | **1,376,256** | **10.50x** |

Turning vision off raises the DFlash KV token pool by 42.6%. The much larger
7.45x language-only gain comes from using the built-in MTP head instead of
keeping the external DFlash2-7 model resident. It is not a vision-only gain.
The language-only profiles do not accept image inputs. The reported KV-token
pool is total allocated capacity, not a promise that every request can use the
entire pool; the configured per-request ceiling and scheduler concurrency still
apply.

## Current measured results

Qualified on two RTX PRO 6000 Blackwell Workstation Edition GPUs (96 GB each),
TP2/EP2/DCP2, NVFP4 MLA KV, prefix cache off, and DFlash2-7. The current quick
speed pass used 600 W limits and +6000 MHz memory offsets. Generation uses the
model defaults (`temperature=1.0`, `top_p=0.95`); the acceptance comparison uses
`reasoning_effort=max`.

| Measurement | Result |
|---|---:|
| Cold prefill, 32K | **6,225 client / 6,277 server tok/s** |
| Cold prefill, 64K | **6,083 client / 6,130 server tok/s** |
| C1 decode, empty context | **145.5 tok/s** |
| C1 decode, 32K context | **147.2 tok/s** |
| C1 decode, 64K context | **151.5 tok/s** |
| DFlash2 acceptance, 5 distinct GSM8K prompts | **5.428 mean / 5.441 token-weighted; 5/5 correct** |
| DFlash2 acceptance, GSM8K first 16 | **5.739 mean / 5.550 token-weighted** |
| DFlash2 acceptance, published reference | 5.78 mean over 128 samples |
| Image smoke | **pass** — correctly identified a mallard |

The clean C1 decode run used a 4,096-token completion budget so the client did
not roll into the next prefill request. A prior 60.1 tok/s row was a harness
rollover artifact and is excluded. The DFlash acceptance fix is material: the partially ported Triton mask scored
1.017 weighted. Restoring the reference semantics—full bidirectional visibility
inside the draft block with a backward-only historical window—raised the same
five-seed probe to 5.068 and the exact GSM8K sample to 5.739. Synthetic padded
long-context decode accepts roughly 2.8–3.0 tokens/step, while five distinct
GSM8K reasoning prompts accepted 4.89–6.03 and all answered correctly; acceptance
is workload-dependent.

Receipts: [600 W prefill JSON](runtime-results/v84/benchmarks/llm-decode-c1-prefill32k64k-600w.json),
[600 W prefill TUI](runtime-results/v84/benchmarks/llm-decode-c1-prefill32k64k-600w.tui.log),
[clean 600 W C1 decode JSON](runtime-results/v84/benchmarks/llm-decode-c1-clean-4096-600w.json),
[clean 600 W C1 decode TUI](runtime-results/v84/benchmarks/llm-decode-c1-clean-4096-600w.tui.log),
[earlier C1-C4 benchmark](runtime-results/v84/benchmarks/llm-decode-c1-c4-64k.json),
[acceptance rows](runtime-results/v84/quality/gsm8k-first16-max-acceptance.jsonl),
[distinct-prompt acceptance](runtime-results/v84/quality/gsm8k-distinct5-language-only-acceptance.json),
[language-only capacity](runtime-results/v84/validation/language-only-capacity.json),
and [release validation](runtime-results/v84/validation/release.json).

## Quality and KLD

v84 changes draft speculation, Triton draft-attention semantics, and vision
packaging; it does not change target-model weights, EXL3 kernels, calibrated
MLA KV scales, or target logits. The current target-quality receipts therefore
remain the repeatedly qualified v75 measurements:

| Test | Result |
|---|---:|
| FP8 MLA KV KLD, five-run full 2,047-position mean | **0.024610591221** |
| NVFP4 MLA KV KLD, five-run full 2,047-position mean | **0.054757372223** |
| Estonia 10x, NVFP4 | **10/10** |
| LAVD-low 10x, FP8 | **8/10 accepted** |
| LAVD-low 10x, NVFP4 | **3/10 accepted** — failed quality gate |
| Needle through 500K, NVFP4 | **17/18 raw; final cell passed on longer retry** |

KLD was measured in eager/no-speculation mode against the sealed BF16 teacher
over every causal position in the 2,048-token window. Draft acceptance does not
alter that target-logit measurement. Hotel was explicitly stopped and is not
presented as a current result.

Receipts: [v75 KLD and quality evidence](runtime-results/v75/). Older tuning
history is retained in [the historical model card](docs/HISTORICAL_MODEL_CARD_2026-08-27.md),
not mixed into the current launch path.

## Vision and implementation notes

The image fixes a packaging defect where GLM-5.3 vision RoPE unconditionally
imported `vllm.vllm_flash_attn.layers.rotary` even when a custom wheel shipped
only the compiled flash-attention extensions. It now uses native PyTorch RoPE
as a correctness fallback. Cold multimodal warmup and a real remote-JPEG chat
request both passed.

The target path remains the fused uniform-K4 EXL3 route-128 SMEM/register
kernel. SM120 in this build does not use a TMEM/TCGEN path. DFlash uses Triton
attention because its noncausal sliding-window semantics are now tested there.

## Provenance and attribution

The image embeds `/opt/glm53/PROVENANCE.json` and OCI source, author,
documentation, revision, checkpoint, and validation labels. The manifest binds
the runtime source and benchmark artifacts with SHA-256 hashes. This is a
transparent provenance fingerprint: there is no telemetry, callback, hidden
output watermark, or inference modification.

```bash
curl -L -o verify-provenance.sh \
  https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime/verify-provenance.sh
chmod +x verify-provenance.sh
./verify-provenance.sh
```

This checkpoint is distributed under the ShapleyMCG License 1.0 in
[LICENSE](LICENSE). Credit goes to turboderp for EXL3, IncoAI for DFlash2, and
Local Inference Lab contributors for the runtime foundation.
