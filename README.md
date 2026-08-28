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
The current daily-driver runtime is v84: TP2/EP2/DCP2, calibrated NVFP4 MLA
KV, DFlash2-7, CUDA graphs, and working image input on two SM120 GPUs. It is a
custom vLLM/B12X build and is not compatible with stock upstream vLLM.

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

## Current measured results

Qualified on two RTX PRO 6000 Blackwell Workstation Edition GPUs (96 GB each),
300 W limits, +6000 MHz memory offset, TP2/EP2/DCP2, NVFP4 MLA KV, prefix cache
off, and DFlash2-7. Generation uses the model defaults (`temperature=1.0`,
`top_p=0.95`); the acceptance comparison uses `reasoning_effort=max`.

| Measurement | Result |
|---|---:|
| Cold prefill, 8K | **3,897 tok/s** |
| Cold prefill, 64K | **4,297 client / 4,320 server tok/s** |
| C1 decode, empty context | **129.45 tok/s** |
| C1 decode, 64K context | **122.24 tok/s** |
| DFlash2 acceptance, GSM8K first 16 | **5.739 mean / 5.550 token-weighted** |
| DFlash2 acceptance, published reference | 5.78 mean over 128 samples |
| Image smoke | **pass** — correctly identified a mallard |

The DFlash acceptance fix is material: the partially ported Triton mask scored
1.017 weighted. Restoring the reference semantics—full bidirectional visibility
inside the draft block with a backward-only historical window—raised the same
five-seed probe to 5.068 and the exact GSM8K sample to 5.739.

Receipts: [benchmark JSON](runtime-results/v84/benchmarks/llm-decode-c1-c4-64k.json),
[native benchmark TUI](runtime-results/v84/benchmarks/llm-decode-c1-c4-64k.tui.log),
[acceptance rows](runtime-results/v84/quality/gsm8k-first16-max-acceptance.jsonl),
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
