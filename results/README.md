# Measured results

## Current SM120 v71 runtime results

The current TP2/EP2 route128 validation artifacts are published under
[`runtime-results/v71`](../runtime-results/v71). They include the complete
five-run FP8 and calibrated-NVFP4 MLA KV-cache KLD receipts, v71 workstation
pair prefill and C1 decode receipts, the later C1-C16 stress matrix through the
nominal 128K harness row, and a postable plain-text `.log` summary. See the main
model card for the exact regime, results, capacity limitations, and thermal
caveats.

## Direct packed TP2 serving result

- Mean BF16-teacher-to-runtime KLD: **`0.022750847877671544`**
- Scope: sealed `final-0000`, 2,047 causal prediction positions
- Context: 2,048 input tokens
- Top-1 agreement: `0.9384465070835368`
- Report receipt SHA-256: `2cfd1b90a65162784de5b67e1b05ef7a15f98979b77156092d5f7500c607ba67`
- Runtime-logits file SHA-256: `280c12452f429a79724cd1715a5bed2462de98a547746a80f04e7b8cdcd34746`
- Report: [`tp2-runtime-window-kld.json`](tp2-runtime-window-kld.json)
- Qualification: [`tp2-runtime-qualified.json`](tp2-runtime-qualified.json)

This is the measured `0.022` result from the actual packed two-GPU runtime. It
is the serving qualification result, not one of the five offline full-panel
runs.

## Five cold offline decoded-K4 runs

| Run | Mean KLD | Report |
|---:|---:|---|
| 1 | `0.024554564249958208` | [`run-1-kld-report.json`](run-1-kld-report.json) |
| 2 | `0.024554564249958208` | [`run-2-kld-report.json`](run-2-kld-report.json) |
| 3 | `0.024554564249958208` | [`run-3-kld-report.json`](run-3-kld-report.json) |
| 4 | `0.024554564249958208` | [`run-4-kld-report.json`](run-4-kld-report.json) |
| 5 | `0.024554564249958208` | [`run-5-kld-report.json`](run-5-kld-report.json) |

Each accepted run covers the same 25 sealed windows and 51,175 causal positions.
The aggregate receipt is [`five-cold-run-kld.json`](five-cold-run-kld.json).

The original fifth attempt received an external SIGTERM before writing logits
and is excluded. Accepted run 5 is the clean `run5b` retry. Its different name
records that operational retry; its measured KLD equals the other four accepted
runs.

## Exact five-run reproduction scripts

The source used for the accepted offline decoded-K4 measurements is preserved
in this repository:

1. [`capture_glm53_packed_k4_student_logits_ep4.py`](../scripts/capture_glm53_packed_k4_student_logits_ep4.py)
   launches with `torchrun --nproc-per-node=4`, loads the immutable BF16 source
   checkpoint under Transformers EP4, independently decodes the hash-verified
   packed K4 experts into each rank's local BF16 expert parameters, and captures
   float32 student logits for the 25 qualification-only windows.
2. [`measure_glm53_packed_student_kld.py`](../scripts/measure_glm53_packed_student_kld.py)
   computes tokenwise `KL(BF16 teacher || packed-K4 student)` in float64 over all
   51,175 jointly valid causal positions and seals the per-token, per-window,
   per-domain, top-1-agreement, and aggregate results.
3. [`aggregate_glm53_five_run_kld.py`](../scripts/aggregate_glm53_five_run_kld.py)
   accepts exactly five independently sealed KLD reports and produces the
   five-cold-run mean and dispersion receipt.

Each accepted run begins with a new capture process and model load. The capture
script is deliberately fail-closed: it requires the exact full-hash BF16
inventory, uniform-K4 contract, content-addressed packed surface, qualified MTP45
adapter receipt, and sealed token-panel receipt used by the campaign. This is
the offline decoded-EP4 measurement that produced `0.024554564249958208`; it is
not the separate packed TP2 runtime measurement that produced
`0.022750847877671544`.
