# GLM-5.3-Flash-EXL3-4bpw

Reproducibility code and sealed quality receipts for
[`brandonmusic/GLM-5.3-Flash-EXL3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw).

The checkpoint uses uniform four-bit EXL3/TR3 MCG for every routed expert matrix,
including the stored MTP45 expert tensors. Non-routed tensors retain the official
source dtype. The source checkpoint is
`zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43`.

## Five cold KLD runs

KLD is teacher-to-student tokenwise KL over the same sealed panel of 25 windows.
Each window has 2,048 input tokens and 2,047 causal prediction positions, for
51,175 positions per run.

| Run | Mean KLD | Positions | Report receipt SHA-256 | Capture receipt SHA-256 |
|---:|---:|---:|---|---|
| 1 | 0.024554564249958 | 51,175 | `ef6a8dedc20f11e582658f94923da3e66c2b6cea4ff62d936abb790e376e2461` | `013759025d8414f8811fa140250e2c79097c1082926edd4ae2cfc6751722fc8d` |
| 2 | 0.024554564249958 | 51,175 | `b7d1cac829f6b21471da4ea724aac479f9db250d4286edd412e099fa747f8257` | `eae08903737bde9f31bf6f8632d2de7b6539f4b1efd8113c5f81461d92aaf671` |
| 3 | 0.024554564249958 | 51,175 | `663629ccd2bda08a4c299d767b7e6e6d622a81ad6830ad1acf08d0eb8ca1a196` | `000896721ea7116322eb31d8e75718985d29240fabd6a921627bb02c03516bec` |
| 4 | 0.024554564249958 | 51,175 | `cdb2d8ee4ce795f695f335f0bb3ce7bd135dcf6df4f48c6e3862b40cd1340586` | `7ece4defa651c3693bffd624ad7d07ff85c0dceb7674a7752ec136dea6370c3f` |
| 5 | 0.024554564249958 | 51,175 | `ac4d6d94aef27b09ca9b2dd513516e793cf5f4afe3d1f2b008a3fb4ed64ae243` | `5b59145332206b4c0fb82f791e2c09be8fadb16d18e6e58818b78e919294cb65` |

The mean of means is `0.024554564249958`, with population standard deviation
`0.0`. All five accepted runs have the same tokenwise-KLD SHA-256,
`2a596810dcdd52fc654eb94fffe1cf394b826ea6b25d8f411049d8354e52f562`,
while their backend and capture receipts are distinct cold-execution records.

### Why the fifth run is named `run5b`

The first attempt at the fifth capture received an external SIGTERM before it
wrote any logits. It left only a plan and reader-identity file, so it is not a
measurement and is excluded from the five-run aggregate. The accepted fifth row
is a clean restart (`run5b`): it wrote all 25 logit files, sealed a distinct
capture receipt, and produced the same KLD as accepted runs 1-4. Thus the
difference is operational provenance, not a different score.

The exact aggregate and five individual reports are under [`results/`](results/).

## Packed TP2 runtime qualification

The custom two-GPU serving path keeps the EXL3 packed kernels active. Gate and up
projections return to the model's BF16 boundary; selected down-projection
partials are accumulated in FP32 and reduced once per layer before the BF16
boundary. This removes repeated per-expert collectives and does not reconstruct
dense BF16 expert weights.

On the sealed `final-0000` window, direct packed TP2 teacher-to-runtime mean KLD
was `0.022750847877672` over 2,047 positions, with top-1 agreement
`0.9384465070835368`. Both TP ranks produced byte-identical logits, all 36,288
packed routed matrices loaded, zero persistent BF16 routed-weight parameters
were present, and 2,048-token prefill plus four-token generation completed.

That one-window runtime result is separate from the five full-panel offline KLD
runs above. A stricter decoded-reference raw-logit absolute-error diagnostic did
not pass (`mean 0.2410409301519394`, `max 10.546875`) and is deliberately retained
in the qualification receipts. This is a semantic KLD qualification, not a claim
of raw-logit identity, stock vLLM compatibility, or stock ExLlamaV3 model support.

CUDA graphs and RTX PRO 6000 Blackwell performance have not yet been qualified;
they should be treated as later latency work, not as evidence for a different KLD.

## Layout

- `results/five-cold-run-kld.json`: sealed five-run aggregate.
- `results/run-{1..5}-kld-report.json`: the five accepted reports.
- `results/tp2-runtime-window-kld.json`: direct packed TP2 window metric.
- `results/tp2-runtime-qualified.json`: sealed TP2 qualification receipt.
- `results/materialization-receipt.json`: complete checkpoint census and hashes.
- `src/quant_pipeline/`: runtime and receipt implementation.
- `scripts/`: launch, qualification, KLD, and aggregation entry points.

The HF model bundles the same runtime package. It requires Transformers 5.16.1
and ExLlamaV3 commit `c5d9c657966ffeeaa9353f0cc899f18629da4a13` with its CUDA
extension compiled.

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 \
  scripts/run_glm53_custom_tp_runtime.py \
  --model /path/to/GLM-5.3-Flash-EXL3-4bpw \
  --exllamav3-source /path/to/exllamav3 \
  --prompt "Hello"
```

## License

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
