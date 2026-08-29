# KL Divergence as a Quantization-Fidelity Metric

## Measurement protocol, statistical validity, determinism, engine error, and plan of action

**Prepared for:** Brandon M. Music
**Date:** 29 August 2026
**Scope:** teacher (full-precision) vs. student (quantized) LLM evaluation, with emphasis on MoE models (Qwen3.5-397B-A17B, GLM-5.2-class), trellis/EXL3 formats (TR3, ShapleyMcg allocation) and NVFP4/BMM formats, on 4× RTX PRO 6000 Blackwell (SM120, PCIe Gen5, no NVLink) running SGLang, ExLlamaV3, and b12x.

Bracketed references such as [S4] point to the source list in Appendix B.

---

## 0. Direct answers to the questions asked

| Question | Short answer |
|---|---|
| **Should the output KLD number be deterministic across runs?** | Yes, for a fixed pipeline (same weights, same token IDs, same kernels, same hardware, same batch composition). A teacher-forced forward pass contains no random variable. Any run-to-run variation is a *pipeline* property (batch-dependent reduction order, atomics, kernel autotuning, multi-GPU reduction order, hardware faults), not a property of the metric or of the model. Across pipelines (different engine, batch size, GPU, library version) variation is expected and must be characterized rather than assumed away. |
| **Does MoE routing make the number non-deterministic?** | No. The quantized model routing a token to a different expert than the teacher is *systematic divergence*, which is exactly what KLD is supposed to measure. What MoE does do is make the *measurement* hypersensitive to numerical noise: a token whose router logits are near a top-k boundary can flip experts under a sub-ulp perturbation, so a non-reproducible pipeline shows run-to-run KLD variation that concentrates in the tail (max, 99.9th percentile) rather than the mean. The remedy is a reproducible pipeline plus routing-agreement diagnostics, not a different metric. |
| **Should it be run through a particular inference engine?** | Two runs, not one. A *reference-path* run (teacher and dequantized student in the same reference framework, e.g. HF transformers) isolates representation error. A *production-engine* run (quantized kernels in SGLang/ExLlamaV3/b12x) measures what users actually receive. The difference between the two is the engine's kernel contribution. Serving HTTP APIs should never be the measurement path (top-k truncated or post-processed logprobs, dynamic batching); use in-process APIs that return full-vocabulary logits. |
| **Should we build a mechanism to correct for engine error?** | Not by subtraction. Measure the engine's floor (unquantized model through the engine vs. the reference), gate on it, and report the decomposition. In the small-error regime KL is quadratic in logit error, so independent error sources add approximately; use that as a consistency check, not as a correction. |
| **Should it be just a Python script?** | Yes, but a pinned, versioned harness with a frozen protocol document, per-token outputs saved to disk, and analysis code separated from model execution. Not a notebook and not an ad hoc script per experiment. |
| **How large a sample?** | Pilot-measure the per-token coefficient of variation (CV) and the design effect from within-document correlation, then size. From observed data: roughly 250K–500K held-out tokens across at least three domains gives a ±2–3% relative 95% CI on mean KLD and detects ~1–2% differences between two allocators in a paired design. Tail statistics (99.9th percentile) need ≥1M tokens to be stable; "maximum KLD" is never comparable across different N. |
| **What are reasonable values?** | Model- and size-dependent; define acceptance relative to the same model's 8-bit quant, not as absolute numbers. Observed reference points from llama.cpp runs: mean KLD ≈ 0.002 nats with ≈98% top-1 agreement (near-lossless), and ≈0.04 nats with ≈91% top-1 agreement (typical 4-bit on an 8B dense model) [S15, S16]. |
| **What can make it error?** | Silent-wrong failures dominate: tokenizer mismatch, misaligned positions, padded vocab, truncated logprobs, fp16 logit overflow, calibration/eval overlap, KV-cache quantization left on, router numerics differing between engines. Section 8 is a catalog with a detection test for each. |

---

## 1. What the number is (and what is and is not random in it)

### 1.1 Definition

For token position *i* in an evaluation corpus, let ℓ<sub>i</sub> ∈ ℝ<sup>V</sup> be the teacher's logits over the vocabulary and ℓ̃<sub>i</sub> the student's. With p<sub>i</sub> = softmax(ℓ<sub>i</sub>) and q<sub>i</sub> = softmax(ℓ̃<sub>i</sub>):

```
d_i = KL(p_i || q_i) = Σ_v p_i(v) · [ ln p_i(v) − ln q_i(v) ]          (per-token KLD, nats)

D̂  = (1/N) Σ_{i=1..N} d_i                                            (mean KLD over the corpus)
```

Three properties matter for everything that follows:

1. **The expectation over the vocabulary is exact.** Given full-vocabulary logits, d<sub>i</sub> is computed in closed form. There is no Monte-Carlo estimation over tokens-in-the-vocabulary, and therefore no sampling noise from that source. The only statistical estimation in D̂ is over *contexts*: the N positions are a sample from the (infinite) distribution of text the model will see. This is the correct frame for sample-size and confidence-interval reasoning (Section 6).

2. **Direction.** KL(teacher ‖ student) is the convention used by llama.cpp [S1], ExLlamaV3 [S8], Unsloth [S10], and the NeurIPS evaluation paper that motivated the metric's adoption [S3]. It is mass-covering: the student is penalized heavily for assigning low probability where the teacher assigns high probability, which is the failure a user experiences as "the quant said something the original would never say." Reverse KL is appropriate for *training* (distillation) objectives but not for fidelity *evaluation*. The direction must be stated on every result.

3. **The measurement is teacher-forced on fixed text.** Both models are fed the identical token-ID sequence and scored at every position. No sampling occurs anywhere in the measurement. Temperature, top-p, repetition penalties, and seeds are irrelevant to it. If anyone's KLD procedure involves generation, it is measuring something else (trajectory divergence, cf. Unsloth's Divergence-300 @32, Section 2.3).

### 1.2 Companion statistics that should always travel with the mean

The mean is dominated by a heavy right tail (Section 6.1), so it is reported with:

- **Median and percentiles** of d<sub>i</sub> (90, 95, 99, 99.9) and the maximum.
- **Top-1 agreement** ("same top p" in llama.cpp): fraction of positions where argmax p<sub>i</sub> = argmax q<sub>i</sub>. This is the greedy-decoding flip rate and is a binomial proportion with a clean confidence interval.
- **Δp on the realized token**: q<sub>i</sub>(t<sub>i+1</sub>) − p<sub>i</sub>(t<sub>i+1</sub>), its mean, RMS, and percentiles. Symmetric Δp indicates noise-like error; asymmetric (negative-skewed) Δp indicates real degradation [S1].
- **ln PPL ratio** ln(PPL<sub>Q</sub>/PPL<sub>base</sub>) for continuity with legacy reporting, with the explicit caveat that perplexity can cancel (Section 2.1).

---

## 2. Why KLD, and what the field currently does

### 2.1 The academic basis

Dutta et al. (Microsoft Research, NeurIPS 2024) evaluated six quantization schemes across Llama-2 and Yi families on seven benchmarks and found aggregate accuracy differences of ≤2% while 5–14% of individual answers *flipped* between correct and incorrect in both directions, netting out to unchanged accuracy [S3]. They show the same cancellation applies to perplexity: perplexity is the inverse geometric mean of token probabilities, so lower probabilities on some tokens are cancelled by higher probabilities on others. Their conclusion is that accuracy and perplexity are necessary but not sufficient, and that compression methods should be evaluated with distance metrics, specifically KL divergence and flip rate, which they show are well correlated [S3]. This paper is the citation Unsloth uses when calling KLD the gold standard for reporting quantization error [S10].

### 2.2 llama.cpp

KLD support was added to `llama-perplexity` by ikawrakow in PR #5076 (merged 22 January 2024), following an earlier Python implementation contributed by Ttl in PR #4739 [S2]. The workflow is: run the FP16/BF16 model once with `--kl-divergence-base FILE` to record logits, then run each quant with `--kl-divergence-base FILE --kl-divergence`. The README documents that the base file stores logits as 16-bit unsigned integers with scaling rather than the original FP32 values, that the file is very large (11 GiB for LLaMA-2, 37 GiB for LLaMA-3 on Wikitext-2), and that the reported uncertainty on mean KLD assumes the per-token KLD is Gaussian [S1]. Both caveats matter (Sections 4 and 6). The README also states plainly that llama.cpp numbers are not directly comparable with those of other projects because the values depend strongly on implementation details [S1].

### 2.3 ExLlamaV3 / EXL3

ExLlamaV3 measures KLD and top-K agreement between a reference model and a quant with `eval/model_diff.py` (and the multi-model `eval/compare_q.py`), typically with eval_len 2048 and stride 512 on a chosen dataset [S8, S23]. Two findings from the community around it are directly relevant:

- EXL3's default calibration data and its perplexity/KLD evaluation data overlap, so perplexity on that data is optimistic; a 3-bit quant can show *lower* perplexity than the FP16 original. KLD against the original distribution is the appropriate quality measure, and results are sensitive to the number of rows evaluated, so comparisons must use identical row counts [S8].
- A GGUF BF16 conversion of Qwen3-30B-A3B showed a small but non-zero KLD against the HF transformers BF16 baseline [S9]. That is an engine floor (Section 5.1) observed in the wild: the same weights, in two implementations, do not produce the same distribution.

### 2.4 Unsloth Dynamic 2.0 / 3.0

Unsloth adopted KLD as its headline metric and documented two protocol failures that inflate reported quality: (i) calibrating on Wikipedia-like data and then evaluating on Wikitext, which overfits the quant to the evaluation domain, and (ii) calibrating instruct models on plain text without their chat template [S10]. Their Dynamic 3.0 release (August 2026) adds Divergence-300 @32: 300 held-out prompts from agentic coding, math, non-Latin, and long-prompt sources, decoded greedily for 32 tokens by both the BF16 model and the quant, scoring trajectory agreement [S10]. This is a *trajectory* metric and is complementary to (not a substitute for) teacher-forced KLD.

### 2.5 What this establishes as the standard of practice

- KLD(teacher ‖ student) over full-vocabulary logits on held-out text is the accepted fidelity metric.
- Calibration and evaluation data must be disjoint, and the evaluation set must include the target usage domain (chat-formatted, code, etc.).
- Mean KLD is reported with percentiles and top-1 agreement.
- Numbers are only comparable within one protocol (same tokens, context/stride, tool, precision).

---

## 3. Determinism: should the number vary, and why would it?

### 3.1 The clean statement

Fix the model weights, the token-ID sequence, the chunking, the software build, the hardware, and the batch composition. The forward pass is a deterministic function of its inputs. The per-token KLD values, and therefore D̂, should then be bitwise identical on every run. If they are not, one of the mechanisms in Section 3.3 is present and should be identified, not averaged over.

This is the position taken by Thinking Machines (He et al., September 2025) after showing that even greedy decoding on a single GPU is not reproducible in practice: the cause is almost never true randomness but the lack of *batch invariance* in kernels, so a request's numerical output depends on what else is in the batch [S4]. They report that atomics are rarely used in the forward-pass hot path, that individual kernels are typically run-to-run deterministic for a fixed shape, and that RMSNorm, matmul, and attention are the three operations whose reduction strategy changes with batch size [S4]. Their batch-invariant kernels achieve exact reproducibility at a cost they measured at roughly 2× on an unoptimized path [S4, S22]. SGLang shipped `--enable-deterministic-inference` on this basis (v0.5.3+), compatible with chunked prefill, CUDA graphs, radix cache, and seeded non-greedy sampling, on the FlashInfer, FA3, and Triton attention backends, with a measured average slowdown of about 34% [S5]. The documentation explicitly lists an MoE model (Qwen3-30B-A3B) as supported and ships `sglang.test.test_deterministic` to verify "Unique samples: 1" across varying batch sizes, prefix lengths, and cache states [S5].

### 3.2 The MoE question, answered directly

Your intuition ("the quant might not make the same routed choice through every layer because expert projections overlap or the hidden state differs a tiny bit") is correct about the mechanism and incorrect about the category.

**Mechanism.** A top-k router computes scores s = W<sub>r</sub>h (plus, in DeepSeek/Qwen3-style routers, sigmoid or softmax normalization and a per-expert bias) and selects the k largest. This is a discontinuous function of h. Quantization perturbs h at every layer; when the margin between the k-th and (k+1)-th score is smaller than the perturbation, the selected expert set changes, the computation path changes, and the divergence at that token can be large. This is documented empirically: EAQuant shows that small quantization-induced perturbations in router outputs significantly alter top-k assignment across OLMoE, DeepSeek-MoE, and Mixtral [S13]; VSRAQ (June 2026) states that MoE models are sensitive to routing instability because small quantization-induced perturbations can change the top-k selection and alter the computation path [S12]; QuantMoE-Bench characterizes expert-usage imbalance and precision sensitivity across MoE architectures [S14]; SAMEQuant documents quantization-induced routing shifts and explicitly aligns the quantized router's selections with the full-precision router's [S21].

**Category.** None of that is non-determinism. Given the same inputs and the same kernels, the quantized model makes the same (different-from-teacher) routing choice every time. That divergence *is the signal*. KLD at that token will be large, correctly, because the student has become a different function at that input.

**Where MoE genuinely interacts with determinism.** Near-boundary tokens are chaotic amplifiers. If the pipeline is not bitwise reproducible (Section 3.3), a perturbation of one ulp in a router score can flip an expert and change the output distribution at that position by orders of magnitude more than the same perturbation would in a dense model. Consequences:

- Run-to-run variation in a non-reproducible MoE pipeline shows up as a small number of tokens whose d<sub>i</sub> changes drastically. The mean moves a little; the max and 99.9th percentile can move a lot.
- MoE serving kernels are *more* batch-dependent than dense kernels, not less: fused-MoE implementations in vLLM and SGLang select kernel configurations from the per-expert token counts and pad expert-token workloads to block sizes before the grouped GEMM [S19]. The number of tokens each expert receives depends on the batch, so the reduction order inside each expert's GEMM depends on the batch.
- In HF transformers-style MoE implementations that combine expert outputs with `index_add_` or `scatter_add_`, PyTorch documents these CUDA ops as nondeterministic [S6], and an H100 sweep identified scatter_reduce and index_add as the dominant nondeterministic PyTorch operators, with elementwise variability on the order of 3–5×10<sup>−6</sup> [S18]. That is small per element, but it sits exactly in front of the next layer's router.

**Your "overlapping expert projections" point cuts the other way.** If two experts compute nearly the same function on a given token (redundant experts), a flip between them costs little KLD. The harmful flip is to a dissimilar expert. So the flip rate by itself is not the quality metric; the *output-weighted* effect of flips is, and per-token KLD already integrates that. The right MoE diagnostic is therefore routing agreement *alongside* KLD (Section 9): low agreement with low KLD means the flips are benign; low agreement with high tail KLD means they are not.

### 3.3 Catalog of causes of run-to-run and configuration-to-configuration variation

| # | Mechanism | Affects | Detection | Remedy |
|---|---|---|---|---|
| 1 | Batch-dependent reduction order (matmul split-K/tiling, RMSNorm, attention split-KV) [S4] | same-machine, different batch composition | Run bs=1 vs bs=8 vs shuffled order; compare per-token d<sub>i</sub> bitwise | Fixed batch composition; batch-invariant kernels (`--enable-deterministic-inference`, `batch_invariant_ops`) |
| 2 | Atomic accumulation (index_add_/scatter_add_ in MoE combine, some split-K GEMMs) [S6, S18] | same machine, same batch | Repeat run 3×; any bit difference | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` [S6]; kernels that combine via sorted/segmented reduction |
| 3 | cuBLAS/cuBLASLt heuristic algorithm selection and workspace [S6] | across processes | Same as 2 | `CUBLAS_WORKSPACE_CONFIG`; pin library versions |
| 4 | cuDNN autotuning (`cudnn.benchmark=True`) | across processes | Same as 2 | `cudnn.benchmark=False`, `cudnn.deterministic=True` |
| 5 | SDPA backend dispatch (flash / mem-efficient / cuDNN / math) [S6] | across configs | Force one backend | Pin backend explicitly |
| 6 | TF32 in fp32 matmuls (Ampere+) | reference path | `allow_tf32` flags | Disable TF32 in the reference path |
| 7 | Prefill vs decode kernels, chunked prefill boundaries, CUDA-graph vs eager | across configs | Vary chunk size; compare | Fix chunk size; evaluate in one mode |
| 8 | KV-cache quantization (FP8/Q8/Q4 KV) enabled by default in some engines | all | Read engine config | Disable for measurement; measure separately as its own error source |
| 9 | Tensor-parallel all-reduce order over NCCL; expert-parallel all-to-all and combine | multi-GPU | Compare TP=1 layer-streamed vs TP=4 | Prefer pipeline-parallel (no cross-GPU reductions) for the reference path; fix NCCL algo/proto env; treat TP results as their own configuration |
| 10 | Hardware/driver/library version drift; different GPU SKUs | cross-machine | Record and hash the environment | Pin; never compare across environments without a bridging run |
| 11 | Hardware faults (memory errors, thermal instability) | any | Canary: teacher self-KLD at session start/end must be exactly 0 | Discard sessions with a non-zero canary. JohannesGaessler withdrew a published 70B logit file for exactly this reason [S17] |
| 12 | MoE expert-token padding/config selection by problem size [S19] | MoE engines | bs sweep as in 1 | As in 1; or bs=1 measurement |

### 3.4 Policy recommendation on determinism

1. **Bitwise reproducibility is the target for the reference path** (teacher logits, dequantized-student logits). It is achievable with a single-process, pipeline-parallel or layer-streamed HF/torch forward with deterministic flags set (Section 10).
2. **For the production engine, measure the spread, then decide.** Run the student evaluation 3–5 times with deliberately varied batch composition (bs=1, bs=8, shuffled chunk order, cache on/off). Report the run-to-run standard deviation of D̂ as σ<sub>run</sub>. Gate: σ<sub>run</sub> must be small relative to the statistical standard error (Section 6) and to the smallest difference you intend to claim. If it is not, enable deterministic mode (accepting the throughput cost for evaluation only [S5, S20]) or fix the batch composition. Do not "average over" σ<sub>run</sub> silently.
3. **Report σ<sub>run</sub> separately from the statistical CI** and, if it cannot be driven to zero, combine in quadrature: SE<sub>total</sub><sup>2</sup> = SE<sub>stat</sub><sup>2</sup> + σ<sub>run</sub><sup>2</sup>. The two sources are independent (one is sampling of contexts, the other is numerical), so quadrature is the appropriate combination.
4. **Never compare a KLD number produced under one pipeline with a number produced under another** as though they were commensurable. Bridge them with an unquantized-model run through both.

---

## 4. Numerical error budget: how large is the floor, and why

### 4.1 KL is quadratic in logit error

Let the student's logits be the teacher's plus a perturbation, ℓ̃ = ℓ + δ. Then

```
KL(p || q) = LSE(ℓ + δ) − LSE(ℓ) − Σ_v p(v) δ(v)
```

and a second-order expansion of the log-sum-exp gives

```
KL(p || q) ≈ ½ · Var_{v ~ p}[ δ(v) ]                                   (small-δ regime)
```

i.e. one half of the *p-weighted variance* of the logit perturbation. Three practical consequences:

- **Uniform shifts are free.** A constant added to all logits has zero p-variance and zero KLD. Only *relative* logit error matters.
- **Halving numerical error quarters the KLD floor.** A BF16 path (8-bit mantissa, relative rounding ≈ 2<sup>−8</sup>) will show a floor roughly 4× that of a path with one more bit of accumulated precision, and far above an FP32 path. Order of magnitude: logit noise with p-weighted standard deviation 0.02–0.05 nats implies a floor of 2×10<sup>−4</sup> to 1.3×10<sup>−3</sup> nats. This matches the community observation that BF16-vs-BF16 across implementations, or BF16-vs-FP16 conversions, land in the 10<sup>−4</sup>–10<sup>−3</sup> range [S9].
- **Independent error sources add approximately** in this regime, because variances of independent perturbations add. This is the sole justification for treating "engine floor" and "quantization error" as separable, and it fails once perturbations are large enough to move routing decisions or change the argmax (Section 5.3).

### 4.2 Floors introduced by the measurement tooling itself

- **llama.cpp's base-logit file** stores teacher log-probabilities as scaled 16-bit unsigned integers; the README states the "f16" results should be read as the difference arising only from that downcast [S1]. The visible symptom is that the *minimum* per-token KLD reported in llama.cpp runs is slightly negative (e.g. −7×10<sup>−5</sup> and −3×10<sup>−5</sup> in the cited runs [S15, S16]), which is impossible in exact arithmetic. That magnitude, ~10<sup>−4</sup> nats per token, is the tool's own noise floor.
- **FP16 softmax or FP16 log-probabilities** in the harness introduce errors of the same order. Compute log-softmax in FP32 from logits (FP64 accumulation for the sum is cheap and removes the question), and store teacher log-probs in FP32 if you can afford it, or store BF16/FP16 *logits* and recompute log-softmax in FP32 at analysis time.
- **Vocabulary padding.** Engines commonly pad the LM head to a multiple of 64 or 128. Padded logits must be masked to −∞ (dropped) before softmax in both teacher and student, or the student's distribution has extra mass.

### 4.3 Reference precision

The "teacher" is itself a numerical approximation of the model. For the reference path, run the teacher in FP32 (with TF32 disabled) on at least a subset of the evaluation set, and run it in BF16 on the full set. KL(FP32 ‖ BF16) is the *precision floor* of your reference and the smallest quantization effect you can meaningfully resolve. Every published result should state which reference precision the KLD is against.

---

## 5. Engine choice: reference path vs. production engine, and "correcting" for engine error

### 5.1 The three-way decomposition

Run the following, all on the identical token-ID file and chunking:

| Run | Teacher side | Student side | What it measures |
|---|---|---|---|
| **R0 — canary** | Reference impl., full precision | Same model, same config, same process, repeated | Pipeline reproducibility. Must be exactly 0.0 at every token. Non-zero ⇒ fix before proceeding. |
| **R1 — precision floor** | Reference impl., FP32 | Reference impl., BF16 | Smallest resolvable effect (Section 4.3). |
| **R2 — engine floor** | Reference impl., BF16 | *Unquantized* weights through the production engine (SGLang / ExLlamaV3 / b12x), bs=1 | Everything the engine changes that is not the quantizer: attention kernel numerics, fused-op precision, KV-cache handling, TP/EP reduction order, router implementation differences. |
| **R3 — representation error** | Reference impl., BF16 | Quantized weights *dequantized* to BF16 and run through the reference impl. | Pure information loss of the format/allocator (EXL3 trellis, ShapleyMcg allocation, BMM/NVFP4 codebook). This is the number that ranks *quantization methods*. |
| **R4 — production fidelity** | Reference impl., BF16 | Quantized weights on real kernels in the production engine | What users get. This is the number that ranks *deployable artifacts*. |
| **R5 — batch/config sweep** | — | R4 repeated at bs=1/8/32, shuffled order, cache on/off, TP vs PP | σ<sub>run</sub> and batch-dependence (Section 3.4). |

Consistency check, valid only in the small-error regime of Section 4.1: R4 ≈ R2 + R3. A residual much larger than either term indicates a *kernel-specific* problem (activation quantization, block-scale rounding, FP4 accumulation, a router numerics mismatch) that neither R2 nor R3 sees, and is the most valuable single diagnostic this design produces. For NVFP4 on SM120, where the MMA consumes E2M1 values with block scales and the activation path is also quantized, expect R4 − R3 to be materially non-zero; that gap is the cost of the kernel format and should be published as such.

### 5.2 Why not measure through the serving API

- **Truncated distributions.** OpenAI-style endpoints return top-k logprobs (default k=20 in vLLM, cap set by `--max-logprobs`; `-1` returns all vocab-size logprobs but the documentation warns it may OOM [S7]). SGLang's `top_logprobs_num` is similarly top-k. A KLD computed over a top-k prefix plus an assumed tail is biased, and the bias depends on the entropy of each position.
- **Post-processing.** vLLM's `logprobs_mode` distinguishes raw from processed logprobs/logits; processed values have temperature, top-k/top-p, and penalties applied [S7]. For *prompt* logprobs the two coincide because prompt tokens do not pass through sampling processors [S7], but this must be verified per engine and version rather than assumed.
- **Dynamic batching.** A serving endpoint places your evaluation chunks into batches with other requests, which is precisely mechanism #1 in Section 3.3 [S4].
- **Throughput.** Serializing full-vocabulary logprobs through JSON for 10<sup>5</sup>–10<sup>6</sup> positions is impractical.

Use in-process APIs that hand back the logits tensor: HF transformers (`output.logits`), ExLlamaV3's Python model API (what `model_diff.py` uses [S8]), an SGLang offline `Engine` with return of full logprobs (verify shape equals vocab size), or llama.cpp's `llama-perplexity` binary, whose protocol you then adopt wholesale.

### 5.3 On "correcting" for the engine

Three approaches, in decreasing order of validity:

1. **Isolate and gate (recommended).** Report R2, R3, R4 separately. Set a gate: R2 must be below some fraction (e.g. 10%) of R3 for the engine to be considered transparent for that model; otherwise investigate the engine before publishing R4 numbers. This is how the ExLlamaV3 community discovered a GGUF BF16 floor against HF [S9]: by measuring it, not by subtracting it.
2. **Difference-in-differences.** When ranking two quants A and B *deployed in the same engine*, the engine floor is common to both and cancels to first order in a paired comparison of d<sub>i</sub><sup>A</sup> − d<sub>i</sub><sup>B</sup> (Section 6.4). This is legitimate for ranking, not for reporting absolute fidelity.
3. **Subtraction (not recommended).** "KLD<sub>quant</sub> = R4 − R2" is only meaningful when Section 4.1's quadratic approximation holds, is negative-prone, and hides the kernel-specific residual that you most want to see. Do not publish subtracted numbers.

### 5.4 Practical note on your rig

The BF16 teacher for Qwen3.5-397B-A17B is on the order of 800 GB and does not fit in 4×96 GB. That does not require a proxy teacher. Because the evaluation set is fixed, the exact teacher forward can be computed **layer-streamed**: load layer *l* (attention + all experts) onto GPU, run all N tokens of the evaluation set through it chunk by chunk, write the N×d hidden states to disk, free the layer, load layer *l+1*. Storage per layer boundary is N×d×2 bytes (≈4–8 GB at N=500K); total time is dominated by one pass of the checkpoint over the PCIe bus. This is the same sequential-layer pattern the EXL3 quantizer already uses, so the machinery is at hand. It also removes multi-GPU reductions from the reference path entirely (no TP), which is the single largest step toward bitwise reproducibility. Run it once per (model, evaluation set, precision) and cache the resulting logits or log-probs; treat that cache as an artifact with a hash, the way JohannesGaessler publishes Wikitext logit files for the llama.cpp community [S17].

If a proxy teacher is nonetheless used (e.g. an 8-bit EXL3 or FP8 checkpoint because a true reference is unavailable), state it, and publish the proxy's own KLD against the exact teacher on at least a 50K-token subset so readers can bound the error.

---

## 6. Statistics: estimator, uncertainty, sample size

### 6.1 The shape of the data

Per-token KLD is strictly non-negative and extremely right-skewed. In the cited llama.cpp runs a Llama-3.1-8B-Instruct-class quant showed mean 0.038, median 0.021, 99th percentile 0.33, 99.9th percentile 1.10, maximum 5.70 [S16]; a near-lossless KV-cache experiment on Qwen2.5-7B showed mean 0.0018, median 0.00076, 99.9th percentile 0.087, maximum 0.99 [S15]. In both cases the maximum is two to three orders of magnitude above the mean. Working back from the reported standard errors and the Wikitext-2 token count, the per-token coefficient of variation (CV = σ<sub>d</sub>/D̂) is roughly 2.5 for the 4-bit-class quant and roughly 5 for the near-lossless one: the closer to lossless, the more the mean is a tail phenomenon.

Consequences: (a) the sample mean is the right estimator of expected KLD (unbiased, consistent), but its sampling distribution converges slowly and CIs must respect the skew; (b) llama.cpp's Gaussian assumption [S1] yields a serviceable standard error for the mean at N ≳ 10<sup>5</sup> but is not appropriate for percentiles or for small N; (c) "maximum KLD" is not an estimator of anything stable: it grows with N, and differs between two quants evaluated on different numbers of tokens even when the quants are identical.

### 6.2 Standard error of the mean

**Naive (i.i.d.) standard error.** SE<sub>naive</sub> = s<sub>d</sub>/√N, where s<sub>d</sub> is the sample standard deviation of d<sub>i</sub>. 95% CI: D̂ ± 1.96·SE. This is the "±" printed by llama.cpp.

**Clustered standard error.** Tokens inside one document or evaluation chunk share context and are positively correlated; adjacent positions especially so. Treat each document (or non-overlapping chunk) *c* as a cluster with S<sub>c</sub> = Σ<sub>i∈c</sub>(d<sub>i</sub> − D̂):

```
SE_cluster = (1/N) · sqrt( Σ_c S_c² )       (× C/(C−1) small-sample factor optional)
```

This is the cluster-robust estimator recommended for LLM evals by Miller (Anthropic, 2024), who reports that cluster-adjusted standard errors can be roughly 3× the naive ones on clustered evals [S11]. The **design effect** DEFF = (SE<sub>cluster</sub>/SE<sub>naive</sub>)<sup>2</sup> should be computed from your own data in the pilot and then used for sizing. If evaluation windows overlap (stride < length, as in the ExLlamaV3 default of 2048/512), positions are scored multiple times under different contexts and clusters must be the *documents*, not the windows.

**Block bootstrap (preferred for reporting).** Resample clusters with replacement B = 2,000–10,000 times, recompute the token-weighted mean D* = Σ<sub>c∈sample</sub> Σ<sub>i∈c</sub> d<sub>i</sub> / Σ<sub>c∈sample</sub> n<sub>c</sub>, and take the 2.5th and 97.5th percentiles (or BCa). This respects both the clustering and the skew without a Gaussian assumption, and the same resampling gives CIs for percentiles, top-1 agreement, and ratios between quants at no extra model cost.

### 6.3 Other statistics

- **Top-1 agreement** a = (1/N) Σ 1[argmax p<sub>i</sub> = argmax q<sub>i</sub>]: Wilson interval, or clustered SE √(Σ<sub>c</sub> S<sub>c</sub><sup>2</sup>)/N with S<sub>c</sub> defined on the indicator. At a = 0.95 and N = 200K the naive SE is 5×10<sup>−4</sup>; clustering typically doubles it.
- **Percentiles.** Use the bootstrap. As a stability heuristic, the *q*-th upper percentile needs on the order of ≥100 exceedances to be quoted at all and ≥1,000 to be quoted to two figures: the 99.9th percentile therefore needs N ≥ 10<sup>5</sup> to exist and N ≥ 10<sup>6</sup> to be stable.
- **Per-domain and per-position breakdowns.** Report D̂ separately by domain (Section 7) and by position bucket within the context (e.g. 0–256, 256–1K, 1K–4K, 4K+), each with its own CI. Early positions are high-entropy and inflate KLD; long positions are where KV/attention numerics and error compounding show.

### 6.4 Comparing two quants: use paired differences

To rank allocator A against allocator B (e.g. ShapleyMcg vs. the ExLlamaV3 default at the same routed-expert bpw), do not compare two CIs. Compute the per-token paired difference δ<sub>i</sub> = d<sub>i</sub><sup>A</sup> − d<sub>i</sub><sup>B</sup> on the identical token file and test its mean with the clustered/bootstrapped SE of Section 6.2. Because both quants spike on the same hard tokens, d<sup>A</sup> and d<sup>B</sup> are strongly positively correlated and the variance of the difference is far smaller than the variance of either: Var(δ) = σ<sub>A</sub><sup>2</sup> + σ<sub>B</sub><sup>2</sup> − 2ρσ<sub>A</sub>σ<sub>B</sub>. Miller makes the same recommendation for model-vs-model comparisons and shows that paired analysis can reverse conclusions drawn from unpaired summary statistics [S11]. Report: mean difference, its 95% CI, the correlation ρ, and the ratio D̂<sub>A</sub>/D̂<sub>B</sub> with a bootstrap CI. Also report the paired difference in top-1 agreement (a McNemar-type comparison) since a small mean-KLD advantage can coexist with a worse flip rate.

### 6.5 Sample-size formulas

For a target relative half-width *r* of the 95% CI on mean KLD:

```
N  ≥  ( z_{0.975} · CV / r )²  ·  DEFF          with z_{0.975} = 1.96
```

For detecting a relative difference Δ (as a fraction of D̂) between two quants in a paired design with power 1−β:

```
N  ≥  ( z_{0.975} + z_{1−β} )²  ·  (σ_δ / D̂)²  /  Δ²  ·  DEFF      with z_{0.8} = 0.84
```

Illustrative values using the observed CVs (2.5 for 4-bit-class, 5 for near-lossless), DEFF = 3, and σ<sub>δ</sub>/D̂ ≈ 1.1 (which follows from ρ ≈ 0.9 at CV 2.5; measure your own):

| Goal | CV = 2.5 | CV = 5 |
|---|---|---|
| ±5% relative CI on mean KLD | ≈ 29K tokens | ≈ 115K tokens |
| ±2% | ≈ 180K | ≈ 720K |
| ±1% | ≈ 720K | ≈ 2.9M |
| Detect 5% paired difference, 80% power | ≈ 11K | (depends on ρ; typically < 50K) |
| Detect 2% paired difference | ≈ 71K | — |
| Detect 1% paired difference | ≈ 285K | — |
| Stable 99.9th percentile | ≥ 1M | ≥ 1M |

Reading: a corpus of 250K–500K held-out tokens is the sensible standing evaluation set; it supports ±2–3% CIs on mean KLD for 4-bit-class quants and 1–2% paired discrimination between allocators. The dependence on CV means near-lossless quants (8-bit, FP8-KV experiments) need substantially more tokens to produce a tight *absolute* number, but paired comparison remains cheap. These are planning numbers; the pilot (Section 11, Phase 2) replaces them with measured CV, ρ, and DEFF.

### 6.6 What "reasonable" looks like

There is no universal threshold. Two anchors from the cited runs: mean KLD ≈ 0.002 with ≈98% top-1 agreement is effectively indistinguishable from the original for most users; mean KLD ≈ 0.04 with ≈91% top-1 agreement is the ordinary 4-bit regime for an 8B dense model [S15, S16]. Larger models tolerate a given bpw better; MoE models with many small experts and low routed bpw are the hardest case; agentic and code workloads are more sensitive than prose to the same mean KLD, which is why Unsloth added a trajectory metric on held-out coding/math prompts [S10]. Recommended practice: set acceptance thresholds *relative to the same model's 8-bit quant in the same protocol* (e.g. "mean KLD ≤ 3× the 8-bit value, 99.9th percentile ≤ 2× the 8-bit value, top-1 agreement ≥ 97%"), and publish the 8-bit anchor with every table.

---

## 7. Evaluation-set design

1. **Disjoint from calibration.** No document in the evaluation set may appear in the calibration corpus of *any* quant being compared; dedupe at the document level with a hash and, for safety, an n-gram overlap scan. Both Unsloth [S10] and the EXL3 community [S8] document optimistic numbers when this is violated, and it is the first thing a skeptical reader of an allocator comparison will ask.
2. **Domains.** At least: (a) general prose (a Wikitext-2 test slice is acceptable as a legacy comparator but must not be the whole set), (b) chat/instruction data rendered through the model's chat template, including system prompts and tool-call formatting where the model supports them, (c) code, (d) mathematical/reasoning text, (e) non-English if the model is multilingual. Report per-domain D̂. A quant that is fine on prose and poor on code is a common outcome of calibration-set choice.
3. **Size.** 250K–500K tokens total, with ≥ 50K–100K per domain, in documents long enough to fill the evaluation context.
4. **Context length and stride.** Evaluate at a short context (512–1K) for comparability with llama.cpp-style reporting *and* at a long context (4K–8K, longer if the deployment uses it) for compounding and attention-numerics effects. Non-overlapping windows are cleanest for statistics; if overlapping windows are used, clusters are documents. State the exact policy for which positions are scored (llama.cpp by default scores only the second half of each window [S1]).
5. **Freeze as token IDs.** Tokenize once with the reference tokenizer, store the token-ID file with its hash, and feed IDs (never text) to every engine. Verify each engine's own tokenization of the same text against the stored IDs and report the mismatch rate; GGUF pre-tokenizer discrepancies are a known source of silent divergence.
6. **Version it.** The evaluation set is part of the protocol. Changing it invalidates all prior numbers.

---

## 8. Failure-mode catalog (what can cause it to error)

"Error" here means both *silently wrong numbers* (the dangerous case) and *crashes*.

### 8.1 Silent-wrong

| Failure | Symptom | Detection test |
|---|---|---|
| Tokenizer mismatch between teacher and student pipelines (BOS handling, added tokens, pre-tokenizer regex, chat template) | Large, uniform KLD; low top-1 agreement even for 8-bit | Compare student engine's token IDs for the eval text to the frozen ID file; mismatch rate must be 0 |
| Off-by-one alignment of teacher and student logits | KLD ≈ entropy-level (nats), everywhere | R0 canary; a deliberately shifted self-comparison should be huge, the aligned one exactly 0 |
| Padded vocabulary not masked | Small constant KLD offset; student mass leaks | Check logits.shape[-1] == tokenizer vocab size; assert padded columns are dropped |
| Logits/log-softmax computed in FP16 | Floor ≈ 10<sup>−3</sup>; negative minimum KLD | Recompute a chunk in FP64 and diff |
| FP16 logit overflow (large-logit models, soft-capping) | inf/NaN or clipped distributions | NaN/inf assertion; compare BF16/FP32 |
| KV-cache quantization enabled in the engine by default | R2 unexpectedly high, growing with position | Inspect engine config; run R2 with cache quant off |
| Attention implementation differs (sliding window, RoPE theta/scaling read from GGUF vs config) | R2 grows with position; long-context KLD diverges | Position-bucket plot of R2 |
| Router implementation differs between engines (normalization, bias, group-limited routing, `norm_topk_prob`) | R2 much larger than R1 on MoE only | Dump per-layer top-k expert IDs from reference and engine for the same tokens; agreement must be ≈100% at full precision |
| Calibration/evaluation overlap | Too-good numbers, especially perplexity | Document-hash and n-gram overlap scan |
| Comparing numbers across protocols (different N, context, stride, tool, reference precision) | Contradictory rankings across sources | Refuse; bridge with a common unquantized run |
| Serving-API logprobs (top-k truncated / post-processed) | KLD biased downward at high-entropy positions | Verify returned vector length == vocab; use raw mode [S7] |
| Batch-dependent kernels in the student path | Non-zero σ<sub>run</sub>; max KLD jumps between runs | R5 sweep |
| Hardware instability | Non-zero R0 canary; drift between session start and end | Canary at both ends of every session [S17] |
| Stale cache of teacher logits after a tokenizer/model revision | Everything looks worse at once | Hash the model, tokenizer, and eval file into the cache filename |

### 8.2 Crashes and operational failures

- **OOM storing logits.** N×V×2 bytes: at N = 500K and V ≈ 150K that is ≈150 GB in BF16. Options: store FP32 log-probs only for a 50K subset and BF16 logits for the rest; or compute d<sub>i</sub> online with teacher and student in one process and persist only the per-token scalar record (Section 10.2). Never keep it in RAM.
- **OOM from `max_logprobs=-1`** in serving engines [S7].
- **Length mismatch** after tokenization when a document is shorter than the window; assert per chunk.
- **Async CUDA errors** surfacing late; run with `CUDA_LAUNCH_BLOCKING=1` in the canary phase only.
- **Version drift** across a multi-day sweep; the harness should refuse to run if the environment hash differs from the protocol's.

---

## 9. MoE-specific diagnostics

These are cheap to collect once the harness exists (hook the router outputs) and they turn the routing question from an intuition into a measurement.

1. **Routing agreement per layer.** For each MoE layer and token, compare the teacher's top-k expert set with the student's: exact-set agreement, Jaccard overlap, and top-1 expert agreement. Report per-layer curves. Quantization-aware MoE methods use precisely this "expert-selection consistency" as their objective [S12, S21], so it is also the metric by which a per-expert bit allocation such as ShapleyMcg should expect to be judged.
2. **Router margin distribution.** m<sub>i,l</sub> = s<sub>(k)</sub> − s<sub>(k+1)</sub>, the gap between the k-th and (k+1)-th router scores in the teacher. The fraction of tokens with margin below the observed perturbation scale (estimate that scale from the FP32-vs-BF16 R1 run) is the population of tokens that *will* flip under any perturbation, quantization or numerical. Publishing this fraction per model explains, in advance, why some models are more quantization-sensitive than others.
3. **Flip-conditioned KLD.** Split tokens into "routing agreed" and "routing disagreed" and report D̂ for each. This directly answers whether flips are benign (overlapping experts, your hypothesis) or harmful. Expect a small minority of tokens with disagreement to carry a disproportionate share of the mean.
4. **Router precision.** Keep router weights and router arithmetic in BF16/FP32 in every quant and every engine; verify the engine does. Given the discontinuity, quantizing the router buys almost no memory and can dominate the KLD [S12, S13].
5. **Engine router parity at full precision.** Run R2 with routing dumps: at full precision the engine's expert selections should match the reference at ≈100% of tokens. Anything less is an engine implementation discrepancy (sigmoid vs softmax scoring, bias handling, group-limited routing, top-k normalization, shared-expert scaling), not a quantization effect, and must be resolved before R4 is interpretable.
6. **Run-to-run flip tracking.** In the R5 sweep, count tokens whose *student* routing changes between runs. Non-zero counts locate the non-batch-invariant kernel; they also explain any run-to-run motion in the 99.9th percentile.
7. **Expert-level attribution (optional).** Given per-token routing and per-token d<sub>i</sub>, attribute KLD to experts by the tokens they served in the student. This is a natural companion to per-expert allocation work and highlights experts whose bit budget is misallocated.

---

## 10. Should it be a Python script? Harness architecture

Yes. The measurement is simple; the protocol discipline around it is what makes results defensible. Recommended shape:

### 10.1 Principles

- **One frozen protocol document** (a YAML/JSON file under version control) that names: model hash, tokenizer hash, evaluation token-ID file hash, context length, stride, scored-position policy, reference precision, deterministic flags, environment lockfile hash, and the KLD direction. The harness refuses to run if any hash mismatches, and every output file embeds the protocol hash. This is the single most effective defense against "which run was that?"
- **Separate execution from analysis.** Execution produces per-token records; analysis is pure NumPy over those records and never touches a GPU. Bootstraps, paired tests, per-domain breakdowns, and re-plots then cost seconds, and a change in statistical method never requires re-running a model.
- **Pinned environment** (uv/pip lockfile, CUDA/driver/library versions recorded), deterministic flags on by default in the reference path: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.benchmark=False`, TF32 off, one explicit SDPA backend [S6].
- **Feed token IDs, not text**, to every engine (Section 7.5).
- **Canary first and last** (Section 3.3 #11).

### 10.2 Per-token record (persist this, not the logits)

For each scored position: document id, position-in-window, domain tag, teacher entropy H(p<sub>i</sub>), teacher top-1 id and prob, student top-1 id and prob, realized-token log-prob under both, d<sub>i</sub> in FP64, and for MoE the per-layer top-k expert IDs from both (packed). A few hundred bytes per token; at 500K tokens this is ≈100–200 MB and supports every statistic in Sections 6 and 9 without re-running anything. Store teacher *logits* (BF16) or log-probs (FP32) as a separate cache only where storage permits; the record is the primary artifact.

### 10.3 Execution back-ends

- **Reference path:** HF transformers (or a minimal in-house implementation for models transformers handles poorly) with the layer-streamed forward of Section 5.4 for models that exceed VRAM. Pipeline-parallel across the four GPUs for models that fit; avoid tensor parallel in the reference path.
- **Dequantized-student path (R3):** same code path with weights dequantized to BF16 by the format's reference dequantizer. For EXL3/TR3 that is the trellis decoder you already have; for BMM/NVFP4 it is E2M1 × block-scale reconstruction in FP32 then cast.
- **Production path (R4):** ExLlamaV3's model API, b12x's fused W4A16 path, and SGLang's offline engine (with `--enable-deterministic-inference` where the backend supports it on SM120, FlashInfer or Triton; verify with `sglang.test.test_deterministic` first [S5]). llama.cpp's `llama-perplexity` for GGUF comparisons, adopting its protocol and noting its 16-bit base-file floor [S1].
- **Interfaces:** a single `score(token_ids) → logits` adapter per back-end so the harness is engine-agnostic and any new kernel (a new BMM variant, a new b12x path) is a new adapter, not a new script.

### 10.4 Analysis module

`mean_kld_ci` (naive, clustered, block bootstrap), `paired_compare(A, B)` (mean difference, ratio, ρ, McNemar on top-1), `percentiles_ci`, `by_domain`, `by_position`, `routing_report`, `sample_size(cv, deff, r)`, and a `kld_card` renderer (Section 12). Unit-test the KLD kernel against a FP64 reference on synthetic distributions, including the KL(p‖p)=0 identity and the Section 4.1 quadratic approximation for small δ.

---

## 11. Plan of action

Phases are gated; do not advance on a failed gate.

**Phase 0 — Protocol freeze (1–2 days).** Write the protocol document (Section 10.1). Choose models: one small MoE for fast iteration (e.g. a 30B-A3B-class model that fits comfortably), plus the production targets. Decide reference precision (FP32 subset + BF16 full). Decide contexts (512/1K and 4K/8K). Gate: protocol hash committed.

**Phase 1 — Evaluation set (1–2 days).** Assemble ≈500K tokens across ≥4 domains (Section 7), chat-formatted where applicable, deduplicated against every calibration corpus in play (the ShapleyMcg calibration corpus, EXL3's built-in mix, any imatrix data). Tokenize once; freeze IDs with hash. Gate: overlap scan clean; per-domain token counts ≥ 50K.

**Phase 2 — Reference logits and pilot statistics (2–3 days).** Layer-streamed teacher forward for the production models; direct forward for the small model. Produce the FP32 subset and BF16 full-set caches. Run R0 (canary must be exactly 0) and R1 (precision floor). Compute per-token CV, cluster ICC/DEFF, and the router-margin distribution. Recompute the sample-size table with measured values; extend the evaluation set if the 4K-context, per-domain CIs are wider than ±5%. Gate: R0 = 0; R1 reported; CV/DEFF measured.

**Phase 3 — Engine floors (2–3 days).** R2 for each production back-end (ExLlamaV3, b12x, SGLang, llama.cpp if used) with unquantized weights, bs=1, plus routing dumps for MoE parity (Section 9.5). R5-style sweep of batch size and cache settings to obtain σ<sub>run</sub> for each engine; enable deterministic mode where available and re-measure. Gate: engine routing parity ≈100% at full precision; σ<sub>run</sub> characterized; any R2 above ~10% of the expected 4-bit R3 is investigated (KV cache quant, attention backend, RoPE/config mismatch) before proceeding.

**Phase 4 — Representation error, R3 (ongoing).** Dequantized-weights evaluation for every quant/allocation under study. This is the apples-to-apples plane for the ShapleyMcg-vs-default comparison at fixed bpw: same token file, paired per-token differences, clustered bootstrap CI, ratio with CI, top-1 paired comparison, per-domain and per-position breakdowns, routing agreement per layer. Gate: the claimed improvement's 95% CI excludes zero on the paired difference, and the direction holds in every domain (or the exceptions are stated).

**Phase 5 — Production fidelity, R4 (ongoing).** Real kernels. Publish R2, R3, R4 side by side with the R4 − R3 residual per kernel path; for NVFP4/BMM this residual is the honest cost of the SM120 block-scaled path and is itself a result worth reporting. Repeat R5 on the final artifacts. Gate: σ<sub>run</sub> ≤ 20% of SE<sub>stat</sub> (or deterministic mode on); otherwise report SE<sub>total</sub> per Section 3.4.

**Phase 6 — Reporting and regression (ongoing).** Every published quant ships a KLD card (Section 12). Add a CI job that runs R0/R2 on the small model whenever a kernel, engine version, or adapter changes, so engine-floor regressions are caught before they contaminate quant comparisons. Publish the teacher logit caches (or their hashes and generation recipe) so third parties can reproduce.

Approximate cost on the 4× RTX PRO 6000 rig: the layer-streamed 397B teacher pass is I/O-bound (one read of the checkpoint per precision); R3/R4 runs of a 500K-token set take minutes to tens of minutes per quant on the production engines; the bootstraps are seconds. The dominant cost is engineering time in Phases 0–3, which is a one-time investment that every later comparison reuses.

---

## 12. Reporting template (the "KLD card")

Each result should carry, at minimum:

```
Model / revision hash; tokenizer hash
Quant: format, bpw (routed / shared / attention / head), allocator, calibration corpus hash
Reference: implementation, precision (FP32 subset n=…, BF16 full), layer-streamed: yes/no
Engine (R4): name, version, backend, deterministic mode: on/off, bs, TP/EP/PP layout, KV-cache dtype
Evaluation set: hash, N tokens, domains and per-domain N, context, stride, scored-position policy
Direction: KL(teacher || student), nats

R0 canary: 0.0 (exact)          R1 precision floor: …          R2 engine floor: …
R3 representation: mean … [95% CI … , block bootstrap, clusters = documents]
R4 production:     mean … [95% CI …]      R4 − R3 residual: …
σ_run over R5 sweep: …  (n runs = …)
Median / 90 / 95 / 99 / 99.9 / max: …   (max not comparable across N)
Top-1 agreement: … [95% CI …]     Mean Δp / RMS Δp: …     ln PPL ratio: …
Per-domain means with CIs; per-position-bucket means with CIs
MoE: routing agreement per layer (mean, min), fraction of low-margin tokens, flip-conditioned KLD
Paired comparison vs. baseline quant (if any): mean diff [CI], ratio [CI], ρ, Δ top-1 [CI]
Environment lockfile hash; date; GPU SKU/driver/CUDA
```

---

## Appendix A — Formula summary

```
Per-token:        d_i = Σ_v p_i(v) [ln p_i(v) − ln q_i(v)]            (exact; compute log-softmax in FP32, sum in FP64)
Mean:             D̂ = (1/N) Σ d_i
Naive SE:         SE = s_d / √N ;  95% CI = D̂ ± 1.96·SE
Clustered SE:     SE_cl = (1/N)·sqrt( Σ_c S_c² ),  S_c = Σ_{i∈c}(d_i − D̂)
Design effect:    DEFF = (SE_cl / SE)²
Block bootstrap:  resample clusters with replacement, D* = Σ_i d_i / Σ_c n_c over the sample; percentile or BCa CI
Paired diff:      δ_i = d_i^A − d_i^B ;  test mean δ with clustered/bootstrap SE ;  Var(δ) = σ_A² + σ_B² − 2ρσ_Aσ_B
Top-1 agreement:  a = (1/N) Σ 1[argmax p_i = argmax q_i] ;  Wilson or clustered SE
Sample size (CI): N ≥ (1.96·CV / r)² · DEFF
Sample size (paired detection): N ≥ (1.96 + z_{1−β})² · (σ_δ/D̂)² / Δ² · DEFF
Small-error KL:   KL(p||q) ≈ ½ · Var_{v~p}[δ(v)]  for q = softmax(ℓ + δ)
Total SE (non-reproducible pipeline): SE_total² = SE_stat² + σ_run²
```

## Appendix B — Sources

- [S1] llama.cpp, `tools/perplexity/README.md` (KLD workflow, 16-bit base file, Gaussian uncertainty, non-comparability across projects). https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
- [S2] ikawrakow, "KL-divergence", llama.cpp PR #5076, merged 22 Jan 2024 (references Ttl's Python implementation in PR #4739). https://github.com/ggerganov/llama.cpp/pull/5076
- [S3] Dutta, Dhar, Sharma et al., "Accuracy is Not All You Need", NeurIPS 2024. https://arxiv.org/abs/2407.09141
- [S4] He et al. (Thinking Machines Lab), "Defeating Nondeterminism in LLM Inference", Sept 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- [S5] LMSYS, "Towards Deterministic Inference in SGLang and Reproducible RL Training", Sept 2025, and SGLang docs "Deterministic Inference". https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/ ; https://docs.sglang.io/advanced_features/deterministic_inference.html
- [S6] PyTorch, "Reproducibility" and `torch.use_deterministic_algorithms` documentation. https://docs.pytorch.org/docs/stable/notes/randomness.html
- [S7] vLLM, Engine Arguments (`max_logprobs`, `logprobs_mode`). https://docs.vllm.ai/en/stable/configuration/engine_args/
- [S8] turboderp-org/exllamav3 (`eval/model_diff.py`, `eval/compare_q.py`); mratsim GLM-4.7-EXL3 model card on KLD methodology and calibration/eval overlap. https://github.com/turboderp-org/exllamav3 ; https://huggingface.co/mratsim/GLM-4.7-EXL3
- [S9] exllamav3 PR #26 discussion (GGUF BF16 vs HF transformers baseline KLD). https://github.com/turboderp-org/exllamav3/pull/26
- [S10] Unsloth, "Dynamic 2.0 GGUFs" and "Dynamic 3.0 GGUFs" documentation (KLD rationale, calibration overfitting, Divergence-300 @32). https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs ; https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
- [S11] Miller, "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations", Anthropic, Nov 2024. https://arxiv.org/abs/2411.00640
- [S12] Park et al., "Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models" (VSRAQ), June 2026. https://arxiv.org/abs/2606.05688
- [S13] "EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization", 2025. https://arxiv.org/abs/2506.13329
- [S14] "QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts", 2024. https://arxiv.org/abs/2406.08155
- [S15] llama.cpp Discussion #23470 (asymmetric KV cache; full KLD statistics block). https://github.com/ggml-org/llama.cpp/discussions/23470
- [S16] llama.cpp PR #12557 (compilade; KLD statistics for Llama-3.1-8B-Instruct quants). https://github.com/ggml-org/llama.cpp/pull/12557
- [S17] JohannesGaessler, llama.cpp Wikitext logits repository (note on withdrawn file due to hardware instability). https://huggingface.co/JohannesGaessler/llama.cpp_wikitext_logits
- [S18] "Kernel Contracts…" (2026), summarizing Shanmugavelu et al. on index_add / scatter_reduce nondeterminism on H100. https://arxiv.org/pdf/2604.22032
- [S19] "How Much Parallelism Is 'Free'?…" (2026), Appendix E on expert-token padding and kernel-config selection in vLLM/SGLang fused MoE. https://arxiv.org/pdf/2605.30851
- [S20] "LLM-42: Enabling Determinism in LLM Inference with Verified Speculation" (2026) on the performance cost of batch-invariant kernels. https://arxiv.org/pdf/2601.17768
- [S21] "Security-aware post-training quantization for Mixture-of-Experts large language models" (SAMEQuant), 2026. https://www.sciencedirect.com/science/article/abs/pii/S0031320326012513
- [S22] thinking-machines-lab/batch_invariant_ops. https://github.com/thinking-machines-lab/batch_invariant_ops
- [S23] ubergarm, notes on `eval/compare_q.py` dataspec (eval_len 2048, eval_stride 512). https://gist.github.com/ubergarm/9d560bab80241b90dac802e91b656743
