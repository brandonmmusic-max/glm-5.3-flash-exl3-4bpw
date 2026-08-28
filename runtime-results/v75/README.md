# v75 release evidence

This directory contains the compact, publishable receipts for the immutable
`r19-sm120-tp2-ep2-v75` image. Large raw logits and temporary captures are not
included.

- `validation/docker-release.json`: registry digests, runtime contract, and evidence hashes
- `validation/coherence-smoke.json`: NVFP4 coherent-generation smoke test
- `validation/coherence-smoke-fp8.json`: digest-specific FP8 coherent-generation smoke test
- `validation/route128-vs-generic.json`: route-128 numerical comparison and kernel timings
- `benchmarks/`: standalone cold-prefill and sustained C1 decode JSON plus logs
- `kld/`: final five-run FP8 and NVFP4 MLA-cache KLD receipts
- `quality/`: v75 Estonia, LAVD-low, and needle-through-500K receipts and logs

The KLD runs use the full 2,048-token qualification window (2,047 causal
positions per run), eager execution, and no MTP. The speed and quality runs use
the production CUDA-graph/MTP3 configuration unless their receipt says
otherwise.
