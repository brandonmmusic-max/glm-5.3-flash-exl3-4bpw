#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-language-only@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692}"
MODEL="${MODEL:?set MODEL to the local EXL3 checkpoint directory}"
DFLASH_MODEL="${DFLASH_MODEL:?set DFLASH_MODEL to the local incoai/GLM-5.3-Flash-DFlash2 directory}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
PORT="${PORT:-8012}"
NAME="${NAME:-glm53-flash-exl3-k4-language-only-dflash2}"
CACHE_PATH="${GLM53_CACHE_PATH:-${PWD}/glm53-vllm-cache}"

mkdir -p "${CACHE_PATH}"

exec docker run --rm --name "${NAME}" \
  --init --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g \
  -p "${PORT}:${PORT}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_B12X_GLM_NOPE_NVFP4=1 \
  -e VLLM_NVFP4_MLA_DYNAMIC_SCALE=0 \
  -e VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm53/calibration/glm53_nvfp4_mla_outer_scales_mtp_power2_v2.json \
  -e VLLM_EXL3_PREFILL_BLOCK_M=128 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 \
  -e B12X_GL53_ROUTE128_WIDE=1 \
  -e B12X_GL53_ROUTE128_HYBRID_TAIL=1 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e KV_FP8_ROPE=0 \
  -e OMP_NUM_THREADS=2 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=4 \
  -v "${MODEL}:/model:ro" \
  -v "${DFLASH_MODEL}:/draft:ro" \
  -v "${CACHE_PATH}:/cache" \
  "${IMAGE}" serve /model \
  --served-model-name GLM-5.3-Flash-EXL3-4bpw \
  --host 0.0.0.0 --port "${PORT}" \
  --language-model-only \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --decode-context-parallel-size 2 \
  --dcp-comm-backend a2a \
  --dtype bfloat16 \
  --load-format safetensors \
  --moe-backend b12x \
  --attention-backend B12X_MLA_SPARSE \
  --kv-cache-dtype nvfp4_ds_mla \
  --max-model-len 98304 \
  --max-num-batched-tokens 2072 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.986 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --generation-config /model \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}' \
  "$@"
