#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v75@sha256:4605c420cc589be9fd15fc759c7f7c2a6035dab48f885c9466eb2233527bca64}"
MODEL="${MODEL:-/home/brandonmusic/models/GLM-5.3-Flash-tr3-4bpw}"
CACHE="${CACHE:-nvfp4_ds_mla}"
DCP="${DCP:-2}"
PORT="${PORT:-8012}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
NAME="${NAME:-glm53-flash-exl3-k4-${CACHE}-dcp${DCP}}"
PROFILE="${PROFILE:-daily}"
case "${PROFILE}" in
  daily)
    PROFILE_MAX_MODEL_LEN=499968
    PROFILE_MAX_NUM_BATCHED_TOKENS=2048
    PROFILE_MAX_NUM_SEQS=1
    PROFILE_GPU_MEMORY_UTILIZATION=0.986
    ;;
  long500k)
    # Qualified single-request 500K profile. The smaller prefill chunk halves
    # KPool's transient logits matrix at extreme context length.
    PROFILE_MAX_MODEL_LEN=499968
    PROFILE_MAX_NUM_BATCHED_TOKENS=1024
    PROFILE_MAX_NUM_SEQS=1
    PROFILE_GPU_MEMORY_UTILIZATION=0.985
    ;;
  *)
    echo "PROFILE must be daily or long500k" >&2
    exit 2
    ;;
esac

# FP8 MLA stores a wider physical cache row than NVFP4 MLA.  Keep its default
# below the measured per-GPU block budget; users can still set MAX_MODEL_LEN
# explicitly for a separately measured layout.  The 500K profile is an NVFP4
# qualification profile by construction.
if [[ "${CACHE}" == "fp8_ds_mla" ]]; then
  if [[ "${PROFILE}" == "long500k" ]]; then
    echo "PROFILE=long500k requires CACHE=nvfp4_ds_mla" >&2
    exit 2
  fi
  PROFILE_MAX_MODEL_LEN=262144
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${PROFILE_MAX_MODEL_LEN}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${PROFILE_MAX_NUM_BATCHED_TOKENS}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${PROFILE_MAX_NUM_SEQS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${PROFILE_GPU_MEMORY_UTILIZATION}}"
MTP_TOKENS="${MTP_TOKENS:-3}"
PREFIX_CACHING="${PREFIX_CACHING:-0}"
B12X_DCP_A2A="${B12X_DCP_A2A:-1}"
DIRECT_DCP_A2A="${DIRECT_DCP_A2A:-}"
B12X_MLA_CKV_GATHER="${B12X_MLA_CKV_GATHER:-0}"
EXL3_PREFILL_BLOCK_M="${VLLM_EXL3_PREFILL_BLOCK_M:-}"

if [[ "${DCP}" != "1" && "${DCP}" != "2" ]]; then
  echo "DCP must be 1 or 2" >&2
  exit 2
fi
if [[ "${PREFIX_CACHING}" != "0" && "${PREFIX_CACHING}" != "1" ]]; then
  echo "PREFIX_CACHING must be 0 or 1" >&2
  exit 2
fi
case "${CACHE}" in
  nvfp4_ds_mla) ATTENTION_BACKEND=B12X_MLA_SPARSE ;;
  fp8_ds_mla) ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_SM120 ;;
  *)
    echo "CACHE must be nvfp4_ds_mla or fp8_ds_mla" >&2
    exit 2
    ;;
esac

EXTRA_ARGS=()
if [[ "${MTP_TOKENS}" != "0" ]]; then
  EXTRA_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\"}")
fi
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${PREFIX_CACHING}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
else
  EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

DIRECT_DCP_ARGS=()
if [[ -n "${DIRECT_DCP_A2A}" ]]; then
  DIRECT_DCP_ARGS+=(-e "VLLM_USE_DIRECT_DCP_A2A=${DIRECT_DCP_A2A}")
fi

EXL3_PREFILL_BLOCK_ARGS=()
if [[ -n "${EXL3_PREFILL_BLOCK_M}" ]]; then
  EXL3_PREFILL_BLOCK_ARGS+=(-e "VLLM_EXL3_PREFILL_BLOCK_M=${EXL3_PREFILL_BLOCK_M}")
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true
exec docker run --name "${NAME}" \
  --init --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g \
  -p "${PORT}:${PORT}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_B12X_GLM_NOPE_NVFP4=1 \
  -e "VLLM_USE_B12X_DCP_A2A=${B12X_DCP_A2A}" \
  -e "VLLM_B12X_MLA_CKV_GATHER=${B12X_MLA_CKV_GATHER}" \
  "${DIRECT_DCP_ARGS[@]}" \
  "${EXL3_PREFILL_BLOCK_ARGS[@]}" \
  -e OMP_NUM_THREADS=2 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -v "${MODEL}:/model:ro" \
  -v "${GLM53_CACHE_PATH:-/home/brandonmusic/.cache/glm53-exl3-k4}:/root/.cache" \
  --entrypoint vllm \
  "${IMAGE}" serve /model \
  --served-model-name GLM-5.3-Flash-EXL3-4bpw \
  --host 0.0.0.0 --port "${PORT}" \
  --language-model-only \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --decode-context-parallel-size "${DCP}" \
  --dcp-comm-backend a2a \
  --dtype bfloat16 \
  --load-format safetensors \
  --moe-backend b12x \
  --attention-backend "${ATTENTION_BACKEND}" \
  --kv-cache-dtype "${CACHE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-chunked-prefill \
  --generation-config /model \
  --reasoning-parser glm45 \
  --disable-custom-all-reduce \
  "${EXTRA_ARGS[@]}" \
  "$@"
