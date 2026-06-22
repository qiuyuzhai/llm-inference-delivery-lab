#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
# Quantization method. Leave empty for BF16/FP16 (no quantization).
# Valid values: awq_marlin (preferred, faster marlin kernel), awq,
#               gptq_marlin (preferred), gptq.
# AWQ weights -> use awq_marlin (or awq for older vLLM).
# GPTQ-Int4 weights -> use gptq_marlin (or gptq for older vLLM).
QUANTIZATION="${QUANTIZATION:-}"

# Build quantization flag only when QUANTIZATION is non-empty to avoid
# passing an empty string which causes vLLM to error out.
QUANT_ARGS=()
if [ -n "${QUANTIZATION}" ]; then
  QUANT_ARGS=(--quantization "${QUANTIZATION}")
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  "${QUANT_ARGS[@]}"
