#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-llm-vllm-streaming-bench}"
IMAGE="${VLLM_IMAGE:-vllm-m27:gb10}"
MODEL_PATH="${MODEL_PATH:-/home/aaron/Desktop/minimax-m2.7-local/models/Qwen2.5-1.5B-Instruct}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/aaron/Desktop/llm-inference-delivery-lab}"
MODEL_NAME="${MODEL_NAME:-/models/model}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
HOST_PORT="${VLLM_HOST_PORT:-8000}"
# Quantization method passed through to start-vllm.sh.
# Valid values: awq_marlin, awq, gptq_marlin, gptq; empty = BF16 (no quantization).
QUANTIZATION="${QUANTIZATION:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "vLLM image not found: ${IMAGE}" >&2
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [ ! -f "${PROJECT_ROOT}/serving/vllm/start-vllm.sh" ]; then
  echo "start-vllm.sh not found under project root: ${PROJECT_ROOT}" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "vLLM container already running: ${CONTAINER_NAME}"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

docker run --rm --gpus all \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:8000" \
  -v "${MODEL_PATH}:/models/model:ro" \
  -v "${PROJECT_ROOT}/serving/vllm/start-vllm.sh:/app/start-vllm.sh:ro" \
  -e "MODEL_NAME=${MODEL_NAME}" \
  -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  -e "QUANTIZATION=${QUANTIZATION}" \
  "${IMAGE}" "/app/start-vllm.sh"
