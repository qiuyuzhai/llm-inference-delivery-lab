#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aaron/Desktop/llm-inference-delivery-lab}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
DEFAULT_MODEL="${DEFAULT_MODEL:-/models/model}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-change-me}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

VLLM_BASE_URL="${VLLM_BASE_URL}" \
DEFAULT_MODEL="${DEFAULT_MODEL}" \
GATEWAY_API_KEY="${GATEWAY_API_KEY}" \
"${PYTHON_BIN}" -m uvicorn gateway.app.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}"
