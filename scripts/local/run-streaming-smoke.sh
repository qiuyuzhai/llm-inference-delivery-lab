#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aaron/Desktop/llm-inference-delivery-lab}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
BASE_URL="${BENCHMARK_BASE_URL:-http://127.0.0.1:8080}"
API_KEY="${GATEWAY_API_KEY:-change-me}"
WORKLOAD="${WORKLOAD:-benchmark/workloads/knowledge_qa.jsonl}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS="${REQUESTS:-3}"
OUTPUT="${OUTPUT:-benchmark/results/smoke.json}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

curl -fsS "${BASE_URL}/health" >/dev/null

"${PYTHON_BIN}" -m benchmark.harness.cli \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --workload "${WORKLOAD}" \
  --concurrency "${CONCURRENCY}" \
  --requests "${REQUESTS}" \
  --output "${OUTPUT}"
