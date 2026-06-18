#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-llm-vllm-streaming-bench}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"

if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker stop "${CONTAINER_NAME}" >/dev/null
  echo "stopped vLLM container: ${CONTAINER_NAME}"
else
  echo "vLLM container not running: ${CONTAINER_NAME}"
fi

if command -v ss >/dev/null 2>&1; then
  gateway_pids="$(ss -ltnp "sport = :${GATEWAY_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
  if [ -n "${gateway_pids}" ]; then
    for pid in ${gateway_pids}; do
      kill "${pid}"
      echo "stopped Gateway process: ${pid}"
    done
  else
    echo "Gateway not listening on port: ${GATEWAY_PORT}"
  fi
else
  echo "ss command not found; skip Gateway process cleanup" >&2
fi
