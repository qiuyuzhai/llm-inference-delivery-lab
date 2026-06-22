#!/usr/bin/env bash
# sweep-params.sh — parameter sweep for max_model_len × gpu_memory_utilization × max_num_seqs
#
# Usage:
#   bash scripts/local/sweep-params.sh [--dry-run]
#
# Environment overrides (all optional):
#   MODEL_PATH              host path to model weights directory
#   MODEL_NAME              model name passed to vLLM (default /models/model)
#   VLLM_IMAGE              docker image tag (default vllm-m27:gb10)
#   PROJECT_ROOT            repo root (default: directory of this script ../../)
#   PYTHON_BIN              python binary (default: PROJECT_ROOT/.venv/bin/python)
#   GATEWAY_API_KEY         API key for gateway (default: change-me)
#   SWEEP_CONCURRENCY       benchmark concurrency per run (default: 4)
#   SWEEP_REQUESTS          benchmark requests per workload per run (default: 20)
#   SWEEP_HEALTH_TIMEOUT    seconds to wait for vLLM /health (default: 180)
#   SWEEP_HEALTH_INTERVAL   polling interval in seconds (default: 5)
#
# Ports:
#   vLLM  → host 8001  (avoids 8000 used by engineering API)
#   Gateway → 8081    (avoids 8080 used by gunicorn)
#
# Output: benchmark/results/sweep/<label>/  (two JSON files per config)
#
# Dry-run (bash -n or --dry-run flag): prints grid without running anything.
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

# ---------------------------------------------------------------------------
# Model configuration (baseline 7B)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/aaron/Desktop/minimax-m2.7-local/models/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-/models/model}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm-m27:gb10}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-change-me}"

# ---------------------------------------------------------------------------
# Ports (fixed; do not conflict with engineering API / gunicorn)
# ---------------------------------------------------------------------------
VLLM_HOST_PORT=8001
GATEWAY_PORT=8081
VLLM_BASE_URL="http://127.0.0.1:${VLLM_HOST_PORT}"
GATEWAY_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}"
CONTAINER_NAME="llm-vllm-sweep"

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
SWEEP_CONCURRENCY="${SWEEP_CONCURRENCY:-4}"
SWEEP_REQUESTS="${SWEEP_REQUESTS:-20}"
SWEEP_HEALTH_TIMEOUT="${SWEEP_HEALTH_TIMEOUT:-180}"
SWEEP_HEALTH_INTERVAL="${SWEEP_HEALTH_INTERVAL:-5}"

WORKLOAD_THROUGHPUT="benchmark/workloads/knowledge_qa.jsonl"
WORKLOAD_LONGCTX="benchmark/workloads/long_context_qa.jsonl"

RESULTS_DIR="${PROJECT_ROOT}/benchmark/results/sweep"

# ---------------------------------------------------------------------------
# Sweep grid — edit here to extend dimensions
# ---------------------------------------------------------------------------
MAX_MODEL_LENS=(4096 8192 16384 32768)
GPU_MEMORY_UTILS=(0.6 0.75 0.9)
MAX_NUM_SEQS_LIST=(8 16 32)

# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------
DRY_RUN=false
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[sweep] $(date '+%H:%M:%S') $*"; }
err() { echo "[sweep][ERROR] $*" >&2; }

# Stop any running sweep container and gateway on our ports.
teardown() {
  log "teardown: stopping container ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm  "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  # Kill gateway process on GATEWAY_PORT if running
  if command -v ss >/dev/null 2>&1; then
    local gw_pids
    gw_pids="$(ss -ltnp "sport = :${GATEWAY_PORT}" 2>/dev/null \
      | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
    for pid in ${gw_pids}; do
      kill "${pid}" 2>/dev/null || true
      log "teardown: killed gateway pid ${pid}"
    done
  fi
}

# Poll vLLM /health until ready or timeout.
# Returns 0 on success, 1 on timeout or OOM detected.
wait_for_vllm() {
  local label="$1"
  local log_file="$2"
  local elapsed=0

  log "waiting for vLLM health (timeout=${SWEEP_HEALTH_TIMEOUT}s) ..."
  while (( elapsed < SWEEP_HEALTH_TIMEOUT )); do
    if curl -fsS "${VLLM_BASE_URL}/health" >/dev/null 2>&1; then
      log "vLLM healthy after ${elapsed}s"
      return 0
    fi

    # Check OOM / CUDA out-of-memory in container logs
    if docker logs "${CONTAINER_NAME}" 2>&1 \
        | grep -qiE 'out of memory|CUDA error|OOM|torch.cuda.OutOfMemoryError'; then
      log "OOM detected in vLLM logs for ${label}"
      # Capture tail of log as case-study evidence
      docker logs "${CONTAINER_NAME}" 2>&1 | tail -50 >> "${log_file}" || true
      return 1
    fi

    # Container exited unexpectedly
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}" 2>/dev/null; then
      log "vLLM container exited unexpectedly for ${label}"
      docker logs "${CONTAINER_NAME}" 2>&1 | tail -50 >> "${log_file}" 2>/dev/null || true
      return 1
    fi

    sleep "${SWEEP_HEALTH_INTERVAL}"
    (( elapsed += SWEEP_HEALTH_INTERVAL )) || true
  done

  log "vLLM health timeout (${SWEEP_HEALTH_TIMEOUT}s) for ${label}"
  docker logs "${CONTAINER_NAME}" 2>&1 | tail -50 >> "${log_file}" || true
  return 1
}

# Write a FAILED sentinel JSON so compare_runs.py can report it.
write_failed_sentinel() {
  local out_dir="$1"
  local label="$2"
  local model_name="$3"
  local max_model_len="$4"
  local gpu_mem_util="$5"
  local max_num_seqs="$6"
  local reason="$7"

  mkdir -p "${out_dir}"
  cat > "${out_dir}/FAILED.json" <<EOF
{
  "config": {
    "label": "${label}",
    "model_name": "${model_name}",
    "quantization": null,
    "dtype": "auto",
    "max_model_len": ${max_model_len},
    "concurrency": null,
    "requests": null,
    "base_url": "${GATEWAY_BASE_URL}"
  },
  "failed": true,
  "reason": "${reason}",
  "gpu_memory_utilization": ${gpu_mem_util},
  "max_num_seqs": ${max_num_seqs}
}
EOF
  log "wrote FAILED sentinel: ${out_dir}/FAILED.json"
}

# Run a single benchmark workload; on failure write FAILED sentinel.
run_bench() {
  local workload="$1"
  local out_file="$2"
  local label="$3"
  local model_name_cli="$4"
  local max_model_len="$5"

  log "benchmark: workload=${workload} output=${out_file}"
  if ! "${PYTHON_BIN}" -m benchmark.harness.cli \
      --base-url "${GATEWAY_BASE_URL}" \
      --api-key  "${GATEWAY_API_KEY}" \
      --workload "${workload}" \
      --concurrency "${SWEEP_CONCURRENCY}" \
      --requests "${SWEEP_REQUESTS}" \
      --output   "${out_file}" \
      --label    "${label}" \
      --model-name "${model_name_cli}" \
      --max-model-len "${max_model_len}"; then
    err "benchmark failed: ${out_file}"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks (skipped in dry-run)
# ---------------------------------------------------------------------------
if ! "${DRY_RUN}"; then
  if ! command -v docker >/dev/null 2>&1; then
    err "docker not found"
    exit 1
  fi
  if ! docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    err "vLLM image not found: ${VLLM_IMAGE}"
    exit 1
  fi
  if [ ! -d "${MODEL_PATH}" ]; then
    err "model path not found: ${MODEL_PATH}"
    exit 1
  fi
  if [ ! -x "${PYTHON_BIN}" ]; then
    err "python binary not found: ${PYTHON_BIN}"
    exit 1
  fi
  if [ ! -f "${PROJECT_ROOT}/serving/vllm/start-vllm.sh" ]; then
    err "start-vllm.sh not found under: ${PROJECT_ROOT}"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Print sweep grid
# ---------------------------------------------------------------------------
total=$(( ${#MAX_MODEL_LENS[@]} * ${#GPU_MEMORY_UTILS[@]} * ${#MAX_NUM_SEQS_LIST[@]} ))
log "sweep grid: ${total} configurations"
log "  max_model_len:          ${MAX_MODEL_LENS[*]}"
log "  gpu_memory_utilization: ${GPU_MEMORY_UTILS[*]}"
log "  max_num_seqs:           ${MAX_NUM_SEQS_LIST[*]}"
log "  workloads:              ${WORKLOAD_THROUGHPUT}  ${WORKLOAD_LONGCTX}"
log "  results_dir:            ${RESULTS_DIR}"

if "${DRY_RUN}"; then
  log "dry-run mode: printing grid only, no execution"
  idx=0
  for mml in "${MAX_MODEL_LENS[@]}"; do
    for gmu in "${GPU_MEMORY_UTILS[@]}"; do
      for mns in "${MAX_NUM_SEQS_LIST[@]}"; do
        (( idx++ )) || true
        label="mml${mml}-gmu${gmu}-mns${mns}"
        log "  [${idx}/${total}] ${label}"
      done
    done
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"

run_idx=0
passed=0
failed=0

for mml in "${MAX_MODEL_LENS[@]}"; do
  for gmu in "${GPU_MEMORY_UTILS[@]}"; do
    for mns in "${MAX_NUM_SEQS_LIST[@]}"; do
      (( run_idx++ )) || true
      label="mml${mml}-gmu${gmu}-mns${mns}"
      out_dir="${RESULTS_DIR}/${label}"
      log_file="${out_dir}/vllm-startup.log"
      mkdir -p "${out_dir}"

      log "=== run [${run_idx}/${total}] ${label} ==="

      # Ensure clean slate before each run
      teardown

      # --- Start vLLM (detached) ---
      log "starting vLLM: max_model_len=${mml} gpu_mem_util=${gmu} max_num_seqs=${mns}"
      docker run --rm -d --gpus all \
        --name "${CONTAINER_NAME}" \
        -p "${VLLM_HOST_PORT}:8000" \
        -v "${MODEL_PATH}:/models/model:ro" \
        -v "${PROJECT_ROOT}/serving/vllm/start-vllm.sh:/app/start-vllm.sh:ro" \
        -e "MODEL_NAME=${MODEL_NAME}" \
        -e "MAX_MODEL_LEN=${mml}" \
        -e "GPU_MEMORY_UTILIZATION=${gmu}" \
        -e "MAX_NUM_SEQS=${mns}" \
        -e "QUANTIZATION=" \
        "${VLLM_IMAGE}" "/app/start-vllm.sh" \
        > "${log_file}" 2>&1

      # --- Poll /health ---
      if ! wait_for_vllm "${label}" "${log_file}"; then
        err "${label}: vLLM failed to start (OOM or timeout); recording FAILED"
        write_failed_sentinel "${out_dir}" "${label}" "${MODEL_NAME}" \
          "${mml}" "${gmu}" "${mns}" "vLLM startup failed (OOM or timeout)"
        teardown
        (( failed++ )) || true
        continue
      fi

      # --- Start gateway ---
      log "starting gateway on port ${GATEWAY_PORT}"
      VLLM_BASE_URL="${VLLM_BASE_URL}" \
      DEFAULT_MODEL="${MODEL_NAME}" \
      GATEWAY_API_KEY="${GATEWAY_API_KEY}" \
      GATEWAY_HOST="127.0.0.1" \
      GATEWAY_PORT="${GATEWAY_PORT}" \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
        "${SCRIPT_DIR}/start-gateway-local.sh" &
      gw_pid=$!

      # Brief pause for gateway to bind
      sleep 3

      # --- Run benchmarks ---
      bench_ok=true

      run_bench \
        "${WORKLOAD_THROUGHPUT}" \
        "${out_dir}/throughput.json" \
        "${label}-throughput" \
        "${MODEL_NAME}" \
        "${mml}" \
        || bench_ok=false

      run_bench \
        "${WORKLOAD_LONGCTX}" \
        "${out_dir}/longctx.json" \
        "${label}-longctx" \
        "${MODEL_NAME}" \
        "${mml}" \
        || bench_ok=false

      if ! "${bench_ok}"; then
        write_failed_sentinel "${out_dir}" "${label}" "${MODEL_NAME}" \
          "${mml}" "${gmu}" "${mns}" "benchmark run failed"
        (( failed++ )) || true
      else
        log "${label}: benchmarks complete"
        (( passed++ )) || true
      fi

      # --- Teardown ---
      kill "${gw_pid}" 2>/dev/null || true
      teardown
      log "=== end run ${label} ==="
    done
  done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "sweep complete: ${passed} passed, ${failed} failed (total ${run_idx})"
log "results in: ${RESULTS_DIR}"
