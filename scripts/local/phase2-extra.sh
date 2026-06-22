#!/usr/bin/env bash
# phase2-extra.sh — KV-capacity probe (§9.2) + long-context boundary (#10) + reasoning (#11)
#
# Fixes the sweep's log-loss flaw: containers run WITHOUT --rm so that
# `docker logs` (which carries "GPU KV cache size" and the precheck ValueError)
# is captured to disk on BOTH success and fast-fail, then torn down explicitly.
#
# Phases:
#   A (§9.2) 7B-BF16 KV capacity at gmu{0.6,0.75} x mml{4096,32768}
#            -> measures KV token capacity vs gpu_util / max_model_len.
#   B (#10)  7B-AWQ + 14B-AWQ at gmu0.75, mml ladder 8192/16384/32768/65536
#            -> KV tokens per rung; 65536 with & without VLLM_ALLOW_LONG_MAX_MODEL_LEN
#               to separate "native-context boundary" from "KV-exhaustion boundary";
#               long_context_qa workload run once per model at mml32768.
#   C (#11)  7B-BF16 / 7B-AWQ / 7B-GPTQ-Int4 at gmu0.75, mml8192
#            -> reasoning.jsonl (long output) throughput vs §5.2 knowledge_qa.
#
# Iron rules honoured: no fabricated data; failures recorded honestly; tokens
# only from SSE usage (harness already enforces). No git, no physical servers.
set -uo pipefail   # NOTE: no -e; a single failed run must not abort the batch.

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm-m27:gb10}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-change-me}"
MODELS_ROOT="${MODELS_ROOT:-/home/aaron/models/qwen2.5}"

VLLM_HOST_PORT=8001
GATEWAY_PORT=8081
VLLM_BASE_URL="http://127.0.0.1:${VLLM_HOST_PORT}"
GATEWAY_BASE_URL="http://127.0.0.1:${GATEWAY_PORT}"
CONTAINER_NAME="llm-vllm-extra"
MODEL_NAME="/models/model"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-420}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-5}"

RESULTS_ROOT="${PROJECT_ROOT}/benchmark/results"
WORKLOAD_LONGCTX="benchmark/workloads/long_context_qa.jsonl"
WORKLOAD_REASONING="benchmark/workloads/reasoning.jsonl"

log() { echo "[extra] $(date '+%H:%M:%S') $*"; }
err() { echo "[extra][ERROR] $*" >&2; }

teardown() {
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm   "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  if command -v ss >/dev/null 2>&1; then
    local pids
    pids="$(ss -ltnp "sport = :${GATEWAY_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
    for pid in ${pids}; do kill "${pid}" 2>/dev/null || true; done
  fi
}

# Poll vLLM /health. Returns 0 healthy, 1 exited/timeout.
wait_health() {
  local elapsed=0
  while (( elapsed < HEALTH_TIMEOUT )); do
    if curl -fsS "${VLLM_BASE_URL}/health" >/dev/null 2>&1; then
      log "vLLM healthy after ${elapsed}s"; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}" 2>/dev/null; then
      log "container exited after ${elapsed}s"; return 1
    fi
    sleep "${HEALTH_INTERVAL}"; (( elapsed += HEALTH_INTERVAL )) || true
  done
  log "health timeout ${HEALTH_TIMEOUT}s"; return 1
}

# Parse captured vLLM log -> kv.json (real numbers only; null when absent).
parse_kv() {
  local out_dir="$1" vlog="$2" healthy="$3"
  OUT_DIR="${out_dir}" VLOG="${vlog}" HEALTHY="${healthy}" \
  M_PATH="${RUN_MODEL_PATH}" QUANT="${RUN_QUANT}" MML="${RUN_MML}" \
  GMU="${RUN_GMU}" ALLOW="${RUN_ALLOW}" LBL="${RUN_LABEL}" \
  "${PYTHON_BIN}" - <<'PY'
import os, re, json
out=os.environ["OUT_DIR"]; vlog=os.environ["VLOG"]; healthy=os.environ["HEALTHY"]=="1"
txt=open(vlog, encoding="utf-8", errors="replace").read() if os.path.exists(vlog) else ""
def find(pat, cast=str, grp=1):
    m=re.search(pat, txt)
    if not m: return None
    v=m.group(grp).replace(",", "")
    try: return cast(v)
    except Exception: return None
kv_tokens   = find(r"GPU KV cache size:\s*([\d,]+)\s*tokens", int)
max_conc    = find(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", float)
weight_gib  = find(r"Model loading took\s*([\d.]+)\s*GiB", float)
# Failure reason heuristics
reason=None
if not healthy:
    if re.search(r"max_model_len.*greater than.*max_position_embeddings|derived max_model_len", txt):
        reason="native_context_limit (max_model_len > max_position_embeddings; YaRN/rope_scaling required)"
    elif re.search(r"Free memory on device.*less than desired|less than desired GPU memory", txt):
        reason="precheck_valueerror (startup free memory < gpu_util x 119.7 GiB)"
    elif re.search(r"out of memory|OutOfMemoryError|CUDA error", txt, re.I):
        reason="cuda_oom_during_load"
    else:
        reason="startup_failed_unclassified (see vllm.log tail)"
rec=dict(label=os.environ["LBL"], model_path=os.environ["M_PATH"],
         quantization=(os.environ["QUANT"] or None), max_model_len=int(os.environ["MML"]),
         gpu_memory_utilization=float(os.environ["GMU"]),
         allow_long_max_model_len=os.environ["ALLOW"]=="1",
         healthy=healthy, kv_cache_tokens=kv_tokens, max_concurrency=max_conc,
         weight_gib=weight_gib, fail_reason=reason)
json.dump(rec, open(os.path.join(out,"kv.json"),"w"), indent=2, ensure_ascii=False)
print(f"  kv.json: healthy={healthy} kv_tokens={kv_tokens} max_conc={max_conc} weight={weight_gib} reason={reason}")
PY
}

# Run benchmark workload through gateway -> <out_dir>/<name>.json
run_workload() {
  local out_file="$1" workload="$2" label="$3" concurrency="$4" requests="$5" mml="$6"
  log "starting gateway :${GATEWAY_PORT}"
  VLLM_BASE_URL="${VLLM_BASE_URL}" DEFAULT_MODEL="${MODEL_NAME}" \
  GATEWAY_API_KEY="${GATEWAY_API_KEY}" GATEWAY_HOST="127.0.0.1" \
  GATEWAY_PORT="${GATEWAY_PORT}" PROJECT_ROOT="${PROJECT_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    "${SCRIPT_DIR}/start-gateway-local.sh" &
  local gw_pid=$!
  sleep 3
  log "benchmark: ${workload} c=${concurrency} n=${requests} -> ${out_file}"
  "${PYTHON_BIN}" -m benchmark.harness.cli \
    --base-url "${GATEWAY_BASE_URL}" --api-key "${GATEWAY_API_KEY}" \
    --workload "${workload}" --concurrency "${concurrency}" --requests "${requests}" \
    --output "${out_file}" --label "${label}" \
    --model-name "${MODEL_NAME}" --max-model-len "${mml}" \
    || err "benchmark failed: ${out_file}"
  kill "${gw_pid}" 2>/dev/null || true
}

# Core: start vLLM, capture logs (success or fail), parse KV, optional workload.
#   $1 out_dir  $2 model_path  $3 quant  $4 mml  $5 gmu  $6 allow_long(0/1)
#   $7 workload(file|"")  $8 wl_name  $9 wl_conc  $10 wl_req
run_one() {
  RUN_OUT="$1"; RUN_MODEL_PATH="$2"; RUN_QUANT="$3"; RUN_MML="$4"; RUN_GMU="$5"; RUN_ALLOW="$6"
  local workload="${7:-}" wl_name="${8:-}" wl_conc="${9:-4}" wl_req="${10:-20}"
  RUN_LABEL="$(basename "${RUN_OUT}")"
  mkdir -p "${RUN_OUT}"
  local vlog="${RUN_OUT}/vllm.log"

  # Resumable: skip if already complete.
  if [ -f "${RUN_OUT}/kv.json" ]; then
    if [ -z "${workload}" ] || [ -f "${RUN_OUT}/${wl_name}.json" ]; then
      log "skip (exists): ${RUN_LABEL}"; return 0
    fi
  fi

  log "=== run ${RUN_LABEL} (mml=${RUN_MML} gmu=${RUN_GMU} quant='${RUN_QUANT}' allow_long=${RUN_ALLOW}) ==="
  teardown

  local allow_env=()
  [ "${RUN_ALLOW}" = "1" ] && allow_env=(-e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")

  docker run -d --gpus all --name "${CONTAINER_NAME}" \
    -p "${VLLM_HOST_PORT}:8000" \
    -v "${RUN_MODEL_PATH}:/models/model:ro" \
    -v "${PROJECT_ROOT}/serving/vllm/start-vllm.sh:/app/start-vllm.sh:ro" \
    -e "MODEL_NAME=${MODEL_NAME}" \
    -e "MAX_MODEL_LEN=${RUN_MML}" \
    -e "GPU_MEMORY_UTILIZATION=${RUN_GMU}" \
    -e "MAX_NUM_SEQS=32" \
    -e "QUANTIZATION=${RUN_QUANT}" \
    "${allow_env[@]}" \
    "${VLLM_IMAGE}" "/app/start-vllm.sh" >/dev/null 2>&1

  local healthy=0
  if wait_health; then healthy=1; fi

  # CRITICAL: capture logs BEFORE teardown (works whether running or exited).
  docker logs "${CONTAINER_NAME}" > "${vlog}" 2>&1 || true

  parse_kv "${RUN_OUT}" "${vlog}" "${healthy}"

  if [ "${healthy}" = "1" ] && [ -n "${workload}" ]; then
    run_workload "${RUN_OUT}/${wl_name}.json" "${workload}" "${RUN_LABEL}-${wl_name}" "${wl_conc}" "${wl_req}" "${RUN_MML}"
  fi

  teardown
  log "=== end ${RUN_LABEL} ==="
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"
command -v docker >/dev/null 2>&1 || { err "docker missing"; exit 1; }
docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || { err "image missing: ${VLLM_IMAGE}"; exit 1; }
[ -x "${PYTHON_BIN}" ] || { err "python missing: ${PYTHON_BIN}"; exit 1; }

M_7B_BF16="${MODELS_ROOT}/Qwen2.5-7B-Instruct"
M_7B_AWQ="${MODELS_ROOT}/Qwen2.5-7B-Instruct-AWQ"
M_7B_GPTQ="${MODELS_ROOT}/Qwen2.5-7B-Instruct-GPTQ-Int4"
M_14B_AWQ="${MODELS_ROOT}/Qwen2.5-14B-Instruct-AWQ"
for m in "${M_7B_BF16}" "${M_7B_AWQ}" "${M_7B_GPTQ}" "${M_14B_AWQ}"; do
  [ -d "${m}" ] || { err "model missing: ${m}"; exit 1; }
done

log "phase2-extra starting; results under ${RESULTS_ROOT}/{kv-capacity,long-context-oom,reasoning}"

# ===========================================================================
# Phase A — §9.2 KV capacity (7B-BF16)
# ===========================================================================
log "########## Phase A: §9.2 KV capacity (7B-BF16) ##########"
for gmu in 0.6 0.75; do
  for mml in 4096 32768; do
    run_one "${RESULTS_ROOT}/kv-capacity/bf16-gmu${gmu}-mml${mml}" \
            "${M_7B_BF16}" "" "${mml}" "${gmu}" "0"
  done
done

# ===========================================================================
# Phase B — #10 long-context boundary (7B-AWQ, 14B-AWQ @ gmu0.75)
# ===========================================================================
log "########## Phase B: #10 long-context boundary ##########"
oom_dir="${RESULTS_ROOT}/long-context-oom"
# 7B-AWQ ladder
for mml in 8192 16384 32768; do
  wl=""; wn=""
  if [ "${mml}" = "32768" ]; then wl="${WORKLOAD_LONGCTX}"; wn="longctx"; fi
  run_one "${oom_dir}/7B-AWQ-mml${mml}" "${M_7B_AWQ}" "awq_marlin" "${mml}" "0.75" "0" "${wl}" "${wn}" 4 20
done
run_one "${oom_dir}/7B-AWQ-mml65536-noallow"  "${M_7B_AWQ}" "awq_marlin" "65536" "0.75" "0"
run_one "${oom_dir}/7B-AWQ-mml65536-allow"    "${M_7B_AWQ}" "awq_marlin" "65536" "0.75" "1"
# 14B-AWQ ladder
for mml in 8192 16384 32768; do
  wl=""; wn=""
  if [ "${mml}" = "32768" ]; then wl="${WORKLOAD_LONGCTX}"; wn="longctx"; fi
  run_one "${oom_dir}/14B-AWQ-mml${mml}" "${M_14B_AWQ}" "awq_marlin" "${mml}" "0.75" "0" "${wl}" "${wn}" 4 20
done
run_one "${oom_dir}/14B-AWQ-mml65536-noallow" "${M_14B_AWQ}" "awq_marlin" "65536" "0.75" "0"
run_one "${oom_dir}/14B-AWQ-mml65536-allow"   "${M_14B_AWQ}" "awq_marlin" "65536" "0.75" "1"

# ===========================================================================
# Phase C — #11 reasoning workload (7B 3 precisions @ gmu0.75, mml8192, c1)
# ===========================================================================
log "########## Phase C: #11 reasoning workload ##########"
rsn_dir="${RESULTS_ROOT}/reasoning"
run_one "${rsn_dir}/7B-BF16" "${M_7B_BF16}" ""            "8192" "0.75" "0" "${WORKLOAD_REASONING}" "reasoning" 1 10
run_one "${rsn_dir}/7B-AWQ"  "${M_7B_AWQ}"  "awq_marlin"  "8192" "0.75" "0" "${WORKLOAD_REASONING}" "reasoning" 1 10
run_one "${rsn_dir}/7B-GPTQ" "${M_7B_GPTQ}" "gptq_marlin" "8192" "0.75" "0" "${WORKLOAD_REASONING}" "reasoning" 1 10

teardown
log "phase2-extra complete: A(§9.2 KV)+B(#10 OOM)+C(#11 reasoning) done"
