#!/usr/bin/env bash
# Phase 2 量化对比矩阵编排脚本（骨架）
#
# 目标：参数化遍历 (模型, 精度) 组合，对每个组合执行端到端闭环：
#   1) 启动 vLLM（按精度注入 QUANTIZATION）
#   2) 轮询等待 vLLM /health 就绪
#   3) 启动本地 Gateway（后台）并等待其 /health 就绪
#   4) 跑 benchmark harness，结果落盘到 benchmark/results/<label>.json
#   5) 停服清理，进入下一组合
#
# 设计原则（KISS + DRY）：本脚本不重造启动逻辑，而是复用 scripts/local/ 下
# 已有脚本（start-vllm-gb10.sh / start-gateway-local.sh / stop-local-runtime.sh）
# 与 benchmark.harness.cli。真实模型路径用变量占位，由 bench-runner 后续跑真实矩阵。
#
# 安全约定：禁止伪造数据。某组合若 OOM / kernel 不支持 / 启动失败，
# 本脚本如实跳过该组合并在控制台标记 FAILED，不会写出占位/编造的结果文件。
set -euo pipefail

# ---------------------------------------------------------------------------
# 可配置变量（均可通过环境变量覆盖）
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/aaron/Desktop/llm-inference-delivery-lab}"
# 预量化权重落地根目录（与 scripts/local/download-phase2-models.sh 保持一致）。
MODELS_DIR="${MODELS_DIR:-/home/aaron/models/qwen2.5}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

# 服务端口与寻址。
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_HOST_PORT}}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
GATEWAY_BASE_URL="${GATEWAY_BASE_URL:-http://${GATEWAY_HOST}:${GATEWAY_PORT}}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-change-me}"

# benchmark 负载参数。
WORKLOAD="${WORKLOAD:-benchmark/workloads/knowledge_qa.jsonl}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS="${REQUESTS:-8}"
RESULTS_DIR="${RESULTS_DIR:-benchmark/results}"

# 健康检查轮询参数（vLLM 大模型加载可能较慢，给足超时）。
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
HEALTH_INTERVAL_SECONDS="${HEALTH_INTERVAL_SECONDS:-5}"

# vLLM 运行参数（按需在外层覆盖；7B 与 14B 可用不同 GPU_MEMORY_UTILIZATION）。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

# 要遍历的精度矩阵。每行格式： <模型基名>:<精度标签>:<QUANTIZATION值>
#   - 模型基名：MODELS_DIR 下的子目录名（与下载脚本 basename 一致）。
#   - 精度标签：用于结果文件命名（<模型>-<标签>.json）。
#   - QUANTIZATION 值：传给 vLLM 的量化方法；BF16 留空字符串。
#
# 映射依据 serving/vllm/start-vllm.sh 注释：
#   AWQ 权重     -> awq_marlin（更快的 marlin kernel；旧版 vLLM 退回 awq）
#   GPTQ-Int4    -> gptq_marlin（旧版 vLLM 退回 gptq）
#   BF16（无量化）-> 空
#
# 默认覆盖 7B/14B × BF16/AWQ/GPTQ-Int4 共 6 组合；可用 MATRIX 环境变量覆盖。
DEFAULT_MATRIX=(
  "Qwen2.5-7B-Instruct:bf16:"
  "Qwen2.5-7B-Instruct-AWQ:awq:awq_marlin"
  "Qwen2.5-7B-Instruct-GPTQ-Int4:gptq-int4:gptq_marlin"
  "Qwen2.5-14B-Instruct:bf16:"
  "Qwen2.5-14B-Instruct-AWQ:awq:awq_marlin"
  "Qwen2.5-14B-Instruct-GPTQ-Int4:gptq-int4:gptq_marlin"
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
log() { printf '[run-quant-matrix] %s\n' "$*"; }

# 轮询某 /health 端点直到就绪或超时。返回非零表示超时。
wait_for_health() {
  local url="$1" name="$2"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
  log "等待 ${name} 就绪：${url}/health（超时 ${HEALTH_TIMEOUT_SECONDS}s）"
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if curl -fsS "${url}/health" >/dev/null 2>&1; then
      log "${name} 已就绪"
      return 0
    fi
    sleep "${HEALTH_INTERVAL_SECONDS}"
  done
  log "ERROR：${name} 在超时内未就绪"
  return 1
}

# 停止本轮起的 vLLM 容器与 Gateway 进程（复用现有清理脚本 + 兜底）。
GATEWAY_PID=""
teardown() {
  if [ -n "${GATEWAY_PID}" ] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
    wait "${GATEWAY_PID}" 2>/dev/null || true
  fi
  GATEWAY_PID=""
  GATEWAY_PORT="${GATEWAY_PORT}" \
  bash "${PROJECT_ROOT}/scripts/local/stop-local-runtime.sh" >/dev/null 2>&1 || true
}

# 任何退出路径都执行清理，避免残留容器/进程。
trap teardown EXIT

# ---------------------------------------------------------------------------
# 启动 vLLM 并等待其 /health 就绪。成功返回 0，失败返回非 0（已 teardown）。
# 用一个用途专一的函数封装，便于 gptq_marlin→gptq 回退时复用。
start_vllm_wait() {
  local model_path="$1" quant="$2" vllm_log="$3"
  log "启动 vLLM（QUANTIZATION='${quant}'）..."
  MODEL_PATH="${model_path}" \
  QUANTIZATION="${quant}" \
  VLLM_HOST_PORT="${VLLM_HOST_PORT}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  bash "${PROJECT_ROOT}/scripts/local/start-vllm-gb10.sh" \
    > "${vllm_log}" 2>&1 &
  if wait_for_health "${VLLM_BASE_URL}" "vLLM"; then
    return 0
  fi
  # 未就绪：可能 OOM / kernel 不支持 / 加载失败。清理后返回失败，由调用方决定回退。
  teardown
  return 1
}

# 单组合执行：返回 0=成功落盘，非 0=失败（已记录，继续下一组合）
# ---------------------------------------------------------------------------
run_combo() {
  local model_basename="$1" precision_label="$2" quant_value="$3"
  local model_path="${MODELS_DIR}/${model_basename}"
  local label="${model_basename}-${precision_label}"
  local output="${RESULTS_DIR}/${label}.json"

  # dtype 元数据：BF16 组合显式记 bfloat16，量化组合记 auto（由权重决定）。
  local dtype_meta="auto"
  [ -z "${quant_value}" ] && dtype_meta="bfloat16"

  log "=============================================================="
  log "组合：${model_basename}  精度=${precision_label}  QUANTIZATION='${quant_value}'"
  log "模型路径：${model_path}"
  log "结果输出：${output}"

  if [ ! -d "${model_path}" ]; then
    log "SKIP（模型权重缺失，尚未下载）：${model_path}"
    return 1
  fi

  # 1) 启动 vLLM；gptq_marlin 失败时自动回退到 gptq 重试一次（lead 要求）。
  #    effective_quant 记录【实际成功加载】的量化值，写入结果 config，保证诚实可复现。
  local effective_quant="${quant_value}"
  if ! start_vllm_wait "${model_path}" "${quant_value}" "${RESULTS_DIR}/${label}.vllm.log"; then
    if [ "${quant_value}" = "gptq_marlin" ]; then
      log "WARN：gptq_marlin 未就绪，回退 QUANTIZATION=gptq 重试一次 ..."
      if start_vllm_wait "${model_path}" "gptq" "${RESULTS_DIR}/${label}.vllm.gptq-fallback.log"; then
        effective_quant="gptq"
        log "回退成功：实际使用 QUANTIZATION=gptq"
      else
        log "FAILED：gptq_marlin 与 gptq 均未就绪，见 ${label}.vllm.log 与 ${label}.vllm.gptq-fallback.log"
        return 1
      fi
    else
      log "FAILED：vLLM 未就绪（可能 OOM / kernel 不支持 / 加载失败），见 ${label}.vllm.log"
      return 1
    fi
  fi

  # 记录到结果 config 段的量化标识：BF16 用 "none"，量化版用【实际成功加载】的方法名。
  local quant_meta="${effective_quant:-none}"

  # 3) 启动 Gateway（后台）并等待其 /health。
  log "启动 Gateway ..."
  VLLM_BASE_URL="${VLLM_BASE_URL}" \
  GATEWAY_API_KEY="${GATEWAY_API_KEY}" \
  GATEWAY_HOST="${GATEWAY_HOST}" \
  GATEWAY_PORT="${GATEWAY_PORT}" \
  bash "${PROJECT_ROOT}/scripts/local/start-gateway-local.sh" \
    > "${RESULTS_DIR}/${label}.gateway.log" 2>&1 &
  GATEWAY_PID=$!

  if ! wait_for_health "${GATEWAY_BASE_URL}" "Gateway"; then
    log "FAILED：Gateway 未就绪，见 ${label}.gateway.log"
    teardown
    return 1
  fi

  # 4) 跑 benchmark，结果落盘。
  log "执行 benchmark（concurrency=${CONCURRENCY} requests=${REQUESTS}）..."
  # 透传实验元数据，使结果 JSON 的 config 段非空，便于 compare_runs.py 分组。
  if (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -m benchmark.harness.cli \
        --base-url "${GATEWAY_BASE_URL}" \
        --api-key "${GATEWAY_API_KEY}" \
        --workload "${WORKLOAD}" \
        --concurrency "${CONCURRENCY}" \
        --requests "${REQUESTS}" \
        --output "${output}" \
        --label "${label}" \
        --model-name "${model_basename}" \
        --quantization "${quant_meta}" \
        --dtype "${dtype_meta}" \
        --max-model-len "${MAX_MODEL_LEN}"); then
    log "OK：结果已写入 ${output}"
    teardown
    return 0
  else
    log "FAILED：benchmark 执行失败"
    teardown
    return 1
  fi
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
main() {
  cd "${PROJECT_ROOT}"
  mkdir -p "${RESULTS_DIR}"

  if [ ! -x "${PYTHON_BIN}" ]; then
    log "ERROR：python binary 不存在：${PYTHON_BIN}"
    exit 1
  fi

  # 允许外部用空白分隔的 MATRIX 覆盖默认矩阵。
  local -a matrix
  if [ -n "${MATRIX:-}" ]; then
    # shellcheck disable=SC2206
    matrix=(${MATRIX})
  else
    matrix=("${DEFAULT_MATRIX[@]}")
  fi

  local ok_count=0 fail_count=0
  local -a ok_labels=() fail_labels=()

  for entry in "${matrix[@]}"; do
    IFS=':' read -r model_basename precision_label quant_value <<< "${entry}"
    if run_combo "${model_basename}" "${precision_label}" "${quant_value}"; then
      ok_count=$((ok_count + 1))
      ok_labels+=("${model_basename}-${precision_label}")
    else
      fail_count=$((fail_count + 1))
      fail_labels+=("${model_basename}-${precision_label}")
    fi
  done

  log "=============================================================="
  log "矩阵执行完成：成功 ${ok_count} / 失败 ${fail_count}"
  [ "${ok_count}" -gt 0 ] && log "成功组合：${ok_labels[*]}"
  [ "${fail_count}" -gt 0 ] && log "失败/跳过组合：${fail_labels[*]}"
  log "结果目录：${RESULTS_DIR}"
}

main "$@"
