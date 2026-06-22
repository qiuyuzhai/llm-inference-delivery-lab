#!/usr/bin/env bash
# Phase 2 — 7B 三精度矩阵「放行即跑」预置包装脚本
#
# 用途：team-lead 放行后一键执行 Qwen2.5-7B 的 BF16 / AWQ / GPTQ-Int4 对比。
# 本脚本只固化参数，真实编排逻辑全部委托 run-quant-matrix.sh（KISS + DRY）。
#
# 放行前提（由 team-lead 显式确认）：
#   1) GPU 已释放（serving-eng 的 #7 验证已停服）。
#   2) #7 给出 marlin kernel 可用性结论——据此设置下方 AWQ_QUANT / GPTQ_QUANT。
#   3) 7B-BF16 权重下载完成（AWQ / GPTQ-Int4 已就位）。
#
# 控制变量：同一 workload、同一 max_model_len、同一采样口径，仅精度变化。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- #7 结论开关 -----------------------------------------------------------
# marlin 可用（默认，待 #7 确认）：awq_marlin / gptq_marlin
# 若 #7 报 marlin 不支持需回退：把这两个改为 awq / gptq
AWQ_QUANT="${AWQ_QUANT:-awq_marlin}"
GPTQ_QUANT="${GPTQ_QUANT:-gptq_marlin}"

# --- 固定的 7B 三组合矩阵 --------------------------------------------------
# 格式：<模型目录基名>:<精度标签>:<QUANTIZATION 取值>（BF16 留空）
MATRIX="\
Qwen2.5-7B-Instruct:bf16: \
Qwen2.5-7B-Instruct-AWQ:awq:${AWQ_QUANT} \
Qwen2.5-7B-Instruct-GPTQ-Int4:gptq-int4:${GPTQ_QUANT}"

# --- 端口（避开已占用：8000=engineering/api python，8080=gunicorn）---------
# vLLM 用 8001，gateway 用 8081；gateway 上游 VLLM_BASE_URL 指向 vLLM。
export VLLM_HOST_PORT="${VLLM_HOST_PORT:-8001}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_HOST_PORT}}"
export GATEWAY_PORT="${GATEWAY_PORT:-8081}"
export GATEWAY_BASE_URL="${GATEWAY_BASE_URL:-http://127.0.0.1:${GATEWAY_PORT}}"

# --- 统一基准参数（控制变量）----------------------------------------------
export MODELS_DIR="${MODELS_DIR:-/home/aaron/models/qwen2.5}"
export WORKLOAD="${WORKLOAD:-benchmark/workloads/knowledge_qa.jsonl}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

# 并发轮次：run-quant-matrix.sh 按 <model>-<precision>.json 命名结果，不含并发维度，
# 故不同并发轮必须用独立 RESULTS_DIR 隔离，否则 c8 会覆盖 c1。默认 c1 写入
# benchmark/results/c1，高并发轮（见文末示例）写 benchmark/results/c8。
export CONCURRENCY="${CONCURRENCY:-1}"
export REQUESTS="${REQUESTS:-16}"
export RESULTS_DIR="${RESULTS_DIR:-benchmark/results/c${CONCURRENCY}}"

echo "[run-7b-matrix] AWQ_QUANT=${AWQ_QUANT}  GPTQ_QUANT=${GPTQ_QUANT}"
echo "[run-7b-matrix] CONCURRENCY=${CONCURRENCY}  REQUESTS=${REQUESTS}  WORKLOAD=${WORKLOAD}"
echo "[run-7b-matrix] 委托 run-quant-matrix.sh 执行 7B 三组合 ..."

MATRIX="${MATRIX}" bash "${SCRIPT_DIR}/run-quant-matrix.sh"

# 高并发轮（concurrency=8）：RESULTS_DIR 默认按 c${CONCURRENCY} 派生为 benchmark/results/c8，
# 不会覆盖 c1 结果。放行后按需执行：
#   CONCURRENCY=8 AWQ_QUANT=${AWQ_QUANT} GPTQ_QUANT=${GPTQ_QUANT} \
#     bash "${SCRIPT_DIR}/run-7b-matrix.sh"
