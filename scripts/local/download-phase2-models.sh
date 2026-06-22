#!/usr/bin/env bash
# Phase 2 量化对比模型下载脚本
# 通过 hf-mirror 镜像下载 Qwen2.5 7B/14B 的 BF16 / AWQ / GPTQ-Int4 三精度权重。
# 权重不进 Git 仓库，统一存放于项目外独立目录，由启动脚本通过 MODEL_NAME 引用。
set -u

export HF_ENDPOINT="https://hf-mirror.com"
MODELS_DIR="/home/aaron/models/qwen2.5"
HFCLI="/home/aaron/.local/bin/hf"

mkdir -p "$MODELS_DIR"

# 顺序：先下小体积量化版 + 7B，让团队尽早跑通 7B 闭环；14B 随后。
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct-AWQ"
  "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct-AWQ"
  "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
  "Qwen/Qwen2.5-14B-Instruct"
)

for M in "${MODELS[@]}"; do
  DEST="$MODELS_DIR/$(basename "$M")"
  echo "=== downloading $M -> $DEST ==="
  ok=0
  for attempt in 1 2 3; do
    if "$HFCLI" download "$M" --local-dir "$DEST"; then
      echo "OK $M"
      ok=1
      break
    fi
    echo "retry $attempt failed for $M"
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "FAILED $M after 3 attempts"
  fi
done

echo "=== ALL DONE ==="
