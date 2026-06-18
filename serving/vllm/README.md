# vLLM Serving

This directory contains the startup script for the upstream vLLM OpenAI-compatible API server.

## Default model

`Qwen/Qwen2.5-14B-Instruct`

## Key parameters

| Environment variable | Default |
|---|---|
| MODEL_NAME | `Qwen/Qwen2.5-14B-Instruct` |
| MAX_MODEL_LEN | `8192` |
| GPU_MEMORY_UTILIZATION | `0.85` |
| MAX_NUM_SEQS | `32` |
| VLLM_DTYPE | `auto` |

## Run locally

```bash
MODEL_NAME=Qwen/Qwen2.5-14B-Instruct serving/vllm/start-vllm.sh
```
