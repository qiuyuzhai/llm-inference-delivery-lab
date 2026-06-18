# MVP Report

## Goal

Run a 14B-class model through vLLM, expose it through the FastAPI gateway, and collect basic benchmark metrics.

## First-run commands

Default Compose path for online environments:

```bash
cp .env.example .env
docker compose --env-file .env -f deploy/docker-compose/docker-compose.yml up --build
```

GB10 local validation path used for this run:

```bash
docker run --rm --gpus all --name llm-vllm -p 8000:8000 \
  -v "/home/aaron/Desktop/minimax-m2.7-local/models/Qwen2.5-1.5B-Instruct:/models/model:ro" \
  -v "/home/aaron/Desktop/llm-inference-delivery-lab/serving/vllm/start-vllm.sh:/app/start-vllm.sh:ro" \
  -e "MODEL_NAME=/models/model" \
  -e "MAX_MODEL_LEN=4096" \
  -e "GPU_MEMORY_UTILIZATION=0.75" \
  -e "MAX_NUM_SEQS=8" \
  "vllm-m27:gb10" "/app/start-vllm.sh"

VLLM_BASE_URL="http://127.0.0.1:8000" DEFAULT_MODEL="/models/model" GATEWAY_API_KEY="change-me" \
  .venv/bin/python -m uvicorn gateway.app.main:app --host 127.0.0.1 --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Streaming client:

```bash
python serving/clients/stream_chat.py --base-url http://127.0.0.1:8080 --api-key change-me --prompt "请解释 KV Cache 为什么影响大模型推理并发。"
```

Streaming benchmark smoke test:

```bash
python -m benchmark.harness.cli --base-url http://127.0.0.1:8080 --api-key change-me --workload benchmark/workloads/knowledge_qa.jsonl --concurrency 1 --requests 3 --output benchmark/results/smoke.json
```

## Runtime validation results

| Metric | Value |
|---|---|
| Runtime date | 2026-06-08 |
| Hardware | NVIDIA GB10 / CUDA 13.0 / Driver 580.82.09 |
| vLLM image | `vllm-m27:gb10` |
| vLLM version | `0.21.0` |
| Model | local `Qwen2.5-1.5B-Instruct` smoke model at `/models/model` |
| max_model_len | `4096` |
| max_num_seqs | `8` |
| gpu_memory_utilization | `0.75` |
| Successful requests | `3` |
| Failed requests | `0` |
| Measured token requests | `3` |
| Requests per second | `0.2501` |
| Latency P50 | `5.1985s` |
| Latency P95 | `5.8299s` |
| Latency P99 | `5.8299s` |
| TTFT P50 | `0.0520s` |
| TTFT P95 | `0.0531s` |
| Chunk count P50 | `319` |
| Chunk count P95 | `352` |
| Inter-chunk gap avg | `0.0160s` |
| Inter-chunk gap P95 | `0.0208s` |
| Completion tokens P50 | `316` |
| Completion tokens P95 | `349` |
| Tokens per second P50 | `59.8935` |
| Tokens per second P95 | `60.7866` |
| TPOT P50 | `0.0167s` |
| TPOT P95 | `0.0167s` |

## Validation notes

- Gateway `/health` returned `{"status":"ok"}`.
- Non-streaming chat completion returned valid OpenAI-compatible JSON through Gateway.
- Streaming client returned SSE chunks through Gateway.
- Streaming benchmark output was written to `benchmark/results/smoke.json`; generated benchmark artifacts are local-only by default.
- Streaming benchmark requested `stream_options.include_usage=true`, and vLLM emitted OpenAI-compatible `usage` for all 3 requests.

## Known limits

- MVP metrics are not production SLOs.
- This run used the locally available 1.5B smoke model because Docker Hub access timed out and the 14B model was not locally cached.
- TPS and TPOT are reported only when the streaming server emits OpenAI-compatible `usage`; tokens are never estimated from bytes or characters.
- Free GPU platforms are not used as serving evidence.
