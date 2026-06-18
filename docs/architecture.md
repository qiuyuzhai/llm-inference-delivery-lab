# Architecture

## Delivery flow

```text
Client
→ FastAPI Gateway
→ vLLM OpenAI-compatible API
→ Model weights on local GPU
```

## Components

### vLLM service

Runs the model and exposes OpenAI-compatible endpoints.

### FastAPI gateway

Adds a stable delivery boundary in front of vLLM:

- health endpoint
- metrics endpoint
- API key check
- request timeout
- streaming proxy
- error normalization

### Benchmark harness

Sends concurrent chat completion requests and measures:

- TTFT
- total latency
- approximate TPOT
- request success rate
- P50 / P95 / P99 latency

## Non-goals for MVP

- Kubernetes deployment
- full Grafana dashboards
- quantization experiments
- real domestic accelerator benchmark
- RAG or Agent application logic
