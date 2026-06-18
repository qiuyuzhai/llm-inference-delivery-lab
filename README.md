# LLM Inference Delivery Lab

Enterprise-style LLM inference delivery lab for turning 14B-class model weights into a deployable, measurable, and operable inference service.

## What this project builds

This repository simulates a real LLM inference delivery workflow:

```text
model weights
→ vLLM / SGLang serving
→ Docker packaging
→ FastAPI API gateway
→ benchmark harness
→ metrics
→ reports
→ future Kubernetes and heterogeneous migration extensions
```

## MVP scope

The first MVP focuses on:

- vLLM OpenAI-compatible serving.
- FastAPI gateway.
- Streaming chat client.
- Async benchmark harness.
- TTFT / TPOT / TPS / P50 / P95 / P99 metrics.
- Docker Compose delivery package.

## Primary resources

- Local GPU: NVIDIA GB10 / GDX Spark.
- Control plane nodes: three Ubuntu 22.04.5 servers, each with 24 CPU cores.
- Budget strategy: zero-cost first; free GPU platforms are evaluated only for lightweight experiments.

## Model strategy

- Primary model: `Qwen/Qwen2.5-14B-Instruct`.
- Baseline model: `Qwen/Qwen2.5-7B-Instruct`.
- Reasoning workload: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
- Boundary analysis: `Qwen/Qwen2.5-72B-Instruct`, estimate only.

## Quick start

```bash
cp .env.example .env
docker compose -f deploy/docker-compose/docker-compose.yml up --build
```

Then call:

```bash
curl http://localhost:8080/health
```

## Documentation

- Architecture: `docs/architecture.md`
- Scenario/SLO matrix: `docs/scenario-slo-matrix.md`
- Experiment template: `docs/experiment-template.md`
- Report template: `docs/report-template.md`
