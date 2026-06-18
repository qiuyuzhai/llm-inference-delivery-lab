# Scenario and SLO Matrix

## Scenario 1: Enterprise knowledge QA

| Field | Value |
|---|---|
| Primary model | `Qwen/Qwen2.5-14B-Instruct` |
| Baseline model | `Qwen/Qwen2.5-7B-Instruct` |
| Prompt style | fixed system prompt + medium user question + long context |
| Key metrics | TTFT, P95, P99, TPS, error rate |
| MVP target | collect metrics, do not claim production SLO before measurement |

## Scenario 2: Technical assistant

| Field | Value |
|---|---|
| Model | initially reuse `Qwen/Qwen2.5-14B-Instruct` |
| Prompt style | longer prompt, code snippets, longer output |
| Key metrics | TTFT, TPOT, streaming stability, output truncation |

## Scenario 3: Reasoning pressure workload

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Prompt style | analysis tasks with long output |
| Key metrics | TPOT, P99, timeout rate, queueing impact |

## Boundary analysis

`Qwen/Qwen2.5-72B-Instruct` is used only for resource estimation in MVP planning. It is not a required runtime dependency.
