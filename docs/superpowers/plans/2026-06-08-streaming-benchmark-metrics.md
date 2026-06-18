# Streaming Benchmark Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the benchmark harness into a streaming-only chat benchmark with TTFT, chunk rhythm, and server-provided token metrics.

**Architecture:** Keep execution in `benchmark/harness/runner.py`, reusable math/parsing helpers in `benchmark/harness/metrics.py`, CLI wiring in `benchmark/harness/cli.py`, and reporting language in `docs/mvp-report.md` plus learning notes. Token metrics are only populated from real OpenAI-compatible streaming `usage` objects; the harness never estimates tokens from bytes or characters.

**Tech Stack:** Python 3.11+, httpx async streaming, dataclasses, pytest, ruff, Typer.

---

## Scope Check

This plan covers one focused subsystem: streaming benchmark metrics. It intentionally excludes non-streaming benchmark metrics, tokenizer dependencies, Prometheus/Grafana, model download automation, and parameter sweep orchestration.

## File Structure

```text
llm-inference-delivery-lab/
├── benchmark/
│   └── harness/
│       ├── cli.py              # remove meaningful stream toggle; benchmark is streaming-only
│       ├── metrics.py          # add SSE usage parsing, numeric summaries, gap helpers
│       └── runner.py           # record streaming chunk timings and token stats
├── tests/
│   └── benchmark/
│       ├── test_metrics.py     # unit tests for parsing and summaries
│       └── test_runner.py      # unit tests for summary construction helpers
├── docs/
│   └── mvp-report.md           # document streaming-first benchmark semantics
└── learning/
    └── issues-and-fixes.md     # add learning note about token metrics not being estimated
```

Boundary decisions:

- `metrics.py` owns pure functions: percentile, summaries, SSE JSON usage extraction, token metric derivation.
- `runner.py` owns network execution and per-request timing.
- `cli.py` owns argument parsing only.
- Documentation explains benchmark semantics but does not duplicate implementation details.

---

### Task 1: Add streaming metric helper tests

**Files:**
- Modify: `tests/benchmark/test_metrics.py`

- [ ] **Step 1: Replace `tests/benchmark/test_metrics.py` with expanded tests**

```python
from benchmark.harness.metrics import (
    extract_usage_from_sse,
    percentile,
    summarize_latencies,
    summarize_optional_numbers,
)


def test_percentile_returns_nearest_rank_value():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 50) == 2.0
    assert percentile(values, 95) == 4.0


def test_summarize_latencies_returns_expected_fields():
    summary = summarize_latencies([1.0, 2.0, 3.0])

    assert summary["count"] == 3
    assert summary["p50"] == 2.0
    assert summary["p95"] == 3.0
    assert summary["p99"] == 3.0


def test_extract_usage_from_sse_reads_openai_usage():
    chunk = b'data: {"usage":{"prompt_tokens":3,"completion_tokens":7,"total_tokens":10}}\n\n'

    usage = extract_usage_from_sse(chunk)

    assert usage == {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}


def test_extract_usage_from_sse_ignores_missing_usage():
    chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'

    assert extract_usage_from_sse(chunk) is None


def test_extract_usage_from_sse_ignores_done_and_malformed_json():
    assert extract_usage_from_sse(b"data: [DONE]\n\n") is None
    assert extract_usage_from_sse(b"data: not-json\n\n") is None


def test_summarize_optional_numbers_ignores_none_values():
    summary = summarize_optional_numbers([None, 1.0, 3.0])

    assert summary["count"] == 2
    assert summary["p50"] == 1.0
    assert summary["p95"] == 3.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m pytest tests/benchmark/test_metrics.py -q
```

Expected: FAIL with imports missing for `extract_usage_from_sse` and `summarize_optional_numbers`.

---

### Task 2: Implement streaming metric helpers

**Files:**
- Modify: `benchmark/harness/metrics.py`
- Test: `tests/benchmark/test_metrics.py`

- [ ] **Step 1: Replace `benchmark/harness/metrics.py`**

```python
import json
import math
from typing import Any


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil((percent / 100) * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "avg": sum(values) / len(values) if values else 0.0,
    }


def summarize_optional_numbers(values: list[float | None]) -> dict[str, float | int]:
    measured = [value for value in values if value is not None]
    return summarize_latencies(measured)


def extract_usage_from_sse(chunk: bytes) -> dict[str, int] | None:
    for raw_line in chunk.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            continue
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            continue
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    return None


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
```

- [ ] **Step 2: Run metric tests**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m pytest tests/benchmark/test_metrics.py -q
```

Expected: PASS with `6 passed`.

---

### Task 3: Add runner summary tests

**Files:**
- Create: `tests/benchmark/test_runner.py`

- [ ] **Step 1: Create `tests/benchmark/test_runner.py`**

```python
from benchmark.harness.runner import RequestResult, summarize_results


def test_summarize_results_separates_token_measured_requests():
    results = [
        RequestResult(
            id="r1",
            ok=True,
            latency_seconds=2.0,
            ttft_seconds=0.2,
            chunk_count=3,
            first_chunk_bytes=10,
            output_bytes=100,
            inter_chunk_gap_avg_seconds=0.4,
            inter_chunk_gap_p95_seconds=0.5,
            prompt_tokens=5,
            completion_tokens=20,
            total_tokens=25,
            tokens_per_second=10.0,
            tpot_seconds=0.1,
        ),
        RequestResult(
            id="r2",
            ok=True,
            latency_seconds=1.0,
            ttft_seconds=0.1,
            chunk_count=2,
            first_chunk_bytes=8,
            output_bytes=50,
            inter_chunk_gap_avg_seconds=0.3,
            inter_chunk_gap_p95_seconds=0.3,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_second=None,
            tpot_seconds=None,
        ),
        RequestResult(
            id="r3",
            ok=False,
            latency_seconds=0.5,
            ttft_seconds=None,
            chunk_count=0,
            first_chunk_bytes=0,
            output_bytes=0,
            inter_chunk_gap_avg_seconds=None,
            inter_chunk_gap_p95_seconds=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_second=None,
            tpot_seconds=None,
            error="boom",
        ),
    ]

    summary = summarize_results(results, total_seconds=3.0)

    assert summary["total_requests"] == 3
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["measured_token_requests"] == 1
    assert summary["latency_seconds"]["count"] == 2
    assert summary["chunk_count"]["p95"] == 3
    assert summary["completion_tokens"]["count"] == 1
    assert summary["tokens_per_second"]["p50"] == 10.0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m pytest tests/benchmark/test_runner.py -q
```

Expected: FAIL because `summarize_results` does not exist and `RequestResult` lacks new fields.

---

### Task 4: Implement streaming-only runner

**Files:**
- Modify: `benchmark/harness/runner.py`
- Test: `tests/benchmark/test_runner.py`

- [ ] **Step 1: Replace `benchmark/harness/runner.py`**

```python
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from benchmark.harness.metrics import (
    average,
    extract_usage_from_sse,
    percentile,
    summarize_latencies,
    summarize_optional_numbers,
)
from benchmark.harness.workload import WorkloadItem


@dataclass(frozen=True)
class RequestResult:
    id: str
    ok: bool
    latency_seconds: float
    ttft_seconds: float | None
    chunk_count: int
    first_chunk_bytes: int
    output_bytes: int
    inter_chunk_gap_avg_seconds: float | None
    inter_chunk_gap_p95_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    tokens_per_second: float | None
    tpot_seconds: float | None
    error: str | None = None


def build_success_result(
    *,
    item_id: str,
    started: float,
    ended: float,
    chunk_timestamps: list[float],
    chunk_sizes: list[int],
    usage: dict[str, int] | None,
) -> RequestResult:
    latency_seconds = ended - started
    ttft_seconds = (chunk_timestamps[0] - started) if chunk_timestamps else None
    gaps = [current - previous for previous, current in zip(chunk_timestamps, chunk_timestamps[1:])]
    completion_tokens = usage["completion_tokens"] if usage else None
    tokens_per_second = (
        completion_tokens / latency_seconds
        if completion_tokens is not None and latency_seconds > 0
        else None
    )
    tpot_seconds = (
        latency_seconds / completion_tokens
        if completion_tokens is not None and completion_tokens > 0
        else None
    )
    return RequestResult(
        id=item_id,
        ok=True,
        latency_seconds=latency_seconds,
        ttft_seconds=ttft_seconds,
        chunk_count=len(chunk_sizes),
        first_chunk_bytes=chunk_sizes[0] if chunk_sizes else 0,
        output_bytes=sum(chunk_sizes),
        inter_chunk_gap_avg_seconds=average(gaps),
        inter_chunk_gap_p95_seconds=percentile(gaps, 95) if gaps else None,
        prompt_tokens=usage["prompt_tokens"] if usage else None,
        completion_tokens=completion_tokens,
        total_tokens=usage["total_tokens"] if usage else None,
        tokens_per_second=tokens_per_second,
        tpot_seconds=tpot_seconds,
    )


async def run_one(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    item: WorkloadItem,
) -> RequestResult:
    payload = {
        "messages": [{"role": "user", "content": item.prompt}],
        "max_tokens": item.max_tokens,
        "stream": True,
    }
    started = time.perf_counter()
    chunk_timestamps: list[float] = []
    chunk_sizes: list[int] = []
    usage: dict[str, int] | None = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                chunk_timestamps.append(time.perf_counter())
                chunk_sizes.append(len(chunk))
                usage = extract_usage_from_sse(chunk) or usage
        ended = time.perf_counter()
        return build_success_result(
            item_id=item.id,
            started=started,
            ended=ended,
            chunk_timestamps=chunk_timestamps,
            chunk_sizes=chunk_sizes,
            usage=usage,
        )
    except Exception as exc:
        ended = time.perf_counter()
        return RequestResult(
            id=item.id,
            ok=False,
            latency_seconds=ended - started,
            ttft_seconds=None,
            chunk_count=len(chunk_sizes),
            first_chunk_bytes=chunk_sizes[0] if chunk_sizes else 0,
            output_bytes=sum(chunk_sizes),
            inter_chunk_gap_avg_seconds=None,
            inter_chunk_gap_p95_seconds=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_second=None,
            tpot_seconds=None,
            error=str(exc),
        )


def summarize_results(results: list[RequestResult], total_seconds: float) -> dict:
    successful = [result for result in results if result.ok]
    measured_tokens = [result for result in successful if result.completion_tokens is not None]
    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "measured_token_requests": len(measured_tokens),
        "total_seconds": total_seconds,
        "requests_per_second": len(results) / total_seconds if total_seconds > 0 else 0.0,
        "latency_seconds": summarize_latencies([result.latency_seconds for result in successful]),
        "ttft_seconds": summarize_optional_numbers([result.ttft_seconds for result in successful]),
        "output_bytes": summarize_latencies([float(result.output_bytes) for result in successful]),
        "chunk_count": summarize_latencies([float(result.chunk_count) for result in successful]),
        "inter_chunk_gap_avg_seconds": summarize_optional_numbers(
            [result.inter_chunk_gap_avg_seconds for result in successful]
        ),
        "inter_chunk_gap_p95_seconds": summarize_optional_numbers(
            [result.inter_chunk_gap_p95_seconds for result in successful]
        ),
        "completion_tokens": summarize_optional_numbers(
            [float(result.completion_tokens) if result.completion_tokens is not None else None for result in successful]
        ),
        "tokens_per_second": summarize_optional_numbers(
            [result.tokens_per_second for result in successful]
        ),
        "tpot_seconds": summarize_optional_numbers([result.tpot_seconds for result in successful]),
    }


async def run_benchmark(
    *,
    base_url: str,
    api_key: str,
    workload: list[WorkloadItem],
    concurrency: int,
    request_count: int,
    timeout_seconds: int,
) -> dict:
    selected = [workload[index % len(workload)] for index in range(request_count)]
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:

        async def guarded(item: WorkloadItem) -> RequestResult:
            async with semaphore:
                return await run_one(
                    client=client,
                    base_url=base_url,
                    api_key=api_key,
                    item=item,
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(item) for item in selected))
        total_seconds = time.perf_counter() - started

    return {
        "summary": summarize_results(results, total_seconds=total_seconds),
        "results": [asdict(result) for result in results],
    }


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 2: Run runner tests**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m pytest tests/benchmark/test_runner.py -q
```

Expected: PASS with `1 passed`.

---

### Task 5: Update streaming-only CLI

**Files:**
- Modify: `benchmark/harness/cli.py`

- [ ] **Step 1: Replace `benchmark/harness/cli.py`**

```python
import asyncio
from pathlib import Path
from typing import Annotated

import typer

from benchmark.harness.runner import run_benchmark, write_result
from benchmark.harness.workload import load_workload

app = typer.Typer()

WorkloadOption = Annotated[Path, typer.Option()]
OutputOption = Annotated[Path, typer.Option()]


@app.command()
def main(
    base_url: Annotated[str, typer.Option()] = "http://localhost:8080",
    api_key: Annotated[str, typer.Option()] = "change-me",
    workload: WorkloadOption = Path("benchmark/workloads/knowledge_qa.jsonl"),
    concurrency: Annotated[int, typer.Option(min=1)] = 1,
    requests: Annotated[int, typer.Option(min=1)] = 3,
    output: OutputOption = Path("benchmark/results/smoke.json"),
    timeout_seconds: Annotated[int, typer.Option(min=1)] = 300,
) -> None:
    items = load_workload(workload)
    result = asyncio.run(
        run_benchmark(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            workload=items,
            concurrency=concurrency,
            request_count=requests,
            timeout_seconds=timeout_seconds,
        )
    )
    write_result(output, result)
    typer.echo(f"wrote streaming benchmark result to {output}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Run CLI syntax check**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m py_compile benchmark/harness/cli.py benchmark/harness/runner.py
```

Expected: command exits with code 0.

---

### Task 6: Update docs and learning notes

**Files:**
- Modify: `docs/mvp-report.md`
- Modify: `learning/issues-and-fixes.md`

- [ ] **Step 1: Update `docs/mvp-report.md` known limits and benchmark wording**

Replace the benchmark smoke test command block text with:

```markdown
Streaming benchmark smoke test:

```bash
python -m benchmark.harness.cli --base-url http://127.0.0.1:8080 --api-key change-me --workload benchmark/workloads/knowledge_qa.jsonl --concurrency 1 --requests 3 --output benchmark/results/smoke.json
```
```

Add these rows to the runtime validation table if absent:

```markdown
| Measured token requests | `0` when stream usage is not emitted |
| Chunk count P50 | recorded in `benchmark/results/smoke.json` |
| Inter-chunk gap P95 | recorded in `benchmark/results/smoke.json` |
```

Replace the token known-limit line with:

```markdown
- TPS and TPOT are reported only when the streaming server emits OpenAI-compatible `usage`; tokens are never estimated from bytes or characters.
```

- [ ] **Step 2: Append learning note to `learning/issues-and-fixes.md`**

Append:

```markdown
---

## 12. 流式 benchmark 不能把字节数当 token 数

### 现象

流式响应天然会持续返回 SSE chunk。每个 chunk 有字节长度，但不一定包含 token usage。

### 根因

字节数、字符数和 token 数不是同一个概念。中文、英文、标点和 tokenizer 规则都会影响 token 数。用 bytes/sec 冒充 tokens/sec 会让压测报告失真。

### 解决方案

benchmark 只在 OpenAI-compatible stream chunk 明确返回 `usage.completion_tokens` 时计算 TPS 和 TPOT。没有 usage 时，token 字段保持 `null`。

### 工程原则

性能指标宁可缺失，也不能伪造。可观测性系统的可信度比表格完整性更重要。
```

---

### Task 7: Run full validation

**Files:**
- Modify if needed: files touched by failing tests only.

- [ ] **Step 1: Run tests and lint**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m pytest -q && .venv/bin/ruff check gateway benchmark tests
```

Expected:

```text
All tests pass
All checks passed!
```

- [ ] **Step 2: Run benchmark CLI help smoke check**

Run:

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab && .venv/bin/python -m benchmark.harness.cli --help >/tmp/streaming-benchmark-help.txt
```

Expected: command exits with code 0.

---

## Self-Review

### Spec coverage

Covered:

- Streaming-only benchmark path.
- Per-request chunk metrics.
- Token extraction only from real SSE `usage` data.
- Summary fields for chunk, token, TPS, and TPOT.
- Error handling for malformed SSE chunks without failing the whole request.
- Tests and documentation updates.

Not covered and intentionally deferred:

- Non-streaming token metrics.
- Local tokenizer-based token estimation.
- Prometheus/Grafana dashboards.
- Parameter sweep automation.

### Placeholder scan

No `TBD`, `TODO`, placeholder implementation steps, or vague “add handling” requirements remain.

### Type consistency

`RequestResult` fields in Task 3 exactly match the dataclass in Task 4. `run_benchmark()` no longer accepts `stream`, matching the CLI in Task 5. `extract_usage_from_sse()` returns the usage shape consumed by `build_success_result()`.
