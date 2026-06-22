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
    gaps = [
        current - previous
        for previous, current in zip(chunk_timestamps, chunk_timestamps[1:], strict=False)
    ]
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
        "stream_options": {"include_usage": True},
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


def build_config(metadata: dict) -> dict:
    """Assemble the config section from run metadata; strips api_key."""
    return {
        "label": metadata.get("label"),
        "model_name": metadata.get("model_name"),
        "quantization": metadata.get("quantization"),
        "dtype": metadata.get("dtype"),
        "max_model_len": metadata.get("max_model_len"),
        "concurrency": metadata.get("concurrency"),
        "requests": metadata.get("requests"),
        "base_url": metadata.get("base_url"),
    }


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
            [
                float(result.completion_tokens) if result.completion_tokens is not None else None
                for result in successful
            ]
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
    metadata: dict | None = None,
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

    config = build_config(metadata or {})
    return {
        "config": config,
        "summary": summarize_results(results, total_seconds=total_seconds),
        "results": [asdict(result) for result in results],
    }


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
