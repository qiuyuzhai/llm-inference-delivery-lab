import json
import math
from typing import Any


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil((percent / 100) * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "avg": average(values) or 0.0,
    }


def summarize_optional_numbers(values: list[float | None]) -> dict[str, float | int]:
    measured = [value for value in values if value is not None]
    return summarize_latencies(measured)


def extract_usage_from_sse(chunk: bytes) -> dict[str, int] | None:
    text = chunk.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed: Any = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        if not isinstance(usage, dict):
            continue
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        has_complete_usage = all(
            isinstance(value, int) for value in (prompt_tokens, completion_tokens, total_tokens)
        )
        if has_complete_usage:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    return None
