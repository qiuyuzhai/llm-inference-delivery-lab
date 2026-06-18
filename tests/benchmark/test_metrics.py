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
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["avg"] == 2.0


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


def test_extract_usage_from_sse_reads_usage_from_multiline_chunk():
    chunk = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        b'data: {"usage":{"prompt_tokens":5,"completion_tokens":11,"total_tokens":16}}\n\n'
    )

    assert extract_usage_from_sse(chunk) == {
        "prompt_tokens": 5,
        "completion_tokens": 11,
        "total_tokens": 16,
    }


def test_summarize_optional_numbers_ignores_none_values():
    summary = summarize_optional_numbers([None, 1.0, 3.0])

    assert summary["count"] == 2
    assert summary["p50"] == 1.0
    assert summary["p95"] == 3.0
