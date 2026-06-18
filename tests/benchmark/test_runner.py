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
