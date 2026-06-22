import pytest

from benchmark.harness.runner import RequestResult, build_config, run_benchmark, summarize_results
from benchmark.harness.workload import WorkloadItem


def _make_result(id: str, ok: bool, **kwargs) -> RequestResult:
    defaults = dict(
        latency_seconds=1.0,
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
    )
    defaults.update(kwargs)
    return RequestResult(id=id, ok=ok, **defaults)


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


# --- build_config tests ---


def test_build_config_all_fields():
    metadata = {
        "label": "7B-awq",
        "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "quantization": "awq",
        "dtype": "float16",
        "max_model_len": 4096,
        "concurrency": 4,
        "requests": 100,
        "base_url": "http://localhost:8080",
    }
    config = build_config(metadata)
    assert config["label"] == "7B-awq"
    assert config["model_name"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert config["quantization"] == "awq"
    assert config["dtype"] == "float16"
    assert config["max_model_len"] == 4096
    assert config["concurrency"] == 4
    assert config["requests"] == 100
    assert config["base_url"] == "http://localhost:8080"


def test_build_config_no_api_key_leakage():
    metadata = {
        "api_key": "super-secret",
        "label": "test",
        "model_name": "model-x",
        "quantization": None,
        "dtype": "auto",
        "max_model_len": None,
        "concurrency": 1,
        "requests": 1,
        "base_url": "http://localhost:8080",
    }
    config = build_config(metadata)
    assert "api_key" not in config


def test_build_config_optional_fields_none():
    config = build_config({})
    assert config["label"] is None
    assert config["model_name"] is None
    assert config["quantization"] is None
    assert config["max_model_len"] is None


# --- run_benchmark config integration test ---


@pytest.mark.asyncio
async def test_run_benchmark_includes_config_section(monkeypatch):
    import benchmark.harness.runner as runner_mod

    workload = [WorkloadItem(id="q1", prompt="hello", max_tokens=10)]
    stub_result = _make_result("q1", ok=True, latency_seconds=0.5)

    async def fake_run_one(**_kwargs):
        return stub_result

    monkeypatch.setattr(runner_mod, "run_one", fake_run_one)

    metadata = {
        "label": "smoke",
        "model_name": "test-model",
        "quantization": "awq",
        "dtype": "float16",
        "max_model_len": 512,
        "concurrency": 1,
        "requests": 1,
        "base_url": "http://localhost:8080",
    }
    result = await run_benchmark(
        base_url="http://localhost:8080",
        api_key="test-key",
        workload=workload,
        concurrency=1,
        request_count=1,
        timeout_seconds=10,
        metadata=metadata,
    )

    assert "config" in result
    assert "summary" in result
    assert "results" in result
    cfg = result["config"]
    assert cfg["label"] == "smoke"
    assert cfg["model_name"] == "test-model"
    assert cfg["quantization"] == "awq"
    assert "api_key" not in cfg
