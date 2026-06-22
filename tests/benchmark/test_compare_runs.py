"""compare_runs.py 最小单测。

构造两个假结果 dict，断言表格行数和关键值。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.compare.compare_runs import (
    _build_table,
    _fmt,
    _get_nested,
    _is_failed_sentinel,
    _load_run,
    _parse_sweep_dims,
)

# --------------------------------------------------------------------------- #
# _get_nested                                                                  #
# --------------------------------------------------------------------------- #


def test_get_nested_returns_value_for_existing_path():
    data = {"latency_seconds": {"p50": 1.23}}
    assert _get_nested(data, "latency_seconds.p50") == pytest.approx(1.23)


def test_get_nested_returns_none_for_missing_key():
    data = {"latency_seconds": {}}
    assert _get_nested(data, "latency_seconds.p50") is None


def test_get_nested_returns_none_for_non_dict_intermediate():
    data = {"a": None}
    assert _get_nested(data, "a.b") is None


def test_get_nested_handles_top_level_float():
    data = {"requests_per_second": 5.0}
    assert _get_nested(data, "requests_per_second") == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# _fmt                                                                         #
# --------------------------------------------------------------------------- #


def test_fmt_returns_dash_for_none():
    assert _fmt(None) == "-"


def test_fmt_formats_small_float_with_six_decimals():
    result = _fmt(0.001234)
    assert result == "0.001234"


def test_fmt_formats_large_float_with_two_decimals():
    result = _fmt(123.456)
    assert result == "123.46"


# --------------------------------------------------------------------------- #
# _load_run                                                                    #
# --------------------------------------------------------------------------- #


def test_load_run_uses_stem_as_label_when_config_absent(tmp_path: Path):
    data = {"summary": {"requests_per_second": 2.0}}
    file = tmp_path / "my_run.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    run = _load_run(file)

    assert run["config"]["label"] == "my_run"


def test_load_run_preserves_existing_label(tmp_path: Path):
    data = {
        "config": {"label": "fp16-baseline"},
        "summary": {"requests_per_second": 3.0},
    }
    file = tmp_path / "run.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    run = _load_run(file)

    assert run["config"]["label"] == "fp16-baseline"


def test_load_run_raises_on_missing_summary(tmp_path: Path):
    file = tmp_path / "bad.json"
    file.write_text(json.dumps({"config": {}}), encoding="utf-8")

    with pytest.raises(Exception, match="summary"):
        _load_run(file)


# --------------------------------------------------------------------------- #
# _build_table                                                                 #
# --------------------------------------------------------------------------- #

_RUN_A = {
    "config": {
        "label": "fp16-base",
        "model_name": "qwen2-7b",
        "quantization": None,
        "dtype": "float16",
    },
    "summary": {
        "latency_seconds": {"p50": 2.0, "p95": 3.0, "p99": 3.5},
        "tokens_per_second": {"p50": 60.0},
        "ttft_seconds": {"p50": 0.05},
        "tpot_seconds": {"p50": 0.016},
        "requests_per_second": 1.5,
    },
}

_RUN_B = {
    "config": {
        "label": "w4a16-awq",
        "model_name": "qwen2-7b",
        "quantization": "w4a16",
        "dtype": "float16",
    },
    "summary": {
        "latency_seconds": {"p50": 2.5, "p95": 3.8, "p99": 4.0},
        "tokens_per_second": {"p50": 55.0},
        "ttft_seconds": {"p50": 0.06},
        "tpot_seconds": {"p50": 0.018},
        "requests_per_second": 1.2,
    },
}

_RUN_NULL_METRICS = {
    "config": {
        "label": "int8-test",
        "model_name": "qwen2-7b",
        "quantization": "int8",
        "dtype": None,
    },
    "summary": {
        # ttft_seconds 和 tpot_seconds 故意缺失，模拟可选指标 null
        "latency_seconds": {"p50": 3.0, "p95": 4.5},
        "tokens_per_second": None,
        "requests_per_second": 0.9,
    },
}


def test_build_table_row_count_matches_run_count():
    table = _build_table([_RUN_A, _RUN_B])
    lines = [line for line in table.strip().split("\n") if line.strip()]
    # 表头 + 分隔行 + 2 个数据行 = 4 行
    assert len(lines) == 4


def test_build_table_contains_labels():
    table = _build_table([_RUN_A, _RUN_B])
    assert "fp16-base" in table
    assert "w4a16-awq" in table


def test_build_table_contains_quantization_when_present():
    table = _build_table([_RUN_A, _RUN_B])
    # quantization 列应出现（_RUN_B 有值）
    assert "w4a16" in table


def test_build_table_shows_dash_for_null_optional_metrics():
    table = _build_table([_RUN_NULL_METRICS])
    # ttft_seconds.p50 和 tpot_seconds.p50 缺失，应显示 "-"
    lines = [line for line in table.strip().split("\n") if line.strip()]
    data_row = lines[2]  # 第三行是数据行
    # 数据行中应含有 "-" 占位
    assert "-" in data_row


def test_build_table_does_not_crash_with_null_metrics():
    """核心：可选指标为 null 时不崩溃。"""
    # 不应抛出任何异常
    table = _build_table([_RUN_A, _RUN_B, _RUN_NULL_METRICS])
    assert len(table) > 0


def test_build_table_sorts_by_label():
    # _RUN_B label='w4a16-awq' > _RUN_A label='fp16-base'（字母序）
    table = _build_table([_RUN_B, _RUN_A])
    lines = [line for line in table.strip().split("\n") if line.strip()]
    # 数据行第一行应是 fp16-base（字母序小）
    assert "fp16-base" in lines[2]
    assert "w4a16-awq" in lines[3]


def test_build_table_single_run_produces_valid_markdown():
    table = _build_table([_RUN_A])
    lines = [line for line in table.strip().split("\n") if line.strip()]
    # 表头 + 分隔 + 1 数据行 = 3 行
    assert len(lines) == 3
    # 每行以 '|' 开头和结尾
    for line in lines:
        assert line.startswith("|")
        assert line.endswith("|")


# --------------------------------------------------------------------------- #
# _parse_sweep_dims                                                            #
# --------------------------------------------------------------------------- #


def test_parse_sweep_dims_extracts_all_three_fields():
    dims = _parse_sweep_dims("mml4096-gmu0.6-mns8-throughput")
    assert dims["max_model_len"] == 4096
    assert dims["gpu_memory_utilization"] == pytest.approx(0.6)
    assert dims["max_num_seqs"] == 8


def test_parse_sweep_dims_works_without_workload_suffix():
    dims = _parse_sweep_dims("mml8192-gmu0.75-mns16")
    assert dims["max_model_len"] == 8192
    assert dims["gpu_memory_utilization"] == pytest.approx(0.75)
    assert dims["max_num_seqs"] == 16


def test_parse_sweep_dims_returns_empty_for_non_sweep_label():
    assert _parse_sweep_dims("fp16-base") == {}
    assert _parse_sweep_dims("") == {}


def test_load_run_backfills_sweep_dims_from_label(tmp_path: Path):
    """SUCCESS 格 config 无 gpu_memory_utilization / max_num_seqs，
    _load_run 应从 label 解析并回填。"""
    data = {
        "config": {
            "label": "mml4096-gmu0.6-mns8-throughput",
            "max_model_len": 4096,
        },
        "summary": {"requests_per_second": 1.0},
    }
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    run = _load_run(path)

    assert run["config"]["gpu_memory_utilization"] == pytest.approx(0.6)
    assert run["config"]["max_num_seqs"] == 8


def test_load_run_does_not_overwrite_existing_sweep_dims(tmp_path: Path):
    """config 里已有值时不覆盖（FAILED sentinel 场景）。"""
    data = {
        "config": {
            "label": "mml4096-gmu0.6-mns8",
            "gpu_memory_utilization": 0.9,  # 故意不同，应保留
            "max_num_seqs": 32,
        },
        "summary": {"requests_per_second": 1.0},
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    run = _load_run(path)

    assert run["config"]["gpu_memory_utilization"] == pytest.approx(0.9)
    assert run["config"]["max_num_seqs"] == 32


# --------------------------------------------------------------------------- #
# _is_failed_sentinel                                                          #
# --------------------------------------------------------------------------- #


def test_is_failed_sentinel_detects_FAILED_json_filename(tmp_path: Path):
    path = tmp_path / "FAILED.json"
    assert _is_failed_sentinel(path, {}) is True


def test_is_failed_sentinel_detects_failed_true_field(tmp_path: Path):
    path = tmp_path / "some_run.json"
    assert _is_failed_sentinel(path, {"failed": True}) is True


def test_is_failed_sentinel_returns_false_for_normal_run(tmp_path: Path):
    path = tmp_path / "normal.json"
    assert _is_failed_sentinel(path, {"summary": {}}) is False


def test_is_failed_sentinel_case_sensitive_filename(tmp_path: Path):
    # 小写 failed.json 不应被识别为 sentinel（约定大写）
    path = tmp_path / "failed.json"
    assert _is_failed_sentinel(path, {}) is False


# --------------------------------------------------------------------------- #
# _load_run — FAILED sentinel                                                  #
# --------------------------------------------------------------------------- #


def test_load_run_accepts_FAILED_json_without_summary(tmp_path: Path):
    path = tmp_path / "FAILED.json"
    path.write_text(json.dumps({"config": {"label": "oom-combo"}}), encoding="utf-8")

    run = _load_run(path)

    assert run["_failed"] is True
    assert run["config"]["label"] == "oom-combo"


def test_load_run_accepts_failed_true_without_summary(tmp_path: Path):
    path = tmp_path / "bad_combo.json"
    path.write_text(
        json.dumps({"failed": True, "config": {"label": "oom-run"}}),
        encoding="utf-8",
    )

    run = _load_run(path)

    assert run["_failed"] is True


def test_load_run_uses_stem_as_label_for_FAILED_json(tmp_path: Path):
    path = tmp_path / "FAILED.json"
    path.write_text(json.dumps({}), encoding="utf-8")

    run = _load_run(path)

    assert run["config"]["label"] == "FAILED"
    assert run["_failed"] is True


# --------------------------------------------------------------------------- #
# _build_table — FAILED sentinel 渲染                                          #
# --------------------------------------------------------------------------- #

_RUN_FAILED = {
    "config": {"label": "oom-combo", "quantization": "awq_marlin"},
    "summary": {},
    "_failed": True,
}


def test_build_table_shows_FAILED_in_metric_cells():
    table = _build_table([_RUN_FAILED])
    lines = [line for line in table.strip().split("\n") if line.strip()]
    data_row = lines[2]
    assert "FAILED" in data_row


def test_build_table_does_not_crash_with_failed_run():
    """FAILED run 与正常 run 混合不崩溃。"""
    table = _build_table([_RUN_A, _RUN_FAILED])
    assert len(table) > 0
    assert "FAILED" in table
    assert "fp16-base" in table


def test_build_table_failed_run_config_cells_normal():
    """FAILED run 的 config 列（label/quantization）正常显示。"""
    table = _build_table([_RUN_FAILED])
    assert "oom-combo" in table


# --------------------------------------------------------------------------- #
# sweep FAILED sentinel — 顶层字段提升到 config                               #
# --------------------------------------------------------------------------- #


def test_load_run_promotes_sweep_top_level_fields_to_config(tmp_path: Path):
    """sweep-params.sh 把 gpu_memory_utilization / max_num_seqs 写在顶层；
    _load_run 应将其提升到 config 以供 _build_table 渲染。
    """
    path = tmp_path / "FAILED.json"
    payload = {
        "config": {"label": "mml8192-gmu0.9-mns16"},
        "failed": True,
        "reason": "vLLM startup failed (OOM or timeout)",
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 16,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    run = _load_run(path)

    assert run["_failed"] is True
    assert run["config"]["gpu_memory_utilization"] == pytest.approx(0.9)
    assert run["config"]["max_num_seqs"] == 16


def test_build_table_renders_sweep_fields_from_failed_sentinel(tmp_path: Path):
    """提升后的 sweep 字段在表格中正常显示（列动态出现）。"""
    path = tmp_path / "FAILED.json"
    payload = {
        "config": {"label": "mml8192-gmu0.9-mns16"},
        "failed": True,
        "reason": "OOM",
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 16,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    run = _load_run(path)

    table = _build_table([run])

    assert "gpu_memory_utilization" in table
    assert "0.9" in table
    assert "FAILED" in table
