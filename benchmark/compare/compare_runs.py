"""多 run 对比汇总脚本。

读取多个 benchmark 结果 JSON，输出一张 Markdown 对比表到 stdout 和 --output 指定的 .md 文件。
每个 JSON 文件须包含顶层 config 段和 summary 段。
可选指标缺失时显示 "-" 而非崩溃。

FAILED sentinel 识别规则（满足任一即标记为失败行）：
  1. 文件名为 FAILED.json（大写，任意目录）
  2. JSON 顶层含 "failed": true 字段
失败行在所有指标列显示 "FAILED"，config 列正常填写（若有）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="汇总多个 benchmark run 为 Markdown 对比表")

# --------------------------------------------------------------------------- #
# 常量                                                                         #
# --------------------------------------------------------------------------- #

# config 字段：必填 vs 可选（字段清单以 harness-eng #2 输出为准）
# label 缺失时降级为文件名 stem，其余均为可选——存在于任意 run 时才显示该列
_CONFIG_REQUIRED = ("label",)
_CONFIG_OPTIONAL = (
    "model_name",
    "quantization",           # "awq" / "gptq" / null
    "dtype",                  # 默认 "auto"
    "max_model_len",
    "gpu_memory_utilization", # sweep 维度
    "max_num_seqs",           # sweep 维度
    "concurrency",
    "requests",
    "base_url",
)

# 要展示的 summary 指标及其显示名
_METRICS: list[tuple[str, str, str | None]] = [
    # (summary_path_dotted, 列标题, 单位前缀)
    ("latency_seconds.p50", "Latency p50 (s)", None),
    ("latency_seconds.p95", "Latency p95 (s)", None),
    ("latency_seconds.p99", "Latency p99 (s)", None),
    ("tokens_per_second.p50", "TPS p50 (tok/s)", None),
    ("ttft_seconds.p50", "TTFT p50 (s)", None),
    ("tpot_seconds.p50", "TPOT p50 (s)", None),
    ("requests_per_second", "RPS", None),
]


# --------------------------------------------------------------------------- #
# 工具函数                                                                      #
# --------------------------------------------------------------------------- #


def _get_nested(data: dict, path: str) -> float | None:
    """按点号路径取嵌套字段，任意层缺失均返回 None。"""
    parts = path.split(".")
    node: dict | float | None = data
    for part in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    if isinstance(node, (int, float)):
        return float(node)
    return None


def _fmt(value: float | None) -> str:
    """格式化数值；None 显示 '-'。"""
    if value is None:
        return "-"
    # 小数保留 4 位有效数字
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


# sweep label 格式：mml<N>-gmu<G>-mns<M>[-<workload>]
_SWEEP_LABEL_RE = re.compile(
    r"mml(?P<mml>\d+)-gmu(?P<gmu>[0-9.]+)-mns(?P<mns>\d+)"
)


def _parse_sweep_dims(label: str) -> dict[str, float | int | None]:
    """从 sweep label 提取 max_model_len / gpu_memory_utilization / max_num_seqs。

    label 样例：mml4096-gmu0.6-mns8-throughput
    无法匹配时返回空 dict（非 sweep run，静默忽略）。
    """
    m = _SWEEP_LABEL_RE.search(label)
    if not m:
        return {}
    return {
        "max_model_len": int(m.group("mml")),
        "gpu_memory_utilization": float(m.group("gmu")),
        "max_num_seqs": int(m.group("mns")),
    }


def _is_failed_sentinel(path: Path, raw: dict) -> bool:
    """判断某个 JSON 是否为 FAILED sentinel。

    满足任一条件即为失败：
    - 文件名为 FAILED.json（大写）
    - JSON 顶层 "failed" 字段为真值
    """
    if path.name == "FAILED.json":
        return True
    return bool(raw.get("failed"))


def _load_run(path: Path) -> dict:
    """加载单个结果 JSON，校验必要字段存在。

    失败 sentinel 返回含 _failed=True 标记的 dict，不要求 summary 字段。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    # config 不存在时给空 dict，label 缺失时用文件名替代
    if "config" not in raw:
        raw["config"] = {}
    if not raw["config"].get("label"):
        raw["config"]["label"] = path.stem

    if _is_failed_sentinel(path, raw):
        raw["_failed"] = True
        # sweep-params.sh 把 gpu_memory_utilization / max_num_seqs 写在顶层而非 config；
        # 提升到 config 以便 _build_table 正常渲染这两列。
        for top_key in ("gpu_memory_utilization", "max_num_seqs"):
            if top_key in raw and raw["config"].get(top_key) is None:
                raw["config"][top_key] = raw[top_key]
        # 失败行不需要 summary
        if "summary" not in raw:
            raw["summary"] = {}
        return raw

    if "summary" not in raw:
        raise typer.BadParameter(f"{path}: 缺少顶层 'summary' 字段")

    # sweep SUCCESS 格：config 里没有 gpu_memory_utilization / max_num_seqs，
    # 从 label（mml<N>-gmu<G>-mns<M>[-workload]）解析并回填，使热力表列可用。
    dims = _parse_sweep_dims(raw["config"].get("label", ""))
    for key, val in dims.items():
        if raw["config"].get(key) is None:
            raw["config"][key] = val

    return raw


def _build_table(runs: list[dict]) -> str:
    """构造 Markdown 对比表字符串。"""
    # 配置列：必填 + 存在于任意 run 的可选字段
    present_optional = [
        col
        for col in _CONFIG_OPTIONAL
        if any(run["config"].get(col) is not None for run in runs)
    ]
    config_cols = list(_CONFIG_REQUIRED) + present_optional

    # 表头
    header_labels = config_cols + [m[1] for m in _METRICS]
    header_row = "| " + " | ".join(header_labels) + " |"
    sep_row = "| " + " | ".join(["---"] * len(header_labels)) + " |"

    # 数据行：按 label 排序
    sorted_runs = sorted(runs, key=lambda r: r["config"].get("label", ""))
    data_rows: list[str] = []
    for run in sorted_runs:
        cfg = run["config"]
        summary = run.get("summary", {})
        failed = run.get("_failed", False)
        cells: list[str] = []
        for col in config_cols:
            val = cfg.get(col)
            cells.append(str(val) if val is not None else "-")
        for metric_path, _, _ in _METRICS:
            cells.append("FAILED" if failed else _fmt(_get_nested(summary, metric_path)))
        data_rows.append("| " + " | ".join(cells) + " |")

    lines = [header_row, sep_row] + data_rows
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


@app.command()
def main(
    results: Annotated[
        list[Path],
        typer.Argument(help="一个或多个 benchmark 结果 JSON 文件路径"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="输出 Markdown 文件路径（可选）"),
    ] = None,
) -> None:
    """汇总多个 benchmark run 为 Markdown 对比表。"""
    if not results:
        typer.echo("错误：至少需要提供一个结果 JSON 文件。", err=True)
        raise typer.Exit(1)

    runs: list[dict] = []
    for path in results:
        if not path.exists():
            typer.echo(f"错误：文件不存在 {path}", err=True)
            raise typer.Exit(1)
        runs.append(_load_run(path))

    table = _build_table(runs)

    typer.echo(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        typer.echo(f"已写入 {output}", err=True)


if __name__ == "__main__":
    app()
