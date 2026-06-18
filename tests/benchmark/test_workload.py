from pathlib import Path

from benchmark.harness.workload import WorkloadItem, load_workload


def test_load_workload_reads_jsonl(tmp_path: Path):
    path = tmp_path / "workload.jsonl"
    path.write_text('{"id":"q1","prompt":"hello","max_tokens":32}\n', encoding="utf-8")

    items = load_workload(path)

    assert items == [WorkloadItem(id="q1", prompt="hello", max_tokens=32)]


def test_load_workload_uses_default_max_tokens(tmp_path: Path):
    path = tmp_path / "workload.jsonl"
    path.write_text('{"id":"q1","prompt":"hello"}\n', encoding="utf-8")

    items = load_workload(path)

    assert items[0].max_tokens == 512
