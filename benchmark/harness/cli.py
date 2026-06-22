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
    model_name: Annotated[str | None, typer.Option()] = None,
    quantization: Annotated[str | None, typer.Option()] = None,
    dtype: Annotated[str, typer.Option()] = "auto",
    max_model_len: Annotated[int | None, typer.Option()] = None,
    label: Annotated[str | None, typer.Option()] = None,
) -> None:
    metadata = {
        "label": label,
        "model_name": model_name,
        "quantization": quantization,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "concurrency": concurrency,
        "requests": requests,
        "base_url": base_url.rstrip("/"),
    }
    items = load_workload(workload)
    result = asyncio.run(
        run_benchmark(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            workload=items,
            concurrency=concurrency,
            request_count=requests,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )
    )
    write_result(output, result)
    typer.echo(f"wrote streaming benchmark result to {output}")


if __name__ == "__main__":
    app()
