import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadItem:
    id: str
    prompt: str
    max_tokens: int = 512


def load_workload(path: Path) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt = raw.get("prompt")
            if not prompt:
                raise ValueError(f"line {line_number}: prompt is required")
            items.append(
                WorkloadItem(
                    id=str(raw.get("id", line_number)),
                    prompt=prompt,
                    max_tokens=int(raw.get("max_tokens", 512)),
                )
            )
    if not items:
        raise ValueError("workload must contain at least one item")
    return items
