"""Golden dataset loading and management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestCase:
    id: str
    input: Any                          # raw input — string or dict
    expected_output: str | None = None  # ground-truth text (optional)
    expected_labels: list[str] = field(default_factory=list)  # classification labels
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenDataset:
    name: str
    cases: list[TestCase]

    def __len__(self) -> int:
        return len(self.cases)

    def filter_by_tag(self, tag: str) -> "GoldenDataset":
        return GoldenDataset(
            name=self.name,
            cases=[c for c in self.cases if tag in c.tags],
        )


def load_dataset(path: str | Path) -> GoldenDataset:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    name = raw.get("name", path.stem)
    cases = []
    for item in raw.get("cases", []):
        cases.append(
            TestCase(
                id=item["id"],
                input=item["input"],
                expected_output=item.get("expected_output"),
                expected_labels=item.get("expected_labels", []),
                tags=item.get("tags", []),
                metadata=item.get("metadata", {}),
            )
        )
    return GoldenDataset(name=name, cases=cases)


def save_dataset(dataset: GoldenDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": dataset.name,
        "cases": [
            {
                "id": c.id,
                "input": c.input,
                "expected_output": c.expected_output,
                "expected_labels": c.expected_labels,
                "tags": c.tags,
                "metadata": c.metadata,
            }
            for c in dataset.cases
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
