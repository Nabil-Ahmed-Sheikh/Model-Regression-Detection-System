"""Unit tests for dataset loading."""

import json
import tempfile
from pathlib import Path

from mrds.dataset import load_dataset, save_dataset, GoldenDataset, TestCase


def _write_dataset(data: dict, tmp_dir: Path) -> Path:
    p = tmp_dir / "test.json"
    p.write_text(json.dumps(data))
    return p


def test_load_basic_dataset(tmp_path):
    data = {
        "name": "my-dataset",
        "cases": [
            {"id": "c1", "input": "hello", "expected_output": "hi", "expected_labels": [], "tags": [], "metadata": {}}
        ],
    }
    p = _write_dataset(data, tmp_path)
    ds = load_dataset(p)
    assert ds.name == "my-dataset"
    assert len(ds) == 1
    assert ds.cases[0].id == "c1"
    assert ds.cases[0].input == "hello"


def test_load_dataset_defaults(tmp_path):
    data = {"cases": [{"id": "c1", "input": "x"}]}
    p = _write_dataset(data, tmp_path)
    ds = load_dataset(p)
    assert ds.name == "test"  # stems from filename
    assert ds.cases[0].expected_output is None
    assert ds.cases[0].tags == []


def test_save_and_reload(tmp_path):
    ds = GoldenDataset(
        name="roundtrip",
        cases=[TestCase(id="t1", input="foo", expected_output="bar", expected_labels=["bar"], tags=["x"])],
    )
    out = tmp_path / "out.json"
    save_dataset(ds, out)
    reloaded = load_dataset(out)
    assert reloaded.name == "roundtrip"
    assert reloaded.cases[0].expected_output == "bar"
    assert reloaded.cases[0].tags == ["x"]


def test_filter_by_tag():
    ds = GoldenDataset(
        name="t",
        cases=[
            TestCase(id="a", input="x", tags=["foo"]),
            TestCase(id="b", input="y", tags=["bar"]),
            TestCase(id="c", input="z", tags=["foo", "bar"]),
        ],
    )
    filtered = ds.filter_by_tag("foo")
    assert len(filtered) == 2
    assert all("foo" in c.tags for c in filtered.cases)
