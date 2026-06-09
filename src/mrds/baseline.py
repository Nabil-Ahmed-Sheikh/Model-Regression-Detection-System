"""Baseline management — save, load, and compare eval results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .evaluator import EvalResult, CaseResult


def _result_to_dict(result: EvalResult) -> dict:
    return {
        "dataset_name": result.dataset_name,
        "model_id": result.model_id,
        "timestamp": result.timestamp,
        "mean_score": result.mean_score,
        "pass_rate": result.pass_rate,
        "mean_latency_ms": result.mean_latency_ms,
        "cases": [
            {
                "case_id": r.case_id,
                "weighted_score": r.weighted_score,
                "metric_scores": r.metric_scores,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in result.case_results
        ],
    }


def _baseline_path(baseline_dir: str, dataset_name: str, model_id: str) -> Path:
    safe_model = model_id.replace("/", "_").replace(":", "_")
    return Path(baseline_dir) / f"{dataset_name}__{safe_model}.json"


def save_baseline(result: EvalResult, baseline_dir: str) -> Path:
    path = _baseline_path(baseline_dir, result.dataset_name, result.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_result_to_dict(result), f, indent=2)
    return path


def load_baseline(baseline_dir: str, dataset_name: str, model_id: str) -> dict | None:
    path = _baseline_path(baseline_dir, dataset_name, model_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def baseline_exists(baseline_dir: str, dataset_name: str, model_id: str) -> bool:
    return _baseline_path(baseline_dir, dataset_name, model_id).exists()
