"""Run model inference against a golden dataset and score each case."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from jinja2 import Template

from .config import Config, MetricConfig
from .dataset import GoldenDataset, TestCase
from .metrics import compute_metric


@dataclass
class CaseResult:
    case_id: str
    input: Any
    expected_output: str | None
    actual_output: str
    metric_scores: dict[str, float]
    weighted_score: float
    passed: bool
    latency_ms: float
    error: str | None = None


@dataclass
class EvalResult:
    dataset_name: str
    model_id: str
    case_results: list[CaseResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def mean_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.weighted_score for r in self.case_results) / len(self.case_results)

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.passed) / len(self.case_results)

    @property
    def mean_latency_ms(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.latency_ms for r in self.case_results) / len(self.case_results)


def _build_prompt(template_str: str, case: TestCase) -> str:
    tpl = Template(template_str)
    return tpl.render(input=case.input, metadata=case.metadata)


def _call_model(client: anthropic.Anthropic, cfg: "Config", prompt: str) -> tuple[str, float]:
    """Call the model and return (output_text, latency_ms)."""
    kwargs: dict[str, Any] = {
        "model": cfg.model.id,
        "max_tokens": cfg.model.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg.model.temperature is not None:
        kwargs["temperature"] = cfg.model.temperature
    if cfg.model.system_prompt:
        kwargs["system"] = cfg.model.system_prompt
    kwargs.update(cfg.model.extra)

    t0 = time.perf_counter()
    response = client.messages.create(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    text = response.content[0].text if response.content else ""
    return text, latency_ms


def _score_case(
    case: TestCase,
    actual: str,
    metrics: list[MetricConfig],
) -> tuple[dict[str, float], float, bool]:
    scores: dict[str, float] = {}
    total_weight = sum(m.weight for m in metrics)

    for mc in metrics:
        score = compute_metric(
            name=mc.name,
            actual=actual,
            expected=case.expected_output or "",
            expected_labels=case.expected_labels,
            params=mc.params,
        )
        scores[mc.name] = score

    weighted = (
        sum(scores[m.name] * m.weight for m in metrics) / total_weight
        if total_weight > 0
        else 0.0
    )

    # A case passes if every metric with a per-metric threshold meets it
    passed = all(
        scores[m.name] >= m.threshold
        for m in metrics
        if m.threshold > 0
    )

    return scores, weighted, passed


def run_evaluation(cfg: Config, dataset: GoldenDataset, verbose: bool = False) -> EvalResult:
    """Run all test cases against the model and return aggregated results."""
    from datetime import datetime, timezone

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    result = EvalResult(
        dataset_name=dataset.name,
        model_id=cfg.model.id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    for case in dataset.cases:
        error: str | None = None
        actual = ""
        latency_ms = 0.0

        try:
            prompt = _build_prompt(cfg.prompt_template, case)
            actual, latency_ms = _call_model(client, cfg, prompt)
        except Exception as exc:
            error = str(exc)
            actual = ""

        metric_scores, weighted, passed = _score_case(case, actual, cfg.metrics)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {case.id}  score={weighted:.3f}  latency={latency_ms:.0f}ms")

        result.case_results.append(
            CaseResult(
                case_id=case.id,
                input=case.input,
                expected_output=case.expected_output,
                actual_output=actual,
                metric_scores=metric_scores,
                weighted_score=weighted,
                passed=passed,
                latency_ms=latency_ms,
                error=error,
            )
        )

    return result
