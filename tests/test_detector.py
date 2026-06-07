"""Unit tests for the regression detector."""

import pytest
from mrds.config import RegressionConfig
from mrds.detector import detect_regressions, RegressionReport
from mrds.evaluator import EvalResult, CaseResult


def _make_result(scores: list[float], pass_rate_override: float | None = None) -> EvalResult:
    cases = [
        CaseResult(
            case_id=f"case-{i}",
            input=f"input {i}",
            expected_output="expected",
            actual_output="actual",
            metric_scores={"exact_match": s},
            weighted_score=s,
            passed=s >= 0.5,
            latency_ms=100.0,
        )
        for i, s in enumerate(scores)
    ]
    result = EvalResult(dataset_name="test", model_id="claude-test", case_results=cases)
    return result


def _make_baseline(scores: list[float]) -> dict:
    return {
        "mean_score": sum(scores) / len(scores),
        "pass_rate": sum(1 for s in scores if s >= 0.5) / len(scores),
        "cases": [
            {
                "case_id": f"case-{i}",
                "weighted_score": s,
                "metric_scores": {"exact_match": s},
            }
            for i, s in enumerate(scores)
        ],
    }


def test_no_regression_when_scores_equal():
    cfg = RegressionConfig(score_drop_threshold=0.05, pass_rate_min=0.80, fail_on_regression=True)
    baseline = _make_baseline([1.0, 1.0, 1.0, 1.0])
    current = _make_result([1.0, 1.0, 1.0, 1.0])
    report = detect_regressions(current, baseline, cfg)
    assert not report.is_regression


def test_regression_detected_score_drop():
    cfg = RegressionConfig(score_drop_threshold=0.05, pass_rate_min=0.80, fail_on_regression=True)
    baseline = _make_baseline([1.0, 1.0, 1.0, 1.0])
    current = _make_result([0.8, 0.8, 0.8, 0.8])  # 0.2 drop
    report = detect_regressions(current, baseline, cfg)
    assert report.is_regression
    assert report.score_delta == pytest.approx(-0.2)


def test_regression_detected_pass_rate_low():
    cfg = RegressionConfig(score_drop_threshold=0.05, pass_rate_min=0.80, fail_on_regression=True)
    # baseline high, current just below pass_rate_min
    baseline = _make_baseline([1.0, 1.0, 1.0, 1.0, 1.0])
    current = _make_result([1.0, 0.0, 0.0, 1.0, 0.0])  # pass rate 0.4
    report = detect_regressions(current, baseline, cfg)
    assert report.is_regression


def test_no_regression_small_drop():
    cfg = RegressionConfig(score_drop_threshold=0.05, pass_rate_min=0.80, fail_on_regression=True)
    baseline = _make_baseline([1.0, 1.0, 1.0, 1.0])
    current = _make_result([0.97, 0.98, 0.99, 0.96])  # tiny drop, within threshold
    report = detect_regressions(current, baseline, cfg)
    assert not report.is_regression


def test_regressed_cases_listed():
    cfg = RegressionConfig(score_drop_threshold=0.05, pass_rate_min=0.0, fail_on_regression=True)
    baseline = _make_baseline([1.0, 1.0, 1.0])
    current = _make_result([0.5, 1.0, 0.5])
    report = detect_regressions(current, baseline, cfg)
    assert len(report.regressed_cases) == 2
