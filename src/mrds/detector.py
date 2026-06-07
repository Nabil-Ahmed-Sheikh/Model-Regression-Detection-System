"""Regression detection logic — compare current run vs. baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RegressionConfig
from .evaluator import EvalResult


@dataclass
class CaseRegression:
    case_id: str
    baseline_score: float
    current_score: float
    delta: float
    metric_deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class RegressionReport:
    dataset_name: str
    model_id: str
    baseline_mean_score: float
    current_mean_score: float
    score_delta: float
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    regressed_cases: list[CaseRegression]
    is_regression: bool
    reasons: list[str]


def detect_regressions(
    current: EvalResult,
    baseline: dict,
    cfg: RegressionConfig,
) -> RegressionReport:
    baseline_mean = baseline.get("mean_score", 0.0)
    current_mean = current.mean_score
    score_delta = current_mean - baseline_mean

    baseline_pass_rate = baseline.get("pass_rate", 0.0)
    current_pass_rate = current.pass_rate

    # Index baseline cases
    baseline_by_id = {c["case_id"]: c for c in baseline.get("cases", [])}

    regressed_cases: list[CaseRegression] = []
    for case in current.case_results:
        b = baseline_by_id.get(case.case_id)
        if b is None:
            continue
        delta = case.weighted_score - b["weighted_score"]
        metric_deltas = {
            m: case.metric_scores.get(m, 0.0) - b["metric_scores"].get(m, 0.0)
            for m in case.metric_scores
        }
        if delta < -cfg.score_drop_threshold:
            regressed_cases.append(
                CaseRegression(
                    case_id=case.case_id,
                    baseline_score=b["weighted_score"],
                    current_score=case.weighted_score,
                    delta=delta,
                    metric_deltas=metric_deltas,
                )
            )

    reasons: list[str] = []
    is_regression = False

    if score_delta < -cfg.score_drop_threshold:
        is_regression = True
        reasons.append(
            f"Mean score dropped {abs(score_delta):.3f} (threshold {cfg.score_drop_threshold}): "
            f"{baseline_mean:.3f} → {current_mean:.3f}"
        )

    if current_pass_rate < cfg.pass_rate_min:
        is_regression = True
        reasons.append(
            f"Pass rate {current_pass_rate:.1%} is below minimum {cfg.pass_rate_min:.1%}"
        )

    if regressed_cases:
        is_regression = True
        reasons.append(f"{len(regressed_cases)} individual case(s) regressed beyond threshold")

    return RegressionReport(
        dataset_name=current.dataset_name,
        model_id=current.model_id,
        baseline_mean_score=baseline_mean,
        current_mean_score=current_mean,
        score_delta=score_delta,
        baseline_pass_rate=baseline_pass_rate,
        current_pass_rate=current_pass_rate,
        pass_rate_delta=current_pass_rate - baseline_pass_rate,
        regressed_cases=regressed_cases,
        is_regression=is_regression,
        reasons=reasons,
    )
