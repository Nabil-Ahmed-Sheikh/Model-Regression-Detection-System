"""Rich terminal and JSON reporting."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from .detector import RegressionReport
from .evaluator import EvalResult

console = Console()


def print_eval_summary(result: EvalResult) -> None:
    table = Table(title=f"Eval Results — {result.model_id} / {result.dataset_name}", box=box.ROUNDED)
    table.add_column("Case ID", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    for metric in (result.case_results[0].metric_scores if result.case_results else {}):
        table.add_column(metric, justify="right")
    table.add_column("Pass", justify="center")
    table.add_column("Latency (ms)", justify="right")

    for r in result.case_results:
        row = [
            r.case_id,
            f"{r.weighted_score:.3f}",
            *[f"{r.metric_scores.get(m, 0):.3f}" for m in (result.case_results[0].metric_scores if result.case_results else {})],
            ":white_check_mark:" if r.passed else ":x:",
            f"{r.latency_ms:.0f}",
        ]
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]Mean score:[/bold] {result.mean_score:.3f}  "
        f"[bold]Pass rate:[/bold] {result.pass_rate:.1%}  "
        f"[bold]Avg latency:[/bold] {result.mean_latency_ms:.0f} ms\n"
    )


def print_regression_report(report: RegressionReport) -> None:
    if report.is_regression:
        console.print(f"\n[bold red]:rotating_light: REGRESSION DETECTED on `{report.dataset_name}`[/bold red]")
        for reason in report.reasons:
            console.print(f"  [red]• {reason}[/red]")
    else:
        console.print(f"\n[bold green]:white_check_mark: No regression on `{report.dataset_name}`[/bold green]")

    console.print(
        f"\n  Score:     {report.baseline_mean_score:.3f} → {report.current_mean_score:.3f} "
        f"({report.score_delta:+.3f})"
    )
    console.print(
        f"  Pass rate: {report.baseline_pass_rate:.1%} → {report.current_pass_rate:.1%} "
        f"({report.pass_rate_delta:+.1%})\n"
    )

    if report.regressed_cases:
        tbl = Table(title="Regressed Cases", box=box.SIMPLE)
        tbl.add_column("Case ID", style="cyan")
        tbl.add_column("Baseline", justify="right")
        tbl.add_column("Current", justify="right")
        tbl.add_column("Delta", justify="right")
        for rc in report.regressed_cases:
            tbl.add_row(
                rc.case_id,
                f"{rc.baseline_score:.3f}",
                f"{rc.current_score:.3f}",
                f"[red]{rc.delta:+.3f}[/red]",
            )
        console.print(tbl)


def save_json_report(result: EvalResult, report: RegressionReport | None, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_name": result.dataset_name,
        "model_id": result.model_id,
        "timestamp": result.timestamp,
        "mean_score": result.mean_score,
        "pass_rate": result.pass_rate,
        "mean_latency_ms": result.mean_latency_ms,
        "regression_detected": report.is_regression if report else None,
        "regression_reasons": report.reasons if report else [],
        "cases": [
            {
                "case_id": r.case_id,
                "weighted_score": r.weighted_score,
                "metric_scores": r.metric_scores,
                "passed": r.passed,
                "actual_output": r.actual_output,
                "expected_output": r.expected_output,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in result.case_results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[dim]JSON report saved to {path}[/dim]")
