"""CLI entry point for MRDS."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from .config import load_config
from .dataset import load_dataset, save_dataset, GoldenDataset, TestCase
from .evaluator import run_evaluation
from .baseline import save_baseline, load_baseline, baseline_exists
from .detector import detect_regressions
from .alerts import send_slack_alert
from .report import print_eval_summary, print_regression_report, save_json_report

console = Console()


@click.group()
@click.version_option()
def main():
    """Model Regression Detection System — LLM quality CI/CD pipeline."""


@main.command()
@click.option("--config", "-c", default="config.yaml", show_default=True, help="Config file path")
@click.option("--dataset", "-d", multiple=True, help="Override dataset path(s)")
@click.option("--update-baseline", is_flag=True, help="Save results as new baseline after run")
@click.option("--report", "-r", default="", help="Save JSON report to this path")
@click.option("--no-slack", is_flag=True, help="Suppress Slack alerts")
@click.option("--verbose", "-v", is_flag=True)
def run(config, dataset, update_baseline, report, no_slack, verbose):
    """Run evaluations against golden datasets and detect regressions."""
    cfg = load_config(config)

    dataset_paths = list(dataset) if dataset else [d.path for d in cfg.datasets]
    if not dataset_paths:
        console.print("[red]No datasets configured. Add datasets to config.yaml or pass --dataset.[/red]")
        sys.exit(1)

    any_regression = False

    for ds_path in dataset_paths:
        ds = load_dataset(ds_path)
        console.print(f"\n[bold cyan]Evaluating[/bold cyan] {ds.name} ({len(ds)} cases) with {cfg.model.id}")

        result = run_evaluation(cfg, ds, verbose=verbose)
        print_eval_summary(result)

        # Check vs baseline
        baseline = load_baseline(cfg.baseline_dir, ds.name, cfg.model.id)
        regression_report = None

        if baseline:
            regression_report = detect_regressions(result, baseline, cfg.regression)
            print_regression_report(regression_report)

            if not no_slack:
                ok = send_slack_alert(cfg.slack, regression_report, result)
                if ok:
                    console.print("[dim]Slack alert sent.[/dim]")

            if regression_report.is_regression:
                any_regression = True
        else:
            console.print("[yellow]No baseline found — saving current results as baseline.[/yellow]")
            update_baseline = True

        if update_baseline:
            path = save_baseline(result, cfg.baseline_dir)
            console.print(f"[green]Baseline saved to {path}[/green]")

        if report:
            report_path = report if len(dataset_paths) == 1 else report.replace(".json", f"_{ds.name}.json")
            save_json_report(result, regression_report, report_path)

    if any_regression and cfg.regression.fail_on_regression:
        console.print("\n[bold red]FAILED: regression(s) detected. Exiting with code 1.[/bold red]")
        sys.exit(1)
    else:
        console.print("\n[bold green]All checks passed.[/bold green]")


@main.command()
@click.option("--config", "-c", default="config.yaml", show_default=True)
@click.option("--dataset", "-d", required=True, help="Dataset path to update baseline for")
def update_baseline(config, dataset):
    """Run evaluation and force-update the baseline (no regression check)."""
    cfg = load_config(config)
    ds = load_dataset(dataset)
    console.print(f"[cyan]Updating baseline for {ds.name}…[/cyan]")
    result = run_evaluation(cfg, ds, verbose=True)
    print_eval_summary(result)
    path = save_baseline(result, cfg.baseline_dir)
    console.print(f"[green]Baseline updated: {path}[/green]")


@main.command()
@click.option("--config", "-c", default="config.yaml", show_default=True)
def list_baselines(config):
    """List all saved baselines."""
    cfg = load_config(config)
    baselines = sorted(Path(cfg.baseline_dir).glob("*.json"))
    if not baselines:
        console.print("[yellow]No baselines found.[/yellow]")
        return
    for p in baselines:
        console.print(f"  {p}")


@main.command()
@click.argument("output", default="golden_datasets/example.json")
def init_dataset(output):
    """Create a starter golden dataset file."""
    ds = GoldenDataset(
        name="example",
        cases=[
            TestCase(
                id="sentiment-positive",
                input="The product is absolutely fantastic, I love it!",
                expected_output="positive",
                expected_labels=["positive"],
                tags=["sentiment"],
            ),
            TestCase(
                id="sentiment-negative",
                input="This is the worst experience I've ever had.",
                expected_output="negative",
                expected_labels=["negative"],
                tags=["sentiment"],
            ),
            TestCase(
                id="summarize-short",
                input="The quick brown fox jumps over the lazy dog. It was a sunny afternoon.",
                expected_output="A fox jumps over a dog on a sunny afternoon.",
                tags=["summarization"],
            ),
        ],
    )
    save_dataset(ds, output)
    console.print(f"[green]Starter dataset written to {output}[/green]")
