"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    id: str
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricConfig:
    name: str
    weight: float = 1.0
    threshold: float = 0.0  # per-metric pass threshold (0 = use global)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    path: str
    name: str = ""


@dataclass
class RegressionConfig:
    """Thresholds that trigger a regression alert."""

    score_drop_threshold: float = 0.05   # absolute drop vs baseline
    pass_rate_min: float = 0.80          # fraction of cases that must pass
    fail_on_regression: bool = True      # exit non-zero when regression detected


@dataclass
class SlackConfig:
    webhook_url: str = ""
    channel: str = "#ml-alerts"
    mention: str = ""  # e.g. "@oncall"
    enabled: bool = True


@dataclass
class Config:
    model: ModelConfig
    datasets: list[DatasetConfig]
    metrics: list[MetricConfig]
    regression: RegressionConfig
    slack: SlackConfig
    baseline_dir: str = "baselines"
    prompt_template: str = ""  # Jinja2 template; receives `input` variable


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    model = ModelConfig(
        id=model_raw["id"],
        temperature=model_raw.get("temperature", 0.0),
        max_tokens=model_raw.get("max_tokens", 1024),
        system_prompt=model_raw.get("system_prompt", ""),
        extra={k: v for k, v in model_raw.items() if k not in {"id", "temperature", "max_tokens", "system_prompt"}},
    )

    datasets = [
        DatasetConfig(path=d["path"], name=d.get("name", Path(d["path"]).stem))
        for d in raw.get("datasets", [])
    ]

    metrics = [
        MetricConfig(
            name=m["name"],
            weight=m.get("weight", 1.0),
            threshold=m.get("threshold", 0.0),
            params=m.get("params", {}),
        )
        for m in raw.get("metrics", [])
    ]

    reg_raw = raw.get("regression", {})
    regression = RegressionConfig(
        score_drop_threshold=reg_raw.get("score_drop_threshold", 0.05),
        pass_rate_min=reg_raw.get("pass_rate_min", 0.80),
        fail_on_regression=reg_raw.get("fail_on_regression", True),
    )

    slack_raw = raw.get("slack", {})
    slack = SlackConfig(
        webhook_url=os.environ.get("SLACK_WEBHOOK_URL", slack_raw.get("webhook_url", "")),
        channel=slack_raw.get("channel", "#ml-alerts"),
        mention=slack_raw.get("mention", ""),
        enabled=slack_raw.get("enabled", True),
    )

    return Config(
        model=model,
        datasets=datasets,
        metrics=metrics,
        regression=regression,
        slack=slack,
        baseline_dir=raw.get("baseline_dir", "baselines"),
        prompt_template=raw.get("prompt_template", "{input}"),
    )
