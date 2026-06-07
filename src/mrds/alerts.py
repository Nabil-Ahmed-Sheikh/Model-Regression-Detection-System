"""Slack alerting for regression events."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from .config import SlackConfig
from .detector import RegressionReport
from .evaluator import EvalResult

if TYPE_CHECKING:
    pass


def _score_emoji(delta: float) -> str:
    if delta >= 0:
        return ":white_check_mark:"
    if delta >= -0.05:
        return ":warning:"
    return ":red_circle:"


def _build_regression_blocks(report: RegressionReport, mention: str) -> list[dict]:
    header = f":rotating_light: *Model Regression Detected* — `{report.model_id}` on `{report.dataset_name}`"
    if mention:
        header = f"{mention} {header}"

    summary_lines = [
        f"• Mean score: `{report.baseline_mean_score:.3f}` → `{report.current_mean_score:.3f}` "
        f"({report.score_delta:+.3f})",
        f"• Pass rate:  `{report.baseline_pass_rate:.1%}` → `{report.current_pass_rate:.1%}` "
        f"({report.pass_rate_delta:+.1%})",
        f"• Regressed cases: *{len(report.regressed_cases)}*",
    ]

    reason_text = "\n".join(f"  › {r}" for r in report.reasons)

    case_lines = []
    for rc in report.regressed_cases[:10]:  # cap to avoid huge messages
        case_lines.append(
            f"  • `{rc.case_id}`: {rc.baseline_score:.3f} → {rc.current_score:.3f} ({rc.delta:+.3f})"
        )
    if len(report.regressed_cases) > 10:
        case_lines.append(f"  … and {len(report.regressed_cases) - 10} more")

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(summary_lines)}},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Reasons:*\n{reason_text}"},
        },
    ]
    if case_lines:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Regressed cases:*\n" + "\n".join(case_lines)},
            }
        )
    return blocks


def _build_success_blocks(result: EvalResult) -> list[dict]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":white_check_mark: *No regression* — `{result.model_id}` on `{result.dataset_name}`\n"
                    f"• Mean score: `{result.mean_score:.3f}`  Pass rate: `{result.pass_rate:.1%}`"
                ),
            },
        }
    ]


def send_slack_alert(
    slack_cfg: SlackConfig,
    report: RegressionReport | None,
    result: EvalResult,
) -> bool:
    """Send a Slack notification. Returns True on success."""
    if not slack_cfg.enabled or not slack_cfg.webhook_url:
        return False

    try:
        from slack_sdk.webhook import WebhookClient

        client = WebhookClient(slack_cfg.webhook_url)

        if report and report.is_regression:
            blocks = _build_regression_blocks(report, slack_cfg.mention)
        else:
            blocks = _build_success_blocks(result)

        response = client.send(blocks=blocks)
        return response.status_code == 200
    except Exception as exc:
        print(f"[MRDS] Slack alert failed: {exc}")
        return False
