# Model Regression Detection System (MRDS)

A CI/CD-style pipeline that continuously tests any LLM-powered feature against a **golden dataset** whenever a prompt or model changes, detects quality regressions, and **alerts your team via Slack** before bad outputs reach users.

---

## How it works

```
Prompt / model change
        │
        ▼
 ┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
 │  Golden      │─────▶│   Evaluator     │─────▶│  Baseline        │
 │  Dataset     │      │  (Anthropic API)│      │  Comparison      │
 └──────────────┘      └─────────────────┘      └──────┬───────────┘
                                                        │
                                          Regression?   │
                                       ┌────────────────┤
                                       ▼                ▼
                               Slack Alert ⚠️    CI passes ✅
```

1. **Golden datasets** — curated JSON files with `(input, expected_output, expected_labels)` test cases
2. **Evaluator** — renders each input through a Jinja2 prompt template, calls the model, scores outputs with pluggable metrics
3. **Baseline** — the last known-good run stored as JSON; created automatically on first run
4. **Detector** — flags a regression if mean score drops, pass rate falls below a threshold, or individual cases regress
5. **Slack alerts** — rich block-kit messages sent on regression; silent on pass
6. **GitHub Actions** — runs on every PR / push that touches prompts, config, or datasets

---

## Quick start

### 1. Install

```bash
pip install -e .
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...   # optional
```

### 3. Run an evaluation

```bash
# First run — no baseline yet, saves results as baseline automatically
mrds run --config config.yaml --verbose

# Subsequent runs — compares to baseline and reports regressions
mrds run --config config.yaml --report report.json
```

### 4. Force-update the baseline

```bash
mrds update-baseline --dataset golden_datasets/sentiment.json
```

---

## Project layout

```
.
├── config.yaml                    # Main config (model, datasets, metrics, thresholds)
├── golden_datasets/
│   ├── sentiment.json             # Example: sentiment classification
│   └── summarization.json         # Example: summarization quality
├── baselines/                     # Auto-generated — do NOT delete
│   └── sentiment__claude-haiku-4-5-20251001.json
├── src/mrds/
│   ├── config.py                  # Config loading
│   ├── dataset.py                 # Golden dataset I/O
│   ├── evaluator.py               # Model inference + scoring
│   ├── metrics.py                 # Pluggable scoring functions
│   ├── baseline.py                # Baseline save/load
│   ├── detector.py                # Regression detection
│   ├── alerts.py                  # Slack notifications
│   ├── report.py                  # Terminal + JSON reporting
│   └── cli.py                     # mrds CLI
├── tests/
│   ├── test_metrics.py
│   ├── test_detector.py
│   └── test_dataset.py
└── .github/workflows/
    └── regression-check.yml       # CI/CD pipeline
```

---

## Configuration (`config.yaml`)

```yaml
model:
  id: claude-haiku-4-5-20251001    # any Anthropic model
  temperature: 0.0                  # 0 = deterministic (best for regression testing)
  max_tokens: 512
  system_prompt: "You are a helpful assistant."

prompt_template: "{{ input }}"      # Jinja2 — use {{ input.field }} for dicts

datasets:
  - path: golden_datasets/sentiment.json

metrics:
  - name: label_match               # checks expected_labels
    weight: 2.0
    threshold: 1.0                  # per-case pass threshold
  - name: semantic_similarity       # embedding cosine similarity
    weight: 1.0
    threshold: 0.6

regression:
  score_drop_threshold: 0.05        # alert if mean score drops > 5 pp
  pass_rate_min: 0.80               # alert if < 80% of cases pass
  fail_on_regression: true          # exit(1) — blocks CI merge

slack:
  enabled: true                     # needs SLACK_WEBHOOK_URL env var
  channel: "#ml-alerts"
  mention: "@channel"
```

---

## Available metrics

| Name | Description |
|------|-------------|
| `exact_match` | 1.0 if output == expected (stripped) |
| `label_match` | 1.0 if any `expected_labels` appears in output |
| `semantic_similarity` | Cosine similarity via `all-MiniLM-L6-v2` embeddings |
| `contains_keywords` | Fraction of required keywords found |
| `json_valid` | 1.0 if output is parseable JSON |
| `length_ratio` | Score based on output length vs. expected length |
| `no_harmful_content` | 0.0 if any banned regex pattern matches |

Add custom metrics by registering a function in `src/mrds/metrics.py`.

---

## GitHub Actions

Add secrets to your repository:

| Secret | Description |
|--------|-------------|
| `ANTHROPIC_API_KEY` | Required — model inference |
| `SLACK_WEBHOOK_URL` | Optional — Slack alerts |

The workflow (`regression-check.yml`) runs on PRs and pushes that touch:
- `golden_datasets/**`
- `config.yaml`
- `src/**`

Baselines are cached between runs. On `main`, passing runs can auto-commit updated baselines.

---

## CLI reference

```
mrds run              Run evaluations and check for regressions
mrds update-baseline  Force-update baseline for a dataset
mrds list-baselines   Show all saved baselines
mrds init-dataset     Write a starter golden dataset file
```

---

## Running tests

```bash
pytest tests/ -v
```

