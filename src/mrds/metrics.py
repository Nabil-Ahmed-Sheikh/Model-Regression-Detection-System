"""Scoring metrics for LLM output evaluation."""

from __future__ import annotations

import re
from typing import Any

# Lazy imports for heavy dependencies
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Individual metric functions
# Each takes (actual: str, expected: str, **params) → float in [0, 1]
# ---------------------------------------------------------------------------

def exact_match(actual: str, expected: str, **_) -> float:
    return 1.0 if actual.strip() == expected.strip() else 0.0


def contains_keywords(actual: str, expected: str, keywords: list[str] | None = None, **_) -> float:
    """Fraction of required keywords found in the actual output."""
    kws = keywords or expected.split()
    if not kws:
        return 1.0
    hits = sum(1 for kw in kws if kw.lower() in actual.lower())
    return hits / len(kws)


def semantic_similarity(actual: str, expected: str, **_) -> float:
    """Cosine similarity between sentence embeddings."""
    import numpy as np

    embedder = _get_embedder()
    vecs = embedder.encode([actual, expected], normalize_embeddings=True)
    return float(np.clip(float(vecs[0] @ vecs[1]), 0.0, 1.0))


def label_match(actual: str, expected_labels: list[str], **_) -> float:
    """Check whether any expected label appears in the actual output."""
    if not expected_labels:
        return 1.0
    actual_lower = actual.lower()
    return 1.0 if any(lbl.lower() in actual_lower for lbl in expected_labels) else 0.0


def json_valid(actual: str, **_) -> float:
    """Check whether output is parseable JSON."""
    import json
    try:
        json.loads(actual)
        return 1.0
    except Exception:
        return 0.0


def length_ratio(actual: str, expected: str, tolerance: float = 0.5, **_) -> float:
    """Score based on how close the output length is to the expected length."""
    if not expected:
        return 1.0
    ratio = len(actual) / len(expected)
    deviation = abs(ratio - 1.0)
    return max(0.0, 1.0 - deviation / tolerance)


def no_harmful_content(actual: str, banned_patterns: list[str] | None = None, **_) -> float:
    """Return 0 if output matches any banned regex pattern, else 1."""
    patterns = banned_patterns or []
    for pat in patterns:
        if re.search(pat, actual, re.IGNORECASE):
            return 0.0
    return 1.0


# Registry — maps metric name → function
METRIC_REGISTRY: dict[str, Any] = {
    "exact_match": exact_match,
    "contains_keywords": contains_keywords,
    "semantic_similarity": semantic_similarity,
    "label_match": label_match,
    "json_valid": json_valid,
    "length_ratio": length_ratio,
    "no_harmful_content": no_harmful_content,
}


def compute_metric(name: str, actual: str, expected: str, expected_labels: list[str], params: dict) -> float:
    fn = METRIC_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown metric: {name!r}. Available: {list(METRIC_REGISTRY)}")
    return fn(actual=actual, expected=expected, expected_labels=expected_labels, **params)
