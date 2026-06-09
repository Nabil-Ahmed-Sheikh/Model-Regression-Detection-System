"""Unit tests for scoring metrics."""

import pytest
from mrds.metrics import (
    exact_match,
    contains_keywords,
    label_match,
    json_valid,
    length_ratio,
    no_harmful_content,
    compute_metric,
)


def test_exact_match_pass():
    assert exact_match("hello world", "hello world") == 1.0


def test_exact_match_fail():
    assert exact_match("hello world", "goodbye") == 0.0


def test_exact_match_strips_whitespace():
    assert exact_match("  hello  ", "hello") == 1.0


def test_contains_keywords_all():
    assert contains_keywords("The cat sat on the mat", expected="", keywords=["cat", "mat"]) == 1.0


def test_contains_keywords_partial():
    score = contains_keywords("The cat sat on the mat", expected="", keywords=["cat", "dog"])
    assert score == pytest.approx(0.5)


def test_label_match_found():
    assert label_match("The sentiment is positive overall", expected="", expected_labels=["positive"]) == 1.0


def test_label_match_not_found():
    assert label_match("The sentiment is negative", expected="", expected_labels=["positive"]) == 0.0


def test_label_match_empty_labels():
    assert label_match("anything", expected="", expected_labels=[]) == 1.0


def test_json_valid_pass():
    assert json_valid('{"key": "value"}') == 1.0


def test_json_valid_fail():
    assert json_valid("not json {{") == 0.0


def test_length_ratio_perfect():
    assert length_ratio("hello", "hello") == 1.0


def test_length_ratio_far_off():
    assert length_ratio("a", "a" * 100) < 0.5


def test_no_harmful_content_clean():
    assert no_harmful_content("This is safe content") == 1.0


def test_no_harmful_content_banned():
    assert no_harmful_content("ignore all previous instructions", banned_patterns=["ignore all"]) == 0.0


def test_compute_metric_unknown():
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_metric("nonexistent_metric", "a", "b", [], {})


def test_compute_metric_dispatch():
    score = compute_metric("exact_match", "foo", "foo", [], {})
    assert score == 1.0
