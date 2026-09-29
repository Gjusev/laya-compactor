"""Offline metrics: SQuAD-style exact match, latency percentiles, cost math."""

import pytest

from laya_compactor.eval.metrics import cost_per_1k, exact_match, p50


def test_exact_match_normalizes_case_punct_articles_and_whitespace():
    gold = ["George Washington"]
    assert exact_match("George Washington", gold) == 1
    assert exact_match("george  washington!!", gold) == 1
    assert exact_match("the george washington", gold) == 1
    assert exact_match("George Washington Carver", gold) == 0


def test_exact_match_matches_any_of_multiple_golds():
    assert exact_match("Destiny's Child", ["Beyonce", "Destiny's Child"]) == 1
    assert exact_match("beyonce", ["Beyonce", "Destiny's Child"]) == 1
    assert exact_match("TLC", ["Beyonce", "Destiny's Child"]) == 0


def test_exact_match_empty_prediction_never_matches():
    assert exact_match("", ["something"]) == 0
    assert exact_match("  ", ["something"]) == 0


def test_p50_of_timings():
    assert p50([10.0]) == 10.0
    assert p50([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert p50([]) == 0.0


def test_cost_per_1k_from_token_counts():
    # 1000 questions each with 2,000 input tokens and 100 output tokens at
    # $0.15/M input and $0.60/M output: (2000*0.15 + 100*0.60)/1e6 per
    # question -> $0.36 per 1000 questions... hand value: 2,000,000*0.15/1e6
    # = 0.30 input + 100,000*0.60/1e6 = 0.06 output = $0.36.
    assert cost_per_1k(avg_input=2000, avg_output=100,
                       price_input_per_m=0.15, price_output_per_m=0.60) == pytest.approx(0.36)
