"""Behavior of compact() at its public seam: kept docs ordered by relevance,
token stats, and one explanation line per cut doc."""

import pytest

from laya_compactor.core import compact


def test_compact_orders_kept_docs_by_relevance_desc(fake_agent, word_counter):
    docs = ["doc a", "doc b", "doc c"]
    agent = fake_agent([0.5, 2.7, 1.9])

    result = compact(
        "query", docs, budget=10_000, agent=agent, min_score=0.0, token_counter=word_counter
    )

    assert [d.doc for d in result.kept] == ["doc b", "doc c", "doc a"]
    assert [d.index for d in result.kept] == [1, 2, 0]


def test_compact_scores_all_docs_in_one_shared_batch(fake_agent, word_counter):
    docs = ["doc a", "doc b"]
    agent = fake_agent([2.0, 1.0])

    compact("query", docs, budget=100, agent=agent, token_counter=word_counter)

    assert len(agent.batch_calls) == 1
    states, questions = agent.batch_calls[0]
    assert len(states) == 2
    assert states[0]["query"] == "query"
    assert states[0]["document"] == "doc a"
    assert questions["relevance"]["type"] == "score"


def test_compact_cuts_lowest_scoring_docs_first_when_budget_is_tight(fake_agent, word_counter):
    # scores: a=3.0, b=2.0, c=1.0; tokens: 1 word each; budget fits only two
    agent = fake_agent([3.0, 2.0, 1.0])

    result = compact(
        "q", ["a", "b", "c"], budget=2, agent=agent, min_score=0.0, token_counter=word_counter
    )

    assert [d.doc for d in result.kept] == ["a", "b"]
    assert [d.doc for d in result.cut] == ["c"]


def test_compact_cuts_docs_below_min_score_even_with_budget_left(fake_agent, word_counter):
    agent = fake_agent([2.5, 0.4])

    result = compact("q", ["good", "bad"], budget=1000, agent=agent, token_counter=word_counter)

    assert [d.doc for d in result.kept] == ["good"]
    assert len(result.cut) == 1
    assert "below min_score" in result.cut[0].reason


def test_every_cut_doc_has_one_explanation_line(fake_agent, word_counter):
    # x scores below min_score; y is kept; z passes the threshold but the
    # budget (1 token) is already spent on y, so z is cut for budget reasons.
    agent = fake_agent([0.1, 3.0, 1.5])

    result = compact(
        "q", ["x", "y", "z"], budget=1, agent=agent, token_counter=word_counter
    )

    assert [d.doc for d in result.kept] == ["y"]
    assert all(isinstance(d.reason, str) and d.reason for d in result.cut)
    budget_cut = [d for d in result.cut if "budget exhausted" in d.reason]
    score_cut = [d for d in result.cut if "below min_score" in d.reason]
    assert len(budget_cut) == 1 and len(score_cut) == 1


def test_compact_reports_token_stats(fake_agent, word_counter):
    docs = ["one two three", "four five", "six"]
    agent = fake_agent([3.0, 2.0, 1.5])

    # budget 4: "one two three" (3 tok) kept, "four five" (2 tok) no longer
    # fits, "six" (1 tok) still does.
    result = compact("q", docs, budget=4, agent=agent, token_counter=word_counter)

    s = result.stats
    assert (s.docs_total, s.docs_kept, s.docs_cut, s.budget) == (3, 2, 1, 4)
    assert s.original_tokens == 6
    assert s.kept_tokens == 4
    assert s.cut_tokens == 2
    assert s.savings_pct == pytest.approx(33.3333, abs=0.001)  # 2 of 6 tokens cut


def test_compact_with_empty_docs_returns_empty_result(fake_agent, word_counter):
    agent = fake_agent([])

    result = compact("q", [], budget=100, agent=agent, token_counter=word_counter)

    assert result.kept == [] and result.cut == []
    assert result.stats.docs_total == 0
    assert result.stats.savings_pct == 0.0
    assert agent.batch_calls == []  # no forward pass for an empty batch


def test_compact_never_calls_laya_load_when_agent_injected(fake_agent, word_counter, monkeypatch):
    import laya_compactor.core as core

    def boom(*args, **kwargs):
        raise AssertionError("compact() must not load a real agent when one is injected")

    monkeypatch.setattr(core, "_load_agent", boom)
    compact("q", ["doc"], budget=10, agent=fake_agent([2.0]), token_counter=word_counter)
