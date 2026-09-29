"""Truncation baselines: what any engineer does in production before reaching
for a decision model."""

from laya_compactor.eval.baselines import full, head_truncate, tail_truncate


def word_counter(text):
    return len(text.split())


DOCS = ["a a a", "b b", "c"]  # 3, 2, 1 tokens in retrieval order


def test_full_keeps_everything_regardless_of_budget():
    kept, stats = full(DOCS, budget=1, token_counter=word_counter)
    assert kept == DOCS
    assert stats["policy"] == "full"
    assert stats["docs_kept"] == 3


def test_head_truncate_keeps_first_retrieved_until_budget():
    kept, stats = head_truncate(DOCS, budget=5, token_counter=word_counter)
    assert kept == ["a a a", "b b"]  # 3 + 2 = 5 fits, 'c' would exceed
    assert stats["policy"] == "head_truncate"
    assert stats["docs_kept"] == 2
    assert stats["kept_tokens"] == 5


def test_tail_truncate_keeps_last_retrieved_until_budget():
    kept, stats = tail_truncate(DOCS, budget=3, token_counter=word_counter)
    assert kept == ["b b", "c"]  # drop from the front until it fits
    assert stats["policy"] == "tail_truncate"
    assert stats["kept_tokens"] == 3


def test_truncation_never_keeps_a_doc_that_alone_exceeds_budget():
    kept, _ = head_truncate(["x y z w"], budget=2, token_counter=word_counter)
    assert kept == []
    kept, _ = tail_truncate(["x y z w"], budget=2, token_counter=word_counter)
    assert kept == []


def test_baselines_report_cut_docs_and_reason():
    _, head_stats = head_truncate(DOCS, budget=3, token_counter=word_counter)
    assert head_stats["docs_cut"] == 2
    assert all("budget" in r for r in head_stats["cut_reasons"])


def test_truncation_handles_duplicate_docs_correctly():
    docs = ["a a", "a a", "b"]  # identical texts, distinct positions
    kept, stats = head_truncate(docs, budget=3, token_counter=word_counter)
    assert kept == ["a a", "b"]  # first duplicate kept, second skipped, b still fits
    assert stats["docs_kept"] == 2 and stats["docs_cut"] == 1
