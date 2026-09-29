"""Truncation baselines: the honest production defaults laya-compactor must beat.

All policies return (kept_docs, stats) where stats carries the policy name,
kept/cut counts, kept tokens and one reason line per cut doc — the same shape
compact() produces, so the harness treats every policy uniformly.
"""

from typing import Callable, List, Tuple


def _truncate(docs: List[str], budget: int, token_counter: Callable[[str], int],
              policy: str, drop_from_front: bool) -> Tuple[List[str], dict]:
    order = list(reversed(docs)) if drop_from_front else list(docs)
    kept: List[str] = []
    kept_tokens = 0
    for doc in order:
        tokens = token_counter(doc)
        if kept_tokens + tokens > budget:
            continue
        kept.append(doc)
        kept_tokens += tokens
    if drop_from_front:
        kept.reverse()  # restore retrieval order for the prompt
    kept_set = set(range(len(kept)))
    # rebuild cut list from the original order for stable reasons
    kept_docs = kept
    cut = []
    seen_kept = list(kept_docs)
    for doc in docs:
        if doc in seen_kept:
            seen_kept.remove(doc)
            continue
        tokens = token_counter(doc)
        cut.append(f"budget: needs {tokens} tokens, no room left")
    stats = {
        "policy": policy,
        "docs_total": len(docs),
        "docs_kept": len(kept_docs),
        "docs_cut": len(cut),
        "kept_tokens": kept_tokens,
        "cut_reasons": cut,
    }
    return kept_docs, stats


def full(docs: List[str], budget: int, token_counter: Callable[[str], int]) -> Tuple[List[str], dict]:
    """The reference: every retrieved doc, no budget applied."""
    kept_tokens = sum(token_counter(doc) for doc in docs)
    return list(docs), {
        "policy": "full",
        "docs_total": len(docs),
        "docs_kept": len(docs),
        "docs_cut": 0,
        "kept_tokens": kept_tokens,
        "cut_reasons": [],
    }


def head_truncate(docs: List[str], budget: int, token_counter: Callable[[str], int]) -> Tuple[List[str], dict]:
    """Keep the first retrieved docs that fit the budget (top of the ranking)."""
    return _truncate(docs, budget, token_counter, "head_truncate", drop_from_front=False)


def tail_truncate(docs: List[str], budget: int, token_counter: Callable[[str], int]) -> Tuple[List[str], dict]:
    """Keep the last retrieved docs that fit (drop from the front of the ranking)."""
    return _truncate(docs, budget, token_counter, "tail_truncate", drop_from_front=True)
