"""Truncation baselines: the honest production defaults laya-compactor must beat.

All policies return (kept_docs, stats) where stats carries the policy name,
kept/cut counts, kept tokens and one reason line per cut doc — the same shape
compact() produces, so the harness treats every policy uniformly.
"""

from typing import Callable, List, Tuple


def _truncate(docs: List[str], budget: int, token_counter: Callable[[str], int],
              policy: str, drop_from_front: bool) -> Tuple[List[str], dict]:
    """Greedily fit docs into the budget, scanning from one end; docs that do
    not fit are skipped (smaller ones after them may still fit)."""
    order = range(len(docs)) if not drop_from_front else range(len(docs) - 1, -1, -1)
    kept_indices: List[int] = []
    kept_tokens = 0
    for i in order:
        tokens = token_counter(docs[i])
        if kept_tokens + tokens <= budget:
            kept_indices.append(i)
            kept_tokens += tokens
    kept = [docs[i] for i in sorted(kept_indices)]
    kept_set = set(kept_indices)
    cut_reasons = [
        f"budget: needs {token_counter(doc)} tokens, no room left"
        for i, doc in enumerate(docs) if i not in kept_set
    ]
    stats = {
        "policy": policy,
        "docs_total": len(docs),
        "docs_kept": len(kept),
        "docs_cut": len(cut_reasons),
        "kept_tokens": kept_tokens,
        "cut_reasons": cut_reasons,
    }
    return kept, stats


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
