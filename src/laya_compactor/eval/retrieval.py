"""BM25 retrieval over a question's candidate docs (rank-bm25, pure Python)."""

from typing import List

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def retrieve(query: str, docs: List[str], k: int) -> List[str]:
    """Top-k docs by BM25 score, best first (order matters: truncation baselines
    cut from this ranking, so retrieval order is part of the measured policy)."""
    if not docs:
        return []
    bm25 = BM25Okapi([_tokenize(doc) for doc in docs])
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
    return [docs[i] for i in ranked[: min(k, len(docs))]]
