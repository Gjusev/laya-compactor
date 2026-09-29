"""Core compaction: score every retrieved doc in one shared forward pass, then
select what fits the token budget."""

from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from laya_compactor.rubrics import QUESTION_ID, get_rubric


def default_token_counter(text: str) -> int:
    """Offline token proxy: whitespace word count.

    Deliberately deterministic and dependency-free so compaction stays testable
    offline; inject a real tokenizer (e.g. tiktoken) when the budget must match
    a specific provider's counts.
    """
    return len(text.split())


@dataclass
class ScoredDoc:
    """A doc selected for the compact context, ordered most relevant first."""

    index: int  # position in the input docs list
    doc: str
    score: float
    tokens: int


@dataclass
class CutDoc:
    """A doc dropped from the context, with the reason line explaining why."""

    index: int
    doc: str
    score: float
    tokens: int
    reason: str


@dataclass
class CompactStats:
    docs_total: int
    docs_kept: int
    docs_cut: int
    budget: int
    original_tokens: int
    kept_tokens: int
    cut_tokens: int

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * self.cut_tokens / self.original_tokens


@dataclass
class CompactResult:
    kept: List[ScoredDoc]
    cut: List[CutDoc]
    stats: CompactStats

    def to_dict(self) -> dict:
        """JSON-friendly view (used by the CLI)."""
        payload = asdict(self)
        payload["stats"]["savings_pct"] = round(self.stats.savings_pct, 2)
        return payload


def _load_agent():
    import laya  # deferred: the real Agent downloads a checkpoint on first load

    return laya.load()


def compact(
    query: str,
    docs: List[str],
    budget: int,
    *,
    agent=None,
    rubric=None,
    min_score: float = 1.0,
    token_counter: Optional[Callable[[str], int]] = None,
) -> CompactResult:
    """Score docs for relevance to the query in one predict_batch call and keep
    the most relevant docs that fit the token budget.

    docs with a score below min_score are cut even when budget remains; the rest
    are cut lowest-score-first once the budget is exhausted.
    """
    if token_counter is None:
        token_counter = default_token_counter
    questions = {QUESTION_ID: get_rubric(rubric)}
    if not docs:
        return CompactResult(
            kept=[],
            cut=[],
            stats=CompactStats(0, 0, 0, budget, 0, 0, 0),
        )
    if agent is None:
        agent = _load_agent()

    states = [{"query": query, "document": doc} for doc in docs]
    results = agent.predict_batch(states, questions)
    scored = [
        ScoredDoc(index=i, doc=doc, score=float(results[i]["answers"][QUESTION_ID]["score"]),
                  tokens=token_counter(doc))
        for i, doc in enumerate(docs)
    ]

    # Most relevant first; ties keep input order.
    ranked = sorted(scored, key=lambda s: (-s.score, s.index))
    kept: List[ScoredDoc] = []
    cut: List[CutDoc] = []
    remaining = budget
    for s in ranked:
        if s.score < min_score:
            cut.append(CutDoc(s.index, s.doc, s.score, s.tokens,
                              reason=f"score {s.score:.2f} below min_score {min_score:.2f}"))
        elif s.tokens > remaining:
            cut.append(CutDoc(s.index, s.doc, s.score, s.tokens,
                              reason=f"budget exhausted: needs {s.tokens} tokens, {remaining} left"))
        else:
            kept.append(s)
            remaining -= s.tokens

    original = sum(s.tokens for s in scored)
    kept_tokens = sum(s.tokens for s in kept)
    return CompactResult(
        kept=kept,
        cut=cut,
        stats=CompactStats(
            docs_total=len(scored),
            docs_kept=len(kept),
            docs_cut=len(cut),
            budget=budget,
            original_tokens=original,
            kept_tokens=kept_tokens,
            cut_tokens=original - kept_tokens,
        ),
    )
