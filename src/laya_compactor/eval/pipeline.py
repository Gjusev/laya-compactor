"""Eval pipeline: retrieve -> apply a context policy -> generate -> score.

Every collaborator (laya agent, generator, judge, token counter) is injected,
so the whole pipeline is unit-testable offline; run_eval wires the real ones.
"""

import time
from typing import Callable, Dict, List, Optional

from laya_compactor.core import compact
from laya_compactor.eval import baselines
from laya_compactor.eval.datasets import Question
from laya_compactor.eval.metrics import exact_match
from laya_compactor.eval.retrieval import retrieve


def _apply_policy(policy: str, query: str, docs: List[str], budget: int,
                  agent, token_counter: Callable[[str], int]) -> (List[str], dict):
    if policy == "laya_compactor":
        t0 = time.perf_counter()
        result = compact(query, docs, budget, agent=agent,
                         min_score=1.0, token_counter=token_counter)
        ms = (time.perf_counter() - t0) * 1000.0
        # restore original relative order for the prompt (delete, don't reorder)
        kept = [d.doc for d in sorted(result.kept, key=lambda s: s.index)]
        stats = {
            "policy": "laya_compactor",
            "docs_kept": len(result.kept),
            "docs_cut": len(result.cut),
            "kept_tokens": result.stats.kept_tokens,
            "compaction_ms": ms,
        }
        return kept, stats
    fn = {"full": baselines.full,
          "head_truncate": baselines.head_truncate,
          "tail_truncate": baselines.tail_truncate}[policy]
    kept, stats = fn(docs, budget, token_counter)
    stats = dict(stats, compaction_ms=None)
    return kept, stats


def run_generation_pass(questions: List[Question], policies: List[str], budget: int,
                        k: int, generator, agent,
                        token_counter: Callable[[str], int]) -> List[dict]:
    """One row per (question, policy): EM, token usage, kept/gold-doc stats."""
    rows: List[dict] = []
    for q in questions:
        retrieved = retrieve(q.question, q.docs, k)
        for policy in policies:
            kept, stats = _apply_policy(policy, q.question, retrieved, budget,
                                        agent, token_counter)
            gen = generator.generate(q.question, kept)
            kept_set = set(kept)
            gold_total = len(q.gold_indices) or 1
            gold_kept = sum(1 for i in q.gold_indices if q.docs[i] in kept_set)
            rows.append({
                "qid": q.id,
                "question": q.question,
                "gold_answers": list(q.gold_answers),
                "policy": policy,
                "answer": gen["answer"],
                "em": exact_match(gen["answer"], q.gold_answers),
                "input_tokens": gen["input_tokens"],
                "output_tokens": gen["output_tokens"],
                "kept_tokens": stats["kept_tokens"],
                "retrieved_docs": len(retrieved),
                "kept_docs": stats["docs_kept"],
                "gold_kept": gold_kept / gold_total,
                "compaction_ms": stats["compaction_ms"],
                "error": None,
            })
    return rows


def run_judge_pass(rows: List[dict], judge) -> List[dict]:
    """Attach pairwise verdicts: every non-full row vs its question's full row."""
    reference = {r["qid"]: r["answer"] for r in rows if r["policy"] == "full"}
    out = []
    for r in rows:
        if r["policy"] == "full" or r["qid"] not in reference:
            out.append(dict(r, judge=None))
            continue
        out.append(dict(r, judge=judge.judge(
            r["question"], r["gold_answers"], reference[r["qid"]], r["answer"])))
    return out
