"""Rubric sensitivity: run every rubric phrasing over a labeled query/doc set
and report how far scores and rounded decisions move when only the phrasing
changes."""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Dict, List, Optional

from laya_compactor.core import _load_agent
from laya_compactor.rubrics import QUESTION_ID, RUBRIC_VARIANTS

DATASET_FILENAME = "relevance_mini_dataset.jsonl"


def load_dataset(path: Optional[str] = None) -> List[dict]:
    """Load labeled rows ({id, query, doc, label}); defaults to the packaged mini-dataset."""
    if path is None:
        text = (
            resources.files("laya_compactor") / "data" / DATASET_FILENAME
        ).read_text(encoding="utf-8")
    else:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass
class SensitivityReport:
    n: int
    per_variant: Dict[str, dict]  # name -> {n, mean_score, agreement, mae}
    spread_mean: float  # mean per-row (max - min) score across variants
    full_agreement: float  # fraction of rows where all variants round to the same level

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "per_variant": self.per_variant,
            "spread_mean": round(self.spread_mean, 4),
            "full_agreement": round(self.full_agreement, 4),
        }


def run_sensitivity(
    rows: List[dict],
    agent=None,
    rubrics: Optional[Dict[str, dict]] = None,
) -> SensitivityReport:
    """Score every row under every rubric phrasing (one predict_batch per phrasing)."""
    if agent is None:
        agent = _load_agent()
    if rubrics is None:
        rubrics = RUBRIC_VARIANTS

    labels = [row["label"] for row in rows]
    scores_by_variant: Dict[str, List[float]] = {}
    per_variant: Dict[str, dict] = {}
    for name, rubric in rubrics.items():
        if rows:
            states = [{"query": r["query"], "document": r["doc"]} for r in rows]
            results = agent.predict_batch(states, {QUESTION_ID: rubric})
            scores = [float(res["answers"][QUESTION_ID]["score"]) for res in results]
        else:
            scores = []
        scores_by_variant[name] = scores
        n = len(scores)
        per_variant[name] = {
            "n": n,
            "mean_score": _mean(scores),
            "agreement": _mean([round(s) == l for s, l in zip(scores, labels)]),
            "mae": _mean([abs(s - l) for s, l in zip(scores, labels)]),
        }

    spreads = []
    agree = []
    for i in range(len(rows)):
        row_scores = [scores_by_variant[name][i] for name in scores_by_variant]
        spreads.append(max(row_scores) - min(row_scores))
        agree.append(len({round(s) for s in row_scores}) == 1)

    return SensitivityReport(
        n=len(rows),
        per_variant=per_variant,
        spread_mean=_mean(spreads),
        full_agreement=_mean(agree),
    )


def _mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: python -m laya_compactor.sensitivity [dataset.jsonl]"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="laya-sensitivity",
        description="Measure rubric-phrasing sensitivity on a labeled query/doc dataset.",
    )
    parser.add_argument("dataset", nargs="?", default=None,
                        help=f"JSONL dataset path (default: packaged {DATASET_FILENAME})")
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset)
    report = run_sensitivity(rows)  # loads the real checkpoint — this is the slow path
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
