"""Eval datasets: HotpotQA (multi-hop) + SQuAD (single-hop).

The plan names HotpotQA + NQ. HotpotQA ships its own 10-paragraph candidate
set per question (supporting docs + distractors) — the standard paragraph
retrieval setting. NQ-open ships questions and answers but no passages, and
streaming full NQ for 200 rows pulls multi-GB shards, so the single-hop arm
uses SQuAD: same public single-hop QA role, self-contained passages, tiny
download. Swapping NQ back in means adding one loader + parser here.

The HF loaders below stream, shuffle deterministically (seed + buffer size)
and take the first n rows; `datasets` caches downloads under ~/.cache.
Determinism holds for a fixed datasets library version — pin it in CI when
the table matters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Question:
    id: str
    question: str
    gold_answers: List[str]  # short gold answers for exact match
    docs: List[str]  # candidate passages retrieval runs over
    gold_indices: List[int] = field(default_factory=list)  # answer-bearing docs


def parse_hotpotqa_row(row: dict) -> Question:
    """One HotpotQA distractor row -> Question with one doc per paragraph."""
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    docs = [f"{title}: {''.join(sents).strip()}" for title, sents in zip(titles, sentences)]
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    gold_indices = [i for i, title in enumerate(titles) if title in gold_titles]
    return Question(
        id=row["_id"] if "_id" in row else row.get("id", ""),
        question=row["question"],
        gold_answers=[row["answer"]],
        docs=docs,
        gold_indices=gold_indices,
    )


def parse_squad_row(row: dict) -> Question:
    """One SQuAD row -> Question whose candidate set starts as its own passage."""
    answers = row.get("answers", {})
    texts = answers.get("text", []) if isinstance(answers, dict) else list(answers)
    return Question(
        id=row["id"],
        question=row["question"],
        gold_answers=texts,
        docs=[row["context"]],
        gold_indices=[0] if row["context"] else [],
    )


def build_squad_corpus(rows: List[Question]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Shared passage corpus from all rows' gold passages; maps each question id
    to its gold doc's index in the corpus. Retrieval then runs over the whole
    corpus, so each question competes with every other question's passages."""
    corpus: List[str] = []
    gold_sets: Dict[str, List[int]] = {}
    for q in rows:
        gold_sets[q.id] = []
        for doc in q.docs:
            corpus.append(doc)
            if doc in [q.docs[i] for i in q.gold_indices]:
                gold_sets[q.id].append(len(corpus) - 1)
    return corpus, gold_sets


def _streaming_subset(name: str, n: int, seed: int, config: str = None, split: str = "validation"):
    from datasets import load_dataset  # deferred: network + heavy imports stay out of unit tests

    ds = load_dataset(name, config, split=split, streaming=True) if config \
        else load_dataset(name, split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=n).take(n)


def load_hotpotqa(n: int = 200, seed: int = 42) -> List[Question]:
    """n HotpotQA distractor-validation questions, deterministic subset."""
    ds = _streaming_subset("hotpotqa/hotpot_qa", n, seed, config="distractor")
    return [parse_hotpotqa_row(row) for row in ds]


def load_squad(n: int = 200, seed: int = 42) -> List[Question]:
    """n SQuAD validation questions whose docs form one shared corpus."""
    ds = _streaming_subset("rajpurkar/squad", n, seed)
    rows = [parse_squad_row(row) for row in ds]
    corpus, gold_sets = build_squad_corpus(rows)
    for q in rows:
        q.docs = corpus
        q.gold_indices = gold_sets[q.id]
    return rows


LOADERS = {"hotpotqa": load_hotpotqa, "squad": load_squad}
