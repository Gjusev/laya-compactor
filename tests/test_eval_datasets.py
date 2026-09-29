"""Dataset loading: pure row parsers (tested here, offline) + thin HF loaders.

The plan's single-hop dataset is NQ; NQ-open ships no passages and full NQ
needs a multi-GB stream for 200 rows, so the single-hop arm uses SQuAD (same
public single-hop role, self-contained passages). Swap is one loader away.
"""

import pytest

from laya_compactor.eval.datasets import (
    Question,
    build_squad_corpus,
    parse_hotpotqa_row,
    parse_squad_row,
)


HOTPOT_ROW = {
    "id": "hp-1",
    "question": "What government position was held by the woman who portrayed Corliss Archer in Kiss and Tell?",
    "answer": "Chief of Protocol",
    "supporting_facts": {"title": ["Shirley Temple"], "sent_id": [1]},
    "context": {
        "title": ["Corliss Archer", "Shirley Temple", "Kiss and Tell (film)", "Physicist"],
        "sentences": [
            ["Corliss Archer is a character..."],
            ["Shirley Temple Black was an American actress and diplomat.",
             "She was appointed Chief of Protocol in 1976."],
            ["Kiss and Tell is a 1945 film..."],
            ["A physicist studies matter."],
        ],
    },
}


def test_parse_hotpotqa_row_builds_one_doc_per_title():
    q = parse_hotpotqa_row(HOTPOT_ROW)
    assert isinstance(q, Question)
    assert q.id == "hp-1"
    assert q.gold_answers == ["Chief of Protocol"]
    assert len(q.docs) == 4
    assert q.docs[0].startswith("Corliss Archer:")
    assert "Chief of Protocol" in q.docs[1]
    assert q.gold_indices == [1]  # only the Shirley Temple doc answers it


def test_parse_hotpotqa_row_without_gold_support_keeps_all_docs_as_candidates():
    # supporting_facts names a title that is not in the context: no gold doc
    row = dict(HOTPOT_ROW, context={"title": ["A"], "sentences": [["text about A."]]})
    q = parse_hotpotqa_row(row)
    assert q.docs == ["A: text about A."]
    assert q.gold_indices == []


SQUAD_ROW = {
    "id": "sq-1",
    "title": "Normans",
    "context": "The Normans were the people who inhabited Normandy in northern France.",
    "question": "In what country is Normandy?",
    "answers": {"text": ["France", "Normandy region"]},
}


def test_parse_squad_row_uses_context_as_gold_doc():
    q = parse_squad_row(SQUAD_ROW)
    assert q.question == "In what country is Normandy?"
    assert q.gold_answers == ["France", "Normandy region"]
    assert q.docs == [SQUAD_ROW["context"]]
    assert q.gold_indices == [0]


def test_build_squad_corpus_merges_docs_and_marks_gold():
    rows = [parse_squad_row(SQUAD_ROW),
            parse_squad_row(dict(SQUAD_ROW, id="sq-2", title="Biology",
                                 context="Cells are the basic unit of life.",
                                 question="What is the basic unit of life?",
                                 answers={"text": ["cells"]}))]

    corpus, gold_sets = build_squad_corpus(rows)

    assert len(corpus) == 2
    assert gold_sets["sq-1"] == [corpus.index(SQUAD_ROW["context"])]
    assert gold_sets["sq-2"] == [corpus.index("Cells are the basic unit of life.")]


def test_question_is_a_plain_dataclass():
    q = Question(id="x", question="q?", gold_answers=["a"], docs=["d"], gold_indices=[0])
    assert (q.id, q.docs) == ("x", ["d"])
