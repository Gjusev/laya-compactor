"""Pipeline seam: retrieve -> apply policy -> generate -> score, all injected."""

from laya_compactor.core import compact  # noqa: F401  (proves import parity)
from laya_compactor.eval.datasets import Question
from laya_compactor.eval.pipeline import run_generation_pass, run_judge_pass


def word_counter(text):
    return len(text.split())


def make_question():
    docs = ["The Eiffel Tower is in Paris France.",
            "Photosynthesis happens in plant leaves.",
            "The tower was designed by Gustave Eiffel the engineer."]
    return Question(id="q1", question="Who designed the Eiffel Tower?",
                    gold_answers=["Gustave Eiffel"],
                    docs=docs, gold_indices=[2])


class ScoreGoldHighAgent:
    """Scores the doc containing 'Gustave' as essential, others as irrelevant."""

    def __init__(self):
        self.calls = 0

    def predict_batch(self, states, questions):
        self.calls += 1
        return [{"answers": {"relevance": {"type": "score",
                                           "score": 3.0 if "Gustave" in s["document"] else 0.2}}}
                for s in states]


class ScriptedGenerator:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate(self, question, docs):
        self.prompts.append(list(docs))
        return {"answer": self.answers.pop(0), "input_tokens": 10 * len(docs),
                "output_tokens": 5}


class ScriptedJudge:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def judge(self, question, gold_answers, reference_answer, candidate_answer):
        return self.verdicts.pop(0)


COMPONENTS = dict(agent=None, token_counter=word_counter)


def test_generation_pass_runs_every_policy_per_question():
    q = make_question()
    gen = ScriptedGenerator(["full answer", "head answer", "tail answer",
                             "Gustave Eiffel"])

    rows = run_generation_pass([q], policies=["full", "head_truncate",
                                              "tail_truncate", "laya_compactor"],
                               budget=100, k=3, generator=gen,
                               agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)

    assert len(rows) == 4
    by_policy = {r["policy"]: r for r in rows}
    assert set(by_policy) == {"full", "head_truncate", "tail_truncate", "laya_compactor"}
    # every row scored exact match, counted tokens, and reported what was kept
    for row in rows:
        assert row["em"] in (0, 1)
        assert row["input_tokens"] > 0
        assert row["kept_docs"] >= 0 and row["retrieved_docs"] == 3
    assert by_policy["laya_compactor"]["em"] == 1  # gold doc survived + scripted answer


def test_compactor_policy_keeps_only_what_scores_and_measures_latency():
    q = make_question()
    gen = ScriptedGenerator(["Gustave Eiffel"])

    rows = run_generation_pass([q], policies=["laya_compactor"], budget=100, k=3,
                               generator=gen, agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)
    row = rows[0]

    assert row["kept_docs"] == 1
    assert row["gold_kept"] == 1.0
    assert row["compaction_ms"] is not None and row["compaction_ms"] >= 0
    assert gen.prompts[0] == ["The tower was designed by Gustave Eiffel the engineer."]


def test_full_policy_has_no_compaction_latency_and_keeps_all():
    q = make_question()
    gen = ScriptedGenerator(["answer"])

    rows = run_generation_pass([q], policies=["full"], budget=1,  # budget ignored
                               k=3, generator=gen, agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)
    row = rows[0]

    assert row["kept_docs"] == 3 and row["gold_kept"] == 1.0
    assert row["compaction_ms"] is None


def test_budget_cut_drops_gold_doc_and_reports_it():
    q = make_question()
    gen = ScriptedGenerator(["I do not know"])

    rows = run_generation_pass([q], policies=["laya_compactor"], budget=1, k=3,
                               generator=gen, agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)
    row = rows[0]

    # budget of 1 word cannot hold the 9-word gold doc; nothing is kept
    assert row["kept_docs"] == 0
    assert row["gold_kept"] == 0.0


def test_judge_pass_compares_against_the_full_reference_answer():
    q = make_question()
    gen = ScriptedGenerator(["full answer", "head answer"])
    rows = run_generation_pass([q], policies=["full", "head_truncate"], budget=100,
                               k=3, generator=gen, agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)
    judge = ScriptedJudge(["candidate"])

    rows = run_judge_pass(rows, judge=judge)

    by_policy = {r["policy"]: r for r in rows}
    assert by_policy["full"]["judge"] is None  # the reference is not judged against itself
    assert by_policy["head_truncate"]["judge"] == "candidate"
    assert by_policy["head_truncate"]["judge_error"] is None


def test_judge_failure_marks_the_row_instead_of_killing_the_pass():
    q = make_question()
    gen = ScriptedGenerator(["full answer", "head answer"])
    rows = run_generation_pass([q], policies=["full", "head_truncate"], budget=100,
                               k=3, generator=gen, agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)

    class BrokenJudge:
        def judge(self, *args, **kwargs):
            raise ValueError("judge returned unparseable verdict: 'banana'")

    rows = run_judge_pass(rows, judge=BrokenJudge())

    by_policy = {r["policy"]: r for r in rows}
    assert by_policy["head_truncate"]["judge"] is None
    assert "unparseable" in by_policy["head_truncate"]["judge_error"]


def test_question_without_gold_indices_reports_gold_kept_none():
    q = Question(id="q2", question="?", gold_answers=["a"],
                 docs=["doc one two"], gold_indices=[])
    gen = ScriptedGenerator(["a"])

    rows = run_generation_pass([q], policies=["full"], budget=10, k=3,
                               generator=gen, agent=ScoreGoldHighAgent(),
                               token_counter=word_counter)

    assert rows[0]["gold_kept"] is None  # no gold marks: not counted as 0.0
