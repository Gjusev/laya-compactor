"""Shared test doubles.

The laya Agent sits at the system boundary (instantiating the real one downloads a
checkpoint), so tests inject FakeAgent and a deterministic offline token counter.
"""

import pytest


class FakeAgent:
    """Deterministic laya.Agent stand-in: returns one fixed score per state, in order.

    Scores are consumed state by state; extra scores are ignored so the same fake
    can be reused across calls.
    """

    def __init__(self, scores):
        self.scores = list(scores)
        self.batch_calls = []

    def predict_batch(self, states, questions):
        self.batch_calls.append((list(states), questions))
        answers = [
            {"answers": {"relevance": {"type": "score", "score": score}}}
            for score in self.scores[: len(states)]
        ]
        if len(answers) < len(states):  # repeat the last score for any overflow
            answers.extend([answers[-1]] * (len(states) - len(answers)))
        return answers


class ScriptedAgent(FakeAgent):
    """FakeAgent that returns a different score list per predict_batch call.

    Calls map 1:1 to rubric variants in run_sensitivity, so the i-th call
    returns the i-th score list.
    """

    def __init__(self, calls):
        super().__init__(calls[0] if calls else [])
        self.calls = list(calls)
        self.seen = 0

    def predict_batch(self, states, questions):
        self.batch_calls.append((list(states), questions))
        scores = self.calls[min(self.seen, len(self.calls) - 1)]
        self.seen += 1
        return [
            {"answers": {"relevance": {"type": "score", "score": score}}}
            for score in scores[: len(states)]
        ]


@pytest.fixture
def fake_agent():
    """Factory: fake_agent([score_per_doc_in_input_order])."""

    def make(scores):
        return FakeAgent(scores)

    return make


@pytest.fixture
def word_counter():
    """Deterministic offline token proxy: whitespace word count."""

    def count(text):
        return len(text.split())

    return count
