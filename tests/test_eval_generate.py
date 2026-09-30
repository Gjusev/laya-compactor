"""LLM boundary: fixed prompts for generation and judging, injectable chat client."""

import pytest

from laya_compactor.eval.generate import (
    OpenAIGenerator,
    OpenAIJudge,
    build_generation_messages,
)


class FakeChat:
    """Scripted stand-in for the OpenAI-compatible chat client."""

    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 20}
        self.calls = []

    def complete(self, messages, temperature):
        self.calls.append({"messages": messages, "temperature": temperature})
        return self.content, self.usage


def test_generation_prompt_contains_numbered_passages_and_question():
    messages = build_generation_messages("Who built the tower?", ["passage one", "passage two"])

    joined = " ".join(m["content"] for m in messages)
    assert "Who built the tower?" in joined
    assert "[1] passage one" in joined and "[2] passage two" in joined
    assert messages[0]["role"] == "system"


def test_generator_returns_answer_and_token_usage():
    chat = FakeChat("Gustave Eiffel")
    gen = OpenAIGenerator(chat=chat)

    out = gen.generate("Who built the tower?", ["Some passage."])

    assert out["answer"] == "Gustave Eiffel"
    assert out["input_tokens"] == 100
    assert out["output_tokens"] == 20
    # generation is deterministic by default
    assert chat.calls[0]["temperature"] == 0.0


def test_judge_prompt_mentions_gold_reference_and_candidate_answers():
    chat = FakeChat("candidate")
    judge = OpenAIJudge(chat=chat)

    verdict = judge.judge(question="Q?", gold_answers=["Paris"],
                          reference_answer="Paris", candidate_answer="Paris, France")

    assert verdict == "candidate"
    prompt = chat.calls[0]["messages"][-1]["content"]
    for needle in ("Paris", "Paris, France", "length"):
        assert needle in prompt, needle
    assert chat.calls[0]["temperature"] == 0.0


def test_judge_parses_each_verdict_word_and_rejects_garbage():
    for word, expected in [("candidate", "candidate"), ("REFERENCE", "reference"),
                           (" tie.", "tie")]:
        assert OpenAIJudge(chat=FakeChat(word)).judge(
            "Q?", ["a"], "ref", "cand") == expected
    with pytest.raises(ValueError, match="judge"):
        OpenAIJudge(chat=FakeChat("banana")).judge("Q?", ["a"], "ref", "cand")


def test_chat_retries_transient_failures_then_succeeds(monkeypatch):
    import requests

    import laya_compactor.eval.generate as gen_mod

    calls = []

    class Resp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code}")

    good = Resp(200, {"choices": [{"message": {"content": "ok"}}],
                      "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    def fake_post(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            return Resp(429, {})
        return good

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(gen_mod.time, "sleep", lambda s: None)

    chat = gen_mod.OpenAIChat()
    content, usage = chat.complete([{"role": "user", "content": "hi"}], 0.0)

    assert content == "ok"
    assert usage == {"prompt_tokens": 5, "completion_tokens": 1}
    assert len(calls) == 3  # two 429s, then success


def test_chat_does_not_retry_hard_client_errors(monkeypatch):
    import requests

    import laya_compactor.eval.generate as gen_mod

    calls = []

    class Resp:
        status_code = 401
        text = '{"error": "bad key"}'

        def json(self):
            return {}

        def raise_for_status(self):
            raise requests.HTTPError("401")

    def fake_post(*args, **kwargs):
        calls.append(args)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    chat = gen_mod.OpenAIChat()
    with pytest.raises(requests.HTTPError):
        chat.complete([{"role": "user", "content": "hi"}], 0.0)
    assert len(calls) == 1
