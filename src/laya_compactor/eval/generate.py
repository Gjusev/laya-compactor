"""Generation and judging at the LLM boundary.

Both collaborators talk to an OpenAI-compatible chat endpoint through a thin
`chat` object (default OpenAIChat, which uses requests against
OPENAI_BASE_URL/OPENAI_API_KEY) so unit tests script the boundary offline.
Prompts are fixed and identical across all eval configurations.
"""

import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple


GENERATION_SYSTEM = (
    "You answer questions using only the provided context passages. "
    "Answer as briefly as the question allows; if the passages do not "
    "contain the answer, say what is missing in a few words."
)


def build_generation_messages(question: str, docs: Sequence[str]) -> List[dict]:
    passages = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))
    user = (
        f"Context passages:\n\n{passages}\n\n"
        f"Question: {question}\n\nAnswer using only the passages above:"
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": user},
    ]


JUDGE_INSTRUCTIONS = (
    "You are grading two answers to the same question against the gold "
    "answer. Rules: correctness against the gold answer decides; answer "
    "length must not influence you (a shorter correct answer beats a longer "
    "one, an equal one ties); if both are wrong or both equally incomplete, "
    "it is a tie. Reply with exactly one word: candidate, reference, or tie."
)


class OpenAIChat:
    """Minimal OpenAI-compatible chat client (POST /chat/completions)."""

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def complete(self, messages: List[dict], temperature: float) -> Tuple[str, Dict[str, int]]:
        import requests  # deferred: keeps unit-test import time tiny

        # A full eval makes ~1000 calls; retrying transient failures is the
        # difference between one bad minute and losing the run.
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(3):
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages,
                      "temperature": temperature},
                timeout=120,
            )
            if response.status_code in retryable and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            if response.status_code >= 400:
                # keep the provider's error body: it names the real cause
                raise requests.HTTPError(
                    f"{response.status_code}: {response.text[:300]}", response=response)
            data = response.json()
            usage = data.get("usage", {})
            return (data["choices"][0]["message"]["content"],
                    {"prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0)})
        raise RuntimeError("unreachable")


class OpenAIGenerator:
    """Answer a question from retrieved docs with a fixed prompt."""

    def __init__(self, chat: Optional[OpenAIChat] = None):
        self.chat = chat or OpenAIChat()

    def generate(self, question: str, docs: Sequence[str]) -> Dict[str, object]:
        content, usage = self.chat.complete(build_generation_messages(question, docs),
                                            temperature=0.0)
        return {
            "answer": content.strip(),
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        }


class OpenAIJudge:
    """Pairwise judge: candidate answer vs the full-context reference answer."""

    def __init__(self, chat: Optional[OpenAIChat] = None):
        self.chat = chat or OpenAIChat()

    def judge(self, question: str, gold_answers: Sequence[str],
              reference_answer: str, candidate_answer: str) -> str:
        user = (
            f"Question: {question}\n"
            f"Gold answer: {' / '.join(gold_answers)}\n"
            f"Reference answer (full context): {reference_answer}\n"
            f"Candidate answer (compacted context): {candidate_answer}\n\n"
            f"{JUDGE_INSTRUCTIONS}"
        )
        content, _ = self.chat.complete(
            [{"role": "user", "content": user}], temperature=0.0)
        match = re.search(r"\b(candidate|reference|tie)\b", content.strip().lower())
        if not match:
            raise ValueError(f"judge returned unparseable verdict: {content!r}")
        return match.group(1)
