"""Offline eval metrics: SQuAD-style exact match, latency percentiles, cost."""

import re
import string
import statistics
from typing import List, Sequence


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD v1 style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> int:
    """1 if the normalized prediction equals any normalized gold answer, else 0."""
    if not prediction.strip():
        return 0
    pred = _normalize(prediction)
    return int(any(pred == _normalize(gold) for gold in gold_answers))


def p50(values: Sequence[float]) -> float:
    """Median of timings; 0.0 for an empty run so tables stay printable."""
    return float(statistics.median(values)) if values else 0.0


def cost_per_1k(avg_input: float, avg_output: float,
                price_input_per_m: float, price_output_per_m: float) -> float:
    """Dollars per 1,000 questions from average per-question token counts.

    Prices are per one million tokens; pass your provider's listed prices.
    """
    per_question = (avg_input * price_input_per_m + avg_output * price_output_per_m) / 1e6
    return per_question * 1000
