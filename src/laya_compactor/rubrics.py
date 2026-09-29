"""Relevance rubrics: the ordinal scale laya scores each retrieved doc against.

Phrasing sensitivity is a documented failure mode of rubric-based scoring, so the
default rubric ships with two alternative phrasings of the same scale; the
sensitivity runner measures how much they disagree on a labeled set.
"""

QUESTION_ID = "relevance"

RELEVANCE_RUBRIC = {
    "type": "score",
    "instructions": "How relevant is this document to answering the question?",
    "criteria": [
        "irrelevant: about a different topic",
        "background: same topic but does not help answer",
        "relevant: contains information that helps answer",
        "essential: directly answers part of the question",
    ],
}

RUBRIC_VARIANTS = {
    "default": RELEVANCE_RUBRIC,
    "v2_question_first": {
        "type": "score",
        "instructions": "To answer the question, how useful is the information in this document?",
        "criteria": [
            "irrelevant: the document discusses an unrelated topic",
            "background: the document is on the same topic but adds nothing toward an answer",
            "relevant: the document provides information that supports answering",
            "essential: the document states information that directly answers the question",
        ],
    },
    "v3_needle": {
        "type": "score",
        "instructions": "Does this document contain the facts needed to answer the question, and how close does it get?",
        "criteria": [
            "irrelevant: no connection to the question",
            "background: related to the question without containing any needed fact",
            "relevant: contains some of the facts needed to answer",
            "essential: contains the specific fact or facts the question asks for",
        ],
    },
}


def get_rubric(name=None):
    """Return a rubric definition: a variant name, None for the default, or a dict passthrough."""
    if name is None:
        return RELEVANCE_RUBRIC
    if isinstance(name, dict):
        return name
    try:
        return RUBRIC_VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown rubric {name!r}; available: {', '.join(sorted(RUBRIC_VARIANTS))}"
        ) from None
