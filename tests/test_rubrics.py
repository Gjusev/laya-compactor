"""Rubric variants: all must be valid laya score questions with the same
4-level scale, and each variant must phrase it differently."""

import pytest

from laya_compactor.rubrics import QUESTION_ID, RELEVANCE_RUBRIC, RUBRIC_VARIANTS, get_rubric


def test_all_variants_are_valid_score_questions():
    assert len(RUBRIC_VARIANTS) == 3
    for name, rubric in RUBRIC_VARIANTS.items():
        assert rubric["type"] == "score", name
        assert isinstance(rubric["instructions"], str) and rubric["instructions"], name
        assert len(rubric["criteria"]) == 4, name
        assert all(isinstance(c, str) and c for c in rubric["criteria"]), name


def test_variant_instructions_are_genuinely_different():
    instructions = [r["instructions"] for r in RUBRIC_VARIANTS.values()]
    assert len(set(instructions)) == 3


def test_variant_criteria_keep_the_same_ordinal_levels():
    # every variant's levels run irrelevant -> background -> relevant -> essential
    for name, rubric in RUBRIC_VARIANTS.items():
        levels = [c.split(":")[0].strip() for c in rubric["criteria"]]
        assert levels == ["irrelevant", "background", "relevant", "essential"], name


def test_get_rubric_resolves_default_name_dict_and_rejects_unknown():
    assert get_rubric() is RELEVANCE_RUBRIC
    assert get_rubric("default") is RELEVANCE_RUBRIC
    assert get_rubric(RELEVANCE_RUBRIC) is RELEVANCE_RUBRIC  # dict passthrough
    with pytest.raises(ValueError, match="unknown rubric"):
        get_rubric("nope")


def test_question_id_is_relevance():
    assert QUESTION_ID == "relevance"
