"""Sensitivity runner: how much do the rubric phrasings disagree with the gold
labels and with each other, on a labeled query/doc set."""

import pytest

from laya_compactor.rubrics import RUBRIC_VARIANTS
from laya_compactor.sensitivity import load_dataset, run_sensitivity

from conftest import ScriptedAgent

# Two rows, gold labels 2 and 0. Each variant's scores are chosen so every
# metric below has a hand-computed value.
ROWS = [
    {"id": "r1", "query": "q1", "doc": "d1", "label": 2},
    {"id": "r2", "query": "q2", "doc": "d2", "label": 0},
]
SCRIPTED = [  # one score list per rubric variant
    [2.0, 0.0],  # default: exact
    [1.6, 0.4],
    [2.4, 0.2],
]


def test_run_sensitivity_scores_one_variant_per_batch_call():
    agent = ScriptedAgent(SCRIPTED)

    run_sensitivity(ROWS, agent=agent)

    assert len(agent.batch_calls) == len(RUBRIC_VARIANTS)
    seen_instructions = [c[1]["relevance"]["instructions"] for c in agent.batch_calls]
    assert seen_instructions == [v["instructions"] for v in RUBRIC_VARIANTS.values()]


def test_run_sensitivity_reports_per_variant_gold_agreement_and_mae():
    report = run_sensitivity(ROWS, agent=ScriptedAgent(SCRIPTED))

    per_variant = report.per_variant
    assert set(per_variant) == set(RUBRIC_VARIANTS)
    # all three variants round to the gold labels here
    assert per_variant["default"]["agreement"] == pytest.approx(1.0)
    assert per_variant["default"]["mae"] == pytest.approx(0.0)
    assert per_variant["v2_question_first"]["mae"] == pytest.approx(0.4)  # (0.4+0.4)/2
    assert per_variant["v3_needle"]["mae"] == pytest.approx(0.3)  # (0.4+0.2)/2
    assert per_variant["default"]["mean_score"] == pytest.approx(1.0)  # (2.0+0.0)/2


def test_run_sensitivity_reports_inter_variant_spread():
    report = run_sensitivity(ROWS, agent=ScriptedAgent(SCRIPTED))

    # row 1 spreads 1.6..2.4 (0.8), row 2 spreads 0.0..0.4 (0.4) -> mean 0.6
    assert report.spread_mean == pytest.approx(0.6)
    # every variant rounds to the same level on both rows
    assert report.full_agreement == pytest.approx(1.0)


def test_run_sensitivity_counts_partial_agreement():
    # variants disagree on the rounded level of row 1 (2 vs 3 vs 2)
    report = run_sensitivity(ROWS, agent=ScriptedAgent([[2.0, 0.0], [2.6, 0.0], [2.0, 0.0]]))

    assert report.full_agreement == pytest.approx(0.5)  # only row 2 agrees across variants


def test_run_sensitivity_with_no_rows_is_all_zeros():
    report = run_sensitivity([], agent=ScriptedAgent([[], [], []]))

    assert report.spread_mean == 0.0
    assert report.full_agreement == 0.0
    assert all(v["n"] == 0 for v in report.per_variant.values())


def test_run_sensitivity_to_dict_is_json_ready():
    report = run_sensitivity(ROWS, agent=ScriptedAgent(SCRIPTED))

    payload = report.to_dict()
    assert set(payload) == {"n", "per_variant", "spread_mean", "full_agreement"}
    assert payload["n"] == 2
    assert payload["per_variant"]["default"]["agreement"] == pytest.approx(1.0)


def test_load_dataset_reads_jsonl_rows(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"id": "a", "query": "q", "doc": "d", "label": 3}\n'
        '{"id": "b", "query": "q2", "doc": "d2", "label": 0}\n',
        encoding="utf-8",
    )

    rows = load_dataset(str(path))

    assert len(rows) == 2
    assert rows[0]["label"] == 3 and rows[1]["id"] == "b"
