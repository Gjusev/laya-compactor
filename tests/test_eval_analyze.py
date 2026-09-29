"""Analyze: per-(dataset, policy) summaries and the README metrics table."""

from laya_compactor.eval.analyze import render_table, summarize


def row(dataset, policy, em=1, inp=1000, out=100, judge=None, ms=None, gold=1.0):
    return {"dataset": dataset, "policy": policy, "em": em, "input_tokens": inp,
            "output_tokens": out, "judge": judge, "compaction_ms": ms,
            "gold_kept": gold, "kept_tokens": inp // 2, "qid": "q", "answer": "a",
            "question": "?", "gold_answers": ["a"], "retrieved_docs": 4,
            "kept_docs": 2, "error": None}


ROWS = [
    row("hotpotqa", "full", em=1, inp=4000),
    row("hotpotqa", "head_truncate", em=0, inp=1000, judge="reference"),
    row("hotpotqa", "tail_truncate", em=1, inp=1000, judge="tie"),
    row("hotpotqa", "laya_compactor", em=1, inp=1000, judge="candidate", ms=35.0),
]


def test_summarize_groups_by_dataset_and_policy():
    s = summarize(ROWS)

    assert set(s) == {("hotpotqa", "full"), ("hotpotqa", "head_truncate"),
                      ("hotpotqa", "tail_truncate"), ("hotpotqa", "laya_compactor")}
    compactor = s[("hotpotqa", "laya_compactor")]
    assert compactor["em"] == 1.0
    assert compactor["avg_input_tokens"] == 1000
    assert compactor["compaction_p50_ms"] == 35.0
    full = s[("hotpotqa", "full")]
    assert full["compaction_p50_ms"] is None


def test_summarize_win_rate_counts_half_credit_for_ties():
    s = summarize(ROWS)
    # laya_compactor: 1 candidate win -> 1.0; head: 1 loss -> 0.0;
    # tail: 1 tie -> 0.5; full is the reference, no verdicts.
    assert s[("hotpotqa", "laya_compactor")]["win_rate_vs_full"] == 1.0
    assert s[("hotpotqa", "head_truncate")]["win_rate_vs_full"] == 0.0
    assert s[("hotpotqa", "tail_truncate")]["win_rate_vs_full"] == 0.5
    assert s[("hotpotqa", "full")]["win_rate_vs_full"] is None


def test_summarize_reports_verdict_counts():
    s = summarize(ROWS)[("hotpotqa", "laya_compactor")]
    assert s["judge_wins"] == 1 and s["judge_ties"] == 0 and s["judge_losses"] == 0


def test_render_table_prints_one_row_per_policy_with_costs():
    table = render_table(summarize(ROWS),
                         price_input_per_m=0.15, price_output_per_m=0.60)

    assert "### hotpotqa" in table
    assert "| Metric |" in table
    for policy in ("full", "head_truncate", "tail_truncate", "laya_compactor"):
        assert policy in table
    # $/1k for full: (4000*0.15 + 100*0.60)/1e6 * 1000 = 0.66
    assert "0.66" in table


def test_summarize_without_judge_verdicts_marks_win_rate_missing():
    rows = [row("squad", "full"), row("squad", "laya_compactor", judge=None)]
    s = summarize(rows)
    assert s[("squad", "laya_compactor")]["win_rate_vs_full"] is None


def test_summarize_tolerates_rows_without_a_judge_field():
    # --no-llm mode skips the judge pass entirely: rows carry no judge key
    rows = [row("squad", "full"), row("squad", "laya_compactor", ms=30.0)]
    for r in rows:
        del r["judge"]
    s = summarize(rows)

    assert s[("squad", "laya_compactor")]["win_rate_vs_full"] is None
    assert s[("squad", "laya_compactor")]["compaction_p50_ms"] == 30.0


def test_summarize_skips_questions_without_gold_marks_and_counts_judge_errors():
    rows = [row("squad", "laya_compactor", gold=None),
            row("squad", "laya_compactor", gold=1.0)]
    rows[0]["judge_error"] = "boom"

    s = summarize(rows)

    # only the question that has gold marks counts toward gold_kept
    assert s[("squad", "laya_compactor")]["gold_kept"] == 1.0
    assert s[("squad", "laya_compactor")]["judge_errors"] == 1
