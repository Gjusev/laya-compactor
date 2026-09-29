"""CLI seam: laya-compact --budget N --query "..." docs.jsonl -> JSON on stdout."""

import json

import pytest

from laya_compactor.cli import main


def write_jsonl(tmp_path, lines):
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_cli_prints_ordered_compaction_json(fake_agent, capsys, tmp_path):
    docs = write_jsonl(tmp_path, ['"alpha"', '"beta"', '"gamma"'])
    agent = fake_agent([0.2, 2.9, 1.8])

    code = main(["--budget", "2", "--query", "q", docs], agent=agent)

    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert [k["doc"] for k in out["kept"]] == ["beta", "gamma"]
    assert out["kept"][0]["score"] == pytest.approx(2.9)
    assert out["stats"]["docs_total"] == 3
    assert out["stats"]["docs_kept"] == 2
    assert all("reason" in c for c in out["cut"])


def test_cli_accepts_text_objects_and_plain_lines(fake_agent, capsys, tmp_path):
    docs = write_jsonl(tmp_path, ['{"text": "the rich doc"}', "plain text doc"])
    agent = fake_agent([2.0, 2.0])

    code = main(["--budget", "100", "--query", "q", docs], agent=agent)

    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert [k["doc"] for k in out["kept"]] == ["the rich doc", "plain text doc"]


def test_cli_missing_file_fails_with_nonzero_exit(fake_agent, capsys):
    code = main(["--budget", "10", "--query", "q", "does-not-exist.jsonl"], agent=fake_agent([]))

    assert code == 2
    assert "does-not-exist.jsonl" in capsys.readouterr().err


def test_cli_passes_rubric_and_min_score_options(fake_agent, capsys, tmp_path):
    docs = write_jsonl(tmp_path, ['"a"', '"b"'])
    agent = fake_agent([2.0, 0.9])

    code = main(
        ["--budget", "10", "--query", "q", "--rubric", "v2_question_first",
         "--min-score", "0.5", docs],
        agent=agent,
    )

    assert code == 0
    questions = agent.batch_calls[0][1]
    assert "To answer the question" in questions["relevance"]["instructions"]
    out = json.loads(capsys.readouterr().out)
    assert out["stats"]["docs_kept"] == 2  # 0.9 >= 0.5 passes the lowered bar
