# laya-compactor

> Context compaction for RAG and agents: score every retrieved chunk in one shared forward pass and cut what does not matter before it reaches the LLM.

Status: early development. Built on [laya](https://github.com/NandhaKishorM/laya),
the open-source System 1 decision engine (Apache 2.0).

## Why

- Context is the dominant cost of RAG. Context compaction is the #2 category by density in the Jev ecosystem (the leading project has 2.7k stars); nothing equivalent exists for the open-source stack.
- laya's `predict_batch` scores a whole retrieval batch in one forward pass, so compaction adds milliseconds, not seconds.

## How it works

1. Every retrieved doc is scored, in **one** `predict_batch` call, against a
   4-level relevance rubric (0 irrelevant → 3 essential).
2. Docs are ranked by score; the highest-scoring docs that fit the token budget
   are kept **verbatim** — nothing is ever rewritten or summarized, only
   deleted (the same delete-don't-rewrite principle as
   [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction),
   the leading Jev-ecosystem compactor — but inverted: they ask N questions
   about one transcript, we ask one question about N docs in a single forward
   pass).
3. Docs scoring below `min_score` are cut even when budget remains; the rest are
   cut lowest-score-first once the budget is exhausted. Every cut doc carries a
   one-line reason.

```python
from laya_compactor import compact

result = compact("Who designed the Eiffel Tower?", retrieved_docs, budget=4000)
result.kept          # docs ordered most relevant first, verbatim
result.cut           # what was dropped, each with .reason
result.stats         # token counts, savings, docs kept/cut
```

`compact()` accepts an injected `agent` (anything exposing
`predict_batch(states, questions)`), an injected `token_counter`, and a `rubric`
variant name. Without an agent it lazily loads the real laya checkpoint
(downloaded from the Hugging Face Hub on first use).

Token counting defaults to a deterministic offline word-count proxy; inject a
real tokenizer when the budget must match a specific provider's counts.

## CLI

```bash
uv pip install -e .
laya-compact --budget 4000 --query "Who designed the Eiffel Tower?" docs.jsonl
```

`docs.jsonl` holds one document per line (plain text, or a JSON object with a
`"text"` field). The result prints as JSON: kept docs in relevance order, cut
docs with reasons, and token stats. Options: `--min-score`, `--rubric
(default|v2_question_first|v3_needle)`.

## Rubric sensitivity

Rubric-based scoring is sensitive to phrasing, so the rubric ships in three
phrasings and the repo publishes a **100-row hand-labeled mini-dataset**
(`src/laya_compactor/data/relevance_mini_dataset.jsonl`; id, query, doc, gold
label 0–3, rationale). Labels follow the convention: 3 states the asked fact,
2 is the exact subject without the fact, 1 is a neighboring subject, 0 is a
different topic. A few queries appear twice on purpose with contrasting docs —
the pair (same query, relevant doc = 3, unrelated doc = 0) tests doc-side
discrimination.

Measure the phrasing sensitivity on your machine (loads the real checkpoint):

```bash
python -m laya_compactor.sensitivity            # packaged dataset
python -m laya_compactor.sensitivity my.jsonl   # your own rows
```

Reported: per-variant agreement with the gold labels and mean absolute error,
mean per-row score spread across phrasings, and the fraction of rows where all
phrasings round to the same level. Measured results are published in the
sensitivity table below — no number in this repo is invented.

Measured on the default English laya checkpoint, CPU, 100 rows (raw report:
`eval/sensitivity_report.json`; rerun `python -m laya_compactor.sensitivity` to
reproduce on your hardware — scoring is deterministic, so numbers repeat):

| Metric (100 rows) | default | v2_question_first | v3_needle |
|---|---|---|---|
| Agreement with gold labels | 0.68 | 0.53 | 0.68 |
| Mean absolute error | 0.43 | 0.47 | 0.42 |
| Mean score spread across variants | 0.24 | | |
| Rows where all variants round to the same level | 62% | | |

The phrasing effect the plan predicted is real and worth knowing: the
question-first phrasing costs 15 points of gold agreement on the same data,
and all three phrasings agree on the rounded level for only 62% of rows. The
default phrasing is kept as the default for that reason; the spread is the
number to watch as the rubric evolves.

## Roadmap

- [x] Core: `compact(query, docs, budget)` with batched relevance scoring and budget-aware selection
- [x] Rubric sensitivity study: three phrasings over a small hand-labeled set, variance published
- [x] CLI for batch compaction of JSONL documents
- [ ] Eval harness: RAG over public QA datasets comparing full context vs head/tail truncation vs laya-compactor
- [ ] Integrations: LangChain retriever wrapper and LlamaIndex node postprocessor

## Honest limitations

- Token counts are word-count estimates by default, not a provider tokenizer.
- A relevance score is calibrated, not a proof: a doc cut at 0.9 might have
  mattered. Tune `min_score` for your risk tolerance.
- v1 is English-first (the default laya checkpoint); multilingual routing is a
  later concern.
- Multi-hop questions can need "background" docs — the eval harness phase
  measures exactly this failure mode instead of hiding it.

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest            # unit tests, fully offline (laya is mocked)
pytest -m slow    # opt-in: loads the real laya checkpoint
```

## License

Apache 2.0. See [LICENSE](LICENSE).
