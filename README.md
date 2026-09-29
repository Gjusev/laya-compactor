# laya-compactor

> Context compaction for RAG and agents: score every retrieved chunk in one shared forward pass and cut what does not matter before it reaches the LLM.

Status: early development. Built on [laya](https://github.com/NandhaKishorM/laya),
the open-source System 1 decision engine (Apache 2.0).

## Why

- Context is the dominant cost of RAG. Context compaction is the #2 category by density in the Jev ecosystem (the leading project has 2.7k stars); nothing equivalent exists for the open-source stack.
- laya's `predict_batch` scores a whole retrieval batch in one forward pass, so compaction adds milliseconds, not seconds.

## Roadmap

- [ ] Core: `compact(query, docs, budget)` with batched relevance scoring and budget-aware selection
- [ ] Rubric sensitivity study: three phrasings over a small hand-labeled set, variance published
- [ ] Eval harness: RAG over public QA datasets comparing full context vs head/tail truncation vs laya-compactor
- [ ] Integrations: LangChain retriever wrapper and LlamaIndex node postprocessor
- [ ] CLI for batch compaction of JSONL documents

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
