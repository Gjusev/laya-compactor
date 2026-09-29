.PHONY: test eval smoke

test:
	uv run pytest

# Full table, from scratch: datasets -> 4 policies -> LLM judge -> table.md.
# Needs OPENAI_API_KEY (or an OpenAI-compatible OPENAI_BASE_URL).
eval:
	uv run python -m laya_compactor.eval.run_eval --n 200

# Offline smoke: real datasets + real laya checkpoint, no LLM calls.
smoke:
	uv run python -m laya_compactor.eval.run_eval --n 3 --no-llm --out eval/results-smoke
