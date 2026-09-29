"""Run the eval: datasets x policies, one row per (question, policy).

Full run (needs OPENAI_API_KEY or an OpenAI-compatible OPENAI_BASE_URL):

    python -m laya_compactor.eval.run_eval --n 200

Offline smoke (real datasets + real laya checkpoint, no LLM calls; EM and
judge columns are meaningless in this mode):

    python -m laya_compactor.eval.run_eval --n 3 --no-llm

Outputs under --out (default eval/results/): rows.jsonl, summary.json and
table.md (the table `make eval` reproduces).
"""

import argparse
import json
import sys
from pathlib import Path

from laya_compactor.eval.analyze import render_table, summarize
from laya_compactor.eval.datasets import LOADERS
from laya_compactor.eval.generate import OpenAIChat, OpenAIGenerator, OpenAIJudge
from laya_compactor.eval.pipeline import run_generation_pass, run_judge_pass

# Listed prices for the default model (USD per 1M tokens); pass yours with
# --price-input / --price-output. Costs are computed, never invented.
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PRICE_INPUT = 0.15
DEFAULT_PRICE_OUTPUT = 0.60


class _NullGenerator:
    """--no-llm mode: no generation, token counts are zero."""

    def generate(self, question, docs):
        return {"answer": "", "input_tokens": 0, "output_tokens": 0}


def _tiktoken_counter():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="laya-eval",
                                     description="RAG compaction eval: 4 context policies, same generator.")
    parser.add_argument("--datasets", default="hotpotqa,squad",
                        help=f"comma-separated: {', '.join(LOADERS)}")
    parser.add_argument("--n", type=int, default=200, help="questions per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20, help="BM25 top-k retrieval")
    parser.add_argument("--budget", type=int, default=1000,
                        help="token budget shared by the three budgeted policies")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--price-input", type=float, default=DEFAULT_PRICE_INPUT,
                        help="USD per 1M input tokens")
    parser.add_argument("--price-output", type=float, default=DEFAULT_PRICE_OUTPUT,
                        help="USD per 1M output tokens")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip generation and judging (token/latency/gold-kept only)")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args(argv)

    token_counter = _tiktoken_counter()
    agent = None
    if not args.no_llm:
        import laya

        agent = laya.load()
        chat = OpenAIChat(model=args.model)
        generator = OpenAIGenerator(chat=chat)
        judge = OpenAIJudge(chat=chat)
    else:
        generator = _NullGenerator()
        judge = None

    policies = ["full", "head_truncate", "tail_truncate", "laya_compactor"]
    all_rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"[{name}] loading {args.n} questions (seed {args.seed})...", file=sys.stderr)
        questions = LOADERS[name](args.n, args.seed)
        print(f"[{name}] running {len(policies)} policies on {len(questions)} questions...",
              file=sys.stderr)
        rows = run_generation_pass(questions, policies, args.budget, args.k,
                                   generator=generator, agent=agent,
                                   token_counter=token_counter)
        if judge is not None:
            rows = run_judge_pass(rows, judge=judge)
        for r in rows:
            r["dataset"] = name
        all_rows.extend(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {f"{ds}|{p}": s for (ds, p), s in summarize(all_rows).items()}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = render_table(summarize(all_rows), args.price_input, args.price_output)
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    from laya_compactor.eval.plot import render_scatter_svg

    (out_dir / "summary.svg").write_text(
        render_scatter_svg(summary), encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
