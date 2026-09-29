"""CLI: laya-compact --budget 4000 --query "..." docs.jsonl

Each input line is a document: either plain text or a JSON object with a "text"
field. Prints the compaction result as JSON on stdout.
"""

import argparse
import json
import sys
from typing import List, Optional

from laya_compactor.core import compact


def _doc_from_line(line: str) -> str:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return line
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return parsed["text"]
    if isinstance(parsed, str):
        return parsed
    return line


def _read_docs(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [_doc_from_line(line.rstrip("\n")) for line in fh if line.strip()]


def main(argv: Optional[List[str]] = None, agent=None) -> int:
    parser = argparse.ArgumentParser(
        prog="laya-compact",
        description="Compact retrieved documents to a token budget using batched relevance scoring.",
    )
    parser.add_argument("--budget", type=int, required=True,
                        help="maximum number of tokens to keep")
    parser.add_argument("--query", required=True, help="the query the docs were retrieved for")
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="drop docs scoring below this relevance level (default: 1.0)")
    parser.add_argument("--rubric", default="default",
                        help="rubric phrasing variant: default, v2_question_first, v3_needle")
    parser.add_argument("docs", help="JSONL file with one document per line")
    args = parser.parse_args(argv)

    try:
        docs = _read_docs(args.docs)
    except OSError as exc:
        print(f"laya-compact: cannot read {args.docs}: {exc}", file=sys.stderr)
        return 2

    result = compact(
        args.query,
        docs,
        args.budget,
        agent=agent,
        rubric=args.rubric,
        min_score=args.min_score,
    )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
