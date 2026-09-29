"""Runnable demo: compact a retrieval batch with the real laya checkpoint.

    uv run python examples/compact_demo.py

Downloads the checkpoint on the first run (~1 GB, cached afterwards).
"""

import laya

from laya_compactor import compact

QUERY = "Who invented the telephone?"

DOCS = [
    "Alexander Graham Bell was awarded the first US patent for the telephone in 1876.",
    "The steam locomotive Rocket, built by George Stephenson, won the Rainhill Trials in 1829.",
    "Bell's work on the telephone grew out of his research on hearing and speech, "
    "driven in part by his mother's deafness and his work teaching deaf students.",
    "Elisha Gray filed a caveat for a similar telephone design on the very same day as Bell.",
    "The Berlin Wall fell in 1989, reunifying East and West Germany.",
    "The first transcontinental telephone call was made in 1915, with Bell in New York "
    "repeating his famous summons to his assistant in San Francisco.",
    "Vaccination against smallpox was pioneered by Edward Jenner in 1796.",
    "The carbon microphone invented by Thomas Edison greatly improved the telephone's "
    "audibility and became standard in early telephone handsets.",
]

BUDGET = 90  # word-count tokens: forces real cuts


def main() -> None:
    agent = laya.load()
    result = compact(QUERY, DOCS, budget=BUDGET, agent=agent)

    print(f"\nquery: {QUERY}")
    print(f"budget: {BUDGET} tokens | retrieved: {len(DOCS)} docs "
          f"({result.stats.original_tokens} tokens)\n")

    print("KEPT (relevance order):")
    for s in result.kept:
        print(f"  [{s.score:4.2f}] {s.tokens:3d} tok  {s.doc[:80]}")

    print("\nCUT:")
    for c in result.cut:
        print(f"  [{c.score:4.2f}] {c.tokens:3d} tok  {c.doc[:60]}...")
        print(f"         reason: {c.reason}")

    s = result.stats
    print(f"\nstats: kept {s.docs_kept}/{s.docs_total} docs, "
          f"{s.kept_tokens}/{s.original_tokens} tokens "
          f"({s.savings_pct:.0f}% saved)")


if __name__ == "__main__":
    main()
