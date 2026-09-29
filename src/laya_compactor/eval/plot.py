"""Cost/quality chart from an eval summary, as a dependency-free SVG.

Usage: python -m laya_compactor.eval.plot eval/results/summary.json [out.svg]
Each dataset gets one color; x = avg input tokens, y = exact match.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def render_scatter_svg(summary: Dict[str, dict], width: int = 640,
                        height: int = 380) -> str:
    """One point per (dataset, policy): cost on x, exact match on y."""
    entries = []
    for key, s in sorted(summary.items()):
        dataset, policy = key.split("|", 1)
        if s.get("avg_input_tokens") or s.get("em"):
            entries.append((dataset, policy, s["avg_input_tokens"], s["em"]))

    left, right, top, bottom = 70, 20, 20, 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max((e[2] for e in entries), default=1.0) * 1.15 or 1.0

    def px(x):
        return left + (x / max_x) * plot_w

    def py(y):
        return top + (1.0 - y) * plot_h

    datasets = sorted({e[0] for e in entries})
    colors = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(datasets)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#666"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#666"/>',
        f'<text x="{left}" y="{top - 6}" font-size="12">exact match</text>',
        f'<text x="{left + plot_w}" y="{top + plot_h + 34}" font-size="12" text-anchor="end">avg input tokens</text>',
    ]
    for frac in (0.0, 0.5, 1.0):
        y = top + (1.0 - frac) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"'
                     f' stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="10" text-anchor="end">'
                     f'{frac:.1f}</text>')
    parts.append(f'<text x="{left + plot_w}" y="{top + plot_h + 16}" font-size="10" '
                 f'text-anchor="middle">{max_x:.0f}</text>')

    for dataset, policy, tokens, em in entries:
        color = colors[dataset]
        x, y = px(tokens), py(em)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" fill-opacity="0.85"/>')
        parts.append(f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" font-size="10" fill="#111">{policy}</text>')

    for i, dataset in enumerate(datasets):
        x = left + 10 + i * 150
        parts.append(f'<rect x="{x}" y="{height - 14}" width="10" height="10" '
                     f'fill="{colors[dataset]}"/>')
        parts.append(f'<text x="{x + 14}" y="{height - 5}" font-size="11">{dataset}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    summary = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    svg = render_scatter_svg(summary)
    out = Path(argv[1]) if len(argv) > 1 else Path(argv[0]).with_suffix(".svg")
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
