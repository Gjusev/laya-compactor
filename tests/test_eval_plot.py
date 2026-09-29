"""Cost/quality SVG chart from a summary (dependency-free, deterministic)."""

import re

from laya_compactor.eval.plot import render_scatter_svg


SUMMARY = {
    "hotpotqa|full": {"avg_input_tokens": 4000.0, "em": 0.40},
    "hotpotqa|head_truncate": {"avg_input_tokens": 1000.0, "em": 0.30},
    "hotpotqa|laya_compactor": {"avg_input_tokens": 1000.0, "em": 0.41},
    "squad|full": {"avg_input_tokens": 3200.0, "em": 0.55},
    "squad|laya_compactor": {"avg_input_tokens": 900.0, "em": 0.54},
}


def test_renders_one_point_per_policy_with_axes_and_legend():
    svg = render_scatter_svg(SUMMARY)

    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    for policy in ("full", "head_truncate", "laya_compactor"):
        assert f">{policy}<" in svg
    assert "hotpotqa" in svg and "squad" in svg  # legend
    assert "avg input tokens" in svg and "exact match" in svg


def test_leftward_points_mean_cheaper_and_marks_are_deterministic():
    svg1 = render_scatter_svg(SUMMARY)
    svg2 = render_scatter_svg(SUMMARY)
    assert svg1 == svg2  # no randomness: same input, same picture

    # circles are emitted in sorted summary order: hotpotqa|full is first
    # and hotpotqa|laya_compactor third; cheaper context sits further left
    xs = [float(x) for x in re.findall(r'<circle cx="([0-9.]+)"', svg1)]
    assert len(xs) == 5
    assert xs[2] < xs[0]  # laya_compactor (1000 tok) left of full (4000 tok)
