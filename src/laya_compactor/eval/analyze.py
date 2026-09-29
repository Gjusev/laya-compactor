"""Summaries and the README-shaped metrics table from result rows."""

from typing import Dict, List, Optional, Tuple

from laya_compactor.eval.metrics import cost_per_1k, p50

POLICY_ORDER = ["full", "head_truncate", "tail_truncate", "laya_compactor"]


def summarize(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Aggregate result rows per (dataset, policy)."""
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["policy"]), []).append(r)

    out = {}
    for key, group in grouped.items():
        n = len(group)
        verdicts = [r["judge"] for r in group if r.get("judge") is not None]
        win_rate = None
        if verdicts:
            wins = verdicts.count("candidate")
            ties = verdicts.count("tie")
            win_rate = (wins + 0.5 * ties) / len(verdicts)
        latencies = [r["compaction_ms"] for r in group if r["compaction_ms"] is not None]
        golds = [g for g in (r.get("gold_kept") for r in group) if g is not None]
        out[key] = {
            "n": n,
            "em": sum(r["em"] for r in group) / n,
            "avg_input_tokens": sum(r["input_tokens"] for r in group) / n,
            "avg_output_tokens": sum(r["output_tokens"] for r in group) / n,
            "avg_kept_tokens": sum(r.get("kept_tokens", 0) for r in group) / n,
            "gold_kept": sum(golds) / len(golds) if golds else None,
            "judge_errors": sum(1 for r in group if r.get("judge_error")),
            "win_rate_vs_full": win_rate,
            "judge_wins": verdicts.count("candidate"),
            "judge_ties": verdicts.count("tie"),
            "judge_losses": verdicts.count("reference"),
            "compaction_p50_ms": p50(latencies) if latencies else None,
        }
    return out


def render_table(summaries: Dict[Tuple[str, str], dict],
                 price_input_per_m: float, price_output_per_m: float) -> str:
    """Markdown table: metrics as rows, one column per policy, per dataset."""
    datasets = sorted({ds for ds, _ in summaries})
    blocks = []
    for dataset in datasets:
        policies = [p for p in POLICY_ORDER if (dataset, p) in summaries]
        if not policies:
            continue
        header = "| Metric | " + " | ".join(policies) + " |"
        sep = "|---" * (len(policies) + 1) + "|"

        def cells(fn):
            return " | ".join(
                _fmt(fn(summaries[(dataset, p)])) for p in policies)

        lines = [f"### {dataset}", "", header, sep]
        lines.append(f"| Avg input tokens | {cells(lambda s: round(s['avg_input_tokens']))} |")
        lines.append(f"| Avg output tokens | {cells(lambda s: round(s['avg_output_tokens']))} |")
        lines.append(f"| Exact match | {cells(lambda s: s['em'])} |")
        lines.append(f"| LLM-judge win-rate vs full | "
                     + " | ".join(
                         _fmt(summaries[(dataset, p)]["win_rate_vs_full"])
                         if p != "full" else "100% (ref)"
                         for p in policies) + " |")
        lines.append(f"| Gold docs kept | {cells(lambda s: s['gold_kept'])} |")
        lines.append(f"| Compaction latency p50 (ms) | "
                     + " | ".join(
                         _fmt(summaries[(dataset, p)]["compaction_p50_ms"])
                         for p in policies) + " |")
        lines.append(f"| Cost per 1,000 questions ($, {price_input_per_m}/M in,"
                     f" {price_output_per_m}/M out) | "
                     + " | ".join(
                         _fmt(cost_per_1k(summaries[(dataset, p)]["avg_input_tokens"],
                                          summaries[(dataset, p)]["avg_output_tokens"],
                                          price_input_per_m, price_output_per_m))
                         for p in policies) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)
