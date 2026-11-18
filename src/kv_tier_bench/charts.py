"""Charts from results JSONL. Only valid points are plotted; invalid ones are counted in the subtitle."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .cost import summarize  # noqa: E402


def ttft_by_tier(rows: list[dict], stage: str, model: str) -> dict[str, list[tuple[int, float]]]:
    """Median read time for each (tier, prefix length), over the reps. Recompute is the fs deployment's cold
    writes; the single recompute read is only a cross-check, in `kvtb cost`."""
    s = summarize(rows, stage)
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (m, tier, n), t in sorted(s.read_s.items()):
        if m == model and tier != "recompute":
            out[tier].append((n, t))
    for (m, dep, n), t in sorted(s.recompute_s.items()):
        if m == model and dep == "fs":
            out["recompute"].append((n, t))
    return dict(out)


def plot_ttft(rows: list[dict], stage: str, model: str, out: str | Path) -> Path:
    series = ttft_by_tier(rows, stage, model)
    invalid = sum(1 for r in rows if r.get("stage") == stage and r.get("model") == model and not r.get("valid"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tier, pts in series.items():
        xs, ys = zip(*pts)
        ax.plot(xs, [y * 1000 for y in ys], marker="o", label=tier)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("cached prefix (tokens)")
    ax.set_ylabel("TTFT, median (ms)")
    ax.set_title(f"{model} on {stage}\n{invalid} invalid points excluded", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    if series:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no valid points yet", ha="center", va="center", transform=ax.transAxes)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
