"""Charts from results JSONL. Only valid points are plotted; invalid ones are counted in the subtitle."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .cost import summarize  # noqa: E402
from .load import summarize as summarize_load  # noqa: E402


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


def load_by_tier(rows: list[dict], stage: str, model: str) -> dict[int, dict[str, list[tuple[int, float, float]]]]:
    """For each prefix length: per tier (recompute included), (concurrency, prefix tokens served per second,
    median request TTFT) at every concurrency measured."""
    s = summarize_load(rows, stage)
    out: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (m, tier, n, c), batch_s in sorted(s.batch_s.items()):
        if m == model:
            out[n][tier].append((c, c * n / batch_s, s.ttft_p50_s[(m, tier, n, c)]))
    return {n: dict(t) for n, t in sorted(out.items())}


def plot_load(rows: list[dict], stage: str, model: str, out: str | Path) -> Path:
    """One column per prefix length: tokens served per second (top) and median request TTFT (bottom) against
    concurrency, one line per tier."""
    data = load_by_tier(rows, stage, model)
    cols = max(1, len(data))
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6), squeeze=False, sharex=True)
    for j, (n, tiers) in enumerate(data.items()):
        top, bottom = axes[0][j], axes[1][j]
        for tier, pts in tiers.items():
            cs, rate, ttft = zip(*pts)
            top.plot(cs, rate, marker="o", label=tier)
            bottom.plot(cs, [t * 1000 for t in ttft], marker="o", label=tier)
        top.set_title(f"{n // 1024}K-token prefixes", fontsize=9)
        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
        bottom.set_xlabel("requests at once")
    axes[0][0].set_ylabel("prefix tokens served / s")
    axes[1][0].set_ylabel("TTFT per request, median (ms)")
    if data:
        axes[0][0].legend(fontsize=7)
    else:
        axes[0][0].text(0.5, 0.5, "no valid batches yet", ha="center", va="center", transform=axes[0][0].transAxes)
    fig.suptitle(f"{model} on {stage}, under load", fontsize=10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
