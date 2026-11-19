#!/usr/bin/env python3
"""Draws the figures for docs/report.md from results/report/*.json, each in a light and a dark version.

    uv run python scripts/export-report-data.py   # first, if the results changed
    uv run python scripts/report-figures.py

Writes docs/figures/<name>-light.svg and <name>-dark.svg. The report shows the one matching the reader's GitHub
theme with a <picture> element. Text is stored as outlines, so the figures look the same without the font.
Each tier keeps one color on every figure (fixed order, never cycled).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "report"
OUT = ROOT / "docs" / "figures"

TIERS = ["gpu", "cpu", "p2p", "obj", "obj-std", "fs", "fs-shared"]
LABEL = {
    "gpu": "GPU memory",
    "cpu": "CPU memory",
    "p2p": "Peer VM (p2p)",
    "obj": "Object, Enhanced",
    "obj-std": "Object, Standard",
    "fs": "Network disk",
    "fs-shared": "Shared filesystem",
}
MODEL = {
    "qwen3.8-27b": "Qwen3.8-27B · hybrid",
    "qwen3-vl-32b": "Qwen3-VL-32B · standard",
    "qwen3-8b": "Qwen3-8B · standard",
    "qwen3.5-9b": "Qwen3.5-9B · hybrid",
}

# Categorical slots in fixed order, light and dark steps of the same hues (validated palette, adjacent pairs).
THEMES = {
    "light": {
        "fg": "#1f2328",
        "muted": "#59636e",
        "grid": "#e4e7eb",
        "axis": "#c8ced5",
        "ref": "#818b98",
        "bg": "#ffffff",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
        "pos": "#2a78d6",
        "neg": "#eb6834",
    },
    "dark": {
        "fg": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#262c36",
        "axis": "#3d444d",
        "ref": "#8b949e",
        "bg": "#0d1117",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9"],
        "pos": "#3987e5",
        "neg": "#d95926",
    },
}


def load(name: str) -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text())


def target(p: int) -> int:
    return 2 ** round(math.log2(p))


def plabel(p: int) -> str:
    t = target(p)
    return f"{t // 1024}K" if t >= 1024 else str(t)


def style(th: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 10,
            "svg.fonttype": "path",
            "text.color": th["fg"],
            "axes.labelcolor": th["muted"],
            "axes.edgecolor": th["axis"],
            "xtick.color": th["muted"],
            "ytick.color": th["muted"],
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
        }
    )


def color(th: dict, tier: str) -> str:
    return th["series"][TIERS.index(tier)]


def mix(c: str, bg: str, a: float) -> tuple:
    c, bg = to_rgb(c), to_rgb(bg)
    return tuple(a * x + (1 - a) * y for x, y in zip(c, bg))


def fmt_s(v: float) -> str:
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"


def log_ticks(lo: float, hi: float, mantissas=(1, 3)) -> list[float]:
    out = [m * 10**e for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 2) for m in mantissas]
    return [v for v in out if lo <= v <= hi]


def end_labels(ax, items: list[tuple[str, float, float, str]], th: dict, min_gap: float = 0.075) -> None:
    """Direct labels at line ends, nudged apart in log-y axis fraction so they never overlap."""
    lo, hi = (math.log10(v) for v in ax.get_ylim())
    pts = sorted(((math.log10(y) - lo) / (hi - lo), x, label, c) for label, x, y, c in items)
    for i in range(1, len(pts)):
        if pts[i][0] - pts[i - 1][0] < min_gap:
            pts[i] = (pts[i - 1][0] + min_gap, *pts[i][1:])
    over = pts[-1][0] - 0.98 if pts else 0
    if over > 0:
        pts = [(f - over, *rest) for f, *rest in pts]
    for f, x, label, c in pts:
        ax.annotate(
            label,
            xy=(x, 10 ** (lo + f * (hi - lo))),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color=c,
        )


def lines_panel(ax, series, ref, th, xs_fmt, y_fmt, ylim, xticks) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    yt = log_ticks(*ylim)
    ax.set_yticks(yt, [y_fmt(v) for v in yt])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(xticks, [xs_fmt(v) for v in xticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis="y", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    labels = []
    if ref:
        rx, ry = zip(*ref)
        ax.plot(rx, ry, color=th["ref"], linewidth=1.6, linestyle=(0, (4, 3)))
        labels.append(("Recompute", rx[-1], ry[-1], th["muted"]))
    for tier, pts in series:
        x, y = zip(*pts)
        c = color(th, tier)
        ax.plot(x, y, color=c, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.scatter(x, y, s=22, color=c, edgecolor=th["bg"], linewidth=1.2, zorder=3)
        labels.append((LABEL[tier], x[-1], y[-1], c))
    end_labels(ax, labels, th)


def fig_s1(th: dict, path: Path) -> None:
    s1 = load("s1")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, m in zip(axes, ["qwen3.8-27b", "qwen3-vl-32b"]):
        rows = [r for r in s1 if r["model"] == m]
        series = []
        for t in TIERS:
            pts = sorted((target(r["prefix_tokens"]), r["read_s"]) for r in rows if r["tier"] == t)
            if pts:
                series.append((t, pts))
        by_p: dict[int, list[float]] = {}
        for r in rows:
            by_p.setdefault(target(r["prefix_tokens"]), []).append(r["recompute_s"])
        ref = sorted((p, sorted(v)[len(v) // 2]) for p, v in by_p.items())
        xs = sorted(by_p)
        lines_panel(
            ax, series, ref, th, plabel, lambda v: f"{v:g} s" if v >= 1 else f"{round(v * 1000)} ms", (0.03, 60), xs
        )
        ax.set_title(MODEL[m], loc="left", fontsize=11, color=th["fg"], pad=10, fontweight="bold")
        ax.set_xlim(xs[0] / 1.15, xs[-1] * 1.15)
        ax.set_xlabel("prefix length, tokens", fontsize=9)
    axes[0].set_ylabel("time to first token (log)", fontsize=9)
    fig.tight_layout(w_pad=7)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_s2_throughput(th: dict, path: Path, prefix: int = 32768) -> None:
    s2 = load("s2")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, m in zip(axes, ["qwen3-8b", "qwen3.5-9b"]):
        rows = [r for r in s2 if r["model"] == m and target(r["prefix_tokens"]) == prefix]
        series = []
        for t in TIERS:
            pts = sorted((r["concurrency"], r["tokens_per_s"]) for r in rows if r["tier"] == t)
            if pts:
                series.append((t, pts))
        ref = sorted({r["concurrency"]: r["recompute_tokens_per_s"] for r in rows}.items())
        xs = [c for c, _ in ref]
        lines_panel(
            ax,
            series,
            ref,
            th,
            lambda v: f"{v}" if v > 1 else "1 request",
            lambda v: f"{v / 1e6:g}M/s" if v >= 1e6 else f"{v / 1e3:g}K/s",
            (3e3, 3e6),
            xs,
        )
        ax.set_title(
            f"{MODEL[m]} · {plabel(prefix)}-token prefixes",
            loc="left",
            fontsize=11,
            color=th["fg"],
            pad=10,
            fontweight="bold",
        )
        ax.set_xlim(xs[0] / 1.15, xs[-1] * 1.15)
        ax.set_xlabel("requests read at once", fontsize=9)
    axes[0].set_ylabel("prefix tokens served per second (log)", fontsize=9)
    fig.tight_layout(w_pad=9)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_savings(th: dict, path: Path) -> None:
    s2 = load("s2")
    models = ["qwen3-8b", "qwen3.5-9b"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), gridspec_kw={"hspace": 0.55})
    for ax, m in zip(axes, models):
        rows = [r for r in s2 if r["model"] == m]
        tiers = [t for t in TIERS if any(r["tier"] == t for r in rows)]
        cols = sorted({(r["prefix_tokens"], r["concurrency"]) for r in rows})
        cell = {(r["tier"], r["prefix_tokens"], r["concurrency"]): r for r in rows}
        xpos, x, prev = [], 0.0, None
        for p, _ in cols:
            if prev is not None and p != prev:
                x += 0.45
            xpos.append(x)
            x += 1
            prev = p
        for i, t in enumerate(tiers):
            for (p, c), cx in zip(cols, xpos):
                r = cell.get((t, p, c))
                if not r:
                    continue
                v = r["saving_vs_recompute"]
                fill = mix(th["pos"] if v >= 0 else th["neg"], th["bg"], min(1.0, abs(v)) * 0.72)
                ax.add_patch(plt.Rectangle((cx + 0.04, i + 0.06), 0.92, 0.88, color=fill, linewidth=0))
                txt = ("−" if v < 0 else "") + f"{abs(round(v * 100))}"
                ax.text(cx + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7.6, color=th["fg"])
        ax.set_xlim(-0.05, x + 0.05)
        ax.set_ylim(len(tiers), 0)
        ax.set_yticks([i + 0.5 for i in range(len(tiers))], [LABEL[t] for t in tiers], fontsize=8.5)
        ax.set_xticks([cx + 0.5 for cx in xpos], [f"×{c}" for _, c in cols], fontsize=7.5)
        ax.xaxis.tick_top()
        for p in sorted({p for p, _ in cols}):
            xs = [cx for (q, _), cx in zip(cols, xpos) if q == p]
            ax.text(
                (xs[0] + xs[-1] + 1) / 2,
                -0.95,
                plabel(p),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=th["fg"],
            )
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(MODEL[m], loc="left", fontsize=11, color=th["fg"], pad=38, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_fs_ratio(th: dict, path: Path) -> None:
    s2 = load("s2")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    c = color(th, "fs-shared")
    for ax, m in zip(axes, ["qwen3.5-9b", "qwen3-8b"]):
        fs = {(r["prefix_tokens"], r["concurrency"]): r["batch_s"] for r in s2 if r["model"] == m and r["tier"] == "fs"}
        pts = sorted(
            (r["prefix_tokens"], r["concurrency"], r["batch_s"] / fs[(r["prefix_tokens"], r["concurrency"])])
            for r in s2
            if r["model"] == m and r["tier"] == "fs-shared"
        )
        groups = sorted({p for p, _, _ in pts})
        x, prev, xs, centers = 0.0, None, [], {}
        for p, _, _ in pts:
            if prev is not None and p != prev:
                x += 1.2
            xs.append(x)
            centers.setdefault(p, []).append(x)
            x += 1
            prev = p
        ax.set_yscale("log")
        ax.set_ylim(0.3, 4.2)
        ax.set_yticks([0.5, 1, 2, 3], ["0.5×", "1×", "2×", "3×"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="y", color=th["grid"], linewidth=0.8)
        ax.axhline(1, color=th["ref"], linewidth=1.2)
        for (p, cc, r), xx in zip(pts, xs):
            ax.plot([xx, xx], [1, r], color=c, alpha=0.45, linewidth=1.4)
            ax.scatter([xx], [r], s=30, color=c if r > 1 else th["bg"], edgecolor=c, linewidth=1.5, zorder=3)
        ax.set_xticks([sum(v) / len(v) for v in centers.values()], [plabel(p) for p in groups])
        ax.set_xlim(-0.8, x - 0.2)
        slower = sum(r > 1 for _, _, r in pts)
        ax.set_title(
            f"{MODEL[m]} · slower at {slower} of {len(pts)} points",
            loc="left",
            fontsize=11,
            color=th["fg"],
            pad=10,
            fontweight="bold",
        )
        ax.set_xlabel("prefix length; batches of 1 to 16 left to right", fontsize=9)
    axes[0].set_ylabel("shared fs ÷ disk batch time\n(above 1×: shared slower)", fontsize=9)
    fig.tight_layout(w_pad=4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, th in THEMES.items():
        style(th)
        fig_s1(th, OUT / f"s1-load-time-{mode}.svg")
        fig_s2_throughput(th, OUT / f"s2-throughput-{mode}.svg")
        fig_savings(th, OUT / f"s2-savings-{mode}.svg")
        fig_fs_ratio(th, OUT / f"fs-shared-vs-disk-{mode}.svg")
    for p in sorted(OUT.glob("*.svg")):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
