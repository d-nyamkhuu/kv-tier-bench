"""Correctness spot check: does a prefix loaded from a tier give the same greedy output as recomputing it?

A cache hit only shows that the lookup succeeded. Wrong KV, or a wrong linear-attention state, can still be
counted as a hit (see LMCache#5600, SGLang#45563). So a few natural-text prompts per (model, tier) are generated greedily (~16 tokens) from the tier and from recompute, and the outputs compared. A mismatch
is looked at by hand; bf16 can flip a near-tie, so one mismatch is not proof of a bug. The runner is not built yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    same: bool
    first_divergence: int | None  # index of the first character that differs


def compare(cached: dict, recomputed: dict) -> Comparison:
    """Takes two /v1/completions responses for the same prompt, both greedy (temperature 0)."""
    a, b = cached["choices"][0]["text"], recomputed["choices"][0]["text"]
    if a == b:
        return Comparison(True, None)
    n = min(len(a), len(b))
    return Comparison(False, next((i for i in range(n) if a[i] != b[i]), n))
