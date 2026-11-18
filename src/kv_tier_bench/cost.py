"""The static cost table: $ per 1M prefix tokens for each tier, against recompute.

Inputs are measured times (from results) and prices (configs/prices.yaml). The arithmetic and its assumptions:

- GPU time: an instance's GPUs are billed while a request waits on a load or a recompute. This is conservative:
  vLLM loads asynchronously and can schedule other work meanwhile, so the table gives an upper bound on
  load cost.
- Storage: the bytes for a prefix (KV for every token, plus one recurrent-state snapshot for hybrid models),
  held for `residency_hours`, at the tier's $/GiB-month. Memory that is part of the GPU VM (HBM, host RAM) is
  priced "included" and costs $0 here. A missing price is an error, never $0.
- Requests and same-region traffic: Enhanced Throughput object storage charges none, so they are left out.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

HOURS_PER_MONTH = 730.0
GIB = 1024**3


@dataclass(frozen=True)
class CostInputs:
    gpu_hour_usd: float  # per GPU
    gpus: int  # GPUs in the instance (its TP size)
    storage_gib_month_usd: float | str  # a price, or "included" (already paid for with the VM)

    def __post_init__(self):
        p = self.storage_gib_month_usd
        if p != "included" and not isinstance(p, (int, float)):
            raise ValueError(f"storage price must be a number or 'included', got {p!r}")


@dataclass(frozen=True)
class Summary:
    """Medians over the results of one stage. Keys are (model, tier, prefix length), or for the store side
    (model, deployment, prefix length), since a job's write belongs to its deployment."""

    read_s: dict  # valid reads; tier "recompute" is the single recompute read, kept as a cross-check
    load_gib_s: dict  # valid reads: the tier's read bytes / read time, as vLLM reports them
    recompute_s: dict  # cold writes (`write_s`): the recompute baseline, one sample per job attempt
    store_gib_s: dict  # cold writes: the secondary tier's write bytes / write time


def summarize(rows: list[dict], stage: str) -> Summary:
    from .tierconfig import DEPLOYMENT_FOR_TIER

    reads, loads, cold, stores = (defaultdict(list) for _ in range(4))
    seen: set[tuple[str, int]] = set()
    for r in rows:
        if r.get("stage") != stage:
            continue
        model, tier, n = r["model"], r["tier"], r["prefix_tokens"]
        if r.get("valid"):
            reads[(model, tier, n)].append(r["read_s"])
            if r.get("read_time_s"):
                loads[(model, tier, n)].append(r["read_bytes"] / r["read_time_s"] / GIB)
        # every row of a job attempt carries the same write; count it once
        if r.get("write_s") is not None and (r["job"], r["attempt"]) not in seen:
            seen.add((r["job"], r["attempt"]))
            key = (model, DEPLOYMENT_FOR_TIER[tier], n)
            cold[key].append(r["write_s"])
            if r.get("write_time_s"):
                stores[key].append(r["write_bytes"] / r["write_time_s"] / GIB)

    def med(d: dict) -> dict:
        return {k: median(v) for k, v in d.items()}

    return Summary(med(reads), med(loads), med(cold), med(stores))


def gpu_usd(seconds: float, c: CostInputs) -> float:
    return seconds * c.gpu_hour_usd * c.gpus / 3600.0


def per_million_tokens(seconds_per_prefix: float, prefix_tokens: int, c: CostInputs) -> float:
    """$ of GPU time per 1M prefix tokens, from the time to serve one prefix of `prefix_tokens`."""
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    return gpu_usd(seconds_per_prefix, c) * 1e6 / prefix_tokens


def storage_usd(prefix_bytes: int, residency_hours: float, c: CostInputs) -> float:
    if c.storage_gib_month_usd == "included":
        return 0.0
    return prefix_bytes / GIB * c.storage_gib_month_usd * residency_hours / HOURS_PER_MONTH


def tier_row(
    tier: str,
    prefix_tokens: int,
    prefix_bytes: int,
    load_s: float,
    recompute_s: float,
    c: CostInputs,
    residency_hours: float = 1.0,
) -> dict:
    load = per_million_tokens(load_s, prefix_tokens, c)
    store = storage_usd(prefix_bytes, residency_hours, c) * 1e6 / prefix_tokens
    recompute = per_million_tokens(recompute_s, prefix_tokens, c)
    return {
        "tier": tier,
        "prefix_tokens": prefix_tokens,
        "load_usd_per_1m": load,
        "storage_usd_per_1m": store,
        "total_usd_per_1m": load + store,
        "recompute_usd_per_1m": recompute,
        "saving_vs_recompute": 1.0 - (load + store) / recompute if recompute else None,
    }
