"""Results under concurrent load (stages with `concurrency`): one row per batch of c requests, read at once.

Keys are (model, tier, prefix length, concurrency). What a batch row measures:

- `read_s`: wall time from the first request sent to the last response. The batch serves c x prefix tokens in
  that time, so it is what the GPU is held for, and what the cost table charges.
- `read_s_each`: each request's own time to its (only) token: the TTFT a user in the batch sees.
- `read_bytes`: bytes the tier under test delivered for the whole batch.

Counter-based fields add up over the batch: `read_time_s` and `promotion_s` are sums over the c requests, not wall
time, so read_bytes / promotion_s (s1's recipe for p2p) understates the rate by up to c. The `engine_*_s` fields are
means over the c requests. The rate here is `batch_gib_s` = read_bytes / read_s, end to end (GPU copy and prefill
included), so it is lower than s1's `load_gib_s` for the same link and must not be compared with it.

The recompute baseline is the recompute batch at the same length and concurrency: c cold prefills sent at once.
The job's own writes cannot be it, because a job writes its prefixes one at a time (measure.run_job).
`engine_queue_s` is not used: in a batch it also holds the wait for batch-mates, not only the load.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from .cost import GIB, CostInputs, per_million_tokens, storage_usd


@dataclass(frozen=True)
class LoadSummary:
    batch_s: dict  # median batch wall time (read_s)
    ttft_p50_s: dict  # median over reps of the batch's median request time
    ttft_max_s: dict  # median over reps of the batch's slowest request
    batch_gib_s: dict  # median of read_bytes / read_s: the whole batch's delivered rate, end to end
    reps: dict  # valid batches behind each median


def _request_times(r: dict) -> list[float]:
    return r.get("read_s_each") or [r["read_s"]]


def summarize(rows: list[dict], stage: str) -> LoadSummary:
    batch, p50, worst, rate, reps = (defaultdict(list) for _ in range(5))
    for r in rows:
        if r.get("stage") != stage or not r.get("valid"):
            continue
        key = (r["model"], r["tier"], r["prefix_tokens"], r.get("concurrency", 1))
        each = _request_times(r)
        batch[key].append(r["read_s"])
        p50[key].append(median(each))
        worst[key].append(max(each))
        if r.get("read_bytes") and r["read_s"]:
            rate[key].append(r["read_bytes"] / r["read_s"] / GIB)
        reps[key].append(1)

    def med(d: dict) -> dict:
        return {k: median(v) for k, v in d.items()}

    return LoadSummary(med(batch), med(p50), med(worst), med(rate), {k: len(v) for k, v in reps.items()})


def cost_row(
    tier: str,
    prefix_tokens: int,
    concurrency: int,
    prefix_bytes: int,
    batch_s: float,
    recompute_batch_s: float,
    c: CostInputs,
    residency_hours: float = 1.0,
) -> dict:
    """$ per 1M prefix tokens served by one batch, against recomputing the same batch.

    The GPU is billed for the batch's wall time, and the batch serves concurrency x prefix_tokens tokens. Storage
    is per prefix and does not depend on how many are read at once.
    """
    served = concurrency * prefix_tokens
    load = per_million_tokens(batch_s, served, c)
    store = storage_usd(prefix_bytes, residency_hours, c) * 1e6 / prefix_tokens
    recompute = per_million_tokens(recompute_batch_s, served, c)
    return {
        "tier": tier,
        "prefix_tokens": prefix_tokens,
        "concurrency": concurrency,
        "tokens_per_s": served / batch_s,
        "recompute_tokens_per_s": served / recompute_batch_s,
        "load_usd_per_1m": load,
        "storage_usd_per_1m": store,
        "total_usd_per_1m": load + store,
        "recompute_usd_per_1m": recompute,
        "saving_vs_recompute": 1.0 - (load + store) / recompute if recompute else None,
    }
