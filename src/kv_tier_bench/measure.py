"""The protocol for one job: write a prefix once, then read it back from each tier, rep after rep.

See docs/plan.md, "Protocol". Nothing here has run against a GPU yet. The steps that rely on UNVERIFIED names
are in constants.py.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from . import constants as C
from . import prom
from .client import Engine, prompt_tokens
from .grid import Job, Point, align_down, job_of

SETTLE_POLL_S = 0.25
SETTLE_STABLE_POLLS = 4
SETTLE_TIMEOUT_S = 300.0
SCRAPE_TIMEOUT_S = 5.0
POKE = [0, 0, 0]  # a prompt shorter than any block: makes an idle engine step, caches nothing
WARMUP_SEED = 0  # the warm-up prompt is never measured, so it can be the same on every run


def tier_label(samples: list[prom.Sample], name: str, index: int | None) -> str | None:
    """The `tier` label of `name` whose index part ("<index>:<type>") is `index`, as this scrape shows it."""
    if index is None:
        return None
    for s in samples:
        t = s.label("tier")
        if s.name in (name, name + "_total") and t and t.partition(":")[0] == str(index):
            return t
    return None


def _indexed(label: str) -> bool:
    head, sep, _ = label.partition(":")
    return bool(sep) and head.isdigit()


def tier_read(by_bytes: dict[str, float], by_time: dict[str, float], index: int | None) -> tuple:
    """(read bytes, read time) of the tier with this index, from one read's deltas per `tier` label.

    None, None if every label has the expected format but none is this tier's: the tier doesn't count its reads
    here (possible for p2p), and the CPU hop's bytes must not stand in for it. If the format is not as expected,
    the sum over all labels; the per-label deltas are kept on the row, so bandwidth can be recomputed.
    """
    own = next((t for t in by_bytes if index is not None and t.partition(":")[0] == str(index)), None)
    if own is not None:
        return by_bytes[own], by_time.get(own, 0.0)
    if index is not None and by_bytes and all(_indexed(t) for t in by_bytes):
        return None, None
    return sum(by_bytes.values()), sum(by_time.values())


async def settle(
    engine: Engine, baseline: list[prom.Sample], tier_index: int | None = None, expect_bytes: int = 0
) -> list[prom.Sample]:
    """Wait until the tier's writes are done: no active jobs, write bytes unchanged for a few polls, and at least
    `expect_bytes` written since `baseline`.

    The byte floor matters because a cascade to fs/obj starts asynchronously: without it, "unchanged and idle" can
    hold before the cascade has even begun.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    last, stable = None, 0
    while time.monotonic() < deadline:
        s = await engine.metrics()
        label = tier_label(s, C.M_TIER_WRITE_BYTES, tier_index)
        labels = {"tier": label} if label else {}  # the secondary tier's own series, or the sum of all
        written = prom.delta(baseline, s, C.M_TIER_WRITE_BYTES, **labels)
        busy = prom.find(s, C.M_TIER_ACTIVE_CASCADES) + prom.find(s, C.M_TIER_ACTIVE_PROMOTIONS)
        if busy and written == last:
            # An idle engine stops stepping, and a store that finished meanwhile is only collected on its next step,
            # so the job looks active forever (p2p-only deployments; vLLM docs, "Threading model"). A request
            # shorter than one block makes it step and caches nothing.
            await engine.complete(POKE, kv_transfer_params={"max_load_tokens": 0})
        if written == last and busy == 0 and written >= expect_bytes:
            stable += 1
            if stable >= SETTLE_STABLE_POLLS:
                return s
        else:
            stable = 0
        last = written
        await asyncio.sleep(SETTLE_POLL_S)
    raise TimeoutError(f"tier writes did not settle ({last} of {expect_bytes} bytes written)")


async def scrape_after(engine: Engine, before: list[prom.Sample], requests: int = 1) -> list[prom.Sample]:
    """Scrape once the engine has recorded the request(s) (its TTFT count has advanced), so deltas are complete."""
    deadline = time.monotonic() + SCRAPE_TIMEOUT_S
    while True:
        s = await engine.metrics()
        done = prom.delta(before, s, C.M_TTFT + "_count") >= requests
        # queue/prefill times land when the request finishes; wait for them too where the engine has them
        if done and _has(s, C.M_PREFILL_TIME + "_count"):
            done = prom.delta(before, s, C.M_PREFILL_TIME + "_count") >= requests
        if done or time.monotonic() > deadline:
            return s
        await asyncio.sleep(0.05)


async def _reset(engine: Engine, reset_external: bool) -> None:
    deadline = time.monotonic() + 60
    while not await engine.reset_prefix_cache(reset_external=reset_external):
        # the reset fails while transfers are in flight; retry until it succeeds
        if time.monotonic() > deadline:
            raise TimeoutError("reset_prefix_cache kept failing")
        await asyncio.sleep(0.5)


async def isolate(writer: Engine, reader: Engine, tier: str) -> None:
    """Make the next read come from `tier` and nothing above it.

    gpu: nothing (the prefix is in GPU from the write, or from the previous rep's reads).
    cpu, recompute: drop the GPU copy (recompute then also turns loads off, see read_params).
    fs, obj: drop the GPU and CPU copies; secondary tiers are kept (TieringOffloadingManager.reset_cache).
    p2p: drop the reader's own copies, which its previous pull left behind; the writer keeps its CPU copy.
    """
    if tier == "gpu":
        return
    if tier == "p2p":
        await _reset(reader, reset_external=True)
    else:
        await _reset(writer, reset_external=tier in ("fs", "obj"))


def read_params(tier: str, source_host: str | None, kv_request_id: str) -> dict | None:
    if tier == "recompute":
        return {"max_load_tokens": 0}
    if tier == "p2p":
        if not source_host:
            raise ValueError("p2p reads need the source instance's IP")
        return {
            "remote_kv_source": {"kv_request_id": kv_request_id, "remote_host": source_host, "remote_port": C.P2P_PORT}
        }
    tiers = C.LOAD_TIERS[tier]
    # an empty list is meaningful: it restricts loads to CPU (CPU is always allowed)
    return {"kv_load_tiers": tiers} if tiers is not None else None


async def warm_up(writer: Engine, block_size: int, reader: Engine | None = None, source_host: str | None = None):
    """One throwaway request per engine before anything is measured, so no measured row pays first-request costs.

    For p2p, the reader's warm-up is a pull from the writer, so the first measured pull doesn't pay for setting up
    the peer connection either. Best effort: the pull may still recompute if the writer's copy hasn't landed yet.
    """
    prompt = prompt_tokens(WARMUP_SEED, 2 * block_size + 1)
    await writer.complete(prompt, kv_transfer_params={"max_load_tokens": 0})
    if reader is not None:
        await reader.complete(prompt, kv_transfer_params=read_params("p2p", source_host, "warmup"))


def _has(samples: list[prom.Sample], name: str) -> bool:
    return any(s.name == name for s in samples)


def _engine_mean(before: list[prom.Sample], after: list[prom.Sample], metric: str) -> float | None:
    """Mean of a vLLM histogram over the request(s) between two scrapes (sum / count)."""
    n = prom.delta(before, after, metric + "_count")
    return prom.delta(before, after, metric + "_sum") / n if n else None


async def run_job(
    job: Job,
    points: list[Point],
    block_size: int,
    writer: Engine,
    reader: Engine | None = None,
    source_host: str | None = None,
    attempt: int = 0,
    kv_bytes_per_token: int = 0,
    emit: Callable[[dict], None] = lambda row: None,
) -> list[dict]:
    """Writes the job's prefix once, then measures `points` (of this job, ordered by rep and tier).

    Store-side numbers (write_s, settle_s, write_bytes, write_time_s) are measured once per job and copied onto
    each of its rows, so count them once per job, not once per row.

    Each row is passed to `emit` as soon as it is measured, so a run that dies mid-job keeps what it measured.
    `reader` is the other instance for p2p; otherwise the writer reads too. `kv_bytes_per_token` (per instance)
    sets the byte floor that fs/obj writes must reach before the tier is read.
    """
    reader = reader or writer
    # the cacheable prefix is whole blocks; one extra token is always computed, so the whole prefix can hit
    prefix = align_down(job.target_tokens, block_size)
    prompt = prompt_tokens(job.seed(attempt), prefix + 1)
    common = {"job": job.id, "attempt": attempt, "prefix_tokens": prefix, "block_size": block_size}

    # 1. write: cold prefill with loads off. Snapshot first: a store can be dropped during the write itself.
    before_write = await writer.metrics()
    common["write_s"], _ = await writer.complete(prompt, kv_transfer_params={"max_load_tokens": 0})

    # 2. settle. The byte floor (a tenth of the instance's KV for this prompt) holds whether the counter sums all
    # TP ranks or reports one (UNVERIFIED), and still proves the cascade ran.
    secondary = job.deployment in ("fs", "obj")
    floor = int(0.1 * prefix * kv_bytes_per_token) if secondary else 0
    index = C.TIER_INDEX[job.deployment] if secondary else None
    t_settle = time.monotonic()
    settled = await settle(writer, before_write, index, floor)
    dropped = prom.delta(before_write, settled, C.M_TIER_STORES_DROPPED)
    # the store side, once per job: how long the cascade took after the prefill, and what the tier reports
    label = tier_label(settled, C.M_TIER_WRITE_BYTES, index)
    labels = {"tier": label} if label else {}
    common["settle_s"] = time.monotonic() - t_settle
    common["write_bytes"] = prom.delta(before_write, settled, C.M_TIER_WRITE_BYTES, **labels)
    common["write_time_s"] = prom.delta(before_write, settled, C.M_TIER_WRITE_TIME, **labels)

    # 3. reads: every rep re-reads the same stored prefix from each tier
    rows = []
    for n_read, p in enumerate(points):
        await isolate(writer, reader, p.tier)
        engine = reader if p.tier == "p2p" else writer
        params = read_params(p.tier, source_host, f"{job.id}#a{attempt}#r{p.rep}")
        before = await engine.metrics()
        read_s, _ = await engine.complete(prompt, kv_transfer_params=params)
        after = await scrape_after(engine, before)
        row = p.as_dict() | common | {"read_index": n_read, "started": time.time()}
        if p.tier == "recompute":
            cached = prom.delta(before, after, C.M_CACHED_BY_SOURCE)
            _finish(row, read_s, before, after, cached, expected=0, dropped=0.0, source="any")
        else:
            source = C.SOURCE_LABEL[p.tier]
            cached = prom.delta(before, after, C.M_CACHED_BY_SOURCE, source=source)
            # per tier label, since a load from fs or obj may also count its hop through the CPU tier
            by_bytes = prom.delta_by_label(before, after, C.M_TIER_READ_BYTES, "tier")
            by_time = prom.delta_by_label(before, after, C.M_TIER_READ_TIME, "tier")
            row["read_bytes_by_tier"], row["read_time_s_by_tier"] = by_bytes, by_time
            row["read_bytes"], row["read_time_s"] = tier_read(by_bytes, by_time, C.TIER_INDEX.get(p.tier))
            promo = prom.delta_by_label(before, after, C.M_TIER_PROMOTION_LATENCY + "_sum", "tier")
            row["promotion_s_by_tier"] = promo
            row["promotion_s"] = tier_read(by_bytes, promo, C.TIER_INDEX.get(p.tier))[1]
            # a dropped store only matters for the tier it was dropped from
            tier_dropped = dropped if p.tier in ("fs", "obj") else 0.0
            _finish(row, read_s, before, after, cached, expected=prefix, dropped=tier_dropped, source=source)
        emit(row)
        rows.append(row)
    return rows


async def run_point(point: Point, block_size: int, writer: Engine, **kw) -> dict:
    """One point as a job of its own (one write, one read). Used by tests and one-off checks."""
    return (await run_job(job_of(point), [point], block_size, writer, **kw))[0]


def _finish(row, read_s, before, after, cached, expected, dropped, source) -> dict:
    """Valid only if the expected source served the whole block-aligned prefix (or, for recompute, nothing)."""
    row["read_s"] = read_s
    row["engine_ttft_s"] = _engine_mean(before, after, C.M_TTFT)
    row["engine_queue_s"] = _engine_mean(before, after, C.M_QUEUE_TIME)  # ~ the KV load (see constants)
    row["engine_prefill_s"] = _engine_mean(before, after, C.M_PREFILL_TIME)
    row["cached_tokens"] = cached
    # every source, so a wrong SOURCE_LABEL can be fixed from the first bad row
    row["cached_by_source"] = prom.delta_by_label(before, after, C.M_CACHED_BY_SOURCE, "source")
    row["expected_cached"] = expected
    row["stores_dropped"] = dropped
    if expected == 0:
        row["valid"] = cached == 0
        row["reason"] = None if row["valid"] else f"recompute run still hit a cache ({cached:g} tokens)"
    else:
        row["valid"] = dropped == 0 and cached >= expected
        if row["valid"]:
            row["reason"] = None
        elif dropped:
            row["reason"] = f"stores dropped: {dropped:g}"
        else:
            row["reason"] = f"source={source} served {cached:g} of {expected:g}"
    return row
