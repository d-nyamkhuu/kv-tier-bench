"""The protocol for one job: write its prefix(es) once, then read them back from each tier, rep after rep.

At concurrency c a job has c distinct prefixes of one length. They are written one at a time (each settled before
the next, so a burst of stores cannot be dropped), and every read sends all c requests at once. See docs/plan.md,
"Protocol". The steps that rely on UNVERIFIED names are in constants.py.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from . import constants as C
from . import prom
from .client import Engine, prompt_tokens
from .grid import Job, Point, align_down, job_of
from .tierconfig import READER_DEPLOYMENTS, STORAGE_TIERS, STORE_DEPLOYMENTS

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


def store_op(before: list[prom.Sample], after: list[prom.Sample], metric: str, operation: str, ok: bool = True):
    """Delta of a Mooncake store counter for one operation: its successful calls, or (ok=False) all of them."""
    labels = {"operation": operation} | ({"status": C.STORE_OK} if ok else {})
    return prom.delta(before, after, metric, **labels)


async def settle(
    engine: Engine,
    baseline: list[prom.Sample],
    tier_index: int | None = None,
    expect_bytes: int = 0,
    store: bool = False,
) -> list[prom.Sample]:
    """Wait until the tier's writes are done: no active jobs, write bytes unchanged for a few polls, and at least
    `expect_bytes` written since `baseline`.

    The byte floor matters because a cascade to fs/obj starts asynchronously: without it, "unchanged and idle" can
    hold before the cascade has even begun. `store`: the Mooncake store's save_put bytes instead of the tiering
    counters (it has no active-job gauge, so only the bytes and the floor count); a failed put ends the wait.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    last, stable = None, 0
    while time.monotonic() < deadline:
        s = await engine.metrics()
        if store:
            written, busy = store_op(baseline, s, C.M_STORE_BYTES, "save_put"), 0.0
            if store_op(baseline, s, C.M_STORE_FAILED_KEYS, "save_put", ok=False):
                return s  # a put failed: the floor may never be reached, and run_job records the drop
        else:
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
    store: drop the GPU copy only. reset_external is left off, so the connector is not asked to reset the store.
    store-p2p: the same on the reader, which loads from the writer's pool.
    p2p: drop the reader's own copies, which its previous pull left behind; the writer keeps its CPU copy.
    """
    if tier == "gpu":
        return
    if tier == "store-p2p":
        await _reset(reader, reset_external=False)
    elif tier == "p2p":
        await _reset(reader, reset_external=True)
    else:
        await _reset(writer, reset_external=tier in STORAGE_TIERS)


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


async def warm_up(
    writer: Engine,
    block_size: int,
    reader: Engine | None = None,
    source_host: str | None = None,
    store: bool = False,
):
    """One throwaway request per engine before anything is measured, so no measured row pays first-request costs.

    Settles the writer's async cascade before returning (and before a p2p reader pulls), so warm-up writes do not
    leak into the first measured job's `write_bytes` or cause the reader's warm-up pull to miss.
    """
    prompt = prompt_tokens(WARMUP_SEED, 2 * block_size + 1)
    before = await writer.metrics()
    await writer.complete(prompt, kv_transfer_params={"max_load_tokens": 0})
    await settle(writer, before, expect_bytes=0, store=store)
    if reader is not None:  # p2p pulls by source host; store-p2p's reader finds the prefix in the writer's pool
        params = read_params("p2p", source_host, "warmup") if source_host else None
        await reader.complete(prompt, kv_transfer_params=params)


def _has(samples: list[prom.Sample], name: str) -> bool:
    return any(s.name == name for s in samples)


async def all_or_nothing(coros) -> list:
    """Run the coroutines at once and return their results in order. If one fails, cancel the rest and raise its
    error unchanged: a request left running would overlap the next read, and run.py classifies httpx errors."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        failed = next((t for t in tasks if t in done and t.exception() is not None), None)
        if failed is not None:
            raise failed.exception()
        return [t.result() for t in tasks]
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
    """Writes the job's prefix(es) once, then measures `points` (of this job, ordered by rep and tier).

    Store-side numbers (write_s, settle_s, write_bytes, write_time_s) are measured once per job and copied onto
    each of its rows, so count them once per job, not once per row. With several prefixes, `write_s` is the mean
    cold prefill of one (`write_s_each` has them all), and the rest are totals over the job.

    Each row is passed to `emit` as soon as it is measured, so a run that dies mid-job keeps what it measured.
    `reader` is the other instance for p2p; otherwise the writer reads too. `kv_bytes_per_token` (per instance)
    sets the byte floor that fs/obj writes must reach before the tier is read.
    """
    reader = reader or writer
    c = job.concurrency
    # the cacheable prefix is whole blocks; one extra token is always computed, so the whole prefix can hit
    prefix = align_down(job.target_tokens, block_size)
    prompts = [prompt_tokens(seed, prefix + 1) for seed in job.prompt_seeds(attempt)]
    common = {"job": job.id, "attempt": attempt, "prefix_tokens": prefix, "block_size": block_size}

    # 1-2. write each prefix (a cold prefill with loads off), and settle its stores before the next. Snapshot first:
    # a store can be dropped during the write itself. The byte floor (a tenth of the instance's KV for one prompt)
    # holds whether the counter sums all TP ranks or reports one (UNVERIFIED), and still proves the cascade ran.
    secondary = job.deployment in STORAGE_TIERS
    is_store = job.deployment in STORE_DEPLOYMENTS
    # The store has no active-job gauge, and its byte counter moves only as each put completes, so its floor is
    # most of the prefix (the read rule's 90%): a low floor would let a read start while puts are still in flight.
    floor = int((0.9 if is_store else 0.1) * prefix * kv_bytes_per_token) if secondary or is_store else 0
    index = C.TIER_INDEX[job.deployment] if secondary else None
    before_write = await writer.metrics()
    writes, settle_s = [], 0.0
    baseline = before_write
    for prompt in prompts:
        write_s, _ = await writer.complete(prompt, kv_transfer_params={"max_load_tokens": 0})
        writes.append(write_s)
        t_settle = time.monotonic()
        settled = await settle(writer, baseline, index, floor, store=is_store)
        settle_s += time.monotonic() - t_settle
        baseline = settled
    common["write_s"] = sum(writes) / c
    if c > 1:
        common["write_s_each"] = writes
    dropped = prom.delta(before_write, settled, C.M_TIER_STORES_DROPPED)
    # the store side, once per job: how long the cascade took after the prefill, and what the tier reports
    common["settle_s"] = settle_s
    if is_store:  # the store's own counters; a key that failed to store counts as a dropped store
        dropped = store_op(before_write, settled, C.M_STORE_FAILED_KEYS, "save_put", ok=False)
        common["write_bytes"] = store_op(before_write, settled, C.M_STORE_BYTES, "save_put")
        common["write_time_s"] = store_op(before_write, settled, C.M_STORE_TIME + "_sum", "save_put")
    elif secondary:
        label = tier_label(settled, C.M_TIER_WRITE_BYTES, index)
        labels = {"tier": label} if label else {}
        common["write_bytes"] = prom.delta(before_write, settled, C.M_TIER_WRITE_BYTES, **labels)
        common["write_time_s"] = prom.delta(before_write, settled, C.M_TIER_WRITE_TIME, **labels)
    else:  # p2p / base do not cascade on write; 1:p2p write_bytes counts previous jobs' peer sends
        common["write_bytes"] = None
        common["write_time_s"] = None

    # 3. reads: every rep re-reads the same stored prefix(es) from each tier, all of a batch at once
    rows = []
    for n_read, p in enumerate(points):
        await isolate(writer, reader, p.tier)
        engine = reader if p.tier in READER_DEPLOYMENTS else writer
        # each request of a p2p batch is its own pull, with its own id
        rid = f"{job.id}#a{attempt}#r{p.rep}"
        params = [read_params(p.tier, source_host, rid if c == 1 else f"{rid}#i{i}") for i in range(c)]
        before = await engine.metrics()
        t0 = time.perf_counter()
        done = await all_or_nothing(engine.complete(pr, kv_transfer_params=pa) for pr, pa in zip(prompts, params))
        # one request: its own wall time, as in s1. A batch: from the first send to the last response
        read_s = done[0][0] if c == 1 else time.perf_counter() - t0
        after = await scrape_after(engine, before, requests=c)
        row = p.as_dict() | common | {"read_index": n_read, "started": time.time()}
        if c > 1:
            row["read_s_each"] = [dt for dt, _ in done]
        if p.tier == "recompute":
            cached = prom.delta(before, after, C.M_CACHED_BY_SOURCE)
            _finish(row, read_s, before, after, cached, expected=0, dropped=0.0, source="any")
        elif p.tier in STORE_DEPLOYMENTS:
            # The source label of a store hit is UNVERIFIED, so every source but the GPU counts; the store's own
            # load_get bytes then prove the tokens came from it.
            cached = prom.delta(before, after, C.M_CACHED_BY_SOURCE) - prom.delta(
                before, after, C.M_CACHED_BY_SOURCE, source=C.SOURCE_LABEL["gpu"]
            )
            row["read_bytes"] = store_op(before, after, C.M_STORE_BYTES, "load_get")
            row["read_time_s"] = store_op(before, after, C.M_STORE_TIME + "_sum", "load_get")
            row["load_failed_keys"] = store_op(before, after, C.M_STORE_FAILED_KEYS, "load_get", ok=False)
            _finish(row, read_s, before, after, cached, expected=c * prefix, dropped=dropped, source="store")
            min_bytes = 0.9 * c * prefix * kv_bytes_per_token
            if row["valid"] and (row["load_failed_keys"] or row["read_bytes"] < min_bytes):
                row["valid"] = False
                row["reason"] = (
                    f"store load_get: {row['read_bytes']:g} bytes (want >= {min_bytes:g}), "
                    f"{row['load_failed_keys']:g} failed keys"
                )
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
            tier_dropped = dropped if p.tier in STORAGE_TIERS else 0.0
            _finish(row, read_s, before, after, cached, expected=c * prefix, dropped=tier_dropped, source=source)
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
    # ~ the KV load with one request (see constants); in a batch it also holds waiting for batch-mates
    row["engine_queue_s"] = _engine_mean(before, after, C.M_QUEUE_TIME)
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
