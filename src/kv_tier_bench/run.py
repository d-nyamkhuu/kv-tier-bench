"""The sweep loop for one deployment: every not-yet-done point of one (stage, model, deployment).

Rerunning the same command resumes: done points are skipped, and a job with points left is rerun with a fresh
prompt for those points only, until each is valid or has `max_attempts` attempts.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable

import httpx

from .client import Engine
from .grid import Point, expand, group_jobs
from .measure import run_job, warm_up
from .store import ResultStore
from .tierconfig import READER_DEPLOYMENTS, STORE_DEPLOYMENTS, deployment_for


class Abort(RuntimeError):
    """The engine is unreachable or the setup is wrong. Stop rather than burn attempts on every point."""


BREAKER = 5  # consecutive invalid attempts with the same reason before the run stops
# MooncakeStoreConnector skips lookups for 30 s after three failed ones (vLLM docs), so a retry right after a failed
# store attempt would miss too. Wait that window out first.
STORE_LOOKUP_BACKOFF_S = 31.0


@dataclass
class Peers:
    writer: Engine
    reader: Engine | None = None  # the second instance, for p2p tiers
    source_host: str | None = None  # the writer's private IP, which the reader pulls from


def reason_kind(reason: str | None) -> str:
    """The reason with its numbers removed, so "served 0 of 960" and "served 0 of 4000" count as the same."""
    return re.sub(r"\d+(\.\d+)?", "#", reason or "")


def points_for(exp: dict, stage: str, model: str, deployment: str, tiers: list[str] | None = None) -> list[Point]:
    pts = [p for p in expand(exp, stage) if p.model == model and deployment_for(stage, p.tier) == deployment]
    return [p for p in pts if p.tier in tiers] if tiers else pts


async def run_deployment(
    points: list[Point],
    peers: Peers,
    block_size: int,
    store: ResultStore,
    max_attempts: int,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True),
    kv_bytes_per_token: int = 0,
    breaker: int = BREAKER,
) -> dict[str, int]:
    """Runs every point until it is done. Returns counts of valid and invalid rows written this run.

    Stops once one tier has failed the same way in `breaker` job attempts in a row (the reps of one job attempt
    count once: they share a prefill, so they fail together). That pattern means a setup error (a wrong source
    label, block size or tier filter), and retrying would only spend GPU time on fresh prefills.
    """
    counts = {"valid": 0, "invalid": 0, "skipped": 0}
    jobs = group_jobs(points)
    left = sum(1 for p in points if not store.is_done(p.id, max_attempts))
    counts["skipped"] = len(points) - left
    log(f"{left} points in {len(jobs)} jobs to run, {counts['skipped']} already done, block size {block_size}")
    if left:
        try:
            await warm_up(peers.writer, block_size, peers.reader, peers.source_host)
        except (httpx.TransportError, TimeoutError) as e:
            raise Abort(f"warm-up: {type(e).__name__}: {e}") from e
        except httpx.HTTPStatusError as e:  # the measured reads record such errors per point
            log(f"warm-up got HTTP {e.response.status_code}; continuing")
    streaks: dict[str, tuple[int, str]] = {}  # tier -> (job attempts in a row that failed this way, reason kind)
    t0 = time.monotonic()
    for n_job, (job, job_points) in enumerate(jobs.items(), 1):
        while missing := [p for p in job_points if not store.is_done(p.id, max_attempts)]:
            # Every job attempt gets a fresh prompt, counted by a start marker. Reusing the prompt of an attempt
            # that died mid-job would be wrong: fs and obj keep its blocks across vLLM restarts, so the write would
            # store nothing new and settle would never see its byte floor.
            marker = f"{job.id}#job"
            attempt = store.next_attempt(marker)
            store.append({"id": marker, "kind": "job_start", "attempt": attempt, "points": len(missing)})
            emitted: set[str] = set()
            bad: dict[str, dict] = {}  # tier -> its first invalid row in this job attempt
            good: set[str] = set()

            def emit(row: dict) -> None:
                store.append(row)
                emitted.add(row["id"])
                ok = bool(row.get("valid"))
                counts["valid" if ok else "invalid"] += 1
                log(
                    f"[job {n_job}/{len(jobs)} {time.monotonic() - t0:6.0f}s] {'ok ' if ok else 'BAD'} {row['id']} "
                    f"a{attempt} read={row.get('read_s', float('nan')):.3f}s {row.get('reason') or ''}"
                )
                if ok:
                    good.add(row["tier"])
                else:
                    bad.setdefault(row["tier"], row)

            two = job.deployment in READER_DEPLOYMENTS
            try:
                await run_job(
                    job,
                    missing,
                    block_size,
                    peers.writer,
                    reader=peers.reader if two else None,
                    source_host=peers.source_host if job.deployment == "p2p" else None,
                    attempt=attempt,
                    kv_bytes_per_token=kv_bytes_per_token,
                    emit=emit,
                )
            except (httpx.TransportError, TimeoutError) as e:
                # the engine died, was preempted, or is wedged: stop so a rerun resumes from here. Rows already
                # emitted are kept; the point in flight is not recorded.
                raise Abort(f"{job.id} attempt {attempt}: {type(e).__name__}: {e}") from e
            except httpx.HTTPStatusError as e:
                reason = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                for p in missing:
                    if p.id not in emitted:
                        emit(
                            p.as_dict()
                            | {
                                "job": job.id,
                                "attempt": attempt,
                                "block_size": block_size,
                                "valid": False,
                                "reason": reason,
                                "started": time.time(),
                            }
                        )
            if bad and job.deployment in STORE_DEPLOYMENTS:
                await asyncio.sleep(STORE_LOOKUP_BACKOFF_S)
            # The breaker counts job attempts, once the attempt is over: its reps share a prefill and fail
            # together, and a tier that only fails from rep 2 on (say, CPU after an fs read) must still count.
            for tier in good - bad.keys():
                streaks.pop(tier, None)
            for tier, row in bad.items():
                kind = reason_kind(row.get("reason"))
                n, last = streaks.get(tier, (0, ""))
                streaks[tier] = (n + 1 if kind == last else 1, kind)
                if streaks[tier][0] >= breaker:
                    raise Abort(
                        f"{tier}: {breaker} job attempts in a row failed the same way: {row.get('reason')}; "
                        f"cached tokens by source: {row.get('cached_by_source')}. "
                        "Fix constants.py (or pass --block-size)."
                    )
    return counts
