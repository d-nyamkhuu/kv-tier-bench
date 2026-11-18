"""The sweep loop and block-size parsing, against fake engines."""

import asyncio

import pytest

from kv_tier_bench.client import block_size
from kv_tier_bench.grid import Point
from kv_tier_bench.measure import run_point
from kv_tier_bench.run import Abort, Peers, run_deployment
from kv_tier_bench.store import ResultStore

from .test_measure import FakeEngine


def pts(tier, n=3, tokens=1000):
    return [Point("s1", "m", tier, tokens, rep) for rep in range(n)]


def measured(store):
    return [r for r in store.rows() if r.get("kind") != "job_start"]


def test_run_deployment_measures_everything_then_resumes(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    eng = FakeEngine()
    points = pts("cpu") + pts("gpu") + pts("recompute")
    counts = asyncio.run(run_deployment(points, Peers(eng), 64, store, max_attempts=3, log=lambda m: None))
    assert counts == {"valid": 9, "invalid": 0, "skipped": 0}
    again = asyncio.run(run_deployment(points, Peers(eng), 64, ResultStore(tmp_path / "r.jsonl"), 3, lambda m: None))
    assert again == {"valid": 0, "invalid": 0, "skipped": 9}


def test_invalid_points_are_retried_then_given_up(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    eng = FakeEngine(serve_from="host")  # every fs read is credited to the wrong source
    counts = asyncio.run(
        run_deployment(pts("fs", 2), Peers(eng), 64, store, max_attempts=3, log=lambda m: None, breaker=100)
    )
    assert counts["invalid"] == 6 and counts["valid"] == 0
    assert sorted(r["attempt"] for r in measured(store)) == [0, 0, 1, 1, 2, 2]


def test_breaker_stops_a_run_that_fails_the_same_way(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    eng = FakeEngine(serve_from="host")
    points = [p for n in (1000, 2000, 3000, 4000, 5000, 6000) for p in pts("fs", 2, tokens=n)]
    with pytest.raises(Abort, match="'host': 4992"):  # the message says where the tokens really came from
        asyncio.run(run_deployment(points, Peers(eng), 64, store, max_attempts=1, log=lambda m: None))
    # the two reps of a job share a prefill, so they count once: it stops after the 5th job, not the 3rd
    assert len(measured(store)) == 5 * 2
    assert measured(store)[0]["cached_by_source"]["host"] == 960


def test_breaker_catches_a_tier_that_fails_only_in_later_reps(tmp_path):
    class LosesCpuAfterRep0(FakeEngine):
        """Rep 0 of each job reads CPU fine; every later reset after a write also empties the CPU tier."""

        async def complete(self, tokens, max_tokens=1, kv_transfer_params=None):
            if (kv_transfer_params or {}).get("max_load_tokens") == 0:
                self.resets = 0  # a fresh write
            return await super().complete(tokens, max_tokens, kv_transfer_params)

        async def reset_prefix_cache(self, reset_external=False):
            self.resets += 1
            if self.resets > 1:
                self.cpu.clear()
            self.gpu.clear()
            return True

    store = ResultStore(tmp_path / "r.jsonl")
    points = [p for n in range(1000, 7000, 1000) for p in pts("cpu", 3, tokens=n)]
    with pytest.raises(Abort, match="cpu: 5 job attempts"):
        asyncio.run(run_deployment(points, Peers(LosesCpuAfterRep0()), 64, store, max_attempts=1, log=lambda m: None))


def test_unreachable_engine_aborts_and_resumes_with_a_fresh_prompt(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    eng = FakeEngine(fail_after=4)  # the warm-up, the write, two reads, then the third read fails
    with pytest.raises(Abort):
        asyncio.run(run_deployment(pts("cpu"), Peers(eng), 64, store, max_attempts=3, log=lambda m: None))
    assert len(measured(store)) == 2  # rows already measured are kept; the read in flight is not recorded
    fresh = FakeEngine()
    counts = asyncio.run(run_deployment(pts("cpu"), Peers(fresh), 64, store, 3, lambda m: None))
    assert counts == {"valid": 1, "invalid": 0, "skipped": 2}
    assert measured(store)[-1]["attempt"] == 1  # a new job attempt, so a new prompt (not left over on fs/obj)


def test_one_prefill_serves_every_rep_of_every_tier(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    eng = FakeEngine()
    points = pts("gpu", 5) + pts("cpu", 5) + pts("fs", 5) + pts("recompute", 2)
    counts = asyncio.run(run_deployment(points, Peers(eng), 64, store, max_attempts=3, log=lambda m: None))
    assert counts["valid"] == 17 and counts["invalid"] == 0
    assert eng.loads_off == 1 + 1 + 2  # the warm-up, one write, plus the two recompute reads
    rows = measured(store)
    assert rows[0]["engine_ttft_s"] == pytest.approx(0.02)  # the engine's own TTFT, next to the wall clock
    assert [r["tier"] for r in rows[:4]] == ["gpu", "cpu", "fs", "recompute"]  # rep 0, in TIER_ORDER
    assert [r["read_index"] for r in rows] == list(range(17)) and len({r["job"] for r in rows}) == 1


def test_p2p_reps_reset_the_reader_so_every_rep_pulls(tmp_path):
    a, b = FakeEngine(host="10.0.0.1"), FakeEngine(host="10.0.0.2")
    store = ResultStore(tmp_path / "r.jsonl")
    counts = asyncio.run(run_deployment(pts("p2p", 3), Peers(a, b, "10.0.0.1"), 64, store, 3, lambda m: None))
    # the reader's warm-up is a pull too (2 blocks of 64), so the first measured pull is not its first
    assert counts["valid"] == 3 and b.cached == {"device": 0.0, "host": 0.0, "disk": 0.0, "p2p": 3 * 960.0 + 128}
    assert a.loads_off == 1 + 1  # the warm-up, then one prefill for the whole job


def test_p2p_reads_pull_from_the_other_peer():
    a, b = FakeEngine(host="10.0.0.1"), FakeEngine(host="10.0.0.2")
    row = asyncio.run(run_point(Point("s1", "m", "p2p", 1000, 0), 64, a, reader=b, source_host="10.0.0.1"))
    assert row["valid"] and row["cached_tokens"] == 960
    assert b.cached["p2p"] == 960 and a.cached["p2p"] == 0


def test_block_size_from_metrics():
    eng = FakeEngine(block_size=528)
    assert block_size(asyncio.run(eng.metrics())) == 528
    with pytest.raises(LookupError):
        block_size([])
