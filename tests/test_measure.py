"""The job protocol against a fake engine that mimics vLLM's metrics. No GPU needed."""

import asyncio

import httpx

from kv_tier_bench import constants as C
from kv_tier_bench import measure
from kv_tier_bench.grid import Point
from kv_tier_bench.prom import Sample

measure.SETTLE_POLL_S = 0.0


class FakeEngine:
    """Holds prefixes in 'gpu', 'cpu' and one secondary tier, and reports cached tokens by source.

    Peers register by host, so a `remote_kv_source` read pulls from the named peer's CPU tier, as vLLM's p2p
    tier does. `fail_after` makes the engine unreachable after that many requests (a preempted node).
    """

    peers: dict = {}

    def __init__(
        self,
        secondary: str = "disk",
        drop_stores: bool = False,
        serve_from: str | None = None,
        host: str = "10.0.0.1",
        block_size: int = 64,
        fail_after: int | None = None,
    ):
        self.host, self.block_size, self.fail_after, self.requests = host, block_size, fail_after, 0
        FakeEngine.peers[host] = self
        self.ttft_sum = self.ttft_count = 0.0
        self.secondary = secondary
        self.drop_stores = drop_stores
        self.serve_from = serve_from  # force a wrong source, to test the validity check
        self.gpu, self.cpu, self.sec = set(), set(), set()
        self.cached = {"device": 0.0, "host": 0.0, "disk": 0.0, "p2p": 0.0}
        self.written = 0.0
        self.dropped = 0.0
        self.loads_off = 0  # requests with loads off: the job's write, plus recompute reads

    async def metrics(self):
        s = [Sample(C.M_CACHED_BY_SOURCE + "_total", (("source", k),), v) for k, v in self.cached.items()]
        s.append(Sample(C.M_TIER_WRITE_BYTES + "_total", (("tier", "1:x"),), self.written))
        s.append(Sample(C.M_TIER_STORES_DROPPED + "_total", (), self.dropped))
        s.append(Sample(C.M_TTFT + "_sum", (), self.ttft_sum))
        s.append(Sample(C.M_TTFT + "_count", (), self.ttft_count))
        s.append(Sample("vllm:cache_config_info", (("block_size", str(self.block_size)), ("engine", "0")), 1.0))
        return s

    async def reset_prefix_cache(self, reset_external=False):
        self.gpu.clear()
        if reset_external:
            self.cpu.clear()
        return True

    async def complete(self, tokens, max_tokens=1, kv_transfer_params=None):
        self.requests += 1
        if self.fail_after is not None and self.requests > self.fail_after:
            raise httpx.ConnectError("connection refused")
        key = tuple(tokens[:-1])
        params = kv_transfer_params or {}
        self.ttft_sum += 0.02
        self.ttft_count += 1
        if params.get("max_load_tokens") == 0:
            self.loads_off += 1
        if "remote_kv_source" in params:
            source = FakeEngine.peers[params["remote_kv_source"]["remote_host"]]
            if key in self.gpu:  # a local GPU hit wins over the pull, as in vLLM
                self.cached["device"] += len(key)
            elif key in source.cpu:
                self.cached["p2p"] += len(key)
        elif params.get("max_load_tokens") != 0:
            src = None
            if key in self.gpu:
                src = "device"
            elif key in self.cpu:
                src = "host"
            elif key in self.sec:
                src = self.secondary
            if src:
                self.cached[self.serve_from or src] += len(key)
        self.gpu.add(key)
        self.cpu.add(key)
        if self.drop_stores:
            self.dropped += 1
        else:
            self.sec.add(key)
            self.written += len(key) * 2
        return 0.01, {}


def run(point, engine):
    return asyncio.run(measure.run_point(point, block_size=64, writer=engine))


def test_each_tier_is_served_from_its_own_source():
    for tier in ("gpu", "cpu", "fs"):
        row = run(Point("s1", "m", tier, 1000, 0), FakeEngine())
        assert row["valid"], (tier, row)
        assert row["prefix_tokens"] == 960 and row["cached_tokens"] == 960


def test_recompute_is_valid_only_without_hits():
    assert run(Point("s1", "m", "recompute", 1000, 0), FakeEngine())["valid"]


def test_wrong_source_is_invalid_not_dropped():
    row = run(Point("s1", "m", "fs", 1000, 0), FakeEngine(serve_from="host"))
    assert not row["valid"] and "source=disk" in row["reason"]


def test_dropped_stores_invalidate_the_point():
    row = run(Point("s1", "m", "fs", 1000, 0), FakeEngine(drop_stores=True))
    assert not row["valid"] and "stores dropped" in row["reason"]


def test_read_params_send_an_empty_tier_list_for_cpu():
    assert measure.read_params("cpu", None, "x") == {"kv_load_tiers": []}  # empty = CPU only
    assert measure.read_params("recompute", None, "x") == {"max_load_tokens": 0}
    p2p = measure.read_params("p2p", "10.0.0.2", "x")["remote_kv_source"]
    assert p2p["remote_host"] == "10.0.0.2" and p2p["remote_port"] == C.P2P_PORT


def test_reads_find_the_tier_own_counters_whatever_the_type_is_called():
    class WithTierCounters(FakeEngine):
        async def metrics(self):
            n = self.requests  # each request adds 1 MiB on the secondary tier and 2 MiB on the CPU hop
            return await super().metrics() + [
                Sample(C.M_TIER_READ_BYTES + "_total", (("tier", "1:SomeFsName"),), n * 2**20),
                Sample(C.M_TIER_READ_BYTES + "_total", (("tier", "0:primary"),), n * 2 * 2**20),
            ]

    row = run(Point("s1", "m", "fs", 1000, 0), WithTierCounters())
    assert row["read_bytes"] == 2**20  # index 1: the fs tier's own series, not fs + the CPU hop
    assert row["read_bytes_by_tier"] == {"0:primary": 2 * 2**20, "1:SomeFsName": 2**20}
    assert row["read_time_s"] == 0.0  # the tier exposes no read-time series
    assert run(Point("s1", "m", "cpu", 1000, 0), WithTierCounters())["read_bytes"] == 2 * 2**20  # index 0


def test_tier_read_never_lets_the_cpu_hop_stand_in_for_another_tier():
    cpu_only = {"0:cpu": 5.0}
    assert measure.tier_read(cpu_only, {"0:cpu": 1.0}, 1) == (None, None)  # e.g. p2p not counted here
    assert measure.tier_read({"fs": 3.0, "cpu": 4.0}, {}, 1) == (7.0, 0.0)  # unexpected format: the sum
    assert measure.tier_read({}, {}, 1) == (0.0, 0.0)  # no series at all


def test_settle_follows_the_secondary_tier_label_from_the_scrape():
    class WithCpuWrites(FakeEngine):
        async def metrics(self):  # the CPU tier also counts writes, under index 0
            cpu = Sample(C.M_TIER_WRITE_BYTES + "_total", (("tier", "0:primary"),), self.requests * 10**6)
            return await super().metrics() + [cpu]

    # the fake's secondary write series is labelled "1:x", a type name the driver has never seen
    row = run(Point("s1", "m", "fs", 1000, 0), WithCpuWrites())
    assert row["valid"] and row["write_bytes"] == 960 * 2  # the secondary tier only, not the CPU writes


def test_retry_uses_a_fresh_prompt():
    eng = FakeEngine()
    first = run(Point("s1", "m", "fs", 1000, 0), eng)
    retry = asyncio.run(measure.run_point(Point("s1", "m", "fs", 1000, 0), 64, eng, attempt=1))
    assert first["valid"] and retry["valid"] and retry["attempt"] == 1
    assert len(eng.sec) == 2  # two distinct prefixes were written
