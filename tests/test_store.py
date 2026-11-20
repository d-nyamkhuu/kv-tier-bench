"""Stage s3: the CPU-only KV store (MooncakeStoreConnector) beside p2p. Grid, config, launch, validity and cost."""

import asyncio
import json
from pathlib import Path

from kv_tier_bench import constants as C
from kv_tier_bench import measure
from kv_tier_bench.cli import main
from kv_tier_bench.grid import Point, expand, group_jobs, job_of, load_experiment
from kv_tier_bench.launch import MOONCAKE_CONFIG, VLLM_IMAGE, docker_run
from kv_tier_bench.prom import Sample
from kv_tier_bench.tierconfig import deployment_for, kv_transfer_config, mooncake_config

from .test_measure import FakeEngine

ROOT = Path(__file__).parents[1]
EXP = load_experiment(ROOT / "configs" / "experiment.yaml")
S3 = EXP["stages"]["s3"]
KVB = 100  # bytes per token in the fake engine


# -- grid --------------------------------------------------------------------------------------------------------


def test_s3_runs_its_baselines_on_one_vm_and_each_remote_tier_on_its_own():
    jobs = group_jobs(expand(EXP, "s3"))
    assert {j.deployment for j in jobs} == {"base", "p2p", "store", "store-p2p"}
    for job, ps in jobs.items():
        want = {"recompute", "gpu", "cpu"} if job.deployment == "base" else {job.deployment}
        assert {p.tier for p in ps} == want, job.id
    # base is the CPU tier alone: the same connector as s1/s2, no secondary tier
    cfg = kv_transfer_config("base", S3, "qwen3-8b", "unused")
    assert cfg["kv_connector"] == "OffloadingConnector"
    assert cfg["kv_connector_extra_config"]["secondary_tiers"] == []
    # the base tiers move only in s3: s1 and s2 keep their fs deployment (and their job ids)
    assert deployment_for("s2", "cpu") == deployment_for("s1", "recompute") == "fs"
    assert job_of(Point("s3", "m", "cpu", 4096, 0, 8)).id == "s3/m/base/4096/c8"


def test_store_batches_fit_the_pool():
    pts = [p for p in expand(EXP, "s3") if p.tier == "store"]
    have = {(p.model, p.target_tokens, p.concurrency) for p in pts}
    assert ("qwen3-8b", 131072, 2) in have and (
        "qwen3-8b",
        131072,
        4,
    ) not in have  # 72 GiB: the GPU's KV room, not the pool, stops it
    assert len({p.id for p in pts}) == len(pts)


# -- config and launch -------------------------------------------------------------------------------------------


def test_mooncake_config_is_standalone_store_over_tcp():
    cfg = mooncake_config(S3, "10.0.0.7")
    assert cfg["mode"] == "standalone-store" and cfg["global_segment_size"] == 0  # vLLM is a pure requester
    assert cfg["master_server_address"] == "10.0.0.7:50051" and cfg["protocol"] == "tcp"
    assert cfg["local_buffer_size"] > 0
    assert kv_transfer_config("store", S3, "qwen3-8b", "unused") == {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
    }


def test_store_p2p_writer_holds_the_pool_and_the_reader_requests():
    writer = mooncake_config(S3, "10.0.0.5", holder=True)
    reader = mooncake_config(S3, "10.0.0.5")
    assert writer["mode"] == "embedded" and writer["global_segment_size"] == S3["store"]["holder_pool_gib"] * 2**30
    assert reader["mode"] == "standalone-store" and reader["global_segment_size"] == 0
    assert writer["master_server_address"] == reader["master_server_address"] == "10.0.0.5:50051"


def test_store_launch_mounts_the_config_and_names_the_requester():
    cmd = docker_run(EXP, "s3", "qwen3-8b", "store", "10.0.0.5", mooncake_config="/home/kvtb/mooncake.json")
    head = cmd[: cmd.index(VLLM_IMAGE)]
    env = {head[i + 1].split("=", 1)[0]: head[i + 1].split("=", 1)[1] for i, a in enumerate(head) if a == "-e"}
    assert f"/home/kvtb/mooncake.json:{MOONCAKE_CONFIG}:ro" in head
    assert env["MOONCAKE_CONFIG_PATH"] == MOONCAKE_CONFIG
    assert env["MOONCAKE_REQUESTER_LOCAL_HOSTNAME"] == "10.0.0.5"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["MC_TCP_LANES_PER_PEER"] == "16" and env["VLLM_MOONCAKE_LOAD_RECV_THREADS"] == "4"
    p2p = docker_run(EXP, "s3", "qwen3-8b", "p2p", "10.0.0.5", mooncake_config="/home/kvtb/mooncake.json")
    assert not any("MOONCAKE" in a for a in p2p)


def test_store_launch_without_its_config_fails():
    try:
        docker_run(EXP, "s3", "qwen3-8b", "store", "10.0.0.5")
    except ValueError as e:
        assert "mooncake" in str(e).lower()
    else:
        raise AssertionError("no error")


# -- validity ----------------------------------------------------------------------------------------------------


class StoreEngine(FakeEngine):
    """vLLM with MooncakeStoreConnector: no CPU tier; a GPU miss loads from the store and counts the bytes."""

    def __init__(self, label="external_unspecified", lose_bytes=False, fail_puts=False, **kw):
        super().__init__(**kw)
        self.store, self.label, self.lose_bytes, self.fail_puts = set(), label, lose_bytes, fail_puts
        self.ops = {(op, st): 0.0 for op in ("save_put", "load_get") for st in ("ok", "failed")}
        self.failed = {"save_put": 0.0, "load_get": 0.0}
        self.cached[label] = 0.0

    async def metrics(self):
        s = [Sample(C.M_CACHED_BY_SOURCE + "_total", (("source", k),), v) for k, v in self.cached.items()]
        s += [Sample(C.M_TTFT + "_sum", (), self.ttft_sum), Sample(C.M_TTFT + "_count", (), self.ttft_count)]
        for (op, st), v in self.ops.items():
            labels = (("operation", op), ("status", st))
            s.append(Sample(C.M_STORE_BYTES, labels, v))
            s.append(Sample(C.M_STORE_TIME + "_sum", labels, v / 1e9))
        for op, v in self.failed.items():
            s.append(Sample(C.M_STORE_FAILED_KEYS, (("operation", op), ("status", "failed")), v))
        return s

    async def reset_prefix_cache(self, reset_external=False):
        assert not reset_external, "a store read must not ask the connector to reset"
        self.gpu.clear()
        return True

    async def complete(self, tokens, max_tokens=1, kv_transfer_params=None):
        key = tuple(tokens[:-1])
        params = kv_transfer_params or {}
        assert "kv_load_tiers" not in params  # an OffloadingConnector parameter
        self.ttft_sum += 0.02
        self.ttft_count += 1
        if params.get("max_load_tokens") != 0:
            if key in self.gpu:
                self.cached["device"] += len(key)
            elif key in self.store:
                self.cached[self.label] += len(key)
                self.ops[("load_get", "ok")] += len(key) * KVB * (0.5 if self.lose_bytes else 1)
        self.gpu.add(key)
        if self.fail_puts:
            self.failed["save_put"] += 1
        else:
            self.store.add(key)
            self.ops[("save_put", "ok")] += len(key) * KVB
        return 0.01, {}


def read(engine, c=1):
    p = Point("s3", "m", "store", 1000, 0, c)
    return asyncio.run(measure.run_job(job_of(p), [p], 64, engine, kv_bytes_per_token=KVB))[0]


def test_a_store_read_counts_its_bytes_and_is_valid():
    row = read(StoreEngine(), c=4)
    assert row["valid"], row
    assert row["cached_tokens"] == row["expected_cached"] == 4 * 960
    assert row["read_bytes"] == 4 * 960 * KVB and row["read_time_s"] > 0
    assert row["write_bytes"] == 4 * 960 * KVB


def test_any_non_gpu_source_label_counts_for_the_store():
    assert read(StoreEngine(label="host"))["valid"]  # the label is unverified; the store's bytes decide


def test_a_store_read_without_the_bytes_is_invalid():
    row = read(StoreEngine(lose_bytes=True))
    assert not row["valid"] and "load_get" in row["reason"]


def test_failed_puts_count_as_dropped_stores():
    row = read(StoreEngine(fail_puts=True))
    assert not row["valid"] and row["stores_dropped"] == 1


def test_store_p2p_reads_on_the_second_instance_from_the_writers_pool():
    writer = StoreEngine(host="10.0.0.5")
    reader = StoreEngine(host="10.0.0.6")
    reader.store = writer.store  # one pool, held by the writer
    p = Point("s3", "m", "store-p2p", 1000, 0, 2)
    rows = asyncio.run(measure.run_job(job_of(p), [p], 64, writer, reader=reader, kv_bytes_per_token=KVB))
    row = rows[0]
    assert row["valid"], row
    assert row["read_bytes"] == 2 * 960 * KVB and reader.ops[("load_get", "ok")] == row["read_bytes"]
    assert writer.ops[("load_get", "ok")] == 0  # the writer only wrote


# -- cost --------------------------------------------------------------------------------------------------------


def test_s3_prices_the_ram_behind_p2p_and_the_store(tmp_path, capsys):
    rows = []
    for tier, batch_s in (("recompute", 2.0), ("p2p", 0.4), ("store", 0.5)):
        rows.append(
            Point("s3", "qwen3-8b", tier, 65536, 0).as_dict()
            | {
                "job": "j",
                "attempt": 0,
                "prefix_tokens": 65536,
                "valid": True,
                "read_s": batch_s,
                "read_bytes": 9 * 2**30,
            }
        )
    path = tmp_path / "s3.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert main(["cost", "--stage", "s3", "--results", str(path)]) == 0
    out = {r["tier"]: r for r in map(json.loads, capsys.readouterr().out.splitlines())}
    for tier in ("p2p", "store"):
        assert out[tier]["storage_usd_per_1m"] > 0 and out[tier]["storage_usd_per_1m_included"] == 0
        assert out[tier]["total_usd_per_1m"] > out[tier]["total_usd_per_1m_included"]
    assert "storage_usd_per_1m_included" not in out.get("gpu", {})
