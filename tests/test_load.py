"""Concurrent load (stage s2): the grid's batches, the batch protocol against fake engines, and the batch cost table."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml

from kv_tier_bench import measure
from kv_tier_bench.cli import main
from kv_tier_bench.cost import CostInputs
from kv_tier_bench.grid import Job, Point, expand, group_jobs, job_of, load_experiment, stage_models
from kv_tier_bench.load import cost_row, summarize
from kv_tier_bench.run import Peers, run_deployment
from kv_tier_bench.store import ResultStore

from .test_measure import FakeEngine

ROOT = Path(__file__).parents[1]
EXP = load_experiment(ROOT / "configs" / "experiment.yaml")
N_S2 = (1032, 215)  # (points, jobs)


# -- grid ------------------------------------------------------------------------------------------------------


def test_s1_ids_are_unchanged_and_all_measured():
    """Adding concurrency must not rename a single s1 point: every one is already done in results/s1.jsonl."""
    done = {r["id"] for r in ResultStore(ROOT / "results" / "s1.jsonl").rows() if r.get("valid")}
    pts = expand(EXP, "s1")
    assert {p.concurrency for p in pts} == {1}
    assert all(p.id in done for p in pts), [p.id for p in pts if p.id not in done][:3]
    assert stage_models(EXP, "s1") == ["qwen3.8-27b", "qwen3-vl-32b"]


def test_concurrency_shows_in_ids_only_above_one():
    assert Point("s2", "m", "cpu", 4096, 0).id == "s2/m/cpu/4096/r0"
    assert Point("s2", "m", "cpu", 4096, 0, 8).id == "s2/m/cpu/4096/r0/c8"
    assert job_of(Point("s2", "m", "cpu", 4096, 0, 8)).id == "s2/m/fs/4096/c8"


def test_every_prefix_of_a_batch_gets_its_own_prompt():
    j = Job("s2", "m", "fs", 4096, 4)
    seeds = j.prompt_seeds(0)
    assert len(set(seeds)) == 4 and seeds == j.prompt_seeds(0)
    assert set(j.prompt_seeds(1)).isdisjoint(seeds)  # a retry writes fresh prefixes
    one = Job("s1", "m", "fs", 4096)
    assert one.prompt_seeds(2) == [one.seed(2)]  # a one-request job keeps its s1 prompt


def test_stage2_grid():
    pts = expand(EXP, "s2")
    assert (len(pts), len(group_jobs(pts))) == N_S2
    assert {p.model for p in pts} == {"qwen3.5-9b", "qwen3-8b"}
    assert {p.concurrency for p in pts} == {1, 2, 4, 8, 16}
    assert {p.tier for p in pts} == {"recompute", "gpu", "cpu", "fs", "fs-shared", "obj", "obj-std", "p2p"}
    assert len({p.id for p in pts}) == len(pts)
    have = {(p.model, p.target_tokens, p.concurrency) for p in pts}
    assert ("qwen3.5-9b", 131072, 8) in have and ("qwen3.5-9b", 131072, 16) not in have  # 64 GiB: too big
    assert ("qwen3-8b", 131072, 2) in have and ("qwen3-8b", 131072, 4) not in have  # 18 GiB per prefix
    # recompute batches are the baseline under load, so they get every rep
    reps = {p.rep for p in pts if p.tier == "recompute"}
    assert reps == set(range(EXP["stages"]["s2"]["recompute_reps"])) == {0, 1, 2}


def test_a_job_holds_one_concurrency():
    for job, ps in group_jobs(expand(EXP, "s2")).items():
        assert {p.concurrency for p in ps} == {job.concurrency}


# -- the batch protocol ------------------------------------------------------------------------------------------


class Overlapping(FakeEngine):
    """Each request takes a moment, so requests sent together really are in flight together."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.in_flight = self.max_in_flight = 0
        self.prompts: list[tuple] = []

    async def complete(self, tokens, max_tokens=1, kv_transfer_params=None):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            self.prompts.append(tuple(tokens))
            return await super().complete(tokens, max_tokens, kv_transfer_params)
        finally:
            self.in_flight -= 1


def run_batch(points, writer, **kw):
    return asyncio.run(measure.run_job(job_of(points[0]), points, 64, writer, **kw))


def test_a_batch_writes_distinct_prefixes_one_at_a_time_and_reads_them_at_once():
    eng = Overlapping()
    rows = run_batch([Point("s2", "m", "cpu", 1000, 0, 4)], eng)
    row = rows[0]
    assert row["valid"], row
    assert row["concurrency"] == 4 and row["expected_cached"] == row["cached_tokens"] == 4 * 960
    assert len(row["read_s_each"]) == 4 and len(row["write_s_each"]) == 4
    assert row["read_s"] >= max(row["read_s_each"])  # first send to last response
    writes = eng.prompts[:4]
    assert len(set(writes)) == 4  # four different prefixes
    assert eng.prompts[4:] == writes or sorted(eng.prompts[4:]) == sorted(writes)
    assert eng.max_in_flight == 4  # the reads overlapped; the writes did not (else max would count them)


def test_writes_never_overlap():
    eng = Overlapping()
    eng.max_in_flight = 0
    job = Job("s2", "m", "fs", 1000, 4)

    async def go():
        await measure.run_job(job, [], 64, eng)  # writes only

    asyncio.run(go())
    assert eng.max_in_flight == 1 and len(eng.sec) == 4


def test_a_batch_is_invalid_if_one_prefix_is_missing():
    class LosesOne(FakeEngine):
        async def reset_prefix_cache(self, reset_external=False):
            await super().reset_prefix_cache(reset_external)
            self.cpu.discard(next(iter(sorted(self.cpu))))  # one batch-mate was evicted
            return True

    row = run_batch([Point("s2", "m", "cpu", 1000, 0, 4)], LosesOne())[0]
    assert not row["valid"] and "served 2880 of 3840" in row["reason"]


def test_each_p2p_request_is_its_own_pull():
    seen = []

    class Reader(FakeEngine):
        async def complete(self, tokens, max_tokens=1, kv_transfer_params=None):
            if kv_transfer_params and "remote_kv_source" in kv_transfer_params:
                seen.append(kv_transfer_params["remote_kv_source"]["kv_request_id"])
            return await super().complete(tokens, max_tokens, kv_transfer_params)

    writer, reader = FakeEngine(host="10.0.0.1"), Reader(host="10.0.0.2")
    row = run_batch([Point("s2", "m", "p2p", 1000, 0, 3)], writer, reader=reader, source_host="10.0.0.1")[0]
    assert row["valid"] and row["cached_by_source"]["p2p"] == 3 * 960
    assert len(set(seen)) == 3 and all(s.endswith(f"#i{i}") for i, s in enumerate(seen))


def test_one_failed_request_cancels_its_batch_mates():
    cancelled = []

    async def ok():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    async def bad():
        await asyncio.sleep(0)
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):  # unwrapped, so run.py's handlers still match it
        asyncio.run(measure.all_or_nothing([ok(), bad(), ok()]))
    assert len(cancelled) == 2


def test_batches_run_and_resume_through_the_sweep_loop(tmp_path):
    store = ResultStore(tmp_path / "r.jsonl")
    points = [Point("s2", "m", t, 1000, r, 4) for r in range(2) for t in ("gpu", "cpu", "recompute")]
    counts = asyncio.run(run_deployment(points, Peers(Overlapping()), 64, store, 3, log=lambda m: None))
    assert counts == {"valid": 6, "invalid": 0, "skipped": 0}
    rows = [r for r in store.rows() if r.get("kind") != "job_start"]
    assert {r["id"] for r in rows} == {p.id for p in points}
    again = asyncio.run(run_deployment(points, Peers(Overlapping()), 64, ResultStore(store.path), 3, lambda m: None))
    assert again["skipped"] == 6


# -- results under load ------------------------------------------------------------------------------------------

GIB = 1024**3


def _row(tier, c, read_s, each=None, **kw):
    r = {"stage": "s2", "model": "m", "prefix_tokens": 4096, "concurrency": c, "tier": tier, "valid": True}
    return r | {"read_s": read_s} | ({"read_s_each": each} if each else {}) | kw


def test_load_summary_keys_on_concurrency():
    rows = [
        _row("cpu", 4, 1.0, [0.5, 0.6, 0.9, 1.0], read_bytes=2 * GIB),
        _row("cpu", 4, 2.0, [1.0, 1.0, 1.0, 2.0], read_bytes=2 * GIB),
        _row("cpu", 1, 0.25),
        _row("cpu", 4, 9.0, valid=False),
        _row("cpu", 4, 9.0, stage="s1"),
    ]
    s = summarize(rows, "s2")
    assert s.batch_s[("m", "cpu", 4096, 4)] == 1.5
    assert s.ttft_p50_s[("m", "cpu", 4096, 4)] == pytest.approx((0.75 + 1.0) / 2)
    assert s.ttft_max_s[("m", "cpu", 4096, 4)] == 1.5
    assert s.batch_gib_s[("m", "cpu", 4096, 4)] == pytest.approx(1.5)  # 2 GiB/1 s and 2 GiB/2 s
    assert s.batch_s[("m", "cpu", 4096, 1)] == 0.25 and s.reps[("m", "cpu", 4096, 4)] == 2


def test_single_request_summary_ignores_batches():
    from kv_tier_bench.cost import summarize as summarize_one

    rows = [_row("cpu", 1, 0.25, job="j", attempt=0), _row("cpu", 4, 9.0, job="j4", attempt=0)]
    assert summarize_one(rows, "s2").read_s == {("m", "cpu", 4096): 0.25}


def test_batch_cost_charges_the_batch_wall_time_over_every_token_it_serves():
    c = CostInputs(gpu_hour_usd=3.6, gpus=1, storage_gib_month_usd="included")
    row = cost_row("cpu", 250_000, 4, 10**9, batch_s=1.0, recompute_batch_s=4.0, c=c)
    assert row["load_usd_per_1m"] == pytest.approx(0.001)  # 1 s of a $3.60/h GPU for 1M tokens
    assert row["tokens_per_s"] == 1_000_000 and row["recompute_tokens_per_s"] == 250_000
    assert row["saving_vs_recompute"] == pytest.approx(0.75)


def test_cli_cost_and_charts_under_load(tmp_path, capsys):
    path = tmp_path / "s2.jsonl"
    store = ResultStore(path)
    for i, (tier, t) in enumerate([("cpu", 0.5), ("recompute", 2.0), ("cpu", 0.25)]):
        c = 1 if i == 2 else 4
        store.append(
            {"id": f"x{i}", "stage": "s2", "model": "qwen3.5-9b", "tier": tier, "prefix_tokens": 4096}
            | {"concurrency": c, "valid": True, "read_s": t, "read_bytes": GIB if tier == "cpu" else None}
            | {"job": f"j{c}", "attempt": 0, "write_s": 0.4}
        )
    assert main(["cost", "--stage", "s2", "--results", str(path)]) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(out) == 1  # the c=1 cpu batch has no recompute batch to compare with
    assert (out[0]["concurrency"], out[0]["batch_s"], out[0]["recompute_batch_s"]) == (4, 0.5, 2.0)
    assert out[0]["saving_vs_recompute"] == pytest.approx(0.75) and out[0]["batch_gib_s"] == 2.0
    assert main(["charts", "--stage", "s2", "--results", str(path), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "load-s2-qwen3.5-9b.png").stat().st_size > 0
    assert (tmp_path / "ttft-s2-qwen3-8b.png").exists()  # every model of the stage gets its charts
    assert not (tmp_path / "ttft-s2-qwen3.8-27b.png").exists()  # an s1 model does not


def test_cli_dry_run_covers_the_whole_stage(tmp_path, capsys):
    total = 0
    for model in ("qwen3.5-9b", "qwen3-8b"):
        for dep in ("fs", "fs-shared", "obj", "obj-std", "p2p"):
            assert main(["run", "--stage", "s2", "--model", model, "--deploy", dep, "--dry-run",
                         "--results", str(tmp_path / "s2.jsonl")]) == 0  # fmt: skip
            line = capsys.readouterr().err
            assert line.startswith(f"{line.split()[0]} of ")  # sweep.sh matches "0 of " to skip a deployment
            total += int(line.split()[2])
    assert total == N_S2[0]


def test_serve_cmd_quotes_the_yarn_override(capsys):
    import shlex

    assert main(["serve-cmd", "--stage", "s2", "--model", "qwen3-8b", "--deploy", "p2p", "--host-ip", "10.0.0.5"]) == 0
    argv = shlex.split(capsys.readouterr().out)
    over = next(a for a in argv if a.startswith("--hf-overrides="))
    assert json.loads(over.split("=", 1)[1])["rope_scaling"]["factor"] == 5.0


def test_standard_class_gets_its_own_bucket_and_the_obj_setup():
    from kv_tier_bench.launch import docker_run

    def kv(dep):
        args = docker_run(EXP, "s2", "qwen3.5-9b", dep, "10.0.0.5", s3_env="/k.env")
        cfg = json.loads(args[args.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
        return args, cfg["secondary_tiers"][0]

    std_args, std = kv("obj-std")
    enh_args, enh = kv("obj")
    assert std["type"] == enh["type"] == "obj"
    assert (std["store_config"]["bucket"], enh["store_config"]["bucket"]) == ("kvtb-cache-std", "kvtb-cache")
    for args in (std_args, enh_args):  # keys, and the CUDA 12 NIXL build with its CA bundle, for both classes
        assert "/k.env" in args and "KVTB_NIXL_CUDA_MAJOR=12" in args
    with pytest.raises(ValueError):
        docker_run(EXP, "s2", "qwen3.5-9b", "obj-std", "10.0.0.5")  # no keys


def test_s2_measures_the_disk_and_the_shared_filesystem_as_two_fs_tiers():
    from kv_tier_bench.launch import docker_run

    st = EXP["stages"]["s2"]
    prices = yaml.safe_load((ROOT / "configs" / "prices.yaml").read_text())["storage_gib_month_usd"]
    for dep, mount, price in (("fs", "/mnt/kv", 0.053), ("fs-shared", "/mnt/fs", 0.08)):
        args = docker_run(EXP, "s2", "qwen3.5-9b", dep, "10.0.0.5")
        extra = json.loads(args[args.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
        tier = extra["secondary_tiers"][0]
        assert tier["type"] == "fs" and tier["root_dir"] == mount
        mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
        assert f"{mount}:{mount}" in mounts
        assert {"/mnt/kv:/mnt/kv", "/mnt/fs:/mnt/fs"} - {f"{mount}:{mount}"} - set(mounts)  # not the other one
    assert st["disk_mount"] == "/mnt/kv" and st["shared_fs_mount"] == "/mnt/fs"
    assert prices[st["disk_price"]] == 0.053 and prices[st["shared_fs_price"]] == 0.08


def test_relabel_moves_fs_tier_rows_to_fs_shared(tmp_path):
    import runpy

    relabel = runpy.run_path(str(ROOT / "scripts" / "relabel-fs-shared.py"), run_name="lib")["relabel"]
    rows = [
        {"id": "s2/qwen3.5-9b/fs/4096#job", "kind": "job_start", "attempt": 0, "points": 12},
        {"id": "s2/qwen3.5-9b/fs/4096/r0", "job": "s2/qwen3.5-9b/fs/4096", "tier": "fs", "stage": "s2"},
        {"id": "s2/qwen3.5-9b/gpu/4096/r0", "job": "s2/qwen3.5-9b/fs/4096", "tier": "gpu", "stage": "s2"},
        {"id": "s2/qwen3-8b/fs/16384/r1/c4", "job": "s2/qwen3-8b/fs/16384/c4", "tier": "fs", "stage": "s2"},
        {"id": "s1/qwen3.8-27b/fs/4096/r0", "job": "s1/qwen3.8-27b/fs/4096", "tier": "fs", "stage": "s1"},
    ]
    out, n = relabel(rows, "s2")
    assert n == 2
    assert out[0] == rows[0] and out[2] == rows[2] and out[4] == rows[4]  # markers, other tiers, other stages
    assert out[1] == {
        "id": "s2/qwen3.5-9b/fs-shared/4096/r0",
        "job": "s2/qwen3.5-9b/fs-shared/4096",
        "tier": "fs-shared",
        "stage": "s2",
    }
    assert out[3]["id"] == "s2/qwen3-8b/fs-shared/16384/r1/c4" and out[3]["job"] == "s2/qwen3-8b/fs-shared/16384/c4"
    assert relabel(out, "s2")[1] == 0  # idempotent
    # the relabelled ids are exactly the fs-shared points the grid expects
    expected = {p.id for p in expand(EXP, "s2") if p.tier == "fs-shared"}
    assert out[1]["id"] in expected and out[3]["id"] in expected


def test_s1_keeps_its_six_tiers():
    assert {p.tier for p in expand(EXP, "s1")} == {"recompute", "gpu", "cpu", "fs", "obj", "p2p"}


def test_cli_lists_a_stage_models(capsys):
    assert main(["models", "--stage", "s2"]) == 0
    assert capsys.readouterr().out.split() == ["qwen3.5-9b", "qwen3-8b"]
