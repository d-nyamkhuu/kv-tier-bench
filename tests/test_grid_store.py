from pathlib import Path

import pytest

from kv_tier_bench.grid import (
    CPU_TIERS,
    FIT_FRACTION,
    GIB,
    Job,
    Point,
    align_down,
    expand,
    gpu_kv_budget,
    group_jobs,
    load_experiment,
)
from kv_tier_bench.models import load_shape
from kv_tier_bench.store import ResultStore

EXP = load_experiment(Path(__file__).parents[1] / "configs" / "experiment.yaml")
N_S1 = (390, 45)  # (points, jobs): the counts docs/plan.md quotes


def test_point_id_and_job_seed_are_stable():
    assert Point("s1", "qwen3.8-27b", "fs", 4096, 2).id == "s1/qwen3.8-27b/fs/4096/r2"
    j = Job("s1", "qwen3.8-27b", "fs", 4096)
    assert j.seed() == Job("s1", "qwen3.8-27b", "fs", 4096).seed()
    assert j.seed() != Job("s1", "qwen3.8-27b", "fs", 8192).seed()
    assert j.seed(1) != j.seed(0)  # a retry gets a fresh prompt


def test_stage1_caps_the_standard_model_at_64k():
    pts = expand(EXP, "s1")
    vl = {p.target_tokens for p in pts if p.model == "qwen3-vl-32b"}
    hy = {p.target_tokens for p in pts if p.model == "qwen3.8-27b"}
    assert max(vl) == 65536 and max(hy) == 131072
    assert {p.tier for p in pts} == {"recompute", "gpu", "cpu", "fs", "obj", "p2p"}
    assert len({p.id for p in pts}) == len(pts)


def test_align_down():
    assert align_down(4096, 800) == 4000
    assert align_down(500, 800) == 800  # never below one block
    with pytest.raises(ValueError):
        align_down(10, 0)


def test_store_skips_done_and_survives_a_torn_line(tmp_path):
    path = tmp_path / "r.jsonl"
    s = ResultStore(path)
    s.append({"id": "a", "valid": True})
    with path.open("a") as f:
        f.write('{"id": "b", "val')  # preempted mid-write
    s2 = ResultStore(path)
    assert s2.is_done("a") and not s2.is_done("b")
    s2.append({"id": "c"})
    assert [r["id"] for r in ResultStore(path).rows()] == ["a", "c"]


def test_invalid_attempts_are_retried_until_max(tmp_path):
    s = ResultStore(tmp_path / "r.jsonl")
    s.append({"id": "p", "attempt": 0, "valid": False})
    assert not s.is_done("p", max_attempts=3) and s.next_attempt("p") == 1
    s.append({"id": "p", "attempt": 1, "valid": True})
    assert s.is_done("p", max_attempts=3)
    s.append({"id": "q", "attempt": 0, "valid": False})
    s.append({"id": "q", "attempt": 1, "valid": False})
    s.append({"id": "q", "attempt": 2, "valid": False})
    assert ResultStore(tmp_path / "r.jsonl").is_done("q", max_attempts=3)


def test_no_point_overflows_the_tiers_it_uses():
    root = Path(__file__).parents[1]
    for stage in EXP["stages"]:
        st = EXP["stages"][stage]
        for p in expand(EXP, stage):
            shape = load_shape(p.model, root / EXP["models"][p.model]["config"])
            need = p.concurrency * shape.prefix_bytes(p.target_tokens, tp=st["tp"])  # a batch is in memory at once
            assert need <= FIT_FRACTION * gpu_kv_budget(EXP, stage, p.model), p.id
            if p.tier in CPU_TIERS:
                assert need <= FIT_FRACTION * st["cpu_tier_gib"] * GIB, p.id


def test_random_prompts_never_contain_special_tokens():
    import json

    from kv_tier_bench.client import prompt_tokens

    root = Path(__file__).parents[1]
    hi = max(prompt_tokens(1, 50000))
    for m in EXP["models"].values():
        cfg = json.loads((root / m["config"]).read_text())
        text = cfg.get("text_config", cfg)
        special = [v for d in (cfg, text) for k, v in d.items() if k.endswith("token_id") and isinstance(v, int)]
        assert hi < min(special), (m["hf_id"], min(special))


def test_jobs_share_one_prefill_and_recompute_gets_one_rep():
    points = expand(EXP, "s1")
    assert (len(points), len(group_jobs(points))) == N_S1
    job, pts = next((j, ps) for j, ps in group_jobs(expand(EXP, "s1")).items() if j.deployment == "fs")
    assert [p.tier for p in pts[:4]] == ["gpu", "cpu", "fs", "recompute"]  # rep 0, read in TIER_ORDER
    assert sum(p.tier == "recompute" for p in pts) == EXP["recompute_reps"]
    assert sum(p.tier == "fs" for p in pts) == EXP["reps"]
