import json
from pathlib import Path

import yaml

from kv_tier_bench.cli import main
from kv_tier_bench.grid import load_experiment
from kv_tier_bench.webdata import export

ROOT = Path(__file__).parents[1]
EXP = load_experiment(ROOT / "configs" / "experiment.yaml")
PRICES = yaml.safe_load((ROOT / "configs" / "prices.yaml").read_text())
GIB = 1024**3


def _row(tier, read_s, **kw):
    r = {"stage": "s1", "model": "qwen3-vl-32b", "prefix_tokens": 1024, "block_size": 64, "valid": True}
    r |= {"job": "j", "attempt": 0, "write_s": 0.2, "tier": tier, "read_s": read_s}
    return r | kw


def test_export_medians_sizes_and_prices():
    rows = [
        _row("cpu", 0.06, engine_queue_s=0.01, engine_prefill_s=0.05),
        _row("cpu", 0.08, engine_queue_s=0.03, engine_prefill_s=0.05),
        _row("fs", 0.3, read_bytes=GIB // 4, read_time_s=0.25),
        _row("p2p", 0.2, job="p", write_s=0.4, read_bytes=GIB // 4, read_time_s=0.0, promotion_s=0.125),
        _row("obj", 9.9, valid=False),
    ]
    d = export(rows, EXP, PRICES, "s1", ROOT)
    m = d["models"]["qwen3-vl-32b"]
    assert (m["kv_bytes_per_token"], m["block_size"], m["read_fixed_bytes"]) == (256 * 1024, 64, 0)
    hybrid = d["models"]["qwen3.8-27b"]["kv_bytes_per_token_by_tp"]  # 4 KV heads: TP=8 stores each head twice
    assert (hybrid["4"], hybrid["8"]) == (64 * 1024, 128 * 1024)
    assert d["cpu_tier_gib"] == 48
    # the page recomputes GPU KV capacity for other GPUs from these, so they must give back grid.gpu_kv_budget
    usable = d["gpu_mem_gib"] * d["gpu_memory_utilization"] - m["weights_bytes"] / GIB - d["activation_gib_per_gpu"]
    assert abs(usable - m["gpu_kv_gib"]) < 0.01
    assert m["recompute"] == [{"tokens": 1024, "ttft_s": 0.2}]  # the fs deployment's cold write, not p2p's
    cpu = m["tiers"]["cpu"][0]
    assert (cpu["ttft_s"], cpu["queue_s"], cpu["prefill_s"], cpu["read_bytes"]) == (0.07, 0.02, 0.05, None)
    assert m["tiers"]["fs"][0]["load_gib_s"] == 1.0 and m["tiers"]["fs"][0]["read_bytes"] == GIB // 4
    assert m["tiers"]["p2p"][0]["load_gib_s"] == 2.0 and m["tiers"]["p2p"][0]["queue_s"] is None
    assert "obj" not in m["tiers"]  # invalid reads are left out
    assert d["prices"]["storage_gib_month_usd"]["obj_enhanced"] == PRICES["storage_gib_month_usd"]["obj_enhanced"]


def test_committed_web_data_is_current(tmp_path):
    """web/data/s1.json is what the page loads; it must match a fresh export of results/s1.jsonl."""
    out = tmp_path / "s1.json"
    assert (
        main(["export-web", "--stage", "s1", "--results", str(ROOT / "results" / "s1.jsonl"), "--out", str(out)]) == 0
    )
    assert json.loads(out.read_text()) == json.loads((ROOT / "web" / "data" / "s1.json").read_text())
