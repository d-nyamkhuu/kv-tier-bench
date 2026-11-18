import pytest

from kv_tier_bench import prom
from kv_tier_bench.cost import CostInputs, per_million_tokens, storage_usd, summarize, tier_row
from kv_tier_bench.validity import compare

METRICS = """
# HELP vllm:kv_offload_tiering_read_bytes x
# TYPE vllm:kv_offload_tiering_read_bytes counter
vllm:kv_offload_tiering_read_bytes_total{engine="0",tier="1:fs"} 1048576.0
vllm:kv_offload_tiering_read_bytes_total{engine="0",tier="0:primary"} 10.0
vllm:prompt_tokens_cached_by_source_total{engine="0",source="disk"} 4000.0
vllm:prompt_tokens_cached_by_source_total{engine="0",source="host"} 0.0
"""


def test_prom_find_accepts_total_suffix_and_filters_labels():
    s = prom.parse(METRICS)
    assert prom.find(s, "vllm:kv_offload_tiering_read_bytes", tier="1:fs") == 1048576.0
    assert prom.find(s, "vllm:kv_offload_tiering_read_bytes") == 1048586.0
    assert prom.find(s, "vllm:prompt_tokens_cached_by_source", source="disk") == 4000.0
    assert prom.delta(prom.parse(""), s, "vllm:prompt_tokens_cached_by_source", source="host") == 0.0


def _resp(text):
    return {"choices": [{"text": text}]}


def test_validity_flags_divergence():
    assert compare(_resp("the cat"), _resp("the cat")).same
    diff = compare(_resp("the cat"), _resp("the dog"))
    assert not diff.same and diff.first_divergence == 4
    assert compare(_resp("the"), _resp("the cat")).first_divergence == 3


def test_summary_pairs_each_deployment_with_its_own_cold_writes():
    GIB = 1024**3
    fs = {"stage": "s1", "model": "m", "prefix_tokens": 960, "job": "fs-job", "attempt": 0, "write_s": 1.0}
    fs |= {"write_bytes": 2 * GIB, "write_time_s": 1.0}
    obj = fs | {"job": "obj-job", "write_s": 5.0, "write_bytes": GIB}
    rows = [
        fs | {"tier": "cpu", "valid": True, "read_s": 0.1, "read_bytes": GIB, "read_time_s": 0.5},
        fs | {"tier": "cpu", "valid": True, "read_s": 0.3, "read_bytes": GIB, "read_time_s": 0.5},
        fs | {"tier": "recompute", "valid": True, "read_s": 2.0},
        fs | {"tier": "fs", "valid": False, "read_s": 9.0},  # invalid reads are left out, its write is not
        fs | {"attempt": 1, "write_s": 3.0, "tier": "fs", "valid": True, "read_s": 0.5},
        obj | {"tier": "obj", "valid": True, "read_s": 0.7},
        fs | {"tier": "p2p", "valid": True, "read_s": 0.2, "read_bytes": GIB, "read_time_s": 0.0, "promotion_s": 0.25},
        fs | {"stage": "other", "tier": "cpu", "valid": True, "read_s": 7.0},
    ]
    s = summarize(rows, "s1")
    assert s.read_s[("m", "cpu", 960)] == pytest.approx(0.2) and s.read_s[("m", "fs", 960)] == 0.5
    assert s.read_s[("m", "recompute", 960)] == 2.0  # the single read: a cross-check, not pooled
    assert s.recompute_s == {("m", "fs", 960): 2.0, ("m", "obj", 960): 5.0}  # writes 1.0 and 3.0; obj's own
    assert s.load_gib_s[("m", "cpu", 960)] == 2.0
    assert s.load_gib_s[("m", "p2p", 960)] == 4.0  # no read time: from the promotion latency
    assert s.store_gib_s[("m", "fs", 960)] == 2.0 and s.store_gib_s[("m", "obj", 960)] == 1.0


def test_cost_arithmetic():
    c = CostInputs(gpu_hour_usd=3.6, gpus=1, storage_gib_month_usd=0.11)
    assert per_million_tokens(1.0, 1_000_000, c) == pytest.approx(0.001)  # 1 s of a $3.60/h GPU
    assert storage_usd(1024**3, 730.0, c) == pytest.approx(0.11)  # 1 GiB for one month
    row = tier_row("obj", 10_000, 0, load_s=0.5, recompute_s=1.0, c=c)
    assert row["saving_vs_recompute"] == pytest.approx(0.5)
    free = tier_row("cpu", 10_000, 10**9, 0.1, 1.0, CostInputs(3.6, 1, "included"))
    assert free["storage_usd_per_1m"] == 0.0
    with pytest.raises(ValueError):  # an unknown price (null in prices.yaml) is never silently $0
        CostInputs(3.6, 1, None)
