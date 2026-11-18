"""vLLM v0.33.0 metric names, source labels and load-tier matchers that the driver relies on.

Everything marked UNVERIFIED is on the day-1 checklist in docs/plan.md. Correct it here once a running vLLM shows
the real value. The container settings (image, env, ports) are in launch.py.
"""

# Per-tier counters, from vllm/v1/kv_offload/tiering/base.py @ v0.33.0 (label `tier`, e.g. "1:fs").
# UNVERIFIED: whether the exposition adds a `_total` suffix to counters. `prom.find` accepts both forms.
M_TIER_READ_BYTES = "vllm:kv_offload_tiering_read_bytes"
M_TIER_READ_TIME = "vllm:kv_offload_tiering_read_time"
M_TIER_WRITE_BYTES = "vllm:kv_offload_tiering_write_bytes"
M_TIER_WRITE_TIME = "vllm:kv_offload_tiering_write_time"
M_TIER_ACTIVE_CASCADES = "vllm:kv_offload_tiering_active_cascade_jobs"
M_TIER_ACTIVE_PROMOTIONS = "vllm:kv_offload_tiering_active_promotion_jobs"
M_TIER_STORES_DROPPED = "vllm:kv_offload_tiering_backpressure_stores_dropped"

# Cached prompt tokens by source; this is what proves where a hit came from.
M_CACHED_BY_SOURCE = "vllm:prompt_tokens_cached_by_source"
M_TTFT = "vllm:time_to_first_token_seconds"

# UNVERIFIED label values of M_CACHED_BY_SOURCE{source=...} for each tier under test.
SOURCE_LABEL = {
    "gpu": "device",
    "cpu": "host",
    "fs": "disk",
    "obj": "disk",  # fs and obj share a label; that is why each run deploys one secondary tier only
    "p2p": "p2p",
}

# `kv_load_tiers` matchers ({"medium", "locality"}), from the offloading scheduler @ v0.33.0.
# CPU is always allowed; an empty list means "CPU only". UNVERIFIED for p2p.
LOAD_TIERS = {
    "gpu": [],
    "cpu": [],
    "fs": [{"medium": "STORAGE"}],
    "obj": [{"medium": "STORAGE"}],
    "p2p": None,  # None = do not send kv_load_tiers; p2p is driven by remote_kv_source
}

# Index part of the per-tier counters' `tier` label, "<index>:<type>" (vLLM docs, "Info Metric Labels"): 0 is the
# CPU tier, 1 the deployment's one secondary tier. The driver finds the full label in each scrape by this index,
# so the type part never has to be known. UNVERIFIED: the format; labels without it fall back to the sum.
TIER_INDEX = {"cpu": 0, "fs": 1, "obj": 1, "p2p": 1}

P2P_PORT = 7777  # llm-d p2p guide's port; vLLM's default is 5710
