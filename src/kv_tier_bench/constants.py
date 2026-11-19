"""vLLM v0.34.0 metric names, source labels and load-tier matchers that the driver relies on.

Everything marked UNVERIFIED is on the day-1 checklist in docs/plan.md. Correct it here once a running vLLM shows
the real value. The container settings (image, env, ports) are in launch.py.
"""

# Per-tier counters, from vllm/v1/kv_offload/tiering @ v0.34.0 (label `tier`: "0:primary" for CPU, "1:fs" etc.).
# UNVERIFIED: whether the exposition adds a `_total` suffix to counters. `prom.find` accepts both forms.
M_TIER_READ_BYTES = "vllm:kv_offload_tiering_read_bytes"
M_TIER_READ_TIME = "vllm:kv_offload_tiering_read_time"
M_TIER_WRITE_BYTES = "vllm:kv_offload_tiering_write_bytes"
M_TIER_WRITE_TIME = "vllm:kv_offload_tiering_write_time"
M_TIER_ACTIVE_CASCADES = "vllm:kv_offload_tiering_active_cascade_jobs"
M_TIER_ACTIVE_PROMOTIONS = "vllm:kv_offload_tiering_active_promotion_jobs"
M_TIER_STORES_DROPPED = "vllm:kv_offload_tiering_backpressure_stores_dropped"
# Histogram (labels tier, result) of each promotion job, from creation to completion, queueing included. p2p reports
# no read time (its READ_TIME series never appears, v0.34.0 docs), so its load time comes from here.
M_TIER_PROMOTION_LATENCY = "vllm:kv_offload_tiering_promotion_latency_seconds"

# MooncakeStoreConnector (s3's store deployment), vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/metrics.
# Labels `operation` (save_exists, save_put, load_get, lookup_exists) and `status`. The store deployment has no
# tiering counters: its writes settle on save_put bytes, and its reads' bytes and time are load_get's.
M_STORE_BYTES = "vllm:mooncake_store_operation_bytes_total"
M_STORE_FAILED_KEYS = "vllm:mooncake_store_operation_failed_keys_total"
M_STORE_TIME = "vllm:mooncake_store_operation_time_seconds"
STORE_OK = "ok"  # the `status` label of a successful operation (seen on v0.34.0, 2026-11-19)

# Cached prompt tokens by source; this is what proves where a hit came from.
M_CACHED_BY_SOURCE = "vllm:prompt_tokens_cached_by_source"
M_TTFT = "vllm:time_to_first_token_seconds"
# Per-request phases, recorded when a request finishes (vllm/v1/metrics/stats.py @ v0.34.0). An async KV load keeps
# the request WAITING_FOR_REMOTE_KVS before its SCHEDULED event, so with one request at a time the queue time is the
# load from CPU/fs/obj/p2p, and the prefill time is the uncached tail plus the first token.
M_QUEUE_TIME = "vllm:request_queue_time_seconds"
M_PREFILL_TIME = "vllm:request_prefill_time_seconds"

# UNVERIFIED label values of M_CACHED_BY_SOURCE{source=...} for each tier under test.
SOURCE_LABEL = {
    "gpu": "device",
    "cpu": "host",
    "fs": "disk",
    "obj": "disk",  # fs and obj share a label; that is why each run deploys one secondary tier only
    "obj-std": "disk",
    "fs-shared": "disk",
    "p2p": "p2p",
    # UNVERIFIED: a hit through a connector other than the OffloadingConnector. s1/s2 rows carry this key with 0.
    "store": "external_unspecified",
    "store-p2p": "external_unspecified",
}

# `kv_load_tiers` matchers ({"medium", "locality"}), from the offloading scheduler @ v0.34.0.
# CPU is always allowed; an empty list means "CPU only". UNVERIFIED for p2p.
LOAD_TIERS = {
    "gpu": [],
    "cpu": [],
    "fs": [{"medium": "STORAGE"}],
    "obj": [{"medium": "STORAGE"}],
    "obj-std": [{"medium": "STORAGE"}],
    "fs-shared": [{"medium": "STORAGE"}],
    "p2p": None,  # None = do not send kv_load_tiers; p2p is driven by remote_kv_source
    "store": None,  # an OffloadingConnector parameter; MooncakeStoreConnector loads whatever it finds
    "store-p2p": None,
}

# Index part of the per-tier counters' `tier` label, "<index>:<type>" (vLLM docs, "Info Metric Labels"): 0 is the
# CPU tier, 1 the deployment's one secondary tier. The driver finds the full label in each scrape by this index,
# so the type part never has to be known. UNVERIFIED: the format; labels without it fall back to the sum.
TIER_INDEX = {"cpu": 0, "fs": 1, "fs-shared": 1, "obj": 1, "obj-std": 1, "p2p": 1}

P2P_PORT = 7777  # llm-d p2p guide's port; vLLM's default is 5710
