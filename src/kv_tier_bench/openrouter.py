"""Provider prices from OpenRouter for the models the web simulator has, as a dated snapshot.

The page compares its modeled $ per 1M input, cached input and output tokens with what providers charge.
`kvtb openrouter-prices` fetches every provider endpoint of each model from OpenRouter's public API and writes
web/data/openrouter.json, which is committed. Prices are prices, not costs: they carry margins and promotions.

Decode speed: the API's per-endpoint throughput_last_30m is null for every endpoint, so the p50 throughput each
model's OpenRouter compare page shows is recorded here by hand (SPEED, dated SPEED_AS_OF) and exported as is: a fetch
of prices never re-dates it. The page uses it as the per-request decode speed a provider holds.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

API = "https://openrouter.ai/api/v1/models/{id}/endpoints"
PAGE = "https://openrouter.ai/{id}/providers"

# simulator model id -> OpenRouter model id
MODEL_MAP = {
    "qwen3.8-27b": "qwen/qwen3.8-27b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
    "minimax-m3": "minimax/minimax-m3",
    "glm-5.3": "z-ai/glm-5.3",
    "mistral-large-4": "mistralai/mistral-large-4-0",
    "kimi-k3": "moonshotai/kimi-k3",
    "qwen3.8-2.4t-a95b": "qwen/qwen3.8-2.4t-a95b",
}


# p50 decode throughput (tokens/s per request) and p50 latency (s) on OpenRouter's compare pages, read SPEED_AS_OF.
# None: the page shows "Not available". The pages do not say how providers are weighted.
SPEED_AS_OF = "2026-11-18"
_CMP = "https://openrouter.ai/compare/"
SPEED = {
    "qwen3.8-27b": (28.0, 4.81, _CMP + "qwen/qwen3.8-27b/qwen/qwen3.8-2.4t-a95b"),
    "qwen3.8-2.4t-a95b": (53.0, 1.78, _CMP + "qwen/qwen3.8-27b/qwen/qwen3.8-2.4t-a95b"),
    "gpt-oss-120b": (65.0, 0.62, _CMP + "openai/gpt-oss-120b/google/gemma-4-31b-it"),
    "gemma-4-31b": (148.0, 0.74, _CMP + "openai/gpt-oss-120b/google/gemma-4-31b-it"),
    "deepseek-v4.1-flash": (154.0, 0.23, _CMP + "deepseek/deepseek-v4.1-flash/moonshotai/kimi-k3"),
    "kimi-k3": (101.0, 1.27, _CMP + "deepseek/deepseek-v4.1-flash/moonshotai/kimi-k3"),
    "glm-5.3": (75.0, 0.26, _CMP + "z-ai/glm-5.3/z-ai/glm-5.3-flash"),
    "glm-5.3-flash": (83.0, 0.94, _CMP + "z-ai/glm-5.3/z-ai/glm-5.3-flash"),
    "minimax-m3": (None, None, _CMP + "minimax/minimax-m3/mistralai/mistral-large-4-0"),
    "mistral-large-4": (54.0, 1.79, _CMP + "minimax/minimax-m3/mistralai/mistral-large-4-0"),
}


def speed(sim_id: str) -> dict | None:
    if sim_id not in SPEED:
        return None
    tok_s, latency, url = SPEED[sim_id]
    return {"p50_tok_s": tok_s, "p50_latency_s": latency, "as_of": SPEED_AS_OF, "url": url}


def _per_million(pricing: dict, key: str) -> float | None:
    v = pricing.get(key)
    if v in (None, ""):
        return None
    x = float(v) * 1e6
    return float(f"{x:.6g}") if x > 0 else None


def summarize(endpoints: list[dict]) -> list[dict]:
    """One row per paid provider endpoint: $ per 1M input, cached input and output tokens (None: not offered)."""
    out = []
    for e in endpoints:
        p = e.get("pricing") or {}
        row = {
            "provider": e.get("provider_name") or e.get("name") or "?",
            "quantization": e.get("quantization") or "unknown",
            "input": _per_million(p, "prompt"),
            "cached": _per_million(p, "input_cache_read"),
            "output": _per_million(p, "completion"),
        }
        if row["input"] is None or row["output"] is None:  # free tiers and incomplete listings
            continue
        out.append(row)
    return sorted(out, key=lambda r: (r["input"], r["output"]))


def fetch_endpoints(model_id: str) -> list[dict]:
    req = urllib.request.Request(API.format(id=model_id), headers={"User-Agent": "kv-tier-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]["endpoints"]


def export(fetch: Callable[[str], list[dict]] = fetch_endpoints, now: datetime | None = None) -> dict:
    models = {}
    for sim_id, or_id in MODEL_MAP.items():
        models[sim_id] = {
            "openrouter_id": or_id, "url": PAGE.format(id=or_id), "providers": summarize(fetch(or_id)), "speed": speed(sim_id),
        }
    return {
        "fetched": (now or datetime.now(UTC)).strftime("%Y-%m-%d"),
        "source": "https://openrouter.ai/api/v1/models/<id>/endpoints",
        "models": models,
    }
