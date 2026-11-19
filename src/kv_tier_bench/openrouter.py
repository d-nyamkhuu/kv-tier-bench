"""Provider prices from OpenRouter for the models the web simulator has, as a dated snapshot.

The page compares its modeled $ per 1M input, cached input and output tokens with what providers charge.
`kvtb openrouter-prices` fetches every provider endpoint of each model from OpenRouter's public API and writes
web/data/openrouter.json, which is committed. Prices are prices, not costs: they carry margins and promotions.
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
        models[sim_id] = {"openrouter_id": or_id, "url": PAGE.format(id=or_id), "providers": summarize(fetch(or_id))}
    return {
        "fetched": (now or datetime.now(UTC)).strftime("%Y-%m-%d"),
        "source": "https://openrouter.ai/api/v1/models/<id>/endpoints",
        "models": models,
    }
