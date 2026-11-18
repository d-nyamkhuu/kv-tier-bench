"""The exact HTTP requests the driver sends to vLLM, checked against a mock transport."""

import asyncio
import json

import httpx

from kv_tier_bench.client import Engine, prompt_tokens

METRICS = 'vllm:cache_config_info{block_size="528",engine="0"} 1.0\nvllm:prompt_tokens_cached_by_source_total{source="host"} 5.0\n'


def test_requests_match_the_vllm_api():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/metrics":
            return httpx.Response(200, text=METRICS)
        if req.url.path == "/reset_prefix_cache":
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json={"choices": [{"text": "x"}]})

    async def go():
        e = Engine("http://10.0.0.1:8000/", "Qwen/Qwen3.8-27B", transport=httpx.MockTransport(handler))
        samples = await e.metrics()
        ok = await e.reset_prefix_cache(reset_external=True)
        dt, body = await e.complete([1, 2, 3], kv_transfer_params={"kv_load_tiers": []})
        await e.aclose()
        return samples, ok, dt, body

    samples, ok, dt, body = asyncio.run(go())
    assert ok and dt >= 0 and body["choices"][0]["text"] == "x"
    assert any(s.name == "vllm:cache_config_info" and s.label("block_size") == "528" for s in samples)
    reset, comp = seen[1], seen[2]
    assert reset.method == "POST" and reset.url.params["reset_external"] == "true"
    sent = json.loads(comp.content)
    assert comp.url.path == "/v1/completions"
    assert sent == {
        "model": "Qwen/Qwen3.8-27B",
        "prompt": [1, 2, 3],  # token IDs, so prefix lengths are exact
        "max_tokens": 1,
        "temperature": 0.0,
        "kv_transfer_params": {"kv_load_tiers": []},  # an empty list is sent, not dropped
    }


def test_prompt_tokens_are_deterministic_and_in_range():
    a, b = prompt_tokens(7, 1000), prompt_tokens(7, 1000)
    assert a == b and len(a) == 1000 and min(a) >= 1000 and max(a) < 150000
    assert prompt_tokens(8, 1000) != a
