from datetime import UTC, datetime

from kv_tier_bench.openrouter import MODEL_MAP, export, summarize


def ep(provider, prompt, completion, cache=None, quant="fp8"):
    pricing = {"prompt": prompt, "completion": completion}
    if cache is not None:
        pricing["input_cache_read"] = cache
    return {"provider_name": provider, "quantization": quant, "pricing": pricing}


def test_summarize_converts_to_per_million_and_drops_free_endpoints():
    rows = summarize([ep("B", "0.0000003", "0.0000012", "0.000000006"), ep("Free", "0", "0"), ep("A", "0.0000001", "0.0000004")])
    assert [r["provider"] for r in rows] == ["A", "B"]  # sorted by input price
    assert rows[1] == {"provider": "B", "quantization": "fp8", "input": 0.3, "cached": 0.006, "output": 1.2}
    assert rows[0]["cached"] is None  # no cache price listed


def test_export_covers_every_mapped_model_with_a_date():
    data = export(lambda _id: [ep("X", "0.000001", "0.000002")], now=datetime(2026, 11, 18, tzinfo=UTC))
    assert data["fetched"] == "2026-11-18"
    assert set(data["models"]) == set(MODEL_MAP)
    m = data["models"]["glm-5.3"]
    assert m["openrouter_id"] == "z-ai/glm-5.3" and m["providers"][0]["input"] == 1.0
