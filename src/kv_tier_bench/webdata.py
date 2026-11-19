"""The measured inputs of the web simulator (web/): medians per model x tier x prefix length, sizes and prices.

The page has no Python step, so `kvtb export-web` writes them once to web/data/<stage>.json, which is committed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .cost import GIB, summarize
from .grid import gpu_kv_budget, stage_models
from .models import load_shape
from .tierconfig import deployment_for

ACTIVATION_GIB_PER_GPU = 3  # what grid.gpu_kv_budget leaves for activations; the page redoes that sum for other GPUs


def _r(x: float | None, digits: int = 6) -> float | None:
    return None if x is None else float(f"{x:.{digits}g}")


def export(rows: list[dict], exp: dict, prices: dict, stage: str, root: Path, source: str = "") -> dict:
    st = exp["stages"][stage]
    s = summarize(rows, stage)
    block = defaultdict(set)
    for r in rows:
        if r.get("stage") == stage and r.get("valid"):
            block[r["model"]].add(r["block_size"])

    models = {}
    for name in stage_models(exp, stage):
        m = exp["models"][name]
        shape = load_shape(name, root / m["config"])
        text = json.loads((root / m["config"]).read_text())
        text = text.get("text_config", text)
        kv = shape.kv_bytes_per_token(tp=st["tp"])
        tiers: dict[str, list[dict]] = defaultdict(list)
        for (model, tier, n), read_s in sorted(s.read_s.items()):
            if model != name:
                continue
            key = (model, tier, n)
            tiers[tier].append(
                {
                    "tokens": n,
                    "ttft_s": _r(read_s),
                    "load_gib_s": _r(s.load_gib_s.get(key), 4),
                    "queue_s": _r(s.queue_s.get(key)),
                    "prefill_s": _r(s.prefill_s.get(key)),
                    "read_bytes": int(s.read_bytes[key]) if s.read_bytes.get(key) else None,
                }
            )
        recompute = [
            {"tokens": n, "ttft_s": _r(t)}
            for (model, dep, n), t in sorted(s.recompute_s.items())
            if model == name and dep == deployment_for(stage, "recompute")
        ]
        # bytes read per prefix = fixed + kv * tokens; the fixed part is the hybrid model's state snapshot
        fixed = [b - kv * n for (model, _, n), b in s.read_bytes.items() if model == name and b]
        if len(block[name]) > 1:
            raise ValueError(f"{name}: more than one block size in {stage}: {sorted(block[name])}")
        models[name] = {
            "hf_id": m["hf_id"],
            "hybrid": shape.is_hybrid,
            "params": m["params"],
            "weights_bytes": m["params"] * 2,  # bf16
            # prefill FLOPs per token = 2 x params + 2 x full-attention layers x width x context (causal)
            "full_attn_layers": shape.full_attn_layers,
            "attn_width": text["num_attention_heads"] * shape.head_dim,
            "kv_bytes_per_token": kv,
            # with tensor parallelism wider than the KV heads, vLLM copies heads: more bytes per token
            "kv_bytes_per_token_by_tp": {str(t): shape.kv_bytes_per_token(tp=t) for t in (1, 2, 4, 8)},
            "state_bytes_per_prefix": shape.state_bytes_per_prefix(),
            "read_fixed_bytes": int(median(fixed)) if fixed else 0,
            "block_size": next(iter(block[name]), None),
            "gpu_kv_gib": round(gpu_kv_budget(exp, stage, name) / GIB, 2),
            "max_prefix_tokens": st.get("max_prefix_tokens", {}).get(name, max(exp["prefix_tokens"])),
            "recompute": recompute,
            "tiers": dict(tiers),
        }

    return {
        "stage": stage,
        "description": st["description"],
        "source": source,
        "generated_by": f"kvtb export-web --stage {stage}",
        "recompute_from": f"cold writes (write_s) of the {deployment_for(stage, 'recompute')} deployment",
        "deployment_for_tier": {t: deployment_for(stage, t) for t in st["tiers"]},
        "gpu": st["gpu"],
        "gpus_per_instance": st["gpus_per_instance"],
        "cpu_tier_gib": st["cpu_tier_gib"],
        "gpu_mem_gib": st["gpu_mem_gib"],
        "gpu_memory_utilization": st["gpu_memory_utilization"],
        "activation_gib_per_gpu": ACTIVATION_GIB_PER_GPU,
        "prices": prices,
        "models": models,
    }
