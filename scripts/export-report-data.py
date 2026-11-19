#!/usr/bin/env python3
"""Writes the aggregated tables behind the results report to results/report/: one JSON array of rows per file.

    uv run python scripts/export-report-data.py

s1.json and s2.json are `kvtb cost` for each stage (medians per point, with the modelled $ per 1M tokens), trimmed
to the columns the report uses and rounded. models.json and setup.json describe what ran. Only aggregates go here:
the raw results stay in results/s1.jsonl and results/s2.jsonl.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from kv_tier_bench.grid import load_experiment, stage_models
from kv_tier_bench.models import load_shape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report"
GIB = 2**30

S1_COLS = [
    "model",
    "tier",
    "prefix_tokens",
    "read_s",
    "recompute_s",
    "load_gib_s",
    "saving_vs_recompute",
    "total_usd_per_1m",
    "recompute_usd_per_1m",
]
S3_EXTRA = ["storage_usd_per_1m", "storage_usd_per_1m_included", "saving_vs_recompute_included", "total_usd_per_1m_included"]
S2_COLS = [
    "model",
    "tier",
    "prefix_tokens",
    "concurrency",
    "batch_s",
    "recompute_batch_s",
    "tokens_per_s",
    "recompute_tokens_per_s",
    "ttft_p50_s",
    "ttft_max_s",
    "batch_gib_s",
    "saving_vs_recompute",
    "total_usd_per_1m",
    "recompute_usd_per_1m",
    "reps",
]


def _round(v):
    if isinstance(v, float):
        return float(f"{v:.4g}")
    return v


def cost_rows(stage: str, cols: list[str]) -> list[dict]:
    out = subprocess.run(
        ["uv", "run", "-q", "kvtb", "cost", "--stage", stage, "--results", f"results/{stage}.jsonl"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    return [{c: _round(r.get(c)) for c in cols} for r in rows]


def measured_rates(stage: str = "s2") -> dict:
    """Delivered GiB/s per point from the rows' own bytes: median of read_bytes / read_s over valid reps."""
    from statistics import median

    acc: dict = {}
    for line in open(ROOT / "results" / f"{stage}.jsonl"):
        r = json.loads(line)
        if r.get("valid") and r.get("read_bytes") and r.get("read_s"):
            key = (r["model"], r["tier"], r["prefix_tokens"], r.get("concurrency", 1))
            acc.setdefault(key, []).append(r["read_bytes"] / r["read_s"] / GIB)
    return {k: median(v) for k, v in acc.items()}


def fio_rows() -> list[dict]:
    """16-way random-read and single-stream read bandwidth (8 MiB direct I/O) of each s2 fs medium, from fio's JSON."""
    rows = []
    for medium, f in (("disk", "fio-kv-disk-us-central1.json"), ("shared_fs", "fio-fs-1tib-us-central1.json")):
        jobs = {j["jobname"]: j for j in json.load(open(ROOT / "results" / f))["jobs"]}
        for job, label in (("read", "16-way random reads"), ("read-qd1", "single-stream reads")):
            rows.append(
                {
                    "item": f"fio_{medium}_{job.replace('-', '_')}_gib_s",
                    "value": _round(jobs[job]["read"]["bw_bytes"] / GIB),
                    "note": f"fio on the s2 {medium.replace('_', ' ')}, {label}, results/{f}",
                }
            )
    return rows


def valid_points(stage: str) -> int:
    return len(
        {
            r["id"]
            for r in map(json.loads, open(ROOT / "results" / f"{stage}.jsonl"))
            if r.get("kind") != "job_start" and r.get("valid")
        }
    )


def main() -> None:
    exp = load_experiment(ROOT / "configs" / "experiment.yaml")
    prices = yaml.safe_load((ROOT / "configs" / "prices.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    s1 = cost_rows("s1", S1_COLS)
    s2 = cost_rows("s2", S2_COLS)
    # s3: the remote tiers measured in eu-south1; recompute, gpu and cpu come from s2 (baseline_from)
    s3 = cost_rows("s3", S2_COLS + S3_EXTRA)
    for stage, rows in (("s2", s2), ("s3", s3)):
        rates = measured_rates(stage)
        for r in rows:
            g = rates.get((r["model"], r["tier"], r["prefix_tokens"], r["concurrency"]))
            r["batch_gib_s"] = _round(g) if g is not None else None

    models = []
    for stage in ("s1", "s2"):
        st = exp["stages"][stage]
        for name in stage_models(exp, stage):
            m = exp["models"][name]
            shape = load_shape(name, ROOT / m["config"])
            models.append(
                {
                    "stage": stage,
                    "model": name,
                    "hf_id": m["hf_id"],
                    "attention": "hybrid" if shape.is_hybrid else "standard",
                    "params_b": round(m["params"] / 1e9, 1),
                    "kv_kib_per_token": round(shape.kv_bytes_per_token(tp=st["tp"]) / 1024),
                    "state_mib_per_prefix": round(shape.state_bytes_per_prefix() / 2**20, 1),
                }
            )

    gpu = prices["gpu_hour_usd"]["rtx-pro-6000"]
    store = prices["storage_gib_month_usd"]
    setup = [
        {"item": "gpu_usd_per_hour", "value": gpu, "note": "RTX PRO 6000, on demand, Nebius; the cost model's price"},
        {"item": "spot_usd_per_hour_s1", "value": 1.38, "note": "eu-south1 spot price the s1 run paid"},
        {"item": "spot_usd_per_hour_s2", "value": 1.19, "note": "us-central1 spot price the s2 run paid"},
        {"item": "disk_usd_per_gib_month", "value": store["fs_network_ssd_nrd"], "note": "non-replicated network SSD"},
        {"item": "shared_fs_usd_per_gib_month", "value": store["fs_shared_network_ssd"], "note": "shared filesystem"},
        {"item": "obj_usd_per_gib_month", "value": store["obj_enhanced"], "note": "Object Storage, Enhanced"},
        {"item": "obj_std_usd_per_gib_month", "value": store["obj_standard"], "note": "Object Storage, Standard"},
        {"item": "residency_hours", "value": 1, "note": "hours each prefix is stored per read, in the cost model"},
        *fio_rows(),
        {"item": "s1_points_valid", "value": valid_points("s1"), "note": "s1, one request at a time"},
        {"item": "s2_points_valid", "value": valid_points("s2"), "note": "s2, batches of 1-16 requests"},
        {"item": "s3_points_valid", "value": valid_points("s3"), "note": "s3, store and p2p, batches of 1-16"},
        {"item": "spot_usd_per_hour_s3", "value": 1.38, "note": "eu-south1 spot price the s3 run paid"},
        {"item": "ram_usd_per_gib_month", "value": store["ram_cpu_vm"], "note": "CPU VM RAM, s3's price for p2p/store"},
    ]

    for name, rows in (("s1", s1), ("s2", s2), ("s3", s3), ("models", models), ("setup", setup)):
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=0) + "\n")
        print(f"results/report/{name}.json: {len(rows)} rows")


if __name__ == "__main__":
    main()
