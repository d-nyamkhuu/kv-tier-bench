"""`kvtb`: sizes, deployments, serve-cmd, run, cost and charts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .grid import load_experiment
from .models import load_shape
from .store import ResultStore

ROOT = Path(__file__).resolve().parents[2]


def _exp(args) -> dict:
    return load_experiment(args.experiment)


def cmd_sizes(args) -> int:
    exp = _exp(args)
    print(f"{'model':<14} {'TP':>2} {'KV/token':>10} {'state/prefix':>13} {'KV copies':>9} {'32K prefix':>11}")
    for name, m in exp["models"].items():
        shape = load_shape(name, ROOT / m["config"])
        for tp in sorted({1, *(s["tp"] for s in exp["stages"].values())}):
            kv = shape.kv_bytes_per_token(tp=tp)
            state = shape.state_bytes_per_prefix()
            total = shape.prefix_bytes(32768, tp=tp)
            print(
                f"{name:<14} {tp:>2} {kv / 1024:>7.0f} KiB {state / 1e6:>10.1f} MB "
                f"{shape.kv_replication(tp):>8}x {total / 1e9:>8.2f} GB"
            )
    return 0


def cmd_deployments(args) -> int:
    from .tierconfig import DEPLOYMENT_FOR_TIER

    print(" ".join(dict.fromkeys(DEPLOYMENT_FOR_TIER[t] for t in _exp(args)["stages"][args.stage]["tiers"])))
    return 0


def cmd_serve_cmd(args) -> int:
    import shlex

    from .launch import docker_run

    argv = docker_run(_exp(args), args.stage, args.model, args.deploy, args.host_ip, args.s3_env, args.models_dir)
    print(shlex.join(argv))
    return 0


def check_endpoints(deploy: str, urls: list[str]) -> str | None:
    """Why these endpoints cannot serve `deploy`, or None. The p2p reader pulls from the writer's host, so both
    must be distinct, routable addresses (not localhost)."""
    from urllib.parse import urlparse

    want = 2 if deploy == "p2p" else 1
    if len(urls) != want:
        return f"{deploy} needs {want} endpoint(s) (writer first), got {len(urls)}"
    hosts = [urlparse(u).hostname or "" for u in urls]
    if deploy == "p2p" and (len(set(hosts)) != 2 or any(h in ("", "localhost") or h.startswith("127.") for h in hosts)):
        return f"p2p needs two distinct, non-loopback hosts, got {hosts}"
    return None


def cmd_run(args) -> int:
    import asyncio
    from urllib.parse import urlparse

    from .client import Engine, block_size
    from .grid import group_jobs
    from .run import Abort, Peers, points_for, run_deployment

    exp = _exp(args)
    hf_id = exp["models"][args.model]["hf_id"]
    points = points_for(exp, args.stage, args.model, args.deploy, args.tiers)
    if not points:
        print(f"no points for {args.stage}/{args.model}/{args.deploy}", file=sys.stderr)
        return 2
    store = ResultStore(args.results or ROOT / "results" / f"{args.stage}.jsonl")
    max_attempts = args.max_attempts or exp.get("max_attempts", 1)
    if args.dry_run:
        todo = [p for p in points if not store.is_done(p.id, max_attempts)]
        jobs = group_jobs(todo)
        print(f"{len(todo)} of {len(points)} points to run, in {len(jobs)} jobs (one prefill each)", file=sys.stderr)
        return 0

    urls = [u for u in (args.endpoints or "").split(",") if u]
    if problem := check_endpoints(args.deploy, urls):
        print(f"abort: {problem}", file=sys.stderr)
        return 2

    async def go() -> dict:
        engines = [Engine(u, hf_id) for u in urls]
        try:
            bs = args.block_size or block_size(await engines[0].metrics())
            writer_host = urlparse(urls[0]).hostname
            peers = Peers(engines[0], engines[1], writer_host) if args.deploy == "p2p" else Peers(engines[0])
            shape = load_shape(args.model, ROOT / exp["models"][args.model]["config"])
            kvb = shape.kv_bytes_per_token(tp=exp["stages"][args.stage]["tp"])
            return await run_deployment(points, peers, bs, store, max_attempts, kv_bytes_per_token=kvb)
        finally:
            for e in engines:
                await e.aclose()

    import httpx

    try:
        counts = asyncio.run(go())
    except (Abort, LookupError, httpx.TransportError) as e:
        print(f"abort: {e} (rerun the same command to resume)", file=sys.stderr)
        return 3
    print(json.dumps(counts))
    return 0


def cmd_cost(args) -> int:
    """Static cost table from valid results, per (model, tier, prefix length), with the measured times and
    bandwidths it is built from. Each tier is compared with the cold writes of its own deployment."""
    import yaml

    from .cost import CostInputs, summarize, tier_row
    from .tierconfig import DEPLOYMENT_FOR_TIER

    exp = _exp(args)
    st = exp["stages"][args.stage]
    prices = yaml.safe_load(Path(args.prices).read_text())
    storage_key = {
        "gpu": "gpu",
        "cpu": "cpu",
        "fs": st["disk_price"],
        "obj": "obj_enhanced",
        "p2p": "p2p",
    }
    s = summarize(ResultStore(args.results).rows(), args.stage)
    rows = []
    for (model, tier, n), load_s in sorted(s.read_s.items()):
        dep = (model, DEPLOYMENT_FOR_TIER[tier], n)
        recompute = s.recompute_s.get(dep)
        if tier == "recompute" or not recompute:
            continue
        shape = load_shape(model, ROOT / exp["models"][model]["config"])
        c = CostInputs(
            gpu_hour_usd=prices["gpu_hour_usd"][st["gpu"]],
            gpus=st["gpus_per_instance"],
            storage_gib_month_usd=prices["storage_gib_month_usd"].get(storage_key[tier]),
        )
        rows.append(
            {"model": model}
            | tier_row(tier, n, shape.prefix_bytes(n, tp=st["tp"]), load_s, recompute, c, args.residency_h)
            | {
                "read_s": load_s,
                "recompute_s": recompute,  # cold writes of this tier's deployment
                "recompute_read_s": s.read_s.get((model, "recompute", n)),  # cross-check, fs deployment only
                "load_gib_s": s.load_gib_s.get((model, tier, n)),
                "store_gib_s": s.store_gib_s.get(dep) if tier in ("fs", "obj") else None,
            }
        )
    for row in rows:
        print(json.dumps(row))
    if not rows:
        print("# no valid tier/recompute pairs yet", file=sys.stderr)
    return 0


def cmd_charts(args) -> int:
    from .charts import plot_ttft

    rows = ResultStore(args.results).rows()
    exp = _exp(args)
    for model in exp["models"]:
        print(plot_ttft(rows, args.stage, model, Path(args.out) / f"ttft-{args.stage}-{model}.png"))
    return 0


def cmd_export_web(args) -> int:
    """Measured medians, sizes and prices for the web simulator, as one JSON file."""
    import yaml

    from .webdata import export

    rows = ResultStore(args.results).rows()
    prices = yaml.safe_load(Path(args.prices).read_text())
    try:
        source = str(Path(args.results).resolve().relative_to(ROOT))
    except ValueError:
        source = Path(args.results).name
    data = export(rows, _exp(args), prices, args.stage, ROOT, source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    n = sum(len(v) for m in data["models"].values() for v in m["tiers"].values())
    print(f"{out}: {len(data['models'])} models, {n} (model, tier, length) medians", file=sys.stderr)
    return 0


def cmd_openrouter_prices(args) -> int:
    """Every provider's price for the simulator's models, from OpenRouter, as a dated JSON snapshot."""
    from .openrouter import export

    data = export()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    n = sum(len(m["providers"]) for m in data["models"].values())
    print(f"{out}: {len(data['models'])} models, {n} provider endpoints ({data['fetched']})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kvtb", description=__doc__)
    ap.add_argument("--experiment", default=str(ROOT / "configs" / "experiment.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sizes", help="KV bytes/token and state bytes/prefix from config.json").set_defaults(fn=cmd_sizes)

    d = sub.add_parser("deployments", help="the vLLM deployments a stage needs, in sweep order")
    d.add_argument("--stage", required=True)
    d.set_defaults(fn=cmd_deployments)

    r = sub.add_parser("serve-cmd", help="print the `docker run` command for one deployment on one VM")
    r.add_argument("--stage", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--deploy", required=True, choices=["fs", "obj", "p2p"])
    r.add_argument("--host-ip", required=True, help="the VM's private IP (vLLM binds to it)")
    r.add_argument("--s3-env", help="env file with the object-store keys (obj only)")
    r.add_argument("--models-dir", default="/models", help="host directory with the pre-staged weights")
    r.set_defaults(fn=cmd_serve_cmd)

    k = sub.add_parser("cost", help="static $ per 1M prefix tokens per tier vs recompute, from results JSONL")
    k.add_argument("--stage", required=True)
    k.add_argument("--results", required=True)
    k.add_argument("--prices", default=str(ROOT / "configs" / "prices.yaml"))
    k.add_argument("--residency-h", type=float, default=1.0, help="hours a prefix is kept in the tier")
    k.set_defaults(fn=cmd_cost)

    u = sub.add_parser("run", help="measure every not-yet-done point of one deployment (resumable)")
    u.add_argument("--stage", required=True)
    u.add_argument("--model", required=True)
    u.add_argument("--deploy", required=True, choices=["fs", "obj", "p2p"])
    u.add_argument("--tiers", nargs="+", help="only these tiers of the deployment")
    u.add_argument("--results", help="JSONL path (default results/<stage>.jsonl)")
    u.add_argument("--endpoints", help="comma-separated vLLM URLs by private IP, writer first (2 for p2p)")
    u.add_argument("--block-size", type=int, help="override the block size read from /metrics")
    u.add_argument("--max-attempts", type=int)
    u.add_argument("--dry-run", action="store_true", help="only count the points left to run")
    u.set_defaults(fn=cmd_run)

    c = sub.add_parser("charts", help="TTFT vs prefix length per tier, from results JSONL")
    c.add_argument("--stage", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--out", default=str(ROOT / "results" / "charts"))
    c.set_defaults(fn=cmd_charts)

    w = sub.add_parser("export-web", help="medians, sizes and prices for the web simulator, as JSON")
    w.add_argument("--stage", required=True)
    w.add_argument("--results", required=True)
    w.add_argument("--prices", default=str(ROOT / "configs" / "prices.yaml"))
    w.add_argument("--out", default=str(ROOT / "web" / "data" / "s1.json"))
    w.set_defaults(fn=cmd_export_web)

    o = sub.add_parser("openrouter-prices", help="provider prices for the web simulator's models, from OpenRouter")
    o.add_argument("--out", default=str(ROOT / "web" / "data" / "openrouter.json"))
    o.set_defaults(fn=cmd_openrouter_prices)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
