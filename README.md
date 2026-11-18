# kv-tier-bench

How long does it take vLLM to **load a cached prefix from each KV-cache tier**, and what does that cost? Each tier
is compared with simply **recomputing** the prefix. The tiers are GPU, CPU, disk, object storage, and a peer
instance's CPU memory. The benchmark runs a hybrid-attention model (Qwen3.8-27B, Gated DeltaNet 3:1) and a
standard-attention model (Qwen3-VL-32B-Instruct) side by side, on two cloud GPU VMs.

> **Status: harness built, not yet run on GPUs.** No measurements yet. The full plan, decisions and open checks are in
> [docs/plan.md](docs/plan.md).

## What it measures

- **Tier profile.** Store and load latency and bandwidth for each tier, for prefixes of 1K–128K tokens.
- **End-to-end.** Time to first token when a prefix comes from tier X, compared with recompute. One request at a
  time.
- **Cost.** $ per 1M prefix tokens served from each tier, compared with recompute. Recompute is each deployment's
  cold prefill of a fresh prompt. `kvtb cost` prints the table together with the read time, load bandwidth and
  store bandwidth it is built from.

Each prefix is prefilled once. vLLM writes it through to every tier, and each rep then resets the tiers above the
one under test and reads it back, so a stage needs one prefill per length rather than one per data point. A data
point counts only if vLLM's own metrics show that the tokens came from the tier under test. A few
natural-text prompts per tier are also generated greedily from the tier and from recompute, and the outputs
compared.

## Stack

- vLLM **v0.34.0**: `OffloadingConnector` with `TieringOffloadingSpec`. The secondary tiers are `fs`, `obj` and
  `p2p`.
- Two Nebius VMs with one RTX PRO 6000 each (TP=1), connected over TCP. vLLM runs in Docker; no Kubernetes.
- Nebius Object Storage, Enhanced Throughput class, for the object-storage tier.

## Layout

```
src/kv_tier_bench/   sizes, sweep grid, result store, tier configs, vLLM launch command, driver (`kvtb run`),
                     validity, cost, charts
configs/             experiment matrix and prices (with sources)
scripts/             sweep.sh (a whole stage, run on VM A), tier cleanup, weight staging, day-1 checks, fio
results/             raw results (JSONL), one file per stage
docs/plan.md         decisions, protocol, budget, day-1 checks, cut order
```

## Quick start (no GPU needed)

```bash
uv sync
uv run kvtb sizes                      # KV bytes/token and state bytes per prefix, from config.json
uv run kvtb serve-cmd --stage s1 --model qwen3.8-27b --deploy fs --host-ip 10.0.0.5   # the `docker run` command
uv run kvtb run --stage s1 --model qwen3.8-27b --deploy fs --dry-run   # points and jobs left to run
uv run pytest
```

On the VMs, `scripts/sweep.sh` runs a whole stage from VM A. For each deployment it starts vLLM in Docker (on both
VMs for p2p), runs `kvtb run` next to it, and stops it. It copies the results to the bucket after every
deployment and resumes after preemption. The steps are in the runbook in [docs/plan.md](docs/plan.md#runbook).

## Limitations (known up front)

- Nebius has no local NVMe on these GPUs. The disk tier is a network SSD.
- Text-only prompts. The linear-attention state is stored in bf16, which is **not** the HF default of fp32.
- GPUDirect Storage runs in compatibility mode on rented VMs, so no GDS numbers are claimed.
- Each length is stored once per job, and its 5 reps re-read the same stored prefix. The reps show how repeatable
  a read is, not how it varies between prefixes.
- Disk numbers are only valid if the disk tier uses O_DIRECT. With buffered I/O, reads can come from the OS page
  cache, so disk numbers are published only once day 1 shows direct I/O (or the page cache is dropped before
  each disk read).
- Every point is a single request: there are no runs under concurrent load.
- One hardware point only: RTX PRO 6000 at TP=1. No H100 or tensor-parallel numbers.
- The p2p tier runs over TCP between two VMs. It is not RDMA, and there is no InfiniBand anywhere in
  this benchmark.
