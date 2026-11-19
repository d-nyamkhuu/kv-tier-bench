# kv-tier-bench

How long does it take vLLM to **load a cached prefix from each KV-cache tier**, and what does that cost? Each tier
is compared with simply **recomputing** the prefix. The tiers are GPU, CPU, disk, object storage, and a peer
instance's CPU memory. The benchmark runs a hybrid-attention model (Qwen3.8-27B, Gated DeltaNet 3:1) and a
standard-attention model (Qwen3-VL-32B-Instruct) side by side, on two cloud GPU VMs.

> **Status: stage s1 measured (2026-11-18), 390 of 390 points valid; p2p re-measured 2026-11-19 with a NIXL
> tuning hook (75 of 75 valid), which made long-prefix p2p loads 5-6x faster.** Raw results are in
> [results/s1.jsonl](results/s1.jsonl); `kvtb cost` and `kvtb charts` summarize them. The plan, decisions, day-1
> findings and the NIXL workarounds behind the p2p and obj numbers are in [docs/plan.md](docs/plan.md).
>
> **Stage s2 measured (2026-11-19), 1032 of 1032 points valid:** concurrent load, batches of 1-16 requests with
> 4K-128K prefixes each, on Qwen3.5-9B (hybrid) and Qwen3-8B (standard) in Nebius us-central1. It adds the
> Standard Object Storage class (`obj-std`) and a Nebius shared filesystem (`fs-shared`) next to s1's tiers. Raw
> results are in [results/s2.jsonl](results/s2.jsonl) (`kvtb cost --stage s2`, `kvtb charts --stage s2`); the
> findings are in [docs/plan.md](docs/plan.md#stage-2-concurrent-load-s2).
>
> **Stage s3 measured (2026-11-19), 315 of 315 points valid:** a CPU-only VM as the KV store (Mooncake Store over
> TCP, `MooncakeStoreConnector`) against p2p, on s2's models and batches in Nebius eu-south1, with s2's recompute,
> GPU and CPU rows as baselines. Raw results are in [results/s3.jsonl](results/s3.jsonl) (`kvtb cost --stage s3`);
> the findings are in [docs/plan.md](docs/plan.md#stage-3-a-cpu-only-kv-store-s3-measured).

## Results

The full write-up, with figures, method and limits, is **[docs/report.md](docs/report.md)**.

**One request at a time (s1, eu-south1).** The longest prefix each model ran. Time to first token, and the modelled
saving against recomputing the prefix:

| Tier | Qwen3.8-27B, 128K tokens | Qwen3-VL-32B, 64K tokens |
|---|---|---|
| GPU memory | 0.15 s (99%) | 0.085 s (100%) |
| CPU memory | 0.31 s (99%) | 0.40 s (98%) |
| Peer VM over TCP (p2p) | 0.69 s (98%), 23 GiB/s | 1.00 s (96%), 28 GiB/s |
| Object Storage, Enhanced | 1.3 s (87%), 10 GiB/s | 2.3 s (72%), 9 GiB/s |
| Network disk | 6.5 s (74%), 1.3 GiB/s | 16 s (28%), 1.0 GiB/s |
| Recompute | 30 s | 26 s |

**Batches of 1–16 requests (s2, us-central1).** Recomputing c prefixes at once takes c times one (median 0.99×), so
recompute gains nothing from batching while every tier overlaps its loads. One 128K-token prefix:

| Tier | Qwen3-8B (standard, 144 KiB/token) | Qwen3.5-9B (hybrid, 32 KiB/token) |
|---|---|---|
| GPU memory | 0.14 s (99%) | 0.12 s (99%) |
| CPU memory | 0.50 s (98%) | 0.21 s (98%) |
| Peer VM over TCP (p2p) | 2.1 s (90%) | 0.66 s (93%) |
| Object Storage, Enhanced | 3.5 s (58%) | 1.2 s (73%) |
| Object Storage, Standard | 11 s (47%) | 2.0 s (76%) |
| Network disk | 13 s (28%) | 2.5 s (66%) |
| Shared filesystem | 37 s (−91%) | 8.1 s (1%) |
| Recompute | 22 s | 9.1 s |

- **Bytes per token decide the storage tiers.** On Qwen3-8B, Object Storage and the disk only pay off from about 32K
  tokens (64K for the disk); below that the storage bill outweighs the faster load.
- **The shared filesystem lost to the plain disk** at 20 of Qwen3.5-9B's 24 points and 14 of Qwen3-8B's 19, though
  fio rated it 4× faster (8.2 against 1.9 GiB/s of 16-way reads). vLLM's fs tier reads a prefix as 10 MiB files with
  16 threads, and one long prefix arrived at about 0.5 GiB/s.
- **p2p in us-central1 is a floor.** It moved 10–12 GiB/s per pull against about 23 GiB/s in eu-south1 with the same
  settings, for a reason that was not diagnosed. s3 re-ran it in eu-south1: a median 1.2× (Qwen3-8B) and 1.6×
  (Qwen3.5-9B) faster.

**A CPU-only KV store (s3, eu-south1).** The same batches, from Mooncake Store on a CPU-only VM against p2p from a
second GPU VM. Batch time and end-to-end rate:

| Prefix × requests | Qwen3-8B p2p | Qwen3-8B store | Qwen3.5-9B p2p | Qwen3.5-9B store |
|---|---|---|---|---|
| 4K × 1 | 0.093 s, 6.1 GiB/s | 0.158 s, 3.6 GiB/s | 0.045 s, 3.3 GiB/s | 0.062 s, 2.3 GiB/s |
| 16K × 16 | 2.4 s, 14.9 GiB/s | 7.3 s, 4.9 GiB/s | 0.50 s, 17.0 GiB/s | 1.6 s, 5.2 GiB/s |
| 128K × 1 | 1.7 s, 10.5 GiB/s | 4.0 s, 4.5 GiB/s | 0.31 s, 12.9 GiB/s | 0.78 s, 5.2 GiB/s |

- **p2p is a median 2.5× faster than the store**, and the gap grows with the batch: Mooncake levels off at
  5–6 GiB/s. With Mooncake's pool on the peer GPU VM instead (store-p2p, Qwen3-8B), times matched the CPU-only VM's.
- **Mooncake's TCP defaults** (1 I/O thread, 4 lanes, 64 KiB slices) were 3.2–4.8× slower than the tuned settings
  in `configs/experiment.yaml`. A bigger CPU VM added pool room, not speed.
- **At production reuse a remote memory tier pays, but not as a CPU-only VM over TCP.** From about 100 reads of a
  prefix an hour, every remote tier saves 68–91% at 128K tokens. The GPU VM's own CPU memory saves more, and on
  Qwen3-8B Object Storage saves at least as much as the store at 1/55 of its price per GiB. A dedicated store VM
  needs RDMA, a hot set bigger than the GPU VMs' spare RAM, or a cache that outlives them
  ([report](docs/report.md#cost-at-production-reuse)).

Savings are modelled, not measured: the GPU at $1.80/h on demand (the runs paid spot), each prefix stored for one
hour per read, Standard-class request fees left out. The aggregated tables behind these numbers are in
[results/report/](results/report/) (`uv run python scripts/export-report-data.py` rebuilds them), the charts in
[results/charts/](results/charts/).

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
scripts/             vms.sh (VMs and the weights filesystem, from the laptop), sweep.sh (a whole stage, run on VM A),
                     store-up.sh (s3's Mooncake store on a CPU VM), tier cleanup, weight staging, day-1 checks, fio
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

## Simulator (web/)

s1 measures the cost of one hit from each tier, not how often a tier hits. [web/](web/) is a static page, the
"KV cache calculator". You pick a GPU, nodes × GPUs per node × TP, cache tiers (CPU memory, local disk, peer nodes,
shared storage) and a load (requests per minute, follow-up share, tokens in and out). It replays the workload
through simulated LRU tiers on that cluster, with FIFO prefill queues and disaggregated batched decode. The result
is the mean TTFT, the $ per 1,000 requests from provisioned GPUs plus storage, and the $ per 1M input, cached input
and output tokens. Each tier row shows how many tokens (and, for models with a fixed state, how many prompts) it holds.
Besides s1's two models, the page has ten current open-weight models (Qwen3.8-2.4T-A95B, Kimi K3, DeepSeek V4.1
Flash, GLM-5.3 and GLM-5.3-Flash, Qwen3.8-Flash-Next, MiniMax-M3, Mistral Large 4, gpt-oss-120b, Gemma 4 31B). Their KV per token, fixed
state and FLOPs are derived from each config.json by one function covering MoE, MLA, sliding-window, sparse and linear
attention. A model too big for one node (Qwen3.8-2.4T, or Kimi K3 on anything smaller than 8 B300s) runs across nodes:
tensor parallel inside each node, pipeline stages across them. The page also sets each model's modeled $ per 1M tokens beside every provider's price on
[OpenRouter](https://openrouter.ai) (a dated snapshot in `web/data/openrouter.json`), from a reference deployment of 8
instances loaded until the prefill GPUs are busy 60% of the time. Decode there holds each model's p50 throughput on
OpenRouter (recorded by hand in `openrouter.py`, since the API's throughput fields are empty) by running the largest
batch that keeps it; an MoE step reads only the experts its batch picked. Decode can add speculative decoding (MTP, EAGLE-3, DFlash, DSpark or
custom numbers) and expert parallelism over more GPUs than one node, with all-to-all traffic at DeepEP's measured rates;
both are checked against published runs (Kimi K2.5 on B200 and GB200, DeepSeek-V3 in production, EAGLE-3 and DFlash). A trade-off map plots every tier
combination with its Pareto frontier, and a diagram shows the cluster. The workload can be synthetic conversations,
a [Mooncake](https://github.com/kvcache-ai/Mooncake) trace, or an uploaded trace. Every input is a published figure
with its source: NVIDIA datasheets, InferenceX vLLM runs, drive and cloud-storage documents (AWS S3, Nebius), and
KV-transfer papers. The rest are marked as assumptions. vLLM's own overheads are published nowhere, so they are fitted
on s1. s1 is otherwise the reference: the page compares the model with s1 and with published vLLM benchmarks. Every
number is editable in the page, and the config imports and exports as JSON.

```bash
uv run kvtb export-web --stage s1 --results results/s1.jsonl --out web/data/s1.json   # after new results
uv run kvtb openrouter-prices                                                        # refresh provider prices
python -m http.server -d web                                                         # then open localhost:8000
node --test "web/test/*.test.js"                                                     # simulator tests
```

[.github/workflows/pages.yml](.github/workflows/pages.yml) publishes `web/` to GitHub Pages on every push to
`main`. The workflow fails until Pages is enabled with "GitHub Actions" as the source.

On the VMs, `scripts/sweep.sh` runs a whole stage from VM A. For each deployment it starts vLLM in Docker (on both
VMs for p2p), runs `kvtb run` next to it, and stops it. It copies the results to the bucket after every
deployment and resumes after preemption. The steps are in the runbook in [docs/plan.md](docs/plan.md#runbook).

## Limitations (known up front)

- Nebius has no local NVMe on these GPUs. The disk tier is a network SSD, and its read rate (~1.0–1.3 GiB/s in s1,
  flat at every prefix length) looks like the disk's provisioned throughput cap. That cap is inferred from the
  results; no standalone `fio` run measured it.
- Object-store reads are of freshly written objects: each prefix is read back seconds to minutes after it is
  stored, possibly while the storage service still has it cached. A prefix stored hours earlier may load slower.
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
  this benchmark. The p2p numbers are "vLLM v0.34.0 + a NIXL tuning hook" (`scripts/nixl_patch`: 32 NIXL agents per
  process, progress threads that sleep, posts off the scheduler thread). The earlier config (16 UCX streams, no hook) loaded
  long prefixes 5-6x slower; its rows are kept in [results/s1-p2p-16stream.jsonl](results/s1-p2p-16stream.jsonl).
