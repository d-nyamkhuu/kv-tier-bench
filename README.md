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

The [KV cache simulator](web/index.html) replays synthetic chat traffic and measures one hour, by default after a
20-minute warm-up (a cold start from empty caches is one setting away). Adjust request rate, average turns per chat,
think time, steady or bursty arrivals, CPU cache capacity, and independent local SSD and S3 tiers. More settings vary
message and reply lengths (fixed, moderate or long-tailed lognormal around the set means), split traffic across distinct
system prompts with Zipf popularity, and set the GPU KV budget, shown in GiB per instance. Four scenarios
set every control at once, all on two nodes with eight B200 GPUs each: Every tier in use (the default: MiniMax-M3 at
480 requests per minute with 1.5 TiB of CPU cache, a 2 TiB Gen5 SSD per node, peer reads and S3 Standard), Plenty of GPU
memory (DeepSeek V4.1 Flash, which keeps the hour in GPU memory), GPU cache fills (Kimi K3 with 1.5 TiB of CPU cache
per node) and SSD fills (Mistral Large 4 with 64 GiB of CPU cache and a 768 GiB SSD per node). Every tier is written as
prompt chunks are computed, so a CPU or SSD tier serves hits only when it is larger than the GPU cache on its node; the
page says so when CPU memory is below twice the GPU cache. Choosing a model, GPU or precision that the
cluster cannot hold raises the node and GPU counts to the smallest layout that fits (a model no layout of the GPU holds
moves to B200, then B300), and a later model that fits the
original cluster returns to it. Settings that differ from the defaults are kept in the
URL hash, so a link reopens the same setup. When a local SSD is enabled, its per-node capacity slider sets both the LRU cache limit and the billed size.
The SSD timeline links to the first eviction and the S3 timeline links to its first read.
Under Model and cluster, choose the model, serving precision (as released, FP8 or FP4), GPU type,
prefill node count, GPUs per node, and inter-node link. Each arrival goes to the next prefill node group in round-robin order, rotating through the replicas within each group. Sticky and random routing are also available. The default lookup order is GPU, local CPU, local SSD, shared storage, then peer CPU. Select peer-first to swap the last two. LMCache MP eagerly saves complete prompt chunks. CPU and bounded SSD caches poll at one-second intervals and evict 20% of resident keys at 80% usage. Prompt chunks become available after prefill, and each backing tier publishes them after the modeled write. Requests compete for each private disk and network interface. Peer reads use both endpoints. AWS backing-tier prices include estimated PUT, GET and HEAD requests. A batch of uncached HEAD probes adds one metadata round trip. Synthetic think time begins after the last reply token. Trace arrivals keep their original timestamps. The peer track shows remote reads using the same CPU cache capacity. The link also carries KV to the separate decode pool and
activations between pipeline stages. The two main results
are average time to first token (including queueing, with p50, p90 and p99) and the hourly serving cost: provisioned
prefill GPUs, the decode GPUs needed at peak and the backing tiers, also given in USD per million input, cached input and
output tokens. A progress bar follows the replay minute by minute. The breakdown under them
shows where all prompt tokens came from (computed tokens include each new message), when GPU evictions began, how busy
the prefill GPUs were, the realized request rate against the target, and, with S3 enabled, how many metadata checks requests made. The time series shows
one-minute averages and downloads as CSV. Its tier detail shows cluster-wide occupancy, request counts and the median
TTFT of requests reading each tier, for a selected minute or the whole replay. Each first-read marker jumps to that minute. Host RAM is
outside the cost. Prefill runs in continuous batches of up to 8,192 new tokens per step by default (vLLM's
max_num_batched_tokens), with cache lookups and loads overlapping other requests' compute; a request alone in its step
keeps the calibrated one-at-a-time TTFT. [Prefill batching](docs/design/prefill-batching.md) gives the step model, the
published benchmarks it was checked against, and its limits. The [LMCache compatibility notes](docs/lmcache-compatibility.md) give the versioned sources and sizing assumptions.

The tier comparison replays GPU-only, local CPU, peer CPU, NVMe and S3 alternatives on the same layout and workload settings. Its bars show mean TTFT and the storage rate per decimal GB-month. SSD rates apply to provisioned capacity and S3 rates to retained data.

The page sets this setup's price per million input, cached input and output tokens beside OpenRouter's rates for the selected model. Each OpenRouter rate is the median of endpoints that list it in the dated pricing snapshot, across provider precisions. The simulator splits its hourly cost over the tokens it served: prefill GPU time by uncached and cached tokens, cache tiers to cached input, decode GPUs to output.

Besides s1's two models, the page has ten current open-weight models (Qwen3.8-2.4T-A95B, Kimi K3, DeepSeek V4.1
Flash, GLM-5.3 and GLM-5.3-Flash, Qwen3.8-Flash-Next, MiniMax-M3, Mistral Large 4, gpt-oss-120b, Gemma 4 31B). Their KV
per token, fixed state and FLOPs are derived from each config.json by one function covering MoE, MLA, sliding-window,
sparse and linear attention. A model too big for one node runs across nodes: tensor parallel inside each node, pipeline
stages across them. Decode batches ready sequences, subject to KV capacity and a target of each model's p50 throughput
on OpenRouter (recorded by hand in `openrouter.py`, since the API's throughput fields are empty). Every input is a
published figure with its source: NVIDIA datasheets, InferenceX vLLM runs, drive and cloud-storage documents (AWS S3,
Nebius), and KV-transfer papers. The rest are marked as assumptions. vLLM's own overheads are published nowhere, so they
are fitted on s1, and `web/checks.js` compares the model with s1 and with published vLLM benchmarks.

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
