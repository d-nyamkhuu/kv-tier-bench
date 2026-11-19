# kv-tier-bench

How long does it take vLLM to **load a cached prefix from each KV-cache tier**, and what does that cost? Each tier
is compared with simply **recomputing** the prefix. The tiers are GPU, CPU, disk, object storage, and a peer
instance's CPU memory. The benchmark runs a hybrid-attention model (Qwen3.8-27B, Gated DeltaNet 3:1) and a
standard-attention model (Qwen3-VL-32B-Instruct) side by side, on two cloud GPU VMs.

> **Status: stage s1 measured (2026-11-18), 390 of 390 points valid.** Raw results are in
> [results/s1.jsonl](results/s1.jsonl); `kvtb cost` and `kvtb charts` summarize them. The plan, decisions, day-1
> findings and the NIXL workarounds behind the p2p and obj numbers are in [docs/plan.md](docs/plan.md).

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
  this benchmark.
