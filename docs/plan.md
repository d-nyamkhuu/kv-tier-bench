# kv-tier-bench: plan

Agreed 2026-11-15. This file records what we decided and why, what is still unverified, and the order in which
things get cut if the two weeks run short.

## Goal

A measured profile of every KV-cache tier that vLLM can use, on two cloud GPU VMs. Two models are compared: one with
hybrid attention and one with standard attention. For each tier we measure the cost to store and load a prefix,
and compare it with recomputing the prefix. The results feed a static cost table.

This is a benchmark and an analysis. It does not change any engine, add a routing rule, or provide a
recommender tool.

Nobody has published this:

- The vLLM tiered-offloading blog (2026-09-10) covers only CPU and filesystem, and only throughput.
- llm-d-benchmark covers only the CPU tier.
- llm-d's calibration script covers only peer pull against recompute, for one model.
- Nothing covers object storage, all tiers side by side, latency against prefix size, or cost per tier.

## Decisions

| Area | Decision | Why |
|---|---|---|
| Engine | vLLM **v0.33.0**, `OffloadingConnector` + `TieringOffloadingSpec` | The only stack with CPU, fs, obj and p2p in-tree and no patching. It has per-tier metrics and per-request load control (`kv_load_tiers`, `max_load_tokens`). |
| Host | **Plain VMs**: vLLM in Docker (`vllm/vllm-openai:v0.33.0`), started by `scripts/sweep.sh` over ssh. No Kubernetes, llm-d, router or Dynamo (decided 2026-11-18). | The benchmark needs two machines that can reach each other. Kubernetes added a driver image, a Job, RBAC, volumes and helper pods without changing any number. Dynamo's KVBM is deprecated and Dynamo pins vLLM 0.31. |
| Machines | 2 × Nebius `gpu-rtx6000` VMs (1 GPU each) in us-central1, same subnet, one stage (s1) | The user has tested RTX PRO 6000 capacity on Nebius. |
| Hybrid model | **Qwen/Qwen3.8-27B**: 64 layers, 16 full attention and 48 Gated DeltaNet; 4 KV heads × 256 | Released 2026-08-05, Apache-2.0. |
| Standard model | **Qwen/Qwen3-VL-32B-Instruct**: 64 full-attention layers; 8 KV heads × 128 | Same KV width per layer as the hybrid model, so the KV difference is just 16 vs 64 layers. Context 262K. |
| Parallelism | TP=1 on RTX PRO 6000 | One card holds either model. PP would distort the recompute baseline. |
| Stage 2 (H100, TP=4) | **Dropped** (decided 2026-11-18) | It was ≤$144 of a ~$200 budget, for a second hardware point (TP=4 on H100) that the tier comparison doesn't need. |
| Precision | bf16 weights and KV. **bf16 linear-attention state on every run** (`--mamba-ssm-cache-dtype bfloat16`). | User decision. The HF default for the state is fp32, so the README must say it was overridden. |
| Prompts | Text only, token-ID prompts through `/v1/completions` | Both models are VLMs, and images would change the prefix hashing. Token IDs give exact lengths. |
| Speculative decoding / MTP | Off | Open issues for hybrid models with offload: vllm#50454, #55503, #53670. |
| Disk tier | 2000 GiB network SSD (non-replicated), ext4, attached to VM A at `/mnt/kv` | Nebius has no local NVMe outside B300; the README states this. A block device should support O_DIRECT, which the fs numbers need. Network SSD read bandwidth grows with size: ~2 GiB/s needs ~1.8 TiB. |
| Object tier | Nebius Object Storage, **Enhanced Throughput** class only | User decision. Requests and same-region traffic are free. |
| Peer tiers | p2p over TCP between the two RTX VMs | InfiniBand is dropped. Same-node p2p is dropped too: it is a host-memory copy, close to the CPU tier. |
| Cost analysis | A static table: $ per 1M prefix tokens per tier, against recompute | User decision. There is no dynamic break-even and no TTL model. |
| Validity | Each point checks the per-tier source metrics. ~5 natural-text prompts per tier and model compare greedy output (~16 tokens) against recompute. | A cache hit counter does not prove the right KV was used. See LMCache#5600, SGLang#45563, vllm PR #65239. |
| Broken tier | File an upstream issue with a minimal repro and **do not debug**. Measure that tier on the standard model only. | User decision. |
| Driver | Our own small async Python driver, run on VM A | Off-the-shelf load tools can't reset tiers between requests, read the metrics, or inject `remote_kv_source`. |

## Tiers measured

On 2 × RTX PRO 6000 in us-central1:

| # | Tier |
|---|---|
| 0 | Recompute (no cache) |
| 1 | GPU prefix cache |
| 2 | CPU pinned (`cpu_bytes_to_use`) |
| 3 | Disk (`fs`): network SSD |
| 4 | Object storage (`obj`, Enhanced Throughput) |
| 5 | p2p over TCP, between the two VMs |

Each model gets three deployments, so three vLLM restarts per model. A deployment is one vLLM container on VM A
(two containers, one per VM, for p2p):
- `fs` covers recompute, GPU, CPU and fs. Its CPU tier sits above the fs tier, so one deployment measures all four.
- `obj` and `p2p` each get their own.

Only one secondary tier runs at a time. This keeps the `fs` and `obj` tiers apart, because both report as medium
`STORAGE`.

## Measurements

1. **Tier profile.** For each tier, store and load latency and bandwidth against prefix length.
   - Lengths are 1K–128K tokens, rounded **at run time** to multiples of the hybrid block size read from the startup log.
   - On RTX, Qwen3-VL-32B stops at 64K: a 128K prefix needs about 32 GiB of KV and the card has about 25 GB free.
2. **End-to-end.** TTFT of `max_tokens=1` requests from each tier against recompute, one request at a time.
3. **Cost table.** $ per 1M prefix tokens for each tier against recompute, computed from items 1–2 and `configs/prices.yaml`.

Sizes are computed from each model's `config.json` (`kvtb sizes`). These are not measurements:

| | KV per token (bf16) | Linear-attention state per cached prefix (bf16) |
|---|---|---|
| Qwen3.8-27B | 64 KiB | ~78.4 MB (48 layers × (1.5 MiB recurrent + 60 KiB conv)) |
| Qwen3-VL-32B-Instruct | 256 KiB | none |

### Protocol: one prefill, many reads

A point is (stage, model, tier, prefix length, rep). Its id is the key for a resumable run: a point that is
already done is skipped. Points that share a deployment and length form a **job**, and a job prefills its prompt
once. Tiering is write-through, so that one write puts the prefix in GPU, CPU and the
deployment's secondary tier, and the reads then "move" it between tiers by resetting the tiers above.

0. **Warm-up**, once per deployment before the first job: one throwaway request to each engine, and for p2p a
   throwaway pull from the writer to the reader. Without it, the first measured write and reads after a restart
   (and the reader's first p2p pull) pay first-request and connection-setup costs.
1. Build a prompt of random token IDs, seeded from the job id and the job attempt, so no tier has seen it before.
2. **Write.** Send the prompt with `max_tokens=1` and `max_load_tokens: 0`, and record the cold prefill time
   (`write_s`). This is the **recompute baseline**: a fresh prompt for every sample, and every deployment has
   them, so each tier is compared with the cold writes of its own deployment.
3. **Settle.** Poll until `kv_offload_tiering_write_bytes` for the tier stops changing, no cascade or promotion
   jobs are active, and at least a tenth of the prefix's bytes have been written. Check that
   `backpressure_stores_dropped` did not grow. Record the store side once per job: `settle_s` (how long the
   cascade ran after the prefill) and the tier's own `write_bytes` and `write_time_s`. Every row of the job carries
   these copies, so count them once per job. Store bandwidth is `write_bytes / write_time_s`; `settle_s` is not a
   store-speed measure, since settling needs 4 stable polls 0.25 s apart and so never reads below about 1 s.
4. **For each rep, read from each tier in turn** (gpu, cpu, fs, obj, p2p, recompute). Before each read, isolate
   the tier:
   - GPU: nothing. The prefix is in GPU from the write or from the previous read.
   - CPU: `POST /reset_prefix_cache`, which clears the GPU only.
   - fs / obj: `POST /reset_prefix_cache?reset_external=true`, which clears GPU and CPU and keeps the secondary
     tiers.
   - p2p: reset the *reader* the same way, so it holds no copy from the previous pull, then send the read to it
     with `remote_kv_source` pointing at the writer. The writer keeps its CPU copy.
   - recompute: clear the GPU, then read with `max_load_tokens: 0`. Recompute gets 1 rep (`recompute_reps`), the
     tiers get 5. This read is a cross-check on the `write_s` baseline (`recompute_read_s` in `kvtb cost`), never
     pooled with it.

   Each reset is retried until it succeeds. The read restricts loads to the tier under test with `kv_load_tiers`
   and records the wall-clock TTFT, the engine's TTFT histogram delta, the read bytes and time for each `tier`
   label (a load from fs or obj may also count its hop through the CPU tier), `prompt_tokens_cached_by_source`,
   and its position in the job (`read_index`). Load bandwidth is the tier's own read bytes / read time.
5. **Valid** only if the cached-token delta for the expected source matches the block-aligned prefix length (for
   recompute: no cached tokens at all). Otherwise record the row as `invalid` with a reason. Never drop it
   silently.
6. **Retry.** A job with invalid points is rerun for those points only, with a fresh prompt. A point is done once
   one attempt is valid, or after `max_attempts` (3) invalid ones. Every attempt stays in the results, and each
   job attempt writes a `job_start` marker row. The marker means a job that died mid-way also gets a fresh prompt
   when it resumes: fs and obj keep the old prompt's blocks across vLLM restarts, so reusing it would store nothing
   and settle would never see its byte floor.

**What the chaining assumes** (UNVERIFIED, day 1):
- A load from fs or obj is promoted through the CPU tier, so the next rep's CPU read still hits. If not, the CPU
  rows from rep 1 on go invalid with source `disk`, and the CPU tier needs its own job.
- The 5 reps re-read the same stored prefix. They measure how repeatable a read is, not how reads vary across
  prefixes, and the store side (write, settle) is measured once per job, so once per length.
- fs reads go to the disk only if the fs tier uses O_DIRECT. With buffered I/O, the write leaves the files in the
  OS page cache, so even the first read can come from RAM; comparing reads by `read_index` cannot rule that out.
  O_DIRECT is therefore a gate for publishing fs numbers (day-1 checks).

**Where the driver runs.** On VM A, next to the writer, talking to both vLLM instances by private IP. A driver
on a laptop, over the internet, would add noise to every TTFT. vLLM binds to the private IP only, because its dev
endpoints (cache reset) must not be reachable from the internet.

**Validity prompts are natural text, not random token IDs.** On random tokens the logits are close to flat, so
bf16 noise flips greedy choices. The check compares greedy outputs only; a mismatch is looked at by hand, so there
are no logprob thresholds and no noise-floor run.

**GPU memory.** vLLM refuses to start if one `max_model_len` sequence does not fit in KV memory.
`max_model_len` is set per model: 66K for Qwen3-VL-32B on RTX, at `--gpu-memory-utilization 0.95`.
A unit test checks the budget.

## Hardware, budget, quotas

| Stage | Where | What | Time | Cost |
|---|---|---|---|---|
| 1 | Nebius **us-central1** | 2 × `gpu-rtx6000` `1gpu-24vcpu-218gb` VMs, **preemptible**, ~300 GB boot disk each; a 2000 GiB network SSD on VM A | about 12 h | about $19–33 at spot, $43 at most on demand, plus disks |

The total is about $45 of GPU time at most, plus disks. Prices are in [../configs/prices.yaml](../configs/prices.yaml) with their sources.

**Why us-central1 for stage 1 (decided 2026-11-16).** It had the cheapest preemptible RTX PRO 6000 at the time:
$0.79/GPU-h, the floor price, against $1.38 in eu-south1 and uk-south2.
- Over the previous 30 days no region stayed cheapest. Prices ranged from $0.79 to $1.94, and the medians were
  $0.94 in us-central1 and uk-south2 and $1.23 in eu-south1.
- The config already targets us-central1: its object-storage endpoint, region and quotas.
- Spot prices are uncapped by default. Set a max price of about $1.40/GPU-h in a Nebius pricing policy. Being
  outbid only preempts the VM, and `kvtb run` resumes.
- Check the live price before launching: [../scripts/spot-prices.sh](../scripts/spot-prices.sh).

- **Preemption.** The harness is resumable and writes each point as soon as it is done. `sweep.sh` copies the
  results to `s3://kvtb-cache/results/s1.jsonl` after every deployment, so a lost VM loses at most one deployment's
  rows. On a fresh VM it restores that file before running, and stops if it can't tell whether a copy exists. If preemption breaks the run twice, switch to on demand.
- **Quotas.** Request these before starting, in us-central1: preemptible GPU VMs ≥ 2 (the default is 2), disks ≥ 3
  (two boot disks and the disk tier).
- **Model weights.** `scripts/stage-weights.sh` downloads both models (~120 GB) to each VM's boot disk, in
  parallel. On GPU VMs this costs GPU time (roughly 10–20 minutes), so start it right after the VMs boot.
- **RAM per VM.** The RTX VM has 218 GiB. That has to hold the CPU tier (in `/dev/shm`), the weights being
  loaded, and the runtime, so the CPU tier is about 120 GiB there.
- **fs and obj never evict.** Add a cleanup step between models and size the disk for the sweep.

## Day-1 checks

These are unverified until checked on the VMs. Each one is in
[../src/kv_tier_bench/constants.py](../src/kv_tier_bench/constants.py) or [../scripts/](../scripts/).

- [ ] `vllm/vllm-openai:v0.33.0` includes the NIXL OBJ plugin and connects to the Nebius endpoint
      (`storage.<region>.nebius.cloud`, `scheme: https`). The obj tier probes the bucket at startup and fails fast.
- [ ] The p2p tier works over TCP. The docs say "over RDMA via NIXL". UCX should fall back to `tcp`; check it with
      `UCX_PROTO_INFO=y`.
- [ ] The hybrid model runs with the offloading connector and HMA on. Pass `--no-disable-hybrid-kv-cache-manager`
      explicitly, because llm-d warns that the connector otherwise turns HMA off. Then read the attention block
      size from the log line `Setting attention block size to N tokens`.
- [ ] The labels of `vllm:prompt_tokens_cached_by_source` (expected `device|host|p2p|disk|...`).
- [ ] Whether counters get a `_total` suffix.
- [ ] The `kv_load_tiers` matcher values for each tier. The code keys are `medium` and `locality`; CPU and
      STORAGE are expected, p2p is unknown.
- [ ] Whether `reset_prefix_cache?reset_external=true` keeps the secondary tiers, as the source suggests.
- [ ] The GPU is visible inside Docker on both VMs: `docker run --rm --gpus all ubuntu nvidia-smi`. The Nebius
      GPU image must have the NVIDIA container toolkit; install it if not.
- [ ] The firewall lets VM A reach VM B on 8000 (vLLM) and 7777 (p2p side channel), plus whatever ports UCX picks
      for the transfer. Check with `curl http://<VM B>:8000/v1/models` and the first p2p rows.
- [ ] A preempted VM keeps its boot disk (results, weights, compile cache). If it doesn't, restore the results from
      the bucket and re-stage the weights before rerunning.
- [ ] `fio` on the network SSD, with KV-chunk-sized I/O ([../scripts/fio-disk.sh](../scripts/fio-disk.sh)).
- [ ] `iperf3` between the two RTX VMs. Inter-VM bandwidth is not documented.
- [ ] Whether a hit from fs or obj reports as `disk` or as `host`, since loads stage through the CPU tier. Check
      `cached_by_source` in the first fs row; the breaker stops the run after 5 jobs with misattributed fs rows.
- [ ] **Gate for fs numbers:** the fs tier uses O_DIRECT. It falls back to buffered I/O and logs a warning when
      the filesystem doesn't support it; `day1-checks.sh` greps for it. With buffered I/O, every fs read, the first
      included, can come from the page cache, so the fs line would show RAM speed. Do not publish fs numbers until
      this passes: fix the mount or filesystem so O_DIRECT works, or drop the page cache before each fs read, and
      say which in the README.
- [ ] A load from fs promotes the blocks to CPU (see "What the chaining assumes"). Check a CPU row from rep 1 or
      later: in rep 0 the recompute read stores the prefix to CPU again, which would hide a missing promotion.
- [ ] `write_s` is a fair recompute baseline. It also does write-through store work. Compare `recompute_s` with
      `recompute_read_s` in `kvtb cost` at each length; if they differ, say so next to the cost table.
- [ ] The per-tier counters' `tier` labels have the form `<index>:<type>`, with 0 = CPU and 1 = the secondary
      tier. The driver picks each tier's series by that index from every scrape, so the type names never need
      to be known. Check `read_bytes_by_tier` in the first fs, obj, cpu and p2p rows: if the labels have another
      form, `read_bytes` / `read_time_s` are the sum over all labels and `TIER_INDEX` in `constants.py` needs
      fixing. A p2p row with `read_bytes: null` means p2p reads are not counted in these metrics.
- [ ] The compile cache under `/models/vllm-cache` is reused across the fs, obj and p2p deployments of a model.
      Unverified: the cache key may include the kv-transfer config, in which case each deployment compiles cold
      (about 5 more minutes per deploy). VM B has its own cache, so its first p2p deploy compiles cold.
- [ ] GDS stays in compatibility mode. The README says so; no GDS numbers are claimed.
- [ ] llm-d's published p2p-vs-recompute crossover. Two readings disagree: 52 vs 85 ms and 35 vs 78 ms at 2K
      tokens. Find which page each one came from before the README quotes either.

## Status

**Built and unit-tested with no GPU** (`uv run pytest`): sizes, the grid and result store, tier configs and the
`docker run` command, the job protocol against a fake engine, the resumable `kvtb run` loop with its guards
(endpoint check, warm-up, abort on an unreachable engine, a circuit breaker after 5 job attempts that fail the
same way, the settle byte floor), and the HTTP client against a mock transport. In a one-off check (the stubs
are not kept in the repo), `scripts/sweep.sh` ran against stub `docker`, `ssh` and `curl`: the command each VM
receives matches the one Python builds, and the container-exited, engine-unreachable, missing-PEER, backup and
restore paths behave as intended.

**What those tests don't prove:** the fake engine and mock transport encode our own reading of vLLM, and the
stubs are not Docker or Nebius. Nothing has run on a real VM.

**Not built yet:** the validity runner.

## Runbook

Create two preemptible `gpu-rtx6000` VMs in us-central1, in the same subnet, from a Nebius Ubuntu GPU image with
Docker, with ~300 GB boot disks. Attach a 2000 GiB non-replicated network SSD to VM A, and create the
`kvtb-cache` bucket (Enhanced Throughput). Everything below runs on VM A unless marked.

```bash
# from your machine: copy the repo to VM A
rsync -a --exclude .venv ./ <user>@<VM A public IP>:kv-tier-bench/

# on VM A, once
sudo mkfs.ext4 /dev/<kv disk> && sudo mkdir -p /mnt/kv && sudo mount /dev/<kv disk> /mnt/kv
ssh-keygen -t ed25519 && ssh-copy-id <user>@<VM B private IP>   # VM A starts vLLM on VM B over ssh
printf 'AWS_ACCESS_KEY_ID=...\nAWS_SECRET_ACCESS_KEY=...\n' > ~/kvtb-s3.env && chmod 600 ~/kvtb-s3.env
curl -LsSf https://astral.sh/uv/install.sh | sh && cd kv-tier-bench && uv sync
export PEER=<user>@<VM B private IP>
./scripts/stage-weights.sh                           # both VMs, in parallel

# day 1: start one deployment by hand, check it, fix constants.py if needed, stop it
eval "$(uv run kvtb serve-cmd --stage s1 --model qwen3.8-27b --deploy fs --host-ip "$(hostname -I | awk '{print $1}')")"
./scripts/day1-checks.sh | tee results/day1-fs.txt
docker rm -f kvtb-vllm

# the sweep, inside tmux; rerun the same command to resume (on a replacement VM too: it restores the results
# from the bucket first)
tmux new -s kvtb
./scripts/sweep.sh s1

# from your machine, afterwards
scp <user>@<VM A public IP>:kv-tier-bench/results/s1.jsonl results/
uv run kvtb cost --stage s1 --results results/s1.jsonl
uv run kvtb charts --stage s1 --results results/s1.jsonl
```

To save GPU time, run the fs and obj deployments with VM A alone (`DEPLOYS="fs obj" ./scripts/sweep.sh s1`), then
start VM B for p2p (`DEPLOYS=p2p`).

The sweep has 390 points in 45 jobs. It runs the deployments in this order: `fs` (recompute, gpu, cpu and fs),
then `obj`, then `p2p`.

**Time estimate** (assumed throughputs, nothing measured): under the earlier 1.3 h, which included concurrency
batches. The 6 deploys are now most of it, assuming the compile cache is warm after each model's first deploy (a
day-1 check; if not, add about 20 minutes).

## Schedule (2 weeks)

| Days | Work |
|---|---|
| 1–2 | Size model, grid, result store, cost maths and the driver, all tested without a GPU. |
| 3 | Nebius setup in us-central1: quotas, two VMs, the disk, the bucket. Run the day-1 checks. |
| 4–5 | Stage 1 sweep on RTX, plus validity runs. File an issue for any broken tier. |
| 6 | Fix the harness. Write the stage-1 analysis and charts. |
| 7–8 | Analysis, cost table, charts. |
| 9–10 | README and write-up. |
| 11–14 | Buffer. |

**What gets cut first if time runs short:** the 64K–128K points on stage 1.

## Out of scope

- An offload library of our own.
- Engine changes, or a break-even rule inside the engine.
- A recommender CLI.
- Time-based demotion, or a PR for it.
- InfiniBand or RDMA.
- A contention experiment, or any run under concurrent load.
- A trace-replay simulator.
- A second hardware point (H100, TP=4), or same-node p2p.
- An FP8-KV sweep.
- Comparing engines (SGLang, LMCache) or routers.
- Kubernetes, llm-d or any orchestration layer.
- A fault-injection suite.

## Known upstream issues to watch

- vllm PR #65239: hybrid models with the tiered offload path are "not yet validated".
- vllm#63718: tracking issue for which layouts the connector supports.
- vllm#49261: fs and obj tiers are not namespaced by model revision.
- vllm#53926: no per-request tier attribution.
- The fs and obj tiers have no capacity eviction.
- LMCache#5600: wrong linear-attention state restored. SGLang#45400: host Mamba pool fills up. Both are context for
  the validity check.
