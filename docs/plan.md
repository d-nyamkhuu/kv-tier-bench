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
| Engine | vLLM **v0.34.0** (switched from v0.33.0 on 2026-11-18, before any measurement), `OffloadingConnector` + `TieringOffloadingSpec` | v0.34.0 hardens the p2p tier and fixes KV-offload correctness bugs. The only stack with CPU, fs, obj and p2p in-tree and no patching. It has per-tier metrics and per-request load control (`kv_load_tiers`, `max_load_tokens`). |
| Host | **Plain VMs**: vLLM in Docker (`vllm/vllm-openai:v0.34.0`), started by `scripts/sweep.sh` over ssh. No Kubernetes, llm-d, router or Dynamo (decided 2026-11-18). | The benchmark needs two machines that can reach each other. Kubernetes added a driver image, a Job, RBAC, volumes and helper pods without changing any number. Dynamo's KVBM is deprecated and Dynamo pins vLLM 0.31. |
| Machines | 2 × Nebius `gpu-rtx6000-a` VMs (1 GPU each) in eu-south1, same subnet, one stage (s1) | The user has tested RTX PRO 6000 capacity on Nebius. |
| Hybrid model | **Qwen/Qwen3.8-27B**: 64 layers, 16 full attention and 48 Gated DeltaNet; 4 KV heads × 256 | Released 2026-08-05, Apache-2.0. |
| Standard model | **Qwen/Qwen3-VL-32B-Instruct**: 64 full-attention layers; 8 KV heads × 128 | Same KV width per layer as the hybrid model, so the KV difference is just 16 vs 64 layers. Context 262K. |
| Parallelism | TP=1 on RTX PRO 6000 | One card holds either model. PP would distort the recompute baseline. |
| Stage 2 (H100, TP=4) | **Dropped** (decided 2026-11-18) | It was ≤$144 of a ~$200 budget, for a second hardware point (TP=4 on H100) that the tier comparison doesn't need. |
| Precision | bf16 weights and KV. **bf16 linear-attention state on every run** (`--mamba-ssm-cache-dtype bfloat16`). | User decision. The HF default for the state is fp32, so the README must say it was overridden. |
| Prompts | Text only, token-ID prompts through `/v1/completions` | Both models are VLMs, and images would change the prefix hashing. Token IDs give exact lengths. |
| Speculative decoding / MTP | Off | Open issues for hybrid models with offload: vllm#50454, #55503, #53670. |
| Disk tier | 1767 GiB network SSD (non-replicated), ext4, attached to VM A at `/mnt/kv` | Nebius has no local NVMe outside B300; the README states this. A block device should support O_DIRECT, which the fs numbers need. Non-replicated disks come in 93 GiB units; 19 units (1767 GiB) reach the 2 GiB/s read maximum. |
| Object tier | Nebius Object Storage, **Enhanced Throughput** class only | User decision. Requests and same-region traffic are free. |
| Peer tiers | p2p over TCP between the two RTX VMs | InfiniBand is dropped. Same-node p2p is dropped too: it is a host-memory copy, close to the CPU tier. |
| Cost analysis | A static table: $ per 1M prefix tokens per tier, against recompute | User decision. There is no dynamic break-even and no TTL model. |
| Validity | Each point checks the per-tier source metrics. ~5 natural-text prompts per tier and model compare greedy output (~16 tokens) against recompute. | A cache hit counter does not prove the right KV was used. See LMCache#5600, SGLang#45563, vllm PR #65239. |
| Broken tier | File an upstream issue with a minimal repro and **do not debug**. Measure that tier on the standard model only. | User decision. |
| Driver | Our own small async Python driver, run on VM A | Off-the-shelf load tools can't reset tiers between requests, read the metrics, or inject `remote_kv_source`. |

## Tiers measured

On 2 × RTX PRO 6000 in eu-south1:

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
| 1 | Nebius **eu-south1** | 2 × `gpu-rtx6000-a` `1gpu-24vcpu-218gb` VMs, **preemptible**, 300 GiB boot disk each; a 1767 GiB non-replicated network SSD on VM A | about 12 h | about $19–33 at spot, $43 at most on demand, plus disks |

The total is about $45 of GPU time at most, plus disks. Prices are in [../configs/prices.yaml](../configs/prices.yaml) with their sources.

**Why eu-south1 (decided 2026-11-18, at launch).** At launch it had the cheapest preemptible RTX PRO 6000
(`gpu-rtx6000-a`): $1.38/GPU-h, against $1.70 in uk-south2 and $1.79 in us-central1 (us-central1 had been the
cheapest on 2026-11-16, at $0.79). `nebius capacity resource-advice list` showed both preemptible slots available
there (availability HIGH), and the project's quotas cover 2 preemptible VMs, 4 disks and 3 public IPs.
- Spot prices are uncapped: the VMs follow the spot price (`--follows-spot-price`); this CLI version (0.12) has no
  spot-price cap. Being outbid only preempts a VM, and `kvtb run` resumes.
- Check the live price before launching: [../scripts/spot-prices.sh](../scripts/spot-prices.sh).

- **Preemption.** The harness is resumable and writes each point as soon as it is done. `sweep.sh` copies the
  results to `s3://kvtb-cache/results/s1.jsonl` after every deployment, so a lost VM loses at most one deployment's
  rows. On a fresh VM it restores that file before running, and stops if it can't tell whether a copy exists. If preemption breaks the run twice, switch to on demand.
- **Quotas.** Request these before starting, in eu-south1: preemptible GPU VMs ≥ 2 (the default is 2), disks ≥ 3
  (two boot disks and the disk tier).
- **Model weights.** `scripts/stage-weights.sh` downloads both models (~120 GB) to each VM's boot disk, in
  parallel. On GPU VMs this costs GPU time (roughly 10–20 minutes), so start it right after the VMs boot.
- **RAM per VM.** The RTX VM has 218 GiB. The CPU tier (in `/dev/shm`) is 48 GiB: 3× the largest prefix
  (16 GiB). At TP=1 the GPU worker and the p2p tier's NIXL/UCX both pin it with `cuMemHostRegister`, so it
  must stay well under half of RAM. With 120 GiB the second pin failed with "out of memory" and vLLM did not
  start.
- **fs and obj never evict.** Add a cleanup step between models and size the disk for the sweep.

## Day-1 checks

These are unverified until checked on the VMs. Each one is in
[../src/kv_tier_bench/constants.py](../src/kv_tier_bench/constants.py) or [../scripts/](../scripts/).

- [ ] `vllm/vllm-openai:v0.34.0` includes the NIXL OBJ plugin and connects to the Nebius endpoint
      (`storage.<region>.nebius.cloud`, `scheme: https`; the AWS CLI reaches `storage.eu-south1.nebius.cloud`
      with the bucket keys, checked 2026-11-18). The obj tier probes the bucket at startup and fails fast.
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
- [x] The GPU is visible inside Docker on both VMs (2026-11-18). The `ubuntu24.04-cuda13.0` image ships Docker
      and the NVIDIA container toolkit; driver 580.173.02, RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- [x] The firewall lets VM A reach VM B on 8000 and 7777 (2026-11-18, default security group). Whatever ports UCX
      picks for the transfer: check in the first p2p rows.
- [ ] A preempted VM keeps its boot disk (results, weights, compile cache). If it doesn't, restore the results from
      the bucket and re-stage the weights before rerunning.
- [ ] `fio` on the network SSD, with KV-chunk-sized I/O ([../scripts/fio-disk.sh](../scripts/fio-disk.sh)).
- [x] `iperf3` between the two RTX VMs (2026-11-18, eu-south1, private IPs, 2 hops, mlx5 NIC at 400 Gbit/s): RTT
      0.25 ms min / 0.7 ms avg; 16 Gbit/s on 1 TCP stream, 46 Gbit/s on 4, 230–277 Gbit/s on 16, a ceiling of
      ~294 Gbit/s from 32 streams on (64: 293, 128: 289). There the receiver spends ~20 of its 24 vCPUs on TCP, so the
      limit is CPU, not the 400 Gbit/s NIC. p2p over UCX/TCP
      will see far less than the 16-stream figure if it uses one or a few connections.
      Measured: a 16K-token p2p pull (1,092 MiB) took 0.71 s of promotion time, about 1.5 GiB/s or 13 Gbit/s,
      close to the 1-stream figure. vLLM's p2p tier reports no read time, so p2p load time comes from
      `kv_offload_tiering_promotion_latency_seconds` (`promotion_s` on each row; it includes tier queueing).
      **p2p runs with a NIXL patch** (`p2p_nixl` in configs/experiment.yaml, `scripts/nixl_patch`). Stock vLLM sends
      each pull over one TCP stream: NIXL splits a transfer across its UCX workers only at >= 1024 descriptors, and
      it first merges a prefix's contiguous blocks into ~1 descriptor. The patch sets 16 workers, split_batch_size
      16 and skip_desc_merge, so a pull uses 16 connections (verified with `ss`: ~equal bytes on each). Result:
      2.3 GiB/s at 16K, 2.7 at 64K, 2.8 at 128K (24 Gbit/s), against 1.5 stock at 16K. The cap is one receive
      thread at 100% on the consumer (UCX/TCP emulates the remote write in software), not the link. The s1 p2p
      rows in results/s1.jsonl were measured with this 16-stream config.
      **p2p tuning, 2026-11-19 (s1 p2p rows re-run with it; old rows in results/s1-p2p-16stream.jsonl).** Three NIXL limits, all fixed in `scripts/nixl_patch` without
      rebuilding NIXL or vLLM. Measured with `scripts/p2p_bench.py` (two NIXL processes in the vLLM image, the p2p
      tier's WRITE pattern, every block's destination verified):
      1. *One receive thread per agent.* NIXL 1.5.2's UCX backend advertises only worker 0's address
         (`getConnInfo`), and every sender worker connects to it, so all receive work lands on the receiver's one
         shared progress thread. Same on NIXL main. Fix: `kvtb_stripe.py` gives vLLM 32 NIXL agents per process and
         splits each pull into 32 stripes, agent i to peer agent i (`KVTB_NIXL_STRIPES`). 64 agents fail to connect
         across VMs (UCX "Destination is unreachable" in loadRemoteMD); 48 were no faster than 32.
      2. *Progress threads never sleep.* The Python `nixl_agent` hardcodes `pthrDelay = 0`, so the shared progress
         thread calls `poll()` with a 0 ms timeout: 100% of a core per agent even when idle (800% for 8 agents),
         holding the UCX worker lock. Fix: `KVTB_NIXL_PTHR_DELAY_US=1000`. Idle CPU fell to ~11%.
      3. *Posting is serial and blocking.* `postXferReq` sends inline and returns late (0.24-0.95 s of a 0.4-1.3 s
         pull), so stripes posted in turn ran in turn, and vLLM's scheduler thread waited inside `write_blocks`.
         Fix: one post thread per agent (the binding releases the GIL), agents in NIXL_THREAD_SYNC_RW;
         `transfer()` returns in ~0.2 ms.
      Ruled out (no effect): UCX fragment size (`UCX_TCP_RX_SEG_SIZE` 64K-8M), AppArmor/seccomp unconfined, socket
      buffers (autotuned to 4-6 MiB), the TCP put ACK (puts are pipelined).
      One VM, two containers on a Docker bridge (8 GiB pull): 6.2 GiB/s with the 16-stream config -> 14.8 striped ->
      **31.0** with all three fixes (across VMs the 16-stream config gives 3.0). Ceilings there: plain sockets with DRAM-sized buffers 32-34 GiB/s (12-16 connections;
      `iperf3` reads 76 because its buffer stays in L3), memcpy ~90 GiB/s. Across the two VMs (VPC, `iperf3` 34-41 GiB/s at 32
      streams): striping alone gave 18.4 GiB/s (8 agents x 2 workers); all three fixes with 32 agents and no dedicated
      UCX workers (`num_threads: 0`), 48 GiB registered as in vLLM: 24-34 GiB/s for an 8 GiB pull, 24-27 for 1 GiB.
      Run-to-run network noise is 15% or more (one 13.2 GiB/s run, 33.7 on the interleaved retry).
      In vLLM (Qwen3-0.6B pair on one VM, 32K prefix = 3.5 GiB): 5.4 -> 26.7-28.3 GiB/s, p2p TTFT 0.78 -> 0.25-0.27 s,
      greedy output equal to recompute on 3 of 3 prompts, all tokens from `p2p`
      (results/p2p-1vm-qwen3-0.6b-check.out). Where a 32K pull's 0.236 s engine TTFT goes: queue 0.223 s, of which
      the deferred lookup 0.145 (almost all promotion, i.e. the transfer: 0.131); `lookup_sync_delay` (0.052 summed)
      is time inside the scheduler's repeated lookup calls during that wait, not an extra stage. The remaining ~0.078 s
      of queue is inferred (by subtraction) to be the CPU->GPU load, ~45 GiB/s; prefill 0.007. Overlapping the load
      with the transfer would need a vLLM change: the CPU tier only loads a chunk once its promotion is complete.
      Across the two VMs in vLLM (Qwen3-0.6B, 131K context via a YaRN override, 114688 B/token;
      results/p2p-2vm-qwen3-0.6b-check.out), end to end from first request to last response, so including the
      CPU->GPU load and prefill: one 32K pull (3.5 GiB) 9.4-14.0 GiB/s, transfer alone (promotion) 0.12-0.23 s, i.e.
      15-29 GiB/s; one 124K pull (13.6 GiB) 15.4-15.6 GiB/s, promotion 0.43-0.44 s = 31 GiB/s; parallel 32K pulls
      2 -> 19.0, 4 -> 23.0-23.9, 8 (28 GiB) -> 26.2-26.3 GiB/s. So a single long pull already runs the link at the
      harness peak, and concurrent pulls hide the load and prefill. Every token came from `p2p`; greedy equal on 3 of
      3. Published p2p numbers are "vLLM v0.34.0 + this NIXL tuning".
      **obj runs on the image's CUDA 12 NIXL build** (`obj_nixl_cuda_major` in configs/experiment.yaml, same hook).
      In vllm/vllm-openai:v0.34.0, nixl_cu13's OBJ plugin links libcuobjclient.so.1, which the image lacks, so the
      obj tier failed at startup (NIXL_ERR_NOT_FOUND, "unsupported backend 'OBJ'"). nixl_cu12 1.5.2's OBJ plugin
      loads, but its AWS SDK reads CA certs from /etc/pki/tls/certs/ca-bundle.crt (curl 77 otherwise), so launch.py
      mounts the host's bundle there. First obj rows: valid, `disk` source, bytes on `1:obj`.
      **s1 p2p re-run, 2026-11-19, with this tuning:** 75 of 75 valid, all rows with `engine_queue_s` /
      `engine_prefill_s`. Median read at the longest prefix, old (16-stream config) -> new: qwen3.8-27b 128K 3.45 ->
      0.69 s (promotion 3.09 -> 0.35 s, 22.8 GiB/s), qwen3-vl-32b 64K 6.10 -> 1.00 s (promotion 0.58 s, 27.7 GiB/s).
      p2p now reads about 2x faster than obj (1.27 and 2.32 s) and about 2x slower than cpu (0.31 and 0.40 s) at every
      length. The 16-stream rows are kept in results/s1-p2p-16stream.jsonl. The sweep has no greedy check (the
      validity runner is not built); greedy equality was checked on the Qwen3-0.6B pairs above.
- [x] (2026-11-18: fs hits report as `disk`, 896 of 896 tokens in the first fs row.) Whether a hit from fs or obj reports as `disk` or as `host`, since loads stage through the CPU tier. Check
      `cached_by_source` in the first fs row; the breaker stops the run after 5 jobs with misattributed fs rows.
- [x] (2026-11-18: passes. ext4 on /mnt/kv, no fallback warning in the vLLM log; v0.34.0 fs/manager.py logs
      "O_DIRECT is not supported at ...; falling back to buffered I/O" when its startup probe fails.) **Gate for fs numbers:** the fs tier uses O_DIRECT. It falls back to buffered I/O and logs a warning when
      the filesystem doesn't support it; `day1-checks.sh` greps for it. With buffered I/O, every fs read, the first
      included, can come from the page cache, so the fs line would show RAM speed. Do not publish fs numbers until
      this passes: fix the mount or filesystem so O_DIRECT works, or drop the page cache before each fs read, and
      say which in the README.
- [ ] A load from fs promotes the blocks to CPU (see "What the chaining assumes"). Check a CPU row from rep 1 or
      later: in rep 0 the recompute read stores the prefix to CPU again, which would hide a missing promotion.
- [ ] `write_s` is a fair recompute baseline. It also does write-through store work. Compare `recompute_s` with
      `recompute_read_s` in `kvtb cost` at each length; if they differ, say so next to the cost table.
- [ ] The per-tier counters' `tier` labels have the form `<index>:<type>`, with 0 = CPU and 1 = the secondary
      tier (the v0.34.0 source says `0:primary` and `1:<type>`; confirm in a live scrape). The driver picks each tier's series by that index from every scrape, so the type names never need
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

Create two preemptible `gpu-rtx6000-a` VMs in eu-south1, in the same subnet, from a Nebius Ubuntu GPU image with
Docker, with 300 GiB boot disks. Attach a 1767 GiB non-replicated network SSD to VM A, and create the
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

## Stage 2: concurrent load (s2)

s1 measures one request at a time. s2 asks how the tiers hold up when several requests load their prefixes at
once: bandwidth that is shared (PCIe, the VPC link, the disk, the bucket) and loads that overlap each other.
Code and tests are in; nothing has run on a GPU yet.

**Where.** us-central1, where RTX PRO 6000 spot was cheapest on 2026-11-19 ($1.19 per GPU-hour vs $1.38 in
eu-south1; $1.80 on demand in both). Two spot VMs on `gpu-rtx6000`, the same GPU and preset as s1's
`gpu-rtx6000-a` (24 vCPU, 218 GB), so the VPC link and host may differ from s1's. VM A has a 1767 GiB
non-replicated network SSD at /mnt/kv, as in s1, and a 1 TiB shared filesystem at /mnt/fs (below). Both buckets are in us-central1: `kvtb-cache` (Enhanced
Throughput; also the results backup) and `kvtb-cache-std` (Standard). The eu-south1 VMs, disks and bucket were
deleted on 2026-11-19 after their logs were copied to results/vm-archive/eu-south1/.

**Models.** On the RTX PRO 6000 the s1 models leave ~25-30 GiB for KV after weights, so at most one 64K prefix
(qwen3-vl-32b) or about three 128K prefixes (qwen3.8-27b) fit at once. s2 uses two ~9B models of the same two kinds
instead (sizes from their config.json, not measured):

| | KV per token | one 128K prefix | GPU KV budget | grid's largest batch |
|---|---|---|---|---|
| Qwen3.5-9B (hybrid, the 27B's layout: 8 full-attention + 24 Gated DeltaNet layers) | 32 KiB | 4 GiB | ~67 GiB | 8 x 128K, 16 x 64K |
| Qwen3-8B (standard, 36 full-attention layers, text-only) | 144 KiB | 18 GiB | ~72 GiB | 2 x 128K, 8 x 32K |

Qwen3-8B is trained to 32K. It runs at 128K with a YaRN `--hf-overrides` (factor 5, so a 128K prompt plus its
token fits; factor 4 gives exactly 131,072 positions), with `rope_theta` repeated inside `rope_scaling`, which the
Qwen3-0.6B p2p check needed. A smaller model also makes recompute cheaper, so tiers save less than on s1's models at
the same length; s2's numbers are not a substitute for s1's.

**Grid** (`stages.s2` in configs/experiment.yaml): prefixes of 4K, 16K, 32K, 64K and 128K tokens, batches of 1, 2,
4, 8 and 16 requests, s1's six tiers plus `obj-std` and `fs-shared`, 3 reps; recompute batches also get 3 reps. A batch is left
out unless it fits as a whole (`grid.fits`: c x prefix bytes within 80% of the GPU KV budget, and of the CPU tier for tiers loaded through
it). That drops 16 x 128K on the 9B, and 4 x 128K, 8 x 64K and 16 x 32K on the 8B. At these powers of two, the
48 GiB CPU tier drops nothing that GPU memory allows. 1032 points in 215 jobs.

**Two fs tiers: the disk and a shared filesystem.** `fs` is vLLM's fs tier on the disk, as in s1. `fs-shared` is
the same tier (10 MiB files, 16 read and 16 write threads) on a 1 TiB Nebius shared filesystem (network_ssd type,
$0.08 per GiB-month, ~$0.11/h for 1 TiB; the disk is $0.053, ~$0.13/h for 1767 GiB), mounted on VM A over virtiofs
at /mnt/fs (`shared_fs_mount`, `shared_fs_price`). Its own deployment. O_DIRECT works on the mount (no fallback
warning in vLLM's log).

fio (2026-11-19, 8 MiB direct I/O, 60 s per phase on 64 GiB of fresh files, run on VM B) favoured the filesystem:
4.97 GiB/s for 16-way writes, 8.17 GiB/s for 16-way random reads and 1.77 GiB/s single-stream, against 1.48, 1.94
and 1.69 GiB/s on the disk. So the filesystem was first meant to replace the disk, and the sweep's first fs runs used
it. vLLM's access pattern disagreed (final runs, median batch read time on Qwen3.5-9B, disk vs filesystem): 4K x 1
0.14 vs 0.18 s, 4K x 16 1.39 vs 0.67 s, 16K x 1 0.49 vs 0.73 s, 32K x 1 0.87 vs 1.37 s, 128K x 1 2.47 vs 8.13 s.
The filesystem wins only on small prefixes read many at a time. A likely cause (not tested): a 32K prefix is ~100
files of 10 MiB, and each open costs more on virtiofs than on ext4. fio read few large files and never paid it. So
s2 measures both.

How the rows came about, so they can be traced:
- The disk's first Qwen3.5-9B fs run (stopped at job 19 of 24; 48 of its rows overlap a fio run on the same disk)
  is kept apart in results/s2-fs-nrd-disk.jsonl and is not in s2's tables. It is also suspect on its own: from 16K
  up it delivered 3.1-3.8 GiB/s (read_bytes / read_s) at 2-4 requests, above the disk's fio read cap of 1.94 GiB/s,
  so part of it was likely served from a cache. The final run reads at 2.0-2.3 GiB/s from
  2 requests up and peaks at 2.6 GiB/s on a few short batches (16K x 4), so short reads may still get a burst
  allowance or some cache; nowhere near the first run's 3.8. The fs
  container's O_DIRECT log line was not checked in s2 (only the filesystem's), and the VM is gone.
- The fs deployment then ran with the tier rooted at /mnt/fs. Its fs-tier rows are fs-shared rows (same vLLM
  config), and `scripts/relabel-fs-shared.py` renames their tier, id and job. Its recompute, gpu and cpu rows stay
  as fs-deployment rows.
- The fs points are then measured on the disk with the disk reattached. The fs jobs get a fresh attempt there, so
  new prompts (cold on a fresh vLLM either way).

**Both Object Storage classes.** `obj` is the Enhanced Throughput class, as in s1 (bucket `kvtb-cache`, $0.11 per
GiB-month, requests free, unthrottled). `obj-std` is the Standard class, in its own bucket `kvtb-cache-std`, because
a bucket's default class applies to every new object and NIXL's OBJ plugin sets no per-object class. Standard costs
$0.0147 per GiB-month plus $0.0034 per 1000 PUTs and $0.00028 per 1000 GETs (`kvtb cost` leaves requests out). Nebius
caps its reads at 20 GBps per tenant per region and quotes Enhanced as up to 10x the throughput per GiB stored
(docs.nebius.com/object-storage/storage-classes, 2026-11-19). The third class, Intelligent (auto-tiering, warm
$0.03), is not measured. `obj-std` is its own deployment, set up like `obj` (keys, the CUDA 12 NIXL build).

**Order.** `sweep.sh` runs p2p first, for every model, and then says VM B can be stopped. The other deployments
(fs, obj, obj-std) need VM A only. It cleans the disk tier and both buckets after each model's last deployment.

**Protocol.** A job at concurrency c stores c distinct prefixes of one length (`Job.prompt_seeds`). They are
written one at a time, each settled before the next, so a burst of stores cannot be dropped. So `write_s` is the
mean cold prefill of one prefix, not a concurrent baseline. Each point resets as in s1, then sends all c reads at
once (`measure.all_or_nothing`: if one fails, the rest are cancelled). Each p2p request is its own pull, with its own
`kv_request_id`. A batch is valid only if the tier served all c prefixes (summed cached tokens by source of at least
c x prefix). Point and job ids gain `/c<n>` above concurrency 1, so s1's ids are unchanged (a test checks every s1
id against results/s1.jsonl).

**What a batch row holds, and the cost table** (`load.py`, `kvtb cost --stage s2`, `kvtb charts --stage s2`):
- `read_s`: from the first request sent to the last response. It is what the GPU is held for, and the cost table
  charges it over the c x prefix tokens the batch serves.
- `read_s_each`: each request's own time to its token. The charts show the median per batch.
- `read_bytes`: what the tier delivered for the whole batch. The table's `batch_gib_s` is read_bytes / read_s, end
  to end (GPU copy and prefill included), so it is lower than s1's `load_gib_s` for the same link: never compare the
  two.
- `read_time_s` and `promotion_s` are counter deltas, so in a batch they are sums over the c requests, not wall time
  (read_bytes / promotion_s understates the rate by up to c). The `engine_*_s` fields are means over the c requests.
- The baseline is the recompute batch at the same length and concurrency.
- `engine_queue_s` is kept on the rows but not used: in a batch it also holds the wait for batch-mates.

**Runbook.** Everything s2 used in us-central1 (both VMs, the disk, the filesystem, both buckets and the access key) was deleted on 2026-11-19 after the run; a rerun creates them again, with keys that reach both buckets. `STAGE=s2 PEER=... ./scripts/stage-weights.sh`,
then `PEER=... ./scripts/sweep.sh s2` (it reads the stage's models). Stop VM B when the script says p2p is done.
Then `kvtb cost` and `kvtb charts` with `--stage s2 --results results/s2.jsonl`. A rerun after VM B is stopped needs
`DEPLOYS="fs fs-shared obj obj-std"`. The filesystem (VM A must be stopped to attach one): create a 1 TiB
network_ssd filesystem in us-central1, stop VM A, attach it with mount tag `kvtbfs` next to the disk, start VM A,
then `sudo mkdir -p /mnt/fs && sudo mount -t virtiofs kvtbfs /mnt/fs` (and an fstab line).

**Day-1 checks for s2:**
- [ ] Qwen3-8B starts with the YaRN override (look for the rope config in the startup log), and a 128K prompt plus
      one token returns 200.
- [ ] Qwen3.5-9B's block size (hybrid: enlarged at startup, as for the 27B): compare the startup log line with what
      the driver reads from /metrics.
- [x] The first s2 batch rows: `cached_by_source` equals c x prefix for the tier under test, and `read_s_each` has c
      entries (every batch row, 2026-11-19).
- [x] Sequential writes settle for fs and obj at c = 8 and 16: no row has `stores_dropped`.
- [x] How v0.34.0 schedules c long prefills sent at once: a recompute batch takes 0.91-1.04x the sum of its job's
      sequential prefills (median 0.98-0.99), so prefill is compute-bound and batching it buys nothing. The savings column rests on the recompute batch: if the
      engine prefills long prompts mostly one after another, the batch takes about c x one prefill. Compare a
      recompute batch's `read_s` with the sum of its job's `write_s_each` (the same prompts, written one at a time).
- [x] Batches through fs and obj: every fs-shared, obj and obj-std batch is valid, up to 16 promotions at once.
- [ ] The first `obj-std` row: bytes land in `kvtb-cache-std`, and its objects show the Standard class.

**Time estimate** (assumed throughputs, nothing measured): ~3-4.5 h of VM A, of which ~1 h also needs VM B (p2p).
Per model and deployment, qwen3-8b stores 272 GiB and qwen3.5-9b 176 GiB. The fs reads are 1,342 GiB at ~1 GiB/s
(~22 min), and Standard-class reads may be as slow. Prefill covers ~31M tokens of writes and ~23M of recompute
batches (60-90 min at a guessed 10-15K tokens/s). Add the 8 deploys.

**First s2 findings (2026-11-19, while the sweep runs):**
- p2p transfers at 10-12 GiB/s per pull in us-central1 (Qwen3.5-9B 128K: 4.0 GiB in 0.39 s; Qwen3-8B 128K: 18 GiB
  in 1.56 s), against 23 GiB/s for s1's 27B in eu-south1, with the same NIXL settings (checked in the launch
  command). Concurrent pulls do not add up past ~10-11 GiB/s in total. iperf3 between the VMs gives 27.4 GiB/s at 32
  streams, so the limit is NIXL/UCX on these hosts (Intel Xeon 6776P on `gpu-rtx6000`), not the link. Not diagnosed:
  `scripts/p2p_bench.py` between the two VMs would say whether it is the platform (then retune) or vLLM's use of
  NIXL. s2's p2p numbers hold for this platform as configured, not as p2p's ceiling.
- fio on VM A's non-replicated network SSD (1767 GiB, 8 MiB direct I/O): 16-way writes 1.48 GiB/s, 16-way random
  reads 1.94 GiB/s, single-stream reads 1.69 GiB/s: the documented caps (~2 GiB/s read, ~1.5 write). s1's fs reads
  (1.0-1.3 GiB/s) were below them, so vLLM's fs path, not the disk, set s1's fs numbers.
- That fio run started late and overlapped the first 48 rows of qwen3.5-9b's first fs run on the disk (all 4K-token
  batches, c = 1-8, 05:08:21-05:08:57 UTC; fio ended 05:09:15). That whole run is kept apart
  (results/s2-fs-nrd-disk.jsonl) and the disk's fs points are measured again.

**s2 results (2026-11-19; 1032 of 1032 points valid; `kvtb cost --stage s2`; charts in
results/charts/load-s2-*.png and ttft-s2-*.png).** Times and rates are measured. Savings are modelled: the GPU at
$1.80/h on demand (the run used $1.19 spot), each prefix stored for 1 h and read once in that hour, storage billed
on bytes used, Standard-class request fees left out. Every "loses money" below is the storage term outweighing a
faster load; a prefix read several times per hour would come out ahead.
- gpu and cpu serve every batch in under 1 s and save 79-99% against recompute at every point.
- p2p is the fastest tier that leaves the box: 2.1 s for one Qwen3-8B 128K prefix (18 GiB; recompute 21.6 s) and
  3.2 s for 8 x 128K on Qwen3.5-9B (recompute 72.6 s). It saves 66-96% on Qwen3.5-9B and 18-92% on Qwen3-8B, the low
  end at 4K, where a pull's fixed cost is most of the time.
- obj (Enhanced Throughput) loads at about 4-5 GiB/s per batch: 3.5 s for a Qwen3-8B 128K prefix, 8.3 s for 8 x 128K
  on Qwen3.5-9B. Its storage price decides the sign: on Qwen3.5-9B (32-40 KiB/token measured) it saves 36-76%, on Qwen3-8B
  (144 KiB/token, 3.6-4.5x the bytes) it loses money below 32K and saves 33-60% at 64K-128K.
- obj-std (Standard) loads at 0.4-0.6x obj's speed on the 9B and 0.25-0.33x on the 8B, but stores for 1/7 of the
  price. On the 9B it saves about as much as obj (24-76%); on the 8B less: it loses money at 4K-16K and at 32K
  beyond 2 requests, and saves 9-47% at 64K-128K. Request fees are not in these numbers.
- fs (the 1767 GiB network SSD) reaches its ~2 GiB/s read cap from 2 requests up (mostly 2.0-2.3 GiB/s of
  measured bytes on Qwen3.5-9B, 2.6 at most; Qwen3.5-9B moves 32-40 KiB per token: KV plus a fixed state per prefix), 1.0-1.6 GiB/s for a single
  prefix. On Qwen3.5-9B it saves 25-72% (5% for one 4K request). On Qwen3-8B, whose prefixes are 3.6-4.5x the bytes,
  it is slower than recompute at 4K-16K and at 32K x 1, faster from 32K x 2 up, and saves 13-45% from 64K x 2 up
  (12.9 s for one 128K prefix, recompute 21.6 s). Between, it is faster but its storage cost keeps it below zero.
- fs-shared is the slowest tier: ~0.5 GiB/s for one long prefix (8.1 s for 4 GiB, 37 s for 18 GiB), rising to
  ~1-1.8 GiB/s with several prefixes at once (2.8 GiB/s for 16 x 4K). On Qwen3.5-9B it breaks even at one request
  (-26% to +1%) and saves 37-58% at 4-16. On Qwen3-8B it loses money at every point. Against the disk: slower at 20
  of Qwen3.5-9B's 24 points (3.3x for one 128K prefix, 1.4-1.9x for most 16K-64K batches); faster only for 4K
  batches of 4-16 (0.48x at 16) and level at 64K x 16. The fio result (filesystem 4x the disk at 16-way reads) does
  not carry over to vLLM's fs tier, which reads a prefix as ~10 MiB files with 16 threads.
- Prefill is compute-bound: a recompute batch costs c single prefills. So every tier's saving grows with batch size
  as its loads overlap, while recompute's cost per token stays flat.

**Not in s2:** background decode streams during the loads (interference with decoding), mixed or trace-driven
workloads, the greedy validity runner, and a bigger GPU (H100/H200, or TP=2) where 8 x 128K fits for a 27-32B model.

## Stage 3: a CPU-only KV store (s3), measured

**Results (2026-11-19, eu-south1; full write-up in [report.md](report.md#stage-3-a-cpu-only-kv-store)).** 315 of 315
points valid, in [results/s3.jsonl](../results/s3.jsonl): p2p and store on both models, store-p2p on Qwen3-8B.
Recompute, GPU and CPU were not re-run; `baseline_from: s2` reuses s2's rows.
- **p2p beats the store.** Over matched points the store took a median 2.5× p2p's time (1.5–3.0× on Qwen3-8B,
  1.4–3.6× on Qwen3.5-9B). The store's best batch moved 5.0 GiB/s (Qwen3-8B) and 6.0 GiB/s (Qwen3.5-9B); p2p's
  14.9 and 21.4.
- **The pool's host does not matter.** store-p2p (the pool inside vLLM on A, read from B) took a median 1.03× the
  store's time (0.99–1.09×).
- **p2p in eu-south1** was a median 1.2× (Qwen3-8B) and 1.6× (Qwen3.5-9B) faster than s2's in us-central1.
- **Tuning, not the VM, set the store's speed.** Mooncake's defaults were 3.2–4.8× slower than the final settings.
  The 16 vCPU CPU VM matched the 64 vCPU one at every shared point (0.96–1.05×), but its 48 GiB pool evicted parts of
  large batches before they were read; the run used 64 vCPU, 256 GiB, a 200 GiB pool and a 60 s lease.
- **Cost at production reuse** (report: "Cost at production reuse"). With RAM priced as a CPU-only VM's and one
  hour of residency shared by 100–1,000 reads, the remote tiers save 68–91% at 128K on Qwen3-8B; a prefix read once
  an hour loses (−630% to −5,200%). Break-even is 15–17 reads per prefix-hour at 128K on Qwen3-8B, about 8 on
  Qwen3.5-9B. The local CPU tier beats every remote tier, and on Qwen3-8B Object Storage saves at least as much as
  the store. A $2.11/h store VM pays for itself at 240 hits an hour on 128K Qwen3-8B prefixes, 134,000 on 4K ones.
- **Recompute baselines from us-central1 hold.** s3's p2p write phase (a prefill on A) took 1–7% longer than s2's
  recompute at the same lengths.


**Question.** Can a CPU-only VM serve as the remote KV tier, as fast as p2p and for less? In s1 and s2, p2p pulls
prefixes from a second *vLLM instance*, so the machine holding the KV needs a GPU. s3 puts the KV on a VM with no
GPU and compares it with p2p on the same day, in the same region, with the same grid as s2.

**How vLLM does it** (docs.vllm.ai, stable, read 2026-11-19):
- **MooncakeStoreConnector, `standalone-store` mode.** Stores KV in MooncakeDistributedStore:
  - `mooncake_master` coordinates;
  - an external `mooncake_client` owns the CPU pool (optionally with an SSD tier behind it);
  - each vLLM rank is a pure requester (`global_segment_size: 0`).
  - Protocols: `rdma` (recommended) or `tcp`. The RTX PRO 6000 VMs have no InfiniBand, so s3 uses `tcp`.
  - **Hybrid models:** supported with conditions. The same `--prefix-match-unit` on every endpoint; entries for
    Mamba/GDN layers are full state snapshots, with the same convolution layout everywhere.
  - **Hashing:** block hashes must agree across processes, so `PYTHONHASHSEED` is fixed.
  - **Lookups:** time out after 1 s; after three failures, lookups are skipped for 30 s.
  - **Hit signals:** `vllm:mooncake_store_operation_bytes_total` (`operation=load_get`, `status=success`) and
    `VLLM_MOONCAKE_STORE_TIER_LOG=1`.
- **LMCacheMPConnector** with a standalone `lmcache server` is the other option. It needs the non-hybrid KV cache
  manager, so it could run Qwen3-8B but not Qwen3.5-9B. The docs only show it on localhost. Kept as optional, and
  only if a remote host works on day 1.
- **Not a tier of our connector.** Both are their own connectors, not secondary tiers of the `OffloadingConnector`
  that s1 and s2 measure. So a `store` deployment is GPU → store, with no CPU tier in front, unless the
  `MultiConnector` can stack them (check on day 1). It is measured like any other tier: a GPU miss read back from
  the store.

**What it costs** (docs.nebius.com/compute/resources/pricing, 2026-11-19):
- **CPU-only VMs (AMD EPYC Genoa):** $0.015 per vCPU-hour plus $0.0045 per GiB-hour of RAM, on demand. No
  preemptible price is listed. Presets give 4 GiB per vCPU, so RAM effectively costs $0.0083 per GiB-hour:
  - `16vcpu-64gb` is $0.53/h;
  - `64vcpu-256gb` is $2.11/h.
- **The GPU VM, for comparison:** $1.80/h for 218 GiB on demand ($0.0083 per GiB-hour too), and spot from $0.79
  ($1.19 at s2's launch).

So a CPU store is **not cheaper per GiB of RAM**. It wins only if:
- it is sized to the cache instead of to a GPU;
- it runs where no GPU is free;
- it is shared by several GPU instances, the case it is built for.

**Cost-model fix that s3 needs.** `kvtb cost` prices p2p storage as "included". That holds when the peer is a GPU
VM serving traffic anyway, not when a VM exists only to hold KV. s3 prices both p2p and the store at the $/GiB-hour
of the RAM that holds the prefix, with the "included" case kept as a second column.

**VMs, all in one region:**
- **VM A:** GPU, the vLLM under test.
- **VM B:** GPU, the p2p peer. Only for p2p and the p2p diagnosis.
- **VM C:** CPU-only, the store: `mooncake_master` and `mooncake_client`.
- **Region.** us-central1, to compare with s2, *after* diagnosing s2's p2p limit there (10-12 GiB/s per pull
  against 23 in eu-south1; `scripts/p2p_bench.py` between A and B). Otherwise eu-south1, where p2p is known to
  reach 23-28 GiB/s.
- **Sizing VM C.** One job's prefixes at a time is at most 36 GiB (Qwen3-8B, 2 x 128K), so 64 GiB of RAM is enough.
  The CPU VMs' network bandwidth is not documented, and on many clouds it scales with vCPU count. So day 1 measures
  iperf3 A→C on `16vcpu-64gb` and `64vcpu-256gb` and picks the smallest one that is not the bottleneck.

**Grid.** s2's: prefixes of 4K-128K tokens, batches of 1-16, 3 reps.
- **Models:** Qwen3-8B (standard, 144 KiB/token, the case where bytes matter most). Qwen3.5-9B (hybrid) only if
  Mooncake's hybrid support works on day 1.
- **Tiers:**
  - recompute, gpu, cpu;
  - p2p, re-measured beside the store;
  - `store` (Mooncake, CPU pool on VM C);
  - optionally `store-ssd`, the same with the client's SSD tier.
- **Optional s3b, a shared prefix.** VM A writes, a second vLLM instance on VM B reads the same prefix from the
  store. That is the store's real use, and p2p's too.

**Harness (written 2026-11-19; nothing of it has run on a VM yet):**
- **Stage `s3`** in `configs/experiment.yaml`: s2's grid and VM settings, tiers `recompute, gpu, cpu, p2p, store`,
  models Qwen3-8B then Qwen3.5-9B. There is no KV disk, so recompute, gpu and cpu run on the p2p deployment's writer
  (`tierconfig.BASE_DEPLOYMENT`). s1 and s2 keep every point and job id, and their `kvtb cost` output is unchanged.
- **The `store` deployment** (`tierconfig`): `{"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}`.
  The rest is in the Mooncake config that `kvtb mooncake-config --stage s3 --store-ip <C>` writes:
  `standalone-store`, `global_segment_size: 0`, tcp, `P2PHANDSHAKE` (the docs' example, so no metadata server), a
  4 GiB `local_buffer_size`, tenant `default`. `launch.py` mounts it and sets `MOONCAKE_CONFIG_PATH`,
  `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` (the private IP) and `VLLM_MOONCAKE_STORE_TIER_LOG=1`. `PYTHONHASHSEED=0` was
  already set.
- **Writes** settle on `vllm:mooncake_store_operation_bytes_total{operation="save_put",status="success"}`. The floor
  is 90% of the prefix's KV bytes, not fs/obj's 10%, because the store has no active-job gauge and its counter moves
  only as each put completes. A failed put key counts as a dropped store, which invalidates the job's store reads.
- **Reads** reset the GPU prefix cache only (`reset_external` off) and send no `kv_load_tiers`.
  - `read_bytes` and `read_time_s` come from `load_get`.
  - Store hits' source label is not known yet, so every non-`device` source counts.
  - A point is valid if cached tokens ≥ c × prefix, `load_get` delivered at least 90% of the prefix's KV bytes, and
    no load key failed.
  - After an invalid store attempt, `kvtb run` waits 31 s before retrying, past Mooncake's 30 s lookup back-off.
- **`store-p2p`: Mooncake as p2p.** VM A's vLLM runs embedded and holds the pool (`pool_gib` of its own RAM),
  and `mooncake_master` runs on A (`store-up.sh` with `MASTER_ONLY=1`). VM B's vLLM is a pure requester and does
  the reads, as in p2p. So it has p2p's holder (a GPU VM's RAM) and the store's transport (Mooncake TCP), which
  tells the transport apart from where the KV lives.
- **Mooncake's TCP defaults are slow** (found on the first run, 2026-11-19).
  - The defaults are 1 TCP I/O thread and 4 lanes per peer, plus 1 vLLM load thread. With them, the store loaded
    one 4K prefix in 0.51 s and one 16K prefix in 1.8 s, about 1.3 GiB/s. Eight 16K prefixes at once loaded at
    1.1 GiB/s.
  - `transfer_engine_bench` from A to C:
    - 1 I/O thread, 4 lanes: 2.0–2.4 GB/s;
    - 8 I/O threads, 16 lanes: 9.5–12.0 GB/s;
    - 24 or 32 lanes collapse to about 4.4 GB/s, whatever the thread count.
  - Into GPU memory, slices set the rate: 2.5 GB/s at 64 KiB, 6.4 at 1 MiB, 7.8 at 4 MiB, 8.2 at 9 MiB. The 64 vCPU
    VM moved 14.1 GB/s into CPU memory with 16 I/O threads, but no more into GPU memory (7.9 GB/s).
  - s3 sets `MC_NUM_TCP_IO_THREADS=8`, `MC_TCP_LANES_PER_PEER=16` and `MC_TCP_SLICE_SIZE=4194304` on both ends,
    plus `VLLM_MOONCAKE_LOAD_RECV_THREADS=4` (`store.env`, `store.load_recv_threads`).
  - The rows measured with the defaults are kept in `results/s3-store-defaults.jsonl`, and those with 8 threads and
    16 lanes only in `results/s3-store-io8-lanes16.jsonl`.
  - iperf3 from A to C topped out at 5.9 GiB/s, below Mooncake's 12 GB/s, so iperf3 (one thread) was the limit
    there, not the NIC.
- **Capacity:** `grid.fits` checks store batches against the pool (`store.pool_gib`, 200 GiB of the 256 GiB VM;
  `store.holder_pool_gib`, 96 GiB, for store-p2p). The GPU, not the pool, leaves out Qwen3-8B 128K × 4. `sweep.sh`
  restarts the store, empty, before each model's store deployment.
  - The first run used a 16 vCPU, 64 GiB VM with a 48 GiB pool. The master's eviction (from 90% full) and its
    default 10 s lease dropped parts of large batches before they were read, and some reads failed with HTTP 500
    when the TCP lanes' queues filled. The fix: the 64 vCPU, 256 GiB VM, `lease_ttl_ms: 60000`, and larger
    `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER` / `MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER`. Those rows are in
    `results/s3-store-16vcpu.jsonl`.
  - Three store-p2p batches (32K × 8, 64K × 4, 128K × 2) lost part of their prefixes to the 96 GiB holder pool on
    the first attempt, and passed on retry.
- **Cost:** `ram_price: ram_cpu_vm` ($6.02 per GiB-month = $0.00825 per GiB-hour, `configs/prices.yaml`) prices the
  RAM behind p2p and the store (`ram_priced_tiers`). Those rows also carry `storage_usd_per_1m_included`,
  `total_usd_per_1m_included` and `saving_vs_recompute_included`, the s1/s2 assumption.
- **Scripts:**
  - `scripts/vms.sh` creates the VMs from the laptop and stages weights without GPU time. A cheap CPU VM downloads
    the models once onto a 128 GiB shared filesystem and is then deleted. Every GPU VM mounts that filesystem at
    `/models` (virtiofs).
    - **Why a shared filesystem:** a network disk attaches to one running VM at a time, so it would need one copy per
      VM.
    - **Speed:** s2 measured 1.77 GiB/s single-stream reads on a 1 TiB filesystem, against 440 MiB/s for a 372 GiB
      disk. That a 128 GiB one reads as fast is unverified.
    - **Compile caches** go to `/models/cache/<private IP>/`, one directory per VM, because A and B start the same
      model at once and Nebius gives no full POSIX guarantees for a file open on two VMs.
  - `scripts/store-up.sh` installs and starts `mooncake_master` and `mooncake_client` on VM C. It installs the
    Mooncake version found in the vLLM image, and stops processes by their pid files.
  - `sweep.sh` takes `STORE`. Without obj deployments it no longer needs `~/kvtb-s3.env`, and then skips the
    bucket backup.
- **Tests:** `tests/test_store.py` covers the grid, the config, the launch, the validity rule against a fake
  Mooncake engine, and the RAM-priced cost columns.

**Runbook (s3).** On the laptop: set `PROJECT_ID`, `SUBNET_ID` and `REGION`, and choose the region first, because
the filesystem is regional.
```bash
STAGE=s3 scripts/vms.sh weights           # 128 GiB filesystem kvtb-models, filled by kvtb-stager, which is then deleted
scripts/vms.sh gpu kvtb-vm-a               # spot, /models mounted; GPU_PLATFORM follows REGION
scripts/vms.sh gpu kvtb-vm-b
scripts/vms.sh cpu kvtb-vm-c 64vcpu-256gb  # on demand: no spot price is published for CPU VMs
```
On VM A, from the repo:
```bash
DEPLOYS=store STORE=kvtb@<C> ./scripts/sweep.sh s3 qwen3-8b qwen3.5-9b     # A and C only
DEPLOYS="p2p store-p2p" PEER=kvtb@<B> ./scripts/sweep.sh s3 qwen3-8b       # then VM B
DEPLOYS=p2p PEER=kvtb@<B> ./scripts/sweep.sh s3 qwen3.5-9b                  # store-p2p ran on Qwen3-8B only
# baselines: baseline_from: s2 reuses s2's rows, so `base` is not run
```
Afterwards, stop or delete the VMs and the `kvtb-models` filesystem, which costs about $0.014/h.

**Day-1 checks for s3:**
- [x] The vLLM v0.34.0 image has Mooncake's Python package (`store-up.sh` stops if not). If it does not, build a
      derived image with `mooncake-transfer-engine` installed and use the same version on VM C.
- [x] That version's wheel ships `mooncake_master` and `mooncake_client`. `--host=<C's private IP>` makes the
      client advertise a routable segment.
- [x] The label of a successful operation is `status="ok"`, not `"success"`, which cost one restart. Store hits
      are labelled `external_unspecified`. The image ships `mooncake-transfer-engine-cuda13` 0.3.14. The CPU VM
      needs the `-non-cuda` build of the same version, because the plain build links libcudart and libcuda.
- [x] `local_buffer_size` does not act as a local cache in front of the store: every valid read showed `load_get`
      bytes.
- [x] Eviction within one model's sweep: the pool fills by about 0.6 GiB per 4K prefix and 18 GiB per 128K prefix.
      48 GiB was too small (see Capacity); 200 GiB with a 60 s lease held.
- [x] A→C on two CPU VM sizes: Mooncake's own benchmark instead of iperf3 (one thread, 5.9 GiB/s). The bigger VM
      was faster into CPU memory only.
- [ ] A store read on Qwen3-8B: cached tokens = prefix, the `load_get` bytes cover it, greedy output equals recompute.
- [x] The same on Qwen3.5-9B with the hybrid conditions above: 72 of 72 valid.
- [ ] The MultiConnector question: can the CPU tier sit in front of the store? Not tried; the store was measured
      alone, and the report says so.
- [ ] p2p's us-central1 limit diagnosed with `scripts/p2p_bench.py`. Not done; s3 ran in eu-south1, where p2p was
      1.2–1.6× faster on the same models.

**Time and cost estimate** (written before the run; the run kept B off except for p2p and store-p2p):
- **Day 1:** about 3 h on A, 1.5 h on B, 3 h on C.
- **The sweep:** about 2 h on A, 1.5 h on B, 1 h on C.
  - B stays up for the baselines too, since they run on the p2p deployment; it can stop once `sweep.sh` reaches the
    store.
  - C is needed only for the store deployments, which run after p2p.
  - Store measurements should take about as long as s2's p2p (about 10 min of measuring per model) plus vLLM starts.
- **Weights:** about 20 min on the staging CPU VM ($0.26/h) instead of GPU time; the shared filesystem costs about
  $0.014/h until deleted.
- **Cost:** at today's spot prices for A and B ($1.69 in us-central1, $1.38 in eu-south1, 2026-11-19) and $0.53/h
  for C, roughly $12-20.

**Not in s3:** RDMA (no InfiniBand on these VMs), a store sharded across several VMs, the SSD tier as its own study,
and LMCache beyond a day-1 attempt.

## Schedule (2 weeks)

| Days | Work |
|---|---|
| 1–2 | Size model, grid, result store, cost maths and the driver, all tested without a GPU. |
| 3 | Nebius setup in eu-south1: quotas, two VMs, the disk, the bucket. Run the day-1 checks. |
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
- Mixed traffic during loads (decodes running alongside, trace replay). s2 measures fixed-size batches only.
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
