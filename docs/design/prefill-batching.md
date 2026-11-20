# Prefill batching

The simulator page batches prefill by default, like vLLM's continuous batching with chunked prefill. Each prefill
instance runs steps. A step takes waiting requests first come, first served up to a token budget (vLLM's
`max_num_batched_tokens`, default 8,192 here), splits a prompt that does not fit across steps, and finishes the requests
it completes together at the end of the step.

## Step cost

A step costs one read of the model weights plus the compute of its tokens:

- **Weight read:** weights per GPU divided by memory bandwidth × `mbu`, capped at the calibrated per-request overhead
  (`software.request_overhead_s`, 52.4 ms). On the calibration setup, Qwen3.8-27B on one RTX PRO 6000, the read takes
  58 ms. That is close to the measured 52.4 ms, which is why the overhead is treated as mostly a per-step cost.
- **Compute:** the FLOPs of each request's new tokens in the step (attention included), plus the per-token overhead,
  plus pipeline-stage hops.
- **Per-request delay:** the rest of the 52.4 ms delays each request without occupying the GPU (tokenisation,
  scheduling, sampling).

A request alone in its step therefore has exactly the one-at-a-time TTFT the model was calibrated on. Batching changes
results only when requests overlap.

Cache lookups and loads run while other requests compute. LMCache's multi-process connector polls lookup status
without blocking the scheduler by default (`lmcache.mp.nonblocking_lookup_status: true`) and offloads and loads
asynchronously ([LMCache MP configuration](https://github.com/lmcache/lmcache/blob/dev/docs/source/mp/configuration.rst),
[architecture](https://github.com/lmcache/lmcache/blob/dev/docs/source/developer_guide/architecture.rst)). The
one-at-a-time model held the GPU during lookups and loads, including the 150 ms S3 metadata round trip, which made a
busy cluster look saturated much earlier.

## What published benchmarks show

### TTFT against load at fixed prompt lengths

[NVIDIA NIM 1.8.0 benchmarks](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html), Llama-3.3-70B-Instruct
FP8, TP2, closed loop at fixed concurrency (aggregated serving: prefill and decode share the GPUs). Mean TTFT in ms:

| Concurrency | 1 | 5 | 25 | 50 | 100 | 150 | 200 | 250 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2×H100, 1,000 in / 1,000 out | 82 | 146 | 250 | 279 | 334 | 429 | 558 | 3,500 |
| 2×H200, 1,000 in / 1,000 out | 81 | 147 | 253 | 279 | 332 | 424 | 582 | 667 |
| 2×H100, 5,000 in / 500 out | 407 | 547 | 689 | 835 | 7,996 | 34,794 | 61,680 | 88,605 |
| 2×H200, 5,000 in / 500 out | 419 | 568 | 708 | 841 | 5,701 | 31,108 | 56,557 | 82,016 |

TTFT roughly doubles from an idle server to about 90% of capacity, then jumps once prefill saturates: with
5,000-token prompts, output throughput flattens at about 920 tokens/s (1.83 requests/s) from concurrency 100 on.

### InferenceX: TTFT against concurrency

[InferenceX](https://inferencex.semianalysis.com/infx/inference/deepseek-v41-flash) reruns its sweeps continuously. The
numbers below come from the JSON its dashboard loads (`/api/v1/benchmarks?model=…`, read 2026-11-19). InferenceX varies
the number of concurrent clients, not the token budget. Its data shows how TTFT grows with load; it does not settle 8,192
against 16,384.

**DeepSeek V4.1 Flash, agentic traces.** One-hour replays of multi-turn coding sessions. Requests average 140k to 170k
input tokens, and the server finds 95–97% of them cached on the GPU. TTFT is in seconds.

| Setup | Concurrency | Requests/s | Median | p90 |
| --- | --- | --- | --- | --- |
| 8×B200, vLLM, TP8 EP8, aggregated ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/41894132869/attempts/1), 2026-11-05) | 224 / 256 / 384 / 512 / 576 | 9.7 / 11 / 16 / 20 / 22 | 1.26 / 1.35 / 1.65 / 2.18 / 2.45 | 2.23 / 2.39 / 2.90 / 4.00 / 4.45 |
| 8×H200, vLLM, TP8, aggregated ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42341901521/attempts/1), 2026-11-09) | 8 / 32 / 64 / 128 | 0.38 / 1.2 / 2.3 / 2.6 | 0.25 / 0.24 / 0.27 / 0.71 | 0.44 / 0.52 / 0.72 / 14.7 |
| B300, Dynamo + vLLM, disaggregated, 4–10 prefill GPUs at TP2 EP2 with 8 decode, CPU offload ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43769338888/attempts/2), 2026-11-17) | 256 / 768 / 1,280 / 2,048 / 3,584 | 12 / 34 / 55 / 83 / 110 | 0.22 / 0.63 / 0.75 / 0.92 / 8.22 | 0.63 / 1.59 / 1.99 / 2.74 / 13.2 |

- TTFT stays flat while the prefill GPUs have headroom. The H200 median holds at 0.24–0.27 s from 8 to 64 clients.
- It then rises steadily. On B200 the median doubles, and so does p90, as load goes from 9.7 to 22 requests/s.
- Past capacity it jumps: H200 at 128 clients, and B300 with 10 prefill GPUs at 3,584 clients.
- The B300 deployment grows its prefill pool from 4 to 10 GPUs as load rises, which holds the median under 1 s up to
  83 requests/s.

**DeepSeek-R1-0528 NVFP4, 8k/1k, on 4×B200.** Each point is a closed loop: C clients, 10 × C prompts, request rate
unlimited, inputs uniform between 0.8 × and 1 × 8,192 tokens (`inferencex-e2e/infx/bench/fixed_seq.py`). TTFT is in ms.

| Engine | Concurrency | Median | Mean | p99 |
| --- | --- | --- | --- | --- |
| TensorRT-LLM 1.3.0rc14, aggregated ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26016880548/attempts/2)) | 4 / 8 / 16 / 32 | 215 / 217 / 223 / 239 | 251 / 303 / 389 / 560 | 729 / 1,417 / 2,733 / 5,331 |
| SGLang 0.5.19, aggregated ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34557177019/attempts/1)) | 1 / 4 / 16 / 32 | 208 / 247 / 491 / 764 | 208 / 288 / 650 / 1,035 | 223 / 739 / 2,933 / 5,685 |
| Dynamo + SGLang 0.5.8, 4 prefill + 8 decode GPUs ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/21888944497/attempts/1)) | 4 / 8 / 16 / 64 | 265 / 327 / 443 / 727 | 330 / 472 / 696 / 1,340 | 799 / 1,436 / 2,756 / 10,220 |
| Dynamo + TensorRT-LLM, 4 prefill + 40 decode GPUs, DP attention ([run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/21484975323/attempts/1)) | 10 / 25 / 50 / 100 | 576 / 822 / 1,017 / 2,851 | 695 / 946 / 1,274 / 3,147 | 1,832 / 3,367 / 5,622 / 10,927 |

- The p99 is roughly C × 170 ms because every client starts at once. The last request of the first wave queues behind
  the other C − 1 prompts, so the p99 measures that start-up wave rather than batching.
- TensorRT-LLM's median barely moves from 4 to 32 clients (215 to 239 ms). SGLang's triples (208 to 764 ms), with
  decode sharing the same GPUs: its time per output token rises from 5 to 19 ms over the same range.

### Token budget and batch caps

| Source | Setup | Setting varied | TTFT |
| --- | --- | --- | --- |
| [TensorRT-LLM tuning guide](https://nvidia.github.io/TensorRT-LLM/1.3.0rc21/legacy/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html) | Llama-3.3-70B, 4×H100 | max_num_tokens 2,048 / 8,192 / 16,384 | 147.6 / 147.8 / 148.0 ms |
| same | same | max_batch_size 64 / 512 / 2,048 (throughput 1,944 / 2,467 / 2,044 tok/s) | 145.8 / 147.8 / 146.7 ms |
| [GLM-5 serving tuning](https://arxiv.org/html/2607.02518v1) | GLM-5 FP8, SGLang, 2×8 H100, ~30k-token prompts, 16 running requests | chunked-prefill-size 2,048 / 3,072 / 4,096 / 6,144 | mean 8.98 / 8.04 / 9.73 / 8.85 s; p90 14.18 / 12.58 / 17.39 / 15.00 s |
| same | chunk 3,072 | max running requests 24 / 32 / 48 | mean 6.69 / 7.01 / 7.39 s |

The budget barely moves TTFT when the GPU has headroom: it caps a step but does not fill it. What raises a request's
latency is sharing a step with other work, and small chunks add per-step costs to long prompts.

### Budgets tuned deployments use

[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) publishes the recipes behind its continuous benchmarks
(`inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes`). Prefill-side settings:

| Recipe | Prefill budget |
| --- | --- |
| DeepSeek-R1 FP4, SGLang disaggregated prefill, B200 TP4, 8k/1k | chunked-prefill-size 24,576, max-running-requests 512 |
| DeepSeek-R1 FP8, SGLang disaggregated prefill, B200 TP8 DP8, 8k/1k | max-prefill-tokens 8,192, chunked-prefill-size 65,536 |
| DeepSeek-R1 FP8, TensorRT-LLM disaggregated prefill, B200, 8k/1k | max_num_tokens 8,320 |
| DeepSeek V4.1 Flash FP4, vLLM disaggregated prefill, B300, AgentX | max-num-batched-tokens 16,384, max-num-seqs 64, long-prefill-token-threshold 8,192, CPU KV offload |
| DeepSeek V4.1 Flash FP4, vLLM aggregated, GB200, AgentX | max-num-batched-tokens 8,192 or 4,096 |

So 8,192 is a conservative default; DeepSeek prefill deployments run 8k to 24k. vLLM's
`long-prefill-token-threshold` also caps how much of one long prompt a step takes, so short requests are not stuck
behind it; the simulator's first-come, first-served chunking does not do that.

### Cache reuse and offload under load

- [LMCache](https://blog.lmcache.ai/en/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/):
  Qwen3-235B-A22B FP8 on 8×H100, vLLM 0.18.1, multi-round chat at 2 QPS, `max-num-batched-tokens 1024`. Mean TTFT
  0.29 s with the multi-process connector against 3.98 s with in-process offload (p99 1.30 s against 13.55 s).
- [Red Hat llm-d](https://developers.redhat.com/articles/2026/01/13/accelerate-multi-turn-workloads-llm-d): 4 vLLM
  replicas on 4×L4, 11 conversations, 110 requests. Prefix-aware routing cut mean TTFT from 209 ms to 113 ms
  (p95 727 ms to 238 ms) against round-robin, with the KV hit rate rising from about 62% to 90%.
- MLPerf Inference holds Llama-2-70B to p99 TTFT of 2 s in the server scenario and 450 ms in the interactive one
  ([NVIDIA on MLPerf v5.0](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0)).

## Check against NIM

`web/test/calibration.test.js` replays the NIM Llama-3.3-70B setup on 2×H100: prefill only, an 8,192-token budget,
no cache, Poisson arrivals at the request rate NIM measured at each concurrency (output throughput / 500 tokens).

| Prompt | Requests/s (NIM concurrency) | Simulated mean TTFT | NIM mean TTFT |
| --- | --- | --- | --- |
| 1,000 tokens | 0.05 (1) | 124 ms | 82 ms |
| 5,000 tokens | 0.10 (1) | 432 ms | 407 ms |
| 5,000 tokens | 0.44 (5) | 481 ms | 547 ms |
| 5,000 tokens | 1.25 (25) | 645 ms | 689 ms |
| 5,000 tokens | 1.63 (50) | 864 ms | 835 ms |
| 5,000 tokens | 1.83 (100) | 987 ms | 7,996 ms |
| 5,000 tokens | 2.5 | 3.8 s batched, 43.9 s one at a time | (beyond NIM's capacity) |

- Long prompts track within 12% up to about 90% of NIM's capacity, and the test holds them within 15%.
- Short prompts come out about 40 ms slow: the 52.4 ms per-request overhead was calibrated on vLLM with LMCache, and
  NIM's TensorRT-LLM engine has less.
- NIM saturates near 1.8 requests/s and the simulator near 2.5, because NIM decodes on the same GPUs. The simulator
  gives decode GPUs of their own, so its prefill capacity is higher; it also misses the decode interference that
  raises NIM's TTFT with 1,000-token prompts (146 and 250 ms at concurrency 5 and 25, against 131 and 146 ms).
- One at a time, the same cluster saturates 10× worse at 2.5 requests/s (43.9 s against 3.8 s): without batching
  every request pays the full fixed cost.

## Check against InferenceX

The R1 runs above were replayed through the simulator as InferenceX's closed loop:
- Each client sends its next prompt when its last reply ends.
- The gap between prompts is the measured decode time, spread like the output lengths.
- R1 is described from its `config.json` (61 MLA layers, 37B active parameters, 423.6 GB of NVFP4 weights) on one
  4×B200 TP4 instance.

The last row of the table below replays the disaggregated SGLang run with the simulator's own decode pool: 1,024 output
tokens, with KV sent over InfiniBand NDR. Median / mean in ms:

| Measured run | Concurrency | Simulated, 8,192 budget | Simulated, 16,384 | Simulated, one at a time | Measured |
| --- | --- | --- | --- | --- | --- |
| TensorRT-LLM aggregated | 4 | 201 / 232 | 200 / 232 | 200 / 232 | 215 / 251 |
| | 32 | 239 / 549 | 243 / 548 | 304 / 624 | 239 / 560 |
| SGLang aggregated | 1 | 192 / 192 | 192 / 192 | 192 / 192 | 208 / 208 |
| | 32 | 241 / 542 | 260 / 557 | 298 / 624 | 764 / 1,035 |
| Dynamo + TensorRT-LLM | 50 | 607 / 979 | 527 / 893 | 1,972 / 2,264 | 1,017 / 1,274 |
| | 100 | saturated (7.4 s) | saturated (6.4 s) | saturated (10.7 s) | 2,851 / 3,147 |
| Dynamo + SGLang, simulated decode | 4 / 64 | 299 / 326, 352 / 920 | 299 / 340, 384 / 930 | 299 / 326, 414 / 1,031 | 265 / 330, 727 / 1,340 |

- **TensorRT-LLM aggregated** stays within 8% on the median, mean and p99 at every concurrency.
  `web/test/calibration.test.js` holds the median and mean within 15%. That test is a regression pin rather than an
  independent check:
  - The prefill MFU was already fitted to InferenceX's Llama 3.3 70B numbers.
  - R1 is computed at the FP8 rate because the FP4 peak does not fit (see Limits).
- **SGLang aggregated** matches when idle and falls behind as concurrency rises. Its decode steps share the GPUs, which
  the simulator does not model.
- **Disaggregated SGLang** matches up to 8 clients once the simulated decode handoff is included (a 59 ms KV transfer
  plus the first decode step). Past that, the measured median climbs to 727 ms at 64 clients against 352 ms simulated.
  The simulator does not explain this. A likely cause is the decode side's admission queue, which has to reserve KV
  memory before the transfer starts, but this is not verified.
- **Disaggregated TensorRT-LLM** sustains 8.2 requests/s on 4 prefill GPUs, where the FP8-rate simulation saturates
  near 5.5. Its prefill runs faster than the FP8 rate and slower than the FP4 peak.
- **Budget size** matters little once arrivals are spread out. 8,192 and 16,384 stay within about 10% of each other on
  the median, and both beat one at a time under load. In a first replay where all clients stayed in step, 16,384
  raised the median from 198 to 342 ms. Two 8k prompts then shared every step, and each waited for the whole step.
  That is the cost the budget can impose when prompts arrive together.

## What the model reproduces

From `web/test/sim.test.js` and replays of the page's scenarios (cold start, one hour):

- **Compute-bound prompts:** two 1,000-token prompts arriving together finish at 1.01 s and 2.02 s one at a time
  (mean 1.515 s), and both at 2.01 s in one step. Batching raises the mean by a third because each waits for the whole
  step.
- **Small budgets:** a 2,048-token budget raised TTFT 15% for Qwen3.8-27B without a cache (1.47 s to 1.68 s at
  300 req/min) and 44% for gpt-oss-120b on H100, because long prompts pay a weight read per chunk. At 8,192 the
  penalty is about 2%; at 32,768 it is gone.
- **Short, cached prompts:** batching amortises the weight read. DeepSeek V4.1 Flash on 2×8 B200 at 1,200 req/min
  keeps a 314 ms mean TTFT batched, while the one-at-a-time model saturates (34.7 s) because it holds the GPU through
  each S3 lookup.
- **Near compute saturation:** Qwen3.8-27B without a cache at 900 req/min: 17.6 s one at a time, 19.0 s at 8,192,
  14.5 s at 32,768.

## Limits

- A step's weight read and compute add, as in the calibrated one-at-a-time fit. A real step overlaps them, so long
  steps are slightly pessimistic.
- First come, first served only: no priority for short prompts and no preemption.
- No limit on the number of requests in a step besides the token budget.
- Prefill and decode run on separate GPUs. On shared GPUs (aggregated serving, as NIM benchmarks), decode steps
  take GPU time from prefill and TTFT saturates earlier.
- The 52.4 ms per-request overhead comes from vLLM with LMCache; other engines differ (NIM: about 40 ms less on a
  1,000-token prompt).
- FP4 models prefill at the FP8 rate (`model.js`, `prefillTflops`); decode still uses the FP4 peak. The FP4 peak
  (4 × BF16 at the same MFU) predicted 130 ms for an idle R1 NVFP4 8k prompt on 4 B200s. InferenceX measures
  208–215 ms, and the FP8 rate gives 192–201 ms. One idle point cannot tell a wrong FP4 rate from a different MFU or
  overhead, so this is a fit, not a measured FP4 rate. TensorRT-LLM's disaggregated prefill outruns the FP8 rate
  (8.2 requests/s where the simulation saturates near 5.5), so some engines get part of the FP4 speed-up.
