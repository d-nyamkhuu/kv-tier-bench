// Tier performance model. DOM-free, so `node --test` covers it.
//
// Everything here is a formula with published or calibrated inputs; the kv-tier-bench s1 measurements are only the
// reference the checks compare against (checks.js). TTFT of a request with n prompt tokens, h of them cached:
//   TTFT = overhead(n) + sum of cache loads + compute(n) - compute(h)
//   overhead(n) = request_overhead_s + per_token_overhead_s x n            (vLLM's own work: scheduling, hashing, ...)
//   compute(n)  = [2 x params x n + 2 x full-attention layers x width x n^2 / attn_eff_ratio] / (peak x prefill_mfu x TP)
// so the uncached tail costs compute(n) - compute(h): attention over the cached prefix still runs. A load from the
// CPU tier is cpu_load_latency_s + bytes / host link; a load from a lower tier is staged through CPU memory, so it
// adds that tier's latency + bytes / its bandwidth.

export const GIB = 1024 ** 3;

// A model's architecture, as layer groups, into the numbers the timing model uses. One function covers dense and MoE
// models, grouped-query (GQA) and latent (MLA) attention, sliding-window and sparse (top-k) attention, and linear
// attention. The spec:
//   { total_params, active_params, weights_bytes, hidden_size, block_size, compute_dtype ("bf16" | "fp8"), kv_dtype_bytes (2),
//     layers: [ { type: "full", count, q_heads, kv_heads, k_dim, v_dim, k_eq_v? }       GQA / MHA: every token
//               { type: "mla", count, q_heads, latent, rope, qk_dim, v_dim }          latent attention: every token
//               { type: "swa", count, q_heads, kv_heads, k_dim, v_dim, window, k_eq_v? }   sliding window
//               { type: "linear", count, state_bytes }                                fixed state per layer
//               { type: "given", kv_bytes_per_token }                                 a published per-token total
//             ] }
//   full and mla groups may add topk (sparse attention over that many tokens), index_heads, index_dim (the sparse
//   indexer's scores, which cost n^2 in keys only) and index_bytes (its key cache per token per layer); a group with
//   stores: false keeps no KV of its own (it reads another layer's), though its indexer keys still count.
// What it returns, and the choices behind it:
//  - KV per token counts the layers that keep every token. GQA heads are split over TP and copied when TP > KV heads.
//    MLA's latent is one head, so every TP rank holds all of it (vLLM: "the KV cache for each GPU will be duplicated
//    for tp_size / H times"): bytes per token x TP.
//  - The fixed part of a cached prefix: linear-attention state (split over TP by vLLM, so the total is constant) plus
//    the last `window` tokens of sliding-window layers. Neither grows with the prefix.
//  - Compute: 2 x active params per token, causal attention n^2 in full layers, n x min(n, window or top-k) in
//    sliding-window and sparse layers.
export function deriveArch(spec) {
  const B = spec.kv_dtype_bytes ?? 2;
  const tps = [1, 2, 4, 8];
  const perTok = Object.fromEntries(tps.map((tp) => [tp, 0]));
  let fixed = 0, windowBytes = 0, quad = 0, quadLayers = 0, linearLayers = 0;
  const windows = [];
  const attend = (g, width) => {
    if (g.topk) windows.push({ width: g.count * width, tokens: g.topk });
    else { quad += g.count * width; quadLayers += g.count; }
    if (g.index_heads) quad += (g.count * g.index_heads * g.index_dim) / 2; // keys only: half a layer's pairs
  };
  for (const g of spec.layers) {
    const kvDims = (x) => x.k_dim + (x.k_eq_v ? 0 : x.v_dim);
    if (g.type === "full") {
      const own = g.stores === false ? 0 : 1;
      for (const tp of tps) perTok[tp] += g.count * (own * Math.max(1, Math.floor(g.kv_heads / tp)) * tp * kvDims(g) * B + (g.index_bytes ?? 0) * tp);
      attend(g, (g.q_heads * (g.k_dim + g.v_dim)) / 2);
    } else if (g.type === "mla") {
      const own = g.stores === false ? 0 : 1;
      for (const tp of tps) perTok[tp] += g.count * (own * (g.latent + g.rope) * B + (g.index_bytes ?? 0)) * tp;
      attend(g, (g.q_heads * (g.qk_dim + g.v_dim)) / 2);
    } else if (g.type === "swa") {
      const w = g.count * g.kv_heads * kvDims(g) * B * g.window;
      fixed += w; windowBytes += w;
      windows.push({ width: (g.count * g.q_heads * (g.k_dim + g.v_dim)) / 2, tokens: g.window });
    } else if (g.type === "linear") {
      fixed += g.count * g.state_bytes;
      linearLayers += g.count;
    } else if (g.type === "given") {
      for (const tp of tps) perTok[tp] += g.kv_bytes_per_token * (g.copies_per_tp ? tp : 1);
    } else throw new Error(`unknown layer type ${g.type}`);
  }
  const fullLayers = quadLayers || (quad > 0 ? 1 : 0);
  return {
    hybrid: linearLayers > 0,
    params: spec.active_params ?? spec.total_params,
    total_params: spec.total_params,
    weights_bytes: spec.weights_bytes,
    full_attn_layers: fullLayers,
    attn_width: fullLayers ? quad / fullLayers : 0,
    attn_windows: windows,
    kv_bytes_per_token: perTok[1],
    kv_bytes_per_token_by_tp: perTok,
    state_bytes_per_prefix: fixed,
    read_fixed_bytes: fixed,
    state_window_bytes: windowBytes, // the sliding-window KV in the fixed part (KV precision applies to it)
    kv_dtype_bytes: B,
    block_size: spec.block_size,
    compute_dtype: spec.compute_dtype ?? "bf16",
    hidden_size: spec.hidden_size ?? null,
    num_layers: spec.num_layers ?? spec.layers.reduce((a, g) => a + (g.count ?? 0), 0),
  };
}

// One Gated DeltaNet layer's state (recurrent + short convolution), as Qwen3.8's config.json describes it; the same
// shape fits Kimi's KDA layers. bf16 by default.
export const gdnStateBytes = ({ k_heads, v_heads, k_dim, v_dim, conv, bytes = 2 }) =>
  v_heads * k_dim * v_dim * bytes + (2 * k_heads * k_dim + v_heads * v_dim) * (conv - 1) * bytes;

// Log-log linear interpolation over points [[x, y], ...] sorted by x; beyond either end, the end segment's slope.
// Used for the s1 reference curves in the checks.
export function interpLogLog(points, x) {
  const pts = points.filter(([px, py]) => px > 0 && py > 0);
  if (!pts.length) return NaN;
  if (pts.length === 1) return pts[0][1];
  if (x <= 0) return 0;
  let i = 1;
  while (i < pts.length - 1 && x > pts[i][0]) i++;
  const [x0, y0] = pts[i - 1];
  const [x1, y1] = pts[i];
  const t = (Math.log(x) - Math.log(x0)) / (Math.log(x1) - Math.log(x0));
  return Math.exp(Math.log(y0) + t * (Math.log(y1) - Math.log(y0)));
}

export const curve = (rows) => rows.map((r) => [r.tokens, r.ttft_s]).sort((a, b) => a[0] - b[0]);

// Bytes a prefix of `tokens` occupies in a tier: KV for every token plus a fixed part (the hybrid model's
// linear-attention state snapshot).
// `state` false: the KV blocks only (a hit spread over tiers reads the state from the tier holding its last block).
export const prefixBytes = (m, tokens, state = true) => (tokens > 0 ? (state ? m.read_fixed_bytes : 0) + m.kv_bytes_per_token * tokens : 0);

// Weighted least squares of y = L + c*x with weights 1/y^2 (relative error), L >= 0. Returns {L, c}.
export function fitLinearRel(xs, ys) {
  let a = 0, b = 0, d = 0, e = 0, f = 0; // sums of w, w*x, w*x^2, w*y, w*x*y
  xs.forEach((x, i) => {
    const w = 1 / (ys[i] * ys[i]);
    a += w; b += w * x; d += w * x * x; e += w * ys[i]; f += w * x * ys[i];
  });
  const det = a * d - b * b;
  let L = (e * d - b * f) / det;
  let c = (a * f - b * e) / det;
  if (!(L >= 0) || !(c > 0)) { // through the origin: minimize sum (c*x/y - 1)^2
    L = 0;
    let s1 = 0, s2 = 0;
    xs.forEach((x, i) => { s1 += x / ys[i]; s2 += (x / ys[i]) ** 2; });
    c = s1 / s2;
  }
  return { L, c };
}

// What s1 measured for one tier, as bandwidth and latency: TTFT_t - TTFT_cpu = L + bytes/BW, fitted against s1's cpu
// tier. The checks compare these with the variants' published figures.
export function fitTier(m, tier) {
  const rows = m.tiers[tier] || [];
  const cpu = curve(m.tiers.cpu);
  const pts = rows
    .map((r) => ({ tokens: r.tokens, ttft: r.ttft_s, extra: r.ttft_s - interpLogLog(cpu, r.tokens), bytes: prefixBytes(m, r.tokens) }))
    .filter((p) => p.extra > 0);
  if (pts.length < 2) return null;
  const { L, c } = fitLinearRel(pts.map((p) => p.bytes), pts.map((p) => p.extra));
  const bw = 1 / c / GIB;
  const residuals = rows.map((r) => {
    const model = interpLogLog(cpu, r.tokens) + L + prefixBytes(m, r.tokens) / (bw * GIB);
    return { tokens: r.tokens, measured: r.ttft_s, model, rel: model / r.ttft_s - 1 };
  });
  return { bw_gib_s: bw, latency_s: L, residuals, maxAbsRel: Math.max(...residuals.map((r) => Math.abs(r.rel))) };
}

// Weighted least squares (weights 1/y^2) of y = X b for any number of columns.
function wls(X, y) {
  const k = X[0].length;
  const A = Array.from({ length: k }, () => new Array(k).fill(0));
  const B = new Array(k).fill(0);
  X.forEach((row, i) => {
    const w = 1 / y[i] ** 2;
    for (let a = 0; a < k; a++) { B[a] += w * row[a] * y[i]; for (let b = 0; b < k; b++) A[a][b] += w * row[a] * row[b]; }
  });
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
    const f = A[j][i] / A[i][i];
    for (let l = i; l < k; l++) A[j][l] -= f * A[i][l];
    B[j] -= f * B[i];
  }
  const x = new Array(k).fill(0);
  for (let i = k - 1; i >= 0; i--) { let sum = B[i]; for (let l = i + 1; l < k; l++) sum -= A[i][l] * x[l]; x[i] = sum / A[i][i]; }
  return x;
}

// The vLLM software constants no one publishes, fitted on s1 (both models, every length): the per-request and
// per-token overhead (GPU hits), the prefill efficiency of s1's GPU and attention's efficiency relative to it (cold
// prefill), and the fixed latency of a CPU-tier load (CPU hits, with the published PCIe rate). presets.js holds the
// result; a test keeps the two in step.
export function calibrate(s1, peakTflops, hostGiBs) {
  const models = Object.values(s1.models);
  const [t0, c] = wls(models.flatMap((m) => m.tiers.gpu.map((r) => [1, r.tokens])), models.flatMap((m) => m.tiers.gpu.map((r) => r.ttft_s)));
  const P = peakTflops * 1e12;
  const rows = models.flatMap((m) => m.recompute.map((r) => ({ m, r })));
  const [ie, iea] = wls(rows.map(({ m, r }) => [2 * m.params * r.tokens / P, 2 * m.full_attn_layers * m.attn_width * r.tokens ** 2 / P]), rows.map(({ r }) => r.ttft_s - t0 - c * r.tokens));
  let sw = 0, swx = 0;
  for (const m of models) for (const r of m.tiers.cpu) {
    const g = m.tiers.gpu.find((x) => x.tokens === r.tokens);
    if (!g) continue;
    const load = r.ttft_s - g.ttft_s;
    const w = 1 / load ** 2;
    sw += w;
    swx += w * (load - prefixBytes(m, r.tokens) / (hostGiBs * GIB));
  }
  return { request_overhead_s: t0, per_token_overhead_s: c, prefill_mfu: 1 / ie, attn_eff_ratio: ie / iea, cpu_load_latency_s: swx / sw };
}

// Effective bandwidth of a variant: a network variant is link x efficiency, capped by a software limit if set; any
// other is its peak x efficiency (what a KV load through vLLM achieves of the device's peak).
export function variantBandwidth(v) {
  if (v.link_gbit_s != null) {
    const link = (v.link_gbit_s * 1e9 * (v.efficiency ?? 1)) / 8 / GIB;
    return v.cap_gib_s ? Math.min(link, v.cap_gib_s) : link;
  }
  return v.bw_gib_s * (v.efficiency ?? 1);
}

// Tensor parallelism inside a node (over NVLink, or PCIe without it): speed-up vs TP1, divided by TP. From vLLM runs of
// Llama 3.3 70B FP8, 8k in / 1k out, concurrency 4, median TTFT (prefill) and TPOT (decode), mean of H200 and B200
// (SemiAnalysis InferenceX, run 2025-10-29). TTFT includes fixed overhead, so these understate pure compute scaling.
export const TP_EFF = {
  prefill: { 1: 1, 2: 0.865, 4: 0.631, 8: 0.47 },
  decode: { 1: 1, 2: 0.793, 4: 0.554, 8: 0.347 },
};
const tpEff = (table, tp) => table[tp] ?? table[8];

// Decode under TP, as a cost per step rather than an efficiency: each layer's all-reduce adds a fixed time that does
// not grow with the batch. Fitted on the same InferenceX runs (Llama 3.3 70B, 80 layers, concurrency 4): with TP1
// paying none, TPOT_tp = TPOT_1 / tp + 80 x o_tp, so o_tp = TPOT_1 / tp x (1 / eff_tp - 1) / 80, mean of H200
// (TPOT_1 26.3 ms) and B200 (18.5 ms). An efficiency measured at batch 4 would wrongly scale a batch-256 step.
const IX_TPOT_TP1_S = [0.0263, 0.0185];
export const TP_SYNC_PER_LAYER_S = Object.fromEntries(Object.entries(TP_EFF.decode).map(([tp, e]) =>
  [tp, IX_TPOT_TP1_S.reduce((a, t) => a + (t / tp) * (1 / e - 1) / 80, 0) / IX_TPOT_TP1_S.length]));

// A model as one TP instance stores it. attention "tp" (vLLM's default): KV heads split over the GPUs, copied when TP >
// KV heads, and a latent (MLA) cache copied on every GPU. "dp" (data-parallel attention, how MLA models are served
// at scale: vLLM --data-parallel-size with --enable-expert-parallel): each GPU keeps whole KV for its own requests,
// so nothing is copied and bytes per token are TP1's.
export const withTp = (m, tp = 1, attention = "tp") =>
  ({ ...m, kv_bytes_per_token: attention === "dp" ? m.kv_bytes_per_token : m.kv_bytes_per_token_by_tp?.[tp] ?? m.kv_bytes_per_token });

// GPU KV cache of one instance (TP GPUs in each of `pp` nodes), as grid.gpu_kv_budget: memory x vLLM's
// --gpu-memory-utilization, minus the weights (split across all its GPUs) and activations per GPU. <= 0: it does not fit.
export function gpuKvGiB(m, gpu, sw, tp = 1, pp = 1) {
  return pp * tp * (gpu.mem_gib * sw.gpu_memory_utilization - sw.activation_gib_per_gpu) - m.weights_bytes / GIB;
}

const perSeqBytes = (m, ctxTokens) => m.state_bytes_per_prefix + m.kv_bytes_per_token * ctxTokens;

// One batched decode step on a TP instance of its own (disaggregated prefill/decode). A step reads the weights once
// (an MoE batch of more than a few dozen tokens touches nearly every expert), plus each sequence's KV and state, at
// `mbu` of peak memory bandwidth; a large batch can instead be bound by its FLOPs (2 x active params per token, at
// the prefill share of peak). Plus the TP all-reduces of every layer.
export function decodeStepS(m, gpu, batch, ctxTokens, tp = 1, eff = TP_EFF) {
  const memory = (m.weights_bytes + batch * perSeqBytes(m, ctxTokens)) / (tp * gpu.mem_bw_tb_s * 1e12 * gpu.mbu);
  const peak = peakTflops(m, gpu);
  const compute = peak && gpu.prefill_mfu ? (2 * m.params * batch) / (peak * 1e12 * gpu.prefill_mfu * tp * tpEff(eff.prefill, tp)) : 0;
  const sync = tp > 1 ? (m.num_layers ?? m.full_attn_layers ?? 0) * (TP_SYNC_PER_LAYER_S[tp] ?? TP_SYNC_PER_LAYER_S[8]) : 0;
  return Math.max(memory, compute) + sync;
}

// The largest decode batch whose KV fits the instance's GPU memory.
export function decodeBatchCap(m, gpu, sw, ctxTokens, tp = 1, pp = 1) {
  return Math.floor((gpuKvGiB(m, gpu, sw, tp, pp) * GIB) / perSeqBytes(m, ctxTokens));
}

// Prefill FLOPs of n tokens: 2 x params per token, plus causal attention in the full-attention layers.
// Query-key pairs of causal attention over a window of w tokens (w = Infinity: full attention), halved.
const windowPairs = (n, w) => (n <= w ? (n * n) / 2 : n * w - (w * w) / 2);
const attnFlops = (m, n) => 2 * m.full_attn_layers * m.attn_width * n * n + (m.attn_windows ?? []).reduce((a, g) => a + 4 * g.width * windowPairs(n, g.tokens), 0);
export const prefillFlops = (m, n) => 2 * m.params * n + attnFlops(m, n);

// Dense tensor-core peak for the precision the model computes in: FP8 when its released weights and activations are
// FP8, else BF16 (weight-only 4-bit models dequantize to BF16).
export const peakTflops = (m, gpu) =>
  (m.compute_dtype === "fp4" && gpu.tflops_fp4) || ((m.compute_dtype === "fp8" || m.compute_dtype === "fp4") && gpu.tflops_fp8) || gpu.tflops_bf16;

// Serving precision. "released": as the checkpoint ships. "fp8": FP8 weights (1 byte per parameter; block scales
// are negligible) and FP8 compute, with vLLM's --kv-cache-dtype fp8. "fp4": NVFP4 weights (4 bits + one FP8 scale per
// 16 values: 4.5 bits per parameter), FP4 compute where the GPU has FP4 tensor cores (Blackwell; else FP8), FP8 KV.
// Weights never grow past what was released; a KV cache already below 1 byte per value stays as it is; the
// linear-attention state stays bf16. Every weight is counted at that precision, embeddings included: an assumption.
export const PRECISIONS = { released: null, fp8: { bits: 8, compute: "fp8" }, fp4: { bits: 4.5, compute: "fp4" } };
export function withPrecision(m, precision = "released") {
  const p = PRECISIONS[precision];
  if (!p) return m;
  const weights = Math.min(m.weights_bytes, ((m.total_params ?? m.params) * p.bits) / 8);
  const kvScale = (m.kv_dtype_bytes ?? 2) > 1 ? 1 / (m.kv_dtype_bytes ?? 2) : 1; // to 1 byte per value
  const win = (m.state_window_bytes ?? 0) * (1 - kvScale);
  const order = { bf16: 0, fp8: 1, fp4: 2 };
  return {
    ...m, weights_bytes: weights,
    compute_dtype: order[p.compute] > order[m.compute_dtype ?? "bf16"] ? p.compute : m.compute_dtype ?? "bf16",
    kv_bytes_per_token: m.kv_bytes_per_token * kvScale,
    kv_bytes_per_token_by_tp: m.kv_bytes_per_token_by_tp && Object.fromEntries(Object.entries(m.kv_bytes_per_token_by_tp).map(([k, v]) => [k, v * kvScale])),
    state_bytes_per_prefix: m.state_bytes_per_prefix - win, read_fixed_bytes: m.read_fixed_bytes - win,
    state_window_bytes: (m.state_window_bytes ?? 0) * kvScale, kv_dtype_bytes: Math.min(m.kv_dtype_bytes ?? 2, 1),
  };
}

// The model as an instance serves it: precision, then TP and attention layout.
export const served = (m, { precision = "released", tp = 1, attention = "tp" } = {}) => withTp(withPrecision(m, precision), tp, attention);

// Pipeline parallelism across nodes (`pp` stages, TP inside each node, as vLLM deploys a model too big for one node):
// a request passes the stages one after another, so its prefill takes as long as on one TP group holding every layer,
// plus a hop per stage boundary: the link's latency + the hidden states (hidden_size x 2 bytes per token) / its
// bandwidth. Cache loads and KV transfers move each stage's layers on its own node at once: pp times the bandwidth.
export const stageHop = (m, link) => (link ? (n) => (link.latency_s ?? 0) + (n * (m.hidden_size ?? 0) * 2) / (variantBandwidth(link) * GIB) : () => 0);

// The per-model timing functions the simulator calls, for one instance of `gpu`: TP GPUs, in `pp.stages` nodes
// joined by `pp.link`. `sw`: the software constants.
export function timing(m, gpu, sw, tp = 1, eff = TP_EFF, { stages = 1, link = null } = {}) {
  const peak = peakTflops(m, gpu) * 1e12 * gpu.prefill_mfu * tp * tpEff(eff.prefill, tp);
  const hop = stageHop(m, link);
  const flops = (n) => (n > 0 ? (2 * m.params * n + attnFlops(m, n) / sw.attn_eff_ratio) / peak : 0);
  const hops = (n) => (n > 0 ? (stages - 1) * hop(n) : 0); // the hidden states of the n tokens computed
  const compute = (n) => flops(n) + hops(n);
  const overhead = (n) => sw.request_overhead_s + sw.per_token_overhead_s * n;
  const R = (n) => (n > 0 ? overhead(n) + compute(n) : 0); // cold prefill
  const host = gpu.host_gib_s * GIB * tp * stages; // one host link per GPU
  const cpuLoad = (k, state = true) => (k > 0 ? sw.cpu_load_latency_s + prefixBytes(m, k, state) / host : 0);
  const loadFor = (v) => {
    if (!v) return () => 0;
    const bw = variantBandwidth(v) * GIB * stages;
    const L = v.latency_s ?? 0;
    return (k, state = true) => (k > 0 ? cpuLoad(k, state) + L + prefixBytes(m, k, state) / bw : 0);
  };
  // prefill -> decode KV transfer over a network variant, GPU to GPU
  const transferFor = (v) => {
    if (!v) return () => 0;
    const bw = variantBandwidth(v) * GIB * stages;
    const L = v.latency_s ?? 0;
    return (k) => (k > 0 ? L + prefixBytes(m, k) / bw : 0);
  };
  const ttft = (n, h, loadSeconds) => overhead(n) + loadSeconds + flops(n) - flops(Math.min(h, n)) + hops(Math.max(1, n - h));
  return { compute, overhead, R, cpuLoad, loadFor, transferFor, ttft };
}
