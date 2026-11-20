// The model against measurements of real setups: published vLLM benchmarks, and kv-tier-bench s1 (RTX PRO 6000,
// Nebius, vLLM v0.34.0) as the reference a real deployment should measure up to. DOM-free: the page shows the
// tables, a test keeps the gaps from drifting. "calibrated" rows set a model parameter, so they agree by construction.

import { decodeStep, decodeStepS, deriveArch, fitTier, GIB, prefixBytes, served, timing, variantBandwidth } from "./model.js";

// Llama 3.3 70B in FP8 (1 byte per weight), bf16 KV: 80 layers x 8 KV heads x 128 x K,V x 2 bytes
const LLAMA70B = { num_layers: 80, params: 70553706496, weights_bytes: 70553706496, kv_bytes_per_token: 80 * 8 * 128 * 2 * 2, state_bytes_per_prefix: 0, read_fixed_bytes: 0, full_attn_layers: 80, attn_width: 64 * 128 };
const FP8_TFLOPS = { h100: 1979, h200: 1979, b200: 4500 }; // dense; NVIDIA datasheets quote 2x "with sparsity"
const IX = "https://inferencex.semianalysis.com/compare/llama-3-3-70b-b200-vs-h100";
const IX_LABEL = "InferenceX: vLLM, Llama 3.3 70B FP8, 8k in / 1k out, concurrency 4, median (run 2025-10-29)";
// Models of the decode checks below, from their config.json; parameter counts from the cards and papers
const mla = (count, q_heads) => ({ type: "mla", count, q_heads, latent: 512, rope: 64, qk_dim: 192, v_dim: 128 });
const KIMI_K25 = { ...deriveArch({ num_layers: 61, hidden_size: 7168, total_params: 1e12, active_params: 32e9, weights_bytes: 2e12, block_size: 64,
  experts: { routed: 384, per_token: 8, params: 60 * 384 * 3 * 7168 * 2048 }, layers: [mla(61, 64)] }) };
const DEEPSEEK_V3 = { ...deriveArch({ num_layers: 61, hidden_size: 7168, total_params: 671e9, active_params: 37e9, weights_bytes: 671e9, compute_dtype: "fp8", block_size: 64,
  experts: { routed: 256, per_token: 8, params: 58 * 256 * 3 * 7168 * 2048 }, layers: [mla(61, 128)] }) };
const gqa = (count, q_heads) => ({ type: "full", count, q_heads, kv_heads: 8, k_dim: 128, v_dim: 128 });
const LLAMA8B = deriveArch({ num_layers: 32, hidden_size: 4096, total_params: 8.03e9, weights_bytes: 2 * 8.03e9, block_size: 16, layers: [gqa(32, 32)] });
const QWEN3_8B = deriveArch({ num_layers: 36, hidden_size: 4096, total_params: 8.19e9, weights_bytes: 2 * 8.19e9, block_size: 16, layers: [gqa(36, 32)] });
const IXK = { label: "InferenceX: Kimi K2.5 NVFP4, 8k in / 1k out, vLLM, no speculative decoding (unofficial copy of the page)", url: "https://inferencex.semianalysis.com/blog/gb200-nvl72-kimi-k2-5-vllm-wide-ep-3x-vs-b200" };
const DS_PROD = { label: "DeepSeek-V3/R1 inference system overview (24-hour production average)", url: "https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md" };
const DFLASH = { label: "DFlash paper, Table 3: SGLang, B200, Qwen3-8B, Math500", url: "https://arxiv.org/abs/2602.06036" };
const EAGLE3 = { label: "EAGLE-3 paper, Table 3: SGLang, H100, LLaMA-Instruct 3.1 8B, MT-Bench", url: "https://arxiv.org/abs/2503.01840" };
const fmt3 = (x) => Number(x.toPrecision(3)).toString();
const S1 = { label: "kv-tier-bench s1: one request at a time, median of 5+ runs", url: "https://github.com/d-nyamkhuu/kv-tier-bench/blob/main/results/s1.jsonl" };
// which configured variant stands for each tier s1 measured
export const S1_TIER_VARIANT = { fs: "fs-nebius-nrd", obj: "obj-nebius-enhanced", p2p: "p2p-tcp-400" };
const S1_TIER_NAME = { fs: "Nebius network SSD (non-replicated, 1767 GiB)", obj: "Nebius Object Storage, Enhanced Throughput", p2p: "a peer VM's CPU memory over TCP (400 Gbit/s NIC)" };

export function checks(s1, c) {
  const g = Object.fromEntries(c.gpus.map((x) => [x.id, x]));
  const sw = c.software;
  const rows = [];
  const add = (r) => rows.push({ ...r, gap: r.predicted / r.published - 1 });

  for (const [id, tp, ttft, tpot] of [["h200", 1, 1.166, 26.3], ["b200", 1, 0.654, 18.5], ["h100", 2, 0.694, 20.7], ["h100", 4, 0.406, 13.4]]) {
    const gpu = g[id];
    if (!gpu) continue;
    const T = timing(LLAMA70B, { ...gpu, tflops_bf16: FP8_TFLOPS[id] }, sw, tp);
    add({
      group: "published", what: `Prefill TTFT, ${gpu.name}, TP${tp}`, unit: "s", published: ttft, predicted: T.R(8192), calibrated: tp === 1,
      source: { label: IX_LABEL, url: IX }, note: tp === 1 ? `Sets prefill_mfu for ${id === "h200" ? "Hopper" : "Blackwell"} GPUs.` : "H100 was not used to calibrate; TP efficiency comes from H200 and B200.",
    });
    add({
      group: "published", what: `Decode time per token, ${gpu.name}, TP${tp}`, unit: "ms", published: tpot, predicted: decodeStepS(LLAMA70B, gpu, 4, 8192 + 512, tp) * 1e3,
      calibrated: false, source: { label: IX_LABEL, url: IX },
      note: `Memory-bound step at ${Math.round(gpu.mbu * 100)}% of peak bandwidth (Databricks), batch 4${tp > 1 ? "; plus TP all-reduce time per layer, fitted on H200/B200" : ""}. Assumes bf16 KV.`,
    });
  }

  // decode beyond one dense instance: an MoE model, expert parallelism and speculative decoding (no row sets a parameter)
  if (g.b200) {
    const ms = (m, gpu, batch, ctx, tp, opts) => decodeStep(m, gpu, batch, ctx, tp, opts).step * 1e3;
    add({ group: "published", what: "Decode time per token, B200, Qwen3-8B, one request", unit: "ms", published: 1000 / 230, predicted: ms(QWEN3_8B, g.b200, 1, 1024, 1), calibrated: false, source: DFLASH,
      note: "SGLang's baseline (230 tokens/s). The model has no fixed cost per step; the gap bounds what scheduling and kernel launches add." });
    const k25 = served(KIMI_K25, { precision: "fp4", tp: 4, attention: "tp" });
    for (const [conc, tps] of [[4, 101.8], [64, 29.3]]) add({
      group: "published", what: `Decode time per token, B200 TP4, Kimi K2.5 (MoE), batch ${conc}`, unit: "ms", published: 1000 / tps, predicted: ms(k25, g.b200, conc, 8192 + 512, 4), calibrated: false, source: IXK,
      note: "Prefill runs on the same GPUs there (not disaggregated), which the model leaves out; read as one 4-GPU instance, since tokens/s per GPU matches 4 GPUs at both concurrencies." });
  }
  if (g.gb200) {
    const k25 = served(KIMI_K25, { precision: "fp4", tp: 1, attention: "dp" });
    for (const [conc, tps] of [[512, 48.8], [4096, 36.3]]) add({
      group: "published", what: `Decode time per token, GB200 NVL72, Kimi K2.5, experts over 16 GPUs, ${conc} requests`, unit: "ms", published: 1000 / tps,
      predicted: decodeStep(k25, g.gb200, conc / 16, 8192 + 512, 1, { ep: 16 }).step * 1e3, calibrated: false, source: IXK,
      note: `Decode TP16 EP16 there; modeled with data-parallel attention (the page does not say), no overlap (with two-batch overlap: ${fmt3(decodeStep(k25, g.gb200, conc / 16, 8192 + 512, 1, { ep: 16, overlap: true }).step * 1e3)} ms).` });
  }
  if (g.h100) {
    const perGpu = 14800 / 8 / 21; // 14.8k output tokens/s per 8-GPU node at ~21 tokens/s per user
    add({ group: "published", what: "Decode time per token, DeepSeek-V3 in production, experts over 144 GPUs", unit: "ms", published: 1000 / 21,
      predicted: decodeStep(DEEPSEEK_V3, g.h100, perGpu, 4989, 1, { ep: 144 }).step * 1e3, calibrated: false, source: DS_PROD,
      note: `H800 (modeled as H100: same HBM and FP8 peak); EP144 with data-parallel attention, ${Math.round(perGpu)} sequences per GPU, 4,989 tokens of KV per output token, all as published. No overlap here; DeepSeek uses two-batch overlap, which the model puts at ${fmt3(decodeStep(DEEPSEEK_V3, g.h100, perGpu, 4989, 1, { ep: 144, overlap: true }).step * 1e3)} ms. Its 20–22 tokens/s is a daily average across loads.` });
    // speculative decoding: the measured speed-up falls as the batch grows; the ratio does not depend on the
    // acceptance length, which these tables do not give for their settings
    const fall = (m, gpu, lo, hi, spec) => {
      const gain = (b) => decodeStep(m, gpu, b, 1024, 1).step / decodeStep(m, gpu, b, 1024, 1, { spec: { ...spec, tau: 1 } }).step;
      return gain(hi) / gain(lo);
    };
    add({ group: "published", what: "Speculative decoding speed-up at batch 64 ÷ at batch 2, EAGLE-3 (chain of 3), H100, Llama 3.1 8B", unit: "×", published: 1.38 / 1.81,
      predicted: fall(LLAMA8B, g.h100, 2, 64, { draft: 3, layers: 1, parallel: false }), calibrated: false, source: EAGLE3,
      note: "1.81× at batch 2, 1.38× at batch 64, falling from batch 8 on. The model keeps the whole gain until the step turns compute-bound, so it overstates speculative decoding at moderate batches." });
    if (g.b200) add({ group: "published", what: "Speculative decoding speed-up at 32 requests ÷ at 1, DFlash (block of 16), B200, Qwen3-8B", unit: "×", published: 2.8 / 5.1,
      predicted: fall(QWEN3_8B, g.b200, 1, 32, { draft: 15, layers: 5, parallel: true }), calibrated: false, source: DFLASH,
      note: "5.1× at 1 request, 4.5× at 4 and 8, 3.9× at 16, 2.8× at 32; 15 drafted tokens per block of 16 (an assumption). The model holds the gain flat to 16 and drops at 32, so the endpoints agree better than the curve." });
    if (g.b200) {
      const gain = decodeStep(QWEN3_8B, g.b200, 1, 1024, 1).step / decodeStep(QWEN3_8B, g.b200, 1, 1024, 1, { spec: { draft: 15, tau: 1, layers: 5, parallel: true } }).step;
      add({ group: "published", what: "Speculative decoding speed-up, one request, DFlash (6.49 tokens per step), Qwen3-8B", unit: "×", published: 4.86, predicted: 6.49 * gain, calibrated: false,
        source: { label: "DFlash paper, Table 1: Qwen3-8B, temperature 0, block 16, average of seven tasks", url: "https://arxiv.org/abs/2602.06036" },
        note: "Tests the drafter's cost at batch 1 (5 layers, one parallel pass). The paper's GPU for this table is not given here; B200 assumed." });
    }
  }

  // s1, on the RTX PRO 6000: the model with the configured GPU profile and variants against what s1 measured
  const rtx = g["rtx-pro-6000"];
  if (rtx) {
    const name = "qwen3-vl-32b";
    const m = { ...s1.models[name], ...(c.models[name] ?? {}) };
    const T = timing(m, rtx, sw, 1);
    const at = (rowsOf, n) => rowsOf.find((r) => r.tokens === n)?.ttft_s;
    const n = 32768;
    add({ group: "s1", what: `Cold prefill, ${n.toLocaleString()} tokens, ${name}`, unit: "s", published: at(m.recompute, n), predicted: T.R(n), calibrated: true, source: S1,
      note: "Sets the RTX PRO 6000's prefill_mfu and attention's relative efficiency (no published RTX PRO 6000 prefill result)." });
    add({ group: "s1", what: `Whole prefix cached on the GPU, ${n.toLocaleString()} tokens`, unit: "s", published: at(m.tiers.gpu, n), predicted: T.ttft(n, n, 0), calibrated: true, source: S1,
      note: "Sets vLLM's per-request and per-token overhead (not published)." });
    const cpuMeasured = at(m.tiers.cpu, n) - at(m.tiers.gpu, n);
    add({ group: "s1", what: "CPU → GPU KV load rate, PCIe 5.0 x16", unit: "GiB/s", published: (prefixBytes(m, n) / GIB) / cpuMeasured, predicted: rtx.host_gib_s, calibrated: false, source: S1,
      note: "The model uses py-kvcache's published 54–56 GB/s (vLLM offloading on H100, same link)." });
    for (const tier of ["fs", "obj", "p2p"]) {
      const v = c.variants.find((x) => x.id === S1_TIER_VARIANT[tier]);
      if (!v || !m.tiers[tier]) continue;
      const measured = at(m.tiers[tier], n);
      if (measured == null) continue;
      const fit = fitTier(m, tier);
      const k = n;
      add({
        group: "s1", what: `Prefix of ${n.toLocaleString()} tokens from ${S1_TIER_NAME[tier]}`, unit: "s", published: measured, predicted: T.ttft(n, n, T.loadFor(v)(k)), calibrated: false, source: S1,
        note: `Model: "${v.name}" at ${variantBandwidth(v).toFixed(2)} GiB/s + ${Math.round((v.latency_s ?? 0) * 1000)} ms. s1 measured ${fit.bw_gib_s.toFixed(2)} GiB/s + ${Math.round(fit.latency_s * 1000)} ms.` +
          (tier === "p2p" ? " s1's tuned 32-agent NIXL transfer outruns the generic 39% UCCL TCP figure." : ""),
      });
    }
  }
  return { rows };
}

// The model against every s1 point, for both models: cold prefill, GPU hit and CPU hit, as relative gaps.
export function s1Residuals(s1, c) {
  const rtx = c.gpus.find((x) => x.id === "rtx-pro-6000");
  if (!rtx) return [];
  return Object.entries(s1.models).map(([name, raw]) => {
    const m = { ...raw, ...(c.models[name] ?? {}) };
    const T = timing(m, rtx, c.software, 1);
    const by = (rows) => Object.fromEntries(rows.map((r) => [r.tokens, r.ttft_s]));
    const cold = by(m.recompute), gpu = by(m.tiers.gpu), cpu = by(m.tiers.cpu);
    const rows = Object.keys(cold).map(Number).sort((a, b) => a - b).map((n) => ({
      tokens: n,
      cold: { measured: cold[n], model: T.R(n) },
      gpu: gpu[n] != null ? { measured: gpu[n], model: T.ttft(n, n, 0) } : null,
      cpu: cpu[n] != null ? { measured: cpu[n], model: T.ttft(n, n, T.cpuLoad(n)) } : null,
    }));
    return { name, rows };
  });
}
