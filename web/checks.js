// The model against measurements of real setups: published vLLM benchmarks, and kv-tier-bench s1 (RTX PRO 6000,
// Nebius, vLLM v0.34.0) as the reference a real deployment should measure up to. DOM-free: the page shows the
// tables, a test keeps the gaps from drifting. "calibrated" rows set a model parameter, so they agree by construction.

import { decodeStepS, fitTier, GIB, prefixBytes, timing, variantBandwidth } from "./model.js";

// Llama 3.3 70B in FP8 (1 byte per weight), bf16 KV: 80 layers x 8 KV heads x 128 x K,V x 2 bytes
const LLAMA70B = { num_layers: 80, params: 70553706496, weights_bytes: 70553706496, kv_bytes_per_token: 80 * 8 * 128 * 2 * 2, state_bytes_per_prefix: 0, read_fixed_bytes: 0, full_attn_layers: 80, attn_width: 64 * 128 };
const FP8_TFLOPS = { h100: 1979, h200: 1979, b200: 4500 }; // dense; NVIDIA datasheets quote 2x "with sparsity"
const IX = "https://inferencex.semianalysis.com/compare/llama-3-3-70b-b200-vs-h100";
const IX_LABEL = "InferenceX: vLLM, Llama 3.3 70B FP8, 8k in / 1k out, concurrency 4, median (run 2025-10-29)";
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
          (tier === "p2p" ? " s1's NIXL transfer was held back by one receive thread; the model assumes none." : ""),
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
