// Prefill TTFT against a published measurement: NVIDIA NIM 1.8.0, Llama-3.3-70B-Instruct FP8, TP2 on 2×H100 80GB,
// 5,000 input / 500 output tokens (https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html). NIM ran a
// closed loop at fixed concurrency; the rates below are its measured output throughput / 500 tokens. The replay is
// prefill only with batching on, no cache, Poisson arrivals. NIM also decodes on the same GPUs, so it saturates
// earlier (concurrency 100, ~1.83 req/s, 8.0 s); the comparison stops below that.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { defaultConfig } from "../config.js";
import { deriveArch } from "../model.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";
import { runScenario } from "../scenario.js";

const s1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
// Llama-3.3-70B-Instruct config.json: 80 layers, hidden 8192, 64 query heads, 8 KV heads of 128; 70.55B params, 141 GB in BF16.
const llama = { name: "Llama-3.3-70B", released: "BF16", ...deriveArch({ num_layers: 80, hidden_size: 8192, total_params: 70.55e9, weights_bytes: 141.1e9,
  compute_dtype: "bf16", block_size: 16, layers: [{ type: "full", count: 80, q_heads: 64, kv_heads: 8, k_dim: 128, v_dim: 128 }] }) };

function poisson(ratePerS, isl, durS = 600) {
  let seed = 7;
  const u = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const requests = [];
  for (let t = 0, i = 0; (t += -Math.log(1 - u()) / ratePerS) <= durS; i++)
    requests.push({ t, n: isl, out: 0, hashes: Array.from({ length: Math.floor(isl / 16) }, (_, j) => `q${i}:${j}`) });
  return { name: "poisson", blockTokens: 16, requests };
}

function ttft(ratePerS, isl) {
  const c = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, { ...MODELS, llama });
  Object.assign(c.sim, { model: "llama", precision: "fp8", gpu: "h100", decodeGpu: "h100", nodes: 1, gpusPerNode: 2, tp: 2, pp: 1,
    attention: "tp", decode: false, cpuGiB: 0, disk: null, shared: null, p2p: null, warmupMin: 1, prefillBatchTokens: 8192 });
  return runScenario(c, {}, poisson(ratePerS, isl), { maxSimNodes: 1 }).ttft.mean * 1000;
}

// DeepSeek-R1-0528 NVFP4 (config.json: 61 MLA layers, hidden 7168, 128 query heads, latent 512 + rope 64; 671B params,
// 37B active; 423.6 GB of safetensors). FP4 prefill runs at the FP8 rate (model.js prefillTflops) because this run
// fits it: the FP4 peak (4 x BF16) predicts 130 ms for an idle 8k prompt on 4 B200s against 215 ms measured. That makes
// the test below a regression pin on a fitted setting, not an independent check. docs/design/prefill-batching.md has
// the engines it does not fit.
const r1 = { name: "DeepSeek-R1 NVFP4", released: "NVFP4", ...deriveArch({ num_layers: 61, hidden_size: 7168, total_params: 671e9, active_params: 37e9,
  weights_bytes: 423.6e9, compute_dtype: "fp4", block_size: 64, layers: [{ type: "mla", count: 61, q_heads: 128, latent: 512, rope: 64, qk_dim: 192, v_dim: 128 }] }) };

// InferenceX's fixed-sequence client (infx/bench/fixed_seq.py): `conc` clients, 10 x conc prompts, request rate inf,
// inputs uniform in [0.8, 1] x 8192. Each client sends its next prompt when the last reply ends; the gap is the
// measured decode time (mean e2el - mean TTFT), spread like the output lengths.
function closedLoop(conc, decodeS, isl = 8192) {
  let seed = 11;
  const u = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const requests = [];
  for (let c = 0; c < conc; c++) {
    let t = 0;
    for (let j = 0; j < 10; j++) {
      const n = Math.round(isl * (0.8 + 0.2 * u()));
      requests.push({ t, conv: `c${c}`, followUp: j > 0, n, out: 0, hashes: Array.from({ length: Math.floor(n / 64) }, (_, k) => `c${c}r${j}:${k}`) });
      t += (decodeS * (0.8 + 0.2 * u())) / 0.9;
    }
  }
  requests.sort((a, b) => a.t - b.t);
  return { name: "closed", blockTokens: 64, durationS: 1e6, requests };
}

test("batched prefill TTFT stays pinned to InferenceX's DeepSeek-R1 NVFP4 run on 4 B200s (TensorRT-LLM, 8k/1k)", () => {
  // conc, measured median and mean TTFT (s), mean decode time (s): TensorRT-LLM 1.3.0rc14, TP4, aggregated, 2026-05-19
  // (https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26016880548/attempts/2). p99 is left out: with every
  // client starting at once it measures the first wave (about conc x 170 ms), not batching.
  for (const [conc, p50, mean, decodeS] of [[4, 0.2154, 0.2513, 7.333], [8, 0.2169, 0.3026, 9.064], [16, 0.223, 0.3885, 12.509], [32, 0.2391, 0.5597, 18.37]]) {
    const c = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, { ...MODELS, r1 });
    Object.assign(c.sim, { model: "r1", precision: "released", gpu: "b200", decodeGpu: "b200", nodes: 1, gpusPerNode: 4, tp: 4, pp: 1,
      attention: "tp", decode: false, cpuGiB: 0, disk: null, shared: null, p2p: null, warmupMin: 0, prefillBatchTokens: 8192 });
    const { ttft } = runScenario(c, {}, closedLoop(conc, decodeS), { maxSimNodes: 1 });
    assert.ok(Math.abs(ttft.p50 / p50 - 1) < 0.15, `c${conc}: simulated median ${(ttft.p50 * 1e3).toFixed(0)} ms, measured ${p50 * 1e3} ms`);
    assert.ok(Math.abs(ttft.mean / mean - 1) < 0.15, `c${conc}: simulated mean ${(ttft.mean * 1e3).toFixed(0)} ms, measured ${mean * 1e3} ms`);
  }
});

test("batched prefill TTFT tracks NIM's measured Llama-3.3-70B TTFT within 15% below saturation", () => {
  // [measured req/s, NIM mean TTFT ms] at concurrency 1, 5, 25 and 50
  for (const [rate, nim] of [[0.103, 406.77], [0.437, 546.7], [1.247, 688.69], [1.629, 834.51]]) {
    const sim = ttft(rate, 5000);
    assert.ok(Math.abs(sim / nim - 1) < 0.15, `${rate} req/s: simulated ${sim.toFixed(0)} ms, NIM ${nim} ms`);
  }
});
