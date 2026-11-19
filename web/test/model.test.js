import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { calibrate, fitLinearRel, fitTier, GIB, interpLogLog, variantBandwidth } from "../model.js";
import { GPUS, SOFTWARE } from "../presets.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const close = (a, b, rel = 1e-9) => assert.ok(Math.abs(a / b - 1) <= rel, `${a} vs ${b}`);

test("log-log interpolation is exact at the points and on a power law", () => {
  const pts = [[1000, 2], [2000, 8], [8000, 128]]; // y = 2e-6 x^2
  close(interpLogLog(pts, 2000), 8);
  close(interpLogLog(pts, 4000), 32);
  close(interpLogLog(pts, 16000), 512); // extrapolated with the last segment's slope
  close(interpLogLog(pts, 500), 0.5); // and the first segment's
  assert.equal(interpLogLog(pts, 0), 0);
});

test("the relative-error fit recovers L and BW, and clamps a negative L to zero", () => {
  const xs = [1, 2, 4, 8, 16].map((g) => g * GIB);
  const fit = fitLinearRel(xs, xs.map((x) => 0.02 + x / (5 * GIB)));
  close(fit.L, 0.02, 1e-9);
  close(1 / fit.c / GIB, 5, 1e-9);
  const neg = fitLinearRel(xs, xs.map((x) => -0.05 + x / (5 * GIB)));
  assert.equal(neg.L, 0);
  assert.ok(neg.c > 0);
});

test("the software constants and the RTX PRO 6000's prefill_mfu in presets.js are what s1 calibrates to", () => {
  const rtx = GPUS.find((g) => g.id === "rtx-pro-6000");
  const c = calibrate(S1, rtx.tflops_bf16, rtx.host_gib_s);
  for (const k of ["request_overhead_s", "per_token_overhead_s", "attn_eff_ratio", "cpu_load_latency_s"]) close(SOFTWARE[k], c[k], 0.01);
  close(rtx.prefill_mfu, c.prefill_mfu, 0.01);
});

test("the s1 fit of every tier s1 measured reproduces its medians within 20%, half within 10%", () => {
  for (const m of Object.values(S1.models)) {
    for (const tier of ["fs", "obj", "p2p"]) {
      const f = fitTier(m, tier);
      assert.ok(f.bw_gib_s > 0.5 && f.bw_gib_s < 20, `${tier} BW ${f.bw_gib_s}`);
      assert.ok(f.maxAbsRel < 0.2, `${tier} max residual ${f.maxAbsRel}`);
      const sorted = f.residuals.map((r) => Math.abs(r.rel)).sort((a, b) => a - b);
      assert.ok(sorted[Math.floor(sorted.length / 2)] < 0.1, `${tier} median residual`);
    }
  }
});

test("a network variant is link x efficiency, capped by its software limit", () => {
  close(variantBandwidth({ link_gbit_s: 100, efficiency: 0.8 }), (100e9 * 0.8) / 8 / GIB);
  assert.equal(variantBandwidth({ link_gbit_s: 400, efficiency: 0.9, cap_gib_s: 2.6 }), 2.6);
  assert.equal(variantBandwidth({ bw_gib_s: 7 }), 7);
  close(variantBandwidth({ bw_gib_s: 10, efficiency: 0.8 }), 8);
});

test("deriveArch reproduces s1's sizes from the two models' config.json", async () => {
  const { deriveArch, gdnStateBytes } = await import("../model.js");
  const qwen27 = deriveArch({ total_params: 1, weights_bytes: 1, block_size: 448, layers: [
    { type: "full", count: 16, q_heads: 24, kv_heads: 4, k_dim: 256, v_dim: 256 },
    { type: "linear", count: 48, state_bytes: gdnStateBytes({ k_heads: 16, v_heads: 48, k_dim: 128, v_dim: 128, conv: 4 }) },
  ] });
  const s1 = S1.models["qwen3.8-27b"];
  assert.equal(qwen27.kv_bytes_per_token, 65536);
  assert.deepEqual(Object.values(qwen27.kv_bytes_per_token_by_tp), Object.values(s1.kv_bytes_per_token_by_tp));
  assert.equal(qwen27.state_bytes_per_prefix, s1.state_bytes_per_prefix); // 78,446,592
  assert.equal(qwen27.full_attn_layers * qwen27.attn_width, s1.full_attn_layers * s1.attn_width);
  assert.equal(qwen27.hybrid, true);
  const vl = deriveArch({ total_params: 1, weights_bytes: 1, block_size: 64, layers: [{ type: "full", count: 64, q_heads: 64, kv_heads: 8, k_dim: 128, v_dim: 128 }] });
  assert.equal(vl.kv_bytes_per_token, 262144);
  assert.equal(vl.attn_width, 8192);
  assert.equal(vl.state_bytes_per_prefix, 0);
  // MLA: the latent is copied on every TP rank; a sliding window is a fixed part, not per token
  const mla = deriveArch({ total_params: 1, weights_bytes: 1, block_size: 64, layers: [
    { type: "mla", count: 2, q_heads: 8, latent: 512, rope: 64, qk_dim: 192, v_dim: 128 },
    { type: "swa", count: 1, q_heads: 8, kv_heads: 2, k_dim: 64, v_dim: 64, window: 128 },
  ] });
  assert.equal(mla.kv_bytes_per_token, 2 * 576 * 2);
  assert.equal(mla.kv_bytes_per_token_by_tp[4], 4 * 2 * 576 * 2);
  assert.equal(mla.state_bytes_per_prefix, 2 * 128 * 2 * 128);
});

test("windowed attention costs n x window, not n^2, once past the window", async () => {
  const { prefillFlops } = await import("../model.js");
  const m = { params: 0, full_attn_layers: 0, attn_width: 0, attn_windows: [{ width: 1, tokens: 100 }] };
  assert.equal(prefillFlops(m, 50), 4 * 50 * 50 / 2);
  assert.equal(prefillFlops(m, 1000), 4 * (1000 * 100 - 100 * 100 / 2));
  const full = { params: 0, full_attn_layers: 1, attn_width: 1 };
  assert.equal(prefillFlops(full, 1000), 2 * 1000 * 1000);
});

test("serving precision: FP8 halves bf16 weights and KV, FP4 is 4.5 bits a weight; nothing grows past the release", async () => {
  const { withPrecision, peakTflops } = await import("../model.js");
  const { MODELS } = await import("../presets.js");
  const g = MODELS["gemma-4-31b"];
  const f8 = withPrecision(g, "fp8"), f4 = withPrecision(g, "fp4");
  assert.equal(f8.weights_bytes, g.total_params);
  assert.equal(f4.weights_bytes, (g.total_params * 4.5) / 8);
  assert.equal(f8.kv_bytes_per_token, g.kv_bytes_per_token / 2);
  assert.equal(f8.read_fixed_bytes, g.read_fixed_bytes / 2); // Gemma's fixed part is all sliding-window KV
  assert.equal(withPrecision(g, "released"), g);
  const ds = MODELS["deepseek-v4.1-flash"]; // released FP8 with FP4 experts and an FP4 KV cache
  assert.equal(withPrecision(ds, "fp8").weights_bytes, ds.weights_bytes);
  assert.equal(withPrecision(ds, "fp8").kv_bytes_per_token, ds.kv_bytes_per_token);
  const kimi = MODELS["kimi-k3"]; // linear-attention state stays bf16
  assert.equal(withPrecision(kimi, "fp8").read_fixed_bytes, kimi.read_fixed_bytes);
  // FP4 computes at FP4 on Blackwell, at FP8 where there are no FP4 tensor cores
  assert.equal(peakTflops(f4, { tflops_bf16: 1, tflops_fp8: 2, tflops_fp4: 4 }), 4);
  assert.equal(peakTflops(f4, { tflops_bf16: 1, tflops_fp8: 2, tflops_fp4: null }), 2);
});
