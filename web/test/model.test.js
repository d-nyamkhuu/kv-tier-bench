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
      assert.ok(f.bw_gib_s > 0.5 && f.bw_gib_s < 46.6, `${tier} BW ${f.bw_gib_s}`); // below the 400 Gbit/s NIC (p2p: 23-27)
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
  const { withPrecision, peakTflops, prefillTflops } = await import("../model.js");
  const { MODELS } = await import("../presets.js");
  const g = MODELS["gemma-4-31b"];
  const f8 = withPrecision(g, "fp8"), f4 = withPrecision(g, "fp4");
  assert.equal(f8.weights_bytes, g.total_params);
  assert.equal(f4.weights_bytes, (g.total_params * 4.5) / 8);
  assert.equal(f8.kv_bytes_per_token, g.kv_bytes_per_token / 2);
  assert.equal(f8.state_window_bytes, g.state_window_bytes / 2); // Gemma's fixed part is sliding-window KV, not linear state
  assert.equal(f8.state_bytes_per_prefix, g.state_bytes_per_prefix / 2);
  assert.equal(g.read_fixed_bytes, 0);
  assert.equal(withPrecision(g, "released"), g);
  const ds = MODELS["deepseek-v4.1-flash"]; // released FP8 with FP4 experts and packed FP8 KV
  assert.equal(withPrecision(ds, "fp8").weights_bytes, ds.weights_bytes);
  assert.equal(withPrecision(ds, "fp8").kv_bytes_per_token, ds.kv_bytes_per_token);
  assert.ok(ds.expert_share < 0.8, `FP4 experts share of 510 GB weights: ${ds.expert_share}`);
  const l40s = GPUS.find((x) => x.id === "l40s");
  assert.ok(l40s.prefill_mfu < 0.7, `L40S prefill MFU should not borrow RTX's 0.878: ${l40s.prefill_mfu}`);
  const kimi = MODELS["kimi-k3"]; // linear-attention state stays bf16
  assert.equal(withPrecision(kimi, "fp8").read_fixed_bytes, kimi.read_fixed_bytes);
  // GLM-5.3 and Mistral Large 4 ship FP8 sparse-indexer keys (132 B/layer): switching KV to FP8 halves MLA latent KV only, not INDEX_BYTES
  const glm = MODELS["glm-5.3"];
  assert.equal(withPrecision(glm, "fp8").kv_bytes_per_token, 78 * (512 + 64) * 1 + 21 * 132);
  const mistral = MODELS["mistral-large-4"];
  assert.equal(withPrecision(mistral, "fp8").kv_bytes_per_token, 63 * (512 + 64) * 1 + 44 * 132);
  // FP4 computes at FP4 on Blackwell, at FP8 where there are no FP4 tensor cores
  assert.equal(peakTflops(f4, { tflops_bf16: 1, tflops_fp8: 2, tflops_fp4: 4 }), 4);
  assert.equal(peakTflops(f4, { tflops_bf16: 1, tflops_fp8: 2, tflops_fp4: null }), 2);
  // prefill runs FP4 at the FP8 rate (fitted to InferenceX's DeepSeek-R1 NVFP4 runs)
  assert.equal(prefillTflops(f4, { tflops_bf16: 1, tflops_fp8: 2, tflops_fp4: 4 }), 2);
  assert.equal(prefillTflops(withPrecision(g, "fp8"), { tflops_bf16: 1, tflops_fp8: 2, tflops_fp4: 4 }), 2);
});

test("DP-attention replicates active dense weights, lower tiers stage through host memory, and NVL72 stage hops use NVLink", async () => {
  const { denseWeightBytes, effectiveLink, instanceWeightBytes, prefixBytes, served, timing } = await import("../model.js");
  const { MODELS, VARIANTS } = await import("../presets.js");

  // 1. Kimi K3 uses mla_use_nope (24 * 512 * 2 = 24,576 B/tok BF16) and replicates active dense weights under DP.
  const kimi = MODELS["kimi-k3"];
  assert.equal(kimi.kv_bytes_per_token, 24 * 512 * 2);
  const kimiDp = served(kimi, { precision: "fp8", tp: 8, attention: "dp" });
  const kimiTp = served(kimi, { precision: "fp8", tp: 8, attention: "tp" });
  assert.ok(instanceWeightBytes(kimiDp, 8) > instanceWeightBytes(kimiTp, 8) + 300e9);

  // 2. DeepSeek V4.1 Flash: excludes 196B Engram lookup table from active dense weights; FP4 gives 890 B/tok; RTX PRO 6000 gets 528B Blackwell row.
  const ds = MODELS["deepseek-v4.1-flash"];
  assert.ok(denseWeightBytes(ds) < 20e9, `active dense weights exclude Engram table: ${denseWeightBytes(ds)}`);
  const rtx = GPUS.find((g) => g.id === "rtx-pro-6000");
  assert.equal(served(ds, { precision: "fp4", tp: 1, attention: "dp", gpu: rtx }).kv_bytes_per_token, 890);
  assert.equal(served(ds, { precision: "fp8", tp: 1, attention: "dp", gpu: rtx }).kv_bytes_per_token, 1650);

  // 3. Local Gen5 NVMe: the published rate is timed disk to GPU, so no host copy is added (~11.5 GB/s). A peer-memory
  // hit over RDMA lands in host memory first and pays the host -> GPU copy; only a gpu_direct device skips it.
  const m = S1.models["qwen3-vl-32b"];
  const h100 = GPUS.find((g) => g.id === "h100");
  const T = timing(m, h100, SOFTWARE, 1);
  const bytes = prefixBytes(m, 32768);
  const nvme = VARIANTS.find((v) => v.id === "fs-nvme-gen5");
  const ndr = VARIANTS.find((v) => v.id === "p2p-ib-ndr");
  assert.ok((bytes / 1e9) / T.loadFor(nvme)(32768) > 11.2, "NVMe load matches py-kvcache 11.2-12.0 GB/s");
  const ndrBw = variantBandwidth(ndr) * GIB, hostBw = h100.host_gib_s * GIB;
  assert.ok(Math.abs(T.loadFor(ndr)(32768) - (ndr.latency_s + SOFTWARE.cpu_load_latency_s + bytes / ndrBw + bytes / hostBw)) < 1e-12,
    "a peer hit is host to host, then host to GPU");
  const icms = VARIANTS.find((v) => v.id === "fs-bf4-icms");
  assert.ok(Math.abs(T.loadFor(icms)(32768) - (icms.latency_s + bytes / (variantBandwidth(icms) * GIB))) < 1e-12, "gpu_direct skips the host copy");

  // 4. GB200 and VR200 NVL72 upgrade intra-rack RDMA links to rack-scale NVLink 5/6, while B200 (8-GPU domain) does not cross nodes over NVLink.
  const gb200 = GPUS.find((g) => g.id === "gb200");
  const vr200 = GPUS.find((g) => g.id === "vr200");
  const b200 = GPUS.find((g) => g.id === "b200");
  assert.ok(variantBandwidth(effectiveLink(ndr, gb200, 4, 8)) > 600, "GB200 NVL72 intra-rack uses NVLink 5");
  assert.ok(variantBandwidth(effectiveLink(ndr, vr200, 8, 8)) > 1200, "VR200 NVL72 intra-rack uses NVLink 6");
  assert.equal(effectiveLink(ndr, gb200, 10, 8), ndr, "80 GPUs > 72-GPU NVL72 domain falls back to scale-out RDMA");
  assert.equal(effectiveLink(ndr, b200, 2, 8), ndr, "HGX B200 8-GPU domain never spans separate nodes over NVLink");
});

test("Kimi K3's dense weights are BF16 as released, and expert-parallel comm ranks links by the rate they achieve", async () => {
  const { decodeStep, denseWeightBytes, served, withPrecision } = await import("../model.js");
  const { MODELS, VARIANTS } = await import("../presets.js");
  const kimi = MODELS["kimi-k3"];
  // The non-expert bytes of the checkpoint are its dense part (BF16); FP8 serving halves them and the total follows.
  assert.ok(Math.abs(denseWeightBytes(kimi) / (kimi.weights_bytes * (1 - kimi.expert_share)) - 1) < 0.05);
  const f8 = withPrecision(kimi, "fp8");
  assert.ok(Math.abs(denseWeightBytes(f8) / denseWeightBytes(kimi) - 0.5) < 1e-9);
  assert.ok(Math.abs((kimi.weights_bytes - f8.weights_bytes) - (denseWeightBytes(kimi) - denseWeightBytes(f8))) < 1);

  const b200 = GPUS.find((g) => g.id === "b200");
  const m = served(MODELS["minimax-m3"], { gpu: b200, precision: "fp8", tp: 8, attention: "dp" });
  const comm = (id) => decodeStep(m, b200, 64, 8192, 1, { ep: 16, link: VARIANTS.find((v) => v.id === id) }).comm;
  const order = ["p2p-tcp-10", "p2p-tcp-100", "p2p-tcp-400", "p2p-ib-hdr", "p2p-ib-ndr", "p2p-ib-xdr"].map(comm);
  for (let i = 1; i < order.length; i++) assert.ok(order[i] < order[i - 1], `a faster link moves expert traffic faster: ${order}`);
  assert.equal(comm("p2p-roce-400"), comm("p2p-ib-ndr"));
});
