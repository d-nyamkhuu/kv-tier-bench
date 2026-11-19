import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { checks, s1Residuals } from "../checks.js";
import { defaultConfig } from "../config.js";
import { GPUS, SOFTWARE, VARIANTS } from "../presets.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const C = defaultConfig(S1, VARIANTS, GPUS, SOFTWARE);

test("the model against published results and s1: calibrated rows agree, the checks stay within their known gaps", () => {
  const { rows } = checks(S1, C);
  const by = (what) => rows.find((r) => r.what.startsWith(what));
  for (const r of rows.filter((x) => x.calibrated && x.group === "published")) assert.ok(Math.abs(r.gap) < 0.01, `${r.what} ${r.gap}`);
  for (const r of rows.filter((x) => x.calibrated && x.group === "s1")) assert.ok(Math.abs(r.gap) < 0.15, `${r.what} ${r.gap}`);
  // independent checks: if a change moves these, the page's claims change too
  assert.ok(Math.abs(by("Decode time per token, H200 SXM, TP1").gap) < 0.15);
  assert.ok(Math.abs(by("Decode time per token, B200 (HGX), TP1").gap) < 0.15);
  assert.ok(Math.abs(by("Prefill TTFT, H100 SXM, TP2").gap) < 0.15);
  assert.ok(Math.abs(by("Prefill TTFT, H100 SXM, TP4").gap) < 0.25);
  assert.ok(Math.abs(by("CPU → GPU KV load rate").gap) < 0.05); // py-kvcache's published rate vs s1's
  for (const tp of [2, 4]) assert.ok(Math.abs(by(`Decode time per token, H100 SXM, TP${tp}`).gap) < 0.15); // all-reduce cost per layer, from H200/B200
  // s1's tiers from published figures, within a factor 2. The TCP peer joined them with the NIXL tuning (32 agents,
  // the 2026-11-19 re-run); before it, the peer was software-capped and far slower than its NIC.
  for (const w of ["Prefix of 32,768 tokens from Nebius network SSD", "Prefix of 32,768 tokens from Nebius Object Storage", "Prefix of 32,768 tokens from a peer VM"]) assert.ok(Math.abs(by(w).gap) < 0.5, `${w} ${by(w).gap}`);
});

test("decode checks beyond one dense instance: their known gaps (none of these rows sets a parameter)", () => {
  const { rows } = checks(S1, C);
  const gap = (what) => rows.find((r) => r.what.startsWith(what)).gap;
  assert.ok(Math.abs(gap("Decode time per token, B200, Qwen3-8B")) < 0.25); // ~0.8 ms per step the model leaves out
  assert.ok(Math.abs(gap("Decode time per token, B200 TP4, Kimi K2.5 (MoE), batch 64")) < 0.2);
  assert.ok(gap("Decode time per token, B200 TP4, Kimi K2.5 (MoE), batch 4") < -0.2); // too fast at small MoE batches
  assert.ok(Math.abs(gap("Decode time per token, GB200 NVL72, Kimi K2.5, experts over 16 GPUs, 4096")) < 0.25);
  assert.ok(Math.abs(gap("Decode time per token, DeepSeek-V3 in production")) < 0.25); // no overlap; with it, 43% too fast
  assert.ok(gap("Decode time per token, GB200 NVL72, Kimi K2.5, experts over 16 GPUs, 512") < -0.15); // too fast at 32 per GPU
  assert.ok(Math.abs(gap("Speculative decoding speed-up, one request, DFlash")) < 0.25);
  assert.ok(Math.abs(gap("Speculative decoding speed-up at 32 requests")) < 0.2); // DFlash's fall with load
  assert.ok(gap("Speculative decoding speed-up at batch 64") > 0.2); // EAGLE-3's fall: the model keeps more of the gain
});

test("against every s1 point, cold prefill, GPU hits and CPU hits stay within 20%", () => {
  for (const { name, rows } of s1Residuals(S1, C)) {
    for (const r of rows) for (const k of ["cold", "gpu", "cpu"]) {
      if (!r[k]) continue;
      assert.ok(Math.abs(r[k].model / r[k].measured - 1) < 0.2, `${name} ${k} ${r.tokens}: ${r[k].model} vs ${r[k].measured}`);
    }
  }
});
