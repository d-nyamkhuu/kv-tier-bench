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
  // s1's tiers from published figures: the disk and S3 within a factor 2; s1's TCP peer was software-capped, far slower
  for (const w of ["Prefix of 32,768 tokens from Nebius network SSD", "Prefix of 32,768 tokens from Nebius Object Storage"]) assert.ok(Math.abs(by(w).gap) < 0.5, `${w} ${by(w).gap}`);
  assert.ok(by("Prefix of 32,768 tokens from a peer VM").gap < -0.5);
});

test("against every s1 point, cold prefill, GPU hits and CPU hits stay within 20%", () => {
  for (const { name, rows } of s1Residuals(S1, C)) {
    for (const r of rows) for (const k of ["cold", "gpu", "cpu"]) {
      if (!r[k]) continue;
      assert.ok(Math.abs(r[k].model / r[k].measured - 1) < 0.2, `${name} ${k} ${r.tokens}: ${r[k].model} vs ${r[k].measured}`);
    }
  }
});
