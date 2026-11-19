import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { clear, defaultConfig, exportConfig, importConfig, loadSaved, save, validate } from "../config.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const defaults = defaultConfig(S1, VARIANTS, GPUS, SOFTWARE);

test("the default config is valid, holds model architectures but none of s1's measurements", () => {
  assert.deepEqual(validate(defaults), []);
  assert.equal(defaults.version, 4);
  assert.equal(defaults.models["qwen3.8-27b"].kv_bytes_per_token, 64 * 1024);
  for (const m of Object.values(defaults.models)) assert.ok(!("tiers" in m) && !("recompute" in m));
  for (const id of [defaults.sim.shared, defaults.sim.transfer, ...Object.values(defaults.sim.candidates).flat()]) assert.ok(VARIANTS.some((v) => v.id === id), id);
});

test("every number has a source or is marked an assumption; no preset is labelled measured", () => {
  for (const v of VARIANTS) {
    assert.ok(!("basis" in v) && !("measured_tier" in v), v.id);
    assert.ok(typeof v.latency_s === "number", `${v.id}: latency`);
    assert.ok(v.sources.length || v.assumptions.length, `${v.id}: no source and no assumption`);
    for (const s of v.sources) assert.match(s.url, /^https:\/\//, `${v.id}: source url`);
  }
  for (const g of GPUS) {
    assert.ok(!("basis" in g), g.id);
    assert.ok(g.sources.length && g.sources.every((s) => /^https:\/\//.test(s.url)), `${g.id}: sources`);
    if (g.usd_per_hour == null) assert.ok(g.assumptions.includes("usd_per_hour"), `${g.id}: missing price must be marked`);
  }
  const rtx = GPUS.find((g) => g.id === "rtx-pro-6000");
  assert.equal(rtx.mem_gib, S1.gpu_mem_gib);
  assert.equal(rtx.usd_per_hour, S1.prices.gpu_hour_usd["rtx-pro-6000"]);
});

test("export then import round-trips the whole config, edits included", () => {
  const c = structuredClone(defaults);
  c.software.request_overhead_s = 0.1;
  c.variants[0].usd_per_gib_month = 9.99;
  c.variants.push({ ...structuredClone(c.variants[0]), id: "mine", name: "mine" });
  assert.deepEqual(importConfig(exportConfig(c), defaults), c);
});

test("import fills missing settings and rejects broken configs with every problem", () => {
  const c = structuredClone(defaults);
  delete c.workload;
  delete c.software;
  const back = importConfig(JSON.stringify(c), defaults);
  assert.deepEqual(back.workload, defaults.workload);
  assert.deepEqual(back.software, defaults.software);
  assert.throws(() => importConfig("{", defaults), /not valid JSON/);
  const bad = structuredClone(defaults);
  bad.version = 99;
  bad.variants[0].family = "tape";
  bad.variants[1].id = bad.variants[0].id;
  bad.sim.tp = 3;
  assert.throws(() => importConfig(JSON.stringify(bad), defaults), (e) => /version/.test(e.message) && /family/.test(e.message) && /duplicate/.test(e.message) && /tp/.test(e.message));
});

test("browser storage is optional: failures are swallowed", () => {
  const broken = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("quota"); }, removeItem() { throw new Error("no"); } };
  assert.equal(loadSaved(broken, defaults), null);
  assert.equal(save(broken, defaults), false);
  clear(broken);
  assert.equal(loadSaved(null, defaults), null);
  const mem = new Map();
  const ok = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: (k) => mem.delete(k) };
  assert.equal(save(ok, defaults), true);
  assert.deepEqual(loadSaved(ok, defaults), defaults);
  ok.setItem("kvtb-sim-config-v4", "{garbage");
  assert.equal(loadSaved(ok, defaults), null);
});

test("a config saved before the MoE decode fields gets them from the presets, edits kept", () => {
  const d = defaultConfig(S1, VARIANTS, GPUS, SOFTWARE, MODELS);
  const old = structuredClone(d);
  delete old.models["kimi-k3"].expert_share; delete old.models["kimi-k3"].expert_pick; delete old.models["kimi-k3"].decode_params;
  old.models["kimi-k3"].weights_bytes = 1.5e12; // an edit
  delete old.models["gpt-oss-120b"];
  const c = importConfig(exportConfig(old), d);
  assert.equal(c.models["kimi-k3"].expert_pick, 16 / 896);
  assert.equal(c.models["kimi-k3"].weights_bytes, 1.5e12);
  assert.ok(c.models["gpt-oss-120b"].expert_share > 0.9);
});

test("GPUs saved before nvlink_domain get it from the presets", () => {
  const d = defaultConfig(S1, VARIANTS, GPUS, SOFTWARE, MODELS);
  const old = structuredClone(d);
  for (const g of old.gpus) delete g.nvlink_domain;
  const c = importConfig(exportConfig(old), d);
  assert.equal(c.gpus.find((g) => g.id === "gb200").nvlink_domain, 72);
  assert.equal(c.gpus.find((g) => g.id === "b200").nvlink_domain, 8);
});
