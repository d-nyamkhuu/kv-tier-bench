import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { defaultConfig, validate } from "../config.js";
import { atBusy, labFrom, logSpace, modelCompare, propOf, propsFor, sweepPoint, TARGET_BUSY, withProp } from "../lab.js";
import { variantBandwidth } from "../model.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";
import { runScenario } from "../scenario.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const config = () => defaultConfig(S1, VARIANTS, GPUS, SOFTWARE, MODELS);
const s3 = (c) => c.variants.find((v) => v.id === "obj-aws-s3-standard");

test("the lab's properties: size only where a tier has one of its own, price per node for peers", () => {
  assert.deepEqual(propsFor("shared").map((p) => p.key), ["latency", "bandwidth", "size", "price"]);
  assert.deepEqual(propsFor("p2p").map((p) => p.key), ["latency", "bandwidth", "price"]);
  assert.equal(propOf("p2p", "price").unit, "$/h per node");
});

test("setting bandwidth on a link-based preset replaces the link derivation and round-trips", () => {
  const c = config();
  const v = s3(c);
  assert.ok(v.link_gbit_s > 0); // an object store's rate comes from its link
  const w = withProp(v, "shared", "bandwidth", 25);
  assert.equal(variantBandwidth(w), 25);
  assert.equal(propOf("shared", "bandwidth").get(w), 25);
  assert.deepEqual(w.edited, ["bw_gib_s"]);
  assert.ok(v.link_gbit_s > 0, "the preset itself is untouched");
  const lat = withProp(w, "shared", "latency", 40);
  assert.equal(lat.latency_s, 0.04);
});

test("a lab variant keeps no sources or assumptions and passes validation", () => {
  const c = config();
  const lv = withProp(labFrom(s3(c), "shared"), "shared", "size", 5000);
  assert.equal(lv.id, "lab-shared");
  assert.deepEqual(lv.sources, []);
  assert.deepEqual(lv.assumptions, []);
  c.variants.push(lv);
  c.sim.shared = lv.id;
  assert.deepEqual(validate(c), []);
});

test("logSpace: evenly spaced on a log scale, with the current value added", () => {
  const xs = logSpace(1, 100, 3, 7);
  assert.deepEqual(xs.map((x) => +x.toFixed(6)), [1, 7, 10, 100]);
  assert.equal(logSpace(1, 100, 3, 10.1).length, 3, "a value next to a step is not added twice");
});

test("a faster shared tier: lower TTFT at the same load, and a lower cost once the GPUs are loaded to the target", () => {
  const c = config();
  const slow = sweepPoint(c, "shared", s3(c), "bandwidth", 0.5);
  const fast = sweepPoint(c, "shared", s3(c), "bandwidth", 50);
  assert.ok(fast.ttft < slow.ttft);
  assert.equal(fast.usd, slow.usd, "at a fixed cluster and load every GPU is paid for anyway");
  assert.ok(fast.capUsd < slow.capUsd, "the headroom serves more requests on the same GPUs");
  assert.ok(fast.capRpm > slow.capRpm);
});

test("atBusy lands near the target busy share", () => {
  const c = config();
  const { cap } = atBusy(c);
  assert.ok(Math.abs(cap.utilization / TARGET_BUSY - 1) < 0.05, `${cap.utilization}`);
});

test("across models: the same traffic for both setups, each with a finite TTFT and cost", () => {
  const c = config();
  const r = modelCompare(c, "qwen3.8-27b");
  assert.ok(!r.error, r.error);
  assert.ok(r.rpm > 0 && r.base.ttft > 0 && r.tier.ttft > 0 && r.base.usd > 0 && r.tier.usd > 0);
  // both TTFTs come from one rate: rerun each setup at r.rpm on the same deployment and get the same numbers
  const cc = { ...c, sim: { ...c.sim, tp: r.tp, pp: r.pp, gpusPerNode: r.tp, nodes: r.pp * r.instances, warmupMin: 10, decodePooled: true },
    workload: { ...c.workload, durationMin: 20, requestsPerMin: r.rpm } };
  assert.equal(runScenario(cc, { disk: null, p2p: null, shared: null }).ttft.mean, r.base.ttft);
  assert.equal(runScenario(cc).ttft.mean, r.tier.ttft);
});
