import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { defaultConfig, validate } from "../config.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";
import { minutesCsv, replay } from "../replay.js";
import { synthetic } from "../sim.js";
import {
  advancedHash, advancedOverrides, applySettings, decodeSettings, DEFAULT_SETTINGS, encodeSettings, readAdvancedHash,
  fitCluster, SCENARIOS, scenarioFor, usdPerHour, withOverrides,
} from "../sim-settings.js";

const s1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const config = () => defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, MODELS);

test("default settings encode to an empty hash and round-trip through the URL", () => {
  assert.equal(encodeSettings(DEFAULT_SETTINGS), "");
  const custom = { ...DEFAULT_SETTINGS, rate: 300, disk: "fs-nvme-gen4", diskSize: null, backing: "" };
  const hash = encodeSettings(custom);
  assert.deepEqual(decodeSettings("#" + hash), custom);
  assert.match(hash, /diskSize=full/);
});

test("decoding ignores unknown keys and malformed numbers", () => {
  const settings = decodeSettings("#rate=abc&nodes=&bogus=1&turns=3");
  assert.equal(settings.rate, DEFAULT_SETTINGS.rate);
  assert.equal(settings.nodes, DEFAULT_SETTINGS.nodes);
  assert.equal(settings.turns, 3);
  assert.ok(!("bogus" in settings));
});

test("every scenario names valid ids and is recognised from its settings", () => {
  const c = config();
  for (const scenario of SCENARIOS) {
    assert.equal(scenarioFor({ ...scenario.settings }), scenario);
    assert.ok(scenario.settings.model in c.models, scenario.id);
    assert.ok(c.gpus.some(g => g.id === scenario.settings.gpu), scenario.id);
    for (const id of [scenario.settings.disk, scenario.settings.backing, scenario.settings.interconnect].filter(Boolean))
      assert.ok(c.variants.some(v => v.id === id), id);
  }
  assert.equal(scenarioFor({ ...DEFAULT_SETTINGS, rate: 130 }), null);
});

test("the scenarios show the pressure each one names", () => {
  const run = id => {
    const c = config();
    const budget = applySettings(c, SCENARIOS.find(s => s.id === id).settings);
    return replay(c, budget);
  };
  const gpu = run("gpu"), ample = run("ample"), disk = run("disk");
  assert.ok(gpu.firstEviction != null, "the default GPU cache fills");
  assert.equal(ample.firstEviction, null, "the large-memory scenario never evicts");
  assert.ok(disk.firstDiskEviction != null && disk.share.disk > 0, "the SSD scenario fills and serves the SSD");
});

test("a model that does not fit grows the cluster to the smallest layout that holds it", () => {
  const c = config();
  assert.equal(fitCluster(c, DEFAULT_SETTINGS), null, "the default already fits");
  const oneGpu = fitCluster(c, { ...DEFAULT_SETTINGS, gpusPerNode: 1 });
  assert.ok(oneGpu.gpusPerNode > 1 && oneGpu.nodes === DEFAULT_SETTINGS.nodes, "more GPUs per node before more nodes");
  const big = { ...DEFAULT_SETTINGS, model: "qwen3.8-2.4t-a95b", nodes: 1 };
  const grown = fitCluster(c, big);
  assert.ok(grown && !grown.error && grown.nodes > 1, JSON.stringify(grown));
  assert.equal(fitCluster(c, { ...big, ...grown }), null, "the grown cluster holds the model");
  assert.match(fitCluster(c, { ...big, gpu: "l40s" }).error, /does not fit/);
});

test("the advanced-lab hand-off produces a valid config with the fitted layout", () => {
  const c = config();
  const settings = { ...DEFAULT_SETTINGS, cpu: 0, gpuBudget: 50, diskSize: null, disk: "fs-nvme-gen5" };
  const overrides = advancedOverrides(settings, { tp: 2, pp: 1 });
  const next = withOverrides(c, readAdvancedHash(advancedHash(overrides)));
  assert.deepEqual(validate(next), []);
  assert.equal(next.sim.tp, 2);
  assert.equal(next.sim.p2p, null, "no peer tier without CPU cache");
  assert.equal(next.sim.gpuBudgetFraction, 0.5);
  assert.equal(next.sim.disk, "fs-nvme-gen5", "the whole drive keeps the product");
  assert.equal(next.sim.diskCapacityGiB, undefined, "no size override hides behind the lab's Size control");
  const sized = withOverrides(c, advancedOverrides({ ...settings, diskSize: 64 }, { tp: 2, pp: 1 }));
  assert.deepEqual(validate(sized), []);
  assert.equal(sized.sim.disk, "lab-disk");
  assert.equal(sized.variants.find(v => v.id === "lab-disk").capacity_gib, 64);
  assert.equal(sized.sim.diskCapacityGiB, undefined);
  assert.equal(next.workload.thinkTimeS, 300);
  assert.equal(readAdvancedHash("#from-simulator=%7Bnot-json"), null);
  // Unknown ids keep the lab's own values.
  assert.equal(withOverrides(c, { sim: { model: "nope", gpu: "nope" } }).sim.model, c.sim.model);
});

test("hourly costs never use exponent notation", () => {
  assert.equal(usdPerHour(0), "$0");
  assert.equal(usdPerHour(0.0000123), "<$0.001");
  assert.equal(usdPerHour(0.1249), "$0.125");
  assert.equal(usdPerHour(0.3), "$0.300");
  assert.equal(usdPerHour(12.345), "$12.35");
  assert.equal(usdPerHour(-0.51), "−$0.510");
});

test("the per-minute CSV has one row per minute and a column per tier", () => {
  const c = config();
  const r = replay(c, applySettings(c, DEFAULT_SETTINGS));
  const lines = minutesCsv(r).trim().split("\n");
  assert.equal(lines.length, r.minutes.length + 1);
  const header = lines[0].split(",");
  assert.ok(header.includes("gpu_used_gib") && header.includes("shared_readers"));
  assert.ok(lines.slice(1).every(line => line.split(",").length === header.length));
  assert.equal(Number(lines[1].split(",")[1]), r.minutes[0].count);
});

test("turns per chat, warm-up and the workload options reach the replay config", () => {
  const c = config();
  applySettings(c, { ...DEFAULT_SETTINGS, turns: 4, start: "warm", spread: "long", prompts: 10, pattern: "bursty" });
  assert.equal(c.workload.followUpShare, 0.75);
  assert.equal(c.sim.warmupMin, 20);
  assert.equal(c.workload.durationMin, 80, "a warm start still measures 60 minutes");
  assert.equal(c.workload.lengthSpread, 1.5);
  assert.equal(c.workload.systemPrompts, 10);
  assert.equal(c.workload.arrivalPattern, "bursty");
  applySettings(c, { ...DEFAULT_SETTINGS, start: "cold", spread: "fixed" });
  assert.equal(c.sim.warmupMin, 0);
  assert.equal(c.workload.durationMin, 60);
  assert.equal(c.workload.lengthSpread, 0);
});

const base = { requestsPerMin: 600, followUpShare: 0.8, systemTokens: 6000, inputTokens: 800, outputTokens: 400, thinkTimeS: 300, durationMin: 60, seed: 3 };
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

test("varied lengths keep the set means and widen the tail", () => {
  const fixed = synthetic(base).requests, varied = synthetic({ ...base, lengthSpread: 1.5 }).requests;
  assert.ok(fixed.every(r => r.out === 400));
  const outs = varied.map(r => r.out).sort((a, b) => a - b);
  assert.ok(Math.abs(mean(outs) / 400 - 1) < 0.1, `mean reply ${mean(outs)}`);
  assert.ok(outs[Math.floor(outs.length * 0.99)] > 4 * 400, "a long tail");
  // Each turn still extends its conversation's prompt, so the prefix cache stays valid.
  const byConv = new Map();
  for (const r of varied) {
    const prev = byConv.get(r.conv);
    if (prev) assert.ok(r.n > prev.n);
    byConv.set(r.conv, r);
  }
});

test("distinct system prompts follow Zipf popularity; one prompt keeps the original workload", () => {
  assert.deepEqual(synthetic({ ...base, systemPrompts: 1 }).requests, synthetic(base).requests);
  const reqs = synthetic({ ...base, systemPrompts: 10 }).requests.filter(r => !r.followUp);
  const counts = Array(10).fill(0);
  for (const r of reqs) counts[r.sysId]++;
  assert.ok(counts.every(n => n > 0));
  assert.ok(counts[0] > 2 * counts[4] && counts[0] / reqs.length > 0.25 && counts[0] / reqs.length < 0.45, counts.join());
});

test("bursty arrivals keep the hourly volume and concentrate new chats in bursts", () => {
  const steady = synthetic(base).requests.filter(r => !r.followUp);
  const bursty = synthetic({ ...base, arrivalPattern: "bursty" }).requests.filter(r => !r.followUp);
  assert.ok(Math.abs(bursty.length / steady.length - 1) < 0.1);
  const inBurst = bursty.filter(r => r.t % 360 < 120).length / bursty.length;
  assert.ok(inBurst > 0.6 && inBurst < 0.74, `burst share ${inBurst}`);
});

test("a warmed-up replay bins the measured hour from its start and reports serving costs", () => {
  const c = config();
  const r = replay(c, applySettings(c, { ...DEFAULT_SETTINGS, start: "warm" }));
  assert.equal(r.minutes.length, 60);
  assert.equal(r.warmupMin, 20);
  assert.ok(r.minutes[0].count > 0 && r.minutes.every(m => m.usedGiB));
  assert.ok(r.ttft.p50 <= r.ttft.p90 && r.ttft.p90 <= r.ttft.p99);
  assert.ok(r.gpuCostPerHour > 100 && r.prefillGpus === 16 && r.decodeGpus > 0);
  assert.ok(Math.abs(r.gpuCostPerHour - r.prefillCostPerHour - r.decodeCostPerHour) < 1e-9);
  assert.ok(r.utilization > 0 && r.utilization < 1);
});
