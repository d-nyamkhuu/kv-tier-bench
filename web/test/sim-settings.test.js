import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { defaultConfig } from "../config.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";
import { minutesCsv, replay } from "../replay.js";
import { synthetic } from "../sim.js";
import {
  applySettings, cpuSizingNote, decodeSettings, modelDefaults, pricesAt, risingTtft, DEFAULT_SETTINGS, encodeSettings,
  fitCluster, SCENARIOS, scenarioFor, usdPerHour,
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
  assert.ok(gpu.share.cpu > 0.02, "CPU memory larger than the GPU cache serves hits");
  const tiers = run("tiers");
  for (const tier of ["gpu", "cpu", "disk", "p2p", "shared", "miss"]) assert.ok(tiers.share[tier] > 0.01, `the default draws on ${tier}: ${tiers.share[tier]}`);
  assert.equal(cpuSizingNote(DEFAULT_SETTINGS, tiers), null, "the default sizes CPU memory past twice the GPU cache");
  for (const s of SCENARIOS) {
    assert.equal(s.settings.gpu, "b200", `${s.id} runs on B200`);
    assert.equal(fitCluster(config(), s.settings), null, `${s.id} fits its own cluster`);
  }
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
  assert.match(fitCluster(c, { ...big, gpu: "l40s", precision: "released" }).error, /does not fit/); // 4.9 TB in BF16
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

test("the CPU note appears when CPU memory is no larger than the GPU cache and the GPU cache evicts", () => {
  // Kimi K3 FP8 on 2 x 8 H200: one TP8 PP2 instance with 619 GiB of KV, about 310 GiB per node
  const kimi = { ...DEFAULT_SETTINGS, model: "kimi-k3", precision: "fp8", gpu: "h200", cpu: 128, disk: "" };
  const result = { firstEviction: -60, gpuGiB: 619.1, instances: 1, share: { cpu: 0 } };
  assert.equal(cpuSizingNote(kimi, result), "CPU memory is no larger than the GPU cache (310 GiB per node), so it served only 0.0% of tokens. Try 464 GiB or more.");
  assert.doesNotMatch(cpuSizingNote(kimi, { ...result, gpuGiB: 3000 }), /Try/); // past the slider
  assert.equal(cpuSizingNote({ ...kimi, cpu: 384 }, result), null); // larger than the GPU cache serves hits
  assert.equal(cpuSizingNote({ ...kimi, cpu: 0 }, result), null);
  assert.equal(cpuSizingNote(kimi, { ...result, firstEviction: null }), null); // the GPU never filled
});

test("a replay reports progress by simulated minute, rising and below 1", () => {
  const c = config();
  const seen = [];
  replay(c, applySettings(c, SCENARIOS.find(s => s.id === "ample").settings), { onProgress: f => seen.push(f) });
  assert.ok(seen.length >= 60, `${seen.length} progress reports`);
  assert.ok(seen.every((f, i) => f > 0 && f < 1 && (i === 0 || f >= seen[i - 1])));
  assert.ok(seen.at(-1) > 0.95);
});

test("a saturated inter-node link is flagged as rising TTFT, and RDMA with FP8 keeps the default flat", () => {
  const run = settings => { const c = config(); return replay(c, applySettings(c, settings)); };
  // MiniMax-M3 as released (TP8) over TCP 400: decode KV transfers of 15-turn chats outgrow the link
  const tcp = risingTtft(run({ ...DEFAULT_SETTINGS, precision: "released", interconnect: "p2p-tcp-400", cpu: 1536, diskSize: 2048 }));
  assert.ok(tcp && tcp.to > 2 * tcp.from, JSON.stringify(tcp));
  assert.equal(tcp.cause, "transfer");
  assert.equal(risingTtft(run(DEFAULT_SETTINGS)), null);
});

test("the measured period sets the replay length", () => {
  const c = config();
  applySettings(c, { ...DEFAULT_SETTINGS, hours: 3 });
  assert.equal(c.workload.durationMin, 180 + 20);
});

test("prices scale the prefill-GPU part to the chosen busy share and keep cache tiers", () => {
  const prices = { input: 2, cachedInput: 0.5, cachedPrefill: 0.4 };
  const at60 = pricesAt(prices, 0.15, "0.6");
  assert.ok(Math.abs(at60.input - 0.5) < 1e-12);
  assert.ok(Math.abs(at60.cachedInput - (0.1 + 0.1)) < 1e-12, "0.4 x 0.25 of prefill plus 0.1 of cache tiers");
  assert.deepEqual(pricesAt(prices, 0.15, "actual"), { input: 2, cachedInput: 0.5, target: null, utilization: 0.15 });
  assert.equal(pricesAt(null, 0.15, "0.6"), null);
});

test("each model's serving defaults: FP8 when released in BF16, RDMA between nodes", () => {
  const c = config();
  assert.deepEqual(modelDefaults(c.models["minimax-m3"]), { precision: "fp8", interconnect: "p2p-ib-ndr" });
  assert.equal(modelDefaults(c.models["deepseek-v4.1-flash"]).precision, "released");
  for (const s of SCENARIOS) assert.equal(s.settings.interconnect, "p2p-ib-ndr", s.id);
});

test("every scenario value is a stop on its slider, so the page shows the scenario exactly", () => {
  const src = readFileSync(new URL("../simulator.js", import.meta.url), "utf8");
  const stops = name => JSON.parse(src.match(new RegExp(`const ${name} = (\\[[^\\]]*\\])`))[1].replace("CPU_SLIDER_MAX_GIB", "2048"));
  const cpu = stops("cpuSizes"), disk = stops("diskSizes");
  for (const s of SCENARIOS) {
    assert.ok(cpu.includes(s.settings.cpu), `${s.id} CPU ${s.settings.cpu}`);
    assert.ok(disk.includes(s.settings.diskSize), `${s.id} SSD ${s.settings.diskSize}`);
  }
});
