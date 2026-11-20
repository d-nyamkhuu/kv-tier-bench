import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { defaultConfig } from "../config.js";
import { instanceWeightBytes, served } from "../model.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";
import { prepareTrace, replay } from "../replay.js";
import { compareTiers } from "../tier-comparison.js";
import { runScenario } from "../scenario.js";
import { parseTrace } from "../sim.js";

const s1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const setup = () => {
  const c = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE);
  Object.assign(c.workload, { requestsPerMin: 120, thinkTimeS: 300 });
  Object.assign(c.sim, { cpuGiB: 128, warmupMin: 0 });
  return c;
};

test("minute bins reconstruct request-weighted mean TTFT and cache-only hourly spend", () => {
  const c = setup();
  const expected = runScenario(c);
  const r = replay(c);
  const bins = r.minutes.filter(Boolean);
  assert.equal(bins.reduce((sum, m) => sum + m.count, 0), expected.requests);
  const mean = bins.reduce((sum, m) => sum + m.mean * m.count, 0) / expected.requests;
  assert.ok(Math.abs(mean - expected.ttft.mean) < 1e-10);
  assert.equal(r.costPerHour, (expected.usd.storage + expected.usd.fixed) * 3600 / expected.windowS);
  assert.ok(r.costPerHour > 0 && r.firstEviction > 0 && r.share.shared > 0);
  c.sim.shared = null;
  assert.equal(replay(c).costPerHour, 0);
});

test("shrinking the GPU budget forces earlier evictions and changes the observed token mix", () => {
  const c = setup();
  c.sim.routing = "sticky";
  const full = replay(c), small = replay(c, 0.1);
  assert.ok(small.firstEviction < full.firstEviction);
  assert.ok(small.share.gpu < full.share.gpu);
  assert.ok(small.mean > full.mean);
});

test("backing reads start after write-through storage receives its first data and report the TTFT of actual readers", () => {
  const c = setup();
  const r = replay(c);
  const first = r.minutes.findIndex(m => m.reads.shared);
  assert.equal(first, 0); // Round-robin can read a prefix written by another node in the same minute.
  assert.ok(r.minutes[0].usedGiB.shared > 0);
  assert.equal(r.tiers.shared, null);
  assert.ok(r.minutes[first].readerTTFT.shared > 0 && r.minutes[first].load > 0);
  for (const m of r.minutes.filter(m => !m.reads.shared)) assert.equal(m.readerTTFT.shared, undefined);
  for (const m of r.minutes) {
    assert.ok(m.usedGiB.gpu <= r.tiers.gpu && m.usedGiB.cpu <= r.tiers.cpu);
    assert.ok(m.load + m.wait <= m.mean);
    assert.ok((m.reads.shared ?? 0) <= m.count);
  }
});

test("idle minutes carry occupancy forward without inventing reads or latency", () => {
  const c = setup();
  Object.assign(c.workload, { requestsPerMin: 0.25, followUpShare: 0 });
  Object.assign(c.sim, { cpuGiB: 0, shared: null });
  const r = replay(c);
  assert.deepEqual(Object.keys(r.tiers), ["gpu"]);
  const idle = r.minutes.findIndex((m, i) => i > 0 && !m.count && r.minutes[i - 1].count);
  assert.ok(idle > 0);
  assert.deepEqual(r.minutes[idle].usedGiB, r.minutes[idle - 1].usedGiB);
  assert.deepEqual(r.minutes[idle].reads, {});
  assert.deepEqual(r.minutes[idle].readerStats, {});
  assert.equal(r.minutes[idle].mean, null);
  assert.equal(r.minutes[0].usedGiB.gpu, 0);
});

test("per-tier statistics attribute each request to its deepest tier read", () => {
  const c = setup();
  c.models[c.sim.model].read_fixed_bytes = 0;
  Object.assign(c.workload, { durationMin: 5, thinkTimeS: 10 });
  const r = replay(c);
  const minuteTimes = r.minutes.map(() => []);
  const samples = [];
  const tierRank = ["gpu", "cpu", "disk", ...(c.sim.peerFirst ? ["p2p", "shared"] : ["shared", "p2p"])];
  let mixed = 0;
  runScenario(c, r.layout, null, { observeRequest(req, ttft, blocks, size) {
    const tiers = Object.keys(blocks).filter(key => blocks[key] > 0);
    if (tiers.length > 1) mixed++;
    let deepest = null;
    for (const t of tierRank) if (blocks[t] > 0) deepest = t;
    if (deepest) {
      const bytes = deepest === "gpu" ? 0 : blocks[deepest] * size * r.kvBytesPerToken;
      const hit = { tier: deepest, ttft, mib: bytes / 2 ** 20 };
      samples.push(hit);
      minuteTimes[Math.floor(req.t / 60)].push(hit);
    }
  } });
  assert.ok(mixed > 0);
  function check(stats, hits) {
    for (const tier of r.order) {
      const subset = hits.filter(hit => hit.tier === tier);
      if (!subset.length) { assert.equal(stats[tier], undefined); continue; }
      const sorted = metric => subset.map(s => s[metric]).sort((a, b) => a - b);
      const times = sorted("ttft"), bytes = sorted("mib"), n = subset.length;
      const median = xs => (xs[Math.floor((n - 1) / 2)] + xs[Math.floor(n / 2)]) / 2;
      assert.deepEqual(stats[tier], { count: n, medianTTFT: median(times), medianMiB: median(bytes) });
    }
  }
  check(r.readerStats, samples);
  r.minutes.forEach((m, i) => check(m.readerStats, minuteTimes[i]));
});

test("the inter-node link changes KV handoff and TTFT on the same workload", () => {
  const c = setup();
  const fast = replay(c);
  c.sim.transfer = c.sim.stageLink = "p2p-tcp-10";
  const slow = replay(c);
  assert.ok(slow.mean > fast.mean);
  assert.ok(slow.minutes[0].transfer > fast.minutes[0].transfer);
  assert.equal(slow.tiers.gpu, fast.tiers.gpu);
  assert.equal(slow.tiers.cpu, fast.tiers.cpu);
});

test("fixed inventory uses pipeline stages when needed and measures every selected node", () => {
  const c = setup();
  Object.assign(c.sim, { nodes: 1, gpusPerNode: 1, gpu: "l40s", decodeGpu: "l40s" });
  assert.throws(() => replay(c), /does not fit this cluster/);
  c.sim.nodes = 12;
  const r = replay(c);
  assert.deepEqual(r.layout, { tp: 1, pp: 2 });
  assert.equal(r.instances, 6);
  assert.equal(r.tiers.cpu, 12 * c.sim.cpuGiB);
  assert.equal(r.tiers.gpu, r.instances * r.gpuGiB);
  c.sim.stageLink = "p2p-tcp-10";
  assert.ok(replay(c).mean > r.mean);
});

test("local disk and S3 coexist, bill independently, and serve nearest-first", () => {
  const c = setup();
  Object.assign(c.sim, { disk: "fs-nvme-gen5", gpu: "rtx-pro-6000", shared: null, cpuGiB: 0 });
  const diskOnly = replay(c, 0.1);
  c.sim.shared = "obj-aws-s3-standard";
  const both = replay(c, 0.1);
  c.sim.disk = null;
  const s3Only = replay(c, 0.1);
  assert.ok(both.tiers.disk > 0 && both.tiers.shared === null);
  assert.ok(both.share.disk > 0 && both.share.shared < s3Only.share.shared);
  assert.ok(both.minutes.at(-1).usedGiB.disk > 0 && both.minutes.at(-1).usedGiB.shared > 0);
  assert.ok(both.s3Requests.get < s3Only.s3Requests.get);
  const sharedRate = c.variants.find(v => v.id === c.sim.shared).usd_per_gib_month / 730;
  const storedCost = r => r.minutes.at(-1).usedGiB.shared * sharedRate * c.sim.retentionH;
  // The billing window includes work completed for its arrivals after minute 60.
  // Zero the object price to isolate the independently provisioned disk exactly.
  c.variants.find(v => v.id === c.sim.shared).usd_per_gib_month = 0;
  c.sim.disk = "fs-nvme-gen5";
  const noRent = replay(c, 0.1);
  assert.ok(Math.abs(noRent.costPerHour - noRent.requestCostPerHour - diskOnly.costPerHour) < 1e-10);
  assert.ok(both.costPerHour - noRent.costPerHour >= storedCost(both) - 1e-10);
});

test("per-node SSD sizing changes actual evictions, S3 reads and the provisioned disk bill", () => {
  const c = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, MODELS);
  Object.assign(c.workload, { requestsPerMin: 600, thinkTimeS: 300 });
  Object.assign(c.sim, { model: "deepseek-v4.1-flash", gpu: "b200", decodeGpu: "b200",
    disk: "fs-nvme-gen5", diskCapacityGiB: 4, gpusPerNode: 2, cpuGiB: 0, warmupMin: 0 });
  const disk = c.variants.find(v => v.id === c.sim.disk);
  c.variants.find(v => v.id === c.sim.shared).usd_per_gib_month = 0; // isolate disk rent and S3 request fees
  const small = replay(c, 0.05);
  c.sim.diskCapacityGiB = 8;
  const large = replay(c, 0.05);
  assert.equal(large.tiers.disk, 8 * c.sim.nodes);
  assert.ok(small.firstDiskEviction < large.firstDiskEviction && small.firstDiskEviction > 0);
  assert.ok(small.share.shared > large.share.shared && large.share.disk > small.share.disk);
  assert.ok(small.mean > large.mean);
  const extraDiskCost = 4 * c.sim.nodes * disk.usd_per_gib_month / 730;
  assert.ok(Math.abs((large.costPerHour - large.requestCostPerHour) - (small.costPerHour - small.requestCostPerHour) - extraDiskCost) < 1e-10);
  c.sim.diskCapacityGiB = null;
  const full = replay(c, 0.05);
  assert.equal(full.firstDiskEviction, null);
  assert.equal(full.tiers.disk, disk.capacity_gib * c.sim.nodes);
});

function deepSeekCluster(gpusPerNode) {
  const c = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, MODELS);
  Object.assign(c.workload, { requestsPerMin: 600, thinkTimeS: 300 });
  Object.assign(c.sim, { model: "deepseek-v4.1-flash", gpu: "b200", decodeGpu: "b200",
    gpusPerNode, cpuGiB: 128, disk: "fs-nvme-gen5", diskCapacityGiB: 8, warmupMin: 0,
    p2p: c.sim.transfer, stageLink: c.sim.transfer });
  return c;
}

test("minute 49 counts each prefill replica's KV headroom once across pipeline stages", () => {
  for (const [gpusPerNode, replicas] of [[2, 1], [4, 2], [8, 4]]) {
    const c = deepSeekCluster(gpusPerNode);
    const r = replay(c);
    const gpu = c.gpus.find(g => g.id === "b200");
    const gpuCount = c.sim.nodes * gpusPerNode;
    const usable = gpuCount * (gpu.mem_gib * c.software.gpu_memory_utilization - c.software.activation_gib_per_gpu);
    const servedModel = served(c.models[c.sim.model], { gpu, precision: c.sim.precision, tp: r.layout.tp, attention: r.layout.attention ?? c.sim.attention });
    const weights = replicas * instanceWeightBytes(servedModel, r.layout.tp) / 2 ** 30;
    assert.equal(r.instances, replicas);
    assert.ok(Math.abs(r.tiers.gpu - (usable - weights)) < 1e-8);
    assert.equal(r.kvBytesPerToken, (528 + 132) * (3 / 2 + 1));
    assert.ok(r.minutes[48].count > 0);
    // Every retained chunk has exactly one fixed-size global payload per KV copy.
    const used = r.minutes[48].usedGiB.gpu * 2 ** 30;
    assert.ok(used > 0 && used <= r.tiers.gpu * 2 ** 30);
    assert.equal(used % (256 * 1650), 0);
    assert.equal(r.order.includes("p2p"), gpusPerNode > 2);
  }
});

test("disk pressure on the default hardware matches the same topology entered directly: TTFT, tier hits and storage bill", () => {
  const defaults = setup();
  assert.equal(defaults.sim.gpusPerNode, 8);
  const c = deepSeekCluster(defaults.sim.gpusPerNode);
  c.sim.cpuGiB = 4;
  const r = replay(c, 0.01);
  // Enter the equivalent topology, disk size, and GPU headroom directly.
  const lab = structuredClone(c);
  Object.assign(lab.sim, { tp: 4, pp: 1, diskCapacityGiB: null, gpuBudgetFraction: 0.01 });
  lab.variants.find(v => v.id === lab.sim.disk).capacity_gib = 8;
  const audit = runScenario(lab);
  assert.deepEqual(r.layout, { tp: lab.sim.tp, pp: lab.sim.pp });
  assert.deepEqual(r.order, ["gpu", "cpu", "disk", "shared", "p2p", "miss"]);
  assert.deepEqual(r.share, audit.share);
  for (const key of ["gpu", "disk", "shared", "miss"]) assert.ok(r.share[key] > 0, key);
  assert.ok(r.share.p2p < 0.001, `stock S3-first keeps peer hits near zero: ${r.share.p2p}`);
  assert.equal(r.mean, audit.ttft.mean);
  assert.equal(r.gpuGiB, audit.kvGiB);
  assert.equal(r.wait, audit.wait);
  assert.equal(r.costPerHour, (audit.usd.storage + audit.usd.fixed) * 3600 / audit.windowS);
  const firstS3 = r.minutes.findIndex(m => m.reads.shared);
  assert.ok(firstS3 >= 0 && r.firstDiskEviction > 0);
  assert.ok(r.minutes[firstS3].readerTTFT.shared > r.minutes[firstS3].mean);
  // Peer hits use CPU capacity already accounted for on the source node.
  assert.equal(r.tiers.cpu, 8);
  assert.equal(r.tiers.p2p, undefined);
});

test("DeepSeek packed KV distinguishes GPU replicas, live windows and serialized objects", () => {
  for (const [gpu, row] of [["b200", 528], ["h100", 584]]) {
    const c = deepSeekCluster(8);
    Object.assign(c.sim, { gpu, decode: false, tp: 8, attention: "tp", nodes: 1 });
    // On H100 the released checkpoint may need two stages for the configured reserve.
    if (gpu === "h100") Object.assign(c.sim, { nodes: 2, pp: 2 });
    const chunks = 3;
    const trace = { blockTokens: 256, requests: [{ t: 0, n: 256 * chunks, hashes: [1, 2, 3] }] };
    let snapshot;
    const r = runScenario(c, {}, trace, { observeMinuteEnd: (_, state) => { snapshot = state; } });
    assert.ok(!r.error, r.error);
    const global = chunks * 256 * (row + 132) * (3 / 2 + 1);
    assert.equal(snapshot.gpu.usedGiB * 2 ** 30, 8 * global);
    assert.equal(snapshot.shared.usedGiB * 2 ** 30, global);
    assert.deepEqual(r.s3Requests, { put: chunks * c.sim.pp, get: 0, head: 0 });
    assert.equal(r.windowBytes, 8 * 128 * 40 * row);
    assert.equal(r.replayTokens, 128);
    const express = runScenario(c, { shared: "obj-aws-s3-express" }, trace);
    assert.equal(express.requestUsd, (chunks * c.sim.pp * .00113) / 1000);
  }
});

test("automatic layout accounts for the active sequence and respects the chosen inventory", () => {
  const c = setup();
  Object.assign(c.sim, { model: "qwen3-vl-32b", gpu: "h100", decodeGpu: "h100", nodes: 1, gpusPerNode: 2 });
  Object.assign(c.workload, { systemTokens: 62000, inputTokens: 1000, outputTokens: 1000,
    durationMin: 1, followUpShare: 0, requestsPerMin: 10 });
  assert.deepEqual(replay(c).layout, { tp: 2, pp: 1 });
  c.sim.gpusPerNode = 1;
  assert.throws(() => replay(c), /63000-token sequence exceeds/);
});

test("tier rows report the median TTFT of requests reading that tier and on-demand prices per GiB-month", () => {
  const c = setup();
  Object.assign(c.workload, { durationMin: 2 });
  Object.assign(c.sim, { nodes: 12, disk: "fs-nvme-gen4", diskCapacityGiB: 4, shared: null, p2p: null });
  const result = replay(c, 0.1);
  const progress = [];
  const rows = compareTiers(c, result.layout, 0.1, { onRow: (partial, done, total) => progress.push([partial.length, done, total]) });
  assert.equal(progress.length, rows.length);
  assert.deepEqual(progress[0], [1, 1, rows.length]);
  assert.deepEqual(progress.at(-1), [rows.length, rows.length, rows.length]);
  const ssd = rows.find(row => row.name === "NVMe Gen4");
  assert.equal(ssd.mean, result.mean, "the SSD row replays the live setup");
  assert.equal(ssd.readers, result.readerStats.disk?.count ?? 0);
  if (ssd.readers) assert.equal(ssd.median, result.readerStats.disk.medianTTFT);
  assert.equal(ssd.rate, 0.065);
  const gpu = c.gpus.find(g => g.id === c.sim.gpu);
  assert.equal(rows.find(row => row.name === "GPU only").rate, gpu.usd_per_hour * 730 / gpu.mem_gib);
  assert.equal(rows.find(row => row.name === "Local CPU").rate, 3.285);
  assert.equal(rows.find(row => row.name === "S3 Standard").rate, 0.023);
  for (const row of rows) {
    assert.ok(row.readers <= row.requests);
    assert.ok(row.readers ? row.median > 0 && !row.error : row.median == null && row.error);
  }
});

test("peer comparison needs CPU cache outside the local pipeline group", () => {
  const c = setup();
  c.workload.durationMin = 1;
  c.sim.p2p = "p2p-tcp-400";
  const names = layout => compareTiers(c, layout, 1).map(row => row.name);
  assert.ok(names({ tp: 1, pp: 1 }).includes("Peer CPU"));
  assert.ok(!names({ tp: 1, pp: 2 }).includes("Peer CPU"));
  c.sim.cpuGiB = 0;
  const noCpu = names({ tp: 1, pp: 1 });
  assert.ok(!noCpu.includes("Local CPU") && !noCpu.includes("Peer CPU"));
});

test("a cache lifetime expires blocks unused for longer than it, so pauses past it miss", () => {
  const c = setup();
  Object.assign(c.workload, { durationMin: 20, followUpShare: 0.8, thinkTimeS: 300 });
  const hit = lifetime => { c.sim.cacheLifetimeS = lifetime; return 1 - runScenario(c).share.miss; };
  const kept = hit(null), hour = hit(3600), short = hit(60);
  assert.ok(Math.abs(kept - hour) < 0.01, `${kept} vs ${hour}`);
  assert.ok(short < kept - 0.1, `a 1-minute lifetime against 5-minute pauses: ${short} vs ${kept}`);
});

test("prompts shorter than the minimum cached length are neither looked up nor saved", () => {
  const c = setup();
  Object.assign(c.workload, { durationMin: 10, followUpShare: 0, systemTokens: 2000, inputTokens: 200 });
  c.sim.minCacheTokens = 0;
  assert.ok(runScenario(c).share.miss < 0.5);
  c.sim.minCacheTokens = 4096;
  assert.equal(runScenario(c).share.miss, 1);
});

test("replay and compareTiers support imported JSONL traces with non-zero start timestamps and short spans", () => {
  const raw = [
    { timestamp: 600_000, input_length: 1024, output_length: 128, hash_ids: [10, 11] },
    { timestamp: 630_000, input_length: 1536, output_length: 128, hash_ids: [10, 11, 12] },
    { timestamp: 720_000, input_length: 2048, output_length: 128, hash_ids: [10, 11, 12, 13] },
  ].map(x => JSON.stringify(x)).join("\n");
  const trace = parseTrace(raw, 512, "unit-trace");
  // Short 2-minute trace clamps a 20-minute warm-up to 0 so all arrivals are measured.
  const prep = prepareTrace(trace, 20);
  assert.equal(prep.warmupMin, 0);
  assert.equal(prep.trace.requests[0].t, 0);
  assert.equal(prep.totalMin, 3);
  const c = setup();
  c.sim.warmupMin = 20;
  const r = replay(c, 1, { trace });
  assert.equal(r.requests, 3);
  assert.ok(r.share.gpu > 0 || r.share.cpu > 0 || r.share.shared > 0);
  assert.ok(r.tpot > 0 && r.decodeUtilization >= 0);
  const rows = compareTiers(c, r.layout, 1, { trace });
  assert.ok(rows.find(row => row.name === "GPU only").requests === 3);
});

test("a context that outgrows one DP-attention worker falls back to TP-attention on the same layout", () => {
  const c = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, MODELS);
  Object.assign(c.workload, { requestsPerMin: 30, followUpShare: 1 - 1 / 30, thinkTimeS: 60, systemTokens: 6000, inputTokens: 3000, outputTokens: 1000, durationMin: 40 });
  Object.assign(c.sim, { model: "qwen3-235b-a22b", gpu: "h200", decodeGpu: "h200", precision: "released", attention: "dp", nodes: 2, gpusPerNode: 8,
    cpuGiB: 128, shared: null, warmupMin: 0, transfer: "p2p-ib-ndr" });
  const r = replay(c);
  assert.deepEqual(r.layout, { tp: 4, pp: 1, attention: "tp" });
  assert.ok(r.requests > 0);
});
