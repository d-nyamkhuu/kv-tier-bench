import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { GIB } from "../model.js";
import { GPUS, SOFTWARE } from "../presets.js";
import { idealHitShare, LRU, parseTrace, simulate, synthetic } from "../sim.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));

test("LRU evicts the least recently used entry, by bytes", () => {
  const l = new LRU(3);
  for (const k of ["a", "b", "c"]) l.put(k, 1);
  l.put("a", 1); // refresh a: b is now the oldest
  l.put("d", 1);
  assert.deepEqual(l.keys(), ["c", "a", "d"]);
  l.put("big", 2);
  assert.deepEqual(l.keys(), ["d", "big"]);
  assert.equal(l.used, 3);
  assert.equal(l.evictions, 3);
  const off = new LRU(0);
  off.put("x", 1);
  assert.equal(off.has("x"), false);
});

// A toy model: 1 block = 100 tokens = 100 bytes; no attention; on a 1 TFLOPS GPU, prefill costs 1 s per 1000 tokens.
// Overhead 0.01 s per request; a CPU load adds 0.01 s (the host link is effectively infinite).
const TOY = { block_size: 100, kv_bytes_per_token: 1, read_fixed_bytes: 0, params: 5e8, full_attn_layers: 0, attn_width: 0, weights_bytes: 0, state_bytes_per_prefix: 0 };
const SW = { request_overhead_s: 0.01, per_token_overhead_s: 0, attn_eff_ratio: 1, cpu_load_latency_s: 0.01, gpu_memory_utilization: 1, activation_gib_per_gpu: 0 };
// 4 prefixes of 10 blocks, visited round-robin, 10 s apart (no queueing): 4000 bytes of working set
const roundRobin = { blockTokens: null, requests: Array.from({ length: 40 }, (_, i) => ({ t: 10 * i, n: 1000, hashes: Array.from({ length: 10 }, (_, j) => `p${i % 4}:${j}`) })) };
const GPU = { usd_per_hour: 3.6, tflops_bf16: 1, prefill_mfu: 1, host_gib_s: 1e9 };
const base = { model: TOY, gpu: GPU, software: SW };
const gib = (bytes) => bytes / GIB;

test("GPU+CPU beats GPU only when prefixes repeat beyond GPU capacity", () => {
  const gpuOnly = simulate(roundRobin, { ...base, gpuGiB: gib(2500), cpuGiB: 0 });
  const withCpu = simulate(roundRobin, { ...base, gpuGiB: gib(2500), cpuGiB: gib(6000) });
  assert.equal(gpuOnly.share.gpu, 0); // LRU with a cyclic working set larger than the cache: every access misses
  assert.equal(gpuOnly.share.miss, 1);
  assert.ok(withCpu.share.cpu > 0.85, `cpu share ${withCpu.share.cpu}`);
  assert.ok(withCpu.ttft.mean < gpuOnly.ttft.mean / 5);
  // the CPU tier is written through, so it never misses what GPU evicted
  const bigGpu = simulate(roundRobin, { ...base, gpuGiB: gib(5000), cpuGiB: gib(6000) });
  assert.equal(bigGpu.share.cpu, 0);
  assert.ok(bigGpu.share.gpu > 0.85);
});

test("local disk takes priority over shared storage", () => {
  const fs = { family: "fs", bw_gib_s: 1000 / GIB, latency_s: 0.48, capacity_gib: gib(10000) }; // 1000 bytes in 1 s
  const r = simulate(roundRobin, { ...base, gpuGiB: gib(1000), cpuGiB: gib(2000), disk: fs });
  assert.equal(r.share.gpu + r.share.cpu, 0);
  assert.equal(r.share.disk, 0.9); // the first visit of each of the 4 prefixes misses
  assert.equal(idealHitShare(roundRobin, 100).share, 0.9); // an unbounded tier reaches the upper bound
  assert.ok(Math.abs(r.ttft.p50 - (0.01 + 0.01 + 0.48 + 1)) < 1e-9); // overhead + CPU staging + disk latency + bytes / BW
  // both hold everything: the local disk is asked first even when shared storage is faster
  const fast = { family: "obj", bw_gib_s: 100, latency_s: 0, shared: true };
  const both = simulate(roundRobin, { ...base, gpuGiB: gib(1000), cpuGiB: gib(2000), disk: fs, shared: fast });
  assert.deepEqual(both.order, ["gpu", "cpu", "disk", "shared", "miss"]);
  assert.equal(both.share.disk, 0.9);
  assert.equal(both.firstPayloadReadS.disk, 40 + fs.latency_s);
  assert.equal(both.firstPayloadReadS.shared, null);
});

test("first physical SSD read waits for a busy drive even across minute boundaries", () => {
  const disk = { family: "fs", bw_gib_s: gib(100), latency_s: 0.2 };
  const requests = [
    { t: 0, n: 100, hashes: ["a"] },
    { t: 24, n: 4000, hashes: Array.from({ length: 40 }, (_, i) => "b" + i) },
    { t: 59, n: 100, hashes: ["a"] },
  ];
  const r = simulate({ requests }, { ...base, gpuGiB: 0, cpuGiB: 0, disk, durationMin: 1 });
  // The second write occupies the drive until 24 + 4.01 prefill + .2 setup + 40 transfer.
  assert.ok(Math.abs(r.firstPayloadReadS.disk - 68.21) < 1e-8);
});

test("restored payload includes the prefix checkpoint once and accounts for external copies", () => {
  const cfg = { ...base, model: { ...TOY, read_fixed_bytes: 40 }, gpuGiB: 0, cpuGiB: 1,
    externalCopiesPerGpuCopy: 0.5 };
  const requests = [0, 10].map(t => ({ t, n: 200, hashes: ["a", "b"] }));
  let bytes;
  simulate({ requests }, cfg, (req, ttft, blocks, size, loads, wait, transfer, peer, restored) => {
    if (blocks.cpu) bytes = restored("cpu");
  });
  assert.equal(bytes, (2 * 100 + 40) / 2);
});

test("with the published figures, S3 loads slower than a local NVMe drive, S3 Express in between", async () => {
  const { VARIANTS } = await import("../presets.js");
  const { timing } = await import("../model.js");
  const m = S1.models["qwen3-vl-32b"];
  const T = timing(m, GPUS[0], SOFTWARE, 1);
  const v = (id) => VARIANTS.find((x) => x.id === id);
  for (const k of [2048, 32768]) {
    const [nvme, express, s3] = ["fs-nvme-gen5", "obj-aws-s3-express", "obj-aws-s3-standard"].map((id) => T.loadFor(v(id))(k));
    assert.ok(nvme < express && express < s3, `${k} tokens: NVMe ${nvme}, Express ${express}, S3 ${s3}`);
  }
});

test("p2p hits come from another node's CPU tier; a shared tier serves every node", () => {
  const conv = { blockTokens: null, requests: [0, 1, 2, 3].map((i) => ({ t: 10 * i, n: 1000, conv: i, hashes: Array.from({ length: 10 }, (_, j) => `x:${j}`) })) };
  const cfg = { ...base, gpuGiB: gib(5000), cpuGiB: gib(6000), routing: "sticky", cluster: { nodes: 2, gpusPerNode: 1, tp: 1 } };
  const link = { family: "p2p", bw_gib_s: 1, latency_s: 0 };
  const p2p = simulate(conv, { ...cfg, p2p: link });
  assert.equal(p2p.tokens.p2p, 1000); // request 1 (node 1) pulls from node 0; later ones hit locally
  assert.equal(simulate(conv, { ...cfg, cluster: { nodes: 1 }, p2p: link }).tokens.p2p, 0);
  // two GPUs in one node share its CPU tier: the second instance hits CPU, not p2p
  const oneNode = simulate(conv, { ...cfg, cluster: { nodes: 1, gpusPerNode: 2, tp: 1 }, p2p: link });
  assert.equal(oneNode.tokens.cpu, 1000);
  const obj = simulate(conv, { ...cfg, cpuGiB: 0, shared: { family: "obj", bw_gib_s: 1, latency_s: 0.1, shared: true } });
  assert.equal(obj.tokens.shared, 1000);
});

test("a busy instance queues requests, and provisioned GPUs are paid for whether busy or not", () => {
  const burst = { blockTokens: null, requests: [0, 0.1, 0.2].map((t, i) => ({ t, n: 1000, hashes: [`b${i}`] })) };
  const r = simulate(burst, { ...base, gpuGiB: 0, cpuGiB: 0 });
  assert.deepEqual([r.wait > 0, r.ttft.p90 > 2.5], [true, true]); // the third waits ~1.8 s behind two ~1 s recomputes
  const two = simulate(burst, { ...base, gpuGiB: 0, cpuGiB: 0, cluster: { nodes: 1, gpusPerNode: 2, tp: 1 }, routing: "sticky" });
  assert.equal(two.prefillGpus, 2);
  const slow = { blockTokens: null, requests: [0, 3600].map((t, i) => ({ t, n: 1000, hashes: [`s${i}`] })) };
  const idle = simulate(slow, { ...base, gpuGiB: 0, cpuGiB: 0, cluster: { nodes: 2, gpusPerNode: 4, tp: 1 } });
  assert.ok(Math.abs(idle.usd.prefill - 8 * 3.6 * (3600 + 1.01) / 3600) < 1e-9); // 8 GPUs x 1 h, though they served two requests
  assert.ok(idle.utilization < 0.001);
});

test("replaying a cold run, a GPU hit and a CPU hit of every s1 length lands within 20% of what s1 measured", () => {
  const rtx = GPUS.find((g) => g.id === "rtx-pro-6000");
  for (const [name, m] of Object.entries(S1.models)) {
    for (const tier of ["recompute", "gpu", "cpu"]) {
      for (const row of tier === "recompute" ? m.recompute : m.tiers[tier]) {
        const hashes = Array.from({ length: row.tokens / m.block_size }, (_, j) => j);
        const w = { blockTokens: m.block_size, requests: [0, 1000].map((t, i) => ({ t, n: row.tokens, conv: i, hashes })) };
        const r = simulate(w, { model: m, gpu: rtx, software: SOFTWARE, routing: "sticky", gpuGiB: tier === "gpu" ? 1000 : 0, cpuGiB: tier === "cpu" ? 1000 : 0 });
        const cold = m.recompute.find((x) => x.tokens === row.tokens).ttft_s;
        const got = tier === "recompute" ? r.ttft.p50 : r.ttft.mean * 2 - Math.max(r.ttft.p50, r.ttft.p90); // the second request
        assert.ok(Math.abs(got / row.ttft_s - 1) < 0.2, `${name} ${tier} ${row.tokens}: ${got} vs ${row.ttft_s} (cold ${cold})`);
      }
    }
  }
});

test("synthetic load: the follow-up share holds, prompts grow by message + answer, and answers are cached", () => {
  const big = synthetic({ requestsPerMin: 600, followUpShare: 0.7, systemTokens: 1000, inputTokens: 300, outputTokens: 200, thinkTimeS: 30, durationMin: 600, seed: 3 });
  const share = big.requests.filter((r) => r.followUp).length / big.requests.length;
  assert.ok(Math.abs(share - 0.7) < 0.01, `follow-up share ${share}`);
  assert.ok(Math.abs(big.requests.length / 6000 / 60 - 1) < 0.05, "requests per minute");
  const turns = new Map();
  for (const r of big.requests) turns.set(r.conv, [...(turns.get(r.conv) ?? []), r.n]);
  for (const ns of [...turns.values()].slice(0, 50)) ns.forEach((n, k) => assert.equal(n, 1000 + k * 500 + 300));

  // one conversation of two turns: the second hits the system prompt, the first message and its answer
  const w = { blockTokens: null, requests: [
    { t: 0, n: 1300, out: 200, conv: 0, sys: 1000 },
    { t: 1, n: 1800, out: 200, conv: 0, sys: 1000, followUp: true },
  ] };
  const r = simulate(w, { ...base, gpuGiB: 1, cpuGiB: 0 });
  assert.equal(r.tokens.gpu, 1500); // blocks of 100 tokens: 1000 + 300 + 200
  assert.equal(r.outputTokens, 400);
  const capped = synthetic({ requestsPerMin: 60, followUpShare: 0.99, systemTokens: 0, inputTokens: 1000, outputTokens: 1000, thinkTimeS: 1, durationMin: 10, maxContextTokens: 8000, seed: 1 });
  assert.ok(Math.max(...capped.requests.map((q) => q.n + q.out)) <= 8000);
  const t = parseTrace('{"timestamp": 2000, "input_length": 1100, "output_length": 5, "hash_ids": [1, 2, 3]}\n{"timestamp": 0, "input_length": 600, "output_length": 5, "hash_ids": [1, 9]}\n', 512);
  assert.deepEqual(t.requests.map((q) => q.t), [0, 2]);
  assert.throws(() => parseTrace('{"timestamp": 0}'), /hash_ids/);
});

test("GPU scaling: compute follows peak x prefill_mfu x TP; overhead does not; KV capacity", async () => {
  const { gpuKvGiB, timing } = await import("../model.js");
  const m = S1.models["qwen3-vl-32b"];
  const rtx = GPUS.find((g) => g.id === "rtx-pro-6000");
  for (const mm of Object.values(S1.models)) assert.ok(Math.abs(gpuKvGiB(mm, rtx, SOFTWARE) - mm.gpu_kv_gib) < 0.01); // s1's own KV budget
  const base = timing(m, rtx, SOFTWARE);
  const fast = timing(m, { ...rtx, tflops_bf16: 1000 }, SOFTWARE);
  assert.ok(Math.abs(fast.compute(16384) / base.compute(16384) - 0.5) < 1e-9);
  assert.equal(fast.overhead(16384), base.overhead(16384));
  assert.ok(Math.abs(base.ttft(16384, 16384, 0) - base.overhead(16384)) < 1e-12); // a full hit costs only the overhead
  const half = timing(m, { ...rtx, host_gib_s: rtx.host_gib_s / 2 }, SOFTWARE);
  assert.ok(half.cpuLoad(16384) > base.cpuLoad(16384) * 1.9);
  assert.ok(gpuKvGiB(m, { mem_gib: 48 }, SOFTWARE) < 0); // a 48 GB card cannot hold qwen3-vl-32b's weights
});

test("per-GB transfer fees are charged on what is written to and read from shared storage", () => {
  const express = { family: "obj", bw_gib_s: 1, latency_s: 0, shared: true, bill: "stored", usd_per_gib_month: 0, usd_per_gib_written: 1, usd_per_gib_read: 2 };
  const r = simulate(roundRobin, { ...base, gpuGiB: 0, cpuGiB: 0, shared: express });
  const written = 4 * 1000, read = r.tokens.shared; // 4 prefixes of 1000 bytes stored once; 1 byte per token
  assert.ok(Math.abs(r.usd.storage - (written + 2 * read) / GIB) < 1e-15, `${r.usd.storage}`);
});

test("disaggregated decode: batched steps read weights once and each sequence's KV; TTFT adds transfer and one step", async () => {
  const { decodeStepS } = await import("../model.js");
  const dgpu = { mem_bw_tb_s: 1, mbu: 0.5, usd_per_hour: 3.6 };
  const bare = { ...TOY, weights_bytes: 0, state_bytes_per_prefix: 0 };
  assert.equal(decodeStepS(bare, dgpu, 4, 1000), (4 * 1000) / 0.5e12); // 1 byte of KV per token
  const m = { ...TOY, weights_bytes: 1e12, state_bytes_per_prefix: 0 };
  const w = { blockTokens: null, requests: [{ t: 0, n: 1000, out: 11, hashes: Array.from({ length: 10 }, (_, j) => j) }] };
  const link = { family: "p2p", bw_gib_s: 1, latency_s: 0.5 };
  const r = simulate(w, { ...base, model: m, gpuGiB: 0, cpuGiB: 0, decode: { gpu: dgpu, batch: 4 }, transfer: link });
  const s = decodeStepS(m, dgpu, 1, 1000 + 5.5);
  assert.equal(r.decodeBatch, 1);
  assert.ok(Math.abs(r.ttft.mean - (0.01 + 1 + 0.5 + 1000 / GIB + s)) < 1e-9); // overhead + recompute 1 s + transfer + first step
  assert.ok(Math.abs(r.e2e.mean - (r.ttft.mean + 10 * s)) < 1e-9);
  // One ready sequence occupies one decode group.
  assert.equal(r.decodeGpus, 1);
  assert.ok(Math.abs(r.usd.decode - (r.decodeGpus * 3.6 * r.windowS) / 3600) < 1e-12);
  const warm = simulate({ ...w, requests: [w.requests[0], { ...w.requests[0], t: 100 }] }, { ...base, model: m, gpuGiB: 1, cpuGiB: 0, warmupS: 50 });
  assert.equal(warm.requests, 1);
  assert.equal(warm.share.gpu, 1); // the warm-up request filled the cache but is not counted
});

test("price sheet: input, cached input and output account for every dollar and every token", () => {
  const s3 = { family: "obj", bw_gib_s: 1, latency_s: 0.05, shared: true, bill: "stored", usd_per_gib_month: 1000, usd_per_hour: 0 };
  const dgpu = { mem_bw_tb_s: 1, mbu: 0.5, usd_per_hour: 2 };
  const w = { blockTokens: null, requests: roundRobin.requests.map((q) => ({ ...q, out: 50 })) };
  const r = simulate(w, { ...base, gpuGiB: 0, cpuGiB: 0, shared: s3, decode: { gpu: dgpu, batch: 4 }, retentionH: 1 });
  const p = r.price;
  assert.ok(Math.abs(p.usd.input + p.usd.cachedInput + p.usd.output - r.usd.total) < 1e-9);
  assert.equal(p.tokens.input + p.tokens.cachedInput, r.promptTokens);
  assert.equal(p.tokens.cachedInput, r.tokens.shared);
  assert.equal(p.tokens.output, r.outputTokens);
  assert.ok(p.input > 0 && p.cachedInput > 0 && p.output > 0);
  const none = simulate(roundRobin, { ...base, gpuGiB: 0, cpuGiB: 0 }); // nothing cached, no decode
  assert.equal(none.price.cachedInput, null);
  assert.equal(none.price.output, null);
  assert.ok(Math.abs(none.price.usd.input - none.usd.total) < 1e-9);
});

test("a model with a fixed state resumes only at a stored snapshot: the end of an earlier sequence", () => {
  // two prefixes share 5 blocks; each is stored whole, so the shared part ends no stored sequence
  const reqs = [
    { t: 0, n: 1000, hashes: Array.from({ length: 10 }, (_, j) => (j < 5 ? `s${j}` : `a${j}`)) },
    { t: 10, n: 1000, hashes: Array.from({ length: 10 }, (_, j) => (j < 5 ? `s${j}` : `b${j}`)) },
    { t: 20, n: 1000, hashes: Array.from({ length: 10 }, (_, j) => (j < 5 ? `s${j}` : `a${j}`)) },
  ];
  const w = { blockTokens: null, requests: reqs };
  const big = { ...base, gpuGiB: gib(1e6), cpuGiB: 0 };
  const kvOnly = simulate(w, big);
  const withState = simulate(w, { ...big, model: { ...TOY, read_fixed_bytes: 50, state_bytes_per_prefix: 50 } });
  assert.equal(kvOnly.tokens.gpu, 500 + 1000); // request 2 reuses the shared 5 blocks, request 3 all of a
  assert.equal(withState.tokens.gpu, 1000); // request 2 has no snapshot to resume from; request 3 does
});

test("end-of-minute occupancy includes completed writes and counts physical cache copies once", () => {
  const requests = [5, 20, 61].map((t, i) => ({
    t, n: 100, conv: i, hashes: [i === 2 ? "b" : "a"],
  }));
  const snapshots = [];
  simulate({ blockTokens: 100, requests }, {
    ...base, gpuGiB: gib(100), cpuGiB: gib(200), cluster: { nodes: 2, gpusPerNode: 2, tp: 1 },
    routing: "sticky", disk: { family: "fs", capacity_gib: gib(200), bw_gib_s: 1 },
    shared: { family: "obj", shared: true, bw_gib_s: 1 },
  }, undefined, (i, snapshot) => { snapshots[i] = snapshot; });
  assert.deepEqual(snapshots[0], {
    gpu: { usedGiB: gib(200), capacityGiB: gib(400) },
    cpu: { usedGiB: gib(100), capacityGiB: gib(400) },
    disk: { usedGiB: gib(100), capacityGiB: gib(400) },
    shared: { usedGiB: gib(100), capacityGiB: null },
  });
  assert.equal(snapshots[1].shared.usedGiB, gib(200));
  assert.equal(snapshots[1].cpu.usedGiB, gib(200));
});

test("each block uses the nearest resident copy across all five configured tiers", () => {
  // Instances 0 and 1 share node A. Instance 2 lives on node B.
  const visits = [[0, "a"], [0, "a"], [1, "a"], [2, "a"], [0, "b"], [0, "c"],
    [0, "a"], [0, "b"], [0, "c"], [0, "d"], [2, "e"], [2, "f"], [0, "a"]];
  const requests = visits.map(([conv, hash], i) => ({ t: i / 100, n: 100, conv, hashes: [hash] }));
  const cfg = { ...base, software: { ...SW, request_overhead_s: .00001, cpu_load_latency_s: .00001 }, gpu: { ...GPU, tflops_bf16: 100 }, gpuGiB: gib(100), cpuGiB: gib(200), routing: "sticky",
    cluster: { nodes: 2, gpusPerNode: 2, tp: 1 },
    disk: { family: "fs", capacity_gib: gib(300), bw_gib_s: 1 },
    p2p: { family: "p2p", bw_gib_s: 1 },
    shared: { family: "obj", shared: true, bw_gib_s: 1 } };
  const servedBy = [];
  simulate({ requests, blockTokens: 100 }, { ...cfg, peerFirst: true }, (_req, _ttft, hits) => {
    servedBy.push(Object.keys(hits).find(key => hits[key]) ?? "miss");
  });
  assert.deepEqual(servedBy, ["miss", "gpu", "cpu", "p2p", "miss", "miss",
    "disk", "disk", "disk", "miss", "miss", "miss", "shared"]);
});

test("the MP controller reclaims 20% of LRU keys at the 80% watermark", () => {
  const cache = new LRU(100);
  for (let i = 0; i < 19; i++) cache.put(i, 4);
  cache.sweep();
  assert.equal(cache.used, 76);
  cache.put(19, 4);
  cache.touch(1);
  cache.sweep();
  assert.equal(cache.used, 64);
  assert.deepEqual(cache.keys(), [...Array.from({ length: 15 }, (_, i) => i + 5), 1]);
});

test("external admission saves complete prompt chunks and optionally synthetic reply chunks", () => {
  const run = (saveDecode, gpuGiB = 0) => simulate({ requests: [
    { t: 0, n: 250, out: 190, conv: 1, sys: 0 }, { t: 1, n: 450, conv: 1, sys: 0 },
  ] }, { ...base, gpuGiB, cpuGiB: 1, saveDecode });
  assert.equal(run(false).tokens.cpu, 200);
  assert.equal(run(true).tokens.cpu, 400);
  assert.equal(run(false, 1).tokens.gpu, 400);
});

test("peer-first skips S3 data and metadata probes when a peer already holds the block", () => {
  const work = { requests: [0, 1, 1].map((conv, t) => ({ t, conv, n: 100, hashes: ["k"] })) };
  const shared = { family: "obj", bw_gib_s: 1, usd_per_1k_put: .005, usd_per_1k_get: .0004 };
  const cfg = { ...base, gpuGiB: 0, cpuGiB: 1, routing: "sticky", cluster: { nodes: 2 },
    p2p: { family: "p2p", bw_gib_s: 1 }, shared, objectRanks: 2 };
  const stock = simulate(work, cfg);
  const peer = simulate(work, { ...cfg, peerFirst: true });
  assert.equal(stock.tokens.shared, 100);
  assert.equal(peer.tokens.p2p, 100);
  assert.equal(peer.tokens.cpu, 100);
  // Cold miss (req 0) sends PUT only (0 wasted HEADs); stock req 1 probes S3 once (2 ranks) + GETs; peer-first skips S3.
  assert.deepEqual(stock.s3Requests, { put: 2, head: 2, get: 2 });
  assert.deepEqual(peer.s3Requests, { put: 2, head: 0, get: 0 });
  assert.equal(peer.firstPayloadReadS.shared, null);
  assert.equal(stock.requestUsd, (.005 * 2 + .0004 * 4) / 1000);
  assert.equal(stock.usd.storage, stock.requestUsd);
  const warm = simulate(work, { ...cfg, warmupS: 1 });
  assert.deepEqual(warm.s3Requests, { put: 0, head: 2, get: 2 });
  const off = simulate(work, { ...cfg, shared: null });
  assert.equal(off.requestUsd, 0);
  assert.deepEqual(off.s3Requests, { put: 0, head: 0, get: 0 });
});

test("bounded replay charges the trailing window on a prefix hit without saving extra objects", () => {
  const work = { requests: [{ t: 0, n: 200, hashes: ["a", "b"] }, { t: 5, n: 220, hashes: ["a", "b"] }] };
  const times = [];
  const r = simulate(work, { ...base, gpuGiB: 1, model: { ...TOY, prefix_replay_tokens: 128 } },
    (_req, ttft) => times.push(ttft));
  assert.equal(r.tokens.gpu, 200);
  // 128 rebuilt tokens plus 20 new ones at 1 ms/token and 10 ms request overhead.
  assert.ok(Math.abs(times[1] - .158) < 1e-10);
});

test("hourly request pricing includes idle time before and after arrivals", () => {
  const shared = { family: "obj", bw_gib_s: 1, usd_per_1k_put: .005, usd_per_1k_get: .0004 };
  const r = simulate({ requests: [{ t: 1800, n: 100, hashes: ["one"] }] },
    { ...base, shared, gpuGiB: 1, cpuGiB: 0, durationMin: 60 });
  assert.equal(r.windowS, 3600);
  assert.equal(r.requestUsd * 3600 / r.windowS, .005 / 1000);
});

test("round-robin visits each node group before its next replica and charges remote bytes once", () => {
  const work = { requests: Array.from({ length: 5 }, (_, i) => ({ t: i * 10, conv: 7, n: 100, hashes: ["prefix"] })) };
  const cfg = { ...base, gpuGiB: 1, cpuGiB: 1, routing: "round-robin", peerFirst: true,
    cluster: { nodes: 4, pp: 2, gpusPerNode: 2, tp: 1 }, p2p: { family: "p2p", bw_gib_s: 1, latency_s: .01 } };
  const rows = [];
  simulate(work, cfg, (_req, ttft, hits, _size, load, _wait, _transfer, peerBytes) => {
    rows.push({ source: Object.keys(hits).find(k => hits[k]) ?? "miss", peerBytes, ttft, load });
  });
  assert.deepEqual(rows.map(r => r.source), ["miss", "p2p", "cpu", "cpu", "gpu"]);
  assert.deepEqual(rows.map(r => r.peerBytes), [0, 100, 0, 0, 0]);
  assert.ok(rows[1].load > rows[2].load && rows[1].ttft > rows[2].ttft);
  const sticky = simulate(work, { ...cfg, routing: "sticky" });
  assert.equal(sticky.tokens.p2p, 0);
});

const arrival = (t, conv, hash = "x", out = 0) => ({ t, conv, n: 100, out, hashes: [hash] });
const twoNodes = { ...base, gpuGiB: 0, cpuGiB: 1, routing: "sticky", cluster: { nodes: 2 } };
const wire = { family: "p2p", bw_gib_s: gib(100) }; // 1 s per block
const disk = { family: "fs", bw_gib_s: gib(100), capacity_gib: gib(10000) };

test("a second node can fetch completed KV only, and a private disk never supplies a remote hit", () => {
  const run = (t, overrides) => simulate({ requests: [arrival(0, 0), arrival(t, 1)] },
    { ...twoNodes, disk, p2p: wire, ...overrides });
  assert.equal(run(.000001).tokens.miss, 200);
  assert.equal(run(.12).tokens.p2p, 100);
  assert.equal(run(3, { cpuGiB: 0 }).tokens.disk, 0);
  assert.equal(run(3, { cpuGiB: 0 }).tokens.miss, 200);
});

test("minute-end snapshots distinguish in-flight prefill, CPU admission and a pending backing write", () => {
  const snapshots = [];
  const requests = [arrival(59.95, 0), arrival(119.5, 0, "y")];
  simulate({ requests }, { ...twoNodes, durationMin: 3, disk }, undefined, (_, s) => snapshots.push(s));
  assert.deepEqual(snapshots.map(s => s.cpu.usedGiB), [0, gib(200), gib(200)]);
  assert.deepEqual(snapshots.map(s => s.disk.usedGiB), [0, gib(100), gib(200)]);
});

test("simultaneous reads serialize on one SSD and run in parallel on different private SSDs", () => {
  const times = (a, b) => {
    const reads = [];
    simulate({ requests: [arrival(0, 0), arrival(0, 2), arrival(5, a), arrival(5, b)] },
      { ...twoNodes, cpuGiB: 0, cluster: { nodes: 2, gpusPerNode: 2 }, disk },
      (req, _ttft, k, _B, load) => { if (req.t === 5) { assert.equal(k.disk, 1); reads.push(load); } });
    return reads.sort((a, b) => a - b);
  };
  const same = times(0, 1), separate = times(0, 2);
  assert.ok(Math.abs(same[1] - same[0] - 1) < 1e-9);
  assert.ok(Math.abs(separate[1] - separate[0]) < 1e-9);
  assert.ok(Math.abs(same[0] - separate[0]) < 1e-9);
});

test("two remote CPU readers compete for their source NIC", () => {
  const loads = [];
  const r = simulate({ requests: [arrival(0, 0), arrival(2, 1), arrival(2, 2)] },
    { ...twoNodes, cluster: { nodes: 3 }, p2p: wire },
    (_req, _ttft, k, _B, load) => { if (k.p2p) loads.push(load); });
  assert.equal(r.tokens.p2p, 200);
  assert.ok(Math.abs(Math.max(...loads) - Math.min(...loads) - 1) < 1e-9);
});

test("S3 lookup pays one metadata RTT on a cold S3 hit, zero on misses or warm S3 hits", () => {
  const shared = { family: "obj", bw_gib_s: gib(200), latency_s: .25, usd_per_1k_get: .0004 };
  const loads = [];
  const requests = [0, 1, 2].map((i) => ({ t: i * 5, conv: Math.min(i, 1), n: 200, hashes: ["a","b"] }));
  const r = simulate({ requests }, { ...twoNodes, cpuGiB: 0, shared },
    (_req, _ttft, _k, _B, load) => loads.push(load));
  assert.deepEqual(r.s3Requests, { head: 2, put: 2, get: 4 });
  assert.equal(r.firstPayloadReadS.shared, 5.5); // first GET payload, after HEAD and GET latency
  assert.ok(Math.abs(loads[0]) < 1e-9); // cold miss does not stall on an S3 HEAD
  assert.ok(Math.abs(loads[1] - loads[2] - .25) < 1e-9); // cold S3 hit pays HEAD + GET, warm S3 hit pays GET only
  assert.ok(Math.abs(loads[2] - 1.26) < 1e-9);
});

test("cache lifetime expires stale blocks from S3 even when a GPU-hot block sits at the LRU head", () => {
  const shared = { family: "obj", bw_gib_s: 1, shared: true, usd_per_gib_month: 730 };
  const requests = [
    { t: 0, n: 200, hashes: ["hot", "dead1"] },
    { t: 5, n: 200, hashes: ["hot", "dead2"] },
    { t: 10, n: 100, hashes: ["hot"] }, // GPU hit keeps "hot" alive
    { t: 20, n: 100, hashes: ["hot"] },
    { t: 30, n: 100, hashes: ["hot"] },
  ];
  let endSnap;
  simulate({ requests }, { ...base, gpuGiB: 1, cpuGiB: 0, shared, cacheLifetimeS: 12, durationMin: 1 },
    undefined, (_, s) => { endSnap = s; });
  // Only "hot" survives past t=30 + 12s? At t=60, even "hot" (last used at 30) expired (>12s); check at t=25s via a request:
  const snaps = [];
  simulate({ requests: [...requests, { t: 25, n: 100, hashes: ["hot"] }] },
    { ...base, gpuGiB: 1, cpuGiB: 0, shared, cacheLifetimeS: 12, durationMin: 1 },
    undefined, (_, s) => snaps.push(s));
  // At t=60s all blocks last used <=30s are expired (>12s), so shared must be completely empty (0 GiB).
  assert.equal(endSnap.shared.usedGiB, 0);
});

test("dedicated RDMA fabric is isolated from S3 Ethernet uploads, and SSD writes use write_bw_gib_s", () => {
  const shared = { family: "obj", bw_gib_s: gib(100), usd_per_1k_get: 0 };
  const rdma = { family: "p2p", bw_gib_s: gib(100), rdma: true };
  const transfers = [];
  // Replica 0 starts a 1 s S3 PUT on ethTx at t=0.11; replica 1 sends KV over RDMA at t=0.31 -> zero ethTx blocking.
  simulate({ requests: [arrival(0, 0), arrival(.2, 1, "other", 1)] },
    { ...twoNodes, cluster: { nodes: 1, gpusPerNode: 2 }, cpuGiB: 0, shared, transfer: rdma,
      decode: { batch: 16, gpu: { mem_bw_tb_s: 1, mbu: 1 } } },
    (req, _ttft, _k, _B, _load, _wait, tr) => { if (req.out) transfers.push(tr); });
  assert.ok(Math.abs(transfers[0] - 1) < 1e-9, `RDMA transfer should not wait on S3 PUT: ${transfers[0]}`);
  // SSD with half write speed takes twice as long to write before a read can start.
  const slowWrite = { family: "fs", bw_gib_s: gib(100), write_bw_gib_s: gib(50), latency_s: 0 };
  const r = simulate({ requests: [arrival(0, 0, "a"), arrival(1, 0, "a")] },
    { ...base, gpuGiB: 0, cpuGiB: 0, disk: slowWrite });
  // Write starts at 0.11, takes 100/50 = 2.0 s -> finishes at 2.11; req at t=1 misses disk because write isn't done yet.
  assert.equal(r.tokens.disk, 0);
});

test("stored-byte retention and writes bill every private disk and exclude warm-up admissions", () => {
  const requests = [arrival(0, 0), arrival(10, 0, "a"), arrival(12, 1, "b")];
  const r = simulate({ requests }, { ...twoNodes, disk: { ...disk, capacity_gib: null,
    bill: "stored", usd_per_gib_month: 730, usd_per_gib_written: 1 }, warmupS: 5, retentionH: 1 });
  // $730/GiB-month at one hour retention gives $1/GiB admitted, plus $1/GiB written.
  assert.ok(Math.abs(r.usd.storage - 2 * gib(200)) < 1e-15);
});

test("one request must fit active KV including live state and peak output, independently of prefix budget", () => {
  const model = { ...TOY, state_bytes_per_prefix: 40 };
  const request = { requests: [arrival(0, 0, "a", 20)] };
  const cfg = { ...base, model, physicalKvGiB: gib(140), gpuGiB: 0,
    decode: { gpu: { mem_bw_tb_s: 1, mbu: 1 }, batch: 8, batchCap: ctx => Math.floor(160 / (ctx + 40)) } };
  assert.equal(simulate(request, cfg).requests, 1);
  assert.match(simulate(request, { ...cfg, physicalKvGiB: gib(139) }).error, /prefill/);
  assert.match(simulate({ requests: [arrival(0, 0, "a", 21)] }, cfg).error, /decode.*batch 1/);
});

test("only overlapping sequences share decode GPU time", () => {
  const cfg = { ...twoNodes, model: { ...TOY, weights_bytes: 1e12 },
    decode: { gpu: { mem_bw_tb_s: 1, mbu: 1, usd_per_hour: 1 }, pooled: true, batch: 1024 } };
  const run = t => simulate({ requests: [arrival(0, 0, "a", 4), arrival(t, 1, "b", 4)] }, cfg);
  const apart = run(10), together = run(0), joined = run(.5);
  assert.equal(apart.decodeBatch, 1);
  assert.equal(together.decodeBatch, 2);
  assert.ok(joined.decodeBatch > 1 && joined.decodeBatch < 2);
  assert.ok(Math.abs(apart.usd.decode / together.usd.decode - 2) < 1e-8);
});

test("synthetic think time begins after the final token, and trace timestamps remain fixed", () => {
  const cfg = { ...twoNodes, model: { ...TOY, weights_bytes: 1e12 },
    decode: { gpu: { mem_bw_tb_s: 1, mbu: 1 }, batch: 4 } };
  const requests = [arrival(0, 0, "a", 3), { ...arrival(.5, 0, "b", 1), followUp: true }];
  const collect = w => {
    const arrivals = [];
    simulate(w, cfg, req => arrivals.push(req.t));
    return arrivals.sort((a,b) => a-b);
  };
  assert.deepEqual(collect({ requests }), [0, .5]);
  const generated = collect({ requests, durationS: 10 });
  assert.ok(Math.abs(generated[1] - (0.11 + 3 + .5)) < 1e-8);
  assert.deepEqual(collect({ requests, durationS: 3 }), [0]);
});

test("S3 and disaggregated handoffs queue on the same node interface", () => {
  const shared = { family: "obj", bw_gib_s: gib(100), usd_per_1k_get: 0 };
  const requests = [arrival(0, 0), arrival(5, 0), arrival(5, 1)];
  const cfg = { ...twoNodes, cluster: { nodes: 1, gpusPerNode: 2 }, cpuGiB: 0, shared };
  const loads = [];
  simulate({ requests }, cfg, (req, _ttft, k, _B, load) => {
    if (req.t === 5) { assert.equal(k.shared, 1); loads.push(load); }
  });
  assert.ok(Math.abs(loads[1] - loads[0] - 1) < 1e-9);
  // At 200 ms the other replica's prompt handoff encounters the first
  // replica's 1 s background PUT, occupying their common NIC since 110 ms.
  const transfers = [];
  simulate({ requests: [arrival(0, 0), arrival(.2, 1, "other", 1)] },
    { ...cfg, transfer: wire, decode: { batch: 16, gpu: { mem_bw_tb_s: 1, mbu: 1 } } },
    (req, _ttft, _k, _B, _load, _wait, tr) => { if (req.out) transfers.push(tr); });
  assert.ok(transfers[0] >= 1.8 - 1e-9);
});

test("an imported reply cannot be read before generation completes", () => {
  // Request timestamps are supplied externally while synthetic IDs let this
  // fixture identify the same generated blocks on a later request.
  const requests = [
    { t: 0, n: 100, out: 200, conv: 0, sys: 0 },
    { t: 1, n: 300, out: 0, conv: 0, sys: 0 },
  ];
  const cfg = { ...twoNodes, routing: "round-robin", p2p: wire, saveDecode: true,
    model: { ...TOY, weights_bytes: 1e12 }, decode: { batch: 2, gpu: { mem_bw_tb_s: 1, mbu: 1 } } };
  assert.equal(simulate({ requests }, cfg).tokens.p2p, 100);
  assert.equal(simulate({ requests: [requests[0], { ...requests[1], t: 205 }] }, cfg).tokens.p2p, 300);
});

// Continuous prefill batching on the toy model. A 1 TB/s GPU reads 8 GB of weights in 8 ms per step; the other 2 ms of
// the 10 ms request overhead delay each request without using the GPU.
const BATCH = { ...base, model: { ...TOY, weights_bytes: 8e9 }, gpu: { ...GPU, mem_bw_tb_s: 1, mbu: 1 }, gpuGiB: gib(1e6), cpuGiB: 0 };
const fresh = (times, n) => ({ blockTokens: null, requests: times.map((t, i) => ({ t, n, hashes: Array.from({ length: n / 100 }, (_, j) => `r${i}:${j}`) })) });
const close = (a, b) => Math.abs(a - b) < 1e-9;

test("a request alone in its prefill step keeps the one-at-a-time TTFT", () => {
  const w = fresh([0], 1000);
  const serial = simulate(w, BATCH), batched = simulate(w, { ...BATCH, prefillBatchTokens: 4096 });
  assert.ok(close(serial.ttft.mean, 1.01) && close(batched.ttft.mean, 1.01), `${serial.ttft.mean} ${batched.ttft.mean}`);
});

test("batching compute-bound prompts makes each wait for the whole step", () => {
  const w = fresh([0, 0], 1000);
  const serial = simulate(w, BATCH), batched = simulate(w, { ...BATCH, prefillBatchTokens: 4096 });
  // one at a time: 1.01 s, then 1.01 s more; together: both finish after 2 ms + 8 ms + 2 s of compute
  assert.ok(close(serial.ttft.mean, 1.515), `${serial.ttft.mean}`);
  assert.ok(close(batched.ttft.p50, 2.01) && close(batched.ttft.mean, 2.01), `${batched.ttft.mean}`);
});

test("a prompt longer than the budget is split across steps, each paying a weight read", () => {
  const w = fresh([0], 3000);
  const batched = simulate(w, { ...BATCH, prefillBatchTokens: 1000 });
  assert.ok(close(batched.ttft.mean, 3.01 + 2 * 0.008), `${batched.ttft.mean}`);
});

test("batching short prompts amortises the weight read and drains a queue the serial model cannot", () => {
  const w = fresh(Array.from({ length: 400 }, (_, i) => i * 0.015), 100); // 0.1 s of compute every 15 ms
  const short = { ...BATCH, model: { ...BATCH.model, params: 5e7 } }; // 10 ms per 100 tokens
  const serial = simulate(w, short), batched = simulate(w, { ...short, prefillBatchTokens: 8192 });
  assert.ok(serial.utilization > 0.99, `${serial.utilization}`); // 20 ms of work every 15 ms: always busy
  assert.ok(batched.utilization < 1 && batched.ttft.mean < serial.ttft.mean / 5, `${batched.utilization} ${batched.ttft.mean} ${serial.ttft.mean}`);
});

test("per-GPU RDMA fabrics give each replica on a node its own independent NIC rail group", () => {
  const ib = { family: "p2p", bw_gib_s: gib(100), rdma: true, per_gpu: true };
  const roce = { family: "p2p", bw_gib_s: gib(100), rdma: true, per_gpu: false };
  const run = transfer => {
    const trs = [];
    simulate({ requests: [arrival(0, 0, "a", 1), arrival(0, 1, "b", 1)] },
      { ...twoNodes, cluster: { nodes: 1, gpusPerNode: 2, tp: 1 }, cpuGiB: 0, transfer,
        decode: { batch: 16, gpu: { mem_bw_tb_s: 1, mbu: 1 } } },
      (_req, _ttft, _k, _B, _load, _wait, tr) => trs.push(tr));
    return trs.sort((a, b) => a - b);
  };
  const perGpu = run(ib), sharedNic = run(roce);
  assert.ok(close(perGpu[0], 1) && close(perGpu[1], 1), `per-GPU IB runs in parallel: ${perGpu}`);
  assert.ok(close(sharedNic[0], 1) && close(sharedNic[1], 2), `shared RoCE serializes: ${sharedNic}`);
});

test("S3 bucket size limit and S3-only object retention bound occupancy and storage cost independently", () => {
  const capped = { family: "obj", bw_gib_s: 1, shared: true, bill: "stored", capacity_gib: gib(200), usd_per_gib_month: 730 };
  const requests = [0, 5, 10].map((t, i) => ({ t, conv: i, n: 100, hashes: [`k${i}`] }));
  let snap;
  const rCap = simulate({ requests }, { ...base, gpuGiB: 0, cpuGiB: 0, shared: capped, retentionH: 10, durationMin: 1 },
    undefined, (_, s) => { snap = s; });
  assert.equal(snap.shared.usedGiB, gib(200));
  assert.equal(rCap.evictions.shared, 1);
  assert.ok(rCap.firstSharedEvictionS > 10);
  // 10h retention of 300B/min would exceed 200B, so bucket capacity_gib caps billed GiB at 200B -> 200B/GiB * (1/60 h).
  assert.ok(close(rCap.usd.storage, gib(200) * (1 / 60)));

  // S3-only TTL expires objects from S3 while local GPU cache retains them.
  const uncapped = { family: "obj", bw_gib_s: 1, shared: true, bill: "stored", usd_per_gib_month: 730 };
  const rTtl = simulate({ requests: [{ t: 0, conv: 0, n: 100, hashes: ["keep"] }, { t: 30, conv: 0, n: 100, hashes: ["keep"] }] },
    { ...base, routing: "sticky", gpuGiB: 1, cpuGiB: 0, shared: uncapped, sharedLifetimeS: 10, retentionH: null, durationMin: 1 },
    undefined, (_, s) => { snap = s; });
  assert.equal(rTtl.tokens.gpu, 100); // GPU still had it at t=30
  assert.equal(snap.shared.usedGiB, 0); // S3 expired it after 10s
});

test("batched prefill serializes concurrent CPU->GPU loads on the same worker's PCIe hostLane, while sibling workers run in parallel", () => {
  const hostLimited = { ...base, gpu: { ...GPU, host_gib_s: gib(100), mem_bw_tb_s: 1, mbu: 1 }, gpuGiB: 0, cpuGiB: 1, prefillBatchTokens: 8192 };
  // Request 0 warms CPU cache; requests 1 & 2 arrive together at t=5 and both hit CPU (100 bytes @ 100 B/s = 1.0 s PCIe transfer each).
  const reqs = (c1, c2) => [
    { t: 0, conv: 0, n: 100, hashes: ["p"] },
    { t: 5, conv: c1, n: 100, hashes: ["p"] },
    { t: 5, conv: c2, n: 100, hashes: ["p"] },
  ];
  const measure = (cluster, c1, c2) => {
    const loads = [];
    simulate({ requests: reqs(c1, c2) }, { ...hostLimited, routing: "sticky", cluster },
      (req, _ttft, k, _B, load) => { if (req.t === 5) { assert.equal(k.cpu, 1); loads.push(load); } });
    return loads.sort((a, b) => a - b);
  };
  const sameGpu = measure({ nodes: 1, gpusPerNode: 1, tp: 1 }, 0, 0);
  const twoGpus = measure({ nodes: 1, gpusPerNode: 2, tp: 1 }, 0, 1);
  assert.ok(Math.abs(sameGpu[1] - sameGpu[0] - 1) < 1e-9, `same GPU PCIe link queues second H2D transfer: ${sameGpu}`);
  assert.ok(Math.abs(twoGpus[1] - twoGpus[0]) < 1e-9, `distinct GPUs run PCIe H2D transfers in parallel: ${twoGpus}`);
});

test("skipSlowLoads bypasses slow storage tiers when GPU recomputation is faster", () => {
  // Toy GPU recomputes 1000 tokens in 1.0 s; slow disk takes 0.48 + 5.0 = 5.48 s.
  const verySlowDisk = { family: "fs", bw_gib_s: 200 / GIB, latency_s: 0.48, capacity_gib: gib(10000) };
  const work = { blockTokens: null, requests: [{ t: 0, n: 1000, hashes: Array.from({ length: 10 }, (_, j) => `h${j}`) },
    { t: 20, n: 1000, hashes: Array.from({ length: 10 }, (_, j) => `h${j}`) }] };
  const stock = simulate(work, { ...base, gpuGiB: 0, cpuGiB: 0, disk: verySlowDisk, skipSlowLoads: false });
  const smart = simulate(work, { ...base, gpuGiB: 0, cpuGiB: 0, disk: verySlowDisk, skipSlowLoads: true });
  assert.equal(stock.tokens.disk, 1000);
  assert.equal(smart.tokens.disk, 0);
  assert.ok(smart.ttft.p90 < stock.ttft.p90 / 4, `cost-aware guard avoids 5.5s disk stall: ${smart.ttft.p90} vs ${stock.ttft.p90}`);
});
