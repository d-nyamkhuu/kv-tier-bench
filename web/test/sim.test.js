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
  const withCpu = simulate(roundRobin, { ...base, gpuGiB: gib(2500), cpuGiB: gib(5000) });
  assert.equal(gpuOnly.share.gpu, 0); // LRU with a cyclic working set larger than the cache: every access misses
  assert.equal(gpuOnly.share.miss, 1);
  assert.ok(withCpu.share.cpu > 0.85, `cpu share ${withCpu.share.cpu}`);
  assert.ok(withCpu.ttft.mean < gpuOnly.ttft.mean / 5);
  // the CPU tier is written through, so it never misses what GPU evicted
  const bigGpu = simulate(roundRobin, { ...base, gpuGiB: gib(5000), cpuGiB: gib(5000) });
  assert.equal(bigGpu.share.cpu, 0);
  assert.ok(bigGpu.share.gpu > 0.85);
});

test("lower tiers take over nearest first: local disk, then peers, then shared storage", () => {
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
  const cfg = { ...base, gpuGiB: gib(5000), cpuGiB: gib(5000), routing: "sticky", cluster: { nodes: 2, gpusPerNode: 1, tp: 1 } };
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
  assert.ok(Math.abs(idle.usd.prefill - 8 * 3.6) < 1e-9); // 8 GPUs x 1 h, though they served two requests
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
  const s = decodeStepS(m, dgpu, 4, 1000 + 5.5);
  assert.ok(Math.abs(r.ttft.mean - (0.01 + 1 + 0.5 + 1000 / GIB + s)) < 1e-9); // overhead + recompute 1 s + transfer + first step
  assert.ok(Math.abs(r.e2e.mean - (r.ttft.mean + 10 * s)) < 1e-9);
  // decode GPUs are provisioned whole: the GPU-seconds the 11 tokens need (a quarter of each step) over the window
  assert.equal(r.decodeGpus, Math.ceil((11 * s) / 4 / r.windowS));
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
