import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { defaultConfig } from "../config.js";
import { GPUS, SOFTWARE, VARIANTS } from "../presets.js";
import { buildRun, combos, paretoFrontier, runScenario } from "../scenario.js";
import { simulate, synthetic } from "../sim.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const config = () => defaultConfig(S1, VARIANTS, GPUS, SOFTWARE);

test("combos: every candidate per slot, plus off", () => {
  const all = combos({ disk: ["a", "b"], p2p: ["c"], shared: [] });
  assert.equal(all.length, 3 * 2 * 1);
  assert.deepEqual(all[0], { disk: null, p2p: null, shared: null });
});

test("Pareto frontier keeps only points no other beats on both cost and latency; unpriced never qualify", () => {
  const pts = [{ id: "a", x: 1, y: 5 }, { id: "b", x: 2, y: 3 }, { id: "c", x: 3, y: 4 }, { id: "d", x: 4, y: 1 }, { id: "free", x: 0, y: 0, priced: false }];
  assert.deepEqual(paretoFrontier(pts).map((p) => p.id), ["a", "b", "d"]);
});

test("a large cluster replayed at 8 nodes with its per-node load matches the full replay", () => {
  const c = config();
  Object.assign(c.workload, { requestsPerMin: 1600, thinkTimeS: 300, durationMin: 30 });
  c.sim = { ...c.sim, nodes: 16, routing: "random", warmupMin: 10 };
  const scaled = runScenario(c);
  const full = buildRun(c);
  full.cfg.cluster.nodes = 16;
  full.cfg.costScale = 1;
  const r = simulate(synthetic(c.workload), full.cfg);
  assert.equal(scaled.simNodes, 8);
  for (const [a, b] of [[scaled.ttft.mean, r.ttft.mean], [scaled.share.miss, r.share.miss], [scaled.usdPer1kRequests, r.usdPer1kRequests]]) {
    assert.ok(Math.abs(a / b - 1) < 0.03, `${a} vs ${b}`);
  }
});

test("a model that does not fit the GPU is an error, not a number; TP lets it fit", () => {
  const c = config();
  assert.match(runScenario(c, { gpu: "l40s", decodeGpu: "l40s" }).error, /does not fit/);
  const ok = runScenario({ ...c, workload: { ...c.workload, durationMin: 25 } }, { gpu: "l40s", decodeGpu: "l40s", tp: 2 });
  assert.equal(ok.error, undefined);
});

test("every model preset fits some priced GPU, in one node or across up to 8, and runs to a finite estimate", async () => {
  const { MODELS } = await import("../presets.js");
  const { gpuKvGiB, withTp } = await import("../model.js");
  const c = defaultConfig(S1, VARIANTS, GPUS, SOFTWARE, MODELS);
  Object.assign(c.workload, { requestsPerMin: 60, durationMin: 20 });
  let multiNode = 0;
  for (const [id, m] of Object.entries(MODELS)) {
    let fit = null;
    for (const pp of [1, 2, 3, 4, 5, 6, 7, 8]) for (const g of GPUS.filter((x) => x.usd_per_hour != null)) for (const tp of [1, 2, 4, 8]) if (!fit && gpuKvGiB(withTp(m, tp), g, SOFTWARE, tp, pp) > 0) fit = { g, tp, pp };
    assert.ok(fit, `${id} fits no GPU`);
    if (fit.pp > 1) multiNode++;
    const r = runScenario({ ...c, sim: { ...c.sim, model: id, gpu: fit.g.id, decodeGpu: fit.g.id, tp: fit.tp, pp: fit.pp, nodes: 2 * fit.pp, gpusPerNode: Math.max(fit.tp, 2), warmupMin: 5 } });
    assert.ok(!r.error, `${id}: ${r.error}`);
    assert.ok(Number.isFinite(r.ttft.mean) && r.ttft.mean > 0, `${id}: ttft ${r.ttft.mean}`);
    assert.ok(Number.isFinite(r.price.input), `${id}: price ${JSON.stringify(r.price)}`);
  }
  assert.ok(multiNode >= 1); // Qwen3.8-2.4T needs more than one node on every GPU
});

test("pipeline stages across nodes: memory adds up, a request crosses every stage, the instance pipelines requests", async () => {
  const { gpuKvGiB, timing } = await import("../model.js");
  const m = { ...S1.models["qwen3-vl-32b"], hidden_size: 5120 };
  const rtx = GPUS.find((g) => g.id === "rtx-pro-6000");
  const one = gpuKvGiB(m, rtx, SOFTWARE, 1, 1), two = gpuKvGiB(m, rtx, SOFTWARE, 1, 2);
  assert.ok(Math.abs(two - (2 * one + m.weights_bytes / 2 ** 30)) < 1e-6); // the weights are held once
  const link = { bw_gib_s: 10, latency_s: 0.001 };
  const T1 = timing(m, rtx, SOFTWARE, 1), T2 = timing(m, rtx, SOFTWARE, 1, undefined, { stages: 2, link });
  const n = 8192;
  assert.ok(Math.abs(T2.R(n) - T1.R(n) - (0.001 + (n * 5120 * 2) / (10 * 2 ** 30))) < 1e-9); // + one hop
  assert.ok(Math.abs(T2.cpuLoad(n) - SOFTWARE.cpu_load_latency_s - (T1.cpuLoad(n) - SOFTWARE.cpu_load_latency_s) / 2) < 1e-12); // two host links
  // two-node instances on 4 nodes: 2 instances; the cost counts all 4 nodes' GPUs
  const c = config();
  c.sim = { ...c.sim, model: "qwen3-vl-32b", nodes: 4, gpusPerNode: 1, tp: 1, pp: 2, decode: false };
  const r = runScenario(c);
  assert.equal(r.instances, 2);
  assert.equal(r.prefillGpus, 4);
  c.sim.nodes = 3;
  assert.match(runScenario(c).error, /instances of 2 nodes/);
});
