// Runs simulations off the page's main thread. A newer job makes an older sweep stop at its next step.

import { atBusy } from "./lab.js";
import { combos, runScenario } from "./scenario.js";

let latest = 0;
const yieldToMessages = () => new Promise((resolve) => setTimeout(resolve, 0));
const BASELINE = { disk: null, p2p: null, shared: null };
const NODE_COUNTS = [1, 2, 4, 8, 16, 32];

// a map point: TTFT at the config's load, and its cost both at that load (x) and at the target busy share (capX)
const point = (r, meta, cap = null) => (r.error ? { ...meta, error: r.error } : {
  ...meta, x: r.usdPer1kRequests, y: r.ttft.mean, p90: r.ttft.p90, priced: r.priced, overloaded: r.utilization >= 1,
  share: r.share, order: r.order, gpuName: r.gpuName, capX: cap?.priced ? cap.usdPer1k : null, capRpm: cap?.rpm ?? null,
});

self.onmessage = async ({ data }) => {
  if (data.kind === "market") return marketJob(data);
  if (data.kind === "sens") return sensJob(data);
  if (data.kind === "models") return modelsJob(data);
  // series: which parts of the map to (re)compute; [] = only the estimate
  const { id, config, trace, series = ["combo", "nodes-yours", "nodes-base", "gpus"] } = data;
  latest = id;
  // 1. the live estimate: your setup and GPU + CPU on the same cluster
  const live = { maxSimNodes: config.sim.nodes };
  const current = runScenario(config, {}, trace, live);
  const baseline = runScenario(config, BASELINE, trace, live);
  self.postMessage({ id, kind: "estimate", current, baseline });
  // what the setup costs with its prefill GPUs loaded to the lab's target (synthetic load only: a trace has its own rate)
  if (!trace && !current.error && current.requests) {
    await yieldToMessages();
    if (latest !== id) return;
    const cap = atBusy(config, {}, { first: current }).cap;
    const base = baseline.error || !baseline.requests ? null : atBusy(config, BASELINE, { first: baseline }).cap;
    if (latest !== id) return;
    self.postMessage({ id, kind: "capacity", cap, base });
  }
  if (!series.length) return;

  // 2. the trade-off map, streamed point by point
  const sim = config.sim;
  const jobs = [
    ...combos(sim.candidates).map((o) => ({ over: o, meta: { series: "combo", ...o } })),
    ...NODE_COUNTS.flatMap((n) => [
      { over: { nodes: n }, meta: { series: "nodes-yours", nodes: n } },
      { over: { ...BASELINE, nodes: n }, meta: { series: "nodes-base", nodes: n } },
    ]),
    ...config.gpus.filter((g) => g.id !== sim.gpu).map((g) => ({ over: { gpu: g.id, decodeGpu: g.id }, meta: { series: "gpus", gpu: g.id } })),
  ].filter((j) => series.includes(j.meta.series));
  for (let i = 0; i < jobs.length; i++) {
    await yieldToMessages();
    if (latest !== id) return;
    const { over, meta } = jobs[i];
    const r = runScenario(config, over, trace);
    const cap = trace || r.error || !r.requests ? null : atBusy(config, over, { first: r }).cap;
    self.postMessage({ id, kind: "point", point: point(r, { ...meta, over }, cap), done: i + 1, total: jobs.length });
  }
  self.postMessage({ id, kind: "sweep-done" });
};

// The tier lab (lab.js): a sweep over one property of one tier, and the same tiers across models. Each kind keeps its
// own latest id, so a sweep and a model comparison can share a worker; a newer job of a kind stops the older one.
const latestLab = { sens: 0, models: 0 };

async function sensJob({ id, config, slot, base, key, values, capacity }) {
  const { sweepPoint } = await import("./lab.js");
  latestLab.sens = id;
  const xs = [null, ...values]; // null: the slot off, the reference line
  for (let i = 0; i < xs.length; i++) {
    await yieldToMessages();
    if (latestLab.sens !== id) return;
    self.postMessage({ id, kind: "sens-point", point: sweepPoint(config, slot, base, key, xs[i], { capacity }), done: i + 1, total: xs.length });
  }
  self.postMessage({ id, kind: "sens-done" });
}

async function modelsJob({ id, config, models }) {
  const { modelCompare } = await import("./lab.js");
  latestLab.models = id;
  for (let i = 0; i < models.length; i++) {
    await yieldToMessages();
    if (latestLab.models !== id) return;
    self.postMessage({ id, kind: "models-row", row: modelCompare(config, models[i]), done: i + 1, total: models.length });
  }
  self.postMessage({ id, kind: "models-done" });
}

// The comparison with provider prices: one reference run per model, streamed row by row (market.js).
let latestMarket = 0;
async function marketJob({ id, config, models, speeds = {} }) {
  const { marketRun } = await import("./market.js");
  latestMarket = id;
  for (let i = 0; i < models.length; i++) {
    await yieldToMessages();
    if (latestMarket !== id) return;
    self.postMessage({ id, kind: "market-row", row: marketRun(config, models[i], { speed: speeds[models[i]] }), done: i + 1, total: models.length });
  }
  self.postMessage({ id, kind: "market-done" });
}
