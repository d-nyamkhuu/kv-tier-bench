// Runs simulations off the page's main thread. A newer job makes an older sweep stop at its next step.

import { combos, runScenario } from "./scenario.js";

let latest = 0;
const yieldToMessages = () => new Promise((resolve) => setTimeout(resolve, 0));
const BASELINE = { disk: null, p2p: null, shared: null };
const NODE_COUNTS = [1, 2, 4, 8, 16, 32];

const point = (r, meta) => (r.error ? { ...meta, error: r.error } : {
  ...meta, x: r.usdPer1kRequests, y: r.ttft.mean, p90: r.ttft.p90, priced: r.priced, overloaded: r.utilization >= 1,
  share: r.share, order: r.order, gpuName: r.gpuName,
});

self.onmessage = async ({ data }) => {
  if (data.kind === "market") return marketJob(data);
  // series: which parts of the map to (re)compute; [] = only the estimate
  const { id, config, trace, series = ["combo", "nodes-yours", "nodes-base", "gpus"] } = data;
  latest = id;
  // 1. the live estimate: your setup and GPU + CPU on the same cluster
  const current = runScenario(config, {}, trace);
  const baseline = runScenario(config, BASELINE, trace);
  self.postMessage({ id, kind: "estimate", current, baseline });
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
    self.postMessage({ id, kind: "point", point: point(runScenario(config, over, trace), { ...meta, over }), done: i + 1, total: jobs.length });
  }
  self.postMessage({ id, kind: "sweep-done" });
};

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
