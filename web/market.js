// The simulator against what providers charge: OpenRouter's prices per provider (web/data/openrouter.json, a dated
// snapshot from `kvtb openrouter-prices`) beside the modeled $ per 1M tokens of each model. DOM-free.
//
// The reference deployment per model: the page's settings (GPU, precision, tiers, serving, workload shape), 8
// instances of the cheapest of three sizes (the fewest GPUs that hold the model, twice, four times: the fewest can
// leave too little KV memory), loaded until the prefill GPUs are busy `util` of the time, with decode GPUs sized to
// that load. A provider runs near capacity, so an idle cluster would overstate cost. Decode holds `speed` tokens/s per
// request (OpenRouter's p50 for the model) by running the largest batch that keeps it, as a provider does; without
// one it fills the batch. Holding OpenRouter's speed makes the simulated speed match it by construction: what the
// comparison then tests is the price of that speed.

import { fitFor, runScenario } from "./scenario.js";

const median = (xs) => {
  const s = xs.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const k = s.length;
  return k % 2 ? s[(k - 1) / 2] : (s[k / 2 - 1] + s[k / 2]) / 2;
};
const range = (xs) => {
  const s = xs.filter((x) => x != null && Number.isFinite(x));
  return s.length ? { median: median(s), min: Math.min(...s), max: Math.max(...s), n: s.length } : null;
};

export function providerStats(providers) {
  return { n: providers.length, input: range(providers.map((p) => p.input)), cached: range(providers.map((p) => p.cached)), output: range(providers.map((p) => p.output)) };
}

export const MARKET_INSTANCES = 8;

// the replay window of a reference run: long enough for caches to warm and conversations to come back
const WINDOW = { durationMin: 20, warmupMin: 10 };
// deployment sizes tried per model: the fewest GPUs that hold it, then twice and four times as many
const growths = (fit) => {
  const out = [fit];
  let { tp, pp } = fit;
  for (let i = 0; i < 2; i++) { if (tp < 8) tp *= 2; else if (pp < 8) pp *= 2; else break; out.push({ tp, pp }); }
  return out;
};

// One shape loaded to `util`: the request rate one instance takes (geometric bisection), then 8 instances.
function loadShape(c, sim, fit, util, steps) {
  const at = (rpm, instances) => runScenario({ ...c, sim: { ...sim, tp: fit.tp, pp: fit.pp, gpusPerNode: fit.tp, nodes: fit.pp * instances },
    workload: { ...c.workload, durationMin: WINDOW.durationMin, requestsPerMin: rpm } });
  // bracket from a light-load run: busy share ~ rate x mean prefill time
  const light = at(1, 1);
  if (light.error) return { error: light.error };
  const est = light.prefill > 0 ? (util * 60) / light.prefill : 1;
  let lo = est / 8, hi = est * 4, found = false;
  if (at(lo, 1).utilization > util) lo = 0.01; // the estimate was far off: search wide
  for (let i = 0; i < steps; i++) {
    const mid = Math.sqrt(lo * hi);
    const r = at(mid, 1);
    if (r.error) return { error: r.error };
    if (r.utilization > util) hi = mid; else { lo = mid; found = true; }
  }
  if (!found) return { error: "no load keeps the prefill GPUs under the target" };
  const r = at(lo * MARKET_INSTANCES, MARKET_INSTANCES);
  return r.error ? r : { ...r, fit, requestsPerMin: lo * MARKET_INSTANCES };
}

// The reference: of the shapes tried, the one with the lowest cost per request, as a provider would pick.
// Decode GPUs are pooled (sized to the load, not rounded to whole instances).
export function marketRun(c, modelId, { util = 0.6, steps = 8, speed = null } = {}) {
  const priced = (g) => g && g.usd_per_hour != null;
  let gpu = c.gpus.find((g) => g.id === c.sim.gpu);
  if (!priced(gpu) || !fitFor(c, modelId, gpu)) gpu = c.gpus.find((g) => g.id === "b200") ?? c.gpus.find(priced);
  const fit = fitFor(c, modelId, gpu);
  if (!fit) return { model: modelId, error: `does not fit 8 nodes of ${gpu.name}` };
  const decodeGpu = c.gpus.find((g) => g.id === c.sim.decodeGpu);
  const sim = {
    ...c.sim, model: modelId, gpu: gpu.id, decodeGpu: priced(decodeGpu) && fitFor(c, modelId, decodeGpu) ? decodeGpu.id : gpu.id,
    warmupMin: WINDOW.warmupMin, decodePooled: true, decodeSpeed: speed ?? null,
  };
  let best = null, lastError = null;
  for (const shape of growths(fit)) {
    const r = loadShape(c, sim, shape, util, steps);
    if (r.error) { lastError = r.error; continue; }
    if (!best || r.usdPer1kRequests < best.usdPer1kRequests) best = r;
  }
  if (!best) return { model: modelId, error: lastError };
  return {
    model: modelId, gpu: gpu.id, gpuName: gpu.name, tp: best.fit.tp, pp: best.fit.pp, instances: MARKET_INSTANCES, requestsPerMin: best.requestsPerMin,
    utilization: best.utilization, decodeGpus: best.decodeGpus, hit: 1 - best.share.miss, ttft: best.ttft.mean, tpot: best.tpot,
    speedTarget: sim.decodeSpeed, decodeBatch: best.decodeBatch, belowSpeed: best.belowSpeed,
    acceptedTokens: best.acceptedTokens, specSpeedup: best.specSpeedup, specUnavailable: best.specUnavailable, decodeEp: best.decodeEp,
    price: { input: best.price.input, cachedInput: best.price.cachedInput, output: best.price.output, cachedParts: best.price.cachedParts },
  };
}
