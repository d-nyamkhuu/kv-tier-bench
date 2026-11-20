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

export { providerStats } from "./openrouter.js";

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

// Calibrate on the reference cluster so the trial includes cross-node cache misses and transfers.
function loadShape(c, sim, fit, util, steps) {
  const cluster = { ...sim, tp: fit.tp, pp: fit.pp, gpusPerNode: fit.tp, nodes: fit.pp * MARKET_INSTANCES };
  const at = rpm => runScenario({ ...c, sim: cluster,
    workload: { ...c.workload, durationMin: WINDOW.durationMin, requestsPerMin: rpm } });
  const light = at(MARKET_INSTANCES);
  if (light.error) return light;
  const est = light.prefill > 0 ? util * 60 * MARKET_INSTANCES / light.prefill : MARKET_INSTANCES;
  let lo = est / 8, hi = est * 4;
  let best = at(lo);
  if (best.utilization > util) { lo = 0.01; best = null; }
  for (let i = 0; i < steps; i++) {
    const mid = Math.sqrt(lo * hi);
    const r = at(mid);
    if (r.error) return r;
    if (r.utilization > util) hi = mid;
    else { lo = mid; best = r; }
  }
  if (!best?.requests) return { error: "no load keeps the prefill GPUs under the target" };
  return { ...best, fit, requestsPerMin: lo };
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
    prefillBatchTokens: null, // the target-busy search assumes one-at-a-time prefill (lab.atBusy)
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
