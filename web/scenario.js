// From the page's settings to one simulator run. DOM-free; the worker and the tests use it.

import { decodeBatchCap, decodeKvGiB, gpuKvGiB, served } from "./model.js";
import { SPEC_METHODS } from "./presets.js";
import { simulate, synthetic } from "./sim.js";

// A larger cluster is replayed at this many nodes with its per-node load, and its totals scaled up. That is exact
// for sticky routing (a conversation stays on one node) and close for random routing; the page says so.
export const MAX_SIM_NODES = 8;

const byId = (list, id) => (id ? list.find((x) => x.id === id) ?? null : null);

// The speculative decoding the settings pick for a model: {draft, tau, layers, parallel} | null (none), with
// unavailable set when the method needs a drafter the checkpoint does not ship (decode then runs without one).
export function specFor(sim, model) {
  const method = SPEC_METHODS.find((x) => x.id === (sim.spec ?? "none"));
  if (!method || method.id === "none") return { spec: null };
  if (method.requires && !(model.drafters ?? []).includes(method.requires)) return { spec: null, unavailable: method };
  // a drafter the checkpoint ships keeps its own shape (DeepSeek V4.1 Flash's DSpark: block 5, 3 layers)
  const p = method.id === "custom" ? { ...method, ...(sim.specCustom ?? {}) } : { ...method, ...(model.drafter_params?.[method.id] ?? {}) };
  return { spec: { id: method.id, draft: Math.max(0, Math.round(p.draft)), tau: sim.specTau ?? p.tau, layers: p.layers, parallel: !!p.parallel } };
}

// Expert-parallel decode width for a model: a multiple of tp above it, for MoE models; null = the TP instance.
export const epFor = (sim, model, tp) => (sim.decodeEp > tp && sim.decodeEp % tp === 0 && model.expert_share > 0 ? sim.decodeEp : null);

// The setup a scenario runs: tier choices may be overridden (the trade-off sweep does) without touching the config.
// Returns {error} when the model does not fit the GPU.
export function buildRun(c, over = {}, trace = null) {
  const sim = { ...c.sim, ...over };
  const gpu = byId(c.gpus, sim.gpu) ?? c.gpus[0];
  const tp = sim.tp, pp = sim.pp ?? 1;
  const sw = c.software;
  const m = served(c.models[sim.model], { precision: sim.precision, tp, attention: sim.attention ?? "tp" });
  const where = pp > 1 ? `TP=${tp} across ${pp} nodes` : `TP=${tp}`;
  if (sim.nodes % pp) return { error: `${sim.nodes} nodes do not split into instances of ${pp} nodes` };
  const kv = gpuKvGiB(m, gpu, sw, tp, pp);
  if (kv <= 0) return { error: `${sim.model} does not fit ${gpu.name} at ${where}` };
  const decodeGpu = byId(c.gpus, sim.decodeGpu) ?? gpu;
  const ep = sim.decode ? epFor(sim, m, tp) : null;
  if (sim.decode && decodeKvGiB(m, decodeGpu, sw, tp, pp, ep) <= 0)
    return { error: `${sim.model} does not fit the decode GPU ${decodeGpu.name} at ${ep ? `TP=${tp}, experts over ${ep} GPUs` : where}` };
  const sp = sim.decode ? specFor(sim, c.models[sim.model]) : { spec: null };
  // whole instances: a multiple of pp nodes
  const simNodes = trace ? sim.nodes : Math.min(sim.nodes, Math.max(pp, Math.floor(MAX_SIM_NODES / pp) * pp));
  const w = trace ?? synthetic({ ...c.workload, requestsPerMin: (c.workload.requestsPerMin * simNodes) / sim.nodes });
  const cfg = {
    model: m, gpu, software: sw,
    cluster: { nodes: simNodes, gpusPerNode: sim.gpusPerNode, tp, pp }, costScale: sim.nodes / simNodes,
    stageLink: pp > 1 ? byId(c.variants, sim.stageLink) : null,
    gpuGiB: kv, cpuGiB: sim.cpuGiB,
    disk: byId(c.variants, sim.disk), p2p: byId(c.variants, sim.p2p), shared: byId(c.variants, sim.shared),
    routing: sim.routing, seed: 7, warmupS: (sim.warmupMin ?? 0) * 60, retentionH: sim.retentionH, simBlockTokens: sim.simBlockTokens,
    decode: sim.decode ? {
      gpu: decodeGpu, batch: sim.decodeBatch, pooled: !!sim.decodePooled, speed: sim.decodeSpeed ?? null,
      spec: sp.spec, ep, overlap: !!sim.decodeOverlap,
      batchCap: (ctx) => decodeBatchCap(m, decodeGpu, sw, ctx, tp, pp, ep),
    } : null,
    transfer: sim.decode ? byId(c.variants, sim.transfer) : null,
  };
  return { cfg, workload: w, gpu, kvGiB: kv, simNodes, specUnavailable: sp.unavailable?.name ?? null, decodeEp: ep };
}

// The fewest nodes per instance (pipeline stages), then the smallest TP, that hold `modelId` on `gpu` (TP 8 when it
// spans nodes); null when even 8 nodes do not.
export function fitFor(c, modelId, gpu, attention = c.sim.attention ?? "tp", precision = c.sim.precision) {
  for (let pp = 1; pp <= 8; pp++) for (const tp of [1, 2, 4, 8]) {
    if (pp > 1 && tp !== 8) continue;
    if (gpuKvGiB(served(c.models[modelId], { precision, tp, attention }), gpu, c.software, tp, pp) > 0) return { tp, pp };
  }
  return null;
}

export function runScenario(c, over = {}, trace = null) {
  const run = buildRun(c, over, trace);
  if (run.error) return { error: run.error };
  const r = simulate(run.workload, run.cfg);
  return { ...r, kvGiB: run.kvGiB, simNodes: run.simNodes, gpuName: run.gpu.name, specUnavailable: run.specUnavailable, decodeEp: run.decodeEp };
}

// Every combination of the candidate variants per slot (null = slot off), for the trade-off map.
export function combos(candidates) {
  const opts = (ids) => [null, ...(ids ?? [])];
  const out = [];
  for (const disk of opts(candidates.disk)) for (const p2p of opts(candidates.p2p)) for (const shared of opts(candidates.shared)) out.push({ disk, p2p, shared });
  return out;
}

// Points not beaten on both axes (lower is better on x and y), sorted by x. Unpriced points never qualify.
export function paretoFrontier(points) {
  const ok = points.filter((p) => p.priced !== false && Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x || a.y - b.y);
  const front = [];
  for (const p of ok) if (!front.length || p.y < front[front.length - 1].y) front.push(p);
  return front;
}
