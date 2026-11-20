// From the page's settings to one simulator run. DOM-free; the worker and the tests use it.

import { decodeBatchCap, decodeKvGiB, effectiveLink, GIB, gpuKvGiB, perSeqBytes, served } from "./model.js";
import { SPEC_METHODS } from "./presets.js";
import { simulate, synthetic } from "./sim.js";

// A larger cluster is replayed at this many nodes with its per-node load, and its totals scaled up. The reduced cluster
// approximates the number of distinct peers and the chance of returning to a replica.
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

// Dense non-MLA models cannot shard weights via data-parallel + expert-parallel attention across TP GPUs.
const effectiveAttention = (rawModel, tp, requested) =>
  tp > 1 && !(rawModel?.expert_share > 0) && !rawModel?.spec?.layers?.some((g) => g.type === "mla") ? "tp" : (requested ?? "tp");

// The setup a scenario runs: tier choices may be overridden (the trade-off sweep does) without touching the config.
// Returns {error} when the model does not fit the GPU.
export function buildRun(c, over = {}, trace = null, maxSimNodes = MAX_SIM_NODES) {
  const sim = { ...c.sim, ...over };
  const gpu = byId(c.gpus, sim.gpu) ?? c.gpus[0];
  const tp = sim.tp, pp = sim.pp ?? 1;
  const sw = c.software;
  const rawModel = c.models[sim.model];
  let attention = effectiveAttention(rawModel, tp, sim.attention);
  let m = served(rawModel, { gpu, precision: sim.precision, tp, attention });
  if (attention === "dp" && tp > 1 && gpuKvGiB(m, gpu, sw, tp, pp) <= 0) {
    attention = "tp";
    m = served(rawModel, { gpu, precision: sim.precision, tp, attention });
  }
  const where = pp > 1 ? `TP=${tp} across ${pp} nodes` : `TP=${tp}`;
  if (sim.nodes % pp) return { error: `${sim.nodes} nodes do not split into instances of ${pp} nodes` };
  const kv = gpuKvGiB(m, gpu, sw, tp, pp);
  if (kv <= 0) return { error: `${sim.model} does not fit ${gpu.name} at ${where}` };
  const decodeGpu = byId(c.gpus, sim.decodeGpu) ?? gpu;
  const decodeModel = decodeGpu.id === gpu.id ? m : served(rawModel, {
    gpu: decodeGpu, precision: sim.precision, tp, attention,
  });
  if (sim.decode && m.kv_layout === "deepseek-v41" && m.kv_bytes_per_token !== decodeModel.kv_bytes_per_token)
    return { error: "DeepSeek V4.1 requires matching KV record formats on the prefill and decode GPUs." };
  const ep = sim.decode ? epFor(sim, m, tp) : null;
  const decTp = decodeModel.attention === "dp" ? 1 : tp;
  const decEp = decodeModel.attention === "dp" ? Math.max(tp, ep ?? tp) : ep;
  const decKvGiB = ep ? decodeKvGiB(decodeModel, decodeGpu, sw, decTp, 1, decEp)
    : gpuKvGiB(decodeModel, decodeGpu, sw, tp, pp) / (decodeModel.attention === "dp" ? tp : 1);
  if (sim.decode && decKvGiB <= 0)
    return { error: `${sim.model} does not fit the decode GPU ${decodeGpu.name} at ${ep ? `TP=${tp}, experts over ${ep} GPUs` : where}` };
  const sp = sim.decode ? specFor(sim, c.models[sim.model]) : { spec: null };
  // whole instances: a multiple of pp nodes
  const simNodes = trace && !trace.syntheticScaled ? sim.nodes : Math.min(sim.nodes, Math.max(pp, Math.floor(maxSimNodes / pp) * pp));
  const w = trace ? (trace.syntheticScaled ? trace.workload : trace) : synthetic({ ...c.workload, requestsPerMin: (c.workload.requestsPerMin * simNodes) / sim.nodes });
  const disk = byId(c.variants, sim.disk);
  const shared = byId(c.variants, sim.shared);
  if (shared && shared.family !== "obj")
    return { error: "The shared tier accepts object storage. Choose a private disk in the local disk slot." };
  const gpuBudget = kv * (sim.gpuBudgetFraction ?? 1);
  const workerScale = m.attention === "dp" ? tp : 1;
  const resolveLink = (id) => effectiveLink(byId(c.variants, id), gpu, sim.nodes, sim.gpusPerNode);
  const transferVariant = sim.decode
    ? (decodeGpu.id === gpu.id ? resolveLink(sim.transfer) : byId(c.variants, sim.transfer))
    : null;
  const cfg = {
    model: m, gpu, software: sw,
    cluster: { nodes: simNodes, gpusPerNode: sim.gpusPerNode, tp, pp }, costScale: sim.nodes / simNodes,
    stageLink: pp > 1 ? resolveLink(sim.stageLink) : null,
    physicalKvGiB: kv / workerScale, gpuGiB: gpuBudget, cpuGiB: sim.cpuGiB,
    disk: disk && sim.diskCapacityGiB != null ? { ...disk, capacity_gib: sim.diskCapacityGiB } : disk,
    p2p: simNodes > pp && sim.cpuGiB > 0 ? resolveLink(sim.p2p) : null,
    shared: shared && sim.sharedCapacityGiB != null ? { ...shared, capacity_gib: sim.sharedCapacityGiB } : shared,
    durationMin: trace && !trace.syntheticScaled ? undefined : c.workload.durationMin, routing: sim.routing,
    peerFirst: sim.peerFirst, saveDecode: sim.saveDecode, skipSlowLoads: !!sim.skipSlowLoads,
    // The v0.5.7 MLA-only connector stores one copy per pipeline stage.
    // Each TP rank still consumes GPU memory for its own replicated KV.
    externalCopiesPerGpuCopy: m.kv_layout === "deepseek-v41" && attention === "tp" ? 1 / tp : 1,
    objectRanks: (attention === "dp" || m.kv_layout === "deepseek-v41" ? 1 : tp) * pp,
    seed: 7, warmupS: (sim.warmupMin ?? 0) * 60, retentionH: sim.retentionH, simBlockTokens: sim.simBlockTokens,
    prefillBatchTokens: sim.prefillBatchTokens ?? null, cacheLifetimeS: sim.cacheLifetimeS ?? null, sharedLifetimeS: sim.sharedLifetimeS ?? null, minCacheTokens: sim.minCacheTokens ?? 0,
    decode: sim.decode ? {
      model: decodeModel, gpu: decodeGpu, batch: sim.decodeBatch, pooled: !!sim.decodePooled, speed: sim.decodeSpeed ?? null,
      spec: sp.spec, tp: decTp, ep: decEp, wideEp: ep, link: byId(c.variants, sim.transfer), overlap: !!sim.decodeOverlap,
      batchCap: (ctx) => Math.floor((decKvGiB * GIB) / perSeqBytes(decodeModel, ctx)),
    } : null,
    transfer: transferVariant,
  };
  return { cfg, workload: w, gpu, kvGiB: gpuBudget, simNodes, specUnavailable: sp.unavailable?.name ?? null, decodeEp: ep };
}

// The fewest nodes per instance (pipeline stages), then the smallest TP, that hold `modelId` on `gpu` (TP 8 when it
// spans nodes). With a fixed inventory, consider only layouts that divide those nodes and GPUs evenly.
export function fitFor(c, modelId, gpu, attention = c.sim.attention ?? "tp", precision = c.sim.precision, inventory = null, context = 0) {
  const rawModel = c.models[modelId];
  for (let pp = 1; pp <= 8; pp++) for (const tp of [1, 2, 4, 8]) {
    if (inventory) {
      if (inventory.nodes % pp || inventory.gpusPerNode % tp) continue;
    } else if (pp > 1 && tp !== 8) continue;
    const prefAttn = effectiveAttention(rawModel, tp, attention);
    for (const attn of [prefAttn, "tp"]) {
      const model = served(rawModel, { gpu, precision, tp, attention: attn });
      const workerScale = model.attention === "dp" ? tp : 1;
      const available = (gpuKvGiB(model, gpu, c.software, tp, pp) * GIB) / workerScale;
      if (available > 0 && (!context || perSeqBytes(model, context) <= available))
        return { tp, pp, ...(attn !== prefAttn ? { attention: attn } : {}) };
      if (attn === "tp") break;
    }
  }
  return null;
}

export function runScenario(c, over = {}, trace = null, { maxSimNodes, observeRequest, observeMinuteEnd } = {}) {
  const run = buildRun(c, over, trace, maxSimNodes);
  if (run.error) return { error: run.error };
  const r = simulate(run.workload, run.cfg, observeRequest, observeMinuteEnd);
  return { ...r, kvBytesPerToken: run.cfg.model.kv_bytes_per_token, windowBytes: run.cfg.model.state_window_bytes ?? 0, replayTokens: run.cfg.model.prefix_replay_tokens ?? 0, kvGiB: run.kvGiB, simNodes: run.simNodes, gpuName: run.gpu.name, specUnavailable: run.specUnavailable, decodeEp: run.decodeEp };
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
