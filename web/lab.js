// The tier lab: one cache tier's properties (latency, bandwidth, size, price) as numbers to set, sweeps over one of
// them, and the cost of serving with the cluster loaded to a target busy share. DOM-free; the worker and tests use it.
//
// Why a target busy share: every GPU is paid for the whole time, so at a fixed cluster and load a faster tier cuts TTFT
// but not the bill. What it buys is headroom: prefill takes less GPU time per request, so the same GPUs serve more.
// Loading each setup until its prefill GPUs are busy TARGET_BUSY of the time (as the provider comparison does) turns
// that headroom into $ per request.

import { variantBandwidth } from "./model.js";
import { fitFor, runScenario } from "./scenario.js";

export const TARGET_BUSY = 0.6;
export const LAB_SLOTS = ["disk", "p2p", "shared"];
export const SLOT_NAME = { disk: "Local disk", p2p: "Peer nodes", shared: "Shared storage" };
export const labId = (slot) => `lab-${slot}`;
export const isLab = (id) => typeof id === "string" && id.startsWith("lab-");
const SWEEP_ID = "__sweep";

// The properties a slot's tier exposes. `zero` puts 0 at the slider's left end (a free tier); `unbounded` puts "no
// limit" (null) at its right end.
export function propsFor(slot) {
  const props = [
    { key: "latency", label: "Latency", unit: "ms", min: 0.01, max: 1000,
      get: (v) => (v.latency_s ?? 0) * 1000, set: (v, x) => { v.latency_s = x / 1000; }, fields: ["latency_s"] },
    { key: "bandwidth", label: "Bandwidth", unit: "GiB/s", min: 0.1, max: 400,
      get: (v) => variantBandwidth(v),
      // a network or object-store preset derives its rate from a link; a set rate replaces the whole derivation
      set: (v, x) => { v.bw_gib_s = x; v.efficiency = 1; v.link_gbit_s = null; v.cap_gib_s = null; }, fields: ["bw_gib_s"] },
  ];
  if (slot !== "p2p") props.push({ key: "size", label: slot === "disk" ? "Size per node" : "Size", unit: "GiB", min: 16, max: 2 ** 20, unbounded: true,
    get: (v) => v.capacity_gib ?? null, set: (v, x) => { v.capacity_gib = x; }, fields: ["capacity_gib"] });
  props.push(slot === "p2p"
    ? { key: "price", label: "Price", unit: "$/h per node", min: 0.01, max: 20, zero: true,
      get: (v) => v.usd_per_hour ?? 0, set: (v, x) => { v.usd_per_hour = x; }, fields: ["usd_per_hour"] }
    : { key: "price", label: "Price", unit: "$/GiB-month", min: 0.001, max: 2, zero: true,
      get: (v) => v.usd_per_gib_month ?? 0, set: (v, x) => { v.usd_per_gib_month = x; }, fields: ["usd_per_gib_month"] });
  return props;
}

export const propOf = (slot, key) => propsFor(slot).find((p) => p.key === key) ?? null;

// A copy of `v` with one property set, marked edited: what a dragged slider or one sweep point runs.
export function withProp(v, slot, key, x, id = v.id) {
  const p = propOf(slot, key);
  const out = { ...structuredClone(v), id };
  p.set(out, x);
  out.edited = [...new Set([...(out.edited ?? []), ...p.fields])];
  return out;
}

// The lab's own variant for a slot, started from a preset. Its numbers are the user's: no sources, no assumptions.
export function labFrom(v, slot) {
  return {
    ...structuredClone(v), id: labId(slot), name: `${v.name} (edited in the tier lab)`, short: `${v.short ?? v.name} (edited)`,
    custom: true, lab: true, from: v.id, sources: [], assumptions: [], edited: [], note: `Started from ${v.name}; the values are set in the tier lab.`,
  };
}

// n values from min to max, evenly spaced on a log scale, with `include` added where it falls (the current value)
export function logSpace(min, max, n, include = null) {
  const out = Array.from({ length: n }, (_, i) => min * (max / min) ** (i / (n - 1)));
  if (include != null && include > min && include < max && !out.some((x) => Math.abs(x / include - 1) < 0.02)) out.push(include);
  return out.sort((a, b) => a - b);
}

const withRate = (c, rpm) => ({ ...c, workload: { ...c.workload, requestsPerMin: rpm } });

// The setup at the config's load (`first`, the TTFT a user sees), then loaded until its prefill GPUs are busy `target`
// of the time (`cap`, what serving costs at that utilization). The busy share grows nearly in proportion to the rate,
// so a few proportional corrections land within a few percent.
export function atBusy(c, over = {}, { target = TARGET_BUSY, first = null, steps = 4 } = {}) {
  const r0 = first ?? runScenario(c, over);
  if (r0.error || !r0.requests) return { first: r0, cap: null };
  let rpm = c.workload.requestsPerMin, r = r0;
  for (let i = 0; i < steps && Math.abs(r.utilization / target - 1) > 0.03; i++) {
    if (!(r.utilization > 0)) break;
    rpm = Math.min(2e5, Math.max(0.05, (rpm * target) / Math.min(r.utilization, 4)));
    r = runScenario(withRate(c, rpm), over);
    if (r.error || !r.requests) return { first: r0, cap: null };
  }
  // the last run lands within a few percent of the target; the rest is closed in proportion: prefill GPUs cost the
  // same per hour, so their $ per request scale with 1 / rate, while decode (sized to the load) and storage (written
  // per request) stay per request
  const k = r.utilization > 0 ? r.utilization / target : 1;
  const prefillShare = r.usd?.total > 0 ? r.usd.prefill / r.usd.total : 0;
  const usdPer1k = r.usdPer1kRequests * (1 - prefillShare + prefillShare * k);
  // rpm: the rate at the target; ran: the rate the last run used (its ttft is at that rate)
  return { first: r0, cap: { rpm: rpm / k, ran: rpm, utilization: r.utilization, usdPer1k, priced: r.priced, ttft: r.ttft.mean } };
}

// One point of a sweep over `key` for `slot`'s tier `base` (null x: the slot off).
export function sweepPoint(c, slot, base, key, x, { capacity = true } = {}) {
  let cc = c, over = { [slot]: null };
  if (x !== null) {
    const v = withProp(base, slot, key, x, SWEEP_ID);
    cc = { ...c, variants: [...c.variants.filter((y) => y.id !== SWEEP_ID), v] };
    over = { [slot]: SWEEP_ID };
  }
  const r = capacity ? atBusy(cc, over) : { first: runScenario(cc, over), cap: null };
  const f = r.first;
  if (f.error) return { x, error: f.error };
  return {
    x, ttft: f.requests ? f.ttft.mean : null, p90: f.requests ? f.ttft.p90 : null, usd: f.usdPer1kRequests, utilization: f.utilization,
    overloaded: f.utilization >= 1, priced: f.priced, hit: f.share ? 1 - f.share.miss : null, slotShare: f.share?.[slot] ?? 0,
    capUsd: r.cap?.priced ? r.cap.usdPer1k : null, capRpm: r.cap?.rpm ?? null,
  };
}

// Across models: each model on the deployment that holds it (the page's GPU when it is priced and holds the model,
// else B200: the smallest TP, then the fewest nodes, `instances` of them), at the request rate that keeps GPU + CPU
// alone `target` busy. At that same rate: TTFT with GPU + CPU and with the page's tiers; then each setup's $ per 1k
// requests loaded to `target`.
export const COMPARE_INSTANCES = 4;
const WINDOW = { durationMin: 20, warmupMin: 10 };
const BASELINE = { disk: null, p2p: null, shared: null };

export function modelCompare(c, modelId, { target = TARGET_BUSY, instances = COMPARE_INSTANCES } = {}) {
  const priced = (g) => g && g.usd_per_hour != null;
  let gpu = c.gpus.find((g) => g.id === c.sim.gpu);
  if (!priced(gpu) || !fitFor(c, modelId, gpu)) gpu = c.gpus.find((g) => g.id === "b200") ?? c.gpus.find(priced);
  const fit = gpu ? fitFor(c, modelId, gpu) : null;
  if (!fit) return { model: modelId, error: `does not fit 8 nodes of ${gpu?.name ?? "any GPU"}` };
  const decodeGpu = c.gpus.find((g) => g.id === c.sim.decodeGpu);
  const cc = {
    ...c,
    sim: { ...c.sim, model: modelId, gpu: gpu.id, decodeGpu: priced(decodeGpu) && fitFor(c, modelId, decodeGpu) ? decodeGpu.id : gpu.id,
      tp: fit.tp, pp: fit.pp, gpusPerNode: fit.tp, nodes: fit.pp * instances, warmupMin: WINDOW.warmupMin, decodePooled: true },
    workload: { ...c.workload, durationMin: WINDOW.durationMin },
  };
  // a light run sizes the first guess at the rate: busy share ~ rate x prefill time per request
  const light = runScenario(withRate(cc, 1));
  if (light.error) return { model: modelId, error: light.error };
  const guess = light.prefill > 0 ? (target * 60 * instances) / light.prefill : 60;
  const base = atBusy(withRate(cc, guess), BASELINE, { target, steps: 5 });
  if (!base.cap) return { model: modelId, error: base.first.error ?? "no requests" };
  // the tier arm gets exactly the traffic the baseline's TTFT was measured at
  const at = withRate(cc, base.cap.ran);
  const tier = atBusy(at, {}, { target });
  if (tier.first.error) return { model: modelId, error: tier.first.error };
  return {
    model: modelId, gpuName: gpu.name, tp: fit.tp, pp: fit.pp, instances, rpm: base.cap.ran,
    base: { ttft: base.cap.ttft, usd: base.cap.priced ? base.cap.usdPer1k : null },
    tier: { ttft: tier.first.ttft.mean, usd: tier.cap?.priced ? tier.cap.usdPer1k : null, rpm: tier.cap?.rpm ?? null, hit: 1 - tier.first.share.miss },
  };
}
