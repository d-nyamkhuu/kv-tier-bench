import { MODELS } from "./presets.js";
import { fitFor } from "./scenario.js";

// The simulator page's settings as plain data: defaults, scenarios, the URL hash and the replay config.
// DOM-free; simulator.js and the tests use it.

// Serving defaults applied whenever a model is chosen: FP8 for models released in BF16 (what providers serve; it halves
// the weights and often the GPUs per instance, so less KV is copied across TP ranks and sent to decode), and RDMA
// between nodes. Over TCP at 400 Gbit/s (about 18 GiB/s in practice) the prefill-to-decode KV transfers of long chats
// saturate the link and TTFT climbs without bound.
export const DEFAULT_LINK = "p2p-ib-ndr";
export const modelDefaults = model => ({ precision: (model?.compute_dtype ?? "bf16") === "bf16" ? "fp8" : "released", interconnect: DEFAULT_LINK });
const forModel = settings => ({ ...settings, ...modelDefaults(MODELS[settings.model]) });

// The default uses every tier: MiniMax-M3 on two B200 nodes fills its GPU cache within the hour, and each tier
// below is larger than the one above, since a tier written as chunks are computed serves hits only then.
export const DEFAULT_SETTINGS = Object.freeze({
  model: "minimax-m3", precision: "fp8", gpu: "b200", nodes: 2, gpusPerNode: 8, interconnect: "p2p-ib-ndr",
  rate: 480, turns: 15, pause: 2, pattern: "steady", routing: "round-robin",
  cpu: 768, disk: "fs-nvme-gen5", diskSize: 1536, lookup: "peer", backing: "obj-aws-s3-standard",
  start: "warm", hours: 1, spread: "moderate", prompts: 1, batch: 8192, priceAt: "0.6",
  admission: "prompt", prefix: 6000, input: 800, output: 400, gpuBudget: 100,
});

// Message and reply lengths vary around their means with this coefficient of variation.
export const LENGTH_SPREAD = { fixed: 0, moderate: 0.5, long: 1.5 };
export const WARMUP_MIN = 20;

// Each scenario is a complete set of settings, so choosing one never leaves a hidden value behind. All run large
// models on two B200 nodes, with CPU and SSD tiers sized past the GPU cache on their node where they should serve hits.
const LIGHT = { ...DEFAULT_SETTINGS, rate: 120, turns: 5, pause: 5, cpu: 128, disk: "", diskSize: 8, lookup: "stock" };
export const SCENARIOS = [
  {
    id: "tiers", name: "Every tier in use",
    description: "MiniMax-M3 (428B, FP8) at 480 req/min with 15-turn chats: every tier serves hits.",
    settings: forModel(DEFAULT_SETTINGS),
  },
  {
    id: "ample", name: "Plenty of GPU memory",
    description: "DeepSeek V4.1 Flash: the whole hour fits in GPU memory.",
    settings: forModel({ ...LIGHT, model: "deepseek-v4.1-flash" }),
  },
  {
    id: "gpu", name: "GPU cache fills",
    description: "Kimi K3 (2.8T) at 480 req/min: the GPU cache fills in minutes.",
    settings: forModel({ ...LIGHT, model: "kimi-k3", rate: 480, cpu: 1536 }),
  },
  {
    id: "disk", name: "SSD fills",
    description: "Mistral Large 4 (1T) at 480 req/min: the SSD fills within the hour.",
    settings: forModel({ ...LIGHT, model: "mistral-large-4", rate: 480, cpu: 64, disk: "fs-nvme-gen5", diskSize: 768 }),
  },
];

// The cluster to use after a model, GPU or precision change. Returns null when the current cluster already holds the
// model, else {nodes, gpusPerNode} for the smallest growth that does (more nodes only when one node cannot hold it),
// or {error} when no supported layout fits this GPU.
export function fitCluster(config, settings) {
  const gpu = config.gpus.find(g => g.id === settings.gpu);
  if (!gpu || !config.models[settings.model]) return null;
  const attention = config.sim.attention;
  const fits = (nodes, gpusPerNode) => fitFor(config, settings.model, gpu, attention, settings.precision, { nodes, gpusPerNode });
  if (fits(settings.nodes, settings.gpusPerNode)) return null;
  const need = fitFor(config, settings.model, gpu, attention, settings.precision);
  if (!need) return { error: `${config.models[settings.model].name} does not fit any layout of up to 8 nodes × 8 ${gpu.name}. Choose a GPU with more memory or a smaller model.` };
  const gpusPerNode = Math.min(8, Math.ceil(Math.max(settings.gpusPerNode, need.tp) / need.tp) * need.tp);
  const nodes = Math.ceil(settings.nodes / need.pp) * need.pp;
  return fits(nodes, gpusPerNode) ? { nodes, gpusPerNode } : { nodes: Math.max(nodes, need.pp), gpusPerNode: 8 };
}

const NUMERIC = new Set(["nodes", "gpusPerNode", "hours", "rate", "turns", "pause", "cpu", "diskSize", "prompts", "batch", "prefix", "input", "output", "gpuBudget"]);
const KEYS = Object.keys(DEFAULT_SETTINGS);

// The price basis changes only how tokens are priced, so it does not make a scenario custom.
export const sameSettings = (a, b) => KEYS.every(key => key === "priceAt" || a[key] === b[key]);
export const scenarioFor = settings => SCENARIOS.find(s => sameSettings(s.settings, settings)) ?? null;

// Only the settings that differ from the defaults go into the hash, so a default page keeps a clean URL.
// diskSize null (the whole drive) is written as "full".
export function encodeSettings(settings) {
  const params = new URLSearchParams();
  for (const key of KEYS) {
    if (settings[key] === DEFAULT_SETTINGS[key]) continue;
    params.set(key, settings[key] == null ? "full" : String(settings[key]));
  }
  return params.toString();
}

// Unknown keys and malformed numbers are ignored; the page checks values against its own options.
export function decodeSettings(hash) {
  const params = new URLSearchParams(String(hash ?? "").replace(/^#/, ""));
  const settings = { ...DEFAULT_SETTINGS };
  for (const key of KEYS) {
    if (!params.has(key)) continue;
    const text = params.get(key);
    if (key === "diskSize" && text === "full") settings.diskSize = null;
    else if (NUMERIC.has(key)) {
      const value = Number(text);
      if (text !== "" && Number.isFinite(value)) settings[key] = value;
    } else settings[key] = text;
  }
  return settings;
}

// Write the settings into a replay config. Returns the GPU budget fraction, which replay() takes separately.
// Turns per chat average 1 / (1 - follow-up share): a chat continues after each turn with that probability.
export function applySettings(config, settings) {
  const warmupMin = settings.start === "cold" ? 0 : WARMUP_MIN;
  Object.assign(config.workload, {
    requestsPerMin: settings.rate, followUpShare: 1 - 1 / Math.max(1, settings.turns), thinkTimeS: settings.pause * 60,
    systemTokens: settings.prefix, inputTokens: settings.input, outputTokens: settings.output,
    durationMin: 60 * (settings.hours || 1) + warmupMin, lengthSpread: LENGTH_SPREAD[settings.spread] ?? 0,
    systemPrompts: settings.prompts, arrivalPattern: settings.pattern === "bursty" ? "bursty" : "steady",
  });
  Object.assign(config.sim, {
    model: settings.model, precision: settings.precision, gpu: settings.gpu, decodeGpu: settings.gpu,
    nodes: settings.nodes, gpusPerNode: settings.gpusPerNode, routing: settings.routing,
    peerFirst: settings.lookup === "peer", saveDecode: settings.admission === "reply",
    p2p: settings.interconnect, stageLink: settings.interconnect, transfer: settings.interconnect,
    cpuGiB: settings.cpu, shared: settings.backing || null, disk: settings.disk || null,
    diskCapacityGiB: settings.diskSize, warmupMin, prefillBatchTokens: settings.batch > 0 ? settings.batch : null,
  });
  return settings.gpuBudget / 100;
}

// A note when CPU memory cannot serve hits, or null. Every computed chunk is copied to GPU and CPU memory as it is
// computed (LMCache's write-through store), and both evict least recently used first, so a CPU tier no larger than the
// GPU cache holds only chunks the GPU still has. Shown only once the GPU cache evicts: before that, nothing needs CPU.
export const CPU_SLIDER_MAX_GIB = 2048; // DGX B200 ships with 2 TB of system memory
export function cpuSizingNote(settings, result) {
  if (!(settings.cpu > 0) || result.firstEviction == null || !(result.gpuGiB > 0)) return null;
  const gpuPerNode = (result.gpuGiB * (result.instances ?? 1)) / settings.nodes;
  if (settings.cpu > gpuPerNode) return null;
  const gib = x => (x >= 10 ? Math.round(x).toLocaleString("en-US") : x.toFixed(1)) + " GiB";
  const share = result.share?.cpu ?? 0;
  const served = share > 0 && share < 0.001 ? "under 0.1%" : (share * 100).toFixed(1) + "%";
  const target = 1.5 * gpuPerNode;
  return `CPU memory is no larger than the GPU cache (${gib(gpuPerNode)} per node), so it served only ${served} of tokens.`
    + (target <= CPU_SLIDER_MAX_GIB ? ` Try ${gib(target)} or more.` : "");
}

// Whether TTFT is still climbing at the end of the run, as {from, to, cause} or null. Compares the request-weighted
// mean of the last sixth of the minutes with the middle fifth: a queue that grows without bound (a saturated link,
// disk or prefill) keeps rising, while a warming cache levels off. The cause is the component that grew the most.
export function risingTtft(result) {
  const minutes = result?.minutes ?? [];
  if (minutes.length < 30) return null;
  const mean = (list, key) => {
    const n = list.reduce((sum, m) => sum + (m[key] != null ? m.count : 0), 0);
    return n ? list.reduce((sum, m) => sum + (m[key] ?? 0) * m.count, 0) / n : null;
  };
  const middle = minutes.slice(Math.floor(minutes.length * 0.4), Math.floor(minutes.length * 0.6));
  const end = minutes.slice(-Math.ceil(minutes.length / 6));
  const from = mean(middle, "mean"), to = mean(end, "mean");
  if (from == null || to == null || to < 1.5 * from || to - from < 0.2) return null;
  const growth = { transfer: mean(end, "transfer") - mean(middle, "transfer"), load: mean(end, "load") - mean(middle, "load"),
    wait: mean(end, "wait") - mean(middle, "wait") };
  const cause = Object.entries(growth).sort((a, b) => b[1] - a[1])[0][0];
  return { from, to, cause };
}

// Per-token prices with the prefill GPUs' cost scaled to a target busy share. A provider runs its GPUs near capacity,
// so pricing an idle cluster overstates the price (the advanced lab priced at 60% busy). priceAt "actual" keeps the
// replay's own load; cache-tier costs do not scale.
export const PRICE_TARGETS = { actual: "This replay's load", "0.4": "40% prefill busy", "0.6": "60% prefill busy", "0.8": "80% prefill busy" };
export function pricesAt(prices, utilization, priceAt) {
  if (!prices) return null;
  const target = Number(priceAt);
  if (!(target > 0) || !(utilization > 0)) return { input: prices.input, cachedInput: prices.cachedInput, target: null, utilization };
  const scale = utilization / target;
  const tiers = (prices.cachedInput ?? 0) - (prices.cachedPrefill ?? 0);
  return {
    input: prices.input == null ? null : prices.input * scale,
    cachedInput: prices.cachedInput == null ? null : (prices.cachedPrefill ?? 0) * scale + tiers,
    target, utilization,
  };
}

// USD per hour for display: whole cents above $1, three decimals below, and never exponent notation.
export function usdPerHour(x) {
  if (x == null || !Number.isFinite(x)) return "–";
  if (x === 0) return "$0";
  const a = Math.abs(x);
  const text = a < 0.001 ? "<$0.001" : "$" + a.toFixed(a < 1 ? 3 : 2);
  return x < 0 ? "−" + text : text;
}
