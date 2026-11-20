import { labFrom, labId } from "./lab.js";
import { fitFor } from "./scenario.js";

// The simulator page's settings as plain data: defaults, scenarios, the URL hash, the replay config and the
// hand-off to the advanced lab. DOM-free; simulator.js, app.js and the tests use it.

export const DEFAULT_SETTINGS = Object.freeze({
  model: "deepseek-v4.1-flash", precision: "released", gpu: "b200", nodes: 2, gpusPerNode: 8, interconnect: "p2p-tcp-400",
  rate: 120, turns: 5, pause: 5, pattern: "steady", routing: "round-robin",
  cpu: 128, disk: "", diskSize: 8, lookup: "stock", backing: "obj-aws-s3-standard",
  start: "warm", spread: "moderate", prompts: 1, batch: 8192,
  admission: "prompt", prefix: 6000, input: 800, output: 400, gpuBudget: 100,
});

// Message and reply lengths vary around their means with this coefficient of variation.
export const LENGTH_SPREAD = { fixed: 0, moderate: 0.5, long: 1.5 };
export const WARMUP_MIN = 20;

// Each scenario is a complete set of settings, so choosing one never leaves a hidden value behind.
const QWEN = { ...DEFAULT_SETTINGS, model: "qwen3.8-27b", gpu: "rtx-pro-6000", cpu: 96 };
export const SCENARIOS = [
  {
    id: "ample", name: "Plenty of GPU memory",
    description: "DeepSeek V4.1 Flash compresses KV to about 1.6 KB per token, so two B200 nodes keep the whole hour in GPU memory.",
    settings: { ...DEFAULT_SETTINGS },
  },
  {
    id: "gpu", name: "GPU cache fills",
    description: "Qwen3.8-27B on RTX PRO 6000 GPUs stores about 64 KiB of KV per token, so the GPU cache fills partway through the hour and older chats are served from CPU memory and S3.",
    settings: QWEN,
  },
  {
    id: "disk", name: "SSD fills",
    description: "Qwen3.8-27B with only 8 GiB of CPU cache per node: evicted chunks move to a 64 GiB local SSD, which fills and starts evicting within the hour.",
    settings: { ...QWEN, cpu: 8, disk: "fs-nvme-gen5", diskSize: 64 },
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

const NUMERIC = new Set(["nodes", "gpusPerNode", "rate", "turns", "pause", "cpu", "diskSize", "prompts", "batch", "prefix", "input", "output", "gpuBudget"]);
const KEYS = Object.keys(DEFAULT_SETTINGS);

export const sameSettings = (a, b) => KEYS.every(key => a[key] === b[key]);
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
    durationMin: 60 + warmupMin, lengthSpread: LENGTH_SPREAD[settings.spread] ?? 0,
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

// The advanced lab's config fields for the same setup. The lab turns peer reads on as a tier of its own, which the
// simulator enables whenever another node holds CPU cache. `layout` is the {tp, pp} the replay fitted. An SSD size
// travels as `diskGiB` and becomes the lab's own edited drive, whose Size control the user can still change.
export function advancedOverrides(settings, layout) {
  const config = { workload: {}, sim: {} };
  applySettings(config, settings);
  const { diskCapacityGiB, ...rest } = config.sim;
  const sim = { ...rest, p2p: settings.cpu > 0 ? settings.interconnect : null, gpuBudgetFraction: settings.gpuBudget / 100 };
  if (layout) Object.assign(sim, { tp: layout.tp, pp: layout.pp });
  if (sim.disk && diskCapacityGiB != null) sim.diskGiB = diskCapacityGiB;
  return { workload: config.workload, sim };
}

export const advancedHash = overrides => "#from-simulator=" + encodeURIComponent(JSON.stringify(overrides));

// The overrides in an advanced-lab URL hash, or null.
export function readAdvancedHash(hash) {
  const match = /^#from-simulator=(.+)$/.exec(String(hash ?? ""));
  if (!match) return null;
  try {
    const overrides = JSON.parse(decodeURIComponent(match[1]));
    return overrides && typeof overrides === "object" ? overrides : null;
  } catch {
    return null;
  }
}

// A copy of `config` with the simulator's choices, keeping only fields the config already has and ids it knows.
export function withOverrides(config, overrides) {
  const next = structuredClone(config);
  const known = {
    model: id => id in next.models, gpu: id => next.gpus.some(g => g.id === id), decodeGpu: id => next.gpus.some(g => g.id === id),
    disk: id => id == null || next.variants.some(v => v.id === id), shared: id => id == null || next.variants.some(v => v.id === id),
    p2p: id => id == null || next.variants.some(v => v.id === id), stageLink: id => next.variants.some(v => v.id === id),
    transfer: id => next.variants.some(v => v.id === id),
  };
  for (const part of ["workload", "sim"]) {
    for (const [key, value] of Object.entries(overrides?.[part] ?? {})) {
      if (!(key in next[part]) && key !== "gpuBudgetFraction") continue;
      if (known[key] && !known[key](value)) continue;
      next[part][key] = value;
    }
  }
  // A size override on the product would hide behind the lab's Size control, so the size goes on a lab drive instead.
  delete next.sim.diskCapacityGiB;
  const size = overrides?.sim?.diskGiB;
  const drive = next.variants.find(v => v.id === next.sim.disk);
  if (drive && Number.isFinite(size) && size > 0) {
    const edited = { ...labFrom(drive.lab ? next.variants.find(v => v.id === drive.from) ?? drive : drive, "disk"), capacity_gib: size };
    edited.edited = ["capacity_gib"];
    next.variants = [...next.variants.filter(v => v.id !== labId("disk")), edited];
    next.sim.disk = edited.id;
  }
  return next;
}

// USD per hour for display: whole cents above $1, three decimals below, and never exponent notation.
export function usdPerHour(x) {
  if (x == null || !Number.isFinite(x)) return "–";
  if (x === 0) return "$0";
  const a = Math.abs(x);
  const text = a < 0.001 ? "<$0.001" : "$" + a.toFixed(a < 1 ? 3 : 2);
  return x < 0 ? "−" + text : text;
}
