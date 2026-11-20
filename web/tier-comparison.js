// Compare cache tiers on the exact layout fitted by the main replay. Each row replays the setup with one tier added
// below GPU memory (CPU memory stays on for the tiers below it) and reports the median TTFT of the requests that read
// from that tier, with the tier's on-demand price per GiB-month.
import { HOST_RAM } from "./presets.js";
import { prepareTrace } from "./replay.js";
import { runScenario } from "./scenario.js";
import { synthetic } from "./sim.js";

const HOURS_PER_MONTH = 730;

export function compareTiers(config, layout, budgetFraction, { onRow, trace = null } = {}) {
  const { sim, variants } = config;
  const gpu = config.gpus.find(g => g.id === sim.gpu);
  const prepared = prepareTrace(trace, sim.warmupMin ?? 0);
  const base = { ...layout, gpuBudgetFraction: budgetFraction, disk: null, shared: null, p2p: null,
    ...(prepared ? { warmupMin: prepared.warmupMin } : {}) };
  const hbm = gpu?.usd_per_hour == null ? null : gpu.usd_per_hour * HOURS_PER_MONTH / gpu.mem_gib;
  const options = [{ name: "GPU only", tier: "gpu", over: { cpuGiB: 0 }, rate: hbm,
    billing: hbm == null ? "Unpriced GPU" : `${gpu.name} rental ÷ ${Math.round(gpu.mem_gib)} GiB of HBM` }];
  if (sim.cpuGiB > 0) {
    options.push({ name: "Local CPU", tier: "cpu", over: {}, rate: HOST_RAM.usd_per_gib_month, billing: "RAM, on-demand" });
    if (sim.p2p && sim.nodes > layout.pp)
      options.push({ name: "Peer CPU", tier: "p2p", over: { p2p: sim.p2p }, rate: HOST_RAM.usd_per_gib_month, billing: "RAM on another node" });
  }
  for (const id of ["fs-nvme-gen4", "fs-nvme-gen5", "obj-aws-s3-standard", "obj-aws-s3-express"]) {
    const product = variants.find(v => v.id === id);
    if (!product) continue;
    const disk = product.family === "fs";
    options.push({ name: product.short, tier: disk ? "disk" : "shared", over: disk ? { disk: id } : { shared: id },
      rate: product.usd_per_gib_month ?? null,
      billing: product.usd_per_gib_month == null ? "Unpriced" : product.bill === "provisioned" ? "Provisioned capacity" : "Stored bytes" });
  }

  // Share a single trace or pre-generated synthetic arrival stream across all rows.
  const sharedWorkload = prepared ? prepared.trace
    : sim.nodes % layout.pp === 0 ? { syntheticScaled: true, workload: synthetic({ ...config.workload }) } : null;
  const rows = [];
  options.forEach(({ name, tier, over, rate, billing }, index) => {
    const times = [];
    const observeRequest = (req, ttft, blocks) => { if (blocks[tier] > 0) times.push(ttft); };
    const result = runScenario(config, { ...base, ...over }, sharedWorkload, { maxSimNodes: sim.nodes, observeRequest });
    times.sort((a, b) => a - b);
    const median = times.length ? (times[(times.length - 1) >> 1] + times[times.length >> 1]) / 2 : null;
    const error = result.error ?? (result.requests ? null : "No measured requests");
    const row = { name, tier, billing, rate, median, readers: times.length, requests: result.requests ?? 0,
      mean: result.requests ? result.ttft.mean : null, error: error ?? (times.length ? null : "No requests read it") };
    rows.push(row);
    onRow?.(rows.slice(), index + 1, options.length);
  });
  return rows;
}
