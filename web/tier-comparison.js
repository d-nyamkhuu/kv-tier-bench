// Compare backing tiers on the exact layout fitted by the main replay.
import { GIB } from "./model.js";
import { runScenario } from "./scenario.js";

export function compareTiers(config, layout, budgetFraction) {
  const { sim, variants } = config;
  const base = { ...layout, gpuBudgetFraction: budgetFraction, disk: null, shared: null, p2p: null };
  const options = [{ name: "GPU only", over: { cpuGiB: 0 }, billing: "In GPU rental" }];
  if (sim.cpuGiB > 0) {
    options.push({ name: "Local CPU", over: {}, billing: "In node rental" });
    if (sim.p2p && sim.nodes > layout.pp)
      options.push({ name: "Peer CPU", over: { p2p: sim.p2p }, billing: "Uses peer RAM" });
  }
  for (const id of ["fs-nvme-gen4", "fs-nvme-gen5", "obj-aws-s3-standard", "obj-aws-s3-express"]) {
    const product = variants.find(v => v.id === id);
    if (!product) continue;
    const disk = product.family === "fs";
    options.push({ name: product.short, over: disk ? { disk: id } : { shared: id },
      rate: product.usd_per_gib_month == null ? null : product.usd_per_gib_month * 1e9 / GIB,
      billing: product.usd_per_gib_month == null ? "Unpriced" : product.bill === "provisioned" ? "Provisioned" : "Stored" });
  }

  return options.map(({ name, over, rate = null, billing }) => {
    const result = runScenario(config, { ...base, ...over }, null, { maxSimNodes: sim.nodes });
    return { name, billing, rate, mean: result.requests ? result.ttft.mean : null,
      error: result.error ?? (result.requests ? null : "No measured requests") };
  });
}
