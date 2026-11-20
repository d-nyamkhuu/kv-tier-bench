// The simulator's whole configuration: model architectures, GPU profiles, tier variants, software constants, workload
// and cluster. DOM-free. The s1 measurements are not part of it: they are the reference the checks use.

export const CONFIG_VERSION = 4;
export const STORAGE_KEY = "kvtb-sim-config-v4";
// what the estimate uses of each model in s1.json; its measured medians stay out of the config
const ARCH_KEYS = ["hf_id", "hybrid", "params", "weights_bytes", "full_attn_layers", "attn_width", "kv_bytes_per_token", "kv_bytes_per_token_by_tp", "state_bytes_per_prefix", "read_fixed_bytes", "block_size"];
const S1_MODEL_INFO = {
  "qwen3.8-27b": { name: "Qwen3.8-27B (s1)", arch: "dense, Gated DeltaNet 3:1 with full attention", released: "BF16", hidden_size: 5120, num_layers: 64, drafters: ["mtp"], drafter_source: '"mtp_num_hidden_layers": 1' },
  "qwen3-vl-32b": { name: "Qwen3-VL-32B-Instruct (s1)", arch: "dense, grouped-query attention", released: "BF16", hidden_size: 5120, num_layers: 64 },
};
export const FAMILIES = ["fs", "obj", "p2p"];

const clone = (x) => JSON.parse(JSON.stringify(x));

export const DEFAULT_WORKLOAD = {
  requestsPerMin: 300, followUpShare: 0.8, systemTokens: 6000, inputTokens: 800, outputTokens: 400, thinkTimeS: 600,
  durationMin: 60, maxContextTokens: 65536, seed: 1,
  // sim.synthetic's optional realism: length variation, distinct system prompts, arrival pattern
  lengthSpread: 0, systemPrompts: 1, arrivalPattern: "steady",
};
export const DEFAULT_SIM = {
  model: "qwen3.8-27b", gpu: "rtx-pro-6000", nodes: 2, gpusPerNode: 8, tp: 1, pp: 1, stageLink: "p2p-ib-ndr", attention: "dp", precision: "released", routing: "round-robin",
  cpuGiB: 96, disk: null, p2p: null, shared: "obj-aws-s3-standard",
  // decodeBatch: vLLM's default max_num_seqs on GPUs with 70+ GiB (256 below), docs.vllm.ai SchedulerConfig
  // decodeSpeed: tokens/s each request decodes at (null: no per-user speed target)
  decode: true, decodeGpu: "rtx-pro-6000", decodeBatch: 1024, decodeSpeed: null, transfer: "p2p-tcp-400",
  // spec: presets.SPEC_METHODS id; specTau: accepted tokens per step, overriding the method's; specCustom: the
  // "custom" method's numbers. decodeEp: GPUs the experts spread over in decode (null: the TP instance);
  // decodeOverlap: two-batch overlap hides the expert all-to-all (off: the measured EP runs sit nearer to no overlap)
  spec: "none", specTau: null, specCustom: { draft: 7, tau: 3, layers: 5, parallel: true }, decodeEp: null, decodeOverlap: false,
  warmupMin: 20, retentionH: 1, simBlockTokens: 256, peerFirst: false, saveDecode: false,
  // prefillBatchTokens: vLLM's max_num_batched_tokens for continuous prefill batching (8192 is the vLLM, TensorRT-LLM
  // and SGLang default; the simulator page uses it). null runs one request at a time, which the lab's target-busy
  // costs assume.
  prefillBatchTokens: null,
  // the variants the trade-off map combines, per slot (plus "off")
  candidates: { disk: ["fs-nvme-gen5", "fs-nebius-nrd"], p2p: ["p2p-tcp-100", "p2p-roce-400"], shared: ["obj-aws-s3-standard", "obj-aws-s3-express"] },
};

// models: s1's two (sizes from their config.json, the state as s1 read it) and then `presets` (presets.MODELS, each
// derived from its config.json by model.deriveArch).
export function defaultConfig(s1, variants, gpus, software, presets = {}) {
  const fromS1 = Object.entries(s1.models).map(([name, m]) => [name, {
    ...(S1_MODEL_INFO[name] ?? { name }), compute_dtype: "bf16",
    ...Object.fromEntries(ARCH_KEYS.filter((k) => k in m).map((k) => [k, clone(m[k])])),
    total_params: m.params, attn_windows: [],
    sources: [{ label: `${m.hf_id} config.json`, url: `https://huggingface.co/${m.hf_id}/blob/main/config.json` }],
  }]);
  const models = Object.fromEntries([...fromS1, ...Object.entries(clone(presets))]);
  return {
    version: CONFIG_VERSION,
    models,
    software: clone(software),
    variants: clone(variants),
    gpus: clone(gpus),
    workload: { ...DEFAULT_WORKLOAD },
    sim: { ...DEFAULT_SIM },
  };
}

const num = (x) => typeof x === "number" && Number.isFinite(x);
const numOrNull = (x) => x == null || num(x);

// Every problem with a config, as messages; [] means it can be used.
export function validate(c) {
  const errs = [];
  if (!c || typeof c !== "object") return ["not a JSON object"];
  if (c.version !== CONFIG_VERSION) errs.push(`version must be ${CONFIG_VERSION}, got ${JSON.stringify(c.version)}`);
  if (!c.models || !Object.keys(c.models).length) errs.push("models: none");
  for (const [name, m] of Object.entries(c.models || {})) {
    for (const k of ["params", "weights_bytes", "full_attn_layers", "attn_width", "kv_bytes_per_token", "read_fixed_bytes", "state_bytes_per_prefix", "block_size"]) if (!num(m[k])) errs.push(`models.${name}.${k} must be a number`);
  }
  const sw = c.software || {};
  for (const k of ["request_overhead_s", "per_token_overhead_s", "cpu_load_latency_s", "activation_gib_per_gpu"]) if (!(num(sw[k]) && sw[k] >= 0)) errs.push(`software.${k} must be a number >= 0`);
  for (const k of ["attn_eff_ratio", "gpu_memory_utilization"]) if (!(num(sw[k]) && sw[k] > 0)) errs.push(`software.${k} must be a number > 0`);
  const sim = c.sim || {};
  for (const k of ["nodes", "gpusPerNode", "tp"]) if (!(Number.isInteger(sim[k]) && sim[k] >= 1)) errs.push(`sim.${k} must be a whole number >= 1`);
  if (![1, 2, 4, 8].includes(sim.tp)) errs.push("sim.tp must be 1, 2, 4 or 8");
  if (sim.gpusPerNode % sim.tp) errs.push("sim.tp must divide sim.gpusPerNode");
  if (sim.gpuBudgetFraction != null && !(num(sim.gpuBudgetFraction) && sim.gpuBudgetFraction > 0 && sim.gpuBudgetFraction <= 1)) errs.push("sim.gpuBudgetFraction must be greater than 0 and at most 1");
  if (sim.diskCapacityGiB != null && !(num(sim.diskCapacityGiB) && sim.diskCapacityGiB > 0)) errs.push("sim.diskCapacityGiB must be a positive number");
  if (sim.prefillBatchTokens != null && !(Number.isInteger(sim.prefillBatchTokens) && sim.prefillBatchTokens >= 1)) errs.push("sim.prefillBatchTokens must be a whole number of tokens, or empty for one request at a time");
  for (const key of ["peerFirst", "saveDecode"]) if (sim[key] != null && typeof sim[key] !== "boolean") errs.push(`sim.${key} must be a boolean`);
  if (sim.precision != null && !["released", "fp8", "fp4"].includes(sim.precision)) errs.push('sim.precision must be "released", "fp8" or "fp4"');
  if (sim.decodeSpeed != null && !(Number.isFinite(sim.decodeSpeed) && sim.decodeSpeed > 0)) errs.push("sim.decodeSpeed must be a positive number of tokens/s, or null");
  if (sim.specTau != null && !(Number.isFinite(sim.specTau) && sim.specTau >= 1)) errs.push("sim.specTau must be at least 1 (tokens per step), or null");
  if (sim.decodeEp != null && !(Number.isInteger(sim.decodeEp) && sim.decodeEp >= 1 && sim.decodeEp <= 1024)) errs.push("sim.decodeEp must be a whole number of GPUs, or null");
  if (sim.attention != null && !["dp", "tp"].includes(sim.attention)) errs.push('sim.attention must be "dp" or "tp"');
  if (sim.pp != null && !(Number.isInteger(sim.pp) && sim.pp >= 1 && sim.pp <= 8)) errs.push("sim.pp must be a whole number from 1 to 8");
  else if (sim.nodes % (sim.pp ?? 1)) errs.push("sim.pp must divide sim.nodes");
  if (!Array.isArray(c.variants)) errs.push("variants must be a list");
  if (!Array.isArray(c.gpus) || !c.gpus.length) errs.push("gpus must be a non-empty list");
  (c.gpus || []).forEach((g, i) => {
    const at = `gpus[${i}] (${g?.id ?? "?"})`;
    for (const k of ["tflops_bf16", "prefill_mfu", "mem_gib", "mem_bw_tb_s", "host_gib_s", "mbu"]) if (!(num(g?.[k]) && g[k] > 0)) errs.push(`${at}: ${k} must be a number > 0`);
    if (!numOrNull(g?.usd_per_hour)) errs.push(`${at}: usd_per_hour must be a number or empty`);
  });
  const ids = new Set();
  (c.variants || []).forEach((v, i) => {
    const at = `variants[${i}] (${v?.id ?? "?"})`;
    if (!v || typeof v.id !== "string" || !v.id) return errs.push(`${at}: id must be a string`);
    if (ids.has(v.id)) errs.push(`${at}: duplicate id`);
    ids.add(v.id);
    if (!FAMILIES.includes(v.family)) errs.push(`${at}: family must be one of ${FAMILIES.join(", ")}`);
    if (!(num(v.bw_gib_s) && v.bw_gib_s > 0) && !(num(v.link_gbit_s) && v.link_gbit_s > 0)) errs.push(`${at}: needs bw_gib_s or link_gbit_s > 0`);
    for (const key of ["usd_per_1k_put", "usd_per_1k_get"]) {
      if (v[key] != null && !(num(v[key]) && v[key] >= 0)) errs.push(`${at}: ${key} must be nonnegative`);
    }
    if (!numOrNull(v.latency_s)) errs.push(`${at}: latency_s must be a number or empty`);
    if (!Array.isArray(v.sources) || v.sources.some((s) => typeof s?.url !== "string" || !/^https:\/\//.test(s.url))) errs.push(`${at}: sources need https:// urls`);
    for (const k of ["capacity_gib", "usd_per_gib_month", "usd_per_gib_written", "usd_per_gib_read", "usd_per_hour", "efficiency", "cap_gib_s"]) if (!numOrNull(v[k])) errs.push(`${at}: ${k} must be a number or empty`);
  });
  return errs;
}

export const exportConfig = (c) => JSON.stringify(c, null, 1);

// Parses an exported config; missing workload/sim settings fall back to the defaults. Throws with every problem.
export function importConfig(text, defaults) {
  let c;
  try { c = JSON.parse(text); } catch (e) { throw new Error(`not valid JSON: ${e.message}`); }
  if (c && typeof c === "object" && defaults) {
    c.workload = { ...defaults.workload, ...(c.workload || {}) };
    c.sim = { ...defaults.sim, ...(c.sim || {}) };
    c.gpus = Array.isArray(c.gpus) ? c.gpus : structuredClone(defaults.gpus);
    // GPUs saved before a field existed (nvlink_domain, say) take it from the current presets
    for (const g of c.gpus) {
      const d = defaults.gpus?.find((x) => x.id === g?.id);
      if (d) for (const [k, v] of Object.entries(d)) if (g[k] === undefined) g[k] = structuredClone(v);
    }
    for (const v of Array.isArray(c.variants) ? c.variants : []) {
      const preset = defaults.variants?.find(p => p.id === v?.id);
      if (preset?.usd_per_1k_put != null && v.usd_per_1k_put === undefined &&
          /request fees.*left out/i.test(v.note ?? "")) v.note = preset.note;
      for (const key of ["usd_per_1k_put", "usd_per_1k_get"]) {
        if (preset?.[key] != null && v[key] === undefined) v[key] = preset[key];
      }
    }
    c.software = { ...defaults.software, ...(c.software || {}) };
    // models saved before a field existed (decode's expert shares, say) take it from the current presets; edited
    // fields stay as saved, and a model the save lacks comes back whole
    if (defaults.models) {
      c.models = c.models && typeof c.models === "object" ? c.models : {};
      for (const [id, d] of Object.entries(defaults.models)) {
        const m = c.models[id];
        if (!m || typeof m !== "object") { c.models[id] = structuredClone(d); continue; }
        if (d.kv_layout === "deepseek-v41" && !m.kv_layout && m.kv_bytes_per_token !== 890) {
          m.kv_layout = "custom";
          m.prefix_replay_tokens ??= 0;
        }
        if (d.kv_layout === "deepseek-v41" && !m.kv_layout && m.kv_bytes_per_token === 890) {
          for (const k of ["kv_bytes_per_token", "kv_bytes_per_token_by_tp", "read_fixed_bytes", "state_bytes_per_prefix", "state_window_bytes", "kv_dtype_bytes", "spec", "arch", "released", "note", "sources", "assumptions"]) m[k] = structuredClone(d[k]);
        }
        for (const [k, v] of Object.entries(d)) if (m[k] === undefined) m[k] = structuredClone(v);
      }
    }
  }
  const errs = validate(c);
  if (errs.length) throw new Error(errs.join("\n"));
  return c;
}

// localStorage may be missing, full or blocked (private windows, previews): the page works without it.
export function loadSaved(storage, defaults) {
  try {
    const text = storage?.getItem(STORAGE_KEY);
    return text ? importConfig(text, defaults) : null;
  } catch {
    return null;
  }
}

export function save(storage, c) {
  try { storage?.setItem(STORAGE_KEY, exportConfig(c)); return true; } catch { return false; }
}

export function clear(storage) {
  try { storage?.removeItem(STORAGE_KEY); } catch { /* nothing to clear */ }
}
