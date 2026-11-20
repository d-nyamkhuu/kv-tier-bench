// The page: the configurator, the live estimate, the cluster diagram and the trade-off map. Simulations run in
// worker.js; the maths is in model.js, sim.js and scenario.js.

import { defaultConfig, loadSaved, save, validate } from "./config.js";
import { readAdvancedHash, withOverrides } from "./sim-settings.js";
import { loadOpenRouterPrices, showInputPrices } from "./openrouter.js";
import { renderDiagram } from "./diagram.js";
import { drawFitChart, initMethods, renderMethods, slotOf } from "./methods.js";
import { gpuKvGiB, PRECISION_LABELS, served, variantBandwidth } from "./model.js";
import { GPUS, MODELS, MOONCAKE_TRACES, SOFTWARE, SPEC_METHODS, VARIANTS } from "./presets.js";
import { COMPARE_INSTANCES, isLab, LAB_SLOTS, labFrom, labId, logSpace, propOf, propsFor, SLOT_NAME, TARGET_BUSY } from "./lab.js";
import { fitFor, MAX_SIM_NODES, paretoFrontier } from "./scenario.js";
import { parseTrace } from "./sim.js";
import { $, assumptionBadge, cssVar, el, fmt, gibf, numField, pctf, secs, usd } from "./ui.js";

const storage = (() => { try { return window.localStorage; } catch { return null; } })();
const TP_SIZES = [1, 2, 4, 8];
const SLOTS = ["disk", "p2p", "shared"];
const TIER_COLOR = { gpu: "--tier-gpu", cpu: "--tier-cpu", disk: "--tier-disk", p2p: "--tier-p2p", shared: "--tier-shared", miss: "--tier-miss" };

let s1, defaults, cfg, market;
let source = "synthetic";
let trace = null; // a parsed workload from Mooncake or an upload
let estimate = null; // the latest {current, baseline}
let capEst = null; // its cost at the target busy share: {cap, base}
const lastPick = { cpuGiB: 96, disk: null, p2p: null, shared: null }; // what a tier switch turns back on

// ------------------------------------------------------------------------------------------------ helpers

const byId = (list, id) => list.find((x) => x.id === id) ?? null;
const variant = (id) => (id ? byId(cfg.variants, id) : null);
const gpuOf = (id) => byId(cfg.gpus, id) ?? cfg.gpus[0];
const gpuShort = (g) => g.name.split(" (")[0].replace(" Blackwell", "").replace(" SXM", "");
const slotVariants = (slot) => cfg.variants.filter((v) => slotOf(v) === slot);
const fill = (host, ...kids) => host.replaceChildren(...kids.filter((k) => k != null && k !== false && k !== ""));

// What `gib` of a tier holds: tokens of KV, and with a fixed per-prefix state, prefixes of the run's mean prompt
function capacity(gib) {
  if (gib == null) return "no size limit";
  if (!(gib > 0)) return "";
  const m = served(cfg.models[cfg.sim.model], { gpu: gpuOf(cfg.sim.gpu), precision: cfg.sim.precision, tp: cfg.sim.tp, attention: cfg.sim.attention });
  const bytes = gib * 2 ** 30;
  const tok = m.kv_bytes_per_token > 0 ? bytes / m.kv_bytes_per_token : null;
  const big = (x) => (x >= 1e9 ? `${fmt(x / 1e9, 3)}B` : x >= 1e6 ? `${fmt(x / 1e6, 3)}M` : x >= 1e3 ? `${fmt(x / 1e3, 3)}k` : fmt(x, 3));
  const r = estimate?.current;
  const meanPrompt = r && !r.error && r.requests ? r.promptTokens / r.requests : null;
  const parts = [];
  if (tok != null) parts.push(`≈ ${big(tok)} tokens`);
  if (m.read_fixed_bytes > 0 || tok == null) {
    if (meanPrompt) parts.push(`≈ ${big(bytes / (m.read_fixed_bytes + m.kv_bytes_per_token * meanPrompt))} prompts of ${big(meanPrompt)} tokens (with ${fmt(m.read_fixed_bytes / 2 ** 20, 3)} MiB of state each)`);
    else parts.push(`plus ${fmt(m.read_fixed_bytes / 2 ** 20, 3)} MiB of state per prefix`);
  }
  return parts.join(", ");
}

// After a model change: the fewest nodes per instance (pipeline stages), then the smallest TP, that hold the model on
// the current GPU; if it takes more than 8 nodes there, the first priced GPU that holds it in fewer.
const PP_MAX = 8;
function fitModel() {
  const sim = cfg.sim;
  const pick = (g) => fitFor(cfg, sim.model, g);
  let g = gpuOf(sim.gpu);
  let f = pick(g);
  if (!f) { g = cfg.gpus.find((x) => x.usd_per_hour != null && pick(x)) ?? cfg.gpus.find((x) => pick(x)); f = g ? pick(g) : null; }
  if (!f) return; // fits nowhere: the page says so
  if (sim.decodeGpu === sim.gpu || kvFor(gpuOf(sim.decodeGpu), f.tp, f.pp) <= 0) sim.decodeGpu = g.id;
  sim.gpu = g.id; sim.tp = f.tp; sim.pp = f.pp;
  if (sim.gpusPerNode % f.tp) sim.gpusPerNode = Math.ceil(sim.gpusPerNode / f.tp) * f.tp;
  if (sim.nodes % f.pp) sim.nodes = Math.ceil(sim.nodes / f.pp) * f.pp;
}

// a per-node tier: each node of a multi-node instance holds its own stages' layers, so the instance's nodes add up
function perGroup(gib) {
  const pp = cfg.sim.pp ?? 1;
  return pp > 1 && gib != null ? `${capacity(gib * pp)} across an instance's ${pp} nodes` : capacity(gib);
}

function kvFor(gpu, tp = cfg.sim.tp, pp = cfg.sim.pp ?? 1) {
  const m = served(cfg.models[cfg.sim.model], { gpu, precision: cfg.sim.precision, tp, attention: cfg.sim.attention });
  return gpuKvGiB(m, gpu, cfg.software, tp, pp) * (cfg.sim.gpuBudgetFraction ?? 1);
}

// "GPU + CPU + NVMe Gen5 + S3 Standard" for a setup, with tier overrides
function setupLabel(over = {}, { cluster = false } = {}) {
  const sim = { ...cfg.sim, ...over };
  const parts = ["GPU", ...(sim.cpuGiB > 0 ? ["CPU"] : []), ...SLOTS.map((k) => variant(sim[k])).filter(Boolean).map((v) => (v.family === "p2p" ? `peers ${v.short}` : v.short))];
  const tiers = parts.join(" + ");
  return cluster ? `${sim.nodes} × ${sim.gpusPerNode} ${gpuShort(gpuOf(sim.gpu))} · ${tiers}` : tiers;
}

function flash(target) {
  const ids = { gpu: "ctl-gpu", cpu: "tier-cpu", disk: "tier-disk", p2p: "tier-p2p", shared: "tier-shared", load: "ctl-load", nodes: "ctl-nodes", routing: "ctl-routing", decode: "ctl-decode" };
  if (LAB_SLOTS.includes(target) && labSlot !== target) { labSlot = target; renderLab(); runSensSoon(0); }
  const node = $(ids[target] ?? target);
  if (!node) return;
  // open whatever hides it: folded sections (nested too) and the GPU choices
  for (let d = node.closest("details"); d; d = d.parentElement?.closest("details")) d.open = true;
  node.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  const box = node.closest(".lab-panel, .fixed, .field, .setup-sec, .panel") ?? node;
  box.classList.remove("flash");
  void box.offsetWidth;
  box.classList.add("flash");
  (node.matches("input, select, button") ? node : node.querySelector("input:not([disabled]), select:not([disabled]), button:not([disabled])"))?.focus({ preventScroll: true });
}

// ------------------------------------------------------------------------------------------------ changes + runs

let runTimer;
function changed({ methods = false, config = true } = {}) {
  const ok = save(storage, cfg);
  $("config-status").textContent = ok ? "Saved in this browser." : "Not saved (browser storage unavailable); export to keep your changes.";
  if (config) keepFocus(renderConfig);
  if (methods) renderMethods();
  renderDiagramNow();
  clearTimeout(runTimer);
  $("estimate").classList.add("busy");
  runTimer = setTimeout(run, 180);
  runSensSoon();
  modelsStatus();
}

// re-rendering replaces the controls: put the focus back on the same one
const FOCUSABLE = "input, select, button";
function keepFocus(render) {
  const a = document.activeElement;
  const host = a?.id ? null : a?.closest?.("[id]");
  const idx = host ? [...host.querySelectorAll(FOCUSABLE)].indexOf(a) : -1;
  const id = a?.id;
  render();
  const next = id ? $(id) : host && idx >= 0 ? $(host.id)?.querySelectorAll(FOCUSABLE)[idx] : null;
  if (next && next !== document.activeElement && !next.disabled) next.focus({ preventScroll: true });
}

function replaceCfg(c, { reset = false } = {}) {
  cfg = c;
  if (reset) { source = "synthetic"; trace = null; }
  syncLastPick();
  renderConfig();
  renderMethods();
  run();
  runSensSoon(0);
}

// ------------------------------------------------------------------------------------------------ worker

let worker = null;
let jobId = 0;
let sweepKey = null;
let sweepComplete = false;
let points = []; // trade-off map points of the current sweep
let frontier = [];
let ghost = null; // the frontier before the last change that moved it
const show = { nodes: false, gpus: false };

// the map's combos override disk, p2p and shared, so a change to only those leaves them valid
const keyOf = () => {
  const cand = Object.values(cfg.sim.candidates ?? {}).flat();
  return JSON.stringify({ ...cfg, variants: cfg.variants.filter((v) => !isLab(v.id) || cand.includes(v.id)), sim: { ...cfg.sim, disk: null, p2p: null, shared: null }, trace: trace ? trace.name : source });
};

function run() {
  if (!worker) return;
  if (source !== "synthetic" && !trace) { // nothing to replay yet
    jobId++;
    estimate = null;
    $("estimate").classList.remove("busy");
    for (const id of ["est-ttft", "est-usd", "est-cap", "mini-ttft", "mini-usd"]) $(id).textContent = "–";
    for (const id of ["est-ttft-sub", "est-usd-sub", "est-cap-sub", "est-delta", "est-foot"]) $(id).textContent = "";
    $("est-alerts").replaceChildren(el("div", { class: "alert", text: "Load a trace in “Load & serving settings” to see an estimate." }));
    $("map-status").textContent = "Load a trace to fill the map.";
    points = []; frontier = []; sweepKey = null; sweepComplete = false;
    renderDiagramNow();
    scheduleChart();
    return;
  }
  const key = keyOf();
  let series;
  if (key !== sweepKey || !sweepComplete) {
    // only a finished sweep's frontier becomes the ghost; an aborted one is simply redone
    if (key !== sweepKey && sweepComplete && frontier.length) ghost = frontier.map((p) => ({ ...p }));
    sweepKey = key;
    sweepComplete = false;
    points = [];
    series = ["combo", "nodes-yours", "nodes-base", "gpus"];
  } else {
    // your tiers changed: the curves for your setup move, the combos stay
    points = points.filter((p) => p.series === "combo" || p.series === "nodes-base");
    series = ["nodes-yours", "gpus"];
  }
  jobId++;
  $("map-status").textContent = "Simulating…";
  worker.postMessage({ id: jobId, config: cfg, trace: source === "synthetic" ? null : trace, series });
}

function onWorker({ data }) {
  if (data.id !== jobId) return; // an older job
  if (data.kind === "estimate") {
    estimate = data;
    capEst = null;
    $("estimate").classList.remove("busy");
    renderEstimate();
    updateLabLive();
    renderDiagramNow();
    updateFrontier();
    scheduleChart();
  } else if (data.kind === "capacity") {
    capEst = { cap: data.cap, base: data.base };
    renderEstimate();
    updateFrontier();
    scheduleChart();
  } else if (data.kind === "point") {
    points.push(withMetric(data.point));
    $("map-status").textContent = `Simulating setups: ${data.done} of ${data.total}`;
    if (data.point.series === "combo") updateFrontier();
    scheduleChart();
  } else if (data.kind === "sweep-done") {
    sweepComplete = true;
    $("map-status").textContent = "";
    renderExcluded();
    scheduleChart();
  }
}

// ------------------------------------------------------------------------------------------------ configurator

function renderConfig() {
  showInputPrices($("openrouter-input-prices"), market, cfg.sim.model);
  renderGpuPicker();
  renderCluster();
  renderLab();
  renderSliders();
  renderSettings();
}

const fitText = (sim, kv) => {
  const where = `TP ${sim.tp}${(sim.pp ?? 1) > 1 ? ` × ${sim.pp} nodes` : ""}`;
  return kv > 0 ? `${fmt(kv, 3)} GiB KV at ${where}` : `model doesn't fit at ${where}`;
};

function renderGpuPicker() {
  const sim = cfg.sim;
  const priceOf = (g) => (g.usd_per_hour != null ? `$${g.usd_per_hour.toFixed(2)}/h` : "no price");
  $("ctl-gpu").replaceChildren(...cfg.gpus.map((g) => el("option", { value: g.id, selected: g.id === sim.gpu,
    text: `${gpuShort(g)} · ${Math.round(g.mem_gib)} GiB · ${priceOf(g)}${kvFor(g) <= 0 ? " · doesn't fit" : ""}` })));
  $("ctl-gpu").onchange = (ev) => {
    const g = gpuOf(ev.target.value);
    if (sim.decodeGpu === sim.gpu) sim.decodeGpu = g.id;
    sim.gpu = g.id;
    if ((sim.pp ?? 1) > 1 || kvFor(g) <= 0) fitModel(); // across nodes: as few as this GPU needs
    changed();
  };
  const cur = gpuOf(sim.gpu), curKv = kvFor(cur);
  $("gpu-summary").className = `pick-meta${curKv > 0 ? "" : " bad"}`;
  $("gpu-summary").textContent = `${fitText(sim, curKv)}${cur.usd_per_hour == null ? " · no price: costs leave it out" : ""}`;
  const models = Object.keys(cfg.models);
  const opt = (k) => el("option", { value: k, text: cfg.models[k].name ?? k, selected: k === sim.model });
  const inS1 = (k) => k in s1.models;
  $("ctl-model").replaceChildren(
    el("optgroup", { label: "Measured in s1" }, models.filter(inS1).map(opt)),
    el("optgroup", { label: "Other open-weight models" }, models.filter((k) => !inS1(k)).map(opt)));
  $("ctl-model").onchange = (ev) => { sim.model = ev.target.value; fitModel(); changed({ methods: true }); };
  const prec = sim.precision ?? "released";
  $("ctl-precision").replaceChildren(...Object.entries(PRECISION_LABELS).map(([v, t]) => el("button", {
    type: "button", role: "radio", "aria-checked": String(v === prec), text: t,
    title: v === "released" ? `as the checkpoint ships: ${cfg.models[sim.model].released ?? "BF16"}` : v === "fp8" ? "FP8 weights, compute and KV cache" : "NVFP4 weights (4.5 bits per parameter), prefill at the FP8 rate, FP4 decode compute on Blackwell, FP8 KV cache",
    onclick: () => { sim.precision = v; fitModel(); changed({ methods: true }); },
  })));
  const mm = cfg.models[sim.model], mt = served(mm, { gpu: gpuOf(sim.gpu), precision: prec, tp: sim.tp, attention: sim.attention });
  const mib = (b) => `${fmt(b / 2 ** 20, 3)} MiB`;
  const params = (n) => (n >= 1e12 ? `${fmt(n / 1e12, 3)}T` : `${fmt(n / 1e9, 3)}B`);
  $("model-brief").textContent = `· ${params(mm.params)} active · ${fmt(mt.kv_bytes_per_token / 1024, 3)} KiB KV per token · ${fmt(mt.weights_bytes / 1e9, 3)} GB weights`;
  fill($("model-hint"),
    `${mm.arch ?? ""}${mm.arch ? ". " : ""}${fmt(mt.kv_bytes_per_token / 1024, 3)} KiB of KV per token at TP ${sim.tp}`,
    mt.prefix_replay_tokens ? `, rebuilding the last ${mt.prefix_replay_tokens} tokens of attention window on a cache hit` : "",
    mt.read_fixed_bytes > 0 ? `, plus ${mib(mt.read_fixed_bytes)} per cached prefix that does not grow with its length` : "",
    `. ${params(mm.params)} parameters active per token (${params(mm.total_params ?? mm.params)} in all). Released as ${mm.released ?? "BF16"}; served ${prec === "released" ? "as released" : `in ${prec.toUpperCase()}`}: ${fmt(mt.weights_bytes / 1e9, 4)} GB of weights, computed in ${(mt.compute_dtype ?? "bf16").toUpperCase()}${mt.compute_dtype === "fp4" ? (gpuOf(sim.gpu).tflops_fp4 ? " (prefill at the FP8 rate, as measured)" : ` (at FP8 speed: ${gpuShort(gpuOf(sim.gpu))} has no FP4 tensor cores)`) : ""}.`,
    mm.assumptions?.length ? el("span", {}, " ", assumptionBadge(mm.assumptions.join("; "))) : null,
    mm.note ? el("span", {}, ` ${mm.note}`) : null,
    mm.sources?.[0] ? el("span", {}, " ", el("a", { href: mm.sources[0].url, text: "config.json", target: "_blank", rel: "noopener" })) : null);
  const g = gpuOf(sim.gpu);
  fill($("gpu-hint"),
    `${fmt(g.tflops_bf16, 4)} TFLOPS dense BF16 at ${pctf(g.prefill_mfu)} for prefill · ${fmt(g.mem_bw_tb_s, 3)} TB/s memory · ${fmt(g.host_gib_s, 3)} GiB/s from CPU memory. `,
    g.assumptions?.length ? assumptionBadge(g.assumptions.join(", ")) : null,
    g.usd_per_hour == null ? el("span", {}, " No price: costs leave this GPU out.") : null,
  );
}

function stepper(host, label, value, min, max, set, step = 1) {
  const input = el("input", { type: "number", min, max, step: 1, value, "aria-label": label,
    onchange: (ev) => { const v = Math.round(Number(ev.target.value)); if (Number.isFinite(v)) set(Math.min(max, Math.max(min, v))); else ev.target.value = value; } });
  host.replaceChildren(el("span", { class: "field" }, el("span", { text: label })),
    el("div", { class: "ctl" },
      el("button", { type: "button", text: "−", "aria-label": `fewer ${label.toLowerCase()}`, disabled: value - step < min, onclick: () => set(value - step) }),
      input,
      el("button", { type: "button", text: "+", "aria-label": `more ${label.toLowerCase()}`, disabled: value + step > max, onclick: () => set(value + step) })));
}

function renderCluster() {
  const sim = cfg.sim;
  const pp = sim.pp ?? 1;
  stepper($("ctl-nodes"), "Nodes", sim.nodes, pp, 64, (v) => { sim.nodes = Math.max(pp, Math.round(v / pp) * pp); changed(); }, pp);
  stepper($("ctl-pp"), "Nodes per instance", pp, 1, PP_MAX, (v) => { sim.pp = v; if (sim.nodes % v) sim.nodes = Math.ceil(sim.nodes / v) * v; changed(); });
  $("ctl-stage-link").hidden = pp === 1;
  $("ctl-stage-link").replaceChildren(el("label", { class: "field" }, el("span", { text: "Network between the nodes of an instance" }),
    el("select", { onchange: (ev) => { sim.stageLink = ev.target.value; changed(); } },
      slotVariants("p2p").map((v) => el("option", { value: v.id, text: v.short, selected: v.id === sim.stageLink })))));
  stepper($("ctl-gpn"), "GPUs per node", sim.gpusPerNode, 1, 8, (v) => {
    sim.gpusPerNode = v;
    if (v % sim.tp) sim.tp = Math.max(...TP_SIZES.filter((t) => t <= v && v % t === 0));
    changed();
  });
  $("ctl-tp").replaceChildren(...TP_SIZES.map((t) => el("button", {
    type: "button", role: "radio", "aria-checked": String(t === sim.tp), text: String(t), disabled: t > sim.gpusPerNode || sim.gpusPerNode % t !== 0,
    title: sim.gpusPerNode % t ? `TP ${t} needs a multiple of ${t} GPUs per node` : `split the model across ${t} GPUs`,
    onclick: () => { sim.tp = t; changed(); },
  })));
  const g = gpuOf(sim.gpu);
  const inst = (sim.nodes * sim.gpusPerNode) / (sim.tp * pp);
  const kv = kvFor(g);
  const where = pp > 1 ? `TP ${sim.tp} × ${pp} nodes` : `TP ${sim.tp}`;
  fill($("cluster-hint"),
    `${sim.nodes * sim.gpusPerNode} prefill GPUs in ${inst} instance${inst === 1 ? "" : "s"}${sim.tp * pp > 1 ? ` of ${sim.tp * pp} GPUs` : ""}${pp > 1 ? `, each across ${pp} nodes (TP ${sim.tp} in a node, ${pp} pipeline stages)` : ""}. `,
    kv > 0 ? `${fmt(kv, 3)} GiB of KV cache per instance.` : el("b", { class: "error", text: `${cfg.models[sim.model].name ?? sim.model} does not fit ${gpuShort(g)} at ${where}.` }),
    pp > 1 ? el("span", {}, " A request passes the stages in turn; the next one enters the first stage as soon as it is free. ", assumptionBadge("ideal pipelining")) : null,
    sim.tp > 1 && g.pcie_only ? el("span", {}, " ", assumptionBadge("TP over PCIe"), " The TP efficiency comes from NVLink machines.") : null,
    sim.nodes > MAX_SIM_NODES ? ` Live estimates replay the full cluster. Sweeps use up to ${MAX_SIM_NODES} nodes at the same load per node.` : null,
  );
}

// ------------------------------------------------------------------------------------------------ tier lab

let labSlot = null; // the tier being tuned: disk, p2p or shared
const SLOT_SUB = { disk: "one per node", p2p: "another node's CPU memory, over the network", shared: "one store for the whole cluster" };
const POS = 1000; // slider steps

function slotBlocked(slot) {
  const sim = cfg.sim;
  if (slot !== "p2p") return null;
  return sim.nodes < 2 ? "needs 2 or more nodes" : !(sim.cpuGiB > 0) ? "needs CPU memory: peers serve from it" : null;
}
const slotOn = (slot) => !!cfg.sim[slot] && !slotBlocked(slot);
// the variant a slot runs, or would run when switched on
const slotBase = (slot) => variant(cfg.sim[slot]) ?? variant(lastPick[slot]) ?? slotVariants(slot).find((v) => !isLab(v.id)) ?? null;
function setSlot(slot, id) { cfg.sim[slot] = id; if (id) lastPick[slot] = id; }

// the lab's own variant for a slot: made from what the slot runs on the first edit, then edited in place
function labVariant(slot) {
  const cur = slotBase(slot);
  if (cur && isLab(cur.id)) { setSlot(slot, cur.id); return cur; }
  const lv = labFrom(cur, slot);
  const i = cfg.variants.findIndex((x) => x.id === lv.id);
  if (i >= 0) cfg.variants[i] = lv; else cfg.variants.push(lv);
  setSlot(slot, lv.id);
  return lv;
}

// log sliders: `zero` at the left end, `unbounded` (null) at the right end
const niceNum = (x) => Number(x.toPrecision(x < 10 ? 2 : 3));
function propPos(p, x) {
  if (x == null) return POS;
  if (p.zero && !(x > 0)) return 0;
  return Math.round((POS * Math.log(Math.min(p.max, Math.max(p.min, x)) / p.min)) / Math.log(p.max / p.min));
}
function posProp(p, pos) {
  if (p.unbounded && pos >= POS) return null;
  if (p.zero && pos <= 0) return 0;
  return niceNum(p.min * (p.max / p.min) ** (pos / POS));
}
const propText = (p, x) => (x == null ? "unbounded" : p.key === "size" ? gibf(x) : p.key === "price" ? `$${fmt(x, 3)}` : fmt(x, 3));

function propRow(slot, p) {
  const v = slotBase(slot);
  const val = v ? p.get(v) : null;
  const id = `lab-${slot}-${p.key}`;
  const shown = (x) => (x == null ? "" : +Number(x).toPrecision(4));
  const range = el("input", { type: "range", id, min: 0, max: POS, step: 1, value: propPos(p, val), "aria-label": `${p.label}, ${p.unit}` });
  const num = el("input", { type: "number", id: `${id}-n`, step: "any", min: 0, value: shown(val), placeholder: p.unbounded ? "no limit" : "0", inputmode: "decimal", "aria-label": `${p.label} in ${p.unit}` });
  const apply = (x, commit) => {
    const lv = labVariant(slot);
    p.set(lv, x);
    lv.edited = [...new Set([...(lv.edited ?? []), ...p.fields])];
    changed({ config: commit, methods: commit });
  };
  range.addEventListener("input", () => { const x = posProp(p, Number(range.value)); num.value = shown(x); apply(x, false); });
  range.addEventListener("change", () => { keepFocus(renderConfig); renderMethods(); });
  num.addEventListener("change", () => {
    const raw = num.value.trim();
    const x = raw === "" ? (p.unbounded ? null : 0) : Number(raw);
    if (x != null && !(Number.isFinite(x) && x >= 0)) { num.value = shown(val); return; }
    if (x != null && !p.zero && x <= 0) { num.value = shown(val); return; }
    apply(x, true);
  });
  return el("div", { class: "prop" },
    el("label", { for: id }, p.label, el("small", { text: p.unit })),
    range,
    el("div", { class: "prop-num" }, num));
}

function labBody(slot) {
  const sim = cfg.sim;
  const blocked = slotBlocked(slot);
  const on = slotOn(slot);
  const v = slotBase(slot);
  const sw = el("label", { class: "switch" },
    el("input", { type: "checkbox", checked: on, disabled: !!blocked, "aria-label": `Use ${SLOT_NAME[slot]}`,
      onchange: (ev) => { setSlot(slot, ev.target.checked ? (lastPick[slot] ?? v?.id ?? null) : null); changed(); } }),
    el("span", { class: "track" }), el("span", { text: blocked ? "unavailable" : on ? "on" : "off" }));
  const selId = sim[slot] ?? lastPick[slot];
  const lab = cfg.variants.find((x) => x.id === labId(slot));
  const chip = (id, text, title) => el("button", { type: "button", class: "chip", "aria-pressed": String(selId === id), title,
    onclick: () => { setSlot(slot, id); changed(); } }, text);
  const presets = el("div", { class: "presets" }, el("span", { class: "presets-label", text: "Start from" }),
    slotVariants(slot).filter((x) => !isLab(x.id)).map((x) => chip(x.id, x.short ?? x.name, x.name)),
    lab ? chip(lab.id, "Your edits", lab.name) : null);
  const extra = [];
  if (v && slot !== "p2p") {
    const fees = [v.usd_per_gib_written ? `$${v.usd_per_gib_written}/GiB written` : null, v.usd_per_gib_read ? `$${v.usd_per_gib_read}/GiB read` : null].filter(Boolean);
    extra.push(el("p", { class: "tier-sub billing" }, v.bill === "provisioned"
      ? `Billed on provisioned size${slot === "disk" ? ", per node" : ""}: size × price.`
      : "Billed on bytes stored: what is written, kept for the retention time.", fees.length ? ` Plus ${fees.join(" and ")}.` : ""));
    if (v.bill !== "provisioned") extra.push(numField("Keep for (hours)", sim, "retentionH", { id: "lab-retention", onchange: () => { sim.retentionH = Math.max(0.01, sim.retentionH ?? 1); changed(); } }));
  }
  const prov = !v ? null : isLab(v.id)
    ? el("p", { class: "tier-sub prov" }, `Your values, started from ${variant(v.from)?.name ?? "a preset"}. `,
      variant(v.from) ? el("button", { type: "button", class: "link", text: "Back to the preset", onclick: () => { setSlot(slot, v.from); changed(); } }) : null)
    : el("div", { class: "tier-tags prov" }, ...(v.assumptions?.length ? [assumptionBadge(v.assumptions.join(", "))] : []),
      v.sources?.[0]?.url ? el("a", { class: "tier-sub", href: v.sources[0].url, target: "_blank", rel: "noopener", text: `Source: ${v.sources[0].label}` }) : null);
  const sticky = slot === "p2p" && sim.routing === "sticky" ? el("p", { class: "tier-sub", text: "Sticky routing keeps a conversation on one node, so peers mostly share the system prompt." }) : null;
  return el("div", { class: `lab-panel${on ? "" : " off"}`, id: `tier-${slot}` },
    el("div", { class: "lab-head" },
      el("div", {}, el("div", { class: "lab-title" }, el("i", { class: "swatch", style: `background:var(${TIER_COLOR[slot]})` }), SLOT_NAME[slot]),
        el("div", { class: "tier-sub", text: blocked ? `${SLOT_SUB[slot]}: ${blocked}` : SLOT_SUB[slot] })),
      sw),
    sticky,
    presets,
    v ? el("div", { class: "props" }, propsFor(slot).map((p) => propRow(slot, p))) : el("p", { class: "hint", text: "No variant for this tier: add one in Methods & data." }),
    extra.length ? el("div", { class: "lab-extra" }, extra) : null,
    el("p", { class: "lab-live", "data-live": slot }),
    prov);
}

function renderFixedTiers() {
  const sim = cfg.sim;
  const cpuOn = sim.cpuGiB > 0;
  const cpuSw = el("label", { class: "switch" },
    el("input", { type: "checkbox", checked: cpuOn, "aria-label": "Use CPU memory",
      onchange: (ev) => { if (!ev.target.checked) lastPick.cpuGiB = sim.cpuGiB || lastPick.cpuGiB; sim.cpuGiB = ev.target.checked ? lastPick.cpuGiB : 0; changed(); } }),
    el("span", { class: "track" }), el("span", { text: cpuOn ? "on" : "off" }));
  $("fixed-tiers").replaceChildren(
    el("div", { class: "fixed", id: "tier-gpu" },
      el("div", { class: "fixed-top" }, el("i", { class: "swatch", style: `background:var(${TIER_COLOR.gpu})` }), el("b", { text: "GPU memory" }),
        el("span", { class: "tier-sub", text: "always on" })),
      el("span", { class: "tier-sub fixed-sub", "data-live": "gpu" })),
    el("div", { class: `fixed${cpuOn ? "" : " off"}`, id: "tier-cpu" },
      el("div", { class: "fixed-top" }, el("i", { class: "swatch", style: `background:var(${TIER_COLOR.cpu})` }), el("b", { text: "CPU memory" }),
        cpuOn ? numField("GiB/node", sim, "cpuGiB", { step: 1, id: "lab-cpu-gib", onchange: () => { sim.cpuGiB = Math.max(1, sim.cpuGiB ?? 1); lastPick.cpuGiB = sim.cpuGiB; changed(); } }) : null,
        cpuSw),
      el("span", { class: "tier-sub fixed-sub", "data-live": "cpu" })));
}

function renderLabTabs() {
  $("lab-tabs").replaceChildren(...LAB_SLOTS.map((slot) => el("button", {
    type: "button", role: "tab", id: `lab-tab-${slot}`, class: "lab-tab", "aria-selected": String(slot === labSlot), "aria-controls": "lab-body", tabindex: slot === labSlot ? "0" : "-1",
    onclick: () => { if (labSlot !== slot) { labSlot = slot; keepFocus(renderLab); runSensSoon(0); } },
    onkeydown: (ev) => {
      const k = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
      if (!k) return;
      ev.preventDefault();
      labSlot = LAB_SLOTS[(LAB_SLOTS.indexOf(slot) + k + LAB_SLOTS.length) % LAB_SLOTS.length];
      renderLab(); runSensSoon(0); $(`lab-tab-${labSlot}`).focus();
    },
  },
  el("i", { class: "swatch", style: `background:var(${TIER_COLOR[slot]})` }),
  el("span", { class: "lab-tab-name", text: SLOT_NAME[slot] }),
  el("small", { text: slotBlocked(slot) ? "unavailable" : slotOn(slot) ? variant(cfg.sim[slot])?.short ?? "on" : "off" }))));
}

function renderLab() {
  if (!labSlot) labSlot = ["shared", "disk", "p2p"].find(slotOn) ?? "shared";
  renderFixedTiers();
  renderLabTabs();
  $("lab-body").setAttribute("aria-labelledby", `lab-tab-${labSlot}`);
  $("lab-body").replaceChildren(labBody(labSlot));
  renderSensControls();
  updateLabLive();
}

// the texts that follow the estimate (capacities in prompts, shares served), updated in place so a drag survives
function updateLabLive() {
  if (!cfg) return;
  const sim = cfg.sim;
  const g = gpuOf(sim.gpu), kv = kvFor(g);
  const r = estimate?.current && !estimate.current.error ? estimate.current : null;
  const served = (k) => (r?.share?.[k] != null ? ` · serves ${pctf(r.share[k], r.share[k] < 0.1 && r.share[k] > 0 ? 1 : 0)} of prompt tokens` : "");
  const set = (k, text) => { const n = document.querySelector(`[data-live="${k}"]`); if (n) n.textContent = text; };
  set("gpu", kv > 0 ? `${gibf(kv)} per instance: ${capacity(kv)} · ${fmt(g.mem_bw_tb_s, 3)} TB/s${served("gpu")}` : "the model does not fit");
  set("cpu", sim.cpuGiB > 0 ? `per node: ${perGroup(sim.cpuGiB)} · ${fmt(g.host_gib_s, 3)} GiB/s to the GPU${served("cpu")}` : "per node, shared by its GPUs");
  const v = slotBase(labSlot);
  if (v) set(labSlot, `${fmt(variantBandwidth(v), 3)} GiB/s + ${fmt((v.latency_s ?? 0) * 1000, 3)} ms · holds ${labSlot === "p2p" ? "what the peers' CPU memory holds" : labSlot === "disk" ? perGroup(v.capacity_gib) : capacity(v.capacity_gib)}${slotOn(labSlot) ? served(labSlot) : " · off"}`);
}

// ------------------------------------------------------------------------------------------------ sensitivity

let labWorker = null;
let sensId = 0, sensPoints = [], sensTimer = null, sensChart = null, sensPending = false;
let sensProp = "bandwidth";

function renderSensControls() {
  const props = propsFor(labSlot);
  if (!props.some((p) => p.key === sensProp)) sensProp = "bandwidth";
  $("sens-prop").replaceChildren(...props.map((p) => el("option", { value: p.key, text: `${p.label} (${p.unit})`, selected: p.key === sensProp })));
  $("sens-prop").onchange = (ev) => { sensProp = ev.target.value; runSensSoon(0); };
}

function runSensSoon(delay = 450) { clearTimeout(sensTimer); sensTimer = setTimeout(runSens, delay); }

function runSens() {
  if (!labWorker || !cfg) return;
  const slot = labSlot, p = propOf(slot, sensProp), base = slotBase(slot);
  sensId++;
  sensPoints = [];
  const why = slotBlocked(slot) ? `${SLOT_NAME[slot]} ${slotBlocked(slot)}.` : !base ? "No variant for this tier." : source !== "synthetic" ? "The sweep uses synthetic conversations: switch the workload source back to see it." : null;
  $("sens-what").textContent = base ? `${SLOT_NAME[slot]}: ${base.short ?? base.name}. TTFT at your load (${fmt(cfg.workload.requestsPerMin, 3)} requests/min); cost with the load set so the prefill GPUs are ${pctf(TARGET_BUSY)} busy. Dashed: ${SLOT_NAME[slot].toLowerCase()} off.` : "";
  $("sens-note").textContent = "";
  if (why) { $("sens-status").textContent = why; drawSens(); return; }
  $("sens-status").textContent = "Simulating…";
  const cur = p.get(base);
  labWorker.postMessage({ kind: "sens", id: sensId, config: cfg, slot, base, key: p.key, values: logSpace(p.min, p.max, 9, cur), capacity: true });
}

function onLab({ data }) {
  if (data.id !== sensId) return;
  if (data.kind === "sens-point") {
    sensPoints.push(data.point);
    $("sens-status").textContent = `Simulating ${data.done} of ${data.total}…`;
    if (!sensPending) { sensPending = true; requestAnimationFrame(() => { sensPending = false; drawSens(); }); }
  } else if (data.kind === "sens-done") {
    $("sens-status").textContent = "";
    drawSens();
    sensNote();
  }
}

// a dashed line and a label where the slider sits now
const markerPlugin = {
  id: "kvtbMarker",
  afterDatasetsDraw(c, _, opts) {
    if (opts?.x == null) return;
    const x = c.scales.x.getPixelForValue(opts.x);
    if (!(x >= c.chartArea.left - 1 && x <= c.chartArea.right + 1)) return;
    const ctx = c.ctx;
    ctx.save();
    ctx.strokeStyle = cssVar("--text-2");
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, c.chartArea.top); ctx.lineTo(x, c.chartArea.bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = `600 12px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillStyle = cssVar("--text");
    const right = x < c.chartArea.right - 110;
    ctx.textAlign = right ? "left" : "right";
    ctx.fillText(opts.label, x + (right ? 6 : -6), c.chartArea.top + 12);
    ctx.restore();
  },
};

function drawSens() {
  if (!window.Chart || $("pane-sens").hidden || !cfg) return;
  const C = window.Chart;
  C.defaults.font.family = getComputedStyle(document.body).fontFamily;
  C.defaults.color = cssVar("--text-2");
  const slot = labSlot, p = propOf(slot, sensProp), base = slotBase(slot);
  const pts = sensPoints.filter((q) => q.x !== null && !q.error).sort((a, b) => (a.x ?? Infinity) - (b.x ?? Infinity));
  const off = sensPoints.find((q) => q.x === null && !q.error);
  const X = (x) => (x == null ? p.max : x > 0 ? x : p.min);
  const accent = cssVar("--accent"), cost = cssVar("--tier-p2p");
  const datasets = [
    { label: "Mean TTFT at your load", data: pts.map((q) => ({ x: X(q.x), y: q.overloaded ? null : q.ttft, q })), yAxisID: "y", borderColor: accent, backgroundColor: accent, borderWidth: 2.5, pointRadius: 3, tension: 0.2 },
    { label: `$ per 1k requests at ${pctf(TARGET_BUSY)} busy`, data: pts.map((q) => ({ x: X(q.x), y: q.capUsd, q })), yAxisID: "y2", borderColor: cost, backgroundColor: cost, borderWidth: 2.5, pointRadius: 3, tension: 0.2 },
  ];
  if (off && !off.overloaded) datasets.push({ label: "Tier off: TTFT", data: [{ x: p.min, y: off.ttft }, { x: p.max, y: off.ttft }], yAxisID: "y", borderColor: accent, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, pointHitRadius: 0 });
  if (off?.capUsd != null) datasets.push({ label: "Tier off: cost", data: [{ x: p.min, y: off.capUsd }, { x: p.max, y: off.capUsd }], yAxisID: "y2", borderColor: cost, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, pointHitRadius: 0 });
  const cur = base ? p.get(base) : null;
  const tick = function (v) {
    const m = v / 10 ** Math.floor(Math.log10(v));
    return (this.chart.width < 560 ? [1] : [1, 2, 5]).some((k) => Math.abs(m - k) < 1e-6) ? propText(p, v) : "";
  };
  const options = {
    maintainAspectRatio: false, animation: false, parsing: false,
    interaction: { mode: "nearest", axis: "x", intersect: false },
    scales: {
      x: { type: "logarithmic", min: p.min, max: p.max, title: { display: true, text: `${p.label} (${p.unit})${p.unbounded ? ", right edge = unbounded" : ""}${p.zero ? ", left edge = free" : ""}` }, grid: { color: (g) => (g.tick?.label ? cssVar("--grid") : "transparent") }, ticks: { callback: tick, autoSkip: false, maxRotation: 0 } },
      y: { beginAtZero: true, title: { display: true, text: "mean TTFT", color: accent }, grid: { color: cssVar("--grid") }, ticks: { callback: (v) => secs(v), maxTicksLimit: 6 } },
      y2: { beginAtZero: true, position: "right", title: { display: true, text: "$ per 1k requests", color: cost }, grid: { display: false }, ticks: { callback: (v) => usd(v), maxTicksLimit: 6 } },
    },
    plugins: {
      legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 14 } },
      tooltip: { filter: (t) => t.datasetIndex < 2, callbacks: {
        title: (items) => { const q = items[0]?.raw.q; return q ? `${p.label}: ${propText(p, q.x)} ${q.x == null ? "" : p.unit}` : ""; },
        label: (t) => { const q = t.raw.q; return t.datasetIndex === 0 ? `TTFT ${q.overloaded ? "overloaded" : secs(q.ttft)} (p90 ${secs(q.p90)}), prefill ${pctf(q.utilization)} busy` : `${usd(q.capUsd)} per 1k requests at ${fmt(q.capRpm, 3)} requests/min`; } } },
      kvtbMarker: { x: X(cur), label: `now: ${propText(p, cur)}${cur == null ? "" : ` ${p.unit}`}` },
    },
  };
  if (sensChart) { sensChart.data.datasets = datasets; sensChart.options = options; sensChart.update("none"); }
  else sensChart = new C($("sens-chart"), { type: "line", data: { datasets }, options, plugins: [markerPlugin] });
}

// one or two sentences on what the curve says
function sensNote() {
  const slot = labSlot, p = propOf(slot, sensProp);
  const pts = sensPoints.filter((q) => q.x !== null && !q.error && !q.overloaded && q.ttft != null).sort((a, b) => (a.x ?? Infinity) - (b.x ?? Infinity));
  const off = sensPoints.find((q) => q.x === null && !q.error);
  if (pts.length < 3) return;
  const notes = [];
  const best = Math.min(...pts.map((q) => q.ttft));
  if (p.key === "bandwidth" || p.key === "size") {
    const k = pts.find((q) => q.ttft <= best * 1.05);
    if (k && k !== pts[0]) notes.push(`TTFT is within 5% of its best from ${propText(p, k.x)}${k.x == null ? "" : ` ${p.unit}`} up.`);
    else if (k === pts[0]) notes.push(`TTFT barely moves with ${p.label.toLowerCase()} here.`);
  } else if (p.key === "latency") {
    const k = [...pts].reverse().find((q) => q.ttft <= best * 1.05);
    if (k && k !== pts.at(-1)) notes.push(`TTFT is within 5% of its best below ${propText(p, k.x)} ms.`);
  } else notes.push("Price changes the cost only: it never touches TTFT.");
  if (off && !off.overloaded) {
    const worse = pts.filter((q) => q.ttft > off.ttft);
    if (worse.length === pts.length) notes.push(`Across this range the tier is slower than recomputing (tier off: ${secs(off.ttft)}): a cached prefix is always loaded, as vLLM does.`);
    else if (worse.length) notes.push(`Where the blue line rises above the dashed one, loading from this tier is slower than recomputing (${secs(off.ttft)} with it off).`);
  }
  $("sens-note").textContent = notes.join(" ");
}

// ------------------------------------------------------------------------------------------------ across models

let modelsWorker = null;
let modelsId = 0, modelsRows = [], modelsKey = null, modelsRunning = false, modelsPending = false;
const modelsCharts = {};
// what the comparison depends on: everything but the page's own model, cluster shape and request rate
const modelsKeyOf = () => JSON.stringify({ ...cfg, sim: { ...cfg.sim, model: null, tp: null, pp: null, nodes: null, gpusPerNode: null },
  workload: { ...cfg.workload, requestsPerMin: null, durationMin: null } });

function modelsHint() {
  $("models-hint").textContent = `Each model on the deployment that holds it: ${gpuShort(gpuOf(cfg.sim.gpu))} when it is priced and holds the model, else B200, ` +
    `at the smallest TP and fewest nodes, ${COMPARE_INSTANCES} instances. Traffic: the rate at which GPU + CPU memory alone keeps prefill ${pctf(TARGET_BUSY)} busy; ` +
    `both setups get that traffic for TTFT. Cost: each setup loaded to ${pctf(TARGET_BUSY)} busy. Your tiers: ${setupLabel()}.`;
}

function modelsStatus() {
  if (modelsRunning) return;
  const stale = modelsRows.length && modelsKey !== modelsKeyOf();
  $("models-status").textContent = source !== "synthetic" ? "The comparison uses synthetic conversations." : stale ? "Settings changed since this run." : "";
  $("models-run").disabled = source !== "synthetic";
  $("models-run").textContent = modelsRows.length ? "Run again with the current settings" : "Run across models";
}

function runModels() {
  if (!modelsWorker || source !== "synthetic") { modelsStatus(); return; }
  modelsId++;
  modelsRows = [];
  modelsKey = modelsKeyOf();
  modelsRunning = true;
  const models = Object.keys(cfg.models);
  $("models-run").disabled = true;
  $("models-status").textContent = `Simulating 0 of ${models.length} models…`;
  modelsHint();
  modelsWorker.postMessage({ kind: "models", id: modelsId, config: cfg, models });
  drawModels();
}

function onModels({ data }) {
  if (data.id !== modelsId) return;
  if (data.kind === "models-row") {
    modelsRows.push(data.row);
    $("models-status").textContent = `Simulating ${data.done} of ${data.total} models…`;
    if (!modelsPending) { modelsPending = true; requestAnimationFrame(() => { modelsPending = false; drawModels(); }); }
  } else if (data.kind === "models-done") {
    modelsRunning = false;
    $("models-run").disabled = false;
    modelsStatus();
    drawModels();
  }
}

const change = (a, b) => (a > 0 && b > 0 ? b / a - 1 : null);
const changeText = (x) => (x == null ? "" : Math.abs(x) < 0.005 ? "same" : `${x > 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(0)}%`);

// "−22%" at the end of each "your tiers" bar: green when lower, red when higher
const changePlugin = {
  id: "kvtbChange",
  afterDatasetsDraw(c, _, opts) {
    const meta = c.getDatasetMeta(1);
    if (!opts?.values || !c.isDatasetVisible(1)) return;
    const ctx = c.ctx;
    ctx.save();
    ctx.font = `600 11px ${getComputedStyle(document.body).fontFamily}`;
    ctx.textBaseline = "middle";
    meta.data.forEach((bar, i) => {
      const x = opts.values[i];
      if (x == null) return;
      const end = Math.max(bar.x, c.getDatasetMeta(0).data[i]?.x ?? 0);
      ctx.fillStyle = Math.abs(x) < 0.005 ? cssVar("--text-2") : x < 0 ? cssVar("--good") : cssVar("--bad");
      ctx.fillText(changeText(x), end + 6, bar.y);
    });
    ctx.restore();
  },
};

function drawModels() {
  if (!cfg) return;
  const ok = modelsRows.filter((r) => !r.error).sort((a, b) => (change(a.base.ttft, a.tier.ttft) ?? 0) - (change(b.base.ttft, b.tier.ttft) ?? 0));
  const name = (r) => cfg.models[r.model]?.name ?? r.model;
  // the table
  const cell = (label, text, cls) => el("td", { "data-label": label, text, class: cls });
  const gainCls = (x) => (x == null ? null : x < -0.005 ? "good" : x > 0.005 ? "bad" : null);
  $("models-table").replaceChildren(
    el("thead", {}, el("tr", {}, ["Model", "Deployment", "Requests/min", "TTFT: GPU + CPU → yours", "Change", "$ per 1k at 60% busy: GPU + CPU → yours", "Change"].map((t) => el("th", { text: t })))),
    el("tbody", {}, ...ok.map((r) => {
      const dt = change(r.base.ttft, r.tier.ttft), dc = change(r.base.usd, r.tier.usd);
      return el("tr", {},
        el("td", { class: "span", text: name(r) }),
        cell("Deployment", `${r.instances} × ${r.tp * r.pp} ${gpuShort({ name: r.gpuName })}`),
        cell("Requests/min", fmt(r.rpm, 3)),
        cell("TTFT", `${secs(r.base.ttft)} → ${secs(r.tier.ttft)}`),
        cell("Change", changeText(dt), gainCls(dt)),
        cell("$ per 1k", `${usd(r.base.usd)} → ${usd(r.tier.usd)}`),
        cell("Change", changeText(dc), gainCls(dc)));
    }), ...modelsRows.filter((r) => r.error).map((r) => el("tr", {}, el("td", { class: "span", text: name(r) }), el("td", { colspan: 6, class: "span note", text: r.error })))));
  if (!window.Chart || $("pane-models").hidden) return;
  const C = window.Chart;
  C.defaults.font.family = getComputedStyle(document.body).fontFamily;
  C.defaults.color = cssVar("--text-2");
  const labels = ok.map(name);
  const h = `${Math.max(120, 44 + labels.length * 30)}px`;
  const bars = (key, fmtv) => ({
    labels,
    datasets: [
      { label: "GPU + CPU memory", data: ok.map((r) => r.base[key]), backgroundColor: cssVar("--tier-miss"), borderRadius: 3, barPercentage: 0.8, categoryPercentage: 0.8 },
      { label: "Your tiers", data: ok.map((r) => r.tier[key]), backgroundColor: cssVar("--accent"), borderRadius: 3, barPercentage: 0.8, categoryPercentage: 0.8 },
    ],
    fmtv,
  });
  for (const [id, key, f] of [["models-ttft", "ttft", secs], ["models-cost", "usd", usd]]) {
    const data = bars(key, f);
    $(id).parentElement.style.height = h;
    const options = {
      indexAxis: "y", maintainAspectRatio: false, animation: false,
      scales: { x: { beginAtZero: true, grid: { color: cssVar("--grid") }, ticks: { callback: (v) => f(v), maxTicksLimit: 5, maxRotation: 0 } }, y: { grid: { display: false } } },
      plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 14 } },
        tooltip: { callbacks: { label: (t) => `${t.dataset.label}: ${f(t.raw)}`, afterBody: (items) => { const r = ok[items[0].dataIndex]; return `${r.instances} × ${r.tp * r.pp} ${r.gpuName}, ${fmt(r.rpm, 3)} requests/min`; } } } },
    };
    options.layout = { padding: { right: 44 } };
    options.plugins.kvtbChange = { values: ok.map((r) => change(r.base[key], r.tier[key])) };
    if (modelsCharts[id]) { modelsCharts[id].data = data; modelsCharts[id].options = options; modelsCharts[id].update("none"); }
    else modelsCharts[id] = new C($(id), { type: "bar", data, options, plugins: [changePlugin] });
  }
}

// ------------------------------------------------------------------------------------------------ chart tabs

const TABS = ["sens", "models", "map"];
function showTab(t) {
  for (const k of TABS) {
    $(`tab-${k}`).setAttribute("aria-selected", String(k === t));
    $(`tab-${k}`).tabIndex = k === t ? 0 : -1;
    $(`pane-${k}`).hidden = k !== t;
  }
  if (t === "sens") drawSens();
  if (t === "models") { modelsHint(); if (!modelsRows.length && !modelsRunning) runModels(); else { modelsStatus(); drawModels(); } }
  if (t === "map") scheduleChart();
}

function bindTabs() {
  for (const k of TABS) {
    $(`tab-${k}`).addEventListener("click", () => showTab(k));
    $(`tab-${k}`).addEventListener("keydown", (ev) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
      if (!d) return;
      ev.preventDefault();
      const next = TABS[(TABS.indexOf(k) + d + TABS.length) % TABS.length];
      showTab(next);
      $(`tab-${next}`).focus();
    });
  }
  $("models-run").addEventListener("click", runModels);
}

// a slider with a number box; `log` maps the slider logarithmically
const SLIDERS = [
  { key: "requestsPerMin", label: "Requests per minute", hint: "all users together", min: 1, max: 100000, log: true },
  { key: "followUpShare", label: "Follow-up messages (%)", hint: "share that continue a conversation (cache reuse)", min: 0, max: 95, scale: 100, unit: "%" },
  { key: "inputTokens", label: "Tokens in per message", hint: "the new text a user sends", min: 10, max: 32000, log: true },
  { key: "outputTokens", label: "Tokens out per answer", hint: "generated, then cached with the conversation", min: 1, max: 16000, log: true },
  { key: "thinkTimeS", label: "Pause between turns (s)", hint: "user reading and typing", min: 1, max: 7200, log: true },
];

function renderSliders() {
  const w = cfg.workload;
  const off = source !== "synthetic";
  $("sliders").classList.toggle("disabled", off);
  if ($("map-metric")) { for (const p of points) withMetric(p); renderMapMetric(); }
  $("trace-note").hidden = !off;
  if (off) $("trace-note").textContent = trace ? `Replaying ${trace.name}: ${trace.requests.length.toLocaleString()} requests over ${fmt((trace.requests.at(-1).t - trace.requests[0].t) / 60, 3)} min. The load sliders apply to synthetic conversations only.` : "Load a trace in “Load & serving settings”. The load sliders apply to synthetic conversations only.";
  $("sliders").replaceChildren(...SLIDERS.map((s) => {
    const scale = s.scale ?? 1;
    const value = w[s.key] * scale;
    const toPos = (v) => (s.log ? (1000 * Math.log(v / s.min)) / Math.log(s.max / s.min) : v);
    const fromPos = (p) => (s.log ? s.min * Math.exp((p / 1000) * Math.log(s.max / s.min)) : p);
    const nice = (v) => (s.log ? Number(v.toPrecision(v < 100 ? 2 : 2)) : Math.round(v));
    const id = `sl-${s.key}`;
    const num = el("input", { type: "number", id: `${id}-n`, min: s.min, max: s.max, step: "any", value: +value.toPrecision(6), disabled: off, "aria-label": s.label });
    const range = el("input", { type: "range", id, min: s.log ? 0 : s.min, max: s.log ? 1000 : s.max, step: 1, value: toPos(Math.min(s.max, Math.max(s.min, value))), disabled: off, "aria-label": s.label });
    const set = (v, rerender) => {
      const c = Math.min(s.max, Math.max(s.min, v));
      w[s.key] = c / scale;
      num.value = +c.toPrecision(6);
      changed({ config: rerender });
    };
    range.addEventListener("input", () => set(nice(fromPos(Number(range.value))), false));
    range.addEventListener("change", () => keepFocus(renderConfig));
    num.addEventListener("change", () => { const v = Number(num.value); if (Number.isFinite(v)) { set(v, true); } else num.value = +value.toPrecision(6); });
    return el("div", { class: "slider" }, el("label", { for: id }, el("span", { text: s.label }), el("small", { text: s.hint })), range, num);
  }));
}

function renderSettings() {
  const w = cfg.workload;
  const sim = cfg.sim;
  const re = () => changed();
  $("ctl-source").replaceChildren(...[["synthetic", "Synthetic"], ["mooncake", "Mooncake trace"], ["upload", "Upload a trace"]].map(([k, t]) =>
    el("button", { type: "button", role: "radio", "aria-checked": String(source === k), text: t, onclick: () => { if (source !== k) { source = k; trace = null; $("workload-status").textContent = ""; changed(); } } })));
  if (source === "synthetic") $("source-body").replaceChildren(el("p", { class: "hint", text: "Conversations start at random; each turn sends a message and gets an answer, both added to the conversation's context." }));
  else if (source === "mooncake") {
    const sel = el("select", { "aria-label": "Mooncake trace" }, MOONCAKE_TRACES.map((t, i) => el("option", { value: i, text: t.name })));
    $("source-body").replaceChildren(
      el("p", { class: "hint" }, "Public traces from ", el("a", { href: "https://github.com/kvcache-ai/Mooncake", text: "kvcache-ai/Mooncake", target: "_blank", rel: "noopener" }),
        ` (${MOONCAKE_TRACES[0].license}), fetched from GitHub when you press Load. They come from a large serving cluster. Each record has a time, an input length and a hash per ${MOONCAKE_TRACES[0].blockTokens}-token block; equal hashes are equal prefixes. Output lengths come from the trace.`),
      el("div", { class: "row" }, sel, el("button", { type: "button", class: "primary", text: "Load", onclick: async () => {
        const t = MOONCAKE_TRACES[+sel.value];
        $("workload-status").textContent = `Loading ${t.name}…`;
        try {
          const res = await fetch(t.url);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          trace = parseTrace(await res.text(), t.blockTokens, t.name);
          $("workload-status").textContent = `Loaded ${trace.requests.length.toLocaleString()} requests.`;
          changed();
        } catch (e) {
          trace = null;
          $("workload-status").textContent = `Could not load ${t.url}: ${e.message}. Download it and use “Upload a trace”.`;
        }
      } })));
  } else {
    const bt = { blockTokens: trace?.blockTokens ?? 512 };
    $("source-body").replaceChildren(
      el("p", { class: "hint", text: 'JSONL or a JSON array of {"timestamp": ms, "input_length": tokens, "output_length": tokens, "hash_ids": [one id per block]} (the Mooncake format), replayed in time order.' }),
      el("div", { class: "grid" }, numField("Tokens per hash (block)", bt, "blockTokens", { step: 1, onchange: () => { if (trace) { trace.blockTokens = bt.blockTokens; changed(); } } }),
        el("label", { class: "field" }, el("span", { text: "Trace file" }), el("input", { type: "file", accept: ".jsonl,.json,.txt", onchange: async (ev) => {
          const f = ev.target.files[0];
          if (!f) return;
          try { trace = parseTrace(await f.text(), bt.blockTokens, f.name); $("workload-status").textContent = `Loaded ${trace.requests.length.toLocaleString()} requests.`; changed(); }
          catch (e) { trace = null; $("workload-status").textContent = `Could not read ${f.name}: ${e.message}`; }
        } }))));
  }
  const pos = (k, min) => () => { w[k] = Math.max(min, w[k] ?? min); re(); };
  $("workload-fields").replaceChildren(
    numField("Shared system prompt (tokens)", w, "systemTokens", { step: 1, onchange: pos("systemTokens", 0) }),
    numField("Max context (tokens)", w, "maxContextTokens", { step: 1, onchange: pos("maxContextTokens", 256) }),
    numField("Duration (min)", w, "durationMin", { onchange: pos("durationMin", 1) }),
    numField("Warm-up, left out (min)", sim, "warmupMin", { onchange: () => { sim.warmupMin = Math.max(0, sim.warmupMin ?? 0); re(); } }),
    numField("Random seed", w, "seed", { step: 1, onchange: re }),
  );
  const dec = el("label", { class: "field check", id: "ctl-decode" }, el("input", { type: "checkbox", checked: !!sim.decode, onchange: (ev) => { sim.decode = ev.target.checked; re(); } }), el("span", { text: "Decode on GPUs of its own (disaggregated)" }));
  $("serving-fields").replaceChildren(
    ...[["peerFirst", "Prefer peer CPU over shared storage after local SSD"], ["saveDecode", "Save generated reply chunks to backing tiers"]].map(([key, label]) =>
      el("label", { class: "field check" }, el("input", { type: "checkbox", checked: !!sim[key], onchange: ev => { sim[key] = ev.target.checked; re(); } }), el("span", { text: label }))),
    numField("Prefill batch budget (tokens per step; empty = one request at a time)", sim, "prefillBatchTokens", {
      step: 1, onchange: () => { const v = sim.prefillBatchTokens; sim.prefillBatchTokens = v > 0 ? Math.round(v) : null; re(); } }),
    el("label", { class: "field" }, el("span", { text: "Routing" }),
      el("select", { id: "ctl-routing", onchange: (ev) => { sim.routing = ev.target.value; re(); } },
        el("option", { value: "round-robin", text: "Round-robin across nodes", selected: sim.routing === "round-robin" }),
        el("option", { value: "sticky", text: "Sticky: a conversation stays on one instance", selected: sim.routing === "sticky" }),
        el("option", { value: "random", text: "Random instance per request", selected: sim.routing === "random" }))),
    el("label", { class: "field" }, el("span", { text: "Attention across a TP group" }),
      el("select", { id: "ctl-attention", onchange: (ev) => { sim.attention = ev.target.value; changed(); } },
        el("option", { value: "dp", text: "Data parallel: each GPU its own requests, no KV copies", selected: (sim.attention ?? "tp") === "dp" }),
        el("option", { value: "tp", text: "Tensor parallel (vLLM's default): latent KV copied on every GPU", selected: (sim.attention ?? "tp") === "tp" }))),
    dec,
    ...(sim.decode ? [
      el("label", { class: "field" }, el("span", { text: "Decode GPU" }),
        el("select", { onchange: (ev) => { sim.decodeGpu = ev.target.value; re(); } }, cfg.gpus.map((g) => el("option", { value: g.id, text: gpuShort(g), selected: g.id === sim.decodeGpu })))),
      numField("Max decode batch", sim, "decodeBatch", { step: 1, onchange: () => { sim.decodeBatch = Math.max(1, Math.round(sim.decodeBatch ?? 1)); re(); } }),
      speedField(sim, re),
      ...specFields(sim, re),
      ...epFields(sim, re),
      el("label", { class: "field" }, el("span", { text: "KV transfer to decode" }),
        el("select", { onchange: (ev) => { sim.transfer = ev.target.value || null; re(); } },
          el("option", { value: "", text: "free (same machine)", selected: !sim.transfer }),
          slotVariants("p2p").map((v) => el("option", { value: v.id, text: v.short, selected: v.id === sim.transfer })))),
    ] : []),
    numField("Simulated block (tokens)", sim, "simBlockTokens", { step: 1, onchange: () => { sim.simBlockTokens = Math.max(16, Math.round(sim.simBlockTokens ?? 256)); re(); } }),
  );
}

// Speculative decoding: the method, its accepted tokens per step (blank: the method's), and the custom method's numbers.
// MTP and DSpark need the drafter the checkpoint ships; the others need a draft model trained for it.
function specFields(sim, re) {
  const model = cfg.models[sim.model] ?? {};
  const has = (m) => !m.requires || (model.drafters ?? []).includes(m.requires);
  const method = SPEC_METHODS.find((x) => x.id === (sim.spec ?? "none")) ?? SPEC_METHODS[0];
  const out = [el("label", { class: "field" }, el("span", { text: "Speculative decoding" }),
    el("select", { id: "ctl-spec", onchange: (ev) => { sim.spec = ev.target.value; sim.specTau = null; re(); } },
      SPEC_METHODS.map((x) => el("option", { value: x.id, selected: x.id === method.id, disabled: !has(x) && x.id !== method.id,
        text: x.id === "none" ? x.name : `${x.name}${x.id === "custom" ? "" : `: ${fmt(x.tau, 3)} tokens per step`}${has(x) ? "" : " (not in this checkpoint)"}` }))))];
  if (method.id === "none") return out;
  const p = method.id === "custom" ? { ...method, ...(sim.specCustom ?? {}) } : method;
  out.push(numField("Accepted tokens per step", sim, "specTau", { id: "ctl-spec-tau", placeholder: `${fmt(p.tau, 3)} (${method.id === "custom" ? "custom" : "measured on 4–14B models"})`,
    assumption: !!(model.expert_share > 0), onchange: () => { if (!(sim.specTau >= 1)) sim.specTau = null; re(); } }));
  if (method.id === "custom") {
    const c = (sim.specCustom = { draft: method.draft, tau: method.tau, layers: method.layers, parallel: method.parallel, ...(sim.specCustom ?? {}) });
    out.push(
      numField("Draft tokens per step", c, "draft", { step: 1, onchange: () => { c.draft = Math.max(1, Math.round(c.draft ?? 1)); re(); } }),
      numField("Custom accepted tokens per step", c, "tau", { onchange: () => { c.tau = Math.max(1, c.tau ?? 1); re(); } }),
      numField("Draft layers (the model's size)", c, "layers", { onchange: () => { c.layers = Math.max(0, c.layers ?? 0); re(); } }),
      el("label", { class: "field" }, el("span", { text: "Drafting" }),
        el("select", { onchange: (ev) => { c.parallel = ev.target.value === "parallel"; re(); } },
          el("option", { value: "parallel", text: "Parallel: one pass for all drafted tokens", selected: !!c.parallel }),
          el("option", { value: "ar", text: "Autoregressive: one pass per token", selected: !c.parallel }))));
  }
  if (!has(method)) out.push(el("p", { class: "note warn", text: `${model.name ?? sim.model} ships no ${method.requires.toUpperCase()} drafter: decode runs without one.` }));
  return out;
}

// Expert parallelism in decode: the routed experts spread over more GPUs than one TP instance (MoE models only).
function epFields(sim, re) {
  const model = cfg.models[sim.model] ?? {};
  if (!(model.expert_share > 0)) return [];
  const widths = [8, 16, 32, 64, 72, 128, 144, 256].filter((n) => n > sim.tp && n % sim.tp === 0);
  const dom = gpuOf(sim.decodeGpu).nvlink_domain ?? 1;
  const out = [el("label", { class: "field" }, el("span", { text: "Decode experts spread over" }),
    el("select", { id: "ctl-ep", onchange: (ev) => { sim.decodeEp = ev.target.value ? Number(ev.target.value) : null; re(); } },
      el("option", { value: "", text: `the TP instance (${sim.tp * (sim.pp ?? 1)} GPUs)`, selected: !sim.decodeEp }),
      widths.map((n) => el("option", { value: n, selected: n === sim.decodeEp, text: `${n} GPUs${n > dom ? " (InfiniBand beyond the NVLink domain)" : " (NVLink)"}` }))))];
  if (sim.decodeEp) out.push(el("label", { class: "field check" },
    el("input", { type: "checkbox", id: "ctl-overlap", checked: !!sim.decodeOverlap, onchange: (ev) => { sim.decodeOverlap = ev.target.checked; re(); } }),
    el("span", { text: "Hide the expert all-to-all behind compute (two-batch overlap)" })));
  return out;
}

// Decode speed per request: an optional target limits the size of a batch of ready sequences. OpenRouter's p50 for
// the model, when the snapshot has one, is a click away.
function speedField(sim, re) {
  const p50 = market?.models?.[sim.model]?.speed;
  const field = numField("Decode speed per request (tok/s)", sim, "decodeSpeed", { id: "ctl-decode-speed", placeholder: "no speed target",
    onchange: () => { if (!(sim.decodeSpeed > 0)) sim.decodeSpeed = null; re(); } });
  if (p50?.p50_tok_s) field.append(el("button", { type: "button", class: "link", id: "ctl-decode-speed-or",
    text: `OpenRouter p50: ${fmt(p50.p50_tok_s, 3)} tok/s (${p50.as_of})`, title: p50.url,
    onclick: () => { sim.decodeSpeed = p50.p50_tok_s; re(); } }));
  return field;
}

// ------------------------------------------------------------------------------------------------ estimate

const TIER_NAME = (k, sim) => ({ gpu: "GPU", cpu: "CPU memory", miss: "Recomputed" }[k] ?? (variant(sim[k])?.short ?? k));

function dl(host, rows) {
  host.replaceChildren(...rows.filter(Boolean).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]));
}

function renderEstimate() {
  if (!estimate) return;
  const { current: r, baseline: b } = estimate;
  const sim = cfg.sim;
  $("est-setup").textContent = setupLabel({}, { cluster: true });
  const alerts = [];
  const alert = (text, bad = false) => alerts.push(el("div", { class: `alert${bad ? " bad" : ""}`, text }));
  if (r.error) {
    $("est-prices").replaceChildren();
    for (const id of ["est-ttft", "est-usd", "est-cap"]) $(id).textContent = "–";
    for (const id of ["est-ttft-sub", "est-usd-sub", "est-cap-sub"]) $(id).textContent = "";
    $("mini-ttft").textContent = "–"; $("mini-usd").textContent = "–";
    $("est-delta").textContent = "";
    alert(`${r.error}. Pick a GPU with more memory or a larger tensor-parallel size.`, true);
    $("est-alerts").replaceChildren(...alerts);
    $("est-bar").replaceChildren(); $("est-legend").replaceChildren();
    for (const id of ["est-latency", "est-cost", "est-util"]) $(id).replaceChildren();
    $("est-foot").textContent = "";
    return;
  }
  $("est-ttft").classList.remove("over");
  if (!r.requests) {
    $("est-prices").replaceChildren();
    for (const id of ["est-ttft", "est-usd", "est-cap", "mini-ttft", "mini-usd"]) $(id).textContent = "–";
    for (const id of ["est-ttft-sub", "est-usd-sub", "est-cap-sub", "est-delta", "est-foot"]) $(id).textContent = "";
    alert(source === "synthetic" ? "No requests after the warm-up: make the duration longer than the warm-up." : "No requests after the warm-up: lower the warm-up for this trace.", true);
    $("est-alerts").replaceChildren(...alerts);
    return;
  }
  const overloaded = r.utilization >= 1;
  $("est-ttft").textContent = overloaded ? "overloaded" : secs(r.ttft.mean);
  $("est-ttft").classList.toggle("over", overloaded);
  $("est-ttft-sub").textContent = overloaded ? `no steady value; ${secs(r.ttft.mean)} mean in this window` : `p50 ${secs(r.ttft.p50)} · p90 ${secs(r.ttft.p90)}`;
  $("est-usd").textContent = r.priced ? usd(r.usdPer1kRequests) : "no price";
  const hours = r.windowS / 3600;
  if (r.specUnavailable) alert(`${cfg.models[sim.model]?.name ?? sim.model} ships no drafter for ${r.specUnavailable}: decode runs without speculative decoding.`);
  if (r.belowSpeed > 0.005) alert(`${pctf(r.belowSpeed, 0)} of requests decode slower than ${fmt(sim.decodeSpeed, 3)} tok/s even one at a time on ${gpuShort(gpuOf(sim.decodeGpu))} (speculative decoding, which reaches it, is not modeled).`);
  $("est-usd-sub").textContent = r.priced ? `${usd(r.usdPer1MTokens)} per 1M tokens · ${usd(r.usd.total / Math.max(hours, 1e-9))} per hour` : "a GPU or tier here has no price";
  const cap = capEst?.cap;
  $("est-cap").textContent = source !== "synthetic" ? "–" : !capEst ? "…" : cap?.priced ? usd(cap.usdPer1k) : "–";
  $("est-cap-sub").textContent = source !== "synthetic" ? "synthetic conversations only" : cap ? `serves ${fmt(cap.rpm, 3)} requests/min on these GPUs` : "";
  $("mini-ttft").textContent = overloaded ? "overloaded" : secs(r.ttft.mean);
  $("mini-usd").textContent = r.priced ? usd(r.usdPer1kRequests) : "no price";

  // against GPU + CPU (or GPU only) on the same cluster
  const isBase = SLOTS.every((k) => !sim[k]) || (!sim.disk && !sim.shared && sim.p2p && sim.nodes < 2);
  const baseName = sim.cpuGiB > 0 ? "GPU + CPU" : "GPU only";
  if (overloaded || b.error || !b.requests) $("est-delta").textContent = "";
  else if (isBase) $("est-delta").replaceChildren(el("div", { class: "delta-base", text: `This is the ${baseName} baseline. Turn on a tier to compare.` }));
  else {
    const dt = r.ttft.mean / b.ttft.mean - 1;
    const cc = capEst?.cap, cb = capEst?.base;
    const dc = cc?.priced && cb?.priced ? cc.usdPer1k / cb.usdPer1k - 1 : null;
    const word = (x, less, more, same) => (Math.abs(x) < 0.005 ? same : `${Math.abs(x * 100).toFixed(0)}% ${x < 0 ? less : more}`);
    $("est-delta").replaceChildren(
      el("div", { class: "delta-main" },
        el("span", { class: dt < -0.005 ? "better" : dt > 0.005 ? "worse" : "", text: word(dt, "lower TTFT", "higher TTFT", "about the same TTFT") }),
        dc == null ? "" : " · ",
        dc == null ? "" : el("span", { class: dc < -0.005 ? "better" : dc > 0.005 ? "worse" : "", text: word(dc, "cheaper", "more expensive", "about the same cost") })),
      el("div", { class: "delta-base", text: `vs the ${baseName} baseline: ${secs(b.ttft.mean)}${cb?.priced ? ` · ${usd(cb.usdPer1k)} at ${pctf(TARGET_BUSY)} busy` : ""}` }));
  }
  if (overloaded) alert(`Overloaded: prefill needs ${pctf(r.utilization)} of the GPUs' time, so the queue grows for as long as the load lasts and TTFT has no steady value. Add nodes or GPUs, or cache more.`, true);
  else if (r.utilization > 0.8) alert(`Prefill GPUs are ${pctf(r.utilization)} busy: waiting in the queue adds ${secs(r.wait)} on average.`);
  if (!r.priced) alert(`No price for ${[gpuOf(sim.gpu), sim.decode ? gpuOf(sim.decodeGpu) : null].filter((g) => g && g.usd_per_hour == null).map(gpuShort).filter((x, i, a) => a.indexOf(x) === i).join(" and ") || "a tier"}: set one in Methods & data to see cost.`);
  const pr = r.price;
  if (pr.cachedInput != null && pr.input != null && pr.cachedInput > pr.input)
    alert("A cached token costs more than computing it here: the tiers' storage and load time outweigh the compute they save.");
  $("est-alerts").replaceChildren(...alerts);

  const priceCell = (x, label, title) => el("div", { title }, el("b", { text: x == null ? "–" : usd(x) }), el("span", { text: label }));
  $("est-prices").replaceChildren(
    priceCell(pr.input, "input", "Prompt tokens recomputed: their share of the prefill GPUs (idle time included)."),
    priceCell(pr.cachedInput, "cached input", "Prompt tokens served from a cache tier: their share of the prefill GPUs (load time) plus storage."),
    priceCell(pr.output, sim.decode ? "output" : "output (decode off)", "Generated tokens: the decode GPUs."));
  const order = r.order;
  $("est-bar").replaceChildren(...order.filter((k) => r.share[k] > 0.0005).map((k) => el("span", { style: `width:${(r.share[k] * 100).toFixed(2)}%;background:var(${TIER_COLOR[k]})`, title: `${TIER_NAME(k, sim)}: ${pctf(r.share[k], 1)}` })));
  $("est-legend").replaceChildren(...order.map((k) => el("li", {}, el("i", { style: `background:var(${TIER_COLOR[k]})` }), el("span", { text: TIER_NAME(k, sim) }), el("b", { text: pctf(r.share[k], r.share[k] < 0.1 && r.share[k] > 0 ? 1 : 0) }))));

  dl($("est-latency"), [
    ["Waiting for a prefill GPU", secs(r.wait)],
    ["Prefill, with cache loads", secs(r.prefill)],
    sim.decode ? ["KV transfer to decode", secs(r.transfer)] : null,
    r.tpot != null ? ["First decode step", secs(r.tpot)] : null,
    ["TTFT p50 / p90", `${secs(r.ttft.p50)} / ${secs(r.ttft.p90)}`],
    r.tpot != null ? ["Time per output token", secs(r.tpot)] : null,
    r.tpot != null ? ["Decode speed per request", `${fmt(1 / r.tpot, 3)} tok/s at a batch of ${fmt(r.decodeBatch, 3)}${r.decodeEp ? ` per ${sim.tp}-GPU group` : ""}${sim.decodeSpeed ? ` (target ${fmt(sim.decodeSpeed, 3)})` : ""}`] : null,
    r.specSpeedup != null ? ["Speculative decoding", r.specSpeedup <= 1.0005 ? "switched off: at this batch checking drafts would slow decode down"
      : `${fmt(r.acceptedTokens, 3)} tokens per step, ${fmt(r.specSpeedup, 3)}× faster than none at this batch`] : null,
    r.decodeEp ? ["Decode experts spread over", `${r.decodeEp} GPUs (${r.decodeEp / sim.tp} attention groups)`] : null,
    ["Whole answer, mean", secs(r.e2e.mean)],
  ]);
  const gp = gpuOf(sim.gpu), gd = gpuOf(sim.decodeGpu);
  const per = (x, g) => (g.usd_per_hour == null ? "no price" : usd(x / hours));
  dl($("est-cost"), [
    ["Prefill GPUs", `${r.prefillGpus} × ${gpuShort(gp)}`],
    sim.decode ? ["Decode GPUs", `${r.decodeGpus} × ${gpuShort(gd)}`] : null,
    ["Prefill GPUs, $/hour", per(r.usd.prefill, gp)],
    sim.decode ? ["Decode GPUs, $/hour", per(r.usd.decode, gd)] : null,
    ["Storage and requests, $/hour", usd(r.usd.storage / hours)],
    r.requestUsd ? ["Included S3 request charges, $/hour", usd(r.requestUsd / hours)] : null,
    r.usd.fixed ? ["Network and other, $/hour", usd(r.usd.fixed / hours)] : null,
    ["Total, $/hour", r.priced ? usd(r.usd.total / hours) : "no price"],
    ["Per 1M tokens (in + out)", r.priced ? usd(r.usdPer1MTokens) : "no price"],
  ]);
  dl($("est-util"), [
    ["Prefill GPU busy", pctf(r.utilization)],
    ["Prefill instances", `${r.instances}${sim.tp > 1 ? ` (TP ${sim.tp})` : ""}`],
    ["KV cache per instance", gibf(r.kvGiB)],
    ["GPU cache evictions", r.evictions.gpu.toLocaleString()],
    sim.cpuGiB > 0 ? ["CPU cache evictions", r.evictions.cpu.toLocaleString()] : null,
    ["Requests simulated", r.requests.toLocaleString()],
  ]);
  const notes = [`Simulated ${r.requests.toLocaleString()} requests over ${fmt(r.windowS / 60, 3)} min after a ${fmt(sim.warmupMin, 3)} min warm-up, in ${r.blockTokens}-token blocks.`];
  if (r.simNodes < sim.nodes) notes.push(`Replayed at ${r.simNodes} of ${sim.nodes} nodes with the same load per node; totals scaled.`);
  $("est-foot").textContent = notes.join(" ");
}

// ------------------------------------------------------------------------------------------------ diagram

function renderDiagramNow() {
  if (!cfg) return;
  const sim = cfg.sim;
  const g = gpuOf(sim.gpu);
  const r = estimate && !estimate.current.error ? estimate.current : null;
  const v = (k) => { const x = variant(sim[k]); return x ? { name: x.name, short: x.short, gib: x.capacity_gib } : null; };
  const p2pOk = sim.nodes >= 2 && sim.cpuGiB > 0;
  const tr = variant(sim.transfer);
  const box = $("diagram");
  const s = {
    nodes: sim.nodes, gpusPerNode: sim.gpusPerNode, tp: sim.tp, pp: sim.pp ?? 1,
    stageLink: variant(sim.stageLink)?.name, stageLinkShort: variant(sim.stageLink)?.short, gpuShort: gpuShort(g), gpuName: g.name, kvGiB: kvFor(g),
    tpLink: g.nvlink_gb_s ? "NVLink" : g.pcie_only ? "PCIe" : "GPU link",
    tpLinkTitle: g.nvlink_gb_s ? `NVLink, ${g.nvlink_gb_s} GB/s per GPU (drawn only; TP efficiency comes from published runs)` : g.pcie_only ? "PCIe (this GPU has no NVLink)" : "GPU interconnect",
    cpuGiB: sim.cpuGiB, disk: v("disk"), p2p: p2pOk ? v("p2p") : null, shared: v("shared"),
    decode: sim.decode ? { gpus: r?.decodeGpus ?? "?", gpuShort: gpuShort(gpuOf(sim.decodeGpu)), batch: sim.decodeBatch } : null,
    transfer: tr ? tr.name : "local copy", transferShort: tr ? tr.short : "local",
    rate: source === "synthetic" ? fmt(cfg.workload.requestsPerMin, 3) : "trace", followUp: source === "synthetic" ? cfg.workload.followUpShare : null,
    routing: sim.routing, share: r?.share ?? {},
  };
  box.innerHTML = renderDiagram(s, { vertical: box.clientWidth < 640 });
}

function bindDiagram() {
  const box = $("diagram");
  const go = (ev) => { const t = ev.target.closest("[data-target]"); if (t) flash(t.dataset.target); };
  box.addEventListener("click", go);
  box.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(ev); } });
  let lastWide = null;
  new ResizeObserver(() => { const wide = box.clientWidth >= 640; if (wide !== lastWide) { lastWide = wide; renderDiagramNow(); } }).observe(box);
}

// ------------------------------------------------------------------------------------------------ trade-off map

let chart = null;
let chartPending = false;
let selected = null;
let mapMetric = "cap"; // the map's cost: "cap" (at the target busy share) or "load" (at your load)
const metricNow = () => (source === "synthetic" ? mapMetric : "load"); // a trace has its own rate: no busy-share cost
const metricText = () => (metricNow() === "cap" ? `at ${pctf(TARGET_BUSY)} busy` : "at your load");
// set a point's x to the cost the map shows, keeping both
function withMetric(p) {
  if (p.xLoad === undefined) p.xLoad = p.x;
  p.x = metricNow() === "cap" ? p.capX ?? null : p.xLoad;
  return p;
}

const currentPoint = () => {
  const r = estimate?.current;
  if (!r || r.error) return null;
  const capX = capEst?.cap?.priced ? capEst.cap.usdPer1k : null;
  return withMetric({ series: "you", x: r.usdPer1kRequests, capX, y: r.ttft.mean, p90: r.ttft.p90, priced: r.priced, overloaded: r.utilization >= 1, share: r.share, order: r.order, over: {} });
};

function renderMapMetric() {
  const off = source !== "synthetic";
  $("map-metric").replaceChildren(...[["cap", `at ${pctf(TARGET_BUSY)} busy`], ["load", "at your load"]].map(([k, t]) => el("button", {
    type: "button", role: "radio", "aria-checked": String(metricNow() === k), text: t, disabled: off && k === "cap",
    title: k === "cap" ? "Each setup loaded until its prefill GPUs are busy 60% of the time: what its headroom is worth" : "Your cluster at your request rate: the bill, where every GPU is paid for anyway",
    onclick: () => { if (mapMetric === k) return; mapMetric = k; for (const p of points) withMetric(p); for (const p of ghost ?? []) withMetric(p); selected = null; renderDrawer(); updateFrontier(); renderMapMetric(); scheduleChart(); },
  })));
}
const plottable = (p) => p && !p.error && p.priced && !p.overloaded && Number.isFinite(p.x) && Number.isFinite(p.y);

function updateFrontier() {
  const you = currentPoint();
  frontier = paretoFrontier([...points.filter((p) => p.series === "combo" && plottable(p)), ...(plottable(you) ? [you] : [])]);
}

function pointLabel(p) {
  if (p.series === "you") return `Your setup: ${setupLabel()}`;
  if (p.series === "combo") return setupLabel(p.over);
  if (p.series === "nodes-yours") return `${p.nodes} node${p.nodes > 1 ? "s" : ""}, your tiers`;
  if (p.series === "nodes-base") return `${p.nodes} node${p.nodes > 1 ? "s" : ""}, ${cfg.sim.cpuGiB > 0 ? "GPU + CPU" : "GPU only"}`;
  if (p.series === "gpus") return `${gpuShort(gpuOf(p.gpu))}, your tiers`;
  return "";
}

function scheduleChart() {
  if (chartPending) return;
  chartPending = true;
  requestAnimationFrame(() => { chartPending = false; drawMap(); });
}

function renderChips() {
  const chip = (key, label, color) => el("button", { type: "button", class: "chip", "aria-pressed": String(show[key]), onclick: () => { show[key] = !show[key]; renderChips(); scheduleChart(); } },
    el("i", { style: `background:var(${color})` }), label);
  $("map-chips").replaceChildren(
    el("span", { class: "chip", "aria-pressed": "true", title: "always shown" }, el("i", { style: "background:var(--accent)" }), "Tier combinations"),
    chip("nodes", "Number of nodes", "--tier-p2p"),
    chip("gpus", "Other GPUs", "--tier-cpu"));
}

function renderExcluded() {
  const bad = points.filter((p) => !plottable(p) && (p.series === "combo" || show.gpus || show.nodes));
  const why = (p) => (p.error ? "doesn't fit" : !p.priced ? "no price" : "overloaded");
  $("map-excluded").textContent = bad.length ? `Not plotted: ${bad.map((p) => `${pointLabel(p)} (${why(p)})`).join("; ")}.` : "";
}

// direct labels for the frontier, your setup and the curves' points. Each label takes the first spot around its point
// that overlaps no point and no label already placed ("You" first); a label with no free spot is left to the tooltip.
const labelPlugin = {
  id: "kvtbLabels",
  afterDatasetsDraw(c) {
    const ctx = c.ctx;
    ctx.save();
    ctx.font = `12px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillStyle = cssVar("--text-2");
    ctx.textBaseline = "middle";
    const narrow = c.width < 520;
    const area = c.chartArea;
    const H = 14, GAP = 9;
    const taken = []; // boxes {l, t, r, b}
    const hits = (a) => taken.some((o) => a.l < o.r && a.r > o.l && a.t < o.b && a.b > o.t);
    const inside = (a) => a.l >= area.left && a.r <= area.right + 8 && a.t >= area.top - 4 && a.b <= area.bottom;
    const wanted = [];
    c.data.datasets.forEach((ds, i) => {
      if (!c.isDatasetVisible(i)) return;
      c.getDatasetMeta(i).data.forEach((pt, j) => {
        const r = (ds.pointRadius ?? 4) + 2;
        taken.push({ l: pt.x - r, t: pt.y - r, r: pt.x + r, b: pt.y + r });
        const text = ds.labels?.[j];
        if (text && !(narrow && ds.labelWide)) wanted.push({ pt, text, below: !!ds.labelBelow, r });
      });
    });
    wanted.sort((a, b) => b.below - a.below);
    for (const { pt, text, below, r } of wanted) {
      const w = ctx.measureText(text).width;
      const off = r + GAP - 4;
      // [x of the box's left edge, y of its middle]
      const spots = below
        ? [[pt.x - w / 2, pt.y + off + H / 2], [pt.x - w / 2, pt.y - off - H / 2], [pt.x + off, pt.y], [pt.x - off - w, pt.y]]
        : [[pt.x + off, pt.y - H / 2 - 2], [pt.x - off - w, pt.y - H / 2 - 2], [pt.x + off, pt.y + H / 2 + 2], [pt.x - off - w, pt.y + H / 2 + 2], [pt.x + off, pt.y], [pt.x - off - w, pt.y], [pt.x - w / 2, pt.y - off - H / 2], [pt.x - w / 2, pt.y + off + H / 2]];
      for (const [x, y] of spots) {
        const box = { l: x - 2, t: y - H / 2, r: x + w + 2, b: y + H / 2 };
        if (!inside(box) || hits(box)) continue;
        taken.push(box);
        ctx.textAlign = "left";
        ctx.fillText(text, x, y);
        break;
      }
    }
    ctx.restore();
  },
};

function drawMap() {
  if (!window.Chart || $("pane-map").hidden) return;
  const C = window.Chart;
  C.defaults.color = cssVar("--text-2");
  C.defaults.borderColor = cssVar("--grid");
  C.defaults.font.family = getComputedStyle(document.body).fontFamily;
  const you = currentPoint();
  const fset = new Set(frontier);
  const combos = points.filter((p) => p.series === "combo" && plottable(p) && !fset.has(p));
  const xy = (p) => ({ x: p.x, y: p.y, p });
  const extra = (p) => SLOTS.map((k) => variant(p.over?.[k])).filter(Boolean).map((v) => v.short).join(" + ") || "GPU + CPU";
  const sameAsYou = (p) => p.series === "combo" && SLOTS.every((k) => (p.over?.[k] ?? null) === (cfg.sim[k] ?? null));
  const sortedNodes = (series) => points.filter((p) => p.series === series && plottable(p)).sort((a, b) => a.nodes - b.nodes);
  const datasets = [
    { label: "Other setups", data: combos.map(xy), backgroundColor: cssVar("--ghost"), borderColor: cssVar("--surface"), borderWidth: 2, pointRadius: 5, pointHoverRadius: 7, order: 5 },
    { label: "Best trade-offs (frontier)", data: frontier.map(xy), showLine: true, borderColor: cssVar("--accent"), backgroundColor: cssVar("--accent"), borderWidth: 2, pointRadius: 6, pointHoverRadius: 8, order: 3,
      labels: show.nodes || show.gpus ? null : frontier.map((p) => (p.series === "you" || sameAsYou(p) ? null : extra(p))), labelWide: true },
  ];
  if (show.nodes) {
    const ny = sortedNodes("nodes-yours"), nb = sortedNodes("nodes-base");
    datasets.push({ label: "Number of nodes, your tiers", data: ny.map(xy), showLine: true, borderColor: cssVar("--tier-p2p"), backgroundColor: cssVar("--tier-p2p"), borderWidth: 2, pointRadius: 4, order: 4, labels: ny.map((p) => `${p.nodes}n`) });
    datasets.push({ label: `Number of nodes, ${cfg.sim.cpuGiB > 0 ? "GPU + CPU" : "GPU only"}`, data: nb.map(xy), showLine: true, borderColor: cssVar("--tier-p2p"), backgroundColor: cssVar("--surface"), borderDash: [4, 3], borderWidth: 1.5, pointRadius: 4, order: 4, labels: nb.map((p) => `${p.nodes}n`) });
  }
  if (show.gpus) {
    const gs = points.filter((p) => p.series === "gpus" && plottable(p));
    datasets.push({ label: "Other GPUs, your tiers", data: gs.map(xy), backgroundColor: cssVar("--tier-cpu"), borderColor: cssVar("--surface"), borderWidth: 2, pointRadius: 6, pointStyle: "rectRot", order: 4, labels: gs.map((p) => gpuShort(gpuOf(p.gpu))) });
  }
  if (plottable(you)) datasets.push({ label: "Your setup", data: [xy(you)], backgroundColor: "transparent", borderColor: cssVar("--text"), borderWidth: 2.5, pointRadius: 11, pointHoverRadius: 12, order: 1, labels: ["You"], labelBelow: true });

  const wideRange = show.nodes || show.gpus;
  // fit the axes to what is current; the ghost is drawn only where it falls inside
  const cur = datasets.flatMap((d) => d.data);
  const niceStep = (raw) => { const e = 10 ** Math.floor(Math.log10(raw)); return [1, 2, 2.5, 5, 10].map((k) => k * e).find((k) => k >= raw); };
  const span = (vals) => {
    if (!vals.length) return {};
    const lo = Math.min(...vals), hi = Math.max(...vals);
    if (wideRange) return { min: lo / 1.2, max: hi * 1.2 };
    const pad = Math.max((hi - lo) * 0.08, hi * 0.02);
    const step = niceStep((hi - lo + 2 * pad) / 6);
    return { min: Math.max(0, Math.floor((lo - pad) / step) * step), max: Math.ceil((hi + pad) / step) * step };
  };
  // the old frontier joins the view when that does not shrink the current points too much (within 5x the spread)
  const withGhost = (k) => {
    const vals = cur.map((d) => d[k]);
    const all = [...vals, ...(ghost ?? []).map((g) => g[k])];
    const w = (v) => (wideRange ? Math.log(Math.max(...v) / Math.min(...v)) : Math.max(...v) - Math.min(...v));
    return ghost?.length && vals.length && w(all) <= 5 * Math.max(w(vals), wideRange ? 0.1 : Math.max(...vals) * 0.05) ? all : vals;
  };
  const xr = span(withGhost("x")), yr = span(withGhost("y"));
  const inView = (p) => p.x >= xr.min && p.x <= xr.max && p.y >= yr.min && p.y <= yr.max;
  if (ghost?.length) datasets.push({ label: `Frontier before your last change${ghost.some(inView) ? "" : " (off the chart)"}`, data: ghost.map(xy), showLine: true, borderColor: cssVar("--ghost"), borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, pointHitRadius: 0, order: 6 });
  // on a log axis, label only 1-2-5 steps, unless the range is narrow enough that every step fits
  const tick = (v, f, r) => { if (!wideRange || r.max / r.min < 5) return f(v); const m = v / 10 ** Math.floor(Math.log10(v)); return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-6) ? f(v) : ""; };
  const options = {
    layout: { padding: { right: 8 } },
    maintainAspectRatio: false, animation: false, parsing: false,
    interaction: { mode: "nearest", intersect: true },
    scales: {
      // node counts and other GPUs span orders of magnitude: log axes then
      x: { type: wideRange ? "logarithmic" : "linear", title: { display: true, text: `$ per 1,000 requests ${metricText()}${wideRange ? " (log)" : ""}` }, grid: { color: cssVar("--grid") }, ...xr, ticks: { maxTicksLimit: 8, callback: (v) => tick(v, usd, xr) } },
      y: { type: wideRange ? "logarithmic" : "linear", title: { display: true, text: `mean time to first token${wideRange ? " (log)" : ""}` }, grid: { color: cssVar("--grid") }, ...yr, ticks: { maxTicksLimit: 8, callback: (v) => tick(v, secs, yr) } },
    },
    plugins: {
      legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 14 } },
      tooltip: { filter: (_, i) => i === 0, callbacks: { label: (t) => { const p = t.raw.p; return [pointLabel(p), `TTFT ${secs(p.y)} (p90 ${secs(p.p90)}) · ${usd(p.x)} per 1k requests ${metricText()}`]; } } },
    },
    onClick: (_, els) => { if (els.length) { selected = els[0].element.$context.raw.p; renderDrawer(); } },
    onHover: (ev, els) => { ev.native.target.style.cursor = els.length ? "pointer" : "default"; },
  };
  if (chart && chart.$wide !== wideRange) { chart.destroy(); chart = null; } // a new axis type: start over
  if (chart) { chart.data.datasets = datasets; chart.options = options; chart.update("none"); }
  else { chart = new C($("map-chart"), { type: "scatter", data: { datasets }, options, plugins: [labelPlugin] }); chart.$wide = wideRange; }
  if (sweepComplete) renderExcluded();
}

function renderDrawer() {
  const p = selected;
  const box = $("map-drawer");
  box.hidden = !p;
  $("map-wrap").classList.toggle("open", !!p);
  $("map-tip").hidden = !!p;
  if (!p) { box.replaceChildren(); return; }
  const close = () => { selected = null; renderDrawer(); $("map-chart").focus?.(); };
  const apply = () => {
    const sim = cfg.sim;
    if (p.series === "gpus") { sim.gpu = p.over.gpu; sim.decodeGpu = p.over.decodeGpu; }
    else for (const [k, v] of Object.entries(p.over ?? {})) sim[k] = v;
    for (const k of SLOTS) if (sim[k]) lastPick[k] = sim[k];
    selected = null;
    renderDrawer();
    changed();
  };
  box.replaceChildren(
    el("div", { class: "drawer-head" }, el("h3", { text: pointLabel(p) }), el("button", { type: "button", text: "×", "aria-label": "Close", title: "Close", onclick: close })),
    el("div", { class: "nums" }, el("div", {}, el("b", { text: secs(p.y) }), el("span", { text: "mean TTFT" })), el("div", {}, el("b", { text: usd(p.x) }), el("span", { text: `per 1k requests ${metricText()}` }))),
    el("p", { class: "hint", text: `p90 TTFT ${secs(p.p90)}` }),
    el("div", { class: "served" }, (p.order ?? []).filter((k) => p.share[k] > 0.0005).map((k) => el("span", { style: `width:${(p.share[k] * 100).toFixed(2)}%;background:var(${TIER_COLOR[k]})`, title: `${k}: ${pctf(p.share[k], 1)}` }))),
    el("ul", { class: "served-legend" }, (p.order ?? []).map((k) => el("li", {}, el("i", { style: `background:var(${TIER_COLOR[k]})` }), el("span", { text: TIER_NAME(k, { ...cfg.sim, ...p.over }) }), el("b", { text: pctf(p.share[k]) })))),
    p.series === "you" ? null : el("button", { type: "button", class: "primary", text: "Use this setup", onclick: apply }),
  );
}

// ------------------------------------------------------------------------------------------------ theme

function initTheme() {
  let saved = "auto";
  try { saved = localStorage.getItem("kvtb-theme") || "auto"; } catch { /* no storage */ }
  const set = (t) => {
    if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
    document.querySelectorAll("[data-theme-choice]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeChoice === t)));
    try { localStorage.setItem("kvtb-theme", t); } catch { /* no storage */ }
    redrawCharts();
  };
  document.querySelectorAll("[data-theme-choice]").forEach((b) => b.addEventListener("click", () => set(b.dataset.themeChoice)));
  set(saved);
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", redrawCharts);
}

function redrawCharts() {
  if (!cfg) return;
  chart?.destroy();
  chart = null;
  sensChart?.destroy();
  sensChart = null;
  for (const k of Object.keys(modelsCharts)) { modelsCharts[k].destroy(); delete modelsCharts[k]; }
  scheduleChart();
  drawSens();
  drawModels();
  drawFitChart();
}

// ------------------------------------------------------------------------------------------------ start

function syncLastPick() {
  const cand = cfg.sim.candidates ?? {};
  for (const k of SLOTS) lastPick[k] = cfg.sim[k] ?? cand[k]?.[0] ?? slotVariants(k)[0]?.id ?? null;
  if (cfg.sim.cpuGiB > 0) lastPick.cpuGiB = cfg.sim.cpuGiB;
}

// the bar shows while the estimate is off screen; hidden, it leaves the keyboard and screen readers alone
function bindMinibar() {
  const bar = $("minibar");
  new IntersectionObserver(([e]) => {
    bar.classList.toggle("hide", e.isIntersecting);
    bar.inert = e.isIntersecting;
    bar.setAttribute("aria-hidden", String(e.isIntersecting));
  }, { threshold: 0.15 }).observe($("estimate"));
}

async function main() {
  try {
    s1 = await fetch("./data/s1.json").then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
    market = await loadOpenRouterPrices();
  } catch (e) {
    $("load-error").hidden = false;
    $("load-error").textContent = `Could not load data/s1.json (${e.message}). Serve this folder over HTTP, e.g. python -m http.server -d web.`;
    return;
  }
  defaults = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, MODELS);
  const saved = loadSaved(storage, defaults);
  cfg = saved ?? structuredClone(defaults);
  if (!cfg.models[cfg.sim.model]) cfg.sim.model = Object.keys(cfg.models)[0];
  // The simulator's "Open this setup" link carries its settings in the hash. They apply to this visit only: nothing is
  // saved until the user changes something here. The hash is cleared so a reload returns to the saved configuration.
  const handoff = readAdvancedHash(location.hash);
  let fromSimulator = false;
  if (handoff) {
    const next = withOverrides(cfg, handoff);
    if (!validate(next).length) {
      cfg = next;
      fromSimulator = true;
    }
    history.replaceState(null, "", location.pathname + location.search);
  }
  syncLastPick();
  initTheme();
  initMethods({ cfg: () => cfg, changed, s1, defaults, replaceCfg, storage, market });
  $("config-status").textContent = fromSimulator ? "Showing the simulator’s setup. It is saved in this browser once you change something here."
    : saved ? "Loaded your saved configuration from this browser." : "Defaults: published figures.";
  renderConfig();
  renderMethods();
  renderChips();
  renderMapMetric();
  document.fonts?.ready.then(redrawCharts); // canvas text keeps the fallback font until redrawn
  bindDiagram();
  bindMinibar();
  renderDiagramNow();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorker;
  worker.onerror = (e) => { $("map-status").textContent = `Simulation failed: ${e.message}`; };
  labWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  labWorker.onmessage = onLab;
  labWorker.onerror = (e) => { $("sens-status").textContent = `Simulation failed: ${e.message}`; };
  modelsWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  modelsWorker.onmessage = onModels;
  modelsWorker.onerror = (e) => { modelsRunning = false; $("models-status").textContent = `Simulation failed: ${e.message}`; };
  bindTabs();
  run();
  runSensSoon(0);
  if (!window.Chart) window.addEventListener("load", () => { scheduleChart(); drawSens(); drawFitChart(); });
}

main();
