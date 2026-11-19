// The page: the configurator, the live estimate, the cluster diagram and the trade-off map. Simulations run in
// worker.js; the maths is in model.js, sim.js and scenario.js.

import { defaultConfig, loadSaved, save } from "./config.js";
import { renderDiagram } from "./diagram.js";
import { drawFitChart, initMethods, renderMethods, slotOf } from "./methods.js";
import { gpuKvGiB, served, variantBandwidth } from "./model.js";
import { GPUS, MODELS, MOONCAKE_TRACES, SOFTWARE, SPEC_METHODS, VARIANTS } from "./presets.js";
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
  const m = served(cfg.models[cfg.sim.model], { precision: cfg.sim.precision, tp: cfg.sim.tp, attention: cfg.sim.attention });
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
  const m = served(cfg.models[cfg.sim.model], { precision: cfg.sim.precision, tp, attention: cfg.sim.attention });
  return gpuKvGiB(m, gpu, cfg.software, tp, pp);
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
  const node = $(ids[target] ?? target);
  if (!node) return;
  if (["routing", "decode"].includes(target)) $("serving").open = true;
  node.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  const box = node.closest(".tier, .panel, .field") ?? node;
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
const keyOf = () => JSON.stringify({ ...cfg, sim: { ...cfg.sim, disk: null, p2p: null, shared: null }, trace: trace ? trace.name : source });

function run() {
  if (!worker) return;
  if (source !== "synthetic" && !trace) { // nothing to replay yet
    jobId++;
    estimate = null;
    $("estimate").classList.remove("busy");
    for (const id of ["est-ttft", "est-usd", "mini-ttft", "mini-usd"]) $(id).textContent = "–";
    for (const id of ["est-ttft-sub", "est-usd-sub", "est-delta", "est-foot"]) $(id).textContent = "";
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
    $("estimate").classList.remove("busy");
    renderEstimate();
    keepFocus(renderTiers);
    renderDiagramNow();
    updateFrontier();
    scheduleChart();
  } else if (data.kind === "point") {
    points.push(data.point);
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
  renderGpuPicker();
  renderCluster();
  renderTiers();
  renderSliders();
  renderSettings();
}

function renderGpuPicker() {
  const sim = cfg.sim;
  $("ctl-gpu").replaceChildren(...cfg.gpus.map((g) => {
    const kv = kvFor(g);
    const on = g.id === sim.gpu;
    return el("button", {
      type: "button", class: `gpu-tile${kv <= 0 ? " nofit" : ""}`, role: "radio", "aria-checked": String(on),
      onclick: () => {
        if (sim.decodeGpu === sim.gpu) sim.decodeGpu = g.id;
        sim.gpu = g.id;
        if ((sim.pp ?? 1) > 1 || kvFor(g) <= 0) fitModel(); // across nodes: as few as this GPU needs
        changed();
      },
    },
    el("span", { class: "name", text: gpuShort(g) }),
    el("span", { class: "meta", text: `${Math.round(g.mem_gib)} GiB · ${g.usd_per_hour != null ? `$${g.usd_per_hour.toFixed(2)}/h` : "no price"}` }),
    el("span", { class: "meta fit", text: kv > 0 ? `${fmt(kv, 3)} GiB KV at TP ${sim.tp}${(sim.pp ?? 1) > 1 ? ` × ${sim.pp} nodes` : ""}` : `model doesn't fit at TP ${sim.tp}${(sim.pp ?? 1) > 1 ? ` × ${sim.pp} nodes` : ""}` }),
    g.assumptions?.length ? el("span", { class: "badge assumption", text: "assumption" }) : null);
  }));
  const models = Object.keys(cfg.models);
  const opt = (k) => el("option", { value: k, text: cfg.models[k].name ?? k, selected: k === sim.model });
  const inS1 = (k) => k in s1.models;
  $("ctl-model").replaceChildren(
    el("optgroup", { label: "Measured in s1" }, models.filter(inS1).map(opt)),
    el("optgroup", { label: "Other open-weight models" }, models.filter((k) => !inS1(k)).map(opt)));
  $("ctl-model").onchange = (ev) => { sim.model = ev.target.value; fitModel(); changed({ methods: true }); };
  const prec = sim.precision ?? "released";
  $("ctl-precision").replaceChildren(...[["released", "As released"], ["fp8", "FP8"], ["fp4", "FP4"]].map(([v, t]) => el("button", {
    type: "button", role: "radio", "aria-checked": String(v === prec), text: t,
    title: v === "released" ? `as the checkpoint ships: ${cfg.models[sim.model].released ?? "BF16"}` : v === "fp8" ? "FP8 weights, compute and KV cache" : "NVFP4 weights (4.5 bits per parameter), FP4 compute on Blackwell, FP8 KV cache",
    onclick: () => { sim.precision = v; fitModel(); changed({ methods: true }); },
  })));
  const mm = cfg.models[sim.model], mt = served(mm, { precision: prec, tp: sim.tp, attention: sim.attention });
  const mib = (b) => `${fmt(b / 2 ** 20, 3)} MiB`;
  const params = (n) => (n >= 1e12 ? `${fmt(n / 1e12, 3)}T` : `${fmt(n / 1e9, 3)}B`);
  fill($("model-hint"),
    `${mm.arch ?? ""}${mm.arch ? ". " : ""}${fmt(mt.kv_bytes_per_token / 1024, 3)} KiB of KV per token at TP ${sim.tp}`,
    mt.read_fixed_bytes > 0 ? `, plus ${mib(mt.read_fixed_bytes)} per cached prefix that does not grow with its length` : "",
    `. ${params(mm.params)} parameters active per token (${params(mm.total_params ?? mm.params)} in all). Released as ${mm.released ?? "BF16"}; served ${prec === "released" ? "as released" : `in ${prec.toUpperCase()}`}: ${fmt(mt.weights_bytes / 1e9, 4)} GB of weights, computed in ${(mt.compute_dtype ?? "bf16").toUpperCase()}${mt.compute_dtype === "fp4" && !gpuOf(sim.gpu).tflops_fp4 ? ` (at FP8 speed: ${gpuShort(gpuOf(sim.gpu))} has no FP4 tensor cores)` : ""}.`,
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
    sim.nodes > MAX_SIM_NODES ? ` Simulated at ${MAX_SIM_NODES} nodes with the same load per node, then scaled.` : null,
  );
}

function tierRow({ key, title, sub, on, toggle, disabledReason, body, tags }) {
  const sw = toggle ? el("label", { class: "switch" },
    el("input", { type: "checkbox", checked: on, disabled: !!disabledReason, "aria-label": `Use ${title}`, onchange: (ev) => toggle(ev.target.checked) }),
    el("span", { class: "track" }), el("span", { text: disabledReason ? "unavailable" : on ? "on" : "off" })) : el("span", { class: "tier-sub", text: "always on" });
  return el("div", { class: `tier${on ? "" : " off"}`, id: `tier-${key}` },
    el("span", { class: "swatch", style: `background:var(${TIER_COLOR[key]})` }),
    el("div", {},
      el("div", { class: "tier-head" }, el("div", {}, el("div", { class: "tier-title", text: title }), el("div", { class: "tier-sub", text: disabledReason ?? sub })), sw),
      on && body ? el("div", { class: `tier-body${body.length === 1 ? " one" : ""}` }, body) : null,
      on && tags?.length ? el("div", { class: "tier-tags", style: "margin-top:8px" }, tags) : null));
}

function variantSelect(slot) {
  const sim = cfg.sim;
  return el("label", { class: "field" }, el("span", { text: "Type" }),
    el("select", { onchange: (ev) => { sim[slot] = ev.target.value; lastPick[slot] = ev.target.value; changed(); } },
      slotVariants(slot).map((v) => el("option", { value: v.id, text: v.short, selected: v.id === sim[slot] }))));
}

function variantTags(v, extra = []) {
  const price = v.family === "p2p" ? (v.usd_per_hour ? `$${v.usd_per_hour}/h per node` : null)
    : v.usd_per_gib_month != null ? `$${v.usd_per_gib_month}/GiB-month${v.bill === "provisioned" ? ", provisioned" : ", stored"}` : "no price";
  const perf = `${fmt(variantBandwidth(v), 3)} GiB/s + ${fmt((v.latency_s ?? 0) * 1000, 3)} ms`;
  return [el("span", { class: "tier-sub", text: perf }), ...(v.assumptions?.length ? [assumptionBadge(v.assumptions.join(", "))] : []),
    ...(v.edited?.length ? [el("span", { class: "badge assumption", text: "edited" })] : []),
    el("span", { class: "tier-sub", text: [price, ...extra].filter(Boolean).join(" · ") })];
}

function renderTiers() {
  const sim = cfg.sim;
  const g = gpuOf(sim.gpu);
  const kv = kvFor(g);
  const rows = [];
  rows.push(tierRow({ key: "gpu", title: "GPU memory", on: true,
    sub: kv > 0 ? `${fmt(kv, 3)} GiB of KV cache per instance, after weights and activations: ${capacity(kv)}` : "the model does not fit",
    tags: [el("span", { class: "tier-sub", text: `${fmt(g.mem_bw_tb_s, 3)} TB/s; part of the GPU price` })] }));
  rows.push(tierRow({ key: "cpu", title: "CPU memory", on: sim.cpuGiB > 0, sub: sim.cpuGiB > 0 ? `per node, shared by its GPUs: ${perGroup(sim.cpuGiB)}` : "per node, shared by its GPUs",
    toggle: (on) => { if (!on) lastPick.cpuGiB = sim.cpuGiB || lastPick.cpuGiB; sim.cpuGiB = on ? lastPick.cpuGiB : 0; changed(); },
    body: [numField("GiB per node", sim, "cpuGiB", { step: 1, onchange: () => { sim.cpuGiB = Math.max(1, sim.cpuGiB ?? 1); lastPick.cpuGiB = sim.cpuGiB; changed(); } })],
    tags: [el("span", { class: "tier-sub", text: `${fmt(g.host_gib_s, 3)} GiB/s to the GPU + ${fmt(cfg.software.cpu_load_latency_s * 1000, 2)} ms · part of the VM: no extra cost` })] }));
  const disk = variant(sim.disk);
  rows.push(tierRow({ key: "disk", title: "Local disk", on: !!disk, sub: disk ? `one per node: ${perGroup(disk.capacity_gib)}` : "one per node",
    toggle: (on) => { sim.disk = on ? lastPick.disk : null; changed(); },
    body: disk ? [variantSelect("disk"), el("div", { class: "field" }, el("span", { text: "Size per node" }), el("span", { class: "tier-sub", style: "color:var(--text)", text: gibf(disk.capacity_gib) }))] : null,
    tags: disk ? variantTags(disk) : null }));
  const p2p = variant(sim.p2p);
  const p2pBlocked = sim.nodes < 2 ? "needs 2 or more nodes" : !(sim.cpuGiB > 0) ? "needs CPU memory: peers serve from it" : null;
  const p2pSub = sim.routing === "sticky" ? "another node's CPU memory. Sticky routing keeps a conversation on one node, so peers mostly share the system prompt" : "another node's CPU memory, over the network";
  rows.push(tierRow({ key: "p2p", title: "Peer nodes", on: !!p2p && !p2pBlocked, sub: p2pSub, disabledReason: p2pBlocked,
    toggle: (on) => { sim.p2p = on ? lastPick.p2p : null; changed(); },
    body: p2p ? [variantSelect("p2p")] : null, tags: p2p ? variantTags(p2p) : null }));
  const shared = variant(sim.shared);
  rows.push(tierRow({ key: "shared", title: "Shared storage", on: !!shared, sub: shared ? `one store for the whole cluster: ${capacity(shared.capacity_gib)}` : "one store for the whole cluster",
    toggle: (on) => { sim.shared = on ? lastPick.shared : null; changed(); },
    body: shared ? [variantSelect("shared"), shared.bill === "provisioned"
      ? el("div", { class: "field" }, el("span", { text: "Size" }), el("span", { class: "tier-sub", style: "color:var(--text)", text: gibf(shared.capacity_gib) }))
      : numField("Keep for (hours)", sim, "retentionH", { onchange: () => { sim.retentionH = Math.max(0.01, sim.retentionH ?? 1); changed(); } })] : null,
    tags: shared ? variantTags(shared, [shared.capacity_gib == null ? "scales without limit" : null]) : null }));
  $("tiers").replaceChildren(...rows);
}

// a slider with a number box; `log` maps the slider logarithmically
const SLIDERS = [
  { key: "requestsPerMin", label: "Requests per minute", hint: "all users together", min: 1, max: 100000, log: true },
  { key: "followUpShare", label: "Follow-up messages", hint: "share that continue a conversation (cache reuse)", min: 0, max: 95, scale: 100, unit: "%" },
  { key: "inputTokens", label: "Tokens in per message", hint: "the new text a user sends", min: 10, max: 32000, log: true },
  { key: "outputTokens", label: "Tokens out per answer", hint: "generated, then cached with the conversation", min: 1, max: 16000, log: true },
  { key: "thinkTimeS", label: "Pause between turns (s)", hint: "user reading and typing", min: 1, max: 7200, log: true },
];

function renderSliders() {
  const w = cfg.workload;
  const off = source !== "synthetic";
  $("sliders").classList.toggle("disabled", off);
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
    el("label", { class: "field" }, el("span", { text: "Routing" }),
      el("select", { id: "ctl-routing", onchange: (ev) => { sim.routing = ev.target.value; re(); } },
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
    numField("Simulated block (tokens)", sim, "simBlockTokens", { step: 1, onchange: () => { sim.simBlockTokens = Math.max(16, Math.round(sim.simBlockTokens ?? 512)); re(); } }),
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

// Decode speed per request: blank fills the batch; a number runs the largest batch that holds it. OpenRouter's p50 for
// the model, when the snapshot has one, is a click away.
function speedField(sim, re) {
  const p50 = market?.models?.[sim.model]?.speed;
  const field = numField("Decode speed per request (tok/s)", sim, "decodeSpeed", { id: "ctl-decode-speed", placeholder: "blank: fill the batch",
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
    $("est-ttft").textContent = "–"; $("est-usd").textContent = "–"; $("est-ttft-sub").textContent = ""; $("est-usd-sub").textContent = "";
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
    for (const id of ["est-ttft", "est-usd", "mini-ttft", "mini-usd"]) $(id).textContent = "–";
    for (const id of ["est-ttft-sub", "est-usd-sub", "est-delta", "est-foot"]) $(id).textContent = "";
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
  $("mini-ttft").textContent = overloaded ? "overloaded" : secs(r.ttft.mean);
  $("mini-usd").textContent = r.priced ? usd(r.usdPer1kRequests) : "no price";

  // against GPU + CPU (or GPU only) on the same cluster
  const isBase = SLOTS.every((k) => !sim[k]) || (!sim.disk && !sim.shared && sim.p2p && sim.nodes < 2);
  const baseName = sim.cpuGiB > 0 ? "GPU + CPU" : "GPU only";
  if (overloaded || b.error || !b.requests) $("est-delta").textContent = "";
  else if (isBase) $("est-delta").replaceChildren(el("span", { class: "hint", text: `This is the ${baseName} baseline. Turn on a tier to compare.` }));
  else {
    const dt = r.ttft.mean / b.ttft.mean - 1;
    const dc = r.priced && b.priced ? r.usdPer1kRequests / b.usdPer1kRequests - 1 : null;
    const word = (x, less, more, same) => (Math.abs(x) < 0.005 ? same : `${Math.abs(x * 100).toFixed(0)}% ${x < 0 ? less : more}`);
    $("est-delta").replaceChildren(
      el("span", { class: dt < -0.005 ? "better" : dt > 0.005 ? "worse" : "", text: word(dt, "faster", "slower", "about as fast") }),
      dc == null ? "" : ", ",
      dc == null ? "" : el("span", { class: dc < -0.005 ? "better" : dc > 0.005 ? "worse" : "", text: word(dc, "cheaper", "more expensive", "about the same cost") }),
      ` vs ${baseName} (${secs(b.ttft.mean)}${b.priced ? `, ${usd(b.usdPer1kRequests)}` : ""})`);
  }
  if (overloaded) alert(`Overloaded: prefill needs ${pctf(r.utilization)} of the GPUs' time, so the queue grows for as long as the load lasts and TTFT has no steady value. Add nodes or GPUs, or cache more.`, true);
  else if (r.utilization > 0.8) alert(`Prefill GPUs are ${pctf(r.utilization)} busy: waiting in the queue adds ${secs(r.wait)} on average.`);
  if (!r.priced) alert(`No price for ${[gpuOf(sim.gpu), sim.decode ? gpuOf(sim.decodeGpu) : null].filter((g) => g && g.usd_per_hour == null).map(gpuShort).filter((x, i, a) => a.indexOf(x) === i).join(" and ") || "a tier"}: set one in Methods & data to see cost.`);
  $("est-alerts").replaceChildren(...alerts);

  const pr = r.price;
  const priceCell = (x, label, title) => el("div", { title }, el("b", { text: x == null ? "–" : usd(x) }), el("span", { text: label }));
  $("est-prices").replaceChildren(
    priceCell(pr.input, "input", "Prompt tokens recomputed: their share of the prefill GPUs (idle time included)."),
    priceCell(pr.cachedInput, "cached input", "Prompt tokens served from a cache tier: their share of the prefill GPUs (load time) plus storage."),
    priceCell(pr.output, sim.decode ? "output" : "output (decode off)", "Generated tokens: the decode GPUs."));
  if (pr.cachedInput != null && pr.input != null && pr.cachedInput > pr.input)
    $("est-prices").append(el("p", { class: "note warn", text: "A cached token costs more than computing it here: the tiers' storage and load time outweigh the compute they save." }));
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
    ["Storage, $/hour", usd(r.usd.storage / hours)],
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

const currentPoint = () => {
  const r = estimate?.current;
  if (!r || r.error) return null;
  return { series: "you", x: r.usdPer1kRequests, y: r.ttft.mean, p90: r.ttft.p90, priced: r.priced, overloaded: r.utilization >= 1, share: r.share, order: r.order, over: {} };
};
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

// direct labels for the frontier, your setup and the curves' points
const labelPlugin = {
  id: "kvtbLabels",
  afterDatasetsDraw(c) {
    const ctx = c.ctx;
    ctx.save();
    ctx.font = `12px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillStyle = cssVar("--text-2");
    const narrow = c.width < 520;
    c.data.datasets.forEach((ds, i) => {
      if (!ds.labels || !c.isDatasetVisible(i)) return;
      c.getDatasetMeta(i).data.forEach((pt, j) => {
        const text = ds.labels[j];
        if (!text || (narrow && ds.labelWide)) return;
        const right = pt.x < c.chartArea.right - 120;
        ctx.textAlign = right ? "left" : "right";
        if (ds.labelBelow) { ctx.textAlign = "center"; ctx.fillText(text, pt.x, pt.y + 26); }
        else ctx.fillText(text, pt.x + (right ? 10 : -10), pt.y - 8);
      });
    });
    ctx.restore();
  },
};

function drawMap() {
  if (!window.Chart) return;
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
  // on a log axis, label only 1-2-5 steps
  const tick = (v, f) => { if (!wideRange) return f(v); const m = v / 10 ** Math.floor(Math.log10(v)); return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-6) ? f(v) : ""; };
  const options = {
    layout: { padding: { right: 8 } },
    maintainAspectRatio: false, animation: false, parsing: false,
    interaction: { mode: "nearest", intersect: true },
    scales: {
      // node counts and other GPUs span orders of magnitude: log axes then
      x: { type: wideRange ? "logarithmic" : "linear", title: { display: true, text: `$ per 1,000 requests${wideRange ? " (log)" : ""}` }, grid: { color: cssVar("--grid") }, ...xr, ticks: { maxTicksLimit: 8, callback: (v) => tick(v, usd) } },
      y: { type: wideRange ? "logarithmic" : "linear", title: { display: true, text: `mean time to first token${wideRange ? " (log)" : ""}` }, grid: { color: cssVar("--grid") }, ...yr, ticks: { maxTicksLimit: 8, callback: (v) => tick(v, secs) } },
    },
    plugins: {
      legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 14 } },
      tooltip: { filter: (_, i) => i === 0, callbacks: { label: (t) => { const p = t.raw.p; return [pointLabel(p), `TTFT ${secs(p.y)} (p90 ${secs(p.p90)}) · ${usd(p.x)} per 1k requests`]; } } },
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
  if (!p) { box.replaceChildren(el("p", { class: "hint", text: "Click a point to see that setup." })); return; }
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
    el("h3", { text: pointLabel(p) }),
    el("div", { class: "nums" }, el("div", {}, el("b", { text: secs(p.y) }), el("span", { text: "mean TTFT" })), el("div", {}, el("b", { text: usd(p.x) }), el("span", { text: "per 1k requests" }))),
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
  scheduleChart();
  drawFitChart();
}

// ------------------------------------------------------------------------------------------------ start

function syncLastPick() {
  const cand = cfg.sim.candidates ?? {};
  for (const k of SLOTS) lastPick[k] = cfg.sim[k] ?? cand[k]?.[0] ?? slotVariants(k)[0]?.id ?? null;
  if (cfg.sim.cpuGiB > 0) lastPick.cpuGiB = cfg.sim.cpuGiB;
}

function bindMinibar() {
  const bar = $("minibar");
  new IntersectionObserver(([e]) => bar.classList.toggle("hide", e.isIntersecting), { threshold: 0.15 }).observe($("estimate"));
}

async function main() {
  try {
    s1 = await fetch("./data/s1.json").then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
    market = await fetch("./data/openrouter.json").then((r) => (r.ok ? r.json() : null)).catch(() => null); // optional
  } catch (e) {
    $("load-error").hidden = false;
    $("load-error").textContent = `Could not load data/s1.json (${e.message}). Serve this folder over HTTP, e.g. python -m http.server -d web.`;
    return;
  }
  defaults = defaultConfig(s1, VARIANTS, GPUS, SOFTWARE, MODELS);
  const saved = loadSaved(storage, defaults);
  cfg = saved ?? structuredClone(defaults);
  if (!cfg.models[cfg.sim.model]) cfg.sim.model = Object.keys(cfg.models)[0];
  syncLastPick();
  initTheme();
  initMethods({ cfg: () => cfg, changed, s1, defaults, replaceCfg, storage, market });
  $("config-status").textContent = saved ? "Loaded your saved configuration from this browser." : "Defaults: published figures.";
  renderConfig();
  renderMethods();
  renderChips();
  bindDiagram();
  bindMinibar();
  renderDiagramNow();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorker;
  worker.onerror = (e) => { $("map-status").textContent = `Simulation failed: ${e.message}`; };
  run();
  if (!window.Chart) window.addEventListener("load", () => { scheduleChart(); drawFitChart(); });
}

main();
