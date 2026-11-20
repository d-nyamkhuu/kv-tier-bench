import { defaultConfig } from "./config.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "./presets.js";
import { PRECISION_LABELS } from "./model.js";
import { loadOpenRouterPrices, showTokenPrices } from "./openrouter.js";
import { tierStyles, timeline } from "./replay-chart.js";
import { minutesCsv } from "./replay.js";
import {
  applySettings, CPU_SLIDER_MAX_GIB, cpuSizingNote, decodeSettings, modelDefaults, pricesAt, risingTtft, DEFAULT_SETTINGS, encodeSettings, fitCluster, SCENARIOS, scenarioFor, usdPerHour,
} from "./sim-settings.js";
import { $, el, gibf, pctf, secs } from "./ui.js";

const controls = $("controls"), backing = $("backing");
const plot = timeline();
let market;
let showPrices = () => {}, simulatedPrices = null; // the token-price table, set up in start()
let config, worker, revision = 0, timer;
let comparisonVisible = false, comparisonLayout = null, lastResult = null;
// `shown` is the result on screen. `baseline` is the result before the change in progress: a slider drag
// counts as one change from press to release.
let shown = null, baseline = null, editing = false;
// The cluster the user chose before a larger model raised it, restored once a model fits it again.
let ownCluster = null;
// Physical KV GiB per instance from the latest replay, so the budget slider can say what its percentage holds.
let kvPerInstanceGiB = null;
const sliders = new Map();

// A slider over `stops` (an array of values) or over min..max in `step`s.
function slider(group, key, label, { min, max, step, stops, units }) {
  const output = el("output", { for: key });
  const input = stops
    ? el("input", { id: key, type: "range", min: 0, max: stops.length - 1, step: 1 })
    : el("input", { id: key, type: "range", min, max, step });
  const value = () => stops ? stops[input.valueAsNumber] : input.valueAsNumber;
  const refresh = () => {
    output.value = typeof units === "function" ? units(value()) : value() + units;
    input.setAttribute("aria-valuetext", output.value);
    input.style.setProperty("--fill", (input.valueAsNumber - input.min) / (input.max - input.min) * 100 + "%");
  };
  const set = v => {
    if (stops) {
      // An exact stop, else the nearest numeric one, so a hand-edited URL still lands on the slider.
      let index = stops.indexOf(v);
      if (index < 0) {
        const numeric = stops.map((s, i) => [s, i]).filter(([s]) => s != null);
        index = v == null ? stops.length - 1 : numeric.reduce((best, cur) => Math.abs(cur[0] - v) < Math.abs(best[0] - v) ? cur : best)[1];
      }
      input.value = index;
    } else input.value = v;
    refresh();
  };
  $(group).append(el("div", { class: "field" },
    el("div", { class: "field-head" }, el("label", { htmlFor: key, text: label }), output), input));
  sliders.set(key, { value, set, refresh });
}

const cpuSizes = [0, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, CPU_SLIDER_MAX_GIB];
// Powers of two give small working sets room on the slider. The final stop uses the drive preset.
const diskSizes = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, null];
slider("workload-controls", "rate", "Target request rate", { min: 10, max: 600, step: 10, units: " req/min" });
slider("workload-controls", "turns", "Average turns per chat", { stops: [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20], units: v => v === 1 ? "1 · no follow-ups" : v + " turns" });
slider("workload-controls", "pause", "Think time after reply", { min: 1, max: 30, step: 1, units: " min" });
slider("cache-controls", "cpu", "CPU cache per node", { stops: cpuSizes, units: v => v === 0 ? "Off" : gibf(v) });
slider("disk-controls", "disk-size", "SSD capacity per node", { stops: diskSizes, units: v => v == null ? "Full drive" : gibf(v) });
slider("detail-controls", "prompts", "Distinct system prompts", { stops: [1, 2, 5, 10, 20, 50, 100, 1000], units: v => v === 1 ? "1 · shared by all" : String(v) });
slider("detail-controls", "prefix", "System prompt length", { min: 0, max: 16000, step: 100, units: " tokens" });
slider("detail-controls", "input", "Average new message", { min: 100, max: 4000, step: 100, units: " tokens" });
slider("detail-controls", "output", "Average reply", { min: 100, max: 4000, step: 100, units: " tokens" });
slider("memory-controls", "batch", "Prefill batch per step", { stops: [0, 2048, 4096, 8192, 16384, 32768],
  units: v => v === 0 ? "Off · one at a time" : v.toLocaleString("en-US") + " tokens" });
slider("memory-controls", "gpu-budget", "GPU KV budget", { min: 1, max: 100, step: 1,
  units: v => v + "%" + (kvPerInstanceGiB ? " · " + gibf(+(kvPerInstanceGiB * v / 100).toPrecision(3)) : "") });

// Settings key → [control id, kind]. Selects hold strings except the two counts.
const FIELDS = {
  model: ["model"], precision: ["precision"], gpu: ["gpu"], nodes: ["nodes", "count"], gpusPerNode: ["gpus-per-node", "count"],
  interconnect: ["interconnect"], rate: ["rate", "slider"], turns: ["turns", "slider"], pause: ["pause", "slider"],
  pattern: ["pattern"], routing: ["routing"], hours: ["hours", "count"], priceAt: ["price-at"], cpu: ["cpu", "slider"], disk: ["disk"], diskSize: ["disk-size", "slider"],
  lookup: ["lookup-policy"], backing: ["backing"], start: ["start"], spread: ["spread"], prompts: ["prompts", "slider"], batch: ["batch", "slider"],
  admission: ["cache-admission"], prefix: ["prefix", "slider"], input: ["input", "slider"],
  output: ["output", "slider"], gpuBudget: ["gpu-budget", "slider"],
};

function readSettings() {
  const settings = {};
  for (const [key, [id, kind]] of Object.entries(FIELDS)) {
    if (kind === "slider") settings[key] = sliders.get(id).value();
    else if (kind === "count") settings[key] = Number($(id).value);
    else settings[key] = $(id).value;
  }
  return settings;
}

// Values the page has no option for keep the default.
function writeSettings(settings) {
  for (const [key, [id, kind]] of Object.entries(FIELDS)) {
    const value = settings[key] ?? DEFAULT_SETTINGS[key];
    if (kind === "slider") sliders.get(id).set(value);
    else {
      const select = $(id), text = String(value);
      select.value = [...select.options].some(o => o.value === text) ? text : String(DEFAULT_SETTINGS[key]);
    }
  }
}

function showScenario(settings) {
  const active = scenarioFor(settings);
  for (const button of $("scenario-buttons").children) button.setAttribute("aria-pressed", String(button.dataset.id === active?.id));
  const note = $("scenario-note");
  if (!active) {
    note.textContent = "Custom settings.";
    return;
  }
  note.textContent = active.description;
}

function setCluster({ nodes, gpusPerNode }) {
  $("nodes").value = String(nodes);
  $("gpus-per-node").value = String(gpusPerNode);
}

function showFit(text) {
  $("fit-note").textContent = text ?? "";
  $("fit-note").hidden = !text;
}

// A model, GPU or precision change grows the cluster to the smallest one that holds the model.
function refit(id) {
  if (id === "nodes" || id === "gpus-per-node") { ownCluster = null; showFit(); return; }
  if (!["model", "gpu", "precision"].includes(id)) return;
  // A new model starts from its serving defaults: FP8 when released in BF16, RDMA between nodes.
  if (id === "model") {
    const defaults = modelDefaults(config.models[$("model").value]);
    $("precision").value = defaults.precision;
    $("interconnect").value = defaults.interconnect;
  }
  const settings = readSettings();
  if (ownCluster && !fitCluster(config, { ...settings, ...ownCluster })) {
    setCluster(ownCluster);
    showFit(`Returned to your cluster of ${ownCluster.nodes} node${ownCluster.nodes === 1 ? "" : "s"} × ${ownCluster.gpusPerNode} GPUs, which holds this model.`);
    ownCluster = null;
    return;
  }
  let fit = fitCluster(config, settings);
  let switched = "";
  // A new model the current GPU cannot hold in this cluster moves to B200 (then B300) when that needs fewer GPUs than
  // growing the cluster on the current GPU, or when no layout of the current GPU holds it at all.
  if (fit && id === "model") {
    const from = config.gpus.find(g => g.id === settings.gpu);
    const count = f => f == null ? settings.nodes * settings.gpusPerNode : f.error ? Infinity : f.nodes * f.gpusPerNode;
    for (const gpu of ["b200", "b300"].filter(g => g !== settings.gpu)) {
      const next = fitCluster(config, { ...settings, gpu });
      if (next?.error || count(next) >= count(fit)) continue;
      const name = config.models[settings.model].name, to = config.gpus.find(g => g.id === gpu).name;
      switched = fit.error
        ? `${name} does not fit any cluster of ${from?.name ?? settings.gpu}, so the GPU changed to ${to}. `
        : `${name} needs ${count(fit)} ${from?.name ?? settings.gpu} GPUs but ${count(next)} ${to} GPUs, so the GPU changed to ${to}. `;
      $("gpu").value = gpu;
      Object.assign(settings, { gpu });
      fit = next;
      break;
    }
  }
  if (!fit) { if (switched || !ownCluster) showFit(switched.trim() || undefined); return; }
  if (fit.error) { showFit(fit.error); return; }
  ownCluster ??= { nodes: settings.nodes, gpusPerNode: settings.gpusPerNode };
  setCluster(fit);
  const gpu = config.gpus.find(g => g.id === settings.gpu);
  showFit(switched + `Raised to ${fit.nodes} node${fit.nodes === 1 ? "" : "s"} × ${fit.gpusPerNode} GPUs, the smallest cluster that holds ${config.models[settings.model].name} on ${gpu.name}.`);
}

function useSettings(settings) {
  ownCluster = null;
  showFit();
  const before = readSettings();
  baseline = shown;
  editing = false;
  writeSettings(settings);
  const after = readSettings();
  const changed = Object.keys(FIELDS).filter(key => before[key] !== after[key]);
  // A changed setting inside a closed section would otherwise be invisible.
  if (changed.some(key => ["start", "spread", "prompts", "batch", "prefix", "input", "output", "gpuBudget", "admission", "lookup"].includes(key))) $("detail-controls").closest("details").open = true;
  showScenario(after);
  schedule();
}

$("scenario-buttons").append(...SCENARIOS.map(s => el("button", {
  type: "button", "data-id": s.id, "aria-pressed": "false", text: s.name, onclick: () => useSettings(s.settings),
})));

function requestComparison() {
  if (!comparisonVisible || !comparisonLayout) return;
  $("comparison-status").textContent = "Comparing cache tiers…";
  worker.postMessage({ id: revision, kind: "comparison", config, layout: comparisonLayout,
    budgetFraction: sliders.get("gpu-budget").value() / 100 });
  comparisonLayout = null;
}

new IntersectionObserver(([entry]) => {
  comparisonVisible = entry.isIntersecting;
  requestComparison();
}, { rootMargin: "200px" }).observe($("tier-comparison"));

function renderComparison(rows) {
  const currentMean = shown.mean;
  const maxTime = Math.max(currentMean, ...rows.map(row => row.mean ?? 0));
  const maxRate = Math.max(0, ...rows.map(row => row.rate ?? 0));
  const bar = (value, max, text, mark = null, title = "") => el("td", { title }, text,
    el("div", { class: "compare-track", "aria-hidden": "true" },
      el("i", { style: `width:${max ? value / max * 100 : 0}%` }),
      mark == null ? null : el("b", { style: `left:${max ? mark / max * 100 : 0}%` })));
  $("tier-bars").tBodies[0].replaceChildren(...rows.map(row => el("tr", {},
    el("th", { scope: "row", text: row.name }),
    row.mean == null ? el("td", { text: row.error }) : bar(row.mean, maxTime, secs(row.mean), currentMean),
    row.rate == null ? el("td", { class: "compare-note", text: row.billing })
      : bar(row.rate, maxRate, "$" + row.rate.toFixed(4), null, row.billing + " capacity"),
  )));
  $("tier-bars").hidden = false;
  $("comparison-status").textContent = `The tick marks the current setup’s average TTFT of ${secs(currentMean)}.`;
}

// The bar under the results header: indeterminate until the worker reports its first minute, hidden when done.
function showProgress(fraction) {
  const bar = $("progress");
  if (fraction == null) {
    bar.hidden = true;
    $("status").classList.remove("busy");
    return;
  }
  bar.hidden = false;
  $("status").classList.add("busy");
  bar.classList.toggle("indeterminate", fraction <= 0);
  const percent = Math.round(fraction * 100);
  bar.firstElementChild.style.width = fraction > 0 ? percent + "%" : "";
  if (fraction > 0) bar.setAttribute("aria-valuenow", String(percent)); else bar.removeAttribute("aria-valuenow");
  $("status").textContent = fraction > 0 ? `Simulating… ${percent}%` : "Simulating…";
}

function runWorker(id) {
  worker = new Worker(new URL("./replay-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.id !== revision) return;
    if (data.kind === "progress") { showProgress(data.fraction); return; }
    if (data.kind === "comparison") {
      if (data.error) $("comparison-status").textContent = data.error;
      else renderComparison(data.result);
    } else if (data.error) failed(data.error);
    else {
      render(data.result);
      comparisonLayout = data.result.layout;
      requestComparison();
    }
  };
  worker.onerror = () => {
    if (id !== revision) return;
    if ($("results").getAttribute("aria-busy") === "true") failed("Could not run the replay worker.");
    else $("comparison-status").textContent = "Could not compare these tiers.";
  };
  worker.postMessage({ id, config, budgetFraction: sliders.get("gpu-budget").value() / 100 });
}

function schedule() {
  $("disk-controls").hidden = !$("disk").value;
  $("disk-size").disabled = !$("disk").value;
  for (const { refresh } of sliders.values()) refresh();
  if (!config || !$("model").value) return;
  worker?.terminate();
  comparisonLayout = null;
  $("tier-bars").hidden = true;
  $("comparison-status").textContent = "Waiting for replay…";
  const id = ++revision;
  clearTimeout(timer);
  const settings = readSettings();
  applySettings(config, settings);
  const hash = encodeSettings(settings);
  history.replaceState(null, "", hash ? "#" + hash : location.pathname + location.search);
  $("results").setAttribute("aria-busy", "true");
  $("status").className = "";
  showProgress(0);
  timer = setTimeout(() => runWorker(id), 120);
}

controls.addEventListener("input", event => {
  if (event.target.id === "price-at") {
    const settings = readSettings();
    if (lastResult) showPrices(pricesAt(lastResult.tokenPrices, lastResult.utilization, settings.priceAt));
    const hash = encodeSettings(settings);
    history.replaceState(null, "", hash ? "#" + hash : location.pathname + location.search);
    showScenario(settings);
    return;
  }
  if (!editing) { baseline = shown; editing = true; }
  if (config) refit(event.target.id);
  showScenario(readSettings());
  schedule();
});
// A slider fires change on release and a select with each pick, so either ends the change in progress.
controls.addEventListener("change", () => { editing = false; });
const editSettings = $("edit-settings");
editSettings.onclick = () => {
  editSettings.setAttribute("aria-expanded", String(editSettings.getAttribute("aria-expanded") !== "true"));
};
$("reset").addEventListener("click", () => useSettings(DEFAULT_SETTINGS));
$("download-csv").addEventListener("click", () => {
  if (!lastResult) return;
  const link = el("a", { href: URL.createObjectURL(new Blob([minutesCsv(lastResult)], { type: "text/csv" })), download: "kv-cache-replay-minutes.csv" });
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
});

function changeText(delta, same, amount) {
  if (Math.abs(delta) < same) return "Same as before your last change";
  return (delta < 0 ? "−" : "+") + amount + " vs before your last change";
}

// The page prices the input side only: prefill GPUs and cache tiers. Decode GPUs are simulated (they set reply timing)
// but their cost belongs to output tokens, so it is left out. A GPU without a listed price counts as zero.
const prefillGpuCost = result => result.gpuCostPerHour == null ? null : result.prefillCostPerHour;
const hourlyCost = result => (prefillGpuCost(result) ?? 0) + result.costPerHour;

// A row of labelled figures under a headline metric.
function stats(id, pairs) {
  $(id).replaceChildren(...pairs.map(([label, value]) => el("div", {}, el("dt", { text: label }), el("dd", { text: value }))));
}

function render(result) {
  const settings = readSettings();
  $("results").classList.remove("failed");
  if (baseline) {
    const latency = result.mean - baseline.mean, cost = hourlyCost(result) - baseline.hourly;
    $("ttft-change").textContent = changeText(latency, 0.0005, secs(Math.abs(latency)));
    $("cost-change").textContent = changeText(cost, 0.0005, usdPerHour(Math.abs(cost)) + "/h");
  } else $("ttft-change").textContent = $("cost-change").textContent = "";
  shown = { mean: result.mean, hourly: hourlyCost(result) };
  lastResult = result;
  $("download-csv").disabled = false;
  $("ttft").textContent = secs(result.mean);
  const { p50, p90, p99 } = result.ttft;
  stats("ttft-stats", [["p50", secs(p50)], ["p90", secs(p90)], ["p99", secs(p99)]]);
  const period = result.minutes.length % 60 === 0 && result.minutes.length > 60 ? `${result.minutes.length / 60} hours` : `${result.minutes.length} minutes`;
  $("ttft-detail").textContent = result.warmupMin ? `Measured over ${period} after a ${result.warmupMin}-minute warm-up` : `Measured over ${period} from empty caches`;
  $("timeline-heading").textContent = result.minutes.length === 60 ? "TTFT over the hour" : `TTFT over ${period}`;
  const trend = risingTtft(result);
  const causes = { transfer: "KV transfers to the decode GPUs are queueing on the inter-node link", load: "cache loads are queueing on the network and disks", wait: "prefill is saturated and requests queue" };
  $("trend-note").hidden = !trend;
  $("trend-note").textContent = trend ? `Still rising at the end (${secs(trend.from)} → ${secs(trend.to)}): ${causes[trend.cause]}.` : "";
  const gpus = prefillGpuCost(result), cache = result.costPerHour;
  $("cost").textContent = usdPerHour(hourlyCost(result)) + " / h";
  stats("cost-stats", [["Prefill GPUs", gpus == null ? "unpriced" : usdPerHour(gpus) + "/h"], ["Cache tiers", usdPerHour(cache) + "/h"]]);
  $("cost-detail").textContent = gpus == null
    ? `${$("gpu").selectedOptions[0].text} has no listed price, so GPUs are left out`
    : `${result.prefillGpus} prefill GPUs at on-demand prices; decode GPUs are not included`;
  showPrices(pricesAt(result.tokenPrices, result.utilization, settings.priceAt));
  const { tp, pp } = result.layout;
  const sim = config.sim;
  $("cluster-summary").textContent = `${config.models[sim.model].name} · ${PRECISION_LABELS[sim.precision]} · ${sim.nodes} nodes × ${sim.gpusPerNode} ${$("gpu").selectedOptions[0].text}`;
  $("layout").textContent = `${result.instances} prefill ${result.instances === 1 ? "instance" : "instances"} · ${tp} GPU${tp === 1 ? "" : "s"} per stage · ${pp} pipeline stage${pp === 1 ? "" : "s"} per instance`;
  kvPerInstanceGiB = result.gpuGiB / (settings.gpuBudget / 100);
  sliders.get("gpu-budget").refresh();
  plot(result);
  const entries = [
    ...result.order.filter(key => key === "p2p" || key in result.tiers)
      .map(key => [tierStyles[key].name, result.share[key], tierStyles[key].color, "white"]),
    ["Computed (new or uncached)", result.share.miss, "#a4acbb", "#132138"],
  ];
  $("bar").replaceChildren(...entries.map(([, share, color, ink]) => el("span", {
    style: 'width:' + share * 100 + '%;background:' + color + ';color:' + ink,
    text: share >= 0.12 ? pctf(share, 1) : "",
  })));
  $("legend").replaceChildren(...entries.map(([name, share, color]) =>
    el("li", {}, el("i", { class: "swatch", style: "background:" + color }), el("span", { text: name }),
      el("b", { text: share > 0 && share < 0.001 ? "<0.1%" : pctf(share, 1) }))));
  // One short line of facts; the design docs carry the explanations.
  const requests = result.minutes.reduce((sum, m) => sum + m.count, 0);
  const realized = Math.round(requests / result.minutes.length);
  const filled = result.firstEviction == null ? "GPU cache never full"
    : result.firstEviction < 0 ? "GPU cache full during warm-up" : `GPU cache full from minute ${Math.floor(result.firstEviction / 60) + 1}`;
  $("run-facts").replaceChildren(...[filled, `Prefill busy ${pctf(result.utilization, 0)}`,
    `${realized.toLocaleString("en-US")} of ${settings.rate} req/min served`,
    ...(sim.shared ? [`S3 requests ${usdPerHour(result.requestCostPerHour)}/h`] : [])].map(text => el("li", { text })));
  const cpuNote = cpuSizingNote(settings, result);
  $("cpu-note").hidden = !cpuNote;
  $("cpu-note").textContent = cpuNote ?? "";
  showProgress(null);
  $("status").textContent = "";
  $("results").setAttribute("aria-busy", "false");
}

function failed(message) {
  shown = null;
  lastResult = null;
  $("download-csv").disabled = true;
  $("results").classList.add("failed");
  $("comparison-status").textContent = "Comparison unavailable for this setup.";
  $("tier-bars").hidden = true;
  $("ttft").textContent = "—";
  $("cost").textContent = "—";
  plot(null);
  $("cpu-note").hidden = true;
  $("trend-note").hidden = true;
  showPrices(null);
  for (const id of ["bar", "legend", "run-facts", "ttft-stats", "cost-stats", "cost-detail", "ttft-detail", "ttft-change", "cost-change", "cluster-summary"]) $(id).replaceChildren();
  $("layout").textContent = message;
  showProgress(null);
  $("status").className = "error";
  $("status").textContent = "Simulation unavailable. " + message;
  $("results").setAttribute("aria-busy", "false");
}

async function start() {
  const response = await fetch("./data/s1.json");
  if (!response.ok) throw new Error("Could not load benchmark data.");
  config = defaultConfig(await response.json(), VARIANTS, GPUS, SOFTWARE, MODELS);
  const options = (id, pairs) => $(id).replaceChildren(...pairs.map(([value, label]) => new Option(label, value)));
  options("model", Object.entries(config.models).map(([id, m]) => [id, m.name]));
  options("precision", Object.entries(PRECISION_LABELS));
  options("gpu", GPUS.map(g => [g.id, g.name]));
  const counts = limit => Array.from({ length: limit }, (_, i) => [i + 1, String(i + 1)]);
  options("nodes", counts(64));
  options("gpus-per-node", counts(8));
  options("interconnect", VARIANTS.filter(v => v.family === "p2p").map(v => [v.id, v.name]));
  const priceHost = $("openrouter-input-prices");
  showPrices = simulated => { simulatedPrices = simulated; showTokenPrices(priceHost, market, $("model").value, simulated); };
  loadOpenRouterPrices().then(snapshot => { market = snapshot; showPrices(simulatedPrices); });
  const initial = decodeSettings(location.hash);
  writeSettings(initial);
  showScenario(readSettings());
  schedule();
}
start().catch(error => failed(error.message));
