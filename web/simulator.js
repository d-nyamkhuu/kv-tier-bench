import { defaultConfig } from "./config.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "./presets.js";
import { PRECISION_LABELS } from "./model.js";
import { loadOpenRouterPrices, showInputPrices } from "./openrouter.js";
import { tierStyles, timeline } from "./replay-chart.js";
import { minutesCsv } from "./replay.js";
import {
  advancedHash, advancedOverrides, applySettings, decodeSettings, DEFAULT_SETTINGS, encodeSettings, fitCluster, SCENARIOS, scenarioFor, usdPerHour,
} from "./sim-settings.js";
import { $, el, fmt, gibf, pctf, secs } from "./ui.js";

const controls = $("controls"), backing = $("backing");
const plot = timeline();
let market;
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

const cpuSizes = [0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512];
// Powers of two give small working sets room on the slider. The final stop uses the drive preset.
const diskSizes = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, null];
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
  pattern: ["pattern"], routing: ["routing"], cpu: ["cpu", "slider"], disk: ["disk"], diskSize: ["disk-size", "slider"],
  lookup: ["lookup-policy"], backing: ["backing"], start: ["start"], spread: ["spread"], prompts: ["prompts", "slider"], batch: ["batch", "slider"],
  admission: ["cache-admission"], prefix: ["prefix", "slider"], input: ["input", "slider"],
  output: ["output", "slider"], gpuBudget: ["gpu-budget", "slider"],
};
const LABELS = {
  model: "model", precision: "precision", gpu: "GPU", nodes: "nodes", gpusPerNode: "GPUs per node", interconnect: "inter-node link",
  rate: "request rate", turns: "turns per chat", pause: "think time", pattern: "arrival pattern", routing: "routing",
  cpu: "CPU cache", disk: "local disk", diskSize: "SSD capacity", lookup: "lookup order", backing: "object storage",
  start: "measurement start", spread: "length variation", prompts: "system prompts", batch: "prefill batch", admission: "saved chunks", prefix: "system prompt length",
  input: "message length", output: "reply length", gpuBudget: "GPU KV budget",
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

function showScenario(settings, changed = []) {
  const active = scenarioFor(settings);
  for (const button of $("scenario-buttons").children) button.setAttribute("aria-pressed", String(button.dataset.id === active?.id));
  const note = $("scenario-note");
  if (!active) {
    note.textContent = "Custom settings. Choose a scenario to start again from a known setup.";
    return;
  }
  note.textContent = active.description + (changed.length ? ` Changed: ${changed.map(key => LABELS[key]).join(", ")}.` : "");
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
  const settings = readSettings();
  if (ownCluster && !fitCluster(config, { ...settings, ...ownCluster })) {
    setCluster(ownCluster);
    showFit(`Returned to your cluster of ${ownCluster.nodes} node${ownCluster.nodes === 1 ? "" : "s"} × ${ownCluster.gpusPerNode} GPUs, which holds this model.`);
    ownCluster = null;
    return;
  }
  const fit = fitCluster(config, settings);
  if (!fit) { if (!ownCluster) showFit(); return; }
  if (fit.error) { showFit(fit.error); return; }
  ownCluster ??= { nodes: settings.nodes, gpusPerNode: settings.gpusPerNode };
  setCluster(fit);
  const gpu = config.gpus.find(g => g.id === settings.gpu);
  showFit(`Raised to ${fit.nodes} node${fit.nodes === 1 ? "" : "s"} × ${fit.gpusPerNode} GPUs, the smallest cluster that holds ${config.models[settings.model].name} on ${gpu.name}.`);
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
  showScenario(after, changed);
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

function runWorker(id) {
  worker = new Worker(new URL("./replay-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.id !== revision) return;
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
  $("status").textContent = "Simulating…";
  timer = setTimeout(() => runWorker(id), 120);
}

controls.addEventListener("input", event => {
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

function render(result) {
  const settings = readSettings();
  $("results").classList.remove("failed");
  if (baseline) {
    const latency = result.mean - baseline.mean, cost = result.costPerHour - baseline.costPerHour;
    $("ttft-change").textContent = changeText(latency, 0.0005, secs(Math.abs(latency)));
    $("cost-change").textContent = changeText(cost, 0.0005, usdPerHour(Math.abs(cost)) + "/h");
  } else $("ttft-change").textContent = $("cost-change").textContent = "";
  shown = { mean: result.mean, costPerHour: result.costPerHour };
  lastResult = result;
  $("download-csv").disabled = false;
  $("ttft").textContent = secs(result.mean);
  const { p50, p90, p99 } = result.ttft;
  $("ttft-detail").textContent = `p50 ${secs(p50)} · p90 ${secs(p90)} · p99 ${secs(p99)}` +
    (result.warmupMin ? ` · after a ${result.warmupMin}-minute warm-up` : " · from empty caches");
  const ops = result.s3Requests;
  const gpus = result.gpuCostPerHour, cache = result.costPerHour;
  const hourly = (gpus ?? 0) + cache;
  $("cost").textContent = usdPerHour(hourly) + " / h";
  const decodeGpus = Math.round(result.decodeGpus);
  const perThousand = result.requests && gpus != null ? hourly * (result.minutes.length / 60) / result.requests * 1000 : null;
  $("cost-detail").textContent = (gpus == null
    ? `${$("gpu").selectedOptions[0].text} has no listed price, so GPUs are left out · `
    : `GPUs ${usdPerHour(gpus)}/h (${result.prefillGpus} prefill + ${decodeGpus} decode) · `)
    + `cache tiers ${usdPerHour(cache)}/h` + (perThousand == null ? "" : ` · ${usdPerHour(perThousand)} per 1,000 requests`);
  const { tp, pp } = result.layout;
  const sim = config.sim;
  const explanation = $("read-explanation");
  const { shared, disk } = result.firstPayloadReadS;
  explanation.hidden = !(sim.disk && sim.nodes > pp && shared != null && (disk == null || shared < disk));
  const routing = sim.routing === "round-robin" ? "With round-robin routing, a request" : "A request";
  explanation.textContent = explanation.hidden ? "" : `Each node has a private SSD. ${routing} can reach a node missing the prefix and read it from S3 while its SSD still has room.`;
  $("cluster-summary").textContent = `${config.models[sim.model].name} · ${PRECISION_LABELS[sim.precision]} · ${sim.nodes} nodes × ${sim.gpusPerNode} ${$("gpu").selectedOptions[0].text}`;
  let peerNote = "Peer reads require CPU cache on another node outside this pipeline group.";
  if (result.order.includes("p2p")) {
    peerNote = sim.shared && !sim.peerFirst
      ? "If S3 and a peer both hold the missing chunk, S3 serves it."
      : "After a local miss, the request can fetch matching chunks from another node’s CPU cache.";
  }
  $("peer-policy").textContent = peerNote;
  $("layout").textContent = `${result.instances} prefill ${result.instances === 1 ? "instance" : "instances"} · ${tp} GPU${tp === 1 ? "" : "s"} per stage · ${pp} pipeline stage${pp === 1 ? "" : "s"} per instance`;
  kvPerInstanceGiB = result.gpuGiB / (settings.gpuBudget / 100);
  sliders.get("gpu-budget").refresh();
  $("gpu-budget-note").textContent = `Each model instance has ${gibf(+kvPerInstanceGiB.toPrecision(3))} of KV cache after weights and runtime reserve. Lower the budget to model memory used by other workloads.`;
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
    el("li", {}, el("i", { class: "swatch", style: "background:" + color }), name + " · " + (share > 0 && share < 0.001 ? "<0.1%" : pctf(share, 1)))));
  const model = config.models[sim.model];
  $("computed-note").textContent = "Computed tokens include each new message"
    + (sim.saveDecode ? "" : " and the previous reply, whose KV stays on the decode GPUs unless reply chunks are saved (Workload and memory details)")
    + "."
    + (model.read_fixed_bytes > 0 ? ` ${model.name} can resume only from a recurrent state saved at the end of an earlier prompt, so a new chat cannot reuse the shared prefix.` : "")
    + " The rest is history that no tier still held.";
  const eviction = result.firstEviction == null ? "GPU memory held every cached block throughout this run."
    : result.firstEviction < 0 ? "The GPU cache was already evicting blocks during the warm-up."
    : "The GPU cache first evicted a block in minute " + (Math.floor(result.firstEviction / 60) + 1) + ".";
  const requests = result.minutes.reduce((sum, m) => sum + m.count, 0);
  const realized = requests / result.minutes.length;
  const ramp = realized < settings.rate * 0.95
    ? ` Each follow-up waits for the previous reply plus the think time, so traffic builds up${result.warmupMin ? " even after the warm-up" : " from a cold start"}: ${result.minutes[0].count} requests in the first measured minute and ${result.minutes.at(-1).count} in the last.`
    : "";
  const busy = result.utilization;
  const load = busy < 0.3 ? "mostly idle, so requests rarely queue" : busy < 0.7 ? "moderately loaded" : busy < 0.95 ? "heavily loaded, so queueing grows quickly" : "saturated, so the queue grows without bound";
  $("pressure").textContent = `${eviction} The prefill GPUs were busy ${pctf(busy, 0)} of the time: ${load}. Requests spent ${secs(result.wait)} in the prefill queue on average. The replay served ${requests.toLocaleString("en-US")} requests, ${fmt(realized, 3)} per minute against the ${settings.rate}/min target.${ramp}`;
  const store = config.variants.find(v => v.id === sim.shared);
  $("s3-note").hidden = !store;
  $("s3-note").textContent = store
    ? `Before loading, each request asks ${store.short ?? store.name} about chunks it does not hold locally, including its new message, which was never saved. A check costs one round trip of ${secs(store.latency_s ?? 0)}, so object storage can raise TTFT even when most tokens come from GPU memory. Over the hour, requests sent ${ops.head.toLocaleString("en-US")} HEAD, ${ops.get.toLocaleString("en-US")} GET and ${ops.put.toLocaleString("en-US")} PUT requests, costing ${usdPerHour(result.requestCostPerHour)}/h.`
    : "";
  $("advanced-link").href = "./advanced.html" + advancedHash(advancedOverrides(settings, result.layout));
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
  $("read-explanation").hidden = true;
  $("s3-note").hidden = true;
  for (const id of ["bar", "legend", "computed-note", "pressure", "cost-detail", "ttft-change", "cost-change", "gpu-budget-note", "peer-policy", "cluster-summary"]) $(id).replaceChildren();
  $("layout").textContent = message;
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
  const updatePrices = () => showInputPrices(priceHost, market, $("model").value);
  controls.addEventListener("input", updatePrices);
  $("reset").addEventListener("click", updatePrices);
  $("scenario-buttons").addEventListener("click", updatePrices);
  loadOpenRouterPrices().then(snapshot => { market = snapshot; updatePrices(); });
  const initial = decodeSettings(location.hash);
  writeSettings(initial);
  showScenario(readSettings());
  schedule();
}
start().catch(error => failed(error.message));
