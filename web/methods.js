// "Checks" and "Methods & data": the model against published results and s1, the editors for tier variants, GPU
// profiles and software constants, s1's medians with its tier fit (read-only: the reference), and config I/O.

import { checks, s1Residuals } from "./checks.js";
import { clear, exportConfig, FAMILIES, importConfig, save } from "./config.js";
import { curve, fitTier, GIB, interpLogLog, prefixBytes, variantBandwidth } from "./model.js";
import { FAMILY_INFO, SPEC_METHODS } from "./presets.js";
import { $, assumptionBadge, cssVar, el, fmt, numField, pctf, sourceList, usd } from "./ui.js";
import { MARKET_INSTANCES, providerStats } from "./market.js";

const SPEC_NAME = (id) => (id && id !== "none" ? `speculative decoding: ${SPEC_METHODS.find((x) => x.id === id)?.name ?? id}` : "no speculative decoding");

const TIERS = ["gpu", "cpu", "fs", "obj", "p2p"];
const fitsFor = (m) => Object.fromEntries(["fs", "obj", "p2p"].map((t) => [t, m.tiers[t] ? fitTier(m, t) : null]));

// which slot of the configurator a variant can fill
export const slotOf = (v) => (v.family === "p2p" ? "p2p" : v.family === "obj" ? "shared" : "disk");

let ctx; // { cfg(), changed(opts), s1, defaults, replaceCfg(c), storage }
let fitChart = null;

export function initMethods(c) {
  ctx = c;
  bindConfigButtons();
  // a canvas inside a closed <details> has no size: draw the fit chart when it opens
  $("fit-details").addEventListener("toggle", () => { if ($("fit-details").open) drawFitChart(); });
}

export function renderMethods() {
  renderChecks();
  renderMarket();
  renderVariants();
  renderGpus();
  renderSoftware();
  renderMeasured();
}

// ------------------------------------------------------------------------------------------------ provider prices

let marketWorker = null, marketId = 0, marketRows = [];

function renderMarket() {
  const data = ctx.market;
  if (!data) return;
  $("market").hidden = false;
  marketHint();
  $("market-run").onclick = runMarket;
  drawMarket();
}

function marketHint() {
  const data = ctx.market, c = ctx.cfg();
  const g = c.gpus.find((x) => x.id === c.sim.gpu);
  $("market-hint").replaceChildren(
    `Each model's $ per 1M tokens from the simulator beside every provider's price on OpenRouter (snapshot of ${data.fetched}). `,
    `Each model runs on your settings (${g?.usd_per_hour != null ? g.name : "B200, as your GPU has no price"}, ${c.sim.precision && c.sim.precision !== "released" ? `served in ${c.sim.precision.toUpperCase()}` : "served as released"}, tiers, serving, workload shape) with ${MARKET_INSTANCES} instances of the cheapest of three sizes (the fewest GPUs that hold it, twice, four times), `,
    "loaded until the prefill GPUs are busy 60% of the time, with decode GPUs sized to that load. A provider runs near capacity and pools decode across clusters; an idle or rounded-up cluster would overstate the cost. ",
    `Speculative decoding and expert parallelism follow your serving settings (${SPEC_NAME(c.sim.spec)}; experts ${c.sim.decodeEp ? `over ${c.sim.decodeEp} GPUs` : "in the TP instance"}). `,
    `Decode batches concurrent requests using each model's p50 speed on OpenRouter (as of ${Object.values(data.models).find((m) => m.speed)?.speed.as_of ?? "–"}), as its throughput target. With no target, arrival times and KV memory determine batch size.`);
}

function runMarket() {
  const data = ctx.market;
  marketWorker ??= new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  marketId += 1;
  marketRows = [];
  marketHint();
  const models = Object.keys(data.models).filter((m) => ctx.cfg().models[m]);
  $("market-status").textContent = `Simulating 0 of ${models.length} models…`;
  $("market-run").disabled = true;
  marketWorker.onmessage = ({ data: msg }) => {
    if (msg.id !== marketId) return;
    if (msg.kind === "market-row") { marketRows.push(msg.row); $("market-status").textContent = `Simulating ${msg.done} of ${msg.total} models…`; drawMarket(); }
    if (msg.kind === "market-done") { $("market-status").textContent = ""; $("market-run").disabled = false; $("market-run").textContent = "Run again with the current settings"; }
  };
  const speeds = Object.fromEntries(models.map((m) => [m, data.models[m].speed?.p50_tok_s ?? null]));
  marketWorker.postMessage({ kind: "market", id: marketId, config: JSON.parse(JSON.stringify(ctx.cfg())), models, speeds });
}

// the provider rows whose details are open, kept across redraws (one per finished model during a run)
const marketOpen = new Set();

function drawMarket() {
  const data = ctx.market, c = ctx.cfg();
  const ratio = (sim, or) => (sim > 0 && or > 0 ? sim / or : null);
  const ratioCell = (label, x) => el("td", { "data-label": label, text: x == null ? "–" : `${fmt(x, 2)}×`, class: x != null && (x > 2 || x < 0.5) ? "bad" : null });
  const head = ["Model", "Simulator, $ per 1M in / cached / out", "Providers' median, $ per 1M in / cached / out", "Simulator ÷ median, input", "Simulator ÷ median, output"];
  const focusId = document.activeElement?.closest?.("#market-table") ? document.activeElement.id : null;
  const rows = Object.entries(data.models).filter(([id]) => c.models[id]).flatMap(([id, or]) => {
    const st = providerStats(or.providers);
    const r = marketRows.find((x) => x.model === id);
    const ok = r && !r.error;
    const trio = (a, b, d) => [a, b, d].map((x) => usd(x)).join(" / ");
    const setup = !r ? "not run yet" : r.error ? r.error : `${r.instances} × ${r.tp * r.pp} ${r.gpuName}${r.pp > 1 ? ` (${r.pp} nodes each)` : ""}, ${fmt(r.requestsPerMin, 3)} req/min, ${fmt(r.decodeGpus, 2)} decode GPUs` +
      `${r.decodeEp ? `, experts over ${r.decodeEp}` : ""}${r.specSpeedup != null ? (r.specSpeedup <= 1.0005 ? ", drafter off (slower at this batch)" : `, drafter ${fmt(r.specSpeedup, 2)}×`) : ""}${r.specUnavailable ? `; no ${r.specUnavailable.split(",")[0]} drafter` : ""}`;
    const p50 = or.speed?.p50_tok_s;
    const short = ok && r.belowSpeed > 0.005;
    const speed = ok && r.tpot ? `${fmt(1 / r.tpot, 3)} tok/s at a batch of ${fmt(r.decodeBatch, 2)}; OpenRouter p50 ${p50 ? `${fmt(p50, 3)} tok/s` : "none"}` : p50 ? `OpenRouter p50 ${fmt(p50, 3)} tok/s` : "–";
    const open = marketOpen.has(id);
    const detailId = `mk-d-${id}`;
    const toggle = el("button", { type: "button", class: "row-toggle", id: `mk-t-${id}`, "aria-expanded": String(open), "aria-controls": detailId, text: "Details",
      onclick: () => { if (marketOpen.has(id)) marketOpen.delete(id); else marketOpen.add(id); drawMarket(); } });
    const main = el("tr", { class: open ? "has-detail" : null },
      el("td", { class: "span" }, el("div", { class: "model-cell" },
        el("a", { href: or.url, target: "_blank", rel: "noopener", text: c.models[id].name ?? id }),
        short ? el("span", { class: "badge bad", text: "below target speed", title: `${pctf(r.belowSpeed, 0)} of requests fall below ${fmt(p50, 3)} tok/s even at batch 1 (no speculative decoding modeled)` }) : null,
        toggle)),
      el("td", { "data-label": "Simulator, in / cached / out", text: ok ? trio(r.price.input, r.price.cachedInput, r.price.output) : "–" }),
      el("td", { "data-label": "Providers' median, in / cached / out", text: trio(st.input?.median, st.cached?.median, st.output?.median) }),
      ratioCell("÷ median, input", ok ? ratio(r.price.input, st.input?.median) : null),
      short ? el("td", { "data-label": "÷ median, output", text: "–" }, el("div", { class: "note", text: "decode misses the speed even at batch 1" }))
        : ratioCell("÷ median, output", ok ? ratio(r.price.output, st.output?.median) : null));
    if (!open) return [main];
    const facts = [
      ["Reference deployment", setup],
      ["Decode speed per request", speed],
      ["Providers' output range", st.output ? `${usd(st.output.min)}–${usd(st.output.max)} per 1M over ${st.n} endpoints` : `${st.n} endpoints`],
      short ? ["Below target", `${pctf(r.belowSpeed, 0)} of requests stay below ${fmt(p50, 3)} tok/s even one at a time, so the output price is what batch 1 costs`] : null,
    ].filter(Boolean);
    return [main, el("tr", { class: "detail", id: detailId }, el("td", { colspan: head.length, class: "span" },
      el("dl", { class: "facts" }, facts.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })]))))];
  });
  $("market-table").replaceChildren(el("thead", {}, el("tr", {}, head.map((t) => el("th", { text: t })))), el("tbody", {}, rows));
  if (focusId) $(focusId)?.focus({ preventScroll: true });
  $("market-note").textContent = "Prices are not costs: they carry margins and promotions, and many providers serve FP4 or FP8 quantizations " +
    "(listed per endpoint in data/openrouter.json); set Serving precision above to compare like with like. Cached-input prices are a pricing choice " +
    "(often a tenth of the input price). The simulator pays Nebius on-demand GPU prices; large providers pay less, and use speculative decoding (several tokens per step), which is not modeled: it is how the fastest endpoints reach their speed at a larger batch.";
}

// ------------------------------------------------------------------------------------------------ checks

function renderChecks() {
  const { rows } = checks(ctx.s1, ctx.cfg());
  const cell = (r, x) => `${fmt(x, 3)} ${r.unit}`;
  const gap = (r) => (Math.abs(r.gap) < 0.005 ? "0%" : `${r.gap > 0 ? "+" : "−"}${Math.abs(r.gap * 100).toFixed(0)}%`);
  const body = (group, title) => [
    el("tr", { class: "group" }, el("th", { colspan: 5, text: title })),
    ...rows.filter((r) => r.group === group).map((r) => el("tr", {},
      el("td", { class: "span" }, el("div", { text: r.what }), r.calibrated ? el("span", { class: "badge calibrated", text: "calibrated" }) : el("span", { class: "badge check", text: "check" }),
        el("div", { class: "note", text: r.note })),
      el("td", { "data-label": "Measured", text: cell(r, r.published) }),
      el("td", { "data-label": "This model", text: cell(r, r.predicted) }),
      el("td", { "data-label": "Gap", class: !r.calibrated && Math.abs(r.gap) > 0.2 ? "bad" : null, text: gap(r) }),
      el("td", { class: "src span" }, el("a", { href: r.source.url, target: "_blank", rel: "noopener", text: r.source.label })))),
  ];
  $("checks-table").replaceChildren(
    el("thead", {}, el("tr", {}, ["What", "Measured", "This model", "Gap", "Source"].map((t) => el("th", { text: t })))),
    el("tbody", {}, ...body("published", "Published vLLM benchmarks"), ...body("s1", "kv-tier-bench s1: RTX PRO 6000 on Nebius, the GPU, CPU memory and three tiers s1 measured")),
  );
  // every s1 point, both models
  const pctCell = (x) => { if (!x) return el("td", { text: "–" }); const g = x.model / x.measured - 1; return el("td", { class: Math.abs(g) > 0.15 ? "bad" : null, text: `${g >= 0 ? "+" : "−"}${Math.abs(g * 100).toFixed(0)}%`, title: `s1 ${fmt(x.measured)} s, model ${fmt(x.model)} s` }); };
  $("residuals-table").replaceChildren(
    el("thead", {}, el("tr", {}, ["model", "tokens", "cold prefill", "cached on GPU", "cached in CPU memory"].map((t) => el("th", { text: t })))),
    el("tbody", {}, s1Residuals(ctx.s1, ctx.cfg()).flatMap(({ name, rows: rr }) => rr.map((r, i) => el("tr", {},
      el("td", { text: i === 0 ? name : "" }), el("td", { text: r.tokens.toLocaleString() }), pctCell(r.cold), pctCell(r.gpu), pctCell(r.cpu))))),
  );
}

// ------------------------------------------------------------------------------------------------ tier variants

function removeVariant(v) {
  const c = ctx.cfg();
  c.variants.splice(c.variants.indexOf(v), 1);
  for (const k of ["disk", "p2p", "shared"]) {
    if (c.sim[k] === v.id) c.sim[k] = null;
    c.sim.candidates[k] = (c.sim.candidates[k] ?? []).filter((id) => id !== v.id);
  }
  if (c.sim.transfer === v.id) c.sim.transfer = null;
}

function setCandidate(v, on) {
  const cand = ctx.cfg().sim.candidates;
  for (const k of ["disk", "p2p", "shared"]) cand[k] = (cand[k] ?? []).filter((id) => id !== v.id);
  if (on) cand[slotOf(v)].push(v.id);
}

function variantCard(v, fits) {
  const c = ctx.cfg();
  const isA = (k) => v.assumptions?.includes(k) || v.edited?.includes(k);
  // an edited value is no longer what its source says: mark it like an assumption
  const edited = (k) => { if (!v.edited?.includes(k)) v.edited = [...(v.edited ?? []), k]; };
  const rerender = () => ctx.changed({ methods: true });
  const fields = [];
  const derived = [];
  if (v.link_gbit_s != null) {
    fields.push(numField("Link (Gbit/s)", v, "link_gbit_s", { assumption: isA("link_gbit_s"), edited, onchange: rerender }),
      numField("Transport efficiency (0–1)", v, "efficiency", { assumption: isA("efficiency"), edited, onchange: rerender }),
      numField("Software cap (GiB/s)", v, "cap_gib_s", { assumption: isA("cap_gib_s"), placeholder: "none", edited, onchange: rerender }));
    derived.push(`Effective ${fmt(variantBandwidth(v))} GiB/s.`);
  } else {
    fields.push(numField("Peak bandwidth (GiB/s)", v, "bw_gib_s", { assumption: isA("bw_gib_s"), edited, onchange: rerender }),
      numField("Efficiency (0–1)", v, "efficiency", { assumption: isA("efficiency"), placeholder: "1", edited, onchange: rerender }));
    derived.push(`Effective ${fmt(variantBandwidth(v))} GiB/s.`);
  }
  fields.push(numField("Latency (ms)", v, "latency_s", { scale: 1000, assumption: isA("latency_s"), edited, onchange: rerender }));
  if (v.family !== "p2p") { // a p2p tier is the peers' CPU memory: no capacity or storage price of its own
    if (v.family === "obj" && v.usd_per_1k_put != null) fields.push(
      numField("$ / 1,000 PUT", v, "usd_per_1k_put", { edited, onchange: rerender }),
      numField("$ / 1,000 GET or HEAD", v, "usd_per_1k_get", { edited, onchange: rerender }));
    fields.push(
      numField("Capacity (GiB, empty = unbounded)", v, "capacity_gib", { assumption: isA("capacity_gib"), placeholder: "unbounded", edited, onchange: rerender }),
      numField("$ / GiB-month", v, "usd_per_gib_month", { assumption: isA("usd_per_gib_month"), placeholder: "no price", edited, onchange: rerender }),
      numField("$ / GiB written", v, "usd_per_gib_written", { assumption: isA("usd_per_gib_written"), placeholder: "0", edited, onchange: rerender }),
      numField("$ / GiB read", v, "usd_per_gib_read", { assumption: isA("usd_per_gib_read"), placeholder: "0", edited, onchange: rerender }),
      el("label", { class: "field" }, el("span", { text: "Billed on" }),
        el("select", { onchange: (ev) => { v.bill = ev.target.value; rerender(); } },
          el("option", { value: "stored", text: "bytes stored", selected: v.bill !== "provisioned" }),
          el("option", { value: "provisioned", text: "provisioned capacity", selected: v.bill === "provisioned" }))),
    );
  }
  fields.push(numField(v.family === "p2p" ? "$ / hour per node" : "$ / hour", v, "usd_per_hour", { assumption: isA("usd_per_hour"), placeholder: "0", edited, onchange: rerender }));
  const slot = slotOf(v);
  return el("div", { class: "card" },
    el("header", {},
      el("input", { type: "text", value: v.name, "aria-label": "Variant name", onchange: (ev) => { v.name = ev.target.value; rerender(); } }),
      v.assumptions?.length ? assumptionBadge(v.assumptions.join(", ")) : null,
      v.edited?.length ? el("span", { class: "badge assumption", text: `edited: ${v.edited.join(", ")}` }) : null),
    el("label", { class: "field check" },
      el("input", { type: "checkbox", checked: !!c.sim.candidates[slot]?.includes(v.id), onchange: (ev) => { setCandidate(v, ev.target.checked); ctx.changed({ methods: true }); } }),
      el("span", { text: `Try it in the trade-off map (${slot === "disk" ? "local disk" : slot === "p2p" ? "peer nodes" : "shared storage"})` })),
    el("div", { class: "grid" }, fields),
    derived.length ? el("p", { class: "derived", text: derived.join(" ") }) : null,
    v.note ? el("p", { class: "note", text: v.note }) : null,
    sourceList(v),
    el("div", { class: "row" },
      el("button", { type: "button", text: "Duplicate", onclick: () => {
        const copy = { ...structuredClone(v), id: `${v.id}-copy-${Date.now().toString(36)}`, name: `${v.name} (copy)`, short: `${v.short ?? v.name} (copy)`, custom: true };
        c.variants.splice(c.variants.indexOf(v) + 1, 0, copy);
        rerender();
      } }),
      el("button", { type: "button", text: "Delete", onclick: () => { removeVariant(v); rerender(); } })),
  );
}

function renderVariants() {
  const c = ctx.cfg();
  const fits = {};
  $("variant-families").replaceChildren(...FAMILIES.map((fam) => el("div", {},
    el("h4", { text: FAMILY_INFO[fam].title }),
    el("div", { class: "cards" }, c.variants.filter((v) => v.family === fam).map((v) => variantCard(v, fits))),
    el("div", { class: "row" }, el("button", { type: "button", text: `Add a custom ${fam} variant`, onclick: () => {
      c.variants.push({
        id: `custom-${fam}-${Date.now().toString(36)}`, family: fam, name: `Custom ${fam}`, short: `Custom ${fam}`, custom: true,
        bw_gib_s: 1, latency_s: 0.01, capacity_gib: null, usd_per_gib_written: 0, usd_per_gib_read: 0, shared: fam === "obj", bill: fam === "fs" ? "provisioned" : "stored",
        usd_per_gib_month: fam === "p2p" ? null : 0, usd_per_hour: 0, sources: [], assumptions: ["bw_gib_s", "latency_s", "usd_per_gib_month"], note: "Added in the editor.",
      });
      ctx.changed({ methods: true });
    } })),
  )));
}

// ------------------------------------------------------------------------------------------------ GPU profiles

function renderGpus() {
  const c = ctx.cfg();
  $("gpu-cards").replaceChildren(...c.gpus.map((g) => {
    const isA = (k) => g.assumptions?.includes(k) || g.edited?.includes(k);
    const edited = (k) => { if (!g.edited?.includes(k)) g.edited = [...(g.edited ?? []), k]; };
    const on = () => ctx.changed({ methods: true });
    const f = (label, key, extra = {}) => numField(label, g, key, { assumption: isA(key), edited, onchange: on, ...extra });
    return el("div", { class: "card" },
      el("header", {}, el("strong", { text: g.name }), g.assumptions?.length ? assumptionBadge(g.assumptions.join(", ")) : null,
        g.edited?.length ? el("span", { class: "badge assumption", text: `edited: ${g.edited.join(", ")}` }) : null),
      el("div", { class: "grid" },
        f("Dense BF16 (TFLOPS)", "tflops_bf16"), f("Dense FP8 (TFLOPS)", "tflops_fp8"), f("Dense FP4 (TFLOPS)", "tflops_fp4", { placeholder: "no FP4 cores" }), f("Prefill share of peak (0–1)", "prefill_mfu"), f("Usable memory (GiB)", "mem_gib"),
        f("Memory bandwidth (TB/s)", "mem_bw_tb_s"), f("Decode bandwidth use (0–1)", "mbu"), f("CPU → GPU (GiB/s)", "host_gib_s"),
        f("$ / GPU-hour", "usd_per_hour", { placeholder: "no price" }), f("NVLink (GB/s, diagram only)", "nvlink_gb_s", { placeholder: "none" })),
      g.note ? el("p", { class: "note", text: g.note }) : null,
      sourceList(g));
  }));
}

// ------------------------------------------------------------------------------------------------ software constants

function renderSoftware() {
  const sw = ctx.cfg().software;
  const isA = (k) => sw.assumptions?.includes(k) || sw.edited?.includes(k);
  const edited = (k) => { if (!sw.edited?.includes(k)) sw.edited = [...(sw.edited ?? []), k]; };
  const f = (label, key, extra = {}) => numField(label, sw, key, { assumption: isA(key), edited, onchange: () => ctx.changed({ methods: true }), ...extra });
  $("software-card").replaceChildren(el("div", { class: "card" },
    el("header", {}, el("strong", { text: "vLLM software constants" }), el("span", { class: "badge calibrated", text: "calibrated" }),
      sw.edited?.length ? el("span", { class: "badge assumption", text: `edited: ${sw.edited.join(", ")}` }) : null),
    el("div", { class: "grid" },
      f("Overhead per request (ms)", "request_overhead_s", { scale: 1000 }), f("Overhead per prompt token (µs)", "per_token_overhead_s", { scale: 1e6 }),
      f("Attention efficiency vs matmul", "attn_eff_ratio"), f("CPU-tier load latency (ms)", "cpu_load_latency_s", { scale: 1000 }),
      f("vLLM GPU memory utilization", "gpu_memory_utilization"), f("Activation reserve per GPU (GiB)", "activation_gib_per_gpu")),
    el("p", { class: "note", text: "No source publishes these. They are fitted on s1 (both models, 896–131k tokens); the residual table under the checks shows how well." }),
    sourceList(sw)));
}

// ------------------------------------------------------------------------------------------------ s1 reference

// the s1 model shown: the selected one when s1 measured it, else s1's standard-attention model
const s1Name = () => (ctx.cfg().sim.model in ctx.s1.models ? ctx.cfg().sim.model : "qwen3-vl-32b" in ctx.s1.models ? "qwen3-vl-32b" : Object.keys(ctx.s1.models)[0]);

function renderMeasured() {
  const name = s1Name();
  const m = ctx.s1.models[name];
  if (!m) return;
  $("measured-model").textContent = `${name}: KV ${m.kv_bytes_per_token / 1024} KiB per token at TP 1, plus ${(m.read_fixed_bytes / 2 ** 20).toFixed(0)} MiB of state per prefix; ${m.block_size}-token blocks.`;
  const cols = [...TIERS.filter((t) => m.tiers[t]), "recompute"];
  const rowsOf = (t) => (t === "recompute" ? m.recompute : m.tiers[t]);
  const lengths = [...new Set(cols.flatMap((t) => rowsOf(t).map((r) => r.tokens)))].sort((a, b) => a - b);
  $("medians-table").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", { text: "tokens" }), cols.map((t) => el("th", { text: t === "recompute" ? "cold (recompute)" : t })))),
    el("tbody", {}, lengths.map((n) => el("tr", {}, el("td", { text: n.toLocaleString() }), cols.map((t) => {
      const row = rowsOf(t).find((r) => r.tokens === n);
      return el("td", { text: row ? fmt(row.ttft_s) : "–" });
    })))),
  );
  const fits = fitsFor(m);
  const fl = FAMILIES.filter((t) => fits[t]);
  const fitLengths = [...new Set(fl.flatMap((t) => fits[t].residuals.map((r) => r.tokens)))].sort((a, b) => a - b);
  $("fit-table").replaceChildren(
    el("thead", {}, el("tr", {}, ["tier", "BW (GiB/s)", "L (ms)", "max |resid.|", ...fitLengths.map((n) => n.toLocaleString())].map((t) => el("th", { text: t })))),
    el("tbody", {}, fl.map((t) => {
      const f = fits[t];
      return el("tr", {}, el("td", { text: t }), el("td", { text: fmt(f.bw_gib_s) }), el("td", { text: fmt(f.latency_s * 1000) }),
        el("td", { text: pctf(f.maxAbsRel, 1), class: f.maxAbsRel > 0.1 ? "bad" : null }),
        fitLengths.map((n) => {
          const r = f.residuals.find((x) => x.tokens === n);
          return el("td", { text: r ? `${r.rel >= 0 ? "+" : ""}${(r.rel * 100).toFixed(1)}%` : "–", class: r && Math.abs(r.rel) > 0.1 ? "bad" : null,
            title: r ? `measured ${fmt(r.measured)} s, fit ${fmt(r.model)} s` : null });
        }));
    })),
  );
  if ($("fit-details").open) drawFitChart();
}

export function drawFitChart() {
  if (!window.Chart || !$("fit-details").open) return;
  const model = ctx.s1.models[s1Name()];
  if (!model) return;
  const fits = fitsFor(model);
  fitChart?.destroy();
  const colors = { cpu: "--tier-cpu", fs: "--tier-disk", obj: "--tier-shared", p2p: "--tier-p2p" };
  const datasets = [];
  const cpu = curve(model.tiers.cpu);
  for (const t of ["cpu", "fs", "obj", "p2p"]) {
    const rows = model.tiers[t];
    if (!rows) continue;
    const color = cssVar(colors[t]);
    datasets.push({ label: `${t} measured`, data: rows.map((r) => ({ x: r.tokens, y: r.ttft_s })), backgroundColor: color, borderColor: color, pointRadius: 4, showLine: false });
    const f = fits[t];
    if (f) datasets.push({
      label: `${t} model`, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0, showLine: true, borderDash: [5, 3],
      data: rows.map((r) => r.tokens).map((x) => ({ x, y: interpLogLog(cpu, x) + f.latency_s + prefixBytes(model, x) / (f.bw_gib_s * GIB) })),
    });
  }
  fitChart = new window.Chart($("fit-chart"), {
    type: "scatter", data: { datasets },
    options: {
      maintainAspectRatio: false, animation: false,
      scales: {
        x: { type: "logarithmic", title: { display: true, text: "prefix tokens" }, grid: { color: cssVar("--grid") } },
        y: { type: "logarithmic", title: { display: true, text: "TTFT (s)" }, grid: { color: cssVar("--grid") } },
      },
      plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${x.parsed.x.toLocaleString()} tokens, ${fmt(x.parsed.y)} s` } } },
    },
  });
}

// ------------------------------------------------------------------------------------------------ config I/O

function bindConfigButtons() {
  $("export-btn").addEventListener("click", () => {
    const blob = new Blob([exportConfig(ctx.cfg())], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: "kvtb-sim-config.json" });
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  });
  $("import-file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    try {
      ctx.replaceCfg(importConfig(await f.text(), ctx.defaults));
      $("config-error").textContent = "";
      save(ctx.storage, ctx.cfg());
      $("config-status").textContent = `Imported ${f.name}.`;
    } catch (e) {
      $("config-error").textContent = `Not imported:\n${e.message}`;
    }
    ev.target.value = "";
  });
  $("reset-btn").addEventListener("click", () => {
    if (!confirm("Reset every value to the defaults? Your edits in this browser are lost.")) return;
    clear(ctx.storage);
    ctx.replaceCfg(structuredClone(ctx.defaults), { reset: true });
    $("config-status").textContent = "Reset to defaults.";
  });
}
