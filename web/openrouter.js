import { el } from "./ui.js";

const median = (xs) => {
  const s = xs.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const k = s.length;
  return k % 2 ? s[(k - 1) / 2] : (s[k / 2 - 1] + s[k / 2]) / 2;
};
const range = (xs) => {
  const s = xs.filter((x) => x != null && Number.isFinite(x));
  return s.length ? { median: median(s), min: Math.min(...s), max: Math.max(...s), n: s.length } : null;
};

export function providerStats(providers) {
  return { n: providers.length, input: range(providers.map((p) => p.input)), cached: range(providers.map((p) => p.cached)), output: range(providers.map((p) => p.output)) };
}

export async function loadOpenRouterPrices() {
  try {
    const response = await fetch("./data/openrouter.json");
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

const dollars = new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 8,
});

// USD per 1M input and cached input tokens, as OpenRouter lists them: this setup's simulated price sheet (null while none is available)
// beside the median OpenRouter endpoint rate (with its range) and the ratio between the two.
const money = x => x == null || !Number.isFinite(x) ? "–" : dollars.format(+x.toPrecision(3));
const ratio = (ours, theirs) => {
  if (!(ours > 0) || !(theirs > 0)) return "–";
  const r = ours / theirs;
  return (r >= 10 ? r.toFixed(0) : r.toFixed(1)) + "×";
};
export function showTokenPrices(host, snapshot, modelId, simulated) {
  const model = snapshot?.models?.[modelId];
  const stats = model?.providers?.length ? providerStats(model.providers) : null;
  const kinds = [["input", "input", "Input"], ["cachedInput", "cached", "Cached input"]];
  if (!simulated && !stats) {
    host.textContent = snapshot ? "No prices: this GPU has no listed price and OpenRouter lists no endpoints for this model." : "Waiting for the replay…";
    return;
  }
  const row = (label, cells, cls = "") => el("tr", { class: cls }, el("th", { scope: "row", text: label }), ...cells);
  const rows = [];
  const pct = x => Math.round(x * 100) + "%";
  const label = simulated?.target ? `This setup at ${pct(simulated.target)} busy` : "This setup at its replay load";
  if (simulated) rows.push(row(label, kinds.map(([ours]) => el("td", { class: "price", text: money(simulated[ours]) })), "ours"));
  if (stats) {
    rows.push(row(`OpenRouter median`, kinds.map(([, theirs]) => {
      const f = stats[theirs];
      return el("td", {}, el("span", { class: "price", text: money(f?.median) }),
        f && f.n > 1 && f.min !== f.max ? el("small", { text: `${money(f.min)}–${money(f.max)}` }) : "");
    })));
    if (simulated) rows.push(row("This setup ÷ OpenRouter", kinds.map(([ours, theirs]) => el("td", { class: "ratio", text: ratio(simulated[ours], stats[theirs]?.median) })), "ratio-row"));
  }
  host.replaceChildren(
    el("table", { class: "price-table" },
      el("thead", {}, el("tr", {}, ...["", ...kinds.map(k => k[2])].map(t => el("th", { scope: "col", text: t })))),
      el("tbody", {}, ...rows)),
    el("p", { class: "price-source" }, "USD per 1M input tokens",
      simulated?.utilization > 0
        ? ` · replay prefill GPUs ${pct(simulated.utilization)} busy` + (simulated.target && simulated.utilization > simulated.target ? ` (above ${pct(simulated.target)} target)` : "")
        : "",
      ...(stats ? [` · OpenRouter: ${stats.n} endpoints, ranges under the medians · `, el("a", { href: model.url, target: "_blank", rel: "noopener", text: "providers" }), " · " + snapshot.fetched]
        : [" · OpenRouter lists no endpoints for this model"])),
  );
}
