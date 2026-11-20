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

// USD per 1M tokens, as OpenRouter lists them: this setup's simulated price sheet (null while none is available)
// next to the median OpenRouter endpoint rates for the same model.
export function showTokenPrices(host, snapshot, modelId, simulated) {
  const model = snapshot?.models?.[modelId];
  const stats = model?.providers?.length ? providerStats(model.providers) : null;
  const cell = x => el("td", { text: x == null || !Number.isFinite(x) ? "–" : dollars.format(+x.toPrecision(3)) });
  const rows = [];
  if (simulated) rows.push(el("tr", {}, el("th", { scope: "row", text: "This setup" }), cell(simulated.input), cell(simulated.cachedInput), cell(simulated.output)));
  if (stats) rows.push(el("tr", {}, el("th", { scope: "row", text: `OpenRouter median (${stats.n} endpoints)` }), cell(stats.input?.median), cell(stats.cached?.median), cell(stats.output?.median)));
  if (!rows.length) {
    host.textContent = snapshot ? "No prices for this model: its GPU has no listed price and OpenRouter lists no endpoints." : "Waiting for the replay…";
    return;
  }
  host.replaceChildren(
    el("table", { class: "reader-table token-prices" },
      el("thead", {}, el("tr", {}, ...["", "Input", "Cached input", "Output"].map(t => el("th", { scope: "col", text: t })))),
      el("tbody", {}, ...rows)),
    el("p", { class: "price-source" }, "USD per 1M tokens",
      ...(stats ? [" · OpenRouter median across endpoints · ", el("a", { href: model.url, target: "_blank", rel: "noopener", text: "providers" }), " · " + snapshot.fetched] : [])),
  );
}
