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

export function showInputPrices(host, snapshot, modelId) {
  const model = snapshot?.models?.[modelId];
  if (!model?.providers?.length) {
    host.textContent = snapshot ? "No OpenRouter rates listed for this model in the snapshot." : "OpenRouter pricing is unavailable.";
    return;
  }

  const stats = providerStats(model.providers);
  const rate = (label, field) => el("span", {}, label + " ",
    el("strong", { text: field ? dollars.format(field.median) : "Unlisted" }));
  host.replaceChildren(
    el("p", { class: "input-rates" }, rate("Regular input", stats.input), rate("Cached input", stats.cached)),
    el("p", { class: "price-source" }, "USD per 1M input tokens · Median endpoint rates across precisions · ",
      el("a", { href: model.url, target: "_blank", rel: "noopener", text: "OpenRouter providers" }), " · " + snapshot.fetched),
  );
}
