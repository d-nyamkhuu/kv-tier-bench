import { checks } from "./checks.js";
import { SPEC_METHODS } from "./presets.js";
import { assumptionBadge, el, fmt } from "./ui.js";

function sourceEntry(kind, title, meta, item) {
  if (!item) return null;
  const badges = (item.assumptions ?? []).map(a => assumptionBadge(a));
  const links = (item.sources ?? []).map(s => el("li", {},
    s.url ? el("a", { href: s.url, target: "_blank", rel: "noopener", text: s.label }) : s.label,
    s.quote ? el("span", { class: "cite-quote", text: ` — “${s.quote}”` }) : ""));
  return el("div", { class: "source-card" },
    el("div", { class: "source-head" },
      el("span", { class: "source-kind", text: kind }),
      el("strong", { text: title }),
      meta ? el("span", { class: "source-meta", text: meta }) : null),
    item.note ? el("p", { class: "source-note", text: item.note }) : null,
    badges.length ? el("div", { class: "badge-row" }, ...badges) : null,
    links.length ? el("ul", { class: "cite-list" }, ...links) : null);
}

export function renderActiveSources(host, config) {
  if (!host || !config) return;
  const { sim } = config;
  const model = config.models[sim.model];
  const gpu = config.gpus.find(g => g.id === sim.gpu);
  const disk = sim.disk ? config.variants.find(v => v.id === sim.disk) : null;
  const shared = sim.shared ? config.variants.find(v => v.id === sim.shared) : null;
  const link = sim.p2p ? config.variants.find(v => v.id === sim.p2p) : null;
  const spec = sim.spec && sim.spec !== "none" ? SPEC_METHODS.find(s => s.id === sim.spec) : null;
  const modelMeta = [model?.arch, model?.released ? `Released ${model.released}` : null].filter(Boolean).join(" · ");
  const cards = [
    sourceEntry("Model", model?.name ?? sim.model, modelMeta, model),
    sourceEntry("GPU", gpu?.name ?? sim.gpu, gpu?.usd_per_hour != null ? `$${gpu.usd_per_hour.toFixed(2)}/GPU-h on-demand` : "No listed on-demand price", gpu),
    disk && sourceEntry("Local disk", disk.name, null, disk),
    shared && sourceEntry("Object storage", shared.name, null, shared),
    link && sourceEntry("Interconnect", link.name, null, link),
    spec && sourceEntry("Speculative drafter", spec.name, `${spec.draft} draft · ${spec.tau} accepted/step`, spec),
    sourceEntry("Engine overheads", "vLLM + LMCache software constants", "Calibrated on kv-tier-bench s1", config.software),
  ].filter(Boolean);
  host.replaceChildren(...cards);
}

export function renderValidationChecks(host, s1, config) {
  if (!host || !s1 || !config) return;
  const { rows } = checks(s1, config);
  const formatVal = (v, unit) => `${fmt(v, 3)}${unit === "×" ? "×" : " " + unit}`;
  const formatGap = gap => {
    const pct = gap * 100;
    return (pct >= 0 ? "+" : "−") + Math.abs(pct).toFixed(1) + "%";
  };
  const gapClass = gap => Math.abs(gap) <= 0.15 ? "gap-ok" : Math.abs(gap) <= 0.25 ? "gap-mid" : "gap-wide";
  const trs = rows.map(r => el("tr", {},
    el("td", {},
      el("div", { class: "check-title" },
        el("span", { text: r.what }),
        r.calibrated ? el("span", { class: "badge calibrated", text: "calibrated" }) : el("span", { class: "badge independent", text: "check" }),
        r.source?.url ? el("a", { class: "check-src", href: r.source.url, target: "_blank", rel: "noopener", text: "source" }) : null),
      r.note ? el("small", { class: "check-note", text: r.note }) : null),
    el("td", { class: "num", text: formatVal(r.published, r.unit) }),
    el("td", { class: "num", text: formatVal(r.predicted, r.unit) }),
    el("td", { class: `num ${gapClass(r.gap)}`, text: formatGap(r.gap) })));
  host.replaceChildren(
    el("p", { class: "track-note", text: "Compares the simulator's formulas against published vLLM, TensorRT-LLM and SGLang benchmarks (InferenceX, NVIDIA NIM, DeepSeek-V3 production, DFlash, EAGLE-3) and kv-tier-bench s1 cloud runs. Rows marked “calibrated” set a model parameter; rows marked “check” are out-of-sample checks." }),
    el("div", { class: "table-scroll" },
      el("table", { class: "validation-table" },
        el("thead", {}, el("tr", {},
          el("th", { scope: "col", text: "Benchmark & setup" }),
          el("th", { scope: "col", class: "num", text: "Measured" }),
          el("th", { scope: "col", class: "num", text: "Simulated" }),
          el("th", { scope: "col", class: "num", text: "Gap" }))),
        el("tbody", {}, ...trs))));
}
