// Small DOM helpers shared by the page's modules.

export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k in e && k !== "list" && !k.startsWith("aria") && !k.startsWith("data")) e[k] = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid);
  return e;
}

export const fmt = (x, d = 3) => (x == null || !Number.isFinite(x) ? "–" : Number(x.toPrecision(d)).toLocaleString("en-US", { maximumFractionDigits: 6 }));
export const pctf = (x, d = 0) => (x == null || !Number.isFinite(x) ? "–" : `${(x * 100).toFixed(d)}%`);
export const parseNum = (s) => (String(s).trim() === "" ? null : Number(s));
export const usd = (x) => (x == null || !Number.isFinite(x) ? "–" : x >= 100 ? `$${Math.round(x).toLocaleString("en-US")}` : x >= 1 ? `$${x.toFixed(2)}` : `$${x.toPrecision(2)}`);
export const secs = (x) => (x == null || !Number.isFinite(x) ? "–" : x >= 100 ? `${Math.round(x).toLocaleString("en-US")} s` : x >= 1 ? `${x.toFixed(2)} s` : `${Math.round(x * 1000)} ms`);
export const gibf = (x) => (x == null ? "unbounded" : x >= 1024 ? `${fmt(x / 1024, 3)} TiB` : `${fmt(x, 3)} GiB`);

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const assumptionBadge = (what) => el("span", { class: "badge assumption", text: what ? `assumption: ${what}` : "assumption" });

// A labelled number input bound to obj[key]; `scale` shows e.g. % for a share.
export function numField(label, obj, key, { scale = 1, step = "any", onchange, assumption = false, placeholder, edited, id } = {}) {
  const v = obj[key];
  const input = el("input", {
    type: "number", step, placeholder, id, inputmode: "decimal",
    value: v == null ? "" : +(v * scale).toPrecision(8),
    onchange: (ev) => {
      const n = parseNum(ev.target.value);
      if (n != null && !Number.isFinite(n)) { ev.target.value = v == null ? "" : v * scale; return; }
      obj[key] = n == null ? null : n / scale;
      if (edited) edited(key);
      onchange?.();
    },
  });
  return el("label", { class: `field${assumption ? " assumption-field" : ""}` }, el("span", { text: label }), input);
}

export function sourceList(x) {
  if (!x.sources?.length) return el("p", { class: "note" }, assumptionBadge("no source"), " Every value here is an assumption.");
  return el("ul", { class: "sources" }, x.sources.map((s) => el("li", {},
    s.url ? el("a", { href: s.url, text: s.label, target: "_blank", rel: "noopener" }) : s.label,
    s.quote ? `: "${s.quote}"` : null)));
}
