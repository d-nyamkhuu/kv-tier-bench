// The cluster as an SVG diagram: users -> router -> nodes (GPUs, CPU memory, disk) -> peers, shared storage, decode.
// Pure: takes the setup and the latest shares, returns SVG markup. Elements with data-target focus that control.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const gib = (x) => (x == null ? "unbounded" : x >= 1024 ? `${(x / 1024).toFixed(1)} TiB` : `${Math.round(x)} GiB`);

// an animated flow whose width follows the share of prompt tokens it carries
function flow(d, share, cls, title) {
  const w = 1.5 + 7 * Math.min(1, share ?? 0);
  const speed = Math.max(0.6, 3 - 2.5 * (share ?? 0));
  return `<path class="dg-edge ${cls}" d="${d}" stroke-width="${w.toFixed(1)}"/>` +
    (share > 0.005 ? `<path class="dg-flow ${cls}" d="${d}" stroke-width="${Math.max(1.5, w - 2).toFixed(1)}" style="animation-duration:${speed.toFixed(2)}s"><title>${esc(title)}</title></path>` : "");
}

function box(x, y, w, h, cls, target, title, inner) {
  return `<g class="dg-box ${cls}"${target ? ` data-target="${target}" tabindex="0" role="button"` : ""}><title>${esc(title)}</title>` +
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/>${inner}</g>`;
}

const text = (x, y, s, cls = "", anchor = "start") => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;

function node(x, y, w, s, idx) {
  const G = s.gpusPerNode, tp = s.tp;
  const pad = 12, chipGap = 6;
  const chipW = Math.min(46, (w - 2 * pad - (G - 1) * chipGap) / G);
  const chipsW = G * chipW + (G - 1) * chipGap;
  const cx = x + (w - chipsW) / 2;
  const pp = s.pp ?? 1;
  let out = text(x + pad, y + 20, pp > 1 ? `Node ${idx} · stage ${((idx - 1) % pp) + 1}/${pp}` : `Node ${idx}`, "dg-title") + text(x + w - pad, y + 20, `${G}× ${s.gpuShort}`, "dg-sub", "end");
  // GPU chips, grouped into TP instances joined by NVLink (or PCIe)
  for (let g = 0; g < G; g++) {
    const gx = cx + g * (chipW + chipGap);
    out += `<g class="dg-chip" data-target="gpu" tabindex="0" role="button"><title>GPU ${g + 1}: ${esc(s.gpuName)}. KV cache ${gib(s.kvGiB)} per ${tp > 1 ? `TP-${tp} instance` : "GPU"}. Served ${pct(s.share.gpu)} of prompt tokens.</title>` +
      `<rect x="${gx}" y="${y + 32}" width="${chipW}" height="34" rx="5"/><rect class="dg-fill" x="${gx + 3}" y="${y + 32 + 31 - 28 * Math.min(1, s.share.gpu ?? 0)}" width="${chipW - 6}" height="${28 * Math.min(1, s.share.gpu ?? 0)}" rx="3"/></g>`;
  }
  for (let t = 0; t < G / tp; t++) {
    if (tp === 1) break;
    const x0 = cx + t * tp * (chipW + chipGap), x1 = x0 + tp * chipW + (tp - 1) * chipGap;
    out += `<g class="dg-link"><title>${esc(`${s.tpLinkTitle}: tensor parallelism across ${tp} GPUs`)}</title><line x1="${x0 + 2}" y1="${y + 72}" x2="${x1 - 2}" y2="${y + 72}"/></g>`;
  }
  if (tp > 1) out += text(x + w / 2, y + 86, `${s.tpLink} · TP ${tp}`, "dg-mini", "middle");
  let yy = y + (tp > 1 ? 94 : 78);
  if (s.cpuGiB > 0) {
    const f = Math.min(1, (s.share.cpu ?? 0) * 4);
    out += box(x + pad, yy, w - 2 * pad, 24, "dg-cpu", "cpu", `CPU memory ${gib(s.cpuGiB)} per node, shared by its GPUs. Served ${pct(s.share.cpu)} of prompt tokens.`,
      `<rect class="dg-fill" x="${x + pad + 3}" y="${yy + 3}" width="${(w - 2 * pad - 6) * f}" height="18" rx="6"/>` + text(x + pad + 10, yy + 16, `CPU memory · ${gib(s.cpuGiB)}`, "dg-label") + text(x + w - pad - 8, yy + 16, pct(s.share.cpu), "dg-label dg-num", "end"));
    yy += 30;
  }
  if (s.disk) {
    out += box(x + pad, yy, w - 2 * pad, 24, "dg-disk", "disk", `${s.disk.name}, ${gib(s.disk.gib)} per node. Served ${pct(s.share.disk)} of prompt tokens.`,
      text(x + pad + 10, yy + 16, `Disk · ${s.disk.short}`, "dg-label") + text(x + w - pad - 8, yy + 16, pct(s.share.disk), "dg-label dg-num", "end"));
    yy += 30;
  }
  return { svg: out, height: yy - y + 6 };
}

export function renderDiagram(s, { vertical = false } = {}) {
  const shown = Math.min(s.nodes, vertical ? 2 : 3);
  const more = s.nodes - shown;
  const W = vertical ? 360 : 1000;
  const nodeW = vertical ? 300 : 320;
  const nodeX = vertical ? 30 : 370;
  // node cards first, to know their height
  const cards = [];
  let y = vertical ? 150 : 24;
  for (let i = 0; i < shown; i++) {
    const n = node(nodeX, y, nodeW, s, i + 1);
    cards.push({ y, h: n.height, svg: n.svg });
    y += n.height + ((s.pp ?? 1) > 1 && i % s.pp !== s.pp - 1 ? 30 : 14);
  }
  // pipeline stages of one instance: joined by the network between nodes
  const pp = s.pp ?? 1;
  const stageLinks = [];
  for (let i = 0; i + 1 < cards.length; i++) {
    if (pp > 1 && i % pp !== pp - 1) {
      const lx = nodeX + nodeW / 2, y0 = cards[i].y + cards[i].h, y1 = cards[i + 1].y;
      stageLinks.push(`<g class="dg-link"><title>${esc(`${s.stageLink ?? "network"}: pipeline stage ${(i % pp) + 1} hands its hidden states to stage ${(i % pp) + 2}`)}</title><line x1="${lx}" y1="${y0 + 1}" x2="${lx}" y2="${y1 - 1}"/></g>` +
        text(lx + 8, (y0 + y1) / 2 + 4, `${s.stageLinkShort ?? "network"} · PP`, "dg-mini"));
    }
  }
  if (more > 0) { cards.push({ y, h: 30, more: true }); y += 44; }
  const nodesBottom = y;
  const mid = (c) => c.y + c.h / 2;
  const H = vertical ? nodesBottom + (s.shared || s.decode ? 190 : 20) : Math.max(nodesBottom + 10, 330);
  let svg = `<svg viewBox="0 0 ${W} ${H}" class="dg" role="img" aria-label="Cluster diagram: requests from users through a router to ${s.nodes} nodes">`;

  // users and router
  const ux = vertical ? W / 2 - 60 : 10, uy = vertical ? 12 : H / 2 - 32;
  const rx = vertical ? W / 2 - 70 : 165, ry = vertical ? 86 : H / 2 - 26;
  svg += box(ux, uy, 120, 56, "dg-users", "load", s.followUp == null ? "Requests replayed from a trace." : `${s.rate} requests per minute, ${pct(s.followUp)} follow-ups.`,
    `<circle cx="${ux + 24}" cy="${uy + 21}" r="7" class="dg-icon"/><path class="dg-icon" d="M${ux + 12} ${uy + 44} q12 -16 24 0"/>` + text(ux + 44, uy + 25, "Users", "dg-title") + text(ux + 44, uy + 42, s.followUp == null ? "trace" : `${s.rate}/min`, "dg-sub"));
  const routing = { "round-robin": "round-robin by node", sticky: "sticky by conversation", random: "random per request" };
  const routeLabel = routing[s.routing] ?? routing.random;
  svg += box(rx, ry, 140, 52, "dg-router", "routing", `Request routing · ${routeLabel}`,
    text(rx + 70, ry + 22, "Router", "dg-title", "middle") + text(rx + 70, ry + 39, routeLabel, "dg-sub", "middle"));
  svg += vertical ? flow(`M${W / 2} ${uy + 56} V${ry}`, 1, "dg-req", "requests") : flow(`M${ux + 120} ${uy + 28} H${rx}`, 1, "dg-req", "requests");

  // router -> nodes
  for (const [i, c] of cards.entries()) {
    if (!c.more && i % pp) continue; // requests enter an instance at its first stage
    const d = vertical ? `M${W / 2} ${ry + 52} V${c.y}` : `M${rx + 140} ${ry + 26} C${rx + 175} ${ry + 26} ${nodeX - 35} ${mid(c)} ${nodeX} ${mid(c)}`;
    svg += flow(d, pp / Math.max(1, s.nodes), "dg-req", pp > 1 ? "requests routed to this instance's first stage" : "requests routed to this node");
  }
  for (const c of cards) {
    svg += c.more
      ? box(nodeX, c.y, nodeW, c.h, "dg-node dg-more", "nodes", `${more} more identical nodes`, text(nodeX + nodeW / 2, c.y + 20, `+ ${more} more node${more > 1 ? "s" : ""}`, "dg-sub", "middle"))
      : `<g class="dg-node"><rect x="${nodeX}" y="${c.y}" width="${nodeW}" height="${c.h}" rx="12"/>${c.svg}</g>`;
  }
  svg += stageLinks.join("");

  // peer network between nodes
  if (s.p2p && cards.length > 1) {
    const fx = vertical ? nodeX + nodeW + 12 : nodeX + nodeW + 26;
    const y0 = mid(cards[0]), y1 = mid(cards[cards.length - 1]);
    svg += `<g class="dg-box dg-p2p" data-target="p2p" tabindex="0" role="button"><title>${esc(s.p2p.name)}: a block in another node's CPU memory. Served ${pct(s.share.p2p)} of prompt tokens.</title>` +
      cards.map((c) => `<path class="dg-edge dg-p2p-edge" d="M${nodeX + nodeW} ${mid(c)} H${fx}"/>`).join("") +
      flow(`M${fx} ${y0} V${y1}`, s.share.p2p, "dg-p2p-edge", "peer pulls") + "</g>";
    if (!vertical) svg += text(fx + 8, (y0 + y1) / 2 - 4, "Peers", "dg-mini") + text(fx + 8, (y0 + y1) / 2 + 10, `${s.p2p.short} · ${pct(s.share.p2p)}`, "dg-mini");
  }

  // shared storage and decode
  const sx = vertical ? 20 : 775, sy = vertical ? nodesBottom + 20 : 30, sw = vertical ? 155 : 215;
  const dx = vertical ? 185 : 775, dy = vertical ? nodesBottom + 20 : H - 120, dw = vertical ? 155 : 215;
  if (s.shared) {
    svg += box(sx, sy, sw, 92, "dg-shared", "shared", `${s.shared.name}: one store for the whole cluster. Served ${pct(s.share.shared)} of prompt tokens.`,
      `<g transform="translate(${sx + sw - (vertical ? 62 : 74)} ${sy + 6}) scale(${vertical ? 0.55 : 0.65})"><path class="dg-cloud" vector-effect="non-scaling-stroke" d="M30 52 a16 16 0 0 1 4 -31 a22 22 0 0 1 40 -6 a16 16 0 0 1 26 14 a14 14 0 0 1 -4 23 z"/></g>` +
      text(sx + 14, sy + 26, vertical ? "Shared" : "Shared storage", "dg-title") +
      text(sx + 14, sy + 50, s.shared.short, vertical ? "dg-mini" : "dg-sub") +
      text(sx + 14, sy + 76, `${pct(s.share.shared)} of prompt`, "dg-sub dg-num"));
    for (const c of cards) {
      const d = vertical ? `M${nodeX + 40} ${nodesBottom - 14} V${sy}` : `M${nodeX + nodeW} ${mid(c) - 10} C${sx - 40} ${mid(c) - 10} ${sx - 60} ${sy + 46} ${sx} ${sy + 46}`;
      svg += flow(d, (s.share.shared ?? 0) / Math.max(1, cards.length), "dg-shared-edge", "loads from shared storage");
      if (vertical) break;
    }
  }
  if (s.decode) {
    svg += box(dx, dy, dw, 92, "dg-decode", "decode", `Decode runs on ${s.decode.gpus} GPU${s.decode.gpus === 1 ? "" : "s"} of its own (disaggregated), batch up to ${s.decode.batch}; prefill nodes send each prompt's KV over ${s.transfer}.`,
      text(dx + 14, dy + 26, "Decode pool", "dg-title") + text(dx + 14, dy + 46, `${s.decode.gpus}× ${s.decode.gpuShort}`, "dg-sub") + text(dx + 14, dy + 64, `batch ≤ ${s.decode.batch}`, "dg-sub") + text(dx + 14, dy + 82, `KV via ${s.transferShort}`, "dg-mini"));
    for (const c of cards) {
      const d = vertical ? `M${nodeX + nodeW - 40} ${nodesBottom - 14} V${dy}` : `M${nodeX + nodeW} ${mid(c) + 10} C${dx - 40} ${mid(c) + 10} ${dx - 60} ${dy + 46} ${dx} ${dy + 46}`;
      svg += flow(d, 0.35, "dg-decode-edge", "KV sent to decode");
      if (vertical) break;
    }
  }
  return `${svg}</svg>`;
}
