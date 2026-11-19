// Cache simulator: replays requests one at a time through LRU tiers and charges each request's TTFT from model.js.
// DOM-free, so `node --test` covers it.
//
// Tiers: each instance's GPU, its node's CPU memory, then optional ones: a local disk per node, peer nodes' CPU memory
// over the network, and shared storage for the whole cluster. As in vLLM's tiered offloading, every tier is written
// through and evicts on its own (LRU).

import { decodeStepS, GIB, timing } from "./model.js";

// Byte-bounded LRU: a doubly linked list (oldest at head) indexed by a Map, so every operation is O(1).
export class LRU {
  constructor(capacityBytes, hooks = {}) {
    this.cap = capacityBytes == null ? Infinity : capacityBytes;
    this.map = new Map(); // id -> node {id, bytes, prev, next}
    this.head = null;
    this.tail = null;
    this.used = 0;
    this.evictions = 0;
    this.added = 0; // bytes newly inserted (not refreshes)
    this.onAdd = hooks.onAdd;
    this.onEvict = hooks.onEvict;
  }
  has(id) { return this.map.has(id); }
  keys() { const out = []; for (let n = this.head; n; n = n.next) out.push(n.id); return out; } // oldest first
  #unlink(n) {
    if (n.prev) n.prev.next = n.next; else this.head = n.next;
    if (n.next) n.next.prev = n.prev; else this.tail = n.prev;
    n.prev = n.next = null;
  }
  #append(n) {
    n.prev = this.tail;
    if (this.tail) this.tail.next = n; else this.head = n;
    this.tail = n;
  }
  // Insert or refresh `id` as most recently used, then evict the oldest entries until it fits.
  put(id, bytes) {
    if (this.cap <= 0) return;
    let n = this.map.get(id);
    if (n) {
      this.#unlink(n);
      if (bytes > n.bytes) { this.used += bytes - n.bytes; n.bytes = bytes; }
    } else {
      n = { id, bytes, prev: null, next: null };
      this.map.set(id, n);
      this.used += bytes;
      this.added += bytes;
      this.onAdd?.(id);
    }
    this.#append(n);
    while (this.used > this.cap && this.head) {
      const old = this.head;
      this.#unlink(old);
      this.map.delete(old.id);
      this.used -= old.bytes;
      this.evictions++;
      this.onEvict?.(old.id);
    }
  }
}

export function rng(seed = 1) { // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- workloads: {name, blockTokens (null: the model's), requests: [{t, n, conv, hashes?}]} -----------------------

// Conversations. `followUpShare` of requests continue an existing conversation; the rest start a new one. New
// conversations arrive as a Poisson process at requestsPerMin x (1 - followUpShare); after each turn a conversation
// continues with probability followUpShare, after an exponential pause of mean thinkTimeS, so in steady state that
// share of requests are follow-ups. Turn k sends the shared system prompt, every earlier message and answer, and the
// new message: system + (k-1) x (input + output) + input tokens. A conversation ends at maxContextTokens.
export function synthetic(p) {
  const r = rng(p.seed ?? 1);
  const exp = (mean) => -Math.log(1 - r()) * mean;
  const share = Math.min(Math.max(p.followUpShare, 0), 0.99);
  const newPerS = (p.requestsPerMin * (1 - share)) / 60;
  const end = p.durationMin * 60;
  const maxCtx = p.maxContextTokens ?? Infinity;
  const requests = [];
  let conv = 0;
  for (let start = exp(1 / newPerS); start < end; start += exp(1 / newPerS), conv++) {
    let t = start;
    for (let k = 1; ; k++) {
      const n = p.systemTokens + (k - 1) * (p.inputTokens + p.outputTokens) + p.inputTokens;
      if (n + p.outputTokens > maxCtx) break;
      requests.push({ t, n, out: p.outputTokens, conv, sys: p.systemTokens, followUp: k > 1 });
      t += exp(p.thinkTimeS);
      if (r() >= share || t >= end) break;
    }
  }
  requests.sort((a, b) => a.t - b.t);
  return { name: "synthetic", blockTokens: null, requests };
}

// Mooncake-style JSONL: {"timestamp": ms, "input_length": n, "output_length": m, "hash_ids": [...]}, one hash per
// `blockTokens`-token block of the prompt (512 in the Mooncake traces). A JSON array of the same objects also works.
export function parseTrace(text, blockTokens = 512, name = "trace") {
  const t = text.trim();
  const items = t.startsWith("[") ? JSON.parse(t) : t.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
  const requests = items.map((o, i) => {
    if (!Array.isArray(o.hash_ids) || !(o.input_length > 0)) throw new Error(`record ${i + 1}: needs input_length and hash_ids`);
    return { t: (o.timestamp ?? 0) / 1000, n: o.input_length, out: o.output_length ?? 0, hashes: o.hash_ids };
  });
  requests.sort((a, b) => a.t - b.t);
  return { name, blockTokens, requests };
}

// Full blocks of a request (a partial last block is never cached). Synthetic block ids are shared while the block
// lies inside the system prompt, and per conversation after it: a valid prefix hash, since prompts only grow.
function blocksOf(req, B) {
  const nb = Math.floor(req.n / B);
  if (req.hashes) return req.hashes.slice(0, nb);
  const sysBlocks = Math.floor(req.sys / B);
  const ids = new Array(nb);
  for (let j = 0; j < nb; j++) ids[j] = j < sysBlocks ? `s${j}` : `c${req.conv}:${j}`;
  return ids;
}

// Share of prompt tokens an infinite cache would serve: the most any tier chain can hit on this workload.
export function idealHitShare(workload, blockTokens) {
  const B = workload.blockTokens ?? blockTokens;
  const seen = new Set();
  let hit = 0, total = 0;
  for (const req of workload.requests) {
    const ids = blocksOf(req, B);
    let i = 0;
    while (i < ids.length && seen.has(ids[i])) i++;
    hit += i * B;
    total += req.n;
    for (const id of ids) seen.add(id);
  }
  return { share: total ? hit / total : 0, uniqueTokens: seen.size * B };
}

const pct = (sorted, q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];

// One replay of a workload through a cluster. cfg:
//   model (architecture, already withTp), gpu, software (presets.SOFTWARE), routing: "random" | "sticky", seed, warmupS,
//   cluster: {nodes, gpusPerNode, tp, pp}: nodes x gpusPerNode / (tp x pp) prefill instances, each spanning pp nodes
//     (pipeline stages, joined by `stageLink`), each one request at a time (FIFO; with pp > 1 the next request enters
//     the first stage when the current one leaves it: ideal pipelining). pp nodes form one cache group: their CPU
//     memories, disks and links serve the instance's layers side by side, so capacities add up,
//   gpuGiB (KV per instance), cpuGiB (per node, 0 = none),
//   disk: fs variant per node | null, p2p: network variant | null (a hit = a block in another node's CPU tier),
//   shared: object storage / shared filesystem variant for the whole cluster | null,
//   decode: {gpu, batch, batchCap(ctx), pooled?} | null: disaggregated, batched decode on GPUs of its own (same TP),
//   transfer: network variant for the prefill -> decode KV | null, simBlockTokens,
//   costScale: real nodes / simulated nodes (a large cluster is replayed at <= 8 nodes with its per-node load).
// Every tier is written through and evicts on its own (LRU). Lookup per block, nearest first: own GPU, own node's CPU,
// local disk, peer nodes, shared storage; a prefix hit ends at the first block no tier holds.
// Statistics leave out requests that arrive in the first `warmupS` seconds; those still fill the caches.
export function simulate(workload, cfg) {
  const m = cfg.model;
  const cl = { nodes: 1, gpusPerNode: 1, tp: 1, pp: 1, ...cfg.cluster };
  const pp = cl.pp;
  const T = timing(m, cfg.gpu, cfg.software, cl.tp, undefined, { stages: pp, link: cfg.stageLink });
  const stepHops = pp > 1 && cfg.stageLink ? (pp - 1) * (cfg.stageLink.latency_s ?? 0) : 0; // a decode step's hidden state is tiny
  const B = workload.blockTokens ?? Math.max(m.block_size, Math.ceil((cfg.simBlockTokens ?? 0) / m.block_size) * m.block_size);
  const blockBytes = B * m.kv_bytes_per_token;
  const perNode = Math.max(1, Math.floor(cl.gpusPerNode / cl.tp));
  const gib = (x) => (x == null ? null : x * GIB);

  // A model with a fixed per-prefix state (linear attention) can only resume where a state snapshot was stored: the end
  // of an earlier stored sequence. vLLM keeps one snapshot per cached sequence, so a hit is cut back to the deepest
  // snapshot it reaches; a model without state can resume at any block.
  const snapAt = m.read_fixed_bytes > 0 ? new Set() : null;
  const holders = new Map(); // block -> number of nodes whose CPU tier holds it (p2p lookups)
  const hook = { onAdd: (k) => holders.set(k, (holders.get(k) ?? 0) + 1), onEvict: (k) => { const c = holders.get(k) - 1; if (c) holders.set(k, c); else holders.delete(k); } };
  // one entry per cache group of pp nodes (a node when pp = 1)
  const nodes = Array.from({ length: Math.max(1, Math.floor(cl.nodes / pp)) }, () => ({
    cpu: new LRU(gib((cfg.cpuGiB ?? 0) * pp), cfg.p2p ? hook : {}),
    disk: cfg.disk ? new LRU(gib(cfg.disk.capacity_gib == null ? null : cfg.disk.capacity_gib * pp)) : null,
  }));
  const inst = nodes.flatMap((node, ni) => Array.from({ length: perNode }, () => ({ node: ni, gpu: new LRU(gib(cfg.gpuGiB)), freeAt: 0 })));
  const shared = cfg.shared ? new LRU(gib(cfg.shared.capacity_gib)) : null;

  // optional tiers, nearest first
  const lower = [
    cfg.disk && { key: "disk", v: cfg.disk, has: (id, n) => n.disk.has(id) },
    cfg.p2p && { key: "p2p", v: cfg.p2p, has: (id) => holders.has(id) }, // own node's CPU was checked first
    shared && { key: "shared", v: cfg.shared, has: (id) => shared.has(id) },
  ].filter(Boolean);
  for (const t of lower) t.load = T.loadFor(t.v);
  const loadCpu = T.cpuLoad;
  const xfer = T.transferFor(cfg.decode ? cfg.transfer : null);
  const dec = cfg.decode;
  const r = rng(cfg.seed ?? 7);

  const keys = ["gpu", "cpu", ...lower.map((t) => t.key), "miss"];
  const tokens = Object.fromEntries(keys.map((k) => [k, 0]));
  const ttfts = [], e2es = [], waits = [], prefills = [], transfers = [], steps = [];
  let promptTokens = 0, outputTokens = 0, busy = 0, decodeGpuSeconds = 0, counted = 0;
  let busyUncached = 0, busyCached = 0, cachedTokens = 0; // prefill time by what it served, for the price sheet
  const reqs = workload.requests;
  const t0 = reqs.length ? reqs[0].t + (cfg.warmupS ?? 0) : 0;
  let sharedAddedAtT0 = null, diskAddedAtT0 = null;
  const diskAdded = () => nodes.reduce((a, n) => a + (n.disk?.added ?? 0), 0);

  for (const req of reqs) {
    if (sharedAddedAtT0 == null && req.t >= t0) { sharedAddedAtT0 = shared ? shared.added : 0; diskAddedAtT0 = diskAdded(); }
    const here = cfg.routing === "sticky" && req.conv != null ? inst[req.conv % inst.length] : inst[Math.floor(r() * inst.length)];
    const node = nodes[here.node];
    const ids = blocksOf(req, B);
    const k = Object.fromEntries(keys.map((kk) => [kk, 0]));
    const where = [];
    for (const id of ids) {
      if (here.gpu.has(id)) { where.push("gpu"); continue; }
      if (node.cpu.has(id)) { where.push("cpu"); continue; }
      const t = lower.find((tt) => tt.has(id, node));
      if (!t) break;
      where.push(t.key);
    }
    if (snapAt) { let j = where.length; while (j > 0 && !snapAt.has(ids[j - 1])) j--; where.length = j; }
    for (const w of where) k[w]++;
    const h = (ids.length && k.gpu + k.cpu + lower.reduce((a, t) => a + k[t.key], 0)) * B;
    const last = where[where.length - 1]; // the tier that holds the hit's last block, and with it the state
    const loads = loadCpu(k.cpu * B, last === "cpu") + lower.reduce((a, t) => a + t.load(k[t.key] * B, last === t.key), 0);
    const prefill = T.ttft(req.n, h, loads);
    const start = Math.max(req.t, here.freeAt);
    here.freeAt = start + prefill / pp;
    if (req.t >= t0) {
      const out = req.out ?? 0;
      const ctx = req.n + out / 2;
      const batch = dec ? Math.max(1, Math.min(dec.batch, dec.batchCap ? dec.batchCap(ctx) : dec.batch)) : 1;
      const step = dec ? decodeStepS(m, dec.gpu, batch, ctx, cl.tp) + stepHops : 0;
      const tr = xfer(req.n);
      const ttft = start - req.t + prefill + tr + step;
      ttfts.push(ttft); waits.push(start - req.t); prefills.push(prefill); transfers.push(tr);
      e2es.push(ttft + Math.max(0, out - 1) * step);
      if (dec) steps.push(step);
      busy += prefill / pp;
      // overhead by token share; recompute (with its attention over the cached part) to uncached, loads to cached
      const over = T.overhead(req.n), hh = Math.min(h, req.n);
      busyCached += loads + (over * hh) / req.n;
      busyUncached += prefill - loads - (over * hh) / req.n;
      cachedTokens += hh;
      decodeGpuSeconds += ((out * step) / batch) * cl.tp;
      promptTokens += req.n;
      outputTokens += out;
      for (const kk of keys) if (kk !== "miss") tokens[kk] += k[kk] * B;
      tokens.miss += req.n - h;
      counted++;
    }
    // write through the prompt and, for synthetic conversations, the answer generated after it (vLLM caches both),
    // last block first, so a prefix's first blocks (the most shared) are evicted last
    const stored = req.hashes ? ids : blocksOf({ ...req, n: req.n + (req.out ?? 0) }, B);
    const tiers = [here.gpu, node.cpu, node.disk, shared].filter(Boolean);
    if (snapAt && stored.length) snapAt.add(stored[stored.length - 1]);
    for (let j = stored.length - 1; j >= 0; j--) {
      const bytes = blockBytes + (j === stored.length - 1 ? m.read_fixed_bytes : 0);
      for (const tier of tiers) tier.put(stored[j], bytes);
    }
  }

  const scaleUp = cfg.costScale ?? 1;
  const windowS = counted ? Math.max(1, reqs[reqs.length - 1].t - t0) : 0;
  const windowH = windowS / 3600;
  const realNodes = cl.nodes * scaleUp;
  const prefillGpus = realNodes * cl.gpusPerNode;
  const usdH = (g) => g?.usd_per_hour ?? 0;
  // provisioned GPUs are paid for the whole window, busy or not; decode in whole TP instances
  // decode instances span pp nodes too and keep pp batches in flight, so a step's GPU time is tp, not tp x pp
  // (decode.pooled: sized to the load in GPU fractions, as a provider pooling many clusters' decode would)
  const decodeNeed = (decodeGpuSeconds * scaleUp) / Math.max(windowS, 1);
  const decodeGpus = !dec ? 0 : dec.pooled ? decodeNeed : Math.ceil(decodeNeed / (cl.tp * pp)) * cl.tp * pp;
  const prefillUsd = prefillGpus * usdH(cfg.gpu) * windowH;
  const decodeUsd = decodeGpus * usdH(dec?.gpu) * windowH;
  const storageOf = (v, lru, copies) => {
    if (!v) return { storage: 0, fixed: 0 };
    const price = v.usd_per_gib_month ?? 0;
    let gibHeld;
    if (v.bill === "provisioned" && v.capacity_gib != null) gibHeld = v.capacity_gib * copies;
    else { // bytes stored: what is written per hour, kept for the retention time (capped by capacity)
      const writtenGiB = ((lru.added - (lru === shared ? sharedAddedAtT0 ?? 0 : 0)) / GIB) * scaleUp;
      gibHeld = (writtenGiB / Math.max(windowH, 1e-9)) * (cfg.retentionH ?? 24);
      if (v.capacity_gib != null) gibHeld = Math.min(gibHeld, v.capacity_gib * copies);
    }
    return { storage: gibHeld * price * (windowH / 730), fixed: (v.usd_per_hour ?? 0) * copies * windowH };
  };
  // per-GB fees (S3 Express): what was written after the warm-up, and what was read
  const fees = (v, writtenBytes, readTokens) => (v ? ((v.usd_per_gib_written ?? 0) * writtenBytes + (v.usd_per_gib_read ?? 0) * readTokens * m.kv_bytes_per_token) / GIB * scaleUp : 0);
  const diskCost = cfg.disk ? storageOf(cfg.disk, nodes[0].disk, realNodes) : { storage: 0, fixed: 0 };
  diskCost.storage += fees(cfg.disk, diskAdded() - (diskAddedAtT0 ?? 0), tokens.disk ?? 0);
  const sharedCost = storageOf(cfg.shared, shared, 1);
  sharedCost.storage += fees(cfg.shared, shared ? shared.added - (sharedAddedAtT0 ?? 0) : 0, tokens.shared ?? 0);
  const p2pFixed = cfg.p2p ? (cfg.p2p.usd_per_hour ?? 0) * realNodes * windowH : 0;
  const storageUsd = diskCost.storage + sharedCost.storage;
  const fixedUsd = diskCost.fixed + sharedCost.fixed + p2pFixed;
  const total = prefillUsd + decodeUsd + storageUsd + fixedUsd;
  const stats = (xs) => {
    const so = [...xs].sort((a, b) => a - b);
    return { mean: xs.reduce((a, b) => a + b, 0) / (xs.length || 1), p50: pct(so, 0.5), p90: pct(so, 0.9) };
  };
  const share = Object.fromEntries(Object.entries(tokens).map(([kk, v]) => [kk, promptTokens ? v / promptTokens : 0]));
  const priced = usdH(cfg.gpu) > 0 && (!dec || usdH(dec.gpu) > 0) && [cfg.disk, cfg.shared].every((v) => !v || v.usd_per_gib_month != null);
  return {
    requests: counted, promptTokens, outputTokens, tokens, share, order: keys, blockTokens: B, windowS,
    ttft: stats(ttfts), e2e: stats(e2es), wait: stats(waits).mean, prefill: stats(prefills).mean, transfer: stats(transfers).mean,
    tpot: steps.length ? stats(steps).mean : null,
    utilization: windowS ? busy / windowS / inst.length : 0, // above 1, requests queue without bound
    instances: inst.length * scaleUp, prefillGpus, decodeGpus, priced,
    usd: { prefill: prefillUsd, decode: decodeUsd, storage: storageUsd, fixed: fixedUsd, total },
    usdPer1kRequests: counted ? ((total / scaleUp / counted) * 1e3) : 0,
    usdPer1MTokens: promptTokens + outputTokens ? (total / scaleUp / (promptTokens + outputTokens)) * 1e6 : 0,
    // an API-style price sheet: every $ goes to exactly one of the three (provisioned prefill GPUs, idle time included,
    // split by the busy time each kind of token caused; storage and fixed tier costs to cached input; decode to output)
    price: (() => {
      const split = busyUncached + busyCached ? busyUncached / (busyUncached + busyCached) : 1;
      const usd = { input: prefillUsd * split, cachedInput: prefillUsd * (1 - split) + storageUsd + fixedUsd, output: decodeUsd };
      const tok = { input: promptTokens - cachedTokens, cachedInput: cachedTokens, output: dec ? outputTokens : 0 };
      const per1M = (k) => (tok[k] ? (usd[k] / scaleUp / tok[k]) * 1e6 : null);
      return { usd, tokens: tok, input: per1M("input"), cachedInput: per1M("cachedInput"), output: per1M("output") };
    })(),
    evictions: { gpu: inst.reduce((a, i) => a + i.gpu.evictions, 0), cpu: nodes.reduce((a, n) => a + n.cpu.evictions, 0) },
  };
}
