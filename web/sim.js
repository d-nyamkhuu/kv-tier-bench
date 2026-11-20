// Cache simulator: replays timestamped events through LRU tiers and charges each request's TTFT from model.js.
// DOM-free, so `node --test` covers it.
//
// Tiers: each instance's GPU, its node's CPU memory, then optional ones: a local disk per node, peer nodes' CPU memory
// over the network, and shared storage for the whole cluster. Newly admitted chunks write through to configured backing stores.
// Host and bounded native filesystem caches use LMCache MP's periodic LRU controller.

import { Events } from "./events.js";
import { decodePool } from "./decode.js";
import { decodeStep, GIB, instanceWeightBytes, perSeqBytes, stagedTransferSeconds, timing, variantBandwidth, writeBandwidth } from "./model.js";

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
  bytes(id) { return this.map.get(id)?.bytes ?? 0; }
  touch(id) {
    const n = this.map.get(id);
    if (n && n !== this.tail) { this.#unlink(n); this.#append(n); }
  }
  sweep() {
    if (!this.head || this.used < 0.8 * this.cap) return;
    const count = Math.max(1, Math.floor(this.map.size * 0.2));
    for (let i = 0; i < count; i++) this.#evictOldest();
  }
  // Remove `id` if present (used by lifetime expiration, not counted as a capacity eviction).
  delete(id) {
    const n = this.map.get(id);
    if (!n) return false;
    this.#unlink(n);
    this.map.delete(id);
    this.used -= n.bytes;
    this.onEvict?.(id);
    return true;
  }
  // Drop entries from the least recently used end while `stale(id)` holds: a cache lifetime, not an eviction.
  expire(stale) {
    while (this.head && stale(this.head.id)) {
      const old = this.head;
      this.#unlink(old);
      this.map.delete(old.id);
      this.used -= old.bytes;
      this.onEvict?.(old.id);
    }
  }
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
      if (bytes > n.bytes) { this.added += bytes - n.bytes; this.used += bytes - n.bytes; n.bytes = bytes; }
    } else {
      n = { id, bytes, prev: null, next: null };
      this.map.set(id, n);
      this.used += bytes;
      this.added += bytes;
      this.onAdd?.(id);
    }
    this.#append(n);
    while (this.used > this.cap && this.head) this.#evictOldest();
  }
  #evictOldest() {
    const old = this.head;
    this.#unlink(old);
    this.map.delete(old.id);
    this.used -= old.bytes;
    this.evictions++;
    this.onEvict?.(old.id);
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
// share of requests are follow-ups. Turn k sends the system prompt, every earlier message and answer, and the
// new message: system + (k-1) x (input + output) + input tokens. A conversation that outgrows maxContextTokens drops
// its oldest turns, as chat apps truncate or compact a long history, which changes everything after the system prompt.
// Optional realism, each off by default:
//   lengthSpread: coefficient of variation of each message and reply length (lognormal with the set mean; 0 = fixed),
//   ownPromptShare: share of conversations with a system prompt of their own, shared with no other conversation,
//   rewriteShare: share of follow-ups that rewrite earlier history (an edited message, a compacted or truncated tool
//     result, a refreshed timestamp), so the turn shares only the system prompt with the cache,
//   systemPrompts: distinct system prompts of the same length, chosen per conversation with Zipf(1) popularity,
//   arrivalPattern: "steady" (Poisson) or "bursty" (new conversations at 2x the mean rate for 2 minutes of every 6,
//     0.5x for the other 4, the same hourly volume).
// Those draws come from a second random stream, so the default workload is identical with or without them.
export function synthetic(p) {
  const r = rng(p.seed ?? 1);
  const extra = rng((p.seed ?? 1) + 7919);
  const exp = (mean) => -Math.log(1 - r()) * mean;
  const share = Math.min(Math.max(p.followUpShare, 0), 0.99);
  const newPerS = (p.requestsPerMin * (1 - share)) / 60;
  const end = p.durationMin * 60;
  const maxCtx = p.maxContextTokens ?? Infinity;
  const cv = Math.max(0, p.lengthSpread ?? 0);
  const sigma = Math.sqrt(Math.log(1 + cv * cv));
  const length = (mean) => {
    if (!cv) return mean;
    const normal = Math.sqrt(-2 * Math.log(1 - extra())) * Math.cos(2 * Math.PI * extra());
    return Math.max(1, Math.round(mean * Math.exp(sigma * normal - (sigma * sigma) / 2)));
  };
  const prompts = Math.max(1, Math.round(p.systemPrompts ?? 1));
  const weights = Array.from({ length: prompts }, (_, i) => 1 / (i + 1));
  const total = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  const cdf = weights.map((w) => (acc += w / total));
  const pickPrompt = () => {
    if (prompts === 1) return 0;
    const u = extra();
    let lo = 0, hi = prompts - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    return lo;
  };
  const bursty = p.arrivalPattern === "bursty";
  const peak = bursty ? 2 : 1;
  const level = (t) => (t % 360 < 120 ? 2 : 0.5);
  const requests = [];
  let conv = 0;
  for (let start = exp(1 / (newPerS * peak)); start < end; start += exp(1 / (newPerS * peak)), conv++) {
    if (bursty && extra() * peak >= level(start)) continue; // thinning to the burst profile
    // A chat with its own system prompt (a user's custom instructions or an app nobody else calls) shares no prefix.
    const sysId = p.ownPromptShare > 0 && extra() < p.ownPromptShare ? `u${conv}` : pickPrompt();
    let t = start, history = 0, ver = 0;
    for (let k = 1; ; k++) {
      const input = length(p.inputTokens), out = length(p.outputTokens);
      if (k > 1 && p.rewriteShare > 0 && extra() < p.rewriteShare) ver++;
      if (p.systemTokens + history + input + out > maxCtx) {
        const kept = Math.max(0, maxCtx - p.systemTokens - input - out);
        if (!kept && p.systemTokens + input + out > maxCtx) break;
        history = Math.min(history, kept);
        ver++;
      }
      const n = p.systemTokens + history + input;
      requests.push({ t, n, out, conv, ver, sys: p.systemTokens, sysId, followUp: k > 1 });
      history += input + out;
      t += exp(p.thinkTimeS);
      if (r() >= share || t >= end) break;
    }
  }
  requests.sort((a, b) => a.t - b.t);
  return { name: "synthetic", durationS: end, blockTokens: null, requests };
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
  const sys = req.sysId ? `s${req.sysId}:` : "s";
  const conv = req.ver ? `c${req.conv}.${req.ver}:` : `c${req.conv}:`;
  for (let j = 0; j < nb; j++) ids[j] = j < sysBlocks ? sys + j : conv + j;
  return ids;
}

// Share of prompt tokens an infinite cache would serve: the most any tier chain can hit on this workload.
export function idealHitShare(workload, blockTokens, minCacheTokens = 0) {
  const B = workload.blockTokens ?? blockTokens;
  const seen = new Set();
  let hit = 0, total = 0;
  for (const req of workload.requests) {
    const ids = req.n < minCacheTokens ? [] : blocksOf(req, B);
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
//   model (architecture, already withTp), gpu, software (presets.SOFTWARE), routing: "round-robin" | "random" | "sticky", seed, warmupS,
//   cluster: {nodes, gpusPerNode, tp, pp}: nodes x gpusPerNode / (tp x pp) prefill instances, each spanning pp nodes
//     (pipeline stages, joined by `stageLink`), each one request at a time (FIFO; with pp > 1 the next request enters
//     the first stage when the current one leaves it: ideal pipelining). pp nodes form one cache group: their CPU
//     memories, disks and links serve the instance's layers side by side, so capacities add up,
//   gpuGiB (KV per instance), cpuGiB (per node, 0 = none),
//   disk: fs variant per node | null, p2p: network variant | null (a hit = a block in another node's CPU tier),
//   shared: object storage variant for the whole cluster | null,
//   decode: {gpu, batch, batchCap(ctx), pooled?, speed?, spec?, ep?, overlap?} | null: disaggregated, batched
//     decode on GPUs of its own (same TP); speed (tokens/s per request): the speed target limits the number of ready
//     requests admitted to a batch alongside the memory and sequence limits; spec: speculative decoding (a token takes step / accepted tokens); ep: experts spread over ep
//     GPUs (decode then needs no pipeline stages), batch per attention group of tp GPUs (model.decodeStep),
//   transfer: network variant for the prefill -> decode KV | null, simBlockTokens,
//   cacheLifetimeS: hosted-API cache lifetime: a block unused for this long expires from every tier (null: kept
//     until evicted, as vLLM and LMCache do), minCacheTokens: prompts shorter than this are neither looked up nor saved,
//   prefillBatchTokens: continuous batching of prefill, vLLM's max_num_batched_tokens (null or 0: one request at a time).
//     Each step takes waiting requests first come, first served up to this many new tokens, splitting the last one
//     (chunked prefill), and finishes them together. A step costs one read of the weights (at the GPU's mbu, capped
//     at software.request_overhead_s) plus the FLOPs and per-token overhead of its tokens; the rest of the
//     calibrated per-request overhead delays each request without occupying the GPU. A request alone in its step
//     therefore keeps the one-at-a-time TTFT exactly. Cache loads run while other requests compute.
//   costScale: real nodes / simulated nodes (a large cluster is replayed at <= 8 nodes with its per-node load).
// Lookup checks the local GPU, CPU and disk, then shared storage and remote CPU in the selected order.
// Statistics leave out requests that arrive in the first `warmupS` seconds; those still fill the caches.
export function simulate(workload, cfg, observeRequest, observeMinuteEnd) {
  const m = cfg.model;
  const cl = { nodes: 1, gpusPerNode: 1, tp: 1, pp: 1, ...cfg.cluster };
  const pp = cl.pp;
  const T = timing(m, cfg.gpu, cfg.software, cl.tp, undefined, { stages: pp, link: cfg.stageLink });
  const decTp = cfg.decode?.tp ?? (m.attention === "dp" ? 1 : cl.tp);
  const wideEp = cfg.decode?.wideEp !== undefined ? cfg.decode.wideEp : (cfg.decode?.ep > cl.tp ? cfg.decode.ep : null);
  const decodeEp = cfg.decode?.ep > decTp ? cfg.decode.ep : (m.attention === "dp" && cl.tp > 1 ? cl.tp : null);
  const stepHops = !wideEp && pp > 1 && cfg.stageLink ? (pp - 1) * (cfg.stageLink.latency_s ?? 0) : 0; // a decode step's hidden state is tiny
  const decodeOpts = cfg.decode ? { spec: cfg.decode.spec ?? null, ep: decodeEp, link: cfg.decode.link ?? cfg.transfer ?? null, overlap: !!cfg.decode.overlap } : null;
  const B = workload.blockTokens ?? Math.max(m.block_size, Math.ceil((cfg.simBlockTokens ?? 0) / m.block_size) * m.block_size);
  const blockBytes = B * m.kv_bytes_per_token;
  const backingScale = cfg.externalCopiesPerGpuCopy ?? 1;
  const backingBlock = blockBytes * backingScale;
  const replicasPerNode = Math.max(1, Math.floor(cl.gpusPerNode / cl.tp));
  const dpRanks = m.attention === "dp" ? cl.tp : 1;
  const tpAttn = m.attention === "dp" ? 1 : cl.tp;
  const perNode = replicasPerNode * dpRanks;
  const gib = (x) => (x == null ? null : x * GIB);

  const externalState = m.read_fixed_bytes * backingScale;
  const s3 = cfg.shared?.usd_per_1k_put != null || cfg.shared?.usd_per_1k_get != null;
  const ranks = cfg.objectRanks ?? 1;
  const operations = { put: 0, get: 0, head: 0 };
  let diskReadBytes = 0, sharedReadBytes = 0, sharedWriteBytes = 0;
  const holders = new Map(); // block -> number of nodes whose CPU tier holds it (p2p lookups)
  const hook = { onAdd: (k) => holders.set(k, (holders.get(k) ?? 0) + 1), onEvict: (k) => { const c = holders.get(k) - 1; if (c) holders.set(k, c); else holders.delete(k); } };
  // one entry per cache group of pp nodes (a node when pp = 1).
  // Full-duplex lanes: frontend Ethernet (S3 and TCP links) is separate from dedicated RDMA rails (IB/RoCE).
  // Per-GPU RDMA fabrics (InfiniBand / NVL72) expose one independent tpAttn-NIC rail group per worker on the node.
  const nodes = Array.from({ length: Math.max(1, Math.floor(cl.nodes / pp)) }, () => ({
    diskLane: { free: 0 }, ethTx: { free: 0 }, ethRx: { free: 0 }, rdmaTx: { free: 0 }, rdmaRx: { free: 0 },
    rdmaTxRails: Array.from({ length: perNode }, () => ({ free: 0 })),
    rdmaRxRails: Array.from({ length: perNode }, () => ({ free: 0 })),
    diskWritten: 0,
    knownObjects: s3 ? new Set() : null,
    knownWindows: s3 && externalState ? new Set() : null,
    cpu: new LRU(gib((cfg.cpuGiB ?? 0) * pp), cfg.p2p ? hook : {}),
    disk: cfg.disk ? new LRU(gib(cfg.disk.capacity_gib == null ? null : cfg.disk.capacity_gib * pp)) : null,
  }));
  const earliest = (lanes) => lanes.reduce((best, l) => (l.free < best.free ? l : best));
  const txLane = (node, v, replica = null) => !v?.rdma ? node.ethTx : !v.per_gpu ? node.rdmaTx
    : replica != null ? node.rdmaTxRails[replica] : earliest(node.rdmaTxRails);
  const rxLane = (node, v, replica = null) => !v?.rdma ? node.ethRx : !v.per_gpu ? node.rdmaRx
    : replica != null ? node.rdmaRxRails[replica] : earliest(node.rdmaRxRails);
  const linkRails = (v) => (v?.per_gpu ? tpAttn : 1);
  const workerGpuGiB = cfg.gpuGiB == null ? null : cfg.gpuGiB / dpRanks;
  const inst = nodes.flatMap((node, ni) => Array.from({ length: perNode }, (_, replica) => ({
    node: ni, replica, hostLane: { free: 0 }, gpu: new LRU(gib(workerGpuGiB)), running: false, queue: [], head: 0, waiting: [], stepping: false,
  })));
  const sharedLifetime = cfg.sharedLifetimeS > 0 ? cfg.sharedLifetimeS : null;
  const sharedLastUse = sharedLifetime ? new Map() : null;
  const touchShared = (id, now) => { if (!sharedLifetime) return; sharedLastUse.delete(id); sharedLastUse.set(id, now); };
  const sharedExpired = (id, now) => now - (sharedLastUse.get(id) ?? -Infinity) > sharedLifetime;
  const shared = cfg.shared ? new LRU(gib(cfg.shared.capacity_gib), {
    onEvict: (id) => {
      for (const n of nodes) { n.knownObjects?.delete(id); n.knownWindows?.delete(id); }
      sharedLastUse?.delete(id);
    },
  }) : null;

  // Occupancy counts physical copies (each instance GPU and each node-group CPU/disk)
  // and counts the cluster-wide shared cache once. Only the timeline requests snapshots.
  const pools = observeMinuteEnd ? {
    gpu: inst.map(instance => instance.gpu), cpu: nodes.map(node => node.cpu),
    ...(cfg.disk ? { disk: nodes.map(node => node.disk) } : {}), ...(shared ? { shared: [shared] } : {}),
  } : null;

  // optional tiers, nearest first
  const peerTier = cfg.p2p && { key: "p2p", v: cfg.p2p, has: id => holders.has(id) };
  const sharedTier = shared && { key: "shared", v: cfg.shared, has: id => shared.has(id) };
  const lower = [
    cfg.disk && { key: "disk", v: cfg.disk, has: (id, n) => n.disk.has(id) },
    ...(cfg.peerFirst ? [peerTier, sharedTier] : [sharedTier, peerTier]),
  ].filter(Boolean);
  const loadCpu = T.cpuLoad;
  const xfer = T.transferFor(cfg.decode ? cfg.transfer : null);
  const dec = cfg.decode;
  const r = rng(cfg.seed ?? 7);
  const batchTokens = cfg.prefillBatchTokens > 0 ? cfg.prefillBatchTokens : null;
  const weightRead = cfg.gpu.mem_bw_tb_s && cfg.gpu.mbu ? (instanceWeightBytes(m, cl.tp) / cl.tp) / (cfg.gpu.mem_bw_tb_s * 1e12 * cfg.gpu.mbu) : 0;
  const stepFloor = Math.min(weightRead, cfg.software.request_overhead_s);
  const requestDelay = cfg.software.request_overhead_s - stepFloor;

  function route(req, index) {
    if (cfg.routing === "round-robin") {
      // Visit each independent node group before using its next local replica.
      const group = index % nodes.length;
      const replica = Math.floor(index / nodes.length) % perNode;
      return inst[group * perNode + replica];
    }
    if (cfg.routing === "sticky" && req.conv != null) return inst[req.conv % inst.length];
    return inst[Math.floor(r() * inst.length)];
  }

  const keys = ["gpu", "cpu", ...lower.map((t) => t.key), "miss"];
  const tokens = Object.fromEntries(keys.map((k) => [k, 0]));
  const ttfts = [], e2es = [], waits = [], prefills = [], transfers = [], steps = [];
  let promptTokens = 0, outputTokens = 0, busy = 0, decodeGpuSeconds = 0, counted = 0, slowRequests = 0;
  const batches = [], accepted = [], speedups = [];
  let busyUncached = 0, busyCached = 0, cachedTokens = 0; // prefill time by what it served, for the price sheet
  let busyLoads = 0; // of busyCached: loading blocks (the rest is the cached tokens' share of vLLM's overhead)
  const reqs = workload.requests;
  const startTime = cfg.durationMin != null || workload.durationS != null ? 0 : reqs[0]?.t ?? 0;
  const t0 = startTime + (cfg.warmupS ?? 0);
  let firstGpuEvictionS = null, firstDiskEvictionS = null, firstSharedEvictionS = null;
  const firstPayloadReadS = { disk: null, shared: null };
  let sharedStored = 0, sharedArea = 0, sharedLastT = t0, lastCompletion = t0;
  const events = new Events();
  const durationS = cfg.durationMin != null ? cfg.durationMin * 60 : workload.durationS;
  const noteShared = (now) => {
    if (!shared || now <= t0) return;
    const clamped = durationS != null ? Math.min(now, durationS) : now;
    if (clamped > sharedLastT) { sharedArea += shared.used * (clamped - sharedLastT); sharedLastT = clamped; }
  };
  const nextTurn = new Map();
  const lastTurn = new Map();
  let routingIndex = 0;
  class CapacityError extends Error {
    constructor(tokens, message) { super(message); this.tokens = tokens; }
  }

  // A single FIFO payload lane for each private disk and each node's NIC. A peer
  // transfer occupies both endpoints concurrently. Independent disks run in parallel.
  function reserve(time, seconds, lanes) {
    const end = Math.max(time, ...lanes.map(lane => lane.free)) + seconds;
    for (const lane of lanes) lane.free = end;
    return end;
  }

  const profile = (batch, ctx) => {
    const model = dec.model ?? m;
    const plain = decodeStep(model, dec.gpu, batch, ctx, decTp, { ...decodeOpts, spec: null });
    const spec = decodeOpts.spec ? decodeStep(model, dec.gpu, batch, ctx, decTp, decodeOpts) : plain;
    const best = (spec.step + stepHops) / spec.tokens < plain.step + stepHops ? spec : plain;
    const step = best.step + stepHops;
    return { step, perToken: step / best.tokens, tokens: best.tokens, plain: plain.step + stepHops };
  };
  const pool = dec && decodePool(events, { ...dec, ranks: decodeEp ? decodeEp / decTp : 1 }, profile);

  // Last read or write of each block, kept in strict oldest-first Map insertion order for O(1) lifetime expiration
  // across every tier (so a GPU-hot block at an LRU's head never blocks expiring stale entries behind it).
  const lifetime = cfg.cacheLifetimeS > 0 ? cfg.cacheLifetimeS : null;
  const lastUse = lifetime ? new Map() : null;
  const touchUse = (id, now) => { if (!lifetime) return; lastUse.delete(id); lastUse.set(id, now); };
  const expired = (id, now) => now - (lastUse.get(id) ?? -Infinity) > lifetime;
  let lastExpiry = 0, lastSharedExpiry = 0;
  function expireAll(now) {
    noteShared(now);
    for (const [id, ts] of lastUse) {
      if (now - ts <= lifetime) break;
      lastUse.delete(id);
      for (const instance of inst) instance.gpu.delete(id);
      for (const n of nodes) { n.cpu.delete(id); n.disk?.delete(id); }
      shared?.delete(id);
    }
  }
  function expireShared(now) {
    noteShared(now);
    for (const [id, ts] of sharedLastUse) {
      if (now - ts <= sharedLifetime) break;
      shared.delete(id);
    }
  }

  let lastPoll = 0;
  function advance(time) {
    if (lifetime && time - lastExpiry >= 1) { lastExpiry = time; expireAll(time); }
    if (shared && sharedLifetime && time - lastSharedExpiry >= 1) { lastSharedExpiry = time; expireShared(time); }
    // The LMCache controller polls each second. Once all pools are below
    // watermark, jumping over a long idle interval has no further effect.
    const end = Math.floor(time);
    while (lastPoll < end) {
      lastPoll++;
      let evicted = false;
      for (const n of nodes) {
        const before = n.cpu.evictions + (n.disk?.evictions ?? 0);
        n.cpu.sweep();
        n.disk?.sweep();
        if (firstDiskEvictionS == null && n.disk?.evictions) firstDiskEvictionS = lastPoll;
        evicted ||= before !== n.cpu.evictions + (n.disk?.evictions ?? 0);
      }
      if (!evicted) { lastPoll = end; break; }
    }
  }

  // Continuous prefill batching on one instance: a step starts when the previous one leaves the first pipeline stage.
  function enqueue(here, job, now) {
    here.waiting.push(job);
    // Priority 3 lets requests ready at the same instant join this step, as one scheduler pass would.
    if (!here.stepping) { here.stepping = true; events.at(now, 3, later => runStep(here, later)); }
  }
  function runStep(here, now) {
    let budget = batchTokens, tokens = 0, work = 0;
    const finishing = [];
    while (here.waiting.length && budget > 0) {
      const job = here.waiting[0];
      const take = Math.min(Math.max(1, job.n - job.pos), budget);
      job.first ??= now;
      work += T.flops(Math.min(job.n, job.pos + take)) - T.flops(Math.min(job.n, job.pos));
      job.pos += take; budget -= take; tokens += take;
      if (job.pos < job.n) break; // the rest of this prompt goes in the next step
      here.waiting.shift();
      work += job.perToken;
      finishing.push(job);
    }
    const seconds = stepFloor + work + T.hops(tokens);
    if (now >= t0) busy += seconds / pp;
    for (const job of finishing) events.at(now + seconds, 0, () => job.done(job.first, now + seconds));
    events.at(now + seconds / pp, 3, later => {
      if (here.waiting.length) runStep(here, later);
      else here.stepping = false;
    });
  }

  function arrive(template, time) {
    const req = time === template.t ? template : { ...template, t: time };
    const peak = req.n + (req.out ?? 0);
    if (cfg.physicalKvGiB != null && perSeqBytes(m, dec ? req.n : peak) > cfg.physicalKvGiB * GIB)
      throw new CapacityError(dec ? req.n : peak, `A ${dec ? req.n : peak}-token sequence exceeds the prefill instance's ${cfg.physicalKvGiB.toFixed(2)} GiB of active KV memory. Reduce the context or choose a larger GPU layout.`);
    if (dec?.batchCap && dec.batchCap(peak) < 1)
      throw new CapacityError(peak, `A ${peak}-token sequence exceeds the decode instance's KV memory at batch 1. Reduce the context or choose a larger decode GPU layout.`);
    const here = route(req, routingIndex++);
    const begin = now => serve(req, template, here, now);
    if (batchTokens) begin(time); // the batch scheduler queues it after its cache loads
    else if (here.running) here.queue.push(begin);
    else { here.running = true; begin(time); }
  }

  function serve(req, template, here, start) {
    const node = nodes[here.node];
    const ids = req.n < (cfg.minCacheTokens ?? 0) ? [] : blocksOf(req, B);
    const k = Object.fromEntries(keys.map((kk) => [kk, 0]));
    const where = [];
    for (const id of ids) {
      if (lifetime && expired(id, start)) break;
      if (shared && sharedLifetime && sharedExpired(id, start)) { noteShared(start); shared.delete(id); }
      if (here.gpu.has(id)) { where.push("gpu"); continue; }
      if (node.cpu.has(id)) { where.push("cpu"); continue; }
      const t = lower.find((tt) => tt.has(id, node));
      if (!t) break;
      where.push(t.key);
    }
    // A stateful prefix can only resume at a checkpoint still resident in a
    // usable tier. A nearer tier may have the global KV without the window.
    const localPools = { gpu: here.gpu, cpu: node.cpu, disk: node.disk, shared };
    if (m.read_fixed_bytes) {
      while (where.length) {
        const id = ids[where.length - 1];
        const endpoint = keys.find(key => key === "p2p"
          ? holders.has(id) && nodes.some(n => n !== node && n.cpu.bytes(id) > backingBlock)
          : localPools[key]?.bytes(id) > (key === "gpu" ? blockBytes : backingBlock));
        if (endpoint) { where[where.length - 1] = endpoint; break; }
        where.pop();
      }
    }
    // Cost-aware load guard (opt-in; LMCache and vLLM load every hit by default): the part of the hit that is slower
    // to load than to recompute, current lane backlog included, is recomputed instead.
    if (cfg.skipSlowLoads && where.length) {
      const cut = where.findIndex(w => w !== "gpu" && w !== "cpu");
      if (cut >= 0) {
        // Seconds to load the lower-tier blocks of where[cut, len), current lane backlog included.
        const estimate = (len) => {
          let seconds = 0;
          for (const tier of lower) {
            let cnt = 0, head = false;
            for (let j = cut; j < len; j++) {
              if (where[j] !== tier.key) continue;
              cnt++;
              head ||= tier.key === "shared" && s3 && !node.knownObjects.has(ids[j]);
            }
            if (!cnt) continue;
            const bytes = cnt * backingBlock + (where[len - 1] === tier.key ? externalState : 0);
            const bw = variantBandwidth(tier.v) * GIB * pp * linkRails(tier.v);
            const lane = tier.key === "disk" ? node.diskLane : tier.key === "shared" ? node.ethRx : rxLane(node, tier.v, here.replica);
            const xferS = tier.v.gpu_direct ? bytes / bw
              : cfg.software.cpu_load_latency_s + stagedTransferSeconds(bytes, bw, T.hostBytesPerS, tier.v);
            seconds += (head ? cfg.shared.latency_s ?? 0 : 0) + Math.max(0, lane.free - start) + (tier.v.latency_s ?? 0) + xferS;
          }
          return seconds;
        };
        const usable = (len) => Math.max(0, len * B - (len > 0 ? m.prefix_replay_tokens ?? 0 : 0));
        // The hit may end where the source tier changes: vLLM's per-tier load filter and load cap recompute what lies
        // beyond, keeping the faster tiers' prefix. A stateful prefix needs a checkpoint at its end, so it is all or
        // nothing below the GPU and CPU tiers.
        const ends = [cut, where.length];
        if (!m.read_fixed_bytes) for (let j = cut + 1; j < where.length; j++) if (where[j] !== where[j - 1]) ends.push(j);
        let best = where.length, bestS = Infinity;
        for (const len of ends) {
          const total = estimate(len) + T.ttft(req.n, usable(len), 0);
          if (total < bestS || (total === bestS && len > best)) { best = len; bestS = total; }
        }
        if (best < where.length) {
          where.length = best;
          if (m.read_fixed_bytes) {
            while (where.length) {
              const id = ids[where.length - 1];
              const key = where[where.length - 1];
              if (localPools[key]?.bytes(id) > (key === "gpu" ? blockBytes : backingBlock)) break;
              where.pop();
            }
          }
        }
      }
    }
    // MP touches a batch in reverse so an earlier prefix survives its tail.
    const peerBytes = new Map();
    for (let j = where.length - 1; j >= 0; j--) {
      const key = where[j];
      k[key]++;
      if (key === "p2p") {
        const owner = nodes.find(n => n !== node && n.cpu.bytes(ids[j]) >=
          (j === where.length - 1 && m.read_fixed_bytes ? backingBlock + externalState : backingBlock));
        owner.cpu.touch(ids[j]);
        peerBytes.set(owner, (peerBytes.get(owner) ?? 0) + backingBlock +
          (j === where.length - 1 ? externalState : 0));
      } else {
        localPools[key].touch(ids[j]);
        if (key === "shared") touchShared(ids[j], start);
      }
    }
    const h = where.length * B;
    const last = where.at(-1);
    // A GPU hit reuses resident KV in place and restores zero bytes.
    const readBytes = key => key === "gpu" ? 0 : k[key] * backingBlock + (last === key ? externalState : 0);
    const measured = req.t >= t0;
    if (measured) {
      if (node.disk) diskReadBytes += readBytes("disk");
      if (shared) sharedReadBytes += readBytes("shared");
      if (s3) operations.get += ranks * (k.shared + Number(last === "shared" && !!m.read_fixed_bytes));
    }
    // S3 size metadata is local to each adapter and primed by a PUT or HEAD.
    // A prefix directory / Bloom filter avoids stalling local hits or brand-new prompt misses on S3 404s:
    // HEAD probes run only for objects actually present in S3 that are not already satisfied by a higher-priority tier.
    let headNeeded = false;
    const metadata = [];
    if (s3) {
      for (let j = 0; j < where.length; j++) {
        const id = ids[j];
        if (where[j] !== "shared") continue;
        if (!node.knownObjects.has(id) && shared.has(id)) {
          headNeeded = true;
          if (measured) operations.head += ranks;
          metadata.push([node.knownObjects, id]);
        }
        if (node.knownWindows && j === where.length - 1 && !node.knownWindows.has(id) && shared.bytes(id) > backingBlock) {
          headNeeded = true;
          if (measured) operations.head += ranks;
          metadata.push([node.knownWindows, id]);
        }
      }
    }
    // Bounded replay restores global KV and recomputes the hit's trailing
    // window. Tier counts describe the global KV source even for those tokens.
    const usable = Math.max(0, h - (m.prefix_replay_tokens ?? 0));
    const out = req.out ?? 0;
    const hostBw = T.hostBytesPerS;
    const phases = [];
    if (k.cpu) {
      const cpuBytes = readBytes("cpu");
      phases.push({
        tier: "cpu",
        latency: cfg.software.cpu_load_latency_s,
        payloads: cpuBytes > 0 && Number.isFinite(hostBw) ? [{ seconds: cpuBytes / hostBw, lanes: [here.hostLane] }] : [],
        hostLatency: 0,
        hostSeconds: 0,
      });
    }
    for (const tier of lower) {
      if (!k[tier.key]) continue;
      const bytes = readBytes(tier.key);
      const bw = variantBandwidth(tier.v) * GIB * pp * linkRails(tier.v);
      const payloads = tier.key === "p2p"
        ? [...peerBytes].map(([owner, size]) => ({ seconds: size / bw, lanes: [txLane(owner, tier.v), rxLane(node, tier.v, here.replica)] }))
        : [{ seconds: bytes / bw, lanes: [tier.key === "disk" ? node.diskLane : node.ethRx] }];
      let hostLatency = 0, hostSeconds = 0;
      if (!tier.v.gpu_direct) {
        hostLatency = cfg.software.cpu_load_latency_s;
        if (bytes > 0 && Number.isFinite(hostBw)) hostSeconds = Math.max(0, stagedTransferSeconds(bytes, bw, hostBw, tier.v) - bytes / bw);
      }
      phases.push({ tier: tier.key, latency: tier.v.latency_s ?? 0, payloads, hostLatency, hostSeconds });
    }
    const afterHead = start + (headNeeded ? cfg.shared.latency_s ?? 0 : 0);
    const loadPhase = (index, time) => {
      if (index === phases.length) { prefillAt(time); return; }
      const phase = phases[index];
      events.at(time + (phase.latency ?? 0), 1, ready => {
        const move = (part, time) => {
          const payload = phase.payloads?.[part];
          if (!payload) {
            events.at(time + (phase.hostLatency ?? 0), 1, hostReady => {
              if (phase.hostSeconds > 0) {
                const hostEnd = reserve(hostReady, phase.hostSeconds, [here.hostLane]);
                events.at(hostEnd, 1, now => loadPhase(index + 1, now));
              } else {
                loadPhase(index + 1, hostReady);
              }
            });
            return;
          }
          if (measured && Object.hasOwn(firstPayloadReadS, phase.tier)) {
            const begin = Math.max(time, ...payload.lanes.map(lane => lane.free));
            firstPayloadReadS[phase.tier] = Math.min(firstPayloadReadS[phase.tier] ?? Infinity, begin);
          }
          const end = reserve(time, payload.seconds, payload.lanes);
          events.at(end, 1, now => move(part + 1, now));
        };
        move(0, ready);
      });
    };
    events.at(afterHead, 1, now => {
      for (const [known, id] of metadata) known.add(id);
      loadPhase(0, now);
    });

    // The one-at-a-time compute, used for the price sheet's split of prefill time in either mode.
    const isolated = T.ttft(req.n, usable, 0);
    let loads = 0, compute = 0, prefill = 0, end = 0, waited = 0;
    function prefillAt(ready) {
      loads = ready - start;
      if (batchTokens) {
        const job = { pos: usable, n: req.n, perToken: cfg.software.per_token_overhead_s * req.n, first: null,
          done: (first, finish) => { waited = first - ready - requestDelay; prefilled(ready, finish); } };
        events.at(ready + requestDelay, 2, now => enqueue(here, job, now));
        return;
      }
      waited = start - req.t;
      events.at(ready + isolated / pp, 2, now => {
        if (here.head < here.queue.length) {
          const next = here.queue[here.head];
          here.queue[here.head++] = null;
          if (here.head === here.queue.length) { here.queue = []; here.head = 0; }
          next(now);
        } else here.running = false;
      });
      prefilled(ready, ready + isolated);
    }

    function prefilled(ready, finish) {
      compute = finish - ready;
      prefill = loads + compute;
      end = finish;
      events.at(end, 0, now => publish(0, ids.length, now));
      events.at(end, 1, now => {
        const transfer = out > 0 ? xfer(req.n) : 0;
        if (transfer) {
          const latency = cfg.transfer.latency_s ?? 0;
          events.at(now + latency, 1, ready => {
            const done = reserve(ready, transfer - latency, [txLane(node, cfg.transfer, here.replica)]);
            events.at(done, 1, beginDecode);
          });
        } else beginDecode(now);
      });

      function beginDecode(time) {
        if (dec && out > 0) {
          pool.submit(time, { measured, left: out, ctx: req.n + out / 2, peak: req.n + out,
            done: (finish, task) => complete(time, task.first, finish, task) });
        } else {
          // Prefill-only mode reports prefill TTFT. Where a GPU decode profile
          // exists, estimate the local reply duration for dependent turns.
          const localStep = !dec && out && cfg.gpu.mem_bw_tb_s && cfg.gpu.mbu
            ? decodeStep(m, cfg.gpu, 1, req.n + out / 2, decTp, decodeOpts ?? {}).step + stepHops : 0;
          const finish = time + out * localStep;
          events.at(finish, 0, () => complete(time, time, finish, null));
        }
      }

      function complete(decodeStart, first, finish, task) {
        publish(ids.length, allIds.length, finish);
        const next = nextTurn.get(template);
        if (next) {
          const time = finish + (next.t - template.t);
          if (time < durationS) events.at(time, 2, now => arrive(next, now));
        }
        if (!measured) return;
        lastCompletion = Math.max(lastCompletion, finish);
        const tr = decodeStart - end, ttft = first - req.t;
        observeRequest?.(req, ttft, k, B, loads, waited, tr, k.p2p ? readBytes("p2p") : 0,
          readBytes);
        ttfts.push(ttft); waits.push(waited); prefills.push(prefill); transfers.push(tr);
        e2es.push(finish - req.t);
        if (task) {
          steps.push(task.decodeSeconds / out);
          batches.push(task.batchSteps / task.steps);
          accepted.push(task.accepted / task.steps);
          if (decodeOpts.spec?.draft) speedups.push(task.plain / task.decodeSeconds);
          if (dec.speed && profile(1, task.ctx).perToken > 1 / dec.speed) slowRequests++;
          decodeGpuSeconds += task.gpuSeconds * decTp;
        }
        if (!batchTokens) busy += loads + isolated / pp;
        // In batched prefill, requestDelay is host/scheduling delay off the GPU; only stepFloor + per-token work
        // occupies GPU time, so split prefill GPU cost on GPU-occupying overhead rather than full request_overhead_s.
        const gpuOver = batchTokens ? stepFloor + cfg.software.per_token_overhead_s * req.n : T.overhead(req.n);
        const gpuIsolated = batchTokens ? isolated - requestDelay : isolated;
        const hh = Math.min(h, req.n);
        const rebuild = usable < h ? isolated - T.ttft(req.n, h, 0) : 0;
        // Batched prefill loads blocks while other requests compute (LMCache MP), so load time costs no GPU time.
        const gpuLoads = batchTokens ? 0 : loads;
        busyCached += gpuLoads + rebuild + (gpuOver * hh) / req.n;
        busyLoads += gpuLoads;
        busyUncached += Math.max(0, gpuIsolated - rebuild - (gpuOver * hh) / req.n);
        cachedTokens += hh;
        promptTokens += req.n; outputTokens += out;
        for (const kk of keys) if (kk !== "miss") tokens[kk] += k[kk] * B;
        tokens.miss += req.n - h;
        counted++;
      }
    }
    // A disaggregated prefill worker never generates the reply. A trace
    // supplies only prompt hashes, so it cannot identify generated chunks.
    const allIds = !ids.length || req.hashes || (dec && !cfg.saveDecode) ? ids : blocksOf({ ...req, n: req.n + (req.out ?? 0) }, B);
    if (lifetime) for (const id of allIds) touchUse(id, start);
    const gpuCount = dec ? ids.length : allIds.length;
    const externalCount = cfg.saveDecode ? allIds.length : ids.length;
    const sysEnd = req.sys > 0 ? Math.floor(req.sys / B) - 1 : -1;
    function publish(from, to, time) {
      const diskWrites = [], sharedWrites = [];
      let diskWriteBytes = 0, sharedWriteNow = 0;
      for (let j = to - 1; j >= from; j--) {
        const id = allIds[j];
        // Hybrid models checkpoint recurrent state at the prompt end, matched prefix end, and shared system-prompt boundary.
        const checkpoint = j === ids.length - 1 || j === where.length - 1 || (sysEnd >= 0 && j === sysEnd);
        if (j < gpuCount) here.gpu.put(id, blockBytes + (checkpoint || j === gpuCount - 1 ? m.read_fixed_bytes : 0));
        if (j >= externalCount) continue;
        const state = checkpoint || j === externalCount - 1 ? externalState : 0;
        const bytes = backingBlock + state;
        if (j < where.length) {
          // Default MP prefetch retains an L2 hit in L1. A GPU hit alone
          // doesn't refresh disk recency or create a redundant S3 write.
          if (where[j] !== "gpu" && where[j] !== "cpu") node.cpu.put(id, bytes);
          continue;
        }
        // A recomputed suffix may still have global KV in L1, particularly
        // when a stateful prefix lacks a usable window checkpoint.
        const prior = node.cpu.bytes(id);
        if (prior >= bytes) continue;
        node.cpu.put(id, bytes);
        // A recomputed chunk goes to every backing store again. The filesystem connectors skip a file that already
        // exists (vLLM's access() check, LMCache fs); LMCache's S3 adapter PUTs without a HEAD, so it is uploaded
        // and billed again.
        if (node.disk && node.disk.bytes(id) < bytes) {
          diskWrites.push({ id, bytes });
          diskWriteBytes += bytes - prior;
        }
        if (shared) {
          sharedWrites.push({ id, bytes, state, prior });
          sharedWriteNow += bytes - prior;
        }
      }
      if (firstGpuEvictionS == null && here.gpu.evictions) firstGpuEvictionS = time;

      // The cache batch becomes readable on each backing tier after its own
      // asynchronous write. SSD writes use write bandwidth; S3 uploads use the frontend TX lane.
      const writeTo = (variant, lane, writeBytes, commit) => {
        events.at(time + (variant.latency_s ?? 0), 1, ready => {
          const done = reserve(ready, writeBytes / (writeBandwidth(variant) * GIB * pp), [lane]);
          events.at(done, 0, commit);
        });
      };
      if (diskWrites.length) writeTo(cfg.disk, node.diskLane, diskWriteBytes, done => {
        const before = node.disk.added;
        for (const { id, bytes } of diskWrites) node.disk.put(id, bytes);
        if (measured) node.diskWritten += node.disk.added - before;
        if (firstDiskEvictionS == null && node.disk.evictions) firstDiskEvictionS = done;
      });
      if (sharedWrites.length) writeTo(cfg.shared, node.ethTx, sharedWriteNow, done => {
        noteShared(done);
        const before = shared.added;
        for (const { id, bytes, state, prior } of sharedWrites) {
          shared.put(id, bytes);
          touchShared(id, done);
          node.knownObjects?.add(id);
          if (state) node.knownWindows?.add(id);
          if (measured && s3) operations.put += ranks * (Number(!prior) + Number(!!state));
        }
        if (firstSharedEvictionS == null && shared.evictions) firstSharedEvictionS = done;
        if (measured) {
          sharedWriteBytes += sharedWriteNow;
          sharedStored += shared.added - before;
        }
      });
    }
  }

  for (const req of reqs) {
    if (durationS != null && req.t >= durationS) continue;
    if (workload.durationS != null && req.followUp && lastTurn.has(req.conv))
      nextTurn.set(lastTurn.get(req.conv), req);
    else events.at(req.t, 2, time => arrive(req, time));
    if (workload.durationS != null) lastTurn.set(req.conv, req);
  }
  if (observeMinuteEnd) {
    const end = durationS != null ? Math.ceil(durationS / 60) : Math.floor((reqs.at(-1)?.t ?? 0) / 60) + 1;
    for (let bucket = Math.floor(startTime / 60); bucket < end; bucket++) {
      events.at((bucket + 1) * 60, 1, () => {
        const snapshot = {};
        for (const [key, copies] of Object.entries(pools)) {
          const capacity = copies.reduce((sum, cache) => sum + cache.cap, 0);
          if (capacity === 0) continue;
          snapshot[key] = {
            usedGiB: copies.reduce((sum, cache) => sum + cache.used, 0) / GIB,
            capacityGiB: Number.isFinite(capacity) ? capacity / GIB : null,
          };
        }
        observeMinuteEnd(bucket, snapshot);
      });
    }
  }

  if (durationS != null) events.at(durationS, 1, () => {});
  try { events.drain(advance); }
  catch (error) {
    if (error instanceof CapacityError) return { error: error.message, requiredContext: error.tokens };
    if (error instanceof RangeError) return { error: error.message };
    throw error;
  }
  const scaleUp = cfg.costScale ?? 1;
  const windowS = durationS != null ? Math.max(0, durationS - t0)
    : counted ? Math.max(1, lastCompletion - t0) : 0;
  noteShared(t0 + windowS);
  const windowH = windowS / 3600;
  const realNodes = cl.nodes * scaleUp;
  const prefillGpus = realNodes * cl.gpusPerNode;
  const usdH = (g) => g?.usd_per_hour ?? 0;
  // provisioned GPUs are paid for the whole window, busy or not; decode in whole TP instances
  // decode instances span pp nodes too and keep pp batches in flight, so a step's GPU time is tp, not tp x pp
  // (decode.pooled: sized to the load in GPU fractions, as a provider pooling many clusters' decode would)
  const decodeNeed = (decodeGpuSeconds * scaleUp) / Math.max(windowS, 1);
  const decodeWidth = wideEp ?? cl.tp * pp; // whole decode instances
  const decodeGpus = !dec ? 0 : dec.pooled ? decodeNeed : Math.ceil(pool.peakGroups * scaleUp * decTp / decodeWidth) * decodeWidth;
  const prefillUsd = prefillGpus * usdH(cfg.gpu) * windowH;
  const decodeUsd = decodeGpus * usdH(dec?.gpu) * windowH;
  const baseHoldH = lifetime ? Math.min(cfg.retentionH ?? Infinity, lifetime / 3600) : cfg.retentionH;
  const storageOf = (v, writtenBytes, copies, scale, holdH = baseHoldH, avgUsedBytes = null) => {
    if (!v) return { storage: 0, fixed: 0, provisioned: 0 };
    const price = v.usd_per_gib_month ?? 0;
    let gibHeld, isProvisioned = false;
    if (v.bill === "provisioned" && v.capacity_gib != null) { gibHeld = v.capacity_gib * copies; isProvisioned = true; }
    else if (holdH == null && avgUsedBytes != null) {
      gibHeld = (avgUsedBytes / GIB) * scale;
      if (v.capacity_gib != null) gibHeld = Math.min(gibHeld, v.capacity_gib * copies);
    } else { // bytes stored: what is written per hour, kept for the effective retention time (capped by capacity)
      const effH = holdH ?? Math.max(windowH, 1);
      const writtenGiB = writtenBytes / GIB * scale;
      gibHeld = (writtenGiB / Math.max(windowH, 1e-9)) * effH;
      if (v.capacity_gib != null) gibHeld = Math.min(gibHeld, v.capacity_gib * copies);
    }
    const storage = gibHeld * price * (windowH / 730);
    return { storage, fixed: (v.usd_per_hour ?? 0) * copies * windowH, provisioned: isProvisioned ? storage : 0 };
  };
  // per-GB fees (S3 Express): what was written after the warm-up, and what was read
  const fees = (v, writtenBytes, readBytes) => (v ? ((v.usd_per_gib_written ?? 0) * writtenBytes + (v.usd_per_gib_read ?? 0) * readBytes) / GIB * scaleUp : 0);
  const diskCost = { storage: 0, fixed: 0, provisioned: 0 };
  for (const node of nodes) {
    const cost = storageOf(cfg.disk, node.diskWritten, pp * scaleUp, scaleUp);
    diskCost.storage += cost.storage;
    diskCost.fixed += cost.fixed;
    diskCost.provisioned += cost.provisioned;
  }
  diskCost.storage += fees(cfg.disk, nodes.reduce((sum, n) => sum + n.diskWritten, 0), diskReadBytes);
  const sharedHoldH = sharedLifetime ? Math.min(baseHoldH ?? Infinity, sharedLifetime / 3600) : baseHoldH;
  const sharedCost = storageOf(cfg.shared, sharedStored, 1, scaleUp, sharedHoldH, windowS > 0 ? sharedArea / windowS : 0);
  sharedCost.storage += fees(cfg.shared, sharedWriteBytes, sharedReadBytes);
  const requestUsd = ((cfg.shared?.usd_per_1k_put ?? 0) * operations.put +
    (cfg.shared?.usd_per_1k_get ?? 0) * (operations.head + operations.get)) * scaleUp / 1000;
  sharedCost.storage += requestUsd;
  const p2pFixed = cfg.p2p ? (cfg.p2p.usd_per_hour ?? 0) * realNodes * windowH : 0;
  const storageUsd = diskCost.storage + sharedCost.storage;
  const fixedUsd = diskCost.fixed + sharedCost.fixed + p2pFixed;
  const provisionedTierUsd = diskCost.provisioned + sharedCost.provisioned + fixedUsd;
  const total = prefillUsd + decodeUsd + storageUsd + fixedUsd;
  const stats = (xs) => {
    const so = [...xs].sort((a, b) => a - b);
    return { mean: xs.reduce((a, b) => a + b, 0) / (xs.length || 1), p50: pct(so, 0.5), p90: pct(so, 0.9), p99: pct(so, 0.99) };
  };
  const share = Object.fromEntries(Object.entries(tokens).map(([kk, v]) => [kk, promptTokens ? v / promptTokens : 0]));
  const priced = usdH(cfg.gpu) > 0 && (!dec || usdH(dec.gpu) > 0) && [cfg.disk, cfg.shared].every((v) => !v || v.usd_per_gib_month != null);
  return {
    s3Requests: Object.fromEntries(Object.entries(operations).map(([k, v]) => [k, v * scaleUp])), requestUsd,
    requests: counted, promptTokens, outputTokens, tokens, share, order: keys, blockTokens: B, windowS, firstPayloadReadS,
    ttft: stats(ttfts), e2e: stats(e2es), wait: stats(waits).mean, prefill: stats(prefills).mean, transfer: stats(transfers).mean,
    tpot: steps.length ? stats(steps).mean : null,
    // decode batch per request (mean), and the share of requests below the speed target even at batch 1
    decodeBatch: batches.length ? stats(batches).mean : null, belowSpeed: counted && dec?.speed ? slowRequests / counted : 0,
    // speculative decoding: tokens accepted per step, and the speed-up over no drafter at the same batch (1 where it was
    // switched off)
    acceptedTokens: accepted.length ? stats(accepted).mean : null, specSpeedup: speedups.length ? stats(speedups).mean : null,
    utilization: windowS ? busy / windowS / inst.length : 0, // above 1, requests queue without bound
    decodeUtilization: decodeGpus > 0 && windowS > 0 ? Math.min(1, decodeNeed / decodeGpus) : 0,
    instances: nodes.length * replicasPerNode * scaleUp, prefillGpus, decodeGpus, priced,
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
      // what a cached token's price is made of, $ per 1M cached tokens
      const busy = busyUncached + busyCached;
      const parts = tok.cachedInput ? Object.fromEntries(Object.entries({
        loads: busy ? (prefillUsd * busyLoads) / busy : 0, overhead: busy ? (prefillUsd * (busyCached - busyLoads)) / busy : 0, storage: storageUsd, fixed: fixedUsd,
      }).map(([k, v]) => [k, (v / scaleUp / tok.cachedInput) * 1e6])) : null;
      // the prefill-GPU and fixed-provisioned-tier parts of the cached-input price (both scale with cluster busy share);
      // usage-billed S3 storage/requests/transfers do not scale with GPU load
      const cachedPrefill = tok.cachedInput ? ((prefillUsd * (1 - split)) / scaleUp / tok.cachedInput) * 1e6 : null;
      const cachedFixed = tok.cachedInput ? (provisionedTierUsd / scaleUp / tok.cachedInput) * 1e6 : null;
      return { usd, tokens: tok, input: per1M("input"), cachedInput: per1M("cachedInput"), cachedPrefill, cachedFixed, output: per1M("output"), cachedParts: parts };
    })(),
    firstGpuEvictionS: durationS != null && firstGpuEvictionS >= durationS ? null : firstGpuEvictionS,
    firstDiskEvictionS: durationS != null && firstDiskEvictionS >= durationS ? null : firstDiskEvictionS,
    firstSharedEvictionS: durationS != null && firstSharedEvictionS >= durationS ? null : firstSharedEvictionS,
    evictions: { gpu: inst.reduce((a, i) => a + i.gpu.evictions, 0), cpu: nodes.reduce((a, n) => a + n.cpu.evictions, 0), ...(shared ? { shared: shared.evictions } : {}) },
  };
}
