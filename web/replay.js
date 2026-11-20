// One measured hour, from a cold start or after sim.warmupMin of warm-up. Each one-minute bucket contains the requests
// that arrived in that minute; times in the result count from the start of measurement, so an event during the
// warm-up has a negative time.
import { fitFor, runScenario } from "./scenario.js";

function readerStats(samples) {
  const stats = {};
  // Median uses the midpoint for even samples.
  const median = sorted => (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
  for (const [tier, { times, bytes }] of Object.entries(samples)) {
    times.sort((a, b) => a - b);
    bytes.sort((a, b) => a - b);
    stats[tier] = { count: times.length, medianTTFT: median(times), medianMiB: median(bytes) / 2 ** 20 };
  }
  return stats;
}

// onProgress(fraction) reports the share of simulated minutes done (warm-up included), below 1 until the run returns.
export function replay(config, budgetFraction = config.sim.gpuBudgetFraction ?? 1, { onProgress } = {}) {
  const sim = config.sim;
  const gpu = config.gpus.find(g => g.id === sim.gpu);
  const layout = context => fitFor(config, sim.model, gpu, sim.attention, sim.precision, sim, context);
  let fit = layout(0);
  if (!fit) {
    const suggested = fitFor(config, sim.model, gpu);
    const advice = suggested
      ? `A layout that fits is ${suggested.pp} node${suggested.pp === 1 ? "" : "s"} with ${suggested.tp} GPU${suggested.tp === 1 ? "" : "s"} per node.`
      : "Choose a GPU with more memory or a smaller model.";
    throw new Error(`${config.models[sim.model].name} does not fit this cluster. ${advice}`);
  }
  for (;;) {
    const result = replayLayout(config, fit, budgetFraction, onProgress);
    if (!result.error) return result;
    const next = result.requiredContext && layout(result.requiredContext);
    if (!next || next.pp < fit.pp || (next.pp === fit.pp && next.tp <= fit.tp)) throw new Error(result.error);
    fit = next;
  }
}

function replayLayout(config, fit, budgetFraction, onProgress) {
  const warmupS = (config.sim.warmupMin ?? 0) * 60, skip = Math.round(warmupS / 60);
  const minutes = Array.from({ length: config.workload.durationMin - skip }, () => ({
    count: 0, seconds: 0, loads: 0, waits: 0, transfers: 0, backingTokens: 0, peerBytes: 0, promptTokens: 0, reads: {}, readerSeconds: {}, samples: {},
  }));
  const runSamples = {};
  const tiers = {};
  let initialUsedGiB = null;
  // Tier depth order (shallowest to deepest) for attributing a request's end-to-end TTFT to its bottleneck tier.
  const tierRank = ["gpu", "cpu", "disk", ...(config.sim.peerFirst ? ["p2p", "shared"] : ["shared", "p2p"])];
  function observeRequest(req, ttft, blocks, blockSize, loads, wait, transfer, peerBytes, readBytes) {
    const minute = minutes[Math.floor((req.t - warmupS) / 60)];
    minute.count++;
    minute.seconds += ttft;
    minute.peerBytes += peerBytes;
    minute.loads += loads;
    minute.waits += wait;
    minute.transfers += transfer;
    minute.promptTokens += req.n;
    minute.backingTokens += ((blocks.shared ?? 0) + (blocks.disk ?? 0)) * blockSize;
    // Attribute each request's TTFT to the deepest (slowest) tier it read so GPU median TTFT isn't polluted by S3 hits.
    let deepest = null;
    for (const tier of tierRank) if (blocks[tier] > 0) deepest = tier;
    if (deepest) {
      minute.reads[deepest] = (minute.reads[deepest] ?? 0) + 1;
      minute.readerSeconds[deepest] = (minute.readerSeconds[deepest] ?? 0) + ttft;
      const bytes = readBytes(deepest);
      for (const scope of [minute.samples, runSamples]) {
        const sample = scope[deepest] ??= { times: [], bytes: [] };
        sample.times.push(ttft);
        sample.bytes.push(bytes);
      }
    }
  }
  function observeMinuteEnd(index, snapshot) {
    onProgress?.(Math.min(0.99, (index + 1) / (config.workload.durationMin || 1)));
    const usedGiB = {};
    for (const [tier, state] of Object.entries(snapshot)) {
      tiers[tier] = state.capacityGiB;
      usedGiB[tier] = state.usedGiB;
    }
    if (index < skip) {
      if (index === skip - 1) initialUsedGiB = usedGiB;
      return;
    }
    minutes[index - skip].usedGiB = usedGiB;
  }
  const result = runScenario(config, { ...fit, gpuBudgetFraction: budgetFraction }, null, {
    maxSimNodes: config.sim.nodes, observeRequest, observeMinuteEnd,
  });
  if (result.error) return result;
  if (!result.requests) throw new Error("This workload produced no requests.");
  const { storage, fixed, prefill, decode } = result.usd;
  const perHour = usd => usd * 3600 / result.windowS;
  const shift = time => time == null ? null : time - warmupS;
  let resident = initialUsedGiB ? { ...initialUsedGiB } : Object.fromEntries(Object.keys(tiers).map(key => [key, 0]));
  return {
    readerStats: readerStats(runSamples), warmupMin: skip, initialUsedGiB: initialUsedGiB ?? Object.fromEntries(Object.keys(tiers).map(key => [key, 0])),
    firstPayloadReadS: Object.fromEntries(Object.entries(result.firstPayloadReadS).map(([key, time]) => [key, shift(time)])),
    layout: fit, instances: result.instances, gpuGiB: result.kvGiB, kvBytesPerToken: result.kvBytesPerToken, order: result.order,
    mean: result.ttft.mean, ttft: result.ttft, wait: result.wait, share: result.share, utilization: result.utilization,
    firstEviction: shift(result.firstGpuEvictionS), firstDiskEviction: shift(result.firstDiskEvictionS),
    windowBytes: result.windowBytes, replayTokens: result.replayTokens, blockTokens: result.blockTokens,
    s3Requests: result.s3Requests, requestCostPerHour: perHour(result.requestUsd),
    costPerHour: perHour(storage + fixed), tiers,
    // Serving GPUs: the provisioned prefill cluster and the decode GPUs its peak needs. Null when a GPU has no listed price.
    gpuCostPerHour: [config.sim.gpu, config.sim.decodeGpu ?? config.sim.gpu].every(id => config.gpus.find(g => g.id === id)?.usd_per_hour > 0)
      ? perHour(prefill + decode) : null,
    prefillCostPerHour: perHour(prefill), decodeCostPerHour: perHour(decode),
    // USD per 1M input, cached-input and output tokens (sim.js price sheet); null when a GPU has no listed price
    tokenPrices: result.priced ? { input: result.price.input, cachedInput: result.price.cachedInput, cachedPrefill: result.price.cachedPrefill,
      cachedFixed: result.price.cachedFixed, output: result.price.output } : null,
    prefillGpus: result.prefillGpus, decodeGpus: result.decodeGpus, requests: result.requests,
    minutes: minutes.map(m => {
      // An idle minute retains resident bytes and has no latency observation.
      resident = m.usedGiB ?? resident;
      return {
        usedGiB: resident, reads: m.reads, count: m.count, peerBytes: m.peerBytes, readerStats: readerStats(m.samples),
        readerTTFT: Object.fromEntries(Object.entries(m.reads).map(([tier, n]) => [tier, m.readerSeconds[tier] / n])),
        mean: m.count ? m.seconds / m.count : null, load: m.count ? m.loads / m.count : null,
        wait: m.count ? m.waits / m.count : null, transfer: m.count ? m.transfers / m.count : null,
        backing: m.promptTokens ? m.backingTokens / m.promptTokens : null,
      };
    }),
  };
}

// One row per arrival minute: request count, the TTFT components, the backing share and each tier's
// minute-end occupancy and reader count.
export function minutesCsv(result) {
  const tiers = result.order.filter(key => key in result.tiers || key === "p2p");
  const occupancy = tiers.filter(key => key in result.tiers);
  const header = ["minute", "requests", "mean_ttft_s", "queue_s", "cache_load_s", "decode_transfer_s", "backing_token_share",
    ...occupancy.map(key => `${key}_used_gib`), ...tiers.map(key => `${key}_readers`)];
  const cell = value => value == null || !Number.isFinite(value) ? "" : String(+value.toPrecision(6));
  const rows = result.minutes.map((m, i) => [i + 1, m.count, m.mean, m.wait, m.load, m.transfer, m.backing,
    ...occupancy.map(key => m.usedGiB[key]), ...tiers.map(key => m.reads[key] ?? 0)].map(cell).join(","));
  return [header.join(","), ...rows].join("\n") + "\n";
}
