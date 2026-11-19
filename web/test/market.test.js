import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { defaultConfig } from "../config.js";
import { marketRun, providerStats } from "../market.js";
import { decodeKvGiB, decodeStep, decodeStepS, decodeWeightBytes, expertCoverage, served, TP_SYNC_PER_LAYER_S, withTp } from "../model.js";
import { runScenario, specFor } from "../scenario.js";
import { GPUS, MODELS, SOFTWARE, SPEC_METHODS, VARIANTS } from "../presets.js";

const S1 = JSON.parse(readFileSync(new URL("../data/s1.json", import.meta.url)));
const OR = JSON.parse(readFileSync(new URL("../data/openrouter.json", import.meta.url)));
const config = () => defaultConfig(S1, VARIANTS, GPUS, SOFTWARE, MODELS);

test("provider stats: median and range per price, cached prices only where listed", () => {
  const st = providerStats([{ input: 1, cached: null, output: 4 }, { input: 3, cached: 0.3, output: 2 }, { input: 2, cached: 0.1, output: 9 }]);
  assert.deepEqual(st.input, { median: 2, min: 1, max: 3, n: 3 });
  assert.equal(st.cached.median, 0.2);
  assert.equal(st.output.max, 9);
});

test("the OpenRouter snapshot names a simulator model and lists paid endpoints for each", () => {
  const c = config();
  for (const [id, m] of Object.entries(OR.models)) {
    assert.ok(c.models[id], id);
    assert.ok(m.providers.length > 0 && m.providers.every((p) => p.input > 0 && p.output > 0), id);
  }
});

test("market reference run: 8 instances loaded to just under the target prefill utilization, every price finite", () => {
  const c = config();
  c.sim.gpu = "b200"; c.sim.decodeGpu = "b200";
  const r = marketRun(c, "gpt-oss-120b");
  assert.equal(r.instances, 8);
  assert.ok(r.utilization < 0.7 && r.utilization > 0.4, `${r.utilization}`);
  for (const k of ["input", "cachedInput", "output"]) assert.ok(Number.isFinite(r.price[k]) && r.price[k] > 0, k);
});

test("decode: TP adds a fixed all-reduce time per layer; a big batch can be bound by FLOPs; DP attention copies no KV", () => {
  const gpu = { mem_bw_tb_s: 1, mbu: 1, tflops_bf16: 1000, prefill_mfu: 1 };
  const m = { weights_bytes: 1e12, params: 1e9, kv_bytes_per_token: 0, state_bytes_per_prefix: 0, num_layers: 10 };
  assert.equal(decodeStepS(m, gpu, 1, 0, 1), 1); // 1 TB at 1 TB/s
  assert.ok(Math.abs(decodeStepS(m, gpu, 1, 0, 2) - (0.5 + 10 * TP_SYNC_PER_LAYER_S[2])) < 1e-12);
  assert.ok(Math.abs(decodeStepS(m, gpu, 1e6, 0, 1) - (2 * 1e9 * 1e6) / 1e15) < 1e-12); // 2 s of FLOPs > 1 s of memory
  const glm = MODELS["glm-5.3"];
  assert.equal(withTp(glm, 8, "tp").kv_bytes_per_token, 8 * glm.kv_bytes_per_token);
  assert.equal(withTp(glm, 8, "dp").kv_bytes_per_token, glm.kv_bytes_per_token);
});

test("MoE decode reads the dense part and only the experts the batch picked; dense models read everything", () => {
  const kimi = MODELS["kimi-k3"], gemma = MODELS["gemma-4-31b"];
  assert.equal(kimi.expert_pick, 16 / 896);
  assert.ok(Math.abs(expertCoverage(kimi, 1) - 16 / 896) < 1e-12);
  assert.ok(expertCoverage(kimi, 1024) > 0.99999);
  const one = decodeWeightBytes(kimi, 1) / kimi.weights_bytes;
  assert.ok(one > 0.05 && one < 0.1, `${one}`); // the dense part (~7%) and 16 experts per layer
  assert.ok(decodeWeightBytes(kimi, 1024) / kimi.weights_bytes > 0.9999);
  assert.equal(decodeWeightBytes(gemma, 1), gemma.weights_bytes);
  // every MoE preset: expert params from config.json near the card's total less the dense part
  for (const [id, m] of Object.entries(MODELS)) {
    if (!m.spec.experts) continue;
    assert.ok(m.expert_share > 0.9 && m.expert_share <= 1, `${id} ${m.expert_share}`);
    assert.ok(m.spec.experts.params < 1.06 * m.total_params, id);
  }
  assert.equal(MODELS["deepseek-v4.1-flash"].decode_params, 16e9); // the card's decode figure
});

test("market reference run holding a decode speed: the batch keeps each request at it, and costs more than a full batch", () => {
  const c = config();
  c.sim.gpu = "b200"; c.sim.decodeGpu = "b200";
  const full = marketRun(c, "glm-5.3", { steps: 4 });
  const held = marketRun(c, "glm-5.3", { steps: 4, speed: OR.models["glm-5.3"].speed.p50_tok_s });
  assert.equal(held.speedTarget, 75);
  assert.ok(1 / held.tpot >= 75 * 0.999, `${1 / held.tpot}`);
  assert.equal(held.belowSpeed, 0);
  assert.ok(held.decodeBatch < full.decodeBatch);
  assert.ok(held.price.output > 2 * full.price.output, `${held.price.output} vs ${full.price.output}`);
  // out of reach even at batch 1: requests run alone and are counted
  const fast = marketRun(c, "glm-5.3", { steps: 4, speed: 1e4 });
  assert.equal(fast.decodeBatch, 1);
  assert.equal(fast.belowSpeed, 1);
});

const B200 = GPUS.find((g) => g.id === "b200");
const glm8 = () => served(MODELS["glm-5.3"], { precision: "fp8", tp: 8, attention: "dp" });

test("decode without a drafter or wider experts is the step it was: the old formula, and experts over the TP GPUs alone", () => {
  const m = glm8();
  for (const b of [1, 15, 256, 1024]) {
    const old = (decodeWeightBytes(m, b) + b * (m.state_bytes_per_prefix + m.kv_bytes_per_token * 10000)) / (8 * B200.mem_bw_tb_s * 1e12 * B200.mbu);
    const compute = (2 * m.decode_params * b) / (B200.tflops_fp8 * 1e12 * B200.prefill_mfu * 8 * 0.47);
    const want = Math.max(old, compute) + m.num_layers * TP_SYNC_PER_LAYER_S[8];
    assert.ok(Math.abs(decodeStepS(m, B200, b, 10000, 8) / want - 1) < 1e-12, `${b}`);
    assert.equal(decodeStep(m, B200, b, 10000, 8, { ep: 8 }).step, decodeStepS(m, B200, b, 10000, 8));
    assert.equal(decodeStep(m, B200, b, 10000, 8, { spec: null }).perToken, decodeStepS(m, B200, b, 10000, 8));
  }
});

test("speculative decoding: tokens per step from the method, a speed-up that shrinks with the batch, drafters the checkpoint lacks are off", () => {
  const m = glm8(), dspark = SPEC_METHODS.find((x) => x.id === "dspark");
  const gain = (b) => decodeStep(m, B200, b, 10000, 8).step / decodeStep(m, B200, b, 10000, 8, { spec: dspark }).perToken;
  assert.equal(decodeStep(m, B200, 1, 10000, 8, { spec: dspark }).tokens, 3.72);
  assert.ok(gain(1) > 2 && gain(1) < 3.72, `${gain(1)}`); // checking 8 positions reads more experts (~22% vs 3%) but the same dense part
  assert.ok(gain(1024) < gain(1));
  assert.equal(specFor({ spec: "mtp" }, MODELS["kimi-k3"]).spec, null); // no MTP head in Kimi K3
  assert.equal(specFor({ spec: "mtp" }, MODELS["kimi-k3"]).unavailable.id, "mtp");
  assert.equal(specFor({ spec: "mtp" }, MODELS["glm-5.3"]).spec.tau, 1.875);
  assert.deepEqual([specFor({ spec: "dspark" }, MODELS["deepseek-v4.1-flash"]).spec.draft, specFor({ spec: "dspark" }, MODELS["deepseek-v4.1-flash"]).spec.layers], [4, 3]); // its own drafter: block 5, 3 layers
  assert.equal(specFor({ spec: "eagle3", specTau: 2 }, MODELS["kimi-k3"]).spec.tau, 2); // a separate drafter, any model
});

test("expert parallelism: fewer weights per GPU, more KV room, traffic over NVLink inside the domain and InfiniBand beyond", () => {
  const m = glm8();
  const at = (ep) => decodeStep(m, B200, 64, 10000, 8, { ep });
  assert.ok(at(32).memory < at(8).memory / 2);
  assert.ok(at(8).comm === 0 && at(32).comm > 0);
  assert.ok(decodeKvGiB(m, B200, SOFTWARE, 8, 1, 32) > decodeKvGiB(m, B200, SOFTWARE, 8, 1));
  const gb = GPUS.find((g) => g.id === "gb200");
  const nv = decodeStep(m, gb, 64, 10000, 8, { ep: 32 }).comm, ib = decodeStep(m, B200, 64, 10000, 8, { ep: 32 }).comm;
  assert.ok(nv < ib, `${nv} vs ${ib}`); // 32 GPUs fit one NVL72 domain, not one HGX board
  // Kimi K3 needs two nodes per instance; with experts over 32 GPUs it decodes without pipeline stages
  const c = config();
  Object.assign(c.sim, { model: "kimi-k3", gpu: "b200", decodeGpu: "b200", tp: 8, pp: 2, gpusPerNode: 8, nodes: 2, decodeEp: 32 });
  const r = runScenario(c);
  assert.ok(!r.error, r.error);
  assert.equal(r.decodeEp, 32);
  assert.equal(r.decodeGpus % 32, 0);
});
