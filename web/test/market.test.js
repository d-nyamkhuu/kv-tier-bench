import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { defaultConfig } from "../config.js";
import { marketRun, providerStats } from "../market.js";
import { decodeStepS, TP_SYNC_PER_LAYER_S, withTp } from "../model.js";
import { GPUS, MODELS, SOFTWARE, VARIANTS } from "../presets.js";

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
