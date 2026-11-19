import assert from "node:assert/strict";
import { test } from "node:test";
import { renderDiagram } from "../diagram.js";

const base = {
  nodes: 12, gpusPerNode: 4, tp: 2, gpuShort: "H100", gpuName: "H100 SXM", kvGiB: 92.7, tpLink: "NVLink", tpLinkTitle: "NVLink",
  cpuGiB: 96, disk: { name: "Local NVMe", short: "NVMe Gen5", gib: 3576 }, p2p: { name: "RoCE", short: "RoCE 400G" }, shared: { name: "S3", short: "S3 Standard" },
  decode: { gpus: 4, gpuShort: "H100", batch: 64 }, transfer: "RoCE", transferShort: "RoCE 400G", rate: "300", followUp: 0.8, routing: "sticky",
};

test("diagram draws every part, caps the nodes it draws and escapes text", () => {
  const svg = renderDiagram({ ...base, share: { gpu: 0.5, cpu: 0.1, disk: 0.1, p2p: 0.05, shared: 0.1, miss: 0.15 } });
  assert.match(svg, /^<svg[^>]*viewBox/);
  for (const t of ["gpu", "cpu", "disk", "p2p", "shared", "decode", "load", "routing", "nodes"]) assert.match(svg, new RegExp(`data-target="${t}"`));
  assert.equal((svg.match(/>Node \d+</g) ?? []).length, 3);
  assert.match(svg, /\+ 9 more nodes/);
  assert.doesNotMatch(renderDiagram({ ...base, gpuName: "<b>", share: {} }), /<b>/);
});

test("diagram works before the first estimate, on a phone, and without optional tiers", () => {
  const svg = renderDiagram({ ...base, nodes: 1, tp: 1, disk: null, p2p: null, shared: null, decode: null, followUp: null, share: {} }, { vertical: true });
  assert.match(svg, /viewBox="0 0 360 /);
  assert.doesNotMatch(svg, /NaN|undefined/);
  assert.doesNotMatch(svg, /data-target="(p2p|shared|decode)"/);
});
