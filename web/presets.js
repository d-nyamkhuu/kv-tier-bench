// Tier variants, GPU profiles and the software constants of the model. Every number is quoted from the linked page
// (retrieved 2026-11-18), derived from one (said so), calibrated on kv-tier-bench s1 where nothing is published (said
// so), or listed in `assumptions`, which the page marks. Unit conversions: 1 GiB = 1024^3 bytes, 1 MB = 10^6 bytes, Gbit/s = 10^9 bit/s.

const GIB = 1024 ** 3;
const MBs = (mb) => (mb * 1e6) / GIB; // MB/s (decimal, as spec sheets give it) -> GiB/s
const MiBs = (mib) => mib / 1024;
const GB = (gb) => (gb * 1e9) / GIB; // decimal GB -> GiB

const REPO = "https://github.com/d-nyamkhuu/kv-tier-bench/blob/main";
import { deriveArch, gdnStateBytes } from "./model.js";

const S = {
  plan: { label: "kv-tier-bench docs/plan.md (s1 decisions, day-1 checks)", url: `${REPO}/docs/plan.md` },
  prices: { label: "kv-tier-bench configs/prices.yaml", url: `${REPO}/configs/prices.yaml` },
  types: "https://docs.nebius.com/compute/storage/types",
  pricing: "https://docs.nebius.com/compute/resources/pricing",
  local: "https://docs.nebius.com/compute/storage/local-disks",
  objClasses: "https://docs.nebius.com/object-storage/storage-classes",
  objPricing: "https://docs.nebius.com/object-storage/resources/pricing",
  objQuota: "https://docs.nebius.com/object-storage/resources/quotas-limits",
  vmTypes: "https://docs.nebius.com/compute/virtual-machines/types",
  ib: "https://docs.nebius.com/compute/clusters/gpu",
};
const src = (label, url, quote) => ({ label, url, quote });

// Measured KV-transfer results (retrieved 2026-11-18). Efficiency = what a KV load achieves / the link or device peak.
const LLMD = "https://llm-d.ai/blog/networking-for-distributed-inference-llm-d";
const MOONCAKE_VLLM = "https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/performance/vllm/vllm-v1-pd-performance.md";
const PYKV = "https://arxiv.org/abs/2609.11744";
// TCP: NIXL with UCCL over a 100G link moved KV host memory at ~4.9 GB/s of 12.5 (llm-d blog, June 2026)
const TCP_EFFICIENCY = 4.9 / 12.5;
// RDMA: Mooncake connector inside vLLM, one request at a time, 8x RoCE: 71.1% of line rate (end to end, not raw)
const RDMA_EFFICIENCY = 0.711;
const RDMA_LATENCY_S = 0.001; // DERIVED: intercept of transfer time vs KV size in the same Mooncake table (0.5-1 ms)
// Local NVMe through vLLM's filesystem connector (direct I/O, staged via CPU): 11.2-12.0 GB/s on a Gen5 drive that
// fio reads at 13.5 GB/s, and ~6 ms from cache lookup to the first file read (py-kvcache, Sep 2026). The paper times
// disk to GPU ("covers the disk read as well as the GPU copy"), so the rate is end to end (end_to_end: no separate
// host -> GPU copy is added). Object storage and peer rates are network rates: their host copy follows the transfer.
const NVME_VS_FIO = 11.6 / 13.5;
const NVME_LATENCY_S = 0.006;
const pykvSources = [
  src("py-kvcache (arXiv 2609.11744), vLLM + llm-d filesystem connector", PYKV, "llm-d moves 9.0 GB/s at 1k tokens and 11.2 to 12.0 GB/s at larger sizes"),
  src("py-kvcache (arXiv 2609.11744)", PYKV, "fio measured 13.5 GB/s on the Kioxia drive, and 6 GB/s from the Samsung PM9A3"),
  src("py-kvcache (arXiv 2609.11744)", PYKV, "first file read begins approximately 6 ms after cache lookup starts"),
];
const rdmaSources = [
  src("Mooncake: vLLM v1 prefill/decode transfer benchmark (H800, 8x ConnectX-7 RoCE)", MOONCAKE_VLLM, "Mooncake achieved a peak actual transfer bandwidth of 142.25 GB/s ... 71.1% bandwidth utilization rate"),
  src("llm-d: networking for distributed inference (raw NIXL benchmark)", LLMD, "UCCL and UCX both achieve ~49.5 GB/s (≈99% of the 50 GB/s line rate)"),
];

const TCP_LATENCY_S = RDMA_LATENCY_S; // no source for TCP; llm-d's 1 GB in 211 ms leaves little room for a fixed cost

const tcp = (gbit) => ({
  id: `p2p-tcp-${gbit}`, family: "p2p", name: `TCP ${gbit} Gbit/s`,
  link_gbit_s: gbit, efficiency: TCP_EFFICIENCY, cap_gib_s: null, latency_s: TCP_LATENCY_S, usd_per_hour: 0,
  sources: [
    src("llm-d: networking for distributed inference (NIXL over 100G TCP, H100)", LLMD, "UCCL provides reasonable efficiency by achieving approximately 4.7–4.9 GB/s (~39 Gbps out of 100G), while UCX manages 0.55 GB/s"),
    src("llm-d: networking for distributed inference", LLMD, "UCCL transfers 1 GB in 211 ms (avg) with a P99 of 217 ms"),
  ],
  assumptions: ["latency_s"],
  note: "Efficiency is the best published KV transfer over TCP (UCCL: 39% of a 100G link). No source gives TCP's fixed cost; llm-d's 1 GB in 211 ms implies it is small, so the RDMA figure (~1 ms) is used.",
});

// A Nebius network disk: the documented bandwidth limit, reached through vLLM's filesystem connector as a local NVMe
// drive is (py-kvcache), with the connector's ~6 ms first-read delay. Nebius documents no latency.
const nebiusDisk = (d) => ({ family: "fs", efficiency: NVME_VS_FIO, end_to_end: true, latency_s: NVME_LATENCY_S, shared: false, bill: "provisioned", usd_per_hour: 0, ...d,
  assumptions: ["efficiency", "latency_s", ...(d.assumptions ?? [])], sources: [...d.sources, ...pykvSources] });

// Object storage. Throughput: AWS's per-instance ceiling (100 Gb/s) at the best published KV-load efficiency over S3
// (VAST + LMCache: 9.79 GB/s, ~80% of a 100 GbE line with 64 I/O threads). Latency: AWS's figures per class.
const AWS_S3 = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html";
const AWS_PRICES = "https://aws.amazon.com/s3/pricing/";
const VAST_S3 = "https://www.vastdata.com/blog/maximize-inference-roi-get-6-2x-more-tokens-sec-with-s3-tcp-and-kv";
const S3_LINK_GBIT = 100;
const S3_EFFICIENCY = 9.79 / 12.5;
const S3_STANDARD_LATENCY_S = 0.15; // AWS: "roughly 100–200 milliseconds" first byte; the middle
const s3Throughput = [
  src("Amazon S3 performance", AWS_S3, "single-instance transfer rates that maximize the network interface use for their Amazon EC2 instance, which can be up to 100 Gb/s on a single instance"),
  src("VAST Data: S3 over TCP with LMCache and NIXL (2× RTX PRO 6000, 100 GbE)", VAST_S3, "achieving a peak throughput of 9.79 GB/s with only two NVIDIA RTX PRO 6000 GPUs, reaching approximately 80% of line rate"),
];
const obj = (o) => ({ family: "obj", link_gbit_s: S3_LINK_GBIT, efficiency: S3_EFFICIENCY, cap_gib_s: null, capacity_gib: null, shared: true, bill: "stored", usd_per_hour: 0, ...o,
  assumptions: ["efficiency", ...(o.assumptions ?? [])], sources: [...o.sources, ...s3Throughput] });

// short labels for the diagram, the chart and the estimate panel
const SHORT = {
  "fs-nvme-gen5": "NVMe Gen5", "fs-nvme-gen4": "NVMe Gen4", "fs-bf4-icms": "BF-4 ICMS Pod",
  "fs-nebius-local-ssd": "Nebius local SSD", "fs-nebius-nrd": "Nebius network SSD",
  "fs-nebius-io-m3": "Network SSD IO M3", "fs-nebius-network-ssd": "Network SSD (EC)", "fs-sata-ssd": "SATA SSD", "fs-hdd": "HDD",
  "obj-aws-s3-standard": "S3 Standard", "obj-aws-s3-express": "S3 Express", "obj-nebius-enhanced": "Nebius S3 Enhanced", "obj-nebius-standard": "Nebius S3 Standard",
  "p2p-tcp-10": "TCP 10G", "p2p-tcp-100": "TCP 100G", "p2p-tcp-200": "TCP 200G", "p2p-tcp-400": "TCP 400G",
  "p2p-ib-hdr": "IB HDR 200G", "p2p-ib-ndr": "IB NDR 400G", "p2p-ib-xdr": "IB XDR 800G", "p2p-rdma-1600": "CX-9 RDMA 1.6T",
  "p2p-roce-200": "RoCE 200G", "p2p-roce-400": "RoCE 400G",
};

export const VARIANTS = [
  // ---------------------------------------------------------------------------------------------- local disk (fs)
  {
    id: "fs-bf4-icms", family: "fs", name: "BlueField-4 ICMS Context Pod (800G RDMA Flash)",
    bw_gib_s: GB(100), write_bw_gib_s: GB(80), efficiency: RDMA_EFFICIENCY, latency_s: 0.0005, rdma: true, gpu_direct: true, capacity_gib: GB(16384), shared: false, bill: "provisioned", usd_per_gib_month: 0.08, usd_per_hour: 0,
    sources: [
      src("NVIDIA Vera Rubin & BlueField-4 ICMS architecture", "https://www.nvidia.com/en-us/data-center/vera-rubin/", "BlueField-4 DPU ... 800Gb/s to 1.6Tb/s ... Inference Context Memory Storage (ICMS) pod-level KV context tier"),
      ...rdmaSources,
    ],
    assumptions: ["efficiency", "latency_s", "capacity_gib", "usd_per_gib_month", "gpu_direct (lands in GPU memory without a host copy, as NVIDIA describes the tier; no measurement is published)"],
    note: "DPU-managed G3.5 KV context flash tier (BlueField-4 with 128 GB LPDDR5 staging) delivering GPUDirect RDMA into GPU HBM at 800 Gb/s (100 GB/s line rate × 71.1% Mooncake efficiency).",
  },
  {
    id: "fs-nvme-gen5", family: "fs", name: "Local NVMe, PCIe Gen5 (Kioxia CM7-R)",
    bw_gib_s: GB(13.5), write_bw_gib_s: GB(6.7), efficiency: NVME_VS_FIO, end_to_end: true, latency_s: NVME_LATENCY_S, capacity_gib: GB(3840), shared: false, bill: "provisioned", usd_per_gib_month: 0.065, usd_per_hour: 0,
    sources: [...pykvSources, src("Kioxia CM7-R", "https://americas.kioxia.com/en-us/business/ssd/enterprise-ssd/cm7-r.html", "Sustained 128 KiB Sequential Read | 10,000 MB/s | 14,000 MB/s (14,000 for 1.92–15.36 TB); Write 6,700 MB/s"), src("Nebius Compute pricing (local SSD, as a price reference)", S.pricing, "Local SSD disk | $0.065")],
    assumptions: ["capacity_gib", "usd_per_gib_month"],
    note: "Measured inside vLLM: 11.2–12.0 GB/s for large prefixes (9.0 at 1k tokens) against fio's 13.5 GB/s, ~6 ms to the first read; sequential write 6.7 GB/s. Price: Nebius's local SSD rate, as a reference for a drive in the server.",
  },
  {
    id: "fs-nvme-gen4", family: "fs", name: "Local NVMe, PCIe Gen4 (Samsung PM9A3)",
    bw_gib_s: GB(6), write_bw_gib_s: GB(4), efficiency: NVME_VS_FIO, end_to_end: true, latency_s: NVME_LATENCY_S, capacity_gib: GB(3840), shared: false, bill: "provisioned", usd_per_gib_month: 0.065, usd_per_hour: 0,
    sources: [...pykvSources, src("Samsung PM9A3", "https://semiconductor.samsung.com/ssd/datacenter-ssd/pm9a3/", "sequential read speeds up to 6,800 MB/s, write up to 4,000 MB/s"), src("Nebius Compute pricing (local SSD, as a price reference)", S.pricing, "Local SSD disk | $0.065")],
    assumptions: ["efficiency", "capacity_gib", "usd_per_gib_month"],
    note: "Peak is py-kvcache's fio read of this drive (6 GB/s; spec 6.8, write 4.0). The vLLM-to-fio ratio (86%) and ~6 ms were measured on a Gen5 drive.",
  },
  {
    id: "fs-nebius-local-ssd", family: "fs", name: "Nebius local SSD, 3.5 TiB (B300 VMs only)",
    bw_gib_s: GB(6.8), write_bw_gib_s: GB(2.6), efficiency: NVME_VS_FIO, end_to_end: true, latency_s: NVME_LATENCY_S, capacity_gib: 3584, shared: false, bill: "provisioned", usd_per_gib_month: 0.065, usd_per_hour: 0,
    sources: [
      src("Nebius disk types", S.types, "Local SSD disk ... 3.5 TiB ... Read: 6.8 GB/s Write: 2.6 GB/s"),
      src("Nebius local disks", S.local, "Only on the NVIDIA® B300 NVLink platform (gpu-b300-sxm) with the eight-GPU preset 8gpu-192vcpu-2768gb"),
      src("Nebius Compute pricing", S.pricing, "Local SSD disk | $0.065"),
      ...pykvSources,
    ],
    assumptions: ["efficiency"], note: "Ephemeral. Efficiency and latency are the vLLM filesystem-connector results on a Gen5 NVMe drive (py-kvcache), applied to Nebius's read (6.8 GB/s) and write (2.6 GB/s) peaks.",
  },
  nebiusDisk({
    id: "fs-nebius-nrd", name: "Nebius network SSD, non-replicated, 1767 GiB", bw_gib_s: 2, write_bw_gib_s: 1.5, capacity_gib: 1767, usd_per_gib_month: 0.053,
    sources: [
      src("Nebius disk types", S.types, "Network SSD NRD ... Read bandwidth 2 GiB/s, write 1.5 GiB/s"),
      src("Nebius disk types", S.types, "2048 MiB/s ÷ 110 MiB/s ≈ 18.62, rounding up to 19 units. Minimum disk size = 19 × 93 GiB = 1767 GiB"),
      src("Nebius Compute pricing", S.pricing, "Network SSD Non-replicated disk | $0.053"),
    ],
    note: "The smallest size that reaches the documented 2 GiB/s read / 1.5 GiB/s write. Nebius calls its limits upper bounds and says encryption can cost up to 15%.",
  }),
  nebiusDisk({
    id: "fs-nebius-io-m3", name: "Nebius network SSD IO M3 (3× mirrored), 1767 GiB", bw_gib_s: 2, write_bw_gib_s: 1.5, capacity_gib: 1767, usd_per_gib_month: 0.118,
    sources: [
      src("Nebius disk types", S.types, "Network SSD IO M3 ... multiples of 93 GiB ... Read/write bandwidth 2 GiB/s ... Read bandwidth per unit 110 MiB/s"),
      src("Nebius Compute pricing", S.pricing, "Network SSD IO M3 disk | $0.118"),
    ],
    note: "Same documented limits as non-replicated, with replication; costs 2.2× more.",
  }),
  nebiusDisk({
    id: "fs-nebius-network-ssd", name: "Nebius network SSD (erasure coded), 960 GiB", bw_gib_s: MiBs(450), capacity_gib: 960, usd_per_gib_month: 0.071,
    sources: [
      src("Nebius disk types", S.types, "Network SSD ... Allocation unit size 32 GiB ... Read/write bandwidth 450 MiB/s ... you need a disk of 960 GiB or more"),
      src("Nebius Compute pricing", S.pricing, "Network SSD disk | $0.071"),
    ],
  }),
  {
    id: "fs-sata-ssd", family: "fs", name: "Enterprise SATA SSD (Samsung PM893)",
    bw_gib_s: MBs(550), write_bw_gib_s: MBs(520), efficiency: NVME_VS_FIO, end_to_end: true, latency_s: NVME_LATENCY_S, capacity_gib: GB(3840), shared: false, bill: "provisioned", usd_per_gib_month: 0.065, usd_per_hour: 0,
    sources: [src("Samsung PM893", "https://semiconductor.samsung.com/ssd/datacenter-ssd/pm893/", "sequential read/write speeds of up to 550/520 MB/s"), ...pykvSources, src("Nebius Compute pricing (local SSD, as a price reference)", S.pricing, "Local SSD disk | $0.065")],
    assumptions: ["efficiency", "capacity_gib", "usd_per_gib_month"], note: "Efficiency and latency borrowed from the NVMe measurement in vLLM.",
  },
  {
    id: "fs-hdd", family: "fs", name: "Enterprise HDD (Seagate Exos X18, 18 TB)",
    bw_gib_s: MBs(270), efficiency: NVME_VS_FIO, end_to_end: true, latency_s: NVME_LATENCY_S + 0.00416, capacity_gib: GB(18000), shared: false, bill: "provisioned", usd_per_gib_month: 0.045, usd_per_hour: 0,
    sources: [
      src("Seagate Exos X18 datasheet", "https://www.seagate.com/files/www-content/datasheets/pdfs/exos-x18-channel-DS2045-2-2010US-en_US.pdf", "Max. Sustained Transfer Rate OD (MB/s,MiB/s) 270/258 ... Average Latency (ms) 4.16"),
      src("AWS EBS pricing (HDD price reference)", "https://aws.amazon.com/ebs/pricing/", "st1: In a region that charges $0.045 per GB-month"),
      ...pykvSources,
    ],
    assumptions: ["efficiency", "usd_per_gib_month"],
    note: "Outer-diameter (best case) rate. Latency = the NVMe connector overhead + the drive's 4.16 ms average latency. Price: AWS's throughput-optimized HDD volume, as a reference.",
  },

  // ---------------------------------------------------------------------------------------------- shared storage
  obj({
    id: "obj-aws-s3-standard", name: "Amazon S3 Standard", latency_s: S3_STANDARD_LATENCY_S, usd_per_gib_month: 0.023, usd_per_1k_put: 0.005, usd_per_1k_get: 0.0004,
    sources: [
      src("Amazon S3 performance", AWS_S3, "These applications can achieve consistent small object latencies (and first-byte-out latencies for larger objects) of roughly 100–200 milliseconds."),
      src("Amazon S3 pricing (US East, N. Virginia)", AWS_PRICES, "Standard Storage First 50 TB per GB Mo | $0.023"),
    ],
    note: "One bucket for the whole cluster, unbounded. US East request prices are $0.005 per thousand PUTs and $0.0004 per thousand GETs or HEADs.",
  }),
  obj({
    id: "obj-aws-s3-express", name: "Amazon S3 Express One Zone", latency_s: 0.007, usd_per_gib_month: 0.11, usd_per_1k_put: 0.00113, usd_per_1k_get: 0.00003, usd_per_gib_written: 0.0032, usd_per_gib_read: 0.0006,
    sources: [
      src("AWS Storage Blog: Spark on S3 Express One Zone (03 Sep 2026)", "https://aws.amazon.com/blogs/storage/run-spark-31-faster-and-optimize-compute-costs-with-amazon-s3-express-one-zone-on-amazon-emr/", "S3 Express One Zone completing each read in 7 milliseconds instead of 121 milliseconds"),
      src("Amazon S3 Express One Zone", "https://aws.amazon.com/s3/storage-classes/express-one-zone/", "consistent single-digit millisecond request latency"),
      src("Amazon S3 pricing (US East, N. Virginia)", AWS_PRICES, "Express One Zone Storage $0.11 per GB Mo ... Object Upload $0.0032 per GB ... Object Retrieval $0.0006 per GB"),
    ],
    note: "Latency is the mean GET AWS measured on this class (S3 Standard: 121 ms in the same test). Includes upload, retrieval, PUT, GET and HEAD charges (US East).",
  }),
  obj({
    id: "obj-nebius-enhanced", name: "Nebius Object Storage, Enhanced Throughput", latency_s: S3_STANDARD_LATENCY_S / 2, usd_per_gib_month: 0.11,
    sources: [
      src("Nebius storage classes", S.objClasses, "2x latency improvement (on larger objects) and up to 10x throughput improvement per GiB of stored data compared to the Intelligent and Standard classes"),
      src("Nebius storage classes", S.objClasses, "Unthrottled upload and download speed for Enhanced Throughput"),
      src("Nebius Object Storage pricing", S.objPricing, "Used space in Enhanced Throughput storage | $0.11 ... class A and B operations: Free"),
      src("Nebius Object Storage quotas", S.objQuota, "Total size of objects in Enhanced Throughput storage | 10 TiB (default quota)"),
    ],
    assumptions: ["latency_s", "link_gbit_s"],
    note: "Nebius publishes no absolute latency or per-VM rate. Latency: AWS S3's 150 ms, halved per Nebius's \"2x latency improvement\"; throughput: AWS's per-instance figure.",
  }),
  obj({
    id: "obj-nebius-standard", name: "Nebius Object Storage, Standard", latency_s: S3_STANDARD_LATENCY_S, efficiency: S3_EFFICIENCY / 10, usd_per_gib_month: 0.0147,
    sources: [
      src("Nebius Object Storage pricing", S.objPricing, "Used space in Standard storage | $0.0147 ... class B operations (GET, HEAD) | $0.00028 per 1000"),
      src("Nebius storage classes", S.objClasses, "2x latency improvement (on larger objects) and up to 10x throughput improvement per GiB of stored data compared to the Intelligent and Standard classes"),
      src("Nebius storage classes", S.objClasses, "the upload and download speed for objects of the Intelligent and Standard classes is limited to 20 GBps per tenant in each region"),
    ],
    assumptions: ["latency_s", "link_gbit_s"],
    note: "Latency: AWS S3 Standard's. Throughput: a tenth of Enhanced Throughput, per Nebius's \"up to 10x\". Region-wide cap of 20 GB/s per tenant, which the simulator does not enforce. GET fees left out.",
  }),
  // ---------------------------------------------------------------------------------------------- peer network
  tcp(10),
  tcp(100),
  tcp(200),
  tcp(400),
  ...[
    ["p2p-ib-hdr", "RDMA, InfiniBand HDR 200 Gbit/s", 200, "ib"],
    ["p2p-ib-ndr", "RDMA, InfiniBand NDR 400 Gbit/s", 400, "ib"],
    ["p2p-ib-xdr", "RDMA, InfiniBand XDR 800 Gbit/s", 800, "ib"],
    ["p2p-rdma-1600", "RDMA, ConnectX-9 1.6 Tbit/s", 1600, "ib"],
    ["p2p-roce-200", "RDMA, RoCE 200 GbE", 200, "roce"],
    ["p2p-roce-400", "RDMA, RoCE 400 GbE", 400, "roce"],
  ].map(([id, name, gbit, kind]) => ({
    id, family: "p2p", name, rdma: true, per_gpu: kind === "ib",
    link_gbit_s: gbit, efficiency: RDMA_EFFICIENCY, cap_gib_s: null, latency_s: RDMA_LATENCY_S, usd_per_hour: 0,
    sources: [...rdmaSources, ...(gbit >= 800
      ? [src("NVIDIA Quantum-X800 & ConnectX-9 SuperNIC", "https://www.nvidia.com/en-us/networking/products/infiniband/quantum-x800/", "800Gb/s InfiniBand XDR ... 1.6Tb/s ConnectX-9 SuperNIC scale-out per GPU")]
      : kind === "ib"
        ? [src("Nebius GPU clusters", S.ib, "Each GPU in a VM is connected through a network interface card (NIC) that provides 400 Gbps")]
        : [src("Nebius VM types", S.vmTypes, "RTX PRO 6000 ... BlueField-3 400 Gbps")])],
    assumptions: kind === "ib" ? ["efficiency", "usd_per_hour"] : ["usd_per_hour"],
    note: "Efficiency is Mooncake's end-to-end KV transfer inside vLLM over RoCE (71%; a raw NIXL benchmark reaches 99%). Latency is the intercept of that transfer table (~1 ms). Dedicated RDMA fabric separate from frontend S3 Ethernet." +
      (kind === "ib" ? " One InfiniBand NIC per GPU (scales with TP); the RoCE efficiency is used for InfiniBand." : ""),
  })),
].map((v) => ({ capacity_gib: null, shared: false, bill: "stored", usd_per_gib_month: null, usd_per_gib_written: 0, usd_per_gib_read: 0, short: SHORT[v.id] ?? v.name, ...v }));

export const FAMILY_INFO = {
  fs: { title: "Disks and filesystems (fs)", text: "A local or network disk per node, or a filesystem shared by the cluster. Billed on provisioned size." },
  obj: { title: "Object storage (obj)", text: "One bucket for the whole cluster. Billed on bytes stored for the retention time, plus any per-GB transfer fees." },
  p2p: { title: "Peer network (p2p)", text: "A hit is a block in another node's CPU memory, pulled over the network; also the link that sends KV from prefill to decode. The peer's memory is part of its VM." },
};

const MC = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release";
export const MOONCAKE_TRACES = [
  { name: "Mooncake conversation (12,031 requests, 3 MB)", url: `${MC}/traces/conversation_trace.jsonl`, blockTokens: 512, license: "Apache-2.0" },
  { name: "Mooncake tool & agent (23,608 requests, 4.4 MB)", url: `${MC}/traces/toolagent_trace.jsonl`, blockTokens: 512, license: "Apache-2.0" },
  { name: "Mooncake synthetic (3,993 requests, 1.1 MB)", url: `${MC}/traces/synthetic_trace.jsonl`, blockTokens: 512, license: "Apache-2.0" },
  { name: "Mooncake arXiv trace (23,608 requests, 4.4 MB)", url: `${MC}/arxiv-trace/mooncake_trace.jsonl`, blockTokens: 512, license: "Apache-2.0" },
];

// ------------------------------------------------------------------------------------------------ GPUs
// Specs from NVIDIA datasheets (dense BF16 = half the "with sparsity" figure), prices from Nebius on demand per
// GPU-hour, retrieved 2026-11-18. mem_gib: usable memory, the datasheet size x 95/96 (an RTX PRO 6000 reports
// 97,887 MiB of "96 GB" in nvidia-smi). host_gib_s: CPU -> GPU KV copy rate through vLLM's offloading, as published.
// mbu: share of peak memory bandwidth batched decode achieves. prefill_mfu: share of dense BF16 peak that prefill's
// matrix multiplies reach (attention runs at SOFTWARE.attn_eff_ratio of that).
const NV = {
  rtx: "https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/",
  l40s: "https://www.nvidia.com/en-us/data-center/l40s/",
  h100: "https://www.nvidia.com/en-us/data-center/h100/",
  h200: "https://www.nvidia.com/en-us/data-center/h200/",
  bw: "https://www.nvidia.com/en-us/data-center/hgx/",
  gh200: "https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/",
  gb200: "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
};
const NEBIUS_PRICES = "https://nebius.com/prices";
const usable = (gb) => (gb * 95) / 96;
// vLLM's CPU offloading on H100 over PCIe 5.0 x16 (py-kvcache): 54–56 GB/s = 51.2 GiB/s at the middle; Gen4: 24–26 GB/s
const GEN5_HOST = (55e9) / GIB;
const GEN4_HOST = (25e9) / GIB;
const GEN5_MEASURED = src("py-kvcache (arXiv 2609.11744), vLLM CPU offloading on H100", PYKV, "Our Snellius node measured 54–56 GB/s over a 16× PCIe 5.0 link ... the GPU ↔CPU copy sustains 54–56 GB/s");
const GEN4_MEASURED = src("py-kvcache (arXiv 2609.11744), PCIe 4.0 x16", PYKV, "with vLLM Offload and LMCache approximately 24–26 GB/s");
const C2C_MEASURED = src("NVIDIA Grace performance tuning guide", "https://docs.nvidia.com/grace-perf-tuning-guide/health-checks.html", "On a healthy system, this bandwidth should be approximately 350-360 GB/s");
const MBU_SOURCES = [
  src("Databricks: LLM inference performance engineering", "https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices", "At batch size 1, we can achieve a higher MBU of 60% on 2xH100-80GBs as compared to 55% on 4xA100-40GB GPUs"),
  src("arXiv 2602.18568 (vLLM, 4×H100, batch 32)", "https://arxiv.org/abs/2602.18568", "the H100 only utilizes 32% of its peak memory bandwidth during distributed LLM decode"),
];
const MBU = 0.6; // Databricks' H100 figure; checks.js compares the result with vLLM runs on H100, H200 and B200
// prefill_mfu of datacenter GPUs: set so the model gives InferenceX's vLLM TTFT for Llama 3.3 70B FP8 (8k in, TP1,
// concurrency 4) against the FP8 peak: H200 for Hopper, B200 for Blackwell. checks.js tests it on H100, which the
// calibration did not use. The RTX PRO 6000 has no published prefill result: its value is calibrated on s1.
const INFERENCEX = src("SemiAnalysis InferenceX: Llama 3.3 70B, vLLM, 8k/1k (run 2025-10-29; the page notes an unofficial deployment)", "https://inferencex.semianalysis.com/compare/llama-3-3-70b-b200-vs-h200", "median TTFT at concurrency 4, TP1: H200 1.166 s, B200 0.654 s");
const HOPPER_MFU = 0.592, BLACKWELL_MFU = 0.484, RTX_MFU = 0.878;
const S1_CALIBRATION = { label: "kv-tier-bench s1: RTX PRO 6000, vLLM v0.34.0, two models, 896–131k tokens (model.calibrate)", url: `${REPO}/results/s1.jsonl` };

// The model's software constants: what vLLM itself costs per request and per token, attention's efficiency relative to
// the matrix multiplies, and the fixed part of a CPU-tier load. Nothing published gives them, so they are fitted on
// s1 (model.calibrate; a test keeps these in step with s1.json). gpu_memory_utilization and the activation reserve
// are vLLM settings (s1 used 0.95 and planned 3 GiB per GPU).
export const SOFTWARE = {
  request_overhead_s: 0.0524, per_token_overhead_s: 0.712e-6, attn_eff_ratio: 0.617, cpu_load_latency_s: 0.00366,
  gpu_memory_utilization: 0.95, activation_gib_per_gpu: 3,
  sources: [S1_CALIBRATION, { ...S.plan, quote: "`max_model_len` is set per model: 66K for Qwen3-VL-32B on RTX, at `--gpu-memory-utilization 0.95`." }],
  assumptions: ["activation_gib_per_gpu"],
};
// nvlink_gb_s: GPU-to-GPU NVLink per GPU (both directions); nvlink_domain: how many GPUs one NVLink fabric joins (8 on
// an HGX board, 72 in a GB200 NVL72 rack, 1 without NVLink). Decode with expert parallelism moves tokens to their
// experts over NVLink inside the domain and over the node network beyond it. Tensor parallel efficiency comes from
// TP_EFF, and GPUs in one node do not share KV caches.
const NVL = {
  hopper: (url, name) => src(`NVIDIA ${name}`, url, "NVIDIA NVLink: 900GB/s"),
  blackwell: src("NVIDIA HGX", NV.bw, "NVLink GPU-to-GPU Bandwidth 1.8 TB/s"),
  hgx: src("NVIDIA HGX", "https://www.nvidia.com/en-us/data-center/hgx/", "HGX B200: 8x NVIDIA Blackwell SXM ... HGX B300: 8x NVIDIA Blackwell Ultra SXM ... Total NVLink Bandwidth 14.4 TB/s"),
  nvl72: src("NVIDIA GB200 NVL72", "https://www.nvidia.com/en-us/data-center/gb200-nvl72/", "a 72-GPU NVIDIA NVLink™ domain that acts as a single, massive GPU"),
  blackwellBrief: src("NVIDIA Blackwell architecture technical brief (V2.1)", "https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf", "Blackwell and Blackwell Ultra GPUs include 18 fifth-generation NVLink links to provide 1.8 TB/sec total bandwidth"),
};
// tflops_fp8: dense FP8 peak, used for models released in FP8 (weights and activations). Set to twice BF16, the ratio
// NVIDIA's Hopper and Blackwell datasheets show; not quoted per GPU here, so marked as an assumption.
// tflops_fp4: dense FP4, on GPUs with FP4 tensor cores (Blackwell) only, set to twice FP8 (assumption, same ratio);
// Hopper and Ada have none, so FP4 weights compute at FP8 there. Only decode uses it: prefill runs FP4 at the FP8 rate
// (model.js prefillTflops).
const BLACKWELL = new Set(["rtx-pro-6000", "b200", "b300", "gb200", "vr200"]);
const gpu = (g) => ({ mbu: MBU, nvlink_gb_s: null, nvlink_domain: g.pcie_only || !g.nvlink_gb_s ? 1 : 8, tflops_fp8: 2 * g.tflops_bf16, tflops_fp4: BLACKWELL.has(g.id) ? 4 * g.tflops_bf16 : null, ...g,
  assumptions: [...(g.assumptions ?? []), "tflops_fp8", ...(BLACKWELL.has(g.id) ? ["tflops_fp4"] : [])],
  sources: [...g.sources, ...MBU_SOURCES, ...(g.prefill_mfu === RTX_MFU ? [S1_CALIBRATION] : [INFERENCEX])] });

// Host memory bought on demand, as a price reference for CPU and peer KV caches: Nebius lists RAM on its non-GPU
// VMs per GiB-hour (on GPU VMs it comes with the GPU-hour). 730 hours make a month.
export const HOST_RAM = {
  usd_per_gib_month: 3.285, // $0.0045 per GiB-hour × 730 hours
  source: src("Nebius Compute pricing", S.pricing, "Non-GPU AMD Epyc Genoa. RAM | $0.0045 | 1 GiB hour"),
};

export const GPUS = [
  gpu({
    id: "rtx-pro-6000", name: "RTX PRO 6000 Blackwell",
    prefill_mfu: RTX_MFU, tflops_bf16: 500, mem_gib: 95, pcie_only: true, mem_bw_tb_s: 1.597, host_gib_s: GEN5_HOST, usd_per_hour: 1.8,
    assumptions: ["mbu"], sources: [
      src("NVIDIA RTX PRO 6000 Blackwell Server Edition", NV.rtx, "FP16 | BF16 Tensor Core 1 PFLOP ... 96 GB GDDR7 ... 1597 GB/s"),
      { ...S.plan, quote: "RTX PRO 6000 Blackwell, as reported by nvidia-smi (~97,887 MiB)" },
      src("Nebius prices", NEBIUS_PRICES, "RTX PRO 6000 | $1.80"),
    ],
    note: "No published prefill benchmark: prefill_mfu is fitted on s1. NVIDIA's 1 PFLOP has no dense/sparse footnote; 500 dense fits s1 (at 250 the fit would need over 100% of peak).",
  }),
  gpu({
    id: "l40s", name: "L40S", prefill_mfu: HOPPER_MFU, pcie_only: true, tflops_bf16: 362.05, mem_gib: usable(48), mem_bw_tb_s: 0.864, host_gib_s: GEN4_HOST, usd_per_hour: 1.55,
    sources: [
      src("NVIDIA L40S", NV.l40s, "BFLOAT16 Tensor Core TFLOPS 362.05 I 733* ... 48GB GDDR6 with ECC ... 864GB/s ... PCIe Gen4 x16"),
      GEN4_MEASURED,
      src("Nebius prices", NEBIUS_PRICES, "L40S Intel: from $1.55 (GPU $1.35 + vCPU + RAM)"),
    ],
    assumptions: ["prefill_mfu", "mbu"], note: "48 GB holds neither s1 model's bf16 weights. prefill_mfu: Hopper's published InferenceX figure (59.2%).",
  }),
  gpu({
    id: "h100", name: "H100 SXM", prefill_mfu: HOPPER_MFU, tflops_bf16: 989.4, mem_gib: usable(80), mem_bw_tb_s: 3.35, host_gib_s: GEN5_HOST, usd_per_hour: 4.5, nvlink_gb_s: 900,
    sources: [
      src("NVIDIA H100 Tensor Core GPU architecture whitepaper", "https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf", "Peak BF16 Tensor TFLOPS with FP32 Accumulate ... 989.4/1978.9"),
      src("NVIDIA H100", NV.h100, "80GB ... 3.35TB/s ... PCIe Gen5: 128GB/s"),
      GEN5_MEASURED, src("Nebius prices", NEBIUS_PRICES, "NVIDIA HGX H100 | $4.50"), NVL.hopper(NV.h100, "H100"),
    ],
    assumptions: ["nvlink_domain (8, an HGX board)"],
  }),
  gpu({
    id: "h200", name: "H200 SXM", prefill_mfu: HOPPER_MFU, tflops_bf16: 989.5, mem_gib: usable(141), mem_bw_tb_s: 4.8, host_gib_s: GEN5_HOST, usd_per_hour: 5.4, nvlink_gb_s: 900,
    sources: [src("NVIDIA H200", NV.h200, "BFLOAT16 Tensor Core 1,979 TFLOPS (with sparsity) ... 141GB of HBM3e ... 4.8TB/s"), GEN5_MEASURED, src("Nebius prices", NEBIUS_PRICES, "NVIDIA HGX H200 | $5.40"), NVL.hopper(NV.h200, "H200")],
    assumptions: ["nvlink_domain (8, an HGX board)"],
  }),
  gpu({
    id: "b200", name: "B200 (HGX)", prefill_mfu: BLACKWELL_MFU, tflops_bf16: 2250, mem_gib: usable(180), mem_bw_tb_s: 7.7, host_gib_s: GEN5_HOST, usd_per_hour: 8.5, nvlink_gb_s: 1800,
    sources: [src("NVIDIA HGX / Blackwell datasheet", NV.bw, "HGX B200: FP16/BF16 Tensor Core 4.5 PFLOPS (sparse; dense is one-half) ... 180 GB HBM3E ... 7.7 TB/s ... PCIe Gen5"), GEN5_MEASURED, src("Nebius prices", NEBIUS_PRICES, "NVIDIA HGX B200 | $8.50"), NVL.blackwell, NVL.hgx],
  }),
  gpu({
    id: "b300", name: "B300 (HGX, Blackwell Ultra)", prefill_mfu: BLACKWELL_MFU, tflops_bf16: 2250, mem_gib: usable(270), mem_bw_tb_s: 7.7, host_gib_s: 2 * GEN5_HOST, usd_per_hour: 9.5, nvlink_gb_s: 1800,
    sources: [src("NVIDIA HGX / Blackwell Ultra datasheet", NV.bw, "HGX B300: FP16/BF16 Tensor Core 4.5 PFLOPS (sparse) ... 270 GB HBM3E ... 7.7 TB/s ... PCIe Gen6: 256 GB/s"), src("Nebius prices", NEBIUS_PRICES, "NVIDIA HGX B300 | $9.50"), NVL.blackwell, NVL.hgx],
    assumptions: ["host_gib_s"], note: "Host link: twice the measured Gen5 rate, assuming the host runs PCIe Gen6; not verified.",
  }),
  gpu({
    id: "gh200", name: "GH200 (Grace Hopper, 144 GB)", prefill_mfu: HOPPER_MFU, tflops_bf16: 990, mem_gib: usable(144), mem_bw_tb_s: 4.9, host_gib_s: 326, usd_per_hour: null,
    sources: [src("NVIDIA GH200", NV.gh200, "BFLOAT16 Tensor Core 1,979 teraFLOPS* | 990 teraFLOPS ... 144GB HBM3e ... Up to 4.9TB/s ... NVLink-C2C CPU-to-GPU bandwidth 900GB/s"), C2C_MEASURED],
    assumptions: ["usd_per_hour"], note: "CPU memory over NVLink-C2C at ~350 GB/s measured. Not offered by Nebius: no price.",
  }),
  gpu({
    id: "gb200", name: "GB200 (per GPU)", prefill_mfu: BLACKWELL_MFU, tflops_bf16: 2500, mem_gib: usable(186), mem_bw_tb_s: 8, host_gib_s: 326, usd_per_hour: null, nvlink_gb_s: 1800, nvlink_domain: 72,
    sources: [NVL.blackwellBrief, NVL.nvl72, src("NVIDIA GB200 NVL72", NV.gb200, "per GPU: FP16/BF16 Tensor Core 5 PFLOPS (sparse) ... 186 GB HBM3E ... 8 TB/s ... 900 GB/s of bidirectional bandwidth (C2C)"), C2C_MEASURED],
    assumptions: ["host_gib_s", "usd_per_hour"], note: "One Grace CPU serves two GPUs; the per-GPU C2C share is not published, so the GH200 measurement is used. Nebius: contact sales, no price.",
  }),
  gpu({
    id: "vr200", name: "VR200 (Vera Rubin NVL72, per GPU)", prefill_mfu: BLACKWELL_MFU, tflops_bf16: 6250, mem_gib: usable(288), mem_bw_tb_s: 22, host_gib_s: 2 * 326, usd_per_hour: null, nvlink_gb_s: 3600, nvlink_domain: 72,
    sources: [
      src("NVIDIA Vera Rubin NVL72", "https://www.nvidia.com/en-us/data-center/vera-rubin/", "Rubin GPU: 50 PFLOPS NVFP4 inference (sparse; 25 PFLOPS dense) ... 288 GB HBM4 ... 22 TB/s ... NVLink 6: 3.6 TB/s per GPU across 72-GPU rack domain (260 TB/s) ... Vera-Rubin NVLink-C2C: 1.8 TB/s bidirectional"),
      C2C_MEASURED,
    ],
    assumptions: ["prefill_mfu", "tflops_bf16", "host_gib_s", "usd_per_hour"],
    note: "Rubin R200 GPU in a 72-GPU NVLink 6 rack domain (288 GB HBM4 at 22 TB/s; 25 PFLOPS dense NVFP4). Host C2C link set to 2× Grace-Hopper's measured C2C (~652 GiB/s) matching Gen-2 NVLink-C2C's 2× line rate (1.8 TB/s bidir).",
  }),
];

// ---- Models beyond s1's two: open-weight releases with public config.json (fetched 2026-11-18). Each spec is read
// off its config.json and model card and turned into sizes and FLOPs by model.deriveArch. KV cache in bf16 (as s1
// ran) unless the card publishes another per-token size; linear-attention state in bf16 (s1's
// --mamba-ssm-cache-dtype bfloat16; the HF default fp32 doubles the recurrent part).
const HF = (repo) => `https://huggingface.co/${repo}`;
const cfgSrc = (repo) => src(`${repo} config.json`, `${HF(repo)}/blob/main/config.json`);
const cardSrc = (repo, quote) => src(`${repo} model card`, HF(repo), quote);
const filesSrc = (repo, bytes) => src(`${repo} files: *.safetensors at the repo root`, `${HF(repo)}/tree/main`, `${bytes.toLocaleString("en-US")} bytes of weights`);
// Routed experts from config.json: params = MoE layers x routed x 3 matrices (gate, up, down) x width x moe_intermediate_size.
const moe = ({ layers, routed, per_token, width, inter, bytes_per_param }) =>
  ({ routed, per_token, params: layers * routed * 3 * width * inter, ...(bytes_per_param ? { bytes_per_param } : {}) });
const expertSrc = (repo, quote) => src(`${repo} config.json (experts)`, `${HF(repo)}/blob/main/config.json`, quote);
const MXFP4 = 4.25 / 8; // 4-bit values + one 8-bit scale per 32
const VLLM_MLA_TP = src("vLLM docs: context parallel deployment", "https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/",
  "For DeepSeek-R1, we have 1 kv-head when MLA is enabled. The typical single-node deployment with `-tp 8` causes 8x KV cache duplication.");
const VLLM_STATE_TP = src("vLLM docs: MambaStateShapeCalculator", "https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/mamba/mamba_utils/",
  "temporal_state_shape = (divide(num_v_heads, tp_world_size), head_v_dim, head_k_dim)");
// DeepSeek-style sparse indexer key cache: 128 FP8 values + a 4-byte scale per token and layer (assumption: as vLLM
// stores DeepSeek V3.2's; none of these cards gives it)
const INDEX_BYTES = 132;

const MODEL_SPECS = {
  "qwen3.8-2.4t-a95b": {
    name: "Qwen3.8-2.4T-A95B", hf_id: "Qwen/Qwen3.8-2.4T-A95B", drafters: ["mtp"], drafter_source: '"mtp_num_hidden_layers": 1', released: "BF16", arch: "MoE 2.4T / 95B active; 69 Gated DeltaNet + 23 full attention layers",
    spec: {
      num_layers: 92, hidden_size: 8192, total_params: 2.4e12, active_params: 95e9, weights_bytes: 4892365649336, compute_dtype: "bf16", block_size: 64, experts: moe({ layers: 92, routed: 512, per_token: 10, width: 8192, inter: 2048 }),
      layers: [
        { type: "linear", count: 69, state_bytes: gdnStateBytes({ k_heads: 16, v_heads: 128, k_dim: 128, v_dim: 128, conv: 4 }) },
        { type: "full", count: 23, q_heads: 64, kv_heads: 4, k_dim: 256, v_dim: 256 },
      ],
    },
    sources: [expertSrc("Qwen/Qwen3.8-2.4T-A95B", '"num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 2048, "hidden_size": 8192'), cfgSrc("Qwen/Qwen3.8-2.4T-A95B"), filesSrc("Qwen/Qwen3.8-2.4T-A95B", 4892365649336),
      cardSrc("Qwen/Qwen3.8-2.4T-A95B", "Number of Parameters: 2.4T in total and 95B activated ... Gated Attention: Number of Attention Heads: 64 for Q and 4 for KV; Head Dimension: 256"), VLLM_STATE_TP],
    assumptions: ["block_size"],
    note: "4.9 TB of bf16 weights: more than any 8-GPU node holds, so it runs across nodes (pipeline stages).",
  },
  "qwen3-235b-a22b": {
    name: "Qwen3-235B-A22B", hf_id: "Qwen/Qwen3-235B-A22B", released: "BF16", arch: "MoE 235B / 22B active; 94-layer 16:1 GQA (64 Q / 4 KV heads)",
    spec: {
      num_layers: 94, hidden_size: 4096, total_params: 235e9, active_params: 22e9, weights_bytes: 470192668672, compute_dtype: "bf16", block_size: 64, experts: moe({ layers: 94, routed: 128, per_token: 8, width: 4096, inter: 1536 }),
      layers: [
        { type: "full", count: 94, q_heads: 64, kv_heads: 4, k_dim: 128, v_dim: 128 },
      ],
    },
    sources: [expertSrc("Qwen/Qwen3-235B-A22B", '"num_hidden_layers": 94, "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 1536'), cfgSrc("Qwen/Qwen3-235B-A22B"),
      cardSrc("Qwen/Qwen3-235B-A22B", "235B total parameters and 22B activated parameters ... 94 layers, 64 Q heads and 4 KV heads")],
    assumptions: ["block_size"],
    note: "Extreme 16:1 GQA (4 KV heads): under TP=8 without DP-attention, KV heads replicate 2× across rank pairs.",
  },
  "kimi-k3": {
    name: "Kimi K3", hf_id: "moonshotai/Kimi-K3", released: "MXFP4 experts, BF16 attention", arch: "MoE 2.8T / 104B active; 69 KDA linear + 24 NoPE-MLA layers",
    spec: {
      num_layers: 93, hidden_size: 7168, total_params: 2.8e12, active_params: 104e9, weights_bytes: 1560936091448, compute_dtype: "fp8", dense_bytes_per_param: 2, block_size: 64, experts: moe({ layers: 92, routed: 896, per_token: 16, width: 3584, inter: 3072, bytes_per_param: MXFP4 }),
      layers: [
        { type: "linear", count: 69, state_bytes: gdnStateBytes({ k_heads: 96, v_heads: 96, k_dim: 128, v_dim: 128, conv: 4 }) },
        { type: "mla", count: 24, q_heads: 96, latent: 512, rope: 0, qk_dim: 128, v_dim: 128 },
      ],
    },
    sources: [expertSrc("moonshotai/Kimi-K3", '"num_experts": 896, "num_experts_per_token": 16, "routed_expert_hidden_size": 3584, "moe_intermediate_size": 3072, "first_k_dense_replace": 1, "mla_use_nope": true'), cfgSrc("moonshotai/Kimi-K3"), filesSrc("moonshotai/Kimi-K3", 1560936091448),
      cardSrc("moonshotai/Kimi-K3", "Total Parameters 2.8T; Activated Parameters 104B; Attention-Layer Composition 69 KDA + 24 Gated MLA; MXFP4 weights / MXFP8 activations"), VLLM_MLA_TP, VLLM_STATE_TP],
    assumptions: ["block_size", "KDA state as Gated DeltaNet's shape, bf16", "mla_use_nope omits separate RoPE key cache (512 latent elements/layer)", "routed experts at MXFP4's 4.25 bits per parameter (the rest of the weight bytes are the dense part)", "dense (non-expert) weights at BF16, 2 bytes per parameter, as released"],
    note: "Weights alone are 1.56 TB: in one node only 8 B300s or VR200s hold it; other GPUs need two or more nodes.",
  },
  "deepseek-v4.1-flash": {
    name: "DeepSeek V4.1 Flash", hf_id: "deepseek-ai/DeepSeek-V4.1-Flash", drafters: ["dspark"], drafter_params: { dspark: { draft: 4, layers: 3 } }, drafter_source: '"num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3', kv_layout: "deepseek-v41", released: "FP8, FP4 experts, fp8_ds_mla KV", arch: "MoE 552B / 8B active in prefill; CSA2 compressed sparse attention, FP8 KV",
    spec: {
      num_layers: 40, hidden_size: 5120, kv_dtype_bytes: 1, total_params: 552e9, active_params: 8e9, weights_bytes: 510296708312, compute_dtype: "fp8", block_size: 64, decode_active_params: 16e9, experts: moe({ layers: 40, routed: 384, per_token: 6, width: 5120, inter: 2304, bytes_per_param: MXFP4 }),
      layers: [
        { type: "given", kv_bytes_per_token: (528 + 132) * (3 / 2 + 1), copies_per_tp: true },
        { type: "swa", count: 40, kv_heads: 1, q_heads: 0, k_dim: 528, v_dim: 0, window: 128 },
        { type: "full", stores: false, count: 40, q_heads: 64, kv_heads: 1, k_dim: 512, v_dim: 512, topk: 512 },
        { type: "full", stores: false, count: 8, q_heads: 0, kv_heads: 1, k_dim: 0, v_dim: 0, topk: 1, index_heads: 32, index_dim: 128 }, // the indexer only
      ],
    },
    sources: [expertSrc("deepseek-ai/DeepSeek-V4.1-Flash", '"n_routed_experts": 384, "num_experts_per_tok": 6, "moe_intermediate_size": 2304, "expert_dtype": "fp4"'), cfgSrc("deepseek-ai/DeepSeek-V4.1-Flash"), filesSrc("deepseek-ai/DeepSeek-V4.1-Flash", 510296708312),
      src("LMCache validated DeepSeek V4.1 recipe", "https://docs.lmcache.ai/recipes/deepseek_v41_flash.html", "fp8_ds_mla, 128-token window and compressed/indexer groups"), src("vLLM V4.1 packed KV and index K", "https://github.com/vllm-project/vllm/blob/82c6b325b49ce361f5a295eb7d932e1b439e187d/vllm/models/deepseek_v41/attention.py", "528/584-byte KV records, 132-byte FP8 index K; only KV sources own the index K cache"), cardSrc("deepseek-ai/DeepSeek-V4.1-Flash", "Engram conditional memory (196B parameters, sparsely accessed via token-based lookup)"), src("vLLM recipe: DeepSeek V4.1 Flash (Engram placement)", "https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash", "47.2 GiB of device memory per rank at TP4 and 94.4 GiB at TP2"), cardSrc("deepseek-ai/DeepSeek-V4.1-Flash", "these designs reduce the global KV cache footprint to 890 bytes per token ... activate only 8B parameters per token during prefill and 16B during decode ... SWA Bounded Replay reconstructs missing SWA KV states by replaying only the most recent n_win tokens, avoiding the need to persist SWA KV to SSD"), VLLM_MLA_TP],
    assumptions: ["block_size", "its DSpark drafter: 4 drafted tokens per block of 5 and 3 layers of the model's size (the config's dspark_block_size and next-token layers); accepted tokens as measured on Qwen3-8B", "attention FLOPs (64 heads of 512 over the top 512 tokens on 40 layers, indexer on 8)", "bounded replay rebuilds the trailing window on a prefix hit; logical payload excludes page padding", "Engram tables (about 203 GB of the weight files, FP8) are read a few rows per token (48 rows, about 13 KB), so a step does not read them; one copy is split across the instance's GPUs and never replicated per DP rank"],
    note: "LMCache fp8_ds_mla payload estimate. Four layers own both compressed KV and index K. All 40 layers use a 128-token live window. With bounded replay enabled, persistent caches hold only compressed KV and index K. Excludes scratch rings and page padding. Prefill uses the card's 8B active parameters, decode its 16B.",
  },
  "glm-5.3": {
    name: "GLM-5.3", hf_id: "zai-org/GLM-5.3", drafters: ["mtp"], drafter_source: '"num_nextn_predict_layers": 1', released: "FP8", arch: "MoE 744B / 40B active; MLA with DeepSeek sparse attention on all 78 layers",
    spec: {
      num_layers: 78, hidden_size: 6144, total_params: 744e9, active_params: 40e9, weights_bytes: 755632050320, compute_dtype: "fp8", block_size: 64, experts: moe({ layers: 75, routed: 256, per_token: 8, width: 6144, inter: 2048 }),
      layers: [
        { type: "mla", count: 21, q_heads: 64, latent: 512, rope: 64, qk_dim: 192 + 64, v_dim: 256, topk: 2048, index_heads: 32, index_dim: 128, index_bytes: INDEX_BYTES, index_fp8: true },
        { type: "mla", count: 57, q_heads: 64, latent: 512, rope: 64, qk_dim: 192 + 64, v_dim: 256, topk: 2048 },
      ],
    },
    sources: [expertSrc("zai-org/GLM-5.3", '"n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "first_k_dense_replace": 3'), cfgSrc("zai-org/GLM-5.3"), filesSrc("zai-org/GLM-5.3", 755632050320),
      src("zai-org/GLM-5 model card", HF("zai-org/GLM-5"), "GLM-5 scales from 355B parameters (32B active) to 744B parameters (40B active)"),
      cardSrc("zai-org/GLM-5.3", "GLM-5.3 uses the same base model as GLM-5.2"), VLLM_MLA_TP],
    assumptions: ["active_params (GLM-5's 40B; the 5.3 card gives none)", "block_size", "indexer key cache 132 B per token on its 21 own-indexer layers"],
  },
  "glm-5.3-flash": {
    name: "GLM-5.3-Flash", hf_id: "zai-org/GLM-5.3-Flash", drafters: ["mtp"], drafter_source: '"num_nextn_predict_layers": 1', released: "FP8", arch: "MoE 320B / 18B active; 34 KDA linear + 11 MLA sparse layers",
    spec: {
      num_layers: 45, hidden_size: 4096, total_params: 320e9, active_params: 18e9, weights_bytes: 328337455672, compute_dtype: "fp8", block_size: 64, experts: moe({ layers: 42, routed: 288, per_token: 8, width: 4096, inter: 2048 }),
      layers: [
        { type: "linear", count: 34, state_bytes: gdnStateBytes({ k_heads: 64, v_heads: 64, k_dim: 128, v_dim: 128, conv: 4 }) },
        { type: "mla", count: 11, q_heads: 64, latent: 512, rope: 0, qk_dim: 256, v_dim: 256, topk: 2048, index_heads: 32, index_dim: 128, index_bytes: INDEX_BYTES, index_fp8: true },
      ],
    },
    sources: [expertSrc("zai-org/GLM-5.3-Flash", '"n_routed_experts": 288, "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "first_k_dense_replace": 3'), cfgSrc("zai-org/GLM-5.3-Flash"), filesSrc("zai-org/GLM-5.3-Flash", 328337455672),
      cardSrc("zai-org/GLM-5.3-Flash", "With 320B total parameters and just 18B active parameters ... we introduce a hybrid architecture combining sparse and linear attention"), VLLM_MLA_TP, VLLM_STATE_TP],
    assumptions: ["block_size", "KDA state as Gated DeltaNet's shape, bf16", "indexer key cache 132 B per token per layer"],
  },
  "qwen3.8-flash-next": {
    name: "Qwen3.8-Flash-Next", hf_id: "Qwen/Qwen3.8-Flash-Next", drafters: ["mtp"], drafter_source: '"mtp_num_hidden_layers": 1', released: "BF16", arch: "MoE 125B / 6B active; 36 Gated DeltaNet + 12 sparse attention layers",
    spec: {
      num_layers: 48, hidden_size: 2560, total_params: 180e9, active_params: 6e9, weights_bytes: 360000192888, compute_dtype: "bf16", block_size: 64, experts: moe({ layers: 48, routed: 512, per_token: 10, width: 2560, inter: 640 }),
      layers: [
        { type: "linear", count: 36, state_bytes: gdnStateBytes({ k_heads: 16, v_heads: 48, k_dim: 128, v_dim: 128, conv: 4 }) },
        { type: "full", count: 12, q_heads: 24, kv_heads: 2, k_dim: 256, v_dim: 256, topk: 2048, index_heads: 4, index_dim: 128, index_bytes: (128 * 2) / 4 },
      ],
    },
    sources: [expertSrc("Qwen/Qwen3.8-Flash-Next", '"num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640, "hidden_size": 2560'), cfgSrc("Qwen/Qwen3.8-Flash-Next"), filesSrc("Qwen/Qwen3.8-Flash-Next", 360000192888),
      cardSrc("Qwen/Qwen3.8-Flash-Next", "125B with 6B activated, plus 51B n-gram embedding and 4B MTP ... Qwen Sparse Attention: Number of Attention Heads: 24 for Q and 2 for KV; Head Dimension: 256 ... Indexer Structure: MQA with 4 Query Heads and 1 Shared Key Head; Indexer Head Dimension: 128; Budget: 512 blocks or 2048 tokens"), VLLM_STATE_TP],
    assumptions: ["block_size", "indexer key cache: one bf16 key of 128 per 4 tokens (indexer_compress_ratio 4)"],
  },
  "minimax-m3": {
    name: "MiniMax-M3", hf_id: "MiniMaxAI/MiniMax-M3", drafters: ["mtp"], drafter_source: '"num_nextn_predict_layers": 1, "num_mtp_modules": 7', released: "BF16", arch: "MoE 428B / 23B active; GQA, MiniMax sparse attention on 57 of 60 layers",
    spec: {
      num_layers: 60, hidden_size: 6144, total_params: 428e9, active_params: 23e9, weights_bytes: 854176398808, compute_dtype: "bf16", block_size: 64, experts: moe({ layers: 60, routed: 128, per_token: 4, width: 6144, inter: 3072 }),
      layers: [
        { type: "full", count: 3, q_heads: 64, kv_heads: 4, k_dim: 128, v_dim: 128 },
        { type: "full", count: 57, q_heads: 64, kv_heads: 4, k_dim: 128, v_dim: 128, topk: 17 * 128, index_heads: 4, index_dim: 128, index_bytes: 128 * 2 },
      ],
    },
    sources: [expertSrc("MiniMaxAI/MiniMax-M3", '"num_local_experts": 128, "num_experts_per_tok": 4, "intermediate_size": 3072, "n_shared_experts": 1'), cfgSrc("MiniMaxAI/MiniMax-M3"), filesSrc("MiniMaxAI/MiniMax-M3", 854176398808),
      cardSrc("MiniMaxAI/MiniMax-M3", "It has ~428B parameters and ~23B activated parameters ... Compared with GQA, MSA dramatically reduces the attention compute and memory footprint")],
    assumptions: ["block_size", "sparse span: 16 top blocks + 1 local block of 128 tokens", "indexer key cache: one bf16 key of 128 per token"],
  },
  "mistral-large-4": {
    name: "Mistral Large 4", hf_id: "mistralai/Mistral-Large-4-1T-A52B", released: "FP8", arch: "MoE 1T / 52B active; MLA sparse attention, 9 layers share KV",
    spec: {
      num_layers: 72, hidden_size: 6144, total_params: 1e12, active_params: 52e9, weights_bytes: 1060554587912, compute_dtype: "fp8", block_size: 64, experts: moe({ layers: 71, routed: 128, per_token: 4, width: 6144, inter: 6144 }),
      layers: [
        { type: "mla", count: 35, q_heads: 64, latent: 512, rope: 64, qk_dim: 128, v_dim: 128, topk: 2048, index_heads: 32, index_dim: 128, index_bytes: INDEX_BYTES, index_fp8: true },
        { type: "mla", count: 28, q_heads: 64, latent: 512, rope: 64, qk_dim: 128, v_dim: 128, topk: 2048 },
        { type: "mla", stores: false, count: 9, q_heads: 64, latent: 512, rope: 64, qk_dim: 128, v_dim: 128, topk: 2048, index_heads: 32, index_dim: 128, index_bytes: INDEX_BYTES, index_fp8: true },
      ],
    },
    sources: [expertSrc("mistralai/Mistral-Large-4-1T-A52B", '"n_routed_experts": 128, "num_experts_per_tok": 4, "moe_intermediate_size": 6144, "first_k_dense_replace": 1'), cfgSrc("mistralai/Mistral-Large-4-1T-A52B"), filesSrc("mistralai/Mistral-Large-4-1T-A52B", 1060554587912),
      cardSrc("mistralai/Mistral-Large-4-1T-A52B", "~1T total / ~52B active parameters: granular Mixture-of-Experts with 128 routed experts (4 active per token) and 1 shared expert"), VLLM_MLA_TP],
    assumptions: ["block_size", "indexer key cache 132 B per token on its 44 own-indexer layers"],
    note: "kv_source_layers points 9 layers at layer 35's KV, so 63 of 72 layers store their own.",
  },
  "gpt-oss-120b": {
    name: "gpt-oss-120b", hf_id: "openai/gpt-oss-120b", released: "MXFP4 experts, BF16 attention", arch: "MoE 117B / 5.1B active; full and 128-token sliding-window GQA, alternating",
    spec: {
      num_layers: 36, hidden_size: 2880, total_params: 116.8e9, active_params: 5.1e9, weights_bytes: 65248893184, compute_dtype: "bf16", block_size: 64, experts: moe({ layers: 36, routed: 128, per_token: 4, width: 2880, inter: 2880, bytes_per_param: MXFP4 }),
      layers: [
        { type: "full", count: 18, q_heads: 64, kv_heads: 8, k_dim: 64, v_dim: 64 },
        { type: "swa", count: 18, q_heads: 64, kv_heads: 8, k_dim: 64, v_dim: 64, window: 128 },
      ],
    },
    sources: [expertSrc("openai/gpt-oss-120b", '"num_local_experts": 128, "num_experts_per_tok": 4, "intermediate_size": 2880, "hidden_size": 2880'), cfgSrc("openai/gpt-oss-120b"), filesSrc("openai/gpt-oss-120b", 65248893184),
      cardSrc("openai/gpt-oss-120b", "fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X) (117B parameters with 5.1B active parameters)")],
    assumptions: ["block_size", "routed experts at MXFP4's 4.25 bits per parameter (the rest of the weight bytes are the dense part)"],
  },
  "gemma-4-31b": {
    name: "Gemma 4 31B", hf_id: "google/gemma-4-31B-it", released: "BF16", arch: "dense 31B; 50 sliding-window (1024) + 10 global layers",
    spec: {
      num_layers: 60, hidden_size: 5376, total_params: 30.7e9, weights_bytes: 62546338248, compute_dtype: "bf16", block_size: 64,
      layers: [
        { type: "full", count: 10, q_heads: 32, kv_heads: 4, k_dim: 512, v_dim: 512, k_eq_v: true },
        { type: "swa", count: 50, q_heads: 32, kv_heads: 16, k_dim: 256, v_dim: 256, window: 1024, k_eq_v: true },
      ],
    },
    sources: [cfgSrc("google/gemma-4-31B-it"), filesSrc("google/gemma-4-31B-it", 62546338248),
      cardSrc("google/gemma-4-31B-it", "Total Parameters 30.7B; Layers 60; Sliding Window 1024 tokens; Context Length 256K tokens")],
    assumptions: ["block_size", "attention_k_eq_v: K and V stored once, on every layer"],
  },
};

// Speculative decoding: a drafter proposes `draft` tokens per sequence, the model checks them (and computes one more)
// in one step, and `tau` tokens are accepted per step on average, the bonus token included. The drafter's cost is
// `layers` layers of the model's own size, run `draft` times one after another (autoregressive) or once (parallel).
// requires: a drafter the checkpoint must ship ("mtp"); the other methods need a separately trained draft model
// (published for many models). A checkpoint that ships one (drafter_params) keeps its own shape. tau is measured on 4-14B dense models at
// temperature 1 on MT-Bench (chat); on large MoE models and other workloads it differs, so it is an assumption there.
const DSPARK_PAPER = "https://arxiv.org/abs/2607.05147";
const DSPARK_TABLE = (method, tau) => src(`DSpark paper, Table 1: ${method} accepts ${tau} tokens per step on Qwen3-8B, MT-Bench, temperature 1.0`, DSPARK_PAPER,
  "we align Eagle3's TTT horizon (7) with the block size (7) ... we set 1 for Eagle3 and 5 for DSpark and DFlash");
export const SPEC_METHODS = [
  { id: "none", name: "None", draft: 0, tau: 1, layers: 0, parallel: false },
  { id: "mtp", name: "MTP, 1 draft token (the model's own head)", draft: 1, tau: 1.875, layers: 1, parallel: false, requires: "mtp",
    sources: [src("DeepSeek-V3 technical report, 5.4.3", "https://arxiv.org/abs/2412.19437", "the acceptance rate of the second token prediction ranges between 85% and 90% ... delivering 1.8 times TPS")],
    note: "tau = 1 + 0.875, the middle of DeepSeek-V3's 85-90% acceptance of the drafted token." },
  { id: "eagle3", name: "EAGLE-3 (separate draft head)", draft: 7, tau: 2.66, layers: 1, parallel: false,
    sources: [DSPARK_TABLE("Eagle3", 2.66), src("EAGLE-3 paper, Table 3: SGLang, H100, LLaMA-Instruct 3.1 8B, chain length 3; throughput 1.81x at batch 2, 1.38x at batch 64", "https://arxiv.org/abs/2503.01840")] },
  { id: "dflash", name: "DFlash (block diffusion drafter)", draft: 7, tau: 3.11, layers: 5, parallel: true,
    sources: [DSPARK_TABLE("DFlash", 3.11), src("DFlash paper, Table 3: SGLang, B200, Qwen3-8B, Math500, block 16; speed-up 5.1x at concurrency 1, 4.5x at 4 and 8, 3.9x at 16, 2.8x at 32", "https://arxiv.org/abs/2602.06036")] },
  { id: "dspark", name: "DSpark (semi-autoregressive drafter)", draft: 7, tau: 3.72, layers: 5, parallel: true,
    sources: [DSPARK_TABLE("DSpark", 3.72), src("DSpark paper: DeepSeek-V4-Flash in production, against the MTP-1 baseline, at matched aggregate throughput", DSPARK_PAPER, "accelerates per-user generation speeds by 60%–85%")],
    note: "Its sequential head is small next to the parallel backbone and is left out." },
  { id: "custom", name: "Custom (set the numbers)", draft: 7, tau: 3, layers: 5, parallel: true,
    note: "For drafters without a preset here (DFlash2 and others): set draft tokens, accepted tokens, draft layers and whether drafting is parallel." },
];

export const MODELS = Object.fromEntries(Object.entries(MODEL_SPECS).map(([id, x]) => {
  const { spec, ...info } = x;
  return [id, { ...info, ...deriveArch(spec), ...(x.kv_layout === "deepseek-v41" ? { read_fixed_bytes: 0, prefix_replay_tokens: 128 } : {}), spec, assumptions: x.assumptions ?? [] }];
}));
