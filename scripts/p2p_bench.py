"""Standalone two-VM NIXL WRITE benchmark: the p2p tier's transfer pattern without vLLM.

Runs inside the vLLM image (same NIXL and UCX), one process per VM:
  receiver (peer):  python3 p2p_bench.py recv --port 7790 [agent opts]
  sender (writer):  python3 p2p_bench.py send --peer 10.88.0.63:7790 [agent opts] --blocks 292 37
The sender pushes `--blocks` descriptors of `--block-mib` each with make_prepped_xfer("WRITE", ...), like vLLM's
NixlTransport.write_blocks, into the receiver's registered DRAM buffer, and reports GiB/s per rep. Both sides report
per-thread CPU over the transfer window (the top -H view). The receiver checks that each sent block landed at the
remote index it was paired with, with a per-rep tag, so a stripe that mispairs indices fails loudly.

Agent options must match on both sides: --stripes (kvtb_stripe), --threads (NIXL UCX num_threads), --split
(split_batch_size). UCX_* env vars pass straight through to UCX.
"""

from __future__ import annotations

import argparse
import base64
import json
import mmap
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nixl_patch"))
import kvtb_stripe  # noqa: E402

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:  # the meta package needs a visible GPU to pick its build
    from nixl_cu13._api import nixl_agent, nixl_agent_config

MIB = 1 << 20
WORDS = (0, 0.5, 1)  # sampled positions in each block (fractions), tagged per rep and checked by the receiver


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("<Q", len(data)) + data)


def recv_msg(sock: socket.socket) -> dict:
    def exact(n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed")
            buf += chunk
        return buf

    (n,) = struct.unpack("<Q", exact(8))
    return json.loads(exact(n))


def thread_cpu() -> dict[str, float]:
    """CPU seconds per thread of this process, keyed 'tid:name'."""
    out, tick = {}, os.sysconf("SC_CLK_TCK")
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                stat = f.read()
        except FileNotFoundError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        out[f"{tid}:{name}"] = (int(fields[11]) + int(fields[12])) / tick
    return out


def cpu_report(before: dict[str, float], after: dict[str, float], wall: float, top: int = 8) -> dict:
    busy = sorted(((after[k] - before.get(k, 0.0)) / wall * 100, k) for k in after)
    busy.reverse()
    return {
        "total_pct": round(sum(b for b, _ in busy), 1),
        "threads_over_50pct": sum(1 for b, _ in busy if b > 50),
        "top": [(k, round(b)) for b, k in busy[:top]],
    }


class Buffer:
    def __init__(self, gib: float, block_mib: float):
        self.block_len = int(block_mib * MIB)
        self.num_blocks = int(gib * 1024 * MIB) // self.block_len
        self.size = self.num_blocks * self.block_len
        self.mm = mmap.mmap(-1, self.size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.arr = np.frombuffer(self.mm, dtype=np.uint64)
        self.arr[:] = 0x5A5A5A5A5A5A5A5A  # fault every page in, as the CPU tier is
        self.base = int(np.frombuffer(self.mm, dtype=np.uint8).ctypes.data)
        self.words_per_block = self.block_len // 8

    def positions(self, block: int) -> list[int]:
        last = self.words_per_block - 1
        return [block * self.words_per_block + int(f * last) for f in WORDS]

    def tag(self, local: list[int], rep_tag: int) -> None:
        for j in local:
            for p in self.positions(j):
                self.arr[p] = (rep_tag << 32) | j

    def check(self, pairs: list[tuple[int, int]], rep_tag: int) -> int:
        bad = 0
        for j, r in pairs:
            want = (rep_tag << 32) | j
            bad += any(int(self.arr[p]) != want for p in self.positions(r))
        return bad


def make_agent(name: str, args):
    cfg = nixl_agent_config(backends=[], capture_telemetry=False)
    agent = kvtb_stripe.StripedAgent(nixl_agent, name, cfg, args.stripes, async_post=args.async_post)
    params = {"engine_config": "TLS=^rc_gda"}
    if args.threads > 0:
        params["num_threads"] = str(args.threads)
        params["split_batch_size"] = str(args.split)
    if args.backend == "UCX":
        agent.create_backend("UCX", params)
    else:
        agent.create_backend(args.backend, {})
    return agent


def setup(name: str, args, buf: Buffer):
    t0 = time.perf_counter()
    agent = make_agent(name, args)
    reg = agent.register_memory([(buf.base, buf.size, 0, "")], "DRAM")
    descs = agent.get_xfer_descs(
        [(buf.base + i * buf.block_len, buf.block_len, 0) for i in range(buf.num_blocks)], "DRAM"
    )
    local = agent.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
    print(f"[{name}] agent ready in {time.perf_counter() - t0:.1f}s: {buf.num_blocks} x {buf.block_len / MIB:.0f} MiB",
          flush=True)
    return agent, reg, local


def maybe_cuda(args) -> None:
    if args.cuda:  # a CUDA context, as in vLLM, so UCX's cuda transports behave as there
        import torch

        torch.zeros(1, device="cuda")


def run_recv(args) -> None:
    maybe_cuda(args)
    buf = Buffer(args.reg_gib, args.block_mib)
    agent, reg, local = setup("recv", args, buf)
    srv = socket.create_server(("0.0.0.0", args.port))
    print(f"[recv] listening on {args.port}", flush=True)
    sock, _ = srv.accept()
    send_msg(sock, {"md": base64.b64encode(agent.get_agent_metadata()).decode(), "base": buf.base,
                    "num_blocks": buf.num_blocks, "block_len": buf.block_len})
    while True:
        msg = recv_msg(sock)
        if msg["op"] == "begin":
            cpu0, t0 = thread_cpu(), time.perf_counter()
            send_msg(sock, {})
        elif msg["op"] == "end":
            wall = time.perf_counter() - t0
            rep = cpu_report(cpu0, thread_cpu(), wall)
            bad = buf.check([tuple(p) for p in msg["pairs"]], msg["tag"])
            send_msg(sock, {"cpu": rep, "bad_blocks": bad})
        elif msg["op"] == "quit":
            break
    agent.release_dlist_handle(local)
    agent.deregister_memory(reg)


def run_send(args) -> None:
    maybe_cuda(args)
    buf = Buffer(args.reg_gib, args.block_mib)
    agent, reg, local = setup("send", args, buf)
    host, port = args.peer.rsplit(":", 1)
    for _ in range(100):
        try:
            sock = socket.create_connection((host, int(port)))
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise SystemExit("receiver not reachable")
    peer = recv_msg(sock)
    t0 = time.perf_counter()
    remote_name = agent.add_remote_agent(base64.b64decode(peer["md"]))
    rdescs = agent.get_xfer_descs(
        [(peer["base"] + i * peer["block_len"], peer["block_len"], 0) for i in range(peer["num_blocks"])], "DRAM"
    )
    remote = agent.prep_xfer_dlist(remote_name, rdescs)
    print(f"[send] peer added in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.idle_s:
        cpu0 = thread_cpu()
        time.sleep(args.idle_s)
        print("IDLE " + json.dumps(cpu_report(cpu0, thread_cpu(), args.idle_s)), flush=True)
    results = []
    tag = 0
    for n in args.blocks:
        n = min(n, buf.num_blocks, peer["num_blocks"])
        for rep in range(args.warmup + args.reps):
            tag += 1
            off = (tag * 37) % peer["num_blocks"]
            li = list(range(n))
            ri = [(off + j) % peer["num_blocks"] for j in li]
            buf.tag(li, tag)
            send_msg(sock, {"op": "begin"})
            recv_msg(sock)
            cpu0 = thread_cpu()
            t0 = time.perf_counter()
            h = agent.make_prepped_xfer("WRITE", local, np.asarray(li, np.int32), remote, np.asarray(ri, np.int32),
                                        skip_desc_merge=args.skip_merge)
            state = agent.transfer(h)
            t_post = time.perf_counter() - t0
            stripe_done = [None] * len(h.parts)  # per-stripe completion time, to spot serialized stripes
            while state == "PROC":
                if args.poll_us:
                    time.sleep(args.poll_us / 1e6)
                state = agent.check_xfer_state(h)
                now = time.perf_counter() - t0
                for j, d in enumerate(h.done):
                    if d and stripe_done[j] is None:
                        stripe_done[j] = round(now, 3)
            dt = time.perf_counter() - t0
            mine = cpu_report(cpu0, thread_cpu(), dt)
            agent.release_xfer_handle(h)
            send_msg(sock, {"op": "end", "pairs": list(zip(li, ri)), "tag": tag})
            theirs = recv_msg(sock)
            gib = n * buf.block_len / (1 << 30)
            row = {"blocks": n, "gib": round(gib, 2), "s": round(dt, 3), "gib_s": round(gib / dt, 2),
                   "post_s": round(t_post, 4), "stripe_done_s": stripe_done,
                   "state": state, "bad_blocks": theirs["bad_blocks"], "warmup": rep < args.warmup,
                   "send_cpu": mine, "recv_cpu": theirs["cpu"]}
            results.append(row)
            print(json.dumps(row), flush=True)
    send_msg(sock, {"op": "quit"})
    agent.release_dlist_handle(remote)
    agent.release_dlist_handle(local)
    agent.remove_remote_agent(remote_name)
    agent.deregister_memory(reg)
    meas = [r for r in results if not r["warmup"]]
    summary = {"variant": args.label, "stripes": args.stripes, "threads": args.threads, "split": args.split,
               "ucx_env": {k: v for k, v in os.environ.items() if k.startswith("UCX_")}}
    for n in sorted({r["blocks"] for r in meas}):
        rows = [r for r in meas if r["blocks"] == n]
        summary[f"gib_s@{n}"] = round(float(np.median([r["gib_s"] for r in rows])), 2)
        summary[f"recv_hot_threads@{n}"] = rows[-1]["recv_cpu"]["threads_over_50pct"]
    summary["all_valid"] = all(r["state"] == "DONE" and r["bad_blocks"] == 0 for r in results)
    print("SUMMARY " + json.dumps(summary), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["send", "recv"])
    p.add_argument("--peer", help="receiver host:port (send)")
    p.add_argument("--port", type=int, default=7790)
    p.add_argument("--stripes", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--split", type=int, default=16)
    p.add_argument("--backend", default="UCX")
    p.add_argument("--skip-merge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reg-gib", type=float, default=8)
    p.add_argument("--block-mib", type=float, default=28)
    p.add_argument("--blocks", type=int, nargs="+", default=[292, 37])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--poll-us", type=float, default=0, help="sleep between completion checks (0 = spin)")
    p.add_argument("--idle-s", type=float, default=0, help="before sending, idle this long (idle-spin check)")
    p.add_argument("--async-post", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--label", default="")
    args = p.parse_args()
    (run_send if args.role == "send" else run_recv)(args)


if __name__ == "__main__":
    main()
