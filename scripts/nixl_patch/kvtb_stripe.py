"""StripedAgent: K NIXL agents behind the single-agent API that vLLM's p2p tier (NixlTransport) uses.

Why: over TCP, all receive work for a NIXL UCX agent lands on one thread of the receiving process. NIXL 1.5.2's
UCX backend advertises only its first worker's address (nixlUcxEngine::getConnInfo returns workers_.front()), and
loadRemoteConnInfo connects every local worker, dedicated ones included, to that one address. The receiver's
worker 0 is a shared worker, polled by the backend's single progress thread, and UCX/TCP emulates the remote write
in software there. So a pull tops out at what one core can receive (~2.8 GiB/s on these VMs), however many sender
threads there are. NIXL main (2026-11) is the same.

K agents per process means K receive threads: agent i on the sender talks only to agent i on the receiver. Each
pull's block list is split into K contiguous stripes, one transfer per agent. The receiver needs no per-pull work:
every agent registers the same memory, and the WRITE lands wherever the descriptors point.

Posting (async_post, the default): each agent has one post thread, and transfer() hands every stripe to its
agent's thread and returns "PROC" at once. NIXL's postXferReq releases the GIL but can take most of a pull to return
(UCP sends inline from the posting thread), so posting the stripes one after another serialized them, and the
caller (vLLM's scheduler thread, under the tier lock) blocked meanwhile. A stripe still posting reads as "PROC".
The agents run in NIXL_THREAD_SYNC_RW so that a post thread and the polling thread may enter one agent together.
In NIXL 1.5.2 the data path (makeXferReq, postXferReq for its whole inline send, getXferStatus, releaseXferReq)
takes the agent's shared lock, so posting never stalls polling or releases. Setup and teardown (registerMem,
deregisterMem, loadRemoteMD, getLocalMD, releasedDlistH) take it exclusively and so wait for a post in flight on
that agent, i.e. at most one stripe's transfer; vLLM registers peers on its own executor thread.

Only the calls NixlTransport makes are implemented. Handles are lists with one entry per agent. NIXL exceptions
propagate unchanged, because vLLM classifies them by class name (nixlRemoteDisconnectError, ...); one raised by an
async post surfaces from check_xfer_state.
"""

from __future__ import annotations

import copy
import struct
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

MAGIC = b"KVTBSTRIPE1"
LOCAL = ("NIXL_INIT_AGENT", "")


def pack_metadata(blobs: Sequence[bytes]) -> bytes:
    out = [MAGIC, struct.pack("<I", len(blobs))]
    for b in blobs:
        out += [struct.pack("<Q", len(b)), b]
    return b"".join(out)


def unpack_metadata(data: bytes) -> list[bytes]:
    if not data.startswith(MAGIC):
        raise ValueError("peer metadata is not striped: run the same KVTB_NIXL_STRIPES on every peer")
    pos = len(MAGIC)
    (n,) = struct.unpack_from("<I", data, pos)
    pos += 4
    blobs = []
    for _ in range(n):
        (size,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        blobs.append(bytes(data[pos : pos + size]))
        pos += size
    if pos != len(data):
        raise ValueError("striped metadata has trailing bytes")
    return blobs


def stripe_bounds(n: int, k: int) -> list[tuple[int, int]]:
    """[start, end) of up to k contiguous, near-equal stripes of n items; empty stripes are dropped."""
    base, extra = divmod(n, k)
    bounds, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < extra else 0)
        if end > start:
            bounds.append((start, end))
        start = end
    return bounds


def _each(items, fn: Callable[[Any], Any]) -> None:
    """Apply fn to every item, then re-raise the first error: one failing release must not skip the others."""
    first = None
    for item in items:
        try:
            fn(item)
        except Exception as exc:  # noqa: BLE001
            first = first or exc
    if first is not None:
        raise first


class _Xfer:
    __slots__ = ("parts", "done", "posts")

    def __init__(self, parts: list[tuple[int, Any]]):
        self.parts = parts  # (agent index, nixl_xfer_handle)
        self.done = [False] * len(parts)
        self.posts: list[Future] | None = None  # async_post: one transfer() future per part


def _rw_config(agent_cls, config):
    """A copy of config with sync_mode NIXL_THREAD_SYNC_RW, when agent_cls's module has the enum (a real NIXL)."""
    enum = getattr(sys.modules.get(getattr(agent_cls, "__module__", "")), "nixl_thread_sync_t", None)
    if enum is None or config is None or not hasattr(config, "sync_mode"):
        return config
    config = copy.copy(config)
    config.sync_mode = enum.NIXL_THREAD_SYNC_RW
    return config


class StripedAgent:
    def __init__(self, agent_cls: Callable[..., Any], name: str, config: Any, stripes: int, async_post: bool = True):
        if stripes < 1:
            raise ValueError("stripes must be >= 1")
        self.name = name
        if async_post:
            config = _rw_config(agent_cls, config)
        self.agents = [agent_cls(f"{name}-s{i}", config) for i in range(stripes)]
        self._remote: dict[str, list[str]] = {}  # composite remote name -> per-agent remote names
        self._lanes = (
            [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"kvtb-post{i}") for i in range(stripes)]
            if async_post
            else None
        )

    # -- setup -----------------------------------------------------------------------------------------------
    def create_backend(self, backend: str, initParams: dict | None = None):
        for a in self.agents:
            a.create_backend(backend, dict(initParams or {}))

    def register_memory(self, reg_list, mem_type=None, backends=()):
        return [a.register_memory(reg_list, mem_type, list(backends)) for a in self.agents]

    def deregister_memory(self, regs, backends=()):
        _each(zip(self.agents, regs), lambda ar: ar[0].deregister_memory(ar[1], list(backends)))

    def get_xfer_descs(self, *args, **kwargs):
        return self.agents[0].get_xfer_descs(*args, **kwargs)

    def prep_xfer_dlist(self, agent_name: str | bytes, xfer_list, mem_type=None, backends=()):
        names = [agent_name] * len(self.agents) if agent_name in LOCAL else self._remote[agent_name]
        return [a.prep_xfer_dlist(n, xfer_list, mem_type, list(backends)) for a, n in zip(self.agents, names)]

    def release_dlist_handle(self, handles):
        _each(zip(self.agents, handles), lambda ah: ah[0].release_dlist_handle(ah[1]))

    # -- peers -----------------------------------------------------------------------------------------------
    def get_agent_metadata(self) -> bytes:
        return pack_metadata([a.get_agent_metadata() for a in self.agents])

    def add_remote_agent(self, metadata: bytes) -> str | bytes:
        blobs = unpack_metadata(metadata)
        if len(blobs) != len(self.agents):
            raise ValueError(f"peer has {len(blobs)} NIXL stripes, this process {len(self.agents)}")
        names: list[str] = []
        try:
            for a, blob in zip(self.agents, blobs):
                names.append(a.add_remote_agent(blob))
        except Exception:
            for a, n in zip(self.agents, names):
                try:
                    a.remove_remote_agent(n)
                except Exception:  # noqa: BLE001
                    pass
            raise
        # nixl_agent returns the name as bytes (str in some builds); the composite keeps the type
        composite = (b"|" if isinstance(names[0], bytes) else "|").join(names)
        self._remote[composite] = names
        return composite

    def remove_remote_agent(self, agent: str | bytes):
        names = self._remote.pop(agent)
        _each(zip(self.agents, names), lambda an: an[0].remove_remote_agent(an[1]))

    # -- transfers -------------------------------------------------------------------------------------------
    def make_prepped_xfer(
        self, operation, local_xfer_side, local_indices, remote_xfer_side, remote_indices, *args, **kwargs
    ) -> _Xfer:
        if len(local_indices) != len(remote_indices):
            raise ValueError("local and remote index lists differ in length")
        parts: list[tuple[int, Any]] = []
        try:
            for i, (s, e) in enumerate(stripe_bounds(len(local_indices), len(self.agents))):
                h = self.agents[i].make_prepped_xfer(
                    operation,
                    local_xfer_side[i],
                    local_indices[s:e],
                    remote_xfer_side[i],
                    remote_indices[s:e],
                    *args,
                    **kwargs,
                )
                parts.append((i, h))
        except Exception:
            for i, h in parts:
                try:
                    self.agents[i].release_xfer_handle(h)
                except Exception:  # noqa: BLE001
                    pass
            raise
        return _Xfer(parts)

    def transfer(self, handle: _Xfer, notif_msg: bytes = b"") -> str:
        if self._lanes is not None:
            handle.posts = [self._lanes[i].submit(self.agents[i].transfer, h, notif_msg) for i, h in handle.parts]
            return "PROC"
        for j, (i, h) in enumerate(handle.parts):
            s = self.agents[i].transfer(h, notif_msg)
            if s == "ERR":
                return "ERR"
            handle.done[j] = s == "DONE"
        return "DONE" if all(handle.done) else "PROC"

    def check_xfer_state(self, handle: _Xfer) -> str:
        for j, (i, h) in enumerate(handle.parts):
            if handle.done[j]:
                continue
            if handle.posts is not None:
                post = handle.posts[j]
                if not post.done():
                    continue  # still posting: in progress
                if post.result() == "ERR":  # result() re-raises a NIXL exception from the post
                    return "ERR"
            s = self.agents[i].check_xfer_state(h)
            if s == "ERR":
                return "ERR"
            handle.done[j] = s == "DONE"
        return "DONE" if all(handle.done) else "PROC"

    def release_xfer_handle(self, handle: _Xfer):
        if handle.posts is not None:  # never release a handle that is still being posted
            for post in handle.posts:
                post.exception()  # waits; the post's own error, if any, was already reported by check_xfer_state
        # nixl_xfer_handle.release() is idempotent, so a retry after a partial failure is safe
        _each(handle.parts, lambda ih: self.agents[ih[0]].release_xfer_handle(ih[1]))


def factory(agent_cls: Callable[..., Any], stripes: int, async_post: bool = True) -> Callable[..., StripedAgent]:
    """A drop-in for the nixl_agent class: factory(nixl_agent, 8)(name, config)."""

    def make(name: str, config: Any = None) -> StripedAgent:
        return StripedAgent(agent_cls, name, config, stripes, async_post)

    return make
