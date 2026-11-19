"""Adds NIXL UCX backend params that vLLM v0.34.0 does not expose. Mounted into the vLLM container and put on
PYTHONPATH, so it runs at interpreter start in every vLLM process. It replaces Ubuntu's own sitecustomize, which
only installs the apport crash hook.

KVTB_NIXL_UCX_PARAMS="split_batch_size=8,..." is merged into every create_backend("UCX", ...) call. Why: vLLM's
p2p tier sends one descriptor per KV block, and NIXL's thread-pool engine splits a transfer across its `num_threads`
workers (each with its own TCP connection to the peer) only at >= split_batch_size descriptors, default 1024. A
128K prefix is 292 blocks, so every pull would otherwise use one TCP stream.

KVTB_NIXL_SKIP_DESC_MERGE=1 also passes skip_desc_merge=True to make_prepped_xfer. NIXL merges contiguous
descriptors by default, and a fresh prefix's blocks are contiguous in the CPU tier on both peers, so a whole pull
collapses to about one descriptor and is never split, whatever split_batch_size is.

KVTB_NIXL_CUDA_MAJOR=12 makes the `nixl` meta package load its CUDA 12 build (nixl_cu12) instead of the one matching
the process's CUDA (13). Why: in the v0.34.0 image, nixl_cu13's OBJ plugin links libcuobjclient.so.1, which the
image lacks, so the obj tier fails at startup with NIXL_ERR_NOT_FOUND; nixl_cu12's OBJ plugin has no such
dependency. The obj tier moves host memory only.

KVTB_NIXL_PTHR_DELAY_US=N sets the agent's progress-thread poll timeout (nixlAgentConfig.pthrDelay). Why: the
Python nixl_agent hardcodes pthrDelay = 0, and the UCX backend's shared progress thread passes ceil(0 us) = 0 ms to
poll(), so it never sleeps: every agent burns a full core even when idle, and in the poster's way it holds the UCX
worker lock almost continuously. With N >= 1 the thread blocks in poll() until the worker's event fd fires (N is
only the fallback wakeup).

KVTB_NIXL_STRIPES=K (K > 1) gives vLLM's p2p tier K NIXL agents instead of one, and splits every pull across them
(kvtb_stripe.py). Why: over TCP, a NIXL UCX agent receives on one thread whatever the sender does, so a pull is
capped at what one core can receive. Every peer must use the same K.
"""

import importlib.abc
import importlib.util
import os
import sys

_EXTRA = dict(kv.split("=", 1) for kv in os.environ.get("KVTB_NIXL_UCX_PARAMS", "").split(",") if "=" in kv)
_SKIP_MERGE = os.environ.get("KVTB_NIXL_SKIP_DESC_MERGE") == "1"
_CUDA_MAJOR = os.environ.get("KVTB_NIXL_CUDA_MAJOR")
_STRIPES = int(os.environ.get("KVTB_NIXL_STRIPES", "1"))
_PTHR_DELAY_US = os.environ.get("KVTB_NIXL_PTHR_DELAY_US")
_P2P_NIXL = "vllm.v1.kv_offload.tiering.p2p.data.nixl"
_MODULES = (
    {"nixl_cu13._api", "nixl_cu12._api", "nixl._api"}
    | ({"nixl_meta_utils"} if _CUDA_MAJOR else set())
    | ({_P2P_NIXL} if _STRIPES > 1 else set())
)


def _patch(module) -> None:
    if module.__name__ == "nixl_meta_utils":
        module.detect_cuda_major = lambda: int(_CUDA_MAJOR)
        print(f"kvtb nixl_patch: NIXL CUDA build forced to {_CUDA_MAJOR}", file=sys.stderr, flush=True)
        return
    if module.__name__ == _P2P_NIXL:
        if module._NixlAgent is not None:  # None when NIXL is missing: leave vLLM's own handling alone
            import kvtb_stripe

            async_post = os.environ.get("KVTB_NIXL_ASYNC_POST", "1") == "1"
            module._NixlAgent = kvtb_stripe.factory(module._NixlAgent, _STRIPES, async_post)
            print(f"kvtb nixl_patch: p2p tier striped over {_STRIPES} NIXL agents", file=sys.stderr, flush=True)
        return
    cls = getattr(module, "nixl_agent", None)
    if cls is None or getattr(cls, "_kvtb_patched", False):
        return
    if _PTHR_DELAY_US is not None:
        module.nixlBind = _BindWithDelay(module.nixlBind, int(_PTHR_DELAY_US))
        print(f"kvtb nixl_patch: progress-thread poll timeout {_PTHR_DELAY_US} us", file=sys.stderr, flush=True)
    create_backend = cls.create_backend

    def patched(self, backend, initParams=None):
        params = dict(initParams or {})
        if backend == "UCX":
            params.update(_EXTRA)
            print(f"kvtb nixl_patch: UCX backend params {params}", file=sys.stderr, flush=True)
        return create_backend(self, backend, params)

    cls.create_backend, cls._kvtb_patched = patched, True
    if _SKIP_MERGE:
        make_prepped_xfer = cls.make_prepped_xfer

        def no_merge(self, *args, **kwargs):
            kwargs["skip_desc_merge"] = True
            return make_prepped_xfer(self, *args, **kwargs)

        cls.make_prepped_xfer = no_merge
        print("kvtb nixl_patch: make_prepped_xfer with skip_desc_merge=True", file=sys.stderr, flush=True)


class _BindWithDelay:
    """The nixl bindings module, except that nixlAgent(name, config) gets config.pthrDelay = delay_us first."""

    def __init__(self, bind, delay_us: int):
        self._bind, self._delay_us = bind, delay_us

    def __getattr__(self, name):
        return getattr(self._bind, name)

    def nixlAgent(self, name, config):
        config.pthrDelay = self._delay_us
        return self._bind.nixlAgent(name, config)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name not in _MODULES:
            return None
        sys.meta_path.remove(self)  # let the normal finders resolve it
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if _EXTRA or _SKIP_MERGE or _CUDA_MAJOR or _STRIPES > 1 or _PTHR_DELAY_US is not None:
    sys.meta_path.insert(0, _PatchOnImport())
