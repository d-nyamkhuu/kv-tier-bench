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
"""

import importlib.abc
import importlib.util
import os
import sys

_EXTRA = dict(kv.split("=", 1) for kv in os.environ.get("KVTB_NIXL_UCX_PARAMS", "").split(",") if "=" in kv)
_SKIP_MERGE = os.environ.get("KVTB_NIXL_SKIP_DESC_MERGE") == "1"
_MODULES = {"nixl_cu13._api", "nixl_cu12._api", "nixl._api"}


def _patch(module) -> None:
    cls = getattr(module, "nixl_agent", None)
    if cls is None or getattr(cls, "_kvtb_patched", False):
        return
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


if _EXTRA or _SKIP_MERGE:
    sys.meta_path.insert(0, _PatchOnImport())
