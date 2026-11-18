"""Thin async client for one vLLM instance (the OpenAI-compatible API plus the dev endpoints), and the block
size it reports in /metrics."""

from __future__ import annotations

import random
import time

import httpx

from . import prom


def prompt_tokens(seed: int, n: int, vocab_lo: int = 1000, vocab_hi: int = 150000) -> list[int]:
    """Deterministic random token IDs. They are unique per job attempt, so no tier has seen the prefix before.

    The range avoids special and control tokens at both ends of the Qwen vocabulary.
    """
    rng = random.Random(seed)
    return [rng.randrange(vocab_lo, vocab_hi) for _ in range(n)]


class Engine:
    def __init__(
        self, base_url: str, model: str, timeout: float = 600.0, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # a long read timeout for 128K prefills, but a short connect one: an unreachable VM must fail fast
        self.http = httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=10.0), transport=transport
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def metrics(self) -> list[prom.Sample]:
        r = await self.http.get("/metrics")
        r.raise_for_status()
        return prom.parse(r.text)

    async def reset_prefix_cache(self, reset_external: bool = False) -> bool:
        """GPU only by default. With reset_external, the CPU tier is cleared too; secondary tiers are kept."""
        r = await self.http.post("/reset_prefix_cache", params={"reset_external": str(reset_external).lower()})
        r.raise_for_status()
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return bool(body.get("success", True))

    async def complete(
        self,
        tokens: list[int],
        max_tokens: int = 1,
        kv_transfer_params: dict | None = None,
    ) -> tuple[float, dict]:
        """Returns (wall-clock seconds, response JSON). With max_tokens=1, the wall-clock time is about the TTFT."""
        body: dict = {
            "model": self.model,
            "prompt": tokens,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if kv_transfer_params is not None:
            body["kv_transfer_params"] = kv_transfer_params
        t0 = time.perf_counter()
        r = await self.http.post("/v1/completions", json=body)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        return dt, r.json()


def block_size(samples: list[prom.Sample]) -> int:
    """Granularity of offloaded hits: the GPU block size times the offload chunk size, in blocks.

    `vllm:cache_config_info` labels every CacheConfig field (v0.34.0 CacheConfig.metrics_info). For hybrid models
    the block size is enlarged at startup, so `mamba_block_size` is checked too and the larger value wins.
    UNVERIFIED: that the label holds the resolved (enlarged) value, not the CLI value. Compare it with the
    startup log line "Setting attention block size to N tokens" on day 1, and use --block-size until then.
    """
    info = next((s for s in samples if s.name == "vllm:cache_config_info"), None)
    if info is None:
        raise LookupError("vllm:cache_config_info not exposed; pass --block-size")
    sizes = [int(v) for k in ("block_size", "mamba_block_size") if (v := info.label(k) or "").isdigit()]
    if not sizes:
        raise LookupError("cache_config_info has no block_size label; pass --block-size")
    chunk = next(
        (s for s in samples if s.name == "vllm:kv_offload_config_info" and s.label("cpu_blocks_per_chunk")), None
    )
    per_chunk = (
        int(chunk.label("cpu_blocks_per_chunk")) if chunk and chunk.label("cpu_blocks_per_chunk").isdigit() else 1
    )
    return max(sizes) * per_chunk
