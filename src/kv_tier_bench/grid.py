"""Expands configs/experiment.yaml into points, and groups them into jobs.

A point is one measurement: (stage, model, tier, prefix length, rep). Its id is stable, so a re-run skips what is
already done. A job is every point that shares one stored prefix: the same (stage, model, deployment, length).
See docs/plan.md, "Protocol".
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .models import load_shape
from .tierconfig import DEPLOYMENT_FOR_TIER

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
FIT_FRACTION = 0.8  # leave a fifth of each tier free: written prompts, padding, fragmentation
CPU_TIERS = {"cpu", "fs", "obj", "p2p"}  # loads into the GPU are staged through the CPU tier
# Order of the reads within one rep. Each read is isolated by a reset first (measure.isolate), so the order only
# matters for cost: reading GPU before the resets that drop it saves a re-promotion.
TIER_ORDER = ["gpu", "cpu", "fs", "obj", "p2p", "recompute"]


@dataclass(frozen=True)
class Point:
    stage: str
    model: str
    tier: str
    target_tokens: int
    rep: int

    @property
    def id(self) -> str:
        return f"{self.stage}/{self.model}/{self.tier}/{self.target_tokens}/r{self.rep}"

    def as_dict(self) -> dict:
        return asdict(self) | {"id": self.id}


@dataclass(frozen=True)
class Job:
    stage: str
    model: str
    deployment: str
    target_tokens: int

    @property
    def id(self) -> str:
        return f"{self.stage}/{self.model}/{self.deployment}/{self.target_tokens}"

    def seed(self, attempt: int = 0) -> int:
        """Seed for the job's prompt: unique per job and attempt, the same on every re-run.

        A new attempt needs a fresh prompt, because the old one is already cached, so its write would no longer
        be a cold prefill.
        """
        return int.from_bytes(hashlib.sha256(f"{self.id}#a{attempt}".encode()).digest()[:8], "big")


def job_of(p: Point) -> Job:
    return Job(p.stage, p.model, DEPLOYMENT_FOR_TIER[p.tier], p.target_tokens)


def group_jobs(points: list[Point]) -> dict[Job, list[Point]]:
    """Points by job, each job's points ordered by rep and then by TIER_ORDER (the order they are read in)."""
    out: dict[Job, list[Point]] = {}
    for p in points:
        out.setdefault(job_of(p), []).append(p)
    return {j: sorted(ps, key=lambda p: (p.rep, TIER_ORDER.index(p.tier))) for j, ps in out.items()}


def load_experiment(path: str | Path = "configs/experiment.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def gpu_kv_budget(exp: dict, stage: str, model: str) -> float:
    """Bytes of GPU KV cache for one instance: usable HBM minus bf16 weights minus 3 GiB/GPU of activations."""
    st = exp["stages"][stage]
    gpus = st["gpus_per_instance"]
    usable = st["gpu_mem_gib"] * GIB * st["gpu_memory_utilization"] * gpus
    return usable - exp["models"][model]["params"] * 2 - 3 * GIB * gpus


def fits(exp: dict, stage: str, model: str, tier: str, tokens: int, root: Path = ROOT) -> bool:
    """Whether a prefix of `tokens` fits the tiers the point uses, with room to spare.

    A prefix that does not fit evicts itself before it is read back, so every attempt would be invalid. Such
    points are left out of the grid instead of being retried.
    """
    st = exp["stages"][stage]
    shape = load_shape(model, root / exp["models"][model]["config"])
    need = shape.prefix_bytes(tokens, tp=st["tp"])
    if need > FIT_FRACTION * gpu_kv_budget(exp, stage, model):
        return False
    return tier not in CPU_TIERS or need <= FIT_FRACTION * st["cpu_tier_gib"] * GIB


def expand(exp: dict, stage: str) -> list[Point]:
    st = exp["stages"][stage]
    points = []
    for model in exp["models"]:
        cap = st.get("max_prefix_tokens", {}).get(model)
        for tier in st["tiers"]:
            reps = min(exp["reps"], exp.get("recompute_reps", exp["reps"])) if tier == "recompute" else exp["reps"]
            for n in exp["prefix_tokens"]:
                if cap and n > cap:
                    continue
                for rep in range(reps):
                    points.append(Point(stage, model, tier, n, rep))
    return [p for p in points if fits(exp, p.stage, p.model, p.tier, p.target_tokens)]


def align_down(tokens: int, block_size: int) -> int:
    """Prefix hits only land on block boundaries, so measured lengths are whole blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(block_size, tokens // block_size * block_size)
