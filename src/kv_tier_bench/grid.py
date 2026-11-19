"""Expands configs/experiment.yaml into points, and groups them into jobs.

A point is one measurement: (stage, model, tier, prefix length, rep, concurrency). Its id is stable, so a re-run
skips what is already done. A job is every point that shares its stored prefixes: the same (stage, model,
deployment, length, concurrency). At concurrency c, a job stores c distinct prefixes of the same length and each
point reads all c back at once, one request per prefix. Concurrency 1 is s1's one request at a time, and its ids
carry no concurrency part. See docs/plan.md, "Protocol".
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .models import load_shape
from .tierconfig import STORE_DEPLOYMENTS as STORE_TIERS
from .tierconfig import deployment_for

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
FIT_FRACTION = 0.8  # leave a fifth of each tier free: written prompts, padding, fragmentation
CPU_TIERS = {"cpu", "fs", "fs-shared", "obj", "obj-std", "p2p"}  # loads into the GPU are staged through the CPU tier
# Order of the reads within one rep. Each read is isolated by a reset first (measure.isolate), so the order only
# matters for cost: reading GPU before the resets that drop it saves a re-promotion.
TIER_ORDER = ["gpu", "cpu", "fs", "fs-shared", "obj", "obj-std", "p2p", "store", "store-p2p", "recompute"]


@dataclass(frozen=True)
class Point:
    stage: str
    model: str
    tier: str
    target_tokens: int
    rep: int
    concurrency: int = 1  # requests read at once, each with its own prefix of target_tokens

    @property
    def id(self) -> str:
        return f"{self.stage}/{self.model}/{self.tier}/{self.target_tokens}/r{self.rep}{_conc(self.concurrency)}"

    def as_dict(self) -> dict:
        return asdict(self) | {"id": self.id}


@dataclass(frozen=True)
class Job:
    stage: str
    model: str
    deployment: str
    target_tokens: int
    concurrency: int = 1

    @property
    def id(self) -> str:
        return f"{self.stage}/{self.model}/{self.deployment}/{self.target_tokens}{_conc(self.concurrency)}"

    def seed(self, attempt: int = 0) -> int:
        """Seed for the job's prompt: unique per job and attempt, the same on every re-run.

        A new attempt needs a fresh prompt, because the old one is already cached, so its write would no longer
        be a cold prefill.
        """
        return int.from_bytes(hashlib.sha256(f"{self.id}#a{attempt}".encode()).digest()[:8], "big")

    def prompt_seeds(self, attempt: int = 0) -> list[int]:
        """One seed per prefix of the job: distinct, or the batch would share one cached prefix. The first is
        `seed(attempt)`, so a one-request job's prompt is what it always was."""
        rest = [
            int.from_bytes(hashlib.sha256(f"{self.id}#a{attempt}#i{i}".encode()).digest()[:8], "big")
            for i in range(1, self.concurrency)
        ]
        return [self.seed(attempt), *rest]


def _conc(c: int) -> str:
    return "" if c == 1 else f"/c{c}"


def job_of(p: Point) -> Job:
    return Job(p.stage, p.model, deployment_for(p.stage, p.tier), p.target_tokens, p.concurrency)


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


def stage_models(exp: dict, stage: str) -> list[str]:
    """The models a stage measures: its own list, or every model."""
    return list(exp["stages"][stage].get("models") or exp["models"])


def stage_setting(exp: dict, stage: str, key: str):
    """A stage's own value for a sweep setting (prefix_tokens, reps, recompute_reps), or the top-level one."""
    return exp["stages"][stage].get(key, exp.get(key))


def fits(exp: dict, stage: str, model: str, tier: str, tokens: int, root: Path = ROOT, concurrency: int = 1) -> bool:
    """Whether `concurrency` prefixes of `tokens` fit the tiers the point uses at once, with room to spare.

    A prefix that does not fit evicts itself (or a batch-mate) before it is read back, so every attempt would be
    invalid. Such points are left out of the grid instead of being retried. A batch is all in GPU memory at once
    whatever tier it is read from, and in the CPU tier when it is loaded through it.
    """
    st = exp["stages"][stage]
    shape = load_shape(model, root / exp["models"][model]["config"])
    need = concurrency * shape.prefix_bytes(tokens, tp=st["tp"])
    if need > FIT_FRACTION * gpu_kv_budget(exp, stage, model):
        return False
    if tier in STORE_TIERS:  # the store VM's pool, or store-p2p's holder (s3)
        pool = st["store"]["holder_pool_gib" if tier == "store-p2p" else "pool_gib"]
        return need <= FIT_FRACTION * pool * GIB
    return tier not in CPU_TIERS or need <= FIT_FRACTION * st["cpu_tier_gib"] * GIB


def expand(exp: dict, stage: str) -> list[Point]:
    st = exp["stages"][stage]
    all_reps = stage_setting(exp, stage, "reps")
    recompute_reps = min(all_reps, stage_setting(exp, stage, "recompute_reps") or all_reps)
    points = []
    for model in stage_models(exp, stage):
        cap = st.get("max_prefix_tokens", {}).get(model)
        for tier in st["tiers"]:
            reps = recompute_reps if tier == "recompute" else all_reps
            for n in stage_setting(exp, stage, "prefix_tokens"):
                if cap and n > cap:
                    continue
                for c in st.get("concurrency", [1]):
                    for rep in range(reps):
                        points.append(Point(stage, model, tier, n, rep, c))
    return [p for p in points if fits(exp, p.stage, p.model, p.tier, p.target_tokens, concurrency=p.concurrency)]


def align_down(tokens: int, block_size: int) -> int:
    """Prefix hits only land on block boundaries, so measured lengths are whole blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(block_size, tokens // block_size * block_size)
