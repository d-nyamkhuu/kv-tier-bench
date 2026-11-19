#!/usr/bin/env python3
"""One-off for s2: the fs rows measured while the fs tier was rooted on the shared filesystem become fs-shared rows.

s2's fs deployment first ran with the fs tier on the 1 TiB shared filesystem (/mnt/fs), when that filesystem was
meant to replace the disk. It then turned out slower than the disk from 16K up, so s2 measures both: fs on the disk,
fs-shared on the filesystem. The vLLM config of those runs is exactly the fs-shared deployment's (same tier type,
threads and root_dir), so their fs-tier rows are fs-shared rows: this renames their tier, id and job. Their
recompute, gpu and cpu rows stay with the fs deployment, and so do the job_start markers (the disk rerun of the fs
points then gets a fresh attempt, i.e. new prompts). Idempotent; keeps a copy of the file it rewrites.

    python3 scripts/relabel-fs-shared.py results/s2.jsonl
"""

import json
import shutil
import sys


def _rename(path_id: str) -> str:
    parts = path_id.split("/")  # stage/model/deployment-or-tier/...
    parts[2] = "fs-shared"
    return "/".join(parts)


def relabel(rows: list[dict], stage: str = "s2") -> tuple[list[dict], int]:
    out, n = [], 0
    for r in rows:
        if r.get("stage") == stage and r.get("tier") == "fs":
            r = {**r, "tier": "fs-shared", "id": _rename(r["id"]), "job": _rename(r["job"])}
            n += 1
        out.append(r)
    return out, n


if __name__ == "__main__":
    path = sys.argv[1]
    rows = [json.loads(line) for line in open(path) if line.strip()]
    out, n = relabel(rows)
    if n:
        shutil.copy2(path, path + ".before-fs-shared-relabel")
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"{path}: {n} fs rows relabelled fs-shared, {len(out)} rows in all")
