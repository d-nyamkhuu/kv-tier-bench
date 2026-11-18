"""Append-only JSONL result store.

Every attempt at a point is kept, valid or not. A point is done once one attempt is valid, or after
`max_attempts` invalid ones. Rows with `kind: job_start` mark the start of a job attempt (see run.py); they
have no `valid` field, so analysis skips them. A transient failure (for example a race just after a reset) is retried with a fresh
prompt instead of losing the point for good.
"""

from __future__ import annotations

import json
from pathlib import Path


class ResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # a torn last line (no newline) would swallow the next row; close it off first
        if self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n"):
            with self.path.open("a") as f:
                f.write("\n")
        self._valid: set[str] = set()
        self._attempts: dict[str, int] = {}
        for row in self.rows():
            self._index(row)

    def _index(self, row: dict) -> None:
        self._attempts[row["id"]] = self._attempts.get(row["id"], 0) + 1
        if row.get("valid"):
            self._valid.add(row["id"])

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # a preempted run can leave a torn last line; the point is simply re-run
                continue
        return out

    def is_done(self, point_id: str, max_attempts: int = 1) -> bool:
        return point_id in self._valid or self._attempts.get(point_id, 0) >= max_attempts

    def next_attempt(self, point_id: str) -> int:
        return self._attempts.get(point_id, 0)

    def append(self, row: dict) -> None:
        if "id" not in row:
            raise ValueError("row needs an id")
        with self.path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        self._index(row)
