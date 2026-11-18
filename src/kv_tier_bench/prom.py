"""Minimal Prometheus text-format parsing, and differences between two scrapes of /metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\d+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label(self, key: str) -> str | None:
        return dict(self.labels).get(key)


def parse(text: str) -> list[Sample]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if not m:
            continue
        name, _, labels, value = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            v = float(value)
        except ValueError:
            continue
        pairs = tuple(sorted(_LABEL.findall(labels or "")))
        out.append(Sample(name, pairs, v))
    return out


def find(samples: list[Sample], name: str, **labels: str) -> float:
    """Sum of every series of `name` (or `name_total`) whose labels include `labels`."""
    names = {name, name + "_total"}
    total = 0.0
    for s in samples:
        if s.name in names and all(s.label(k) == v for k, v in labels.items()):
            total += s.value
    return total


def delta(before: list[Sample], after: list[Sample], name: str, **labels: str) -> float:
    return find(after, name, **labels) - find(before, name, **labels)


def delta_by_label(before: list[Sample], after: list[Sample], name: str, label: str) -> dict[str, float]:
    """Delta of `name` for every value of `label` seen in `after`, e.g. cached tokens for each source."""
    values = {s.label(label) for s in after if s.name in (name, name + "_total") and s.label(label) is not None}
    return {v: delta(before, after, name, **{label: v}) for v in sorted(values)}
