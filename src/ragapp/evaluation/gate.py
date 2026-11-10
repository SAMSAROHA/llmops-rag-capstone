"""Release gate: compare aggregate eval metrics against thresholds in config/gate.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Check:
    metric: str
    value: float | None
    threshold: float
    passed: bool


@dataclass
class GateResult:
    passed: bool
    checks: list[Check]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks]}


def load_gate(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "faithfulness_mean" not in config.get("thresholds", {}):
        raise ValueError(f"{path}: thresholds.faithfulness_mean is required")
    return config


def check_gate(metrics: dict[str, float | None], errors: int, config: dict) -> GateResult:
    checks = []
    for metric, threshold in config["thresholds"].items():
        value = metrics.get(metric)
        # A metric that could not be computed never passes: no silent green builds.
        checks.append(Check(metric, value, threshold, value is not None and value >= threshold))
    max_errors = config.get("max_errors", 0)
    checks.append(Check("errors (max allowed)", float(errors), max_errors, errors <= max_errors))
    return GateResult(all(c.passed for c in checks), checks)
