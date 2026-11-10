"""Per-request cost: token usage x price table, persisted to a SQLite ledger."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .llm import Usage


@dataclass
class Cost:
    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_usd: float = 0.0
    priced: bool = True  # False => model missing from the price table, cost is unknown

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.output_usd + self.cache_usd

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            self.input_usd + other.input_usd,
            self.output_usd + other.output_usd,
            self.cache_usd + other.cache_usd,
            self.priced and other.priced,
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "total_usd": self.total_usd}


class PriceTable:
    """USD per million tokens, loaded from config/pricing.yaml."""

    def __init__(self, models: dict[str, dict[str, float]]):
        self.models = models

    @classmethod
    def from_file(cls, path: str | Path) -> "PriceTable":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["models"])

    def _lookup(self, model: str) -> dict[str, float] | None:
        if model in self.models:
            return self.models[model]
        # Tolerate suffixed ids (dated snapshots, provider prefixes stripped upstream).
        matches = [m for m in self.models if model.startswith(m)]
        return self.models[max(matches, key=len)] if matches else None

    def cost(self, model: str, usage: Usage) -> Cost:
        price = self._lookup(model)
        if price is None:
            return Cost(priced=False)
        cache_read = price.get("cache_read", price["input"] * 0.1)
        cache_write = price.get("cache_write", price["input"] * 1.25)
        return Cost(
            input_usd=usage.input_tokens * price["input"] / 1e6,
            output_usd=usage.output_tokens * price["output"] / 1e6,
            cache_usd=(
                usage.cache_read_input_tokens * cache_read
                + usage.cache_creation_input_tokens * cache_write
            )
            / 1e6,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    priced INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    trace_id TEXT
)
"""


class CostLedger:
    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()

    def record(
        self,
        request_id: str,
        model: str,
        usage: Usage,
        cost: Cost,
        latency_ms: float,
        prompt_version: str | None = None,
        trace_id: str | None = None,
        kind: str = "query",
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    time.time(),
                    kind,
                    model,
                    prompt_version,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens,
                    usage.cache_creation_input_tokens,
                    cost.total_usd,
                    int(cost.priced),
                    latency_ms,
                    trace_id,
                ),
            )
            self._db.commit()

    def get(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        return dict(row) if row else None

    def summary(self) -> dict:
        query = """
            SELECT model, prompt_version, kind,
                   COUNT(*) AS requests,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(cost_usd) AS cost_usd,
                   AVG(cost_usd) AS avg_cost_usd,
                   AVG(latency_ms) AS avg_latency_ms
            FROM requests GROUP BY model, prompt_version, kind ORDER BY cost_usd DESC
        """
        with self._lock:
            groups = [dict(r) for r in self._db.execute(query).fetchall()]
        return {
            "total_requests": sum(g["requests"] for g in groups),
            "total_cost_usd": sum(g["cost_usd"] for g in groups),
            "groups": groups,
        }

    def close(self) -> None:
        self._db.close()
