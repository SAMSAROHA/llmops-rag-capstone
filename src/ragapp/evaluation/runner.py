"""Run the golden set through the pipeline, judge every answer, log to MLflow, apply the gate."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow

from ..llm import LLMError
from ..pipeline import RAGPipeline
from .gate import GateResult, check_gate
from .judges import Judge

METRICS = ("faithfulness", "answer_correctness", "context_recall")


@dataclass
class EvalRow:
    id: str
    question: str
    ground_truth: str
    answerable: bool
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    answer_correctness: float | None = None
    context_recall: float | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    trace_id: str | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalReport:
    meta: dict
    metrics: dict[str, float | None]
    rows: list[EvalRow]
    errors: int
    gate: GateResult | None = None
    mlflow_run_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "metrics": self.metrics,
            "errors": self.errors,
            "gate": self.gate.to_dict() if self.gate else None,
            "mlflow_run_id": self.mlflow_run_id,
            "rows": [asdict(r) for r in self.rows],
        }

    def to_markdown(self) -> str:
        def fmt(v):
            return "n/a" if v is None else f"{v:.3f}"

        lines = []
        if self.gate:
            lines.append(f"## Release gate: {'PASSED' if self.gate.passed else 'FAILED'}\n")
            lines.append("| Check | Value | Threshold | Result |")
            lines.append("|---|---|---|---|")
            for c in self.gate.checks:
                lines.append(
                    f"| {c.metric} | {fmt(c.value)} | {c.threshold:g} | "
                    f"{'pass' if c.passed else '**FAIL**'} |"
                )
            lines.append("")
        m = self.meta
        lines.append(
            f"Prompt `{m['prompt_name']}/{m['prompt_version']}` (sha `{m['prompt_sha']}`), "
            f"model `{m['model']}`, judge `{m['judge']}`, {m['n_samples']} samples, "
            f"generation ${self.metrics['generation_cost_usd']:.4f}, "
            f"judging ${self.metrics['judge_cost_usd']:.4f}.\n"
        )
        lines.append("| Sample | Faithfulness | Correctness | Context recall | Note |")
        lines.append("|---|---|---|---|---|")
        for r in self.rows:
            note = r.error or "; ".join(f"unsupported: {c}" for c in r.unsupported_claims)
            lines.append(
                f"| {r.id} | {fmt(r.faithfulness)} | {fmt(r.answer_correctness)} | "
                f"{fmt(r.context_recall)} | {note.replace('|', '/')} |"
            )
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "eval_report.json").write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        (directory / "eval_report.md").write_text(self.to_markdown(), encoding="utf-8")
        return directory / "eval_report.json"


def load_golden(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"id", "question", "ground_truth"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{n} is missing {sorted(missing)}")
        row.setdefault("answerable", True)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no samples")
    return rows


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


def run_evaluation(
    pipeline: RAGPipeline,
    judge: Judge,
    golden: list[dict],
    prompt_version: str | None = None,
    gate_config: dict | None = None,
    log_to_mlflow: bool = True,
) -> EvalReport:
    prompt = pipeline.prompts.get(pipeline.prompt_name, prompt_version or pipeline.prompt_version)
    meta = {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_sha": prompt.sha,
        "model": pipeline.llm.model,
        "judge": judge.name,
        "embedder": pipeline.store.embedder.name,
        "top_k": pipeline.top_k,
        "n_samples": len(golden),
    }

    run = (
        mlflow.start_run(run_name=f"eval-{prompt.name}-{prompt.version}-{prompt.sha}")
        if log_to_mlflow
        else None
    )
    try:
        rows: list[EvalRow] = []
        for sample in golden:
            row = EvalRow(
                id=sample["id"],
                question=sample["question"],
                ground_truth=sample["ground_truth"],
                answerable=sample["answerable"],
            )
            rows.append(row)
            # Traces created here are linked to the MLflow run automatically.
            response = pipeline.answer(sample["question"], prompt_version=prompt.version, kind="eval")
            row.answer = response.answer
            row.sources = [c.source for c in response.sources]
            row.latency_ms = response.latency_ms
            row.cost_usd = response.cost.total_usd
            row.trace_id = response.trace_id
            if response.stop_reason in ("refusal", "max_tokens"):
                row.error = f"generation stop_reason={response.stop_reason}"
                continue
            try:
                scores = judge.evaluate(
                    sample["question"],
                    response.answer,
                    [c.text for c in response.sources],
                    sample["ground_truth"],
                    sample["answerable"],
                )
            except LLMError as exc:  # judge refused / truncated: count it, keep going
                row.error = f"judge: {exc}"
                continue
            row.faithfulness = scores.faithfulness
            row.answer_correctness = scores.answer_correctness
            row.context_recall = scores.context_recall
            row.judge_cost_usd = pipeline.prices.cost(scores.model, scores.usage).total_usd
            row.unsupported_claims = [
                c["claim"] for c in scores.details.get("claims", []) if not c["supported"]
            ]

        errors = sum(r.error is not None for r in rows)
        metrics: dict[str, float | None] = {}
        for name in METRICS:
            metrics[f"{name}_mean"] = _mean([getattr(r, name) for r in rows])
        faith = [r.faithfulness for r in rows if r.faithfulness is not None]
        metrics["faithfulness_min"] = min(faith) if faith else None
        metrics["latency_ms_p50"] = _percentile([r.latency_ms for r in rows], 0.5)
        metrics["latency_ms_p95"] = _percentile([r.latency_ms for r in rows], 0.95)
        metrics["generation_cost_usd"] = sum(r.cost_usd for r in rows)
        metrics["avg_cost_per_request_usd"] = metrics["generation_cost_usd"] / len(rows)
        metrics["judge_cost_usd"] = sum(r.judge_cost_usd for r in rows)

        report = EvalReport(meta=meta, metrics=metrics, rows=rows, errors=errors)
        if gate_config is not None:
            report.gate = check_gate(metrics, errors, gate_config)

        if run is not None:
            report.mlflow_run_id = run.info.run_id
            mlflow.log_params(meta)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            mlflow.log_metric("errors", errors)
            mlflow.log_table(
                data={
                    key: [getattr(r, key) for r in rows]
                    for key in (
                        "id",
                        "question",
                        "answer",
                        "ground_truth",
                        *METRICS,
                        "cost_usd",
                        "trace_id",
                        "error",
                    )
                },
                artifact_file="eval_results.json",
            )
            mlflow.log_dict(report.to_dict(), "eval_report.json")
            if report.gate is not None:
                mlflow.set_tag("gate", "passed" if report.gate.passed else "failed")
                for name, threshold in gate_config["thresholds"].items():
                    mlflow.log_param(f"gate.{name}", threshold)
        return report
    finally:
        if run is not None:
            mlflow.end_run()
