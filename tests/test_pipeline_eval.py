from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from ragapp.api import create_app
from ragapp.evaluation import HeuristicJudge, LLMJudge, check_gate, load_gate, load_golden
from ragapp.evaluation.judges import (
    ClaimVerdict,
    CorrectnessOutput,
    FaithfulnessOutput,
    RecallOutput,
    StatementVerdict,
)
from ragapp.evaluation.runner import run_evaluation
from ragapp.llm import FakeLLM, LLMError, LLMResult, Usage

ROOT = Path(__file__).resolve().parents[1]
GATE = load_gate(ROOT / "config" / "gate.yaml")
# The offline stack (lexical fake LLM + heuristic judge) only proves the plumbing,
# so it is held to the faithfulness checks, not the answer-quality thresholds.
OFFLINE_GATE = load_gate(ROOT / "config" / "gate.offline.yaml")
GOLDEN = load_golden(ROOT / "data" / "eval" / "golden.jsonl")


# ----------------------------------------------------------------- pipeline


def test_answer_is_grounded_costed_and_recorded(make_pipeline):
    pipeline = make_pipeline()
    result = pipeline.answer("How much does data egress cost?")
    assert "$0.07 per GB" in result.answer
    assert result.sources[0].source == "pricing.md"
    assert result.prompt.version == "v2"
    assert result.usage.input_tokens > 0

    row = pipeline.ledger.get(result.request_id)
    assert row["prompt_version"] == "v2" and row["model"] == "fake-extractive"
    assert row["input_tokens"] == result.usage.input_tokens
    assert row["trace_id"] == result.trace_id


def test_request_produces_a_tagged_trace(make_pipeline):
    result = make_pipeline().answer("How long are audit logs retained on the Enterprise plan?")
    assert result.trace_id
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(result.trace_id)
    assert {s.name for s in trace.data.spans} == {"rag_request", "retrieve", "generate"}
    tags = trace.info.tags
    assert tags["prompt.version"] == "v2" and tags["prompt.sha"] == result.prompt.sha
    assert float(tags["cost_usd"]) == pytest.approx(result.cost.total_usd)
    generate = next(s for s in trace.data.spans if s.name == "generate")
    assert generate.attributes["mlflow.chat.tokenUsage"]["output_tokens"] == result.usage.output_tokens


def test_prompt_version_override(make_pipeline):
    assert make_pipeline().answer("What is the SLA?", prompt_version="v1").prompt.version == "v1"


def test_refusal_is_replaced_with_safe_message(make_pipeline):
    class Refuser(FakeLLM):
        def generate(self, system, user):
            return LLMResult(text="", model=self.model, stop_reason="refusal")

    assert "can't help" in make_pipeline(Refuser()).answer("anything").answer


# -------------------------------------------------------------------- judges


def test_heuristic_judge_separates_grounded_from_invented():
    judge = HeuristicJudge()
    context = ["API keys expire after 90 days by default."]
    good = judge.evaluate("q", "API keys expire after 90 days by default [1].", context, "90 days", True)
    bad = judge.evaluate("q", "Keys are engraved on titanium plates in Geneva.", context, "90 days", True)
    abstain = judge.evaluate("q", "I don't know based on the provided documents.", context, "n/a", False)
    assert good.faithfulness == 1.0 and bad.faithfulness == 0.0
    assert abstain.faithfulness == 1.0 and abstain.answer_correctness == 1.0
    assert abstain.context_recall is None


class ScriptedLLM:
    """Returns canned structured outputs keyed by schema, like a judge model would."""

    model = "claude-opus-5-5"

    def __init__(self, outputs):
        self.outputs = outputs
        self.seen = []

    def structured(self, system, user, schema):
        self.seen.append((schema, user))
        output = self.outputs[schema]
        if isinstance(output, Exception):
            raise output
        return output, LLMResult(text="", model=self.model, usage=Usage(1000, 100))


def test_llm_judge_scores_and_usage():
    llm = ScriptedLLM(
        {
            FaithfulnessOutput: FaithfulnessOutput(
                claims=[
                    ClaimVerdict(claim="a", reason="stated", supported=True),
                    ClaimVerdict(claim="b", reason="stated", supported=True),
                    ClaimVerdict(claim="c", reason="absent", supported=False),
                ]
            ),
            CorrectnessOutput: CorrectnessOutput(reasoning="close", score=4),
            RecallOutput: RecallOutput(
                statements=[StatementVerdict(statement="s", reason="r", in_context=True)]
            ),
        }
    )
    scores = LLMJudge(llm).evaluate("q", "the answer", ["ctx one", "ctx two"], "truth", True)
    assert scores.faithfulness == pytest.approx(2 / 3)
    assert scores.answer_correctness == 0.75
    assert scores.context_recall == 1.0
    assert scores.usage == Usage(3000, 300)
    assert '<excerpt index="2">\nctx two\n</excerpt>' in llm.seen[0][1]

    # unanswerable: no recall call; an answer with no claims is vacuously faithful
    llm.outputs[FaithfulnessOutput] = FaithfulnessOutput(claims=[])
    llm.seen.clear()
    scores = LLMJudge(llm).evaluate("q", "Not covered.", ["ctx"], "not covered", False)
    assert scores.faithfulness == 1.0 and scores.context_recall is None
    assert [s for s, _ in llm.seen] == [FaithfulnessOutput, CorrectnessOutput]


# ------------------------------------------------------------- eval and gate


def test_grounded_pipeline_passes_gate_and_logs_to_mlflow(make_pipeline, tmp_path):
    report = run_evaluation(make_pipeline(ledger=False), HeuristicJudge(), GOLDEN, gate_config=OFFLINE_GATE)
    assert report.errors == 0
    assert report.metrics["faithfulness_mean"] >= 0.9
    assert report.gate.passed, report.to_markdown()

    run = mlflow.get_run(report.mlflow_run_id)
    assert run.data.tags["gate"] == "passed"
    assert run.data.params["prompt_version"] == "v2"
    assert run.data.metrics["faithfulness_mean"] == pytest.approx(report.metrics["faithfulness_mean"])

    report.save(tmp_path)
    assert "Release gate: PASSED" in (tmp_path / "eval_report.md").read_text()


def test_hallucinating_pipeline_fails_gate(make_pipeline):
    pipeline = make_pipeline(FakeLLM(hallucinate=True), ledger=False)
    report = run_evaluation(pipeline, HeuristicJudge(), GOLDEN, gate_config=OFFLINE_GATE)
    assert report.metrics["faithfulness_mean"] < 0.5
    assert not report.gate.passed
    failed = {c.metric for c in report.gate.checks if not c.passed}
    assert "faithfulness_mean" in failed
    assert mlflow.get_run(report.mlflow_run_id).data.tags["gate"] == "failed"
    assert report.rows[0].unsupported_claims


def test_judge_errors_are_counted_and_block_the_gate(make_pipeline):
    judge = LLMJudge(ScriptedLLM({FaithfulnessOutput: LLMError("stop_reason=refusal")}))
    report = run_evaluation(
        make_pipeline(ledger=False), judge, GOLDEN[:2], gate_config=GATE, log_to_mlflow=False
    )
    assert report.errors == 2 and report.metrics["faithfulness_mean"] is None
    assert not report.gate.passed


def test_gate_rules():
    ok = {"faithfulness_mean": 0.95, "faithfulness_min": 0.6, "answer_correctness_mean": 0.8,
          "context_recall_mean": 0.9}
    assert check_gate(ok, 0, GATE).passed
    assert not check_gate({**ok, "faithfulness_mean": 0.89}, 0, GATE).passed
    assert not check_gate({**ok, "faithfulness_min": 0.2}, 0, GATE).passed
    assert not check_gate(ok, 1, GATE).passed
    assert not check_gate({**ok, "context_recall_mean": None}, 0, GATE).passed  # missing never passes


# ----------------------------------------------------------------------- API


def test_api_query_and_costs(make_pipeline):
    with TestClient(create_app(make_pipeline())) as client:
        assert client.get("/health").json()["indexed_chunks"] > 0

        response = client.post("/query", json={"question": "When do API keys expire by default?"})
        assert response.status_code == 200
        body = response.json()
        assert "90 days" in body["answer"]
        assert body["prompt"]["version"] == "v2" and body["sources"]
        assert response.headers["X-Request-Id"] == body["request_id"]
        assert float(response.headers["X-Request-Cost-USD"]) == body["cost"]["total_usd"]

        assert client.get(f"/costs/{body['request_id']}").json()["model"] == "fake-extractive"
        assert client.get("/costs").json()["total_requests"] == 1
        assert client.get("/costs/nope").status_code == 404
        assert client.post("/query", json={"question": "x", "prompt_version": "v999"}).status_code == 400
        assert client.post("/query", json={"question": ""}).status_code == 422
