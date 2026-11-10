from pathlib import Path
from types import SimpleNamespace

import pytest

from ragapp.cost import Cost, CostLedger, PriceTable
from ragapp.ingest import load_chunks, split_text
from ragapp.llm import FALLBACK_BETA, AnthropicLLM, Usage
from ragapp.prompts import PromptRegistry
from ragapp.vectorstore import Chunk

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ chunking


def test_split_respects_max_chars_and_overlaps():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(8))
    chunks = split_text(text, max_chars=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 80 + 2 for c in chunks)
    # the tail of each chunk reappears at the head of the next
    assert chunks[0][-40:].strip() in chunks[1]


def test_split_handles_one_giant_paragraph():
    chunks = split_text("x" * 2500, max_chars=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


def test_chunk_ids_are_deterministic():
    a = load_chunks(ROOT / "data" / "docs", 900, 150)
    b = load_chunks(ROOT / "data" / "docs", 900, 150)
    assert [c.point_id for c in a] == [c.point_id for c in b]
    assert len({c.point_id for c in a}) == len(a)
    assert {c.source for c in a} >= {"sla.md", "pricing.md", "security.md"}


def test_retrieval_finds_the_right_document(store):
    hits = store.search("How much does data egress cost?", top_k=3)
    assert hits[0].source == "pricing.md"
    assert hits[0].score >= hits[-1].score


# --------------------------------------------------------------------- cost


def test_cost_arithmetic():
    table = PriceTable({"m": {"input": 4.0, "output": 20.0, "cache_read": 0.2}})
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    cost = table.cost("m", usage)
    assert cost.input_usd == pytest.approx(4.0)
    assert cost.output_usd == pytest.approx(10.0)
    assert cost.cache_usd == pytest.approx(0.2 + 5.0)  # read + write at 1.25x input
    assert cost.total_usd == pytest.approx(19.2)
    assert cost.priced


def test_cost_prefix_match_and_unknown_model():
    table = PriceTable({"claude-opus-5-5": {"input": 4.0, "output": 20.0}})
    assert table.cost("claude-opus-5-5-snapshot", Usage(1000, 0)).total_usd == pytest.approx(0.004)
    unknown = table.cost("mystery-model", Usage(1000, 1000))
    assert unknown.total_usd == 0 and not unknown.priced


def test_shipped_price_table_covers_default_models():
    table = PriceTable.from_file(ROOT / "config" / "pricing.yaml")
    assert table.cost("claude-opus-5-5", Usage(1_000_000, 1_000_000)).total_usd == pytest.approx(24.0)


def test_ledger_roundtrip_and_summary(tmp_path):
    ledger = CostLedger(tmp_path / "c.db")
    ledger.record("r1", "m", Usage(100, 50), Cost(0.01, 0.02), 120.0, "v2", "tr-1")
    ledger.record("r2", "m", Usage(200, 10), Cost(0.02, 0.01), 80.0, "v2")
    assert ledger.get("r1")["cost_usd"] == pytest.approx(0.03)
    assert ledger.get("r1")["trace_id"] == "tr-1"
    summary = ledger.summary()
    assert summary["total_requests"] == 2
    assert summary["total_cost_usd"] == pytest.approx(0.06)
    assert summary["groups"][0]["input_tokens"] == 300


# ------------------------------------------------------------------ prompts


def test_prompt_versions_and_latest():
    registry = PromptRegistry(ROOT / "prompts")
    assert registry.versions("rag_answer")[:2] == ["v1", "v2"]
    assert registry.get("rag_answer", "latest").version == registry.versions("rag_answer")[-1]
    with pytest.raises(KeyError):
        registry.get("rag_answer", "v999")


def test_prompt_sha_tracks_content(tmp_path):
    registry = PromptRegistry(ROOT / "prompts")
    v1, v2 = registry.get("rag_answer", "v1"), registry.get("rag_answer", "v2")
    assert v1.sha != v2.sha
    assert v2.sha == registry.get("rag_answer", "v2").sha


def test_prompt_render_and_validation(tmp_path):
    prompt = PromptRegistry(ROOT / "prompts").get("rag_answer", "v2")
    _, user = prompt.render("Why?", [Chunk("Because.", "a.md", 0)])
    assert '<document index="1" source="a.md">\nBecause.\n</document>' in user
    assert "<question>\nWhy?\n</question>" in user
    assert "{{" not in user

    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "v1.yaml").write_text("system: s\nuser_template: no placeholders\n")
    with pytest.raises(ValueError):
        PromptRegistry(tmp_path).get("bad", "v1")


def test_prompt_sync_to_mlflow_is_idempotent():
    registry = PromptRegistry(ROOT / "prompts")
    first = registry.sync_to_mlflow("rag_answer")
    assert first == registry.versions("rag_answer")
    assert registry.sync_to_mlflow("rag_answer") == []


# ------------------------------------------------------- Anthropic request shape


class _StubMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=" The answer. "),
            ],
            model="claude-opus-5-5",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            ),
            stop_reason="end_turn",
            _request_id="req_1",
        )


def test_anthropic_request_shape():
    beta, plain = _StubMessages(), _StubMessages()
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))

    llm = AnthropicLLM("claude-opus-5-5", effort="medium", client=client)
    result = llm.generate("sys", "user")
    assert result.text == "The answer." and result.usage == Usage(10, 5)
    call = beta.calls[0]
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "medium"}
    # Opus 5.5 rejects sampling params, thinking budgets and prefill.
    assert not {"temperature", "top_p", "top_k", "thinking"} & call.keys()
    assert call["messages"] == [{"role": "user", "content": "user"}]

    AnthropicLLM("claude-opus-5-5", refusal_fallback=False, client=client).generate("s", "u")
    assert "fallbacks" not in plain.calls[0] and "betas" not in plain.calls[0]
