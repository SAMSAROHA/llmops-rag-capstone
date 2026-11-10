"""The RAG request path: retrieve -> render versioned prompt -> generate -> cost -> trace."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from .config import Settings
from .cost import Cost, CostLedger, PriceTable
from .embeddings import build_embedder
from .llm import LLM, Usage, build_llm
from .prompts import PromptRegistry, PromptVersion
from .vectorstore import Chunk, VectorStore, build_client

REFUSAL_MESSAGE = "I can't help with that request."


@dataclass
class RAGResponse:
    request_id: str
    question: str
    answer: str
    sources: list[Chunk]
    prompt: PromptVersion
    model: str
    usage: Usage
    cost: Cost
    latency_ms: float
    stop_reason: str
    trace_id: str | None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "sources": [
                {"index": i, "source": c.source, "score": round(c.score, 4), "text": c.text}
                for i, c in enumerate(self.sources, start=1)
            ],
            "prompt": {
                "name": self.prompt.name,
                "version": self.prompt.version,
                "sha": self.prompt.sha,
            },
            "model": self.model,
            "usage": asdict(self.usage),
            "cost": self.cost.to_dict(),
            "latency_ms": round(self.latency_ms, 1),
            "stop_reason": self.stop_reason,
            "trace_id": self.trace_id,
        }


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        llm: LLM,
        prompts: PromptRegistry,
        prices: PriceTable,
        ledger: CostLedger | None = None,
        prompt_name: str = "rag_answer",
        prompt_version: str = "latest",
        top_k: int = 4,
    ):
        self.store = store
        self.llm = llm
        self.prompts = prompts
        self.prices = prices
        self.ledger = ledger
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.top_k = top_k

    @classmethod
    def from_settings(cls, settings: Settings, with_ledger: bool = True) -> "RAGPipeline":
        store = VectorStore(build_client(settings), build_embedder(settings), settings.collection)
        return cls(
            store=store,
            llm=build_llm(settings),
            prompts=PromptRegistry(settings.prompts_dir),
            prices=PriceTable.from_file(settings.pricing_file),
            ledger=CostLedger(settings.cost_db_path) if with_ledger else None,
            prompt_name=settings.prompt_name,
            prompt_version=settings.prompt_version,
            top_k=settings.top_k,
        )

    def _retrieve(self, question: str, top_k: int) -> list[Chunk]:
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": question, "top_k": top_k})
            chunks = self.store.search(question, top_k)
            span.set_outputs(
                [
                    {
                        "page_content": c.text,
                        "metadata": {
                            "doc_uri": c.source,
                            "chunk_index": c.chunk_index,
                            "score": c.score,
                        },
                    }
                    for c in chunks
                ]
            )
            return chunks

    def _generate(self, prompt: PromptVersion, question: str, chunks: list[Chunk]):
        system, user = prompt.render(question, chunks)
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_inputs(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                }
            )
            result = self.llm.generate(system, user)
            cost = self.prices.cost(result.model, result.usage)
            span.set_attribute(
                SpanAttributeKey.CHAT_USAGE,
                {
                    "input_tokens": result.usage.input_tokens
                    + result.usage.cache_read_input_tokens
                    + result.usage.cache_creation_input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
            )
            span.set_attributes(
                {
                    SpanAttributeKey.MODEL: result.model,
                    "llm.requested_model": self.llm.model,
                    "llm.stop_reason": result.stop_reason,
                    "llm.request_id": result.request_id or "",
                    "cost.usd": cost.total_usd,
                    "cost.priced": cost.priced,
                }
            )
            span.set_outputs({"role": "assistant", "content": result.text})
            return result, cost

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        prompt_version: str | None = None,
        kind: str = "query",
    ) -> RAGResponse:
        request_id = uuid.uuid4().hex
        prompt = self.prompts.get(self.prompt_name, prompt_version or self.prompt_version)
        started = time.perf_counter()

        with mlflow.start_span(name="rag_request", span_type=SpanType.CHAIN) as root:
            root.set_inputs({"question": question})
            chunks = self._retrieve(question, top_k or self.top_k)
            result, cost = self._generate(prompt, question, chunks)
            answer = REFUSAL_MESSAGE if result.stop_reason == "refusal" else result.text
            latency_ms = (time.perf_counter() - started) * 1000
            root.set_outputs({"answer": answer})
            root.set_attributes(
                {"cost.usd": cost.total_usd, "tokens.total": result.usage.total_tokens}
            )
            trace_id = getattr(root, "trace_id", None)
            # Tags make traces filterable by prompt version / model / kind in the MLflow UI.
            mlflow.update_current_trace(
                client_request_id=request_id,
                tags={
                    "prompt.name": prompt.name,
                    "prompt.version": prompt.version,
                    "prompt.sha": prompt.sha,
                    "model": result.model,
                    "kind": kind,
                    "cost_usd": f"{cost.total_usd:.6f}",
                },
            )

        if self.ledger is not None:
            self.ledger.record(
                request_id=request_id,
                model=result.model,
                usage=result.usage,
                cost=cost,
                latency_ms=latency_ms,
                prompt_version=prompt.version,
                trace_id=trace_id,
                kind=kind,
            )
        return RAGResponse(
            request_id=request_id,
            question=question,
            answer=answer,
            sources=chunks,
            prompt=prompt,
            model=result.model,
            usage=result.usage,
            cost=cost,
            latency_ms=latency_ms,
            stop_reason=result.stop_reason,
            trace_id=trace_id,
        )
