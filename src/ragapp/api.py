"""FastAPI service. Run with `ragapp serve` or `uvicorn ragapp.api:app`."""

from __future__ import annotations

from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RAGPipeline
from .tracing import setup_tracing


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    prompt_version: str | None = None


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            settings = get_settings()
            setup_tracing(settings)
            app.state.pipeline = RAGPipeline.from_settings(settings)
        else:
            app.state.pipeline = pipeline
        yield

    app = FastAPI(title="RAG capstone", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        p: RAGPipeline = app.state.pipeline
        return {
            "status": "ok",
            "indexed_chunks": p.store.count(),
            "model": p.llm.model,
            "prompt": f"{p.prompt_name}/{p.prompts.get(p.prompt_name, p.prompt_version).version}",
        }

    # Sync handler: FastAPI runs it in a threadpool, so the blocking SDK call is fine.
    @app.post("/query")
    def query(request: QueryRequest, response: Response):
        p: RAGPipeline = app.state.pipeline
        try:
            result = p.answer(request.question, request.top_k, request.prompt_version)
        except KeyError as exc:  # unknown prompt version
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise HTTPException(status_code=429, detail="Upstream rate limit, retry later") from exc
        except anthropic.APIStatusError as exc:
            status = 502 if exc.status_code >= 500 else 500
            raise HTTPException(status_code=status, detail=f"LLM error: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise HTTPException(status_code=503, detail="Could not reach the LLM API") from exc
        response.headers["X-Request-Id"] = result.request_id
        response.headers["X-Request-Cost-USD"] = f"{result.cost.total_usd:.6f}"
        response.headers["X-Prompt-Version"] = result.prompt.version
        return result.to_dict()

    @app.get("/costs")
    def costs():
        ledger = app.state.pipeline.ledger
        if ledger is None:
            raise HTTPException(status_code=404, detail="Cost ledger is disabled")
        return ledger.summary()

    @app.get("/costs/{request_id}")
    def cost_for_request(request_id: str):
        ledger = app.state.pipeline.ledger
        row = ledger.get(request_id) if ledger else None
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown request_id")
        return row

    return app


app = create_app()
