"""Central settings. Every value can be overridden by an environment variable or `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    # "anthropic" calls Claude. "fake" is an offline extractive stand-in used by
    # unit tests and key-less smoke runs; it is not a quality signal.
    llm_provider: Literal["anthropic", "fake"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16000
    # Server-side refusal fallback (beta). If a safety classifier declines the
    # request, the API re-runs it on a fallback model inside the same call.
    llm_refusal_fallback: bool = True
    judge_model: str = "claude-opus-5-5"
    judge_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    # --- Retrieval ---------------------------------------------------------
    # "fastembed" runs BAAI/bge-small-en-v1.5 locally (ONNX, no API key).
    # "hash" is a dependency-free hashing embedder for tests / offline runs.
    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Set QDRANT_URL to use a Qdrant server (docker compose). Otherwise the
    # embedded on-disk mode at QDRANT_PATH is used (":memory:" also works).
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = str(PROJECT_ROOT / ".qdrant")
    collection: str = "docs"
    top_k: int = 4
    chunk_chars: int = 900
    chunk_overlap: int = 150

    # --- Prompts -----------------------------------------------------------
    prompts_dir: str = str(PROJECT_ROOT / "prompts")
    prompt_name: str = "rag_answer"
    prompt_version: str = "v2"

    # --- Observability -----------------------------------------------------
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_experiment: str = "rag-capstone"
    tracing_enabled: bool = True
    cost_db_path: str = str(PROJECT_ROOT / "costs.db")
    pricing_file: str = str(PROJECT_ROOT / "config" / "pricing.yaml")

    # --- Evaluation / gate -------------------------------------------------
    docs_dir: str = str(PROJECT_ROOT / "data" / "docs")
    golden_path: str = str(PROJECT_ROOT / "data" / "eval" / "golden.jsonl")
    gate_file: str = str(PROJECT_ROOT / "config" / "gate.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()
