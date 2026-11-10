"""Offline fixtures: hash embeddings, in-memory Qdrant, fake LLM, throwaway MLflow store."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from qdrant_client import QdrantClient

from ragapp.cost import CostLedger, PriceTable
from ragapp.embeddings import HashEmbedder
from ragapp.ingest import ingest
from ragapp.llm import FakeLLM
from ragapp.pipeline import RAGPipeline
from ragapp.prompts import PromptRegistry
from ragapp.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def mlflow_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    # Keep run and trace artifacts in the temp dir instead of ./mlruns.
    mlflow.create_experiment("tests", artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment("tests")
    mlflow.tracing.enable()


@pytest.fixture(scope="session")
def store():
    store = VectorStore(QdrantClient(":memory:"), HashEmbedder(), "docs")
    ingest(store, ROOT / "data" / "docs", max_chars=900, overlap=150)
    return store


@pytest.fixture
def prices():
    return PriceTable.from_file(ROOT / "config" / "pricing.yaml")


@pytest.fixture
def make_pipeline(store, prices, tmp_path):
    def factory(llm=None, ledger=True):
        return RAGPipeline(
            store=store,
            llm=llm or FakeLLM(),
            prompts=PromptRegistry(ROOT / "prompts"),
            prices=prices,
            ledger=CostLedger(tmp_path / "costs.db") if ledger else None,
            prompt_version="v2",
        )

    return factory
