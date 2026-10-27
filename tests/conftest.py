"""
Shared fixtures. Everything here is deterministic and offline:
* embeddings: the `hash` backend (no model download)
* LLM: MockLLM or small stubs
* ChromaDB: a real PersistentClient in a pytest tmp directory
* telemetry: written to a tmp file, never to outputs/telemetry_log.jsonl
"""

import os
import shutil
from pathlib import Path

import pytest

from app.config import Settings, settings

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "evaluation" / "corpus"


@pytest.fixture(autouse=True)
def _isolate_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "telemetry_log_path", str(tmp_path / "telemetry.jsonl"))
    monkeypatch.setattr(settings, "complexity_mode", "auto")
    yield


@pytest.fixture
def hash_embedder():
    from app.ingestion.embeddings import Embedder
    return Embedder(backend="hash")


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    return Settings(_env_file=None, embedding_backend="hash", llm_provider="mock",
                    chroma_persist_dir=str(tmp_path / "chroma"), chroma_collection_name="test_collection",
                    documents_dir=str(docs), eval_output_dir=str(tmp_path / "out"),
                    telemetry_log_path=str(tmp_path / "telemetry.jsonl"), top_k=3)


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    d = tmp_path / "corpus"
    shutil.copytree(CORPUS, d)
    return d


@pytest.fixture
def services(test_settings, corpus_dir, hash_embedder):
    from app.services import Services
    sv = Services(test_settings).init(embedder=hash_embedder)
    sv.ingestion.ingest_directory(str(corpus_dir))
    return sv
