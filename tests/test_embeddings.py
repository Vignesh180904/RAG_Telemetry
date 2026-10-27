"""Embedding interface. Real model backends are exercised by tests/integration (opt-in)."""

import builtins

import numpy as np
import pytest

from app.ingestion import embeddings as emb_mod
from app.ingestion.embeddings import Embedder, EmbeddingError


def test_hash_backend_is_deterministic_and_normalized():
    e = Embedder(backend="hash")
    a, b = e.embed_text("refund policy"), Embedder(backend="hash").embed_text("refund policy")
    assert a == b and len(a) == 384 and abs(np.linalg.norm(a) - 1) < 1e-5
    assert e.model_id == "hash:hash-384"


def test_similar_texts_score_higher_than_unrelated():
    e = Embedder(backend="hash")
    q = e.embed_text("refund within 30 days")
    assert e.cosine_similarity(q, e.embed_text("returns accepted for a refund within 30 days")) > \
           e.cosine_similarity(q, e.embed_text("office located in Chennai"))


def test_batching_matches_single_calls():
    e = Embedder(backend="hash", batch_size=2)
    texts = ["a b c", "d e f", "g h i", "j k l", "m n o"]
    assert e.embed_batch(texts) == [e.embed_text(t) for t in texts]


def test_query_cache_is_bounded():
    e = Embedder(backend="hash", cache_size=2)
    for t in ("one", "two", "three"):
        e.embed_text(t)
    assert list(e._cache) == ["two", "three"]


def test_unknown_backend():
    with pytest.raises(EmbeddingError):
        Embedder(backend="nope")


def test_torch_import_failure_becomes_clear_error(monkeypatch):
    """Simulates the Windows Application Control failure (WinError 4551) on `import sentence_transformers`."""
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise OSError("[WinError 4551] An Application Control policy has blocked this file")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbeddingError, match="WinError 4551.*EMBEDDING_BACKEND=ollama"):
        Embedder(backend="sentence-transformers")


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise emb_mod.requests.HTTPError(str(self.status_code))


def test_ollama_backend_missing_model(monkeypatch):
    monkeypatch.setattr(emb_mod.requests, "get", lambda *a, **k: _Resp({"models": [{"name": "llama3.2:3b"}]}))
    with pytest.raises(EmbeddingError, match="ollama pull all-minilm"):
        Embedder("all-MiniLM-L6-v2", backend="ollama")


def test_ollama_backend_unreachable(monkeypatch):
    def boom(*a, **k):
        raise emb_mod.requests.ConnectionError("refused")
    monkeypatch.setattr(emb_mod.requests, "get", boom)
    with pytest.raises(EmbeddingError, match="not reachable"):
        Embedder(backend="ollama")


def test_ollama_backend_embeds_and_normalizes(monkeypatch):
    monkeypatch.setattr(emb_mod.requests, "get", lambda *a, **k: _Resp({"models": [{"name": "all-minilm:latest"}]}))
    monkeypatch.setattr(emb_mod.requests, "post", lambda *a, **k: _Resp({"embeddings": [[3.0, 4.0], [0.0, 2.0]]}))
    e = Embedder("all-MiniLM-L6-v2", backend="ollama")
    assert e.model_id == "ollama:all-minilm"
    assert np.allclose(e.embed_batch(["a", "b"]), [[0.6, 0.8], [0.0, 1.0]])
