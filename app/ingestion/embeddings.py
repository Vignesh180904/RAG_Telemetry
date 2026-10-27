"""
Embeddings
==========
Turns text into normalized vectors. The rest of the app only uses the `Embedder`
class (`embed_text`, `embed_batch`, `cosine_similarity`, `model_id`), so the backend can be
swapped through EMBEDDING_BACKEND without touching other code.

Backends
--------
sentence-transformers  The original backend. Needs torch. On Windows machines with
                       Application Control / Smart App Control, importing torch can fail with
                       `WinError 4551`. The import is lazy, so the API still starts and /ready
                       reports the problem instead of crashing.
ollama                 Calls Ollama's /api/embed. `ollama pull all-minilm` gives the same
                       all-MiniLM-L6-v2 architecture with no Python ML dependencies at all.
fastembed              ONNX Runtime, no torch (`pip install fastembed`). Downloads the model from
                       HuggingFace on first use.
hash                   Deterministic feature hashing. No downloads, no dependencies, and LOW
                       retrieval quality. Use only for tests and offline smoke runs.

Vectors from different backends/models are NOT interchangeable. The vector store records
`model_id` and refuses to mix them (re-index after switching).
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import requests


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend cannot be loaded or called."""


# --------------------------------------------------------------------------- backends
class _HashBackend:
    dim = 384
    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, model_name: str):
        self.model_name = "hash-384"

    def _vec(self, text: str) -> List[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = self._token.findall(text.lower())
        feats = [t[:6] for t in toks] + [f"{a[:6]}_{b[:6]}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return (v / n).tolist() if n else v.tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


class _SentenceTransformersBackend:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # lazy: torch import can fail on Windows
        except Exception as e:  # ImportError, OSError (WinError 4551), ...
            raise EmbeddingError(
                f"Could not import sentence-transformers/torch ({type(e).__name__}: {e}). "
                "On Windows with Application Control, try EMBEDDING_BACKEND=ollama "
                "(ollama pull all-minilm) or EMBEDDING_BACKEND=fastembed."
            ) from e
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class _FastEmbedBackend:
    _ALIASES = {"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"}

    def __init__(self, model_name: str):
        try:
            from fastembed import TextEmbedding
        except Exception as e:
            raise EmbeddingError(f"fastembed is not installed or failed to import: {e}") from e
        self.model_name = self._ALIASES.get(model_name, model_name)
        self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for v in self._model.embed(texts):
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append((v / n if n else v).tolist())
        return out


class _OllamaBackend:
    _ALIASES = {"all-MiniLM-L6-v2": "all-minilm"}

    def __init__(self, model_name: str, base_url: str, timeout_s: float = 60.0):
        self.model_name = self._ALIASES.get(model_name, model_name)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=5).json()
        except Exception as e:
            raise EmbeddingError(f"Ollama is not reachable at {self.base_url}: {e}") from e
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model_name not in names and f"{self.model_name}:latest" not in names:
            raise EmbeddingError(
                f"Embedding model '{self.model_name}' is not installed in Ollama. "
                f"Run: ollama pull {self.model_name}"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            r = requests.post(f"{self.base_url}/api/embed",
                              json={"model": self.model_name, "input": texts}, timeout=self.timeout_s)
            r.raise_for_status()
            vecs = r.json()["embeddings"]
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding call failed: {e}") from e
        out = []
        for v in vecs:
            a = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(a))
            out.append((a / n if n else a).tolist())
        return out


# --------------------------------------------------------------------------- facade
class Embedder:
    """Backend-agnostic embedder with batching, a small query cache and latency tracking."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", backend: Optional[str] = None,
                 batch_size: int = 32, cache_size: int = 512, ollama_base_url: str = "http://localhost:11434"):
        from app.config import settings

        backend = backend or settings.embedding_backend
        self.backend_name = backend
        self.batch_size = batch_size
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_size = cache_size
        self.last_latency_ms: float = 0.0
        if backend == "hash":
            self._backend = _HashBackend(model_name)
        elif backend == "sentence-transformers":
            self._backend = _SentenceTransformersBackend(model_name)
        elif backend == "fastembed":
            self._backend = _FastEmbedBackend(model_name)
        elif backend == "ollama":
            self._backend = _OllamaBackend(model_name, ollama_base_url or settings.ollama_base_url)
        else:
            raise EmbeddingError(f"Unknown EMBEDDING_BACKEND: {backend}")
        self.model_name = self._backend.model_name

    @property
    def model_id(self) -> str:
        """Identifies the vector space; stored with the collection to prevent mixing."""
        return f"{self.backend_name}:{self.model_name}"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        t0 = time.perf_counter()
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._backend.embed(texts[i:i + self.batch_size]))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return out

    def embed_text(self, text: str) -> List[float]:
        """Embed one string (queries). Results are cached in a small LRU -- embeddings are deterministic."""
        if text in self._cache:
            self._cache.move_to_end(text)
            self.last_latency_ms = 0.0
            return self._cache[text]
        vec = self.embed_batch([text])[0]
        if self._cache_size:
            self._cache[text] = vec
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vec

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        na, nb = float(np.linalg.norm(a_arr)), float(np.linalg.norm(b_arr))
        if not na or not nb or math.isnan(na) or math.isnan(nb):
            return 0.0
        return float(np.dot(a_arr, b_arr) / (na * nb))
