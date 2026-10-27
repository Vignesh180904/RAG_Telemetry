"""
Service container
=================
Builds every component once (API startup, evaluation runner, scripts) and records failures
instead of crashing, so `/health` stays up and `/ready` can say exactly what is missing
(e.g. "embedding backend failed to import torch (WinError 4551)" or "Ollama unreachable").
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.agent.graph import RagAgent
from app.config import Settings, settings as default_settings
from app.generation.llm import BaseLLM, MockLLM, get_llm
from app.generation.model_manager import ModelCall, ModelError, OllamaModelManager, OllamaRoleLLM
from app.generation.prompt import build_rag_prompt
from app.ingestion.embeddings import Embedder
from app.ingestion.pipeline import IngestionService
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import NoReranker, build_reranker
from app.retrieval.retriever import HybridRetriever, Retriever
from app.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)
ROLES = ("generator", "verifier", "synthesizer", "router")


def build_role_llms(s: Settings):
    """role -> LLM. Mock and legacy cloud providers use one object for every role."""
    if s.llm_provider == "ollama":
        mgr = OllamaModelManager(base_url=s.ollama_base_url, timeout_s=s.llm_timeout_s, max_retries=s.llm_max_retries,
                                 backoff_s=s.llm_retry_backoff_s, num_ctx=s.llm_num_ctx, num_predict=s.llm_num_predict,
                                 temperature=s.llm_temperature, keep_alive=s.llm_keep_alive)
        return mgr, {r: OllamaRoleLLM(mgr, r, s.model_for(r)) for r in ROLES}
    llm = MockLLM() if s.llm_provider == "mock" else get_llm()
    return None, {r: llm for r in ROLES}


@dataclass
class Services:
    s: Settings = field(default_factory=lambda: default_settings)
    embedder: Optional[Embedder] = None
    store: Optional[VectorStore] = None
    bm25: Optional[BM25Index] = None
    retriever: Optional[HybridRetriever] = None
    baseline_retriever: Optional[Retriever] = None
    ingestion: Optional[IngestionService] = None
    agent: Optional[RagAgent] = None
    manager: Optional[OllamaModelManager] = None
    llms: Dict[str, BaseLLM] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    warnings: Dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def init(self, embedder: Optional[Embedder] = None) -> "Services":
        s = self.s
        try:
            self.embedder = embedder or Embedder(s.embedding_model, backend=s.embedding_backend,
                                                 batch_size=s.embedding_batch_size,
                                                 cache_size=s.embedding_query_cache_size,
                                                 ollama_base_url=s.ollama_base_url)
        except Exception as e:
            self.errors["embeddings"] = str(e)
            log.error("embedding backend unavailable: %s", e)
        if self.embedder is not None:
            try:
                self.store = VectorStore(s.chroma_persist_dir, s.chroma_collection_name, self.embedder.model_id)
            except Exception as e:
                self.errors["vector_store"] = str(e)
                log.error("vector store unavailable: %s", e)
        try:
            reranker = build_reranker(s.reranker, s.reranker_model)
        except Exception as e:
            reranker = NoReranker()
            self.warnings["reranker"] = f"Reranker disabled: {e}"
        self.manager, self.llms = build_role_llms(s)
        if self.store is not None:
            self.bm25 = BM25Index(k1=s.bm25_k1, b=s.bm25_b)
            self.bm25.build_from_store(self.store)
            self.retriever = HybridRetriever(self.embedder, self.store, self.bm25, reranker, mode=s.retrieval_mode,
                                             top_k=s.top_k, dense_candidates=s.dense_candidates,
                                             sparse_candidates=s.sparse_candidates, rrf_k=s.rrf_k,
                                             min_dense_score=s.min_dense_score, dedupe_threshold=s.dedupe_threshold)
            self.baseline_retriever = Retriever(self.embedder, self.store)
            self.ingestion = IngestionService(self.embedder, self.store, s.chunk_size, s.chunk_overlap,
                                              on_change=self.retriever.refresh_sparse_index)
            self.agent = RagAgent(self.retriever, self.llms["generator"], self.llms["verifier"],
                                  self.llms["synthesizer"], self.llms["router"])
        return self

    # ------------------------------------------------------------------ readiness
    def readiness(self, check_models: bool = True) -> Dict[str, Any]:
        comps: Dict[str, Any] = {
            "embeddings": {"ok": self.embedder is not None, "backend": self.s.embedding_backend,
                           "model_id": self.embedder.model_id if self.embedder else None,
                           "error": self.errors.get("embeddings")},
            "vector_store": {"ok": self.store is not None, "chunks": self.store.count() if self.store else None,
                             "legacy_collection": getattr(self.store, "legacy", None),
                             "error": self.errors.get("vector_store")},
            "sparse_index": {"ok": self.bm25 is not None, "chunks": len(self.bm25) if self.bm25 is not None else None},
        }
        if self.s.llm_provider == "ollama" and check_models and self.manager:
            st = self.manager.status(self.s.role_models())
            ok = st.get("ollama_reachable") and all(r["installed"] for k, r in st["roles"].items()
                                                    if k != "router" or self.s.agent_router == "llm")
            comps["llm"] = {"ok": bool(ok), "provider": "ollama", **st}
        else:
            comps["llm"] = {"ok": True, "provider": self.s.llm_provider,
                            "note": "mock provider: deterministic test model" if self.s.llm_provider == "mock" else ""}
        return {"ready": all(c["ok"] for c in comps.values()), "components": comps, "warnings": self.warnings}

    # ------------------------------------------------------------------ baseline (original pipeline)
    def baseline_answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """The ORIGINAL fixed pipeline: dense top-k -> one prompt -> one generation. Used as the eval baseline."""
        k = top_k or self.s.top_k
        t0 = time.perf_counter()
        chunks = self.baseline_retriever.retrieve(question, top_k=k)
        r_ms = (time.perf_counter() - t0) * 1000
        prompt = build_rag_prompt(question, Retriever.format_context(chunks))
        llm = self.llms["generator"]
        calls: list = []
        t1 = time.perf_counter()
        try:
            resp = llm.chat("You are a helpful assistant.", prompt, calls=calls) if isinstance(llm, OllamaRoleLLM) \
                else llm.generate(prompt)
            answer, err = resp.text, None
        except ModelError as e:
            answer, err = "", {"code": e.code, "message": str(e)}
        l_ms = (time.perf_counter() - t1) * 1000
        return {"answer": answer, "chunks": [c.to_dict() for c in chunks], "citations": [],
                "retrieval_ms": round(r_ms, 2), "llm_ms": round(l_ms, 2), "error": err,
                "model_calls": [c.to_dict() for c in calls]}
