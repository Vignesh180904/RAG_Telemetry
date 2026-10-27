"""
Retrieval
=========
`Retriever`        the ORIGINAL dense-only retriever (kept unchanged in behaviour; it is the
                   "baseline" in evaluations).
`HybridRetriever`  the advanced pipeline:

    query -> normalize -> [dense (Chroma)] + [sparse (BM25)] -> RRF fusion
          -> relevance filter -> near-duplicate removal -> optional rerank -> top-k (+ citations)

Every returned chunk carries its dense cosine similarity (`score`, used by the agent's
relevance gate), its BM25 score, its RRF score and, if a reranker ran, its rerank score.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.ingestion.embeddings import Embedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import fused_order, reciprocal_rank_fusion
from app.retrieval.reranker import NoReranker, Reranker
from app.retrieval.text import jaccard, normalize_query
from app.retrieval.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float                         # dense cosine similarity (higher = more similar)
    page: Optional[int] = None
    doc_id: str = ""
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    found_by: List[str] = field(default_factory=list)   # ["dense"], ["sparse"] or both
    citation: int = 0                    # 1-based number used as [n] in answers

    def label(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _where(source_filter: Optional[str], where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    clauses = []
    if source_filter:
        clauses.append({"source": source_filter})
    if where:
        clauses.extend({k: v} for k, v in where.items())
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Retriever:
    """Original dense-only retriever (baseline)."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
        res = self.vector_store.query(self.embedder.embed_text(query), top_k=top_k, where=_where(source_filter, None))
        out = []
        for i, (_id, doc, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                       res["metadatas"][0], res["distances"][0]), start=1):
            meta = meta or {}
            out.append(RetrievedChunk(id=_id, text=doc, source=meta.get("source", "unknown"),
                                      chunk_index=meta.get("chunk_index", -1), score=round(1 - dist, 4),
                                      page=meta.get("page"), doc_id=meta.get("doc_id", ""),
                                      found_by=["dense"], citation=i))
        return out

    @staticmethod
    def format_context(chunks: List[RetrievedChunk]) -> str:
        return "\n\n".join(f"[Source {i}: {c.source}, chunk {c.chunk_index}, score={c.score}]\n{c.text}"
                           for i, c in enumerate(chunks, start=1))


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]
    stats: Dict[str, Any]


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, bm25: BM25Index,
                 reranker: Optional[Reranker] = None, *, mode: str = "hybrid", top_k: int = 3,
                 dense_candidates: int = 10, sparse_candidates: int = 10, rrf_k: int = 60,
                 min_dense_score: float = 0.0, dedupe_threshold: float = 0.9):
        self.embedder, self.store, self.bm25 = embedder, store, bm25
        self.reranker = reranker or NoReranker()
        self.mode, self.top_k = mode, top_k
        self.dense_candidates, self.sparse_candidates = dense_candidates, sparse_candidates
        self.rrf_k, self.min_dense_score, self.dedupe_threshold = rrf_k, min_dense_score, dedupe_threshold

    def refresh_sparse_index(self) -> None:
        self.bm25.build_from_store(self.store)

    def retrieve(self, query: str, top_k: Optional[int] = None, source_filter: Optional[str] = None,
                 where: Optional[Dict[str, Any]] = None, mode: Optional[str] = None) -> RetrievalResult:
        mode = mode or self.mode
        top_k = top_k or self.top_k
        q = normalize_query(query)
        flt = _where(source_filter, where)
        stats: Dict[str, Any] = {"mode": mode, "query": q, "top_k": top_k, "reranker": self.reranker.name,
                                 "embedding_ms": 0.0, "dense_ms": 0.0, "sparse_ms": 0.0, "rerank_ms": 0.0}
        t_all = time.perf_counter()
        pool: Dict[str, RetrievedChunk] = {}
        rankings: List[List[str]] = []

        # ---- query embedding (always: it also scores keyword-only hits for the relevance gate)
        t0 = time.perf_counter()
        q_vec = self.embedder.embed_text(q)
        stats["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- dense
        if mode in ("hybrid", "dense"):
            t0 = time.perf_counter()
            res = self.store.query(q_vec, top_k=self.dense_candidates, where=flt)
            stats["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            ranking = []
            for _id, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
                sim = round(1 - dist, 4)
                if sim < self.min_dense_score:
                    continue
                meta = meta or {}
                pool[_id] = RetrievedChunk(id=_id, text=doc, source=meta.get("source", "unknown"),
                                           chunk_index=meta.get("chunk_index", -1), score=sim,
                                           page=meta.get("page"), doc_id=meta.get("doc_id", ""), found_by=["dense"])
                ranking.append(_id)
            rankings.append(ranking)
        stats["dense_hits"] = len(rankings[0]) if rankings else 0

        # ---- sparse
        sparse_scores: Dict[str, float] = {}
        if mode in ("hybrid", "sparse"):
            t0 = time.perf_counter()
            hits = self.bm25.search(q, k=self.sparse_candidates, where=flt)
            stats["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            sparse_scores = {i: s for i, s in hits}
            rankings.append([i for i, _ in hits])
            missing = [i for i, _ in hits if i not in pool]
            if missing:  # fetch text + embedding for keyword-only hits so every chunk has a dense score
                got = self.store.collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
                for _id, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
                    meta = meta or {}
                    pool[_id] = RetrievedChunk(id=_id, text=doc, source=meta.get("source", "unknown"),
                                               chunk_index=meta.get("chunk_index", -1),
                                               score=round(Embedder.cosine_similarity(q_vec, list(emb)), 4),
                                               page=meta.get("page"), doc_id=meta.get("doc_id", ""), found_by=[])
            for i in sparse_scores:
                if i in pool:
                    pool[i].sparse_score = round(sparse_scores[i], 4)
                    pool[i].found_by = sorted(set(pool[i].found_by) | {"sparse"})
        stats["sparse_hits"] = len(sparse_scores)

        # ---- fusion
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        order = [i for i in fused_order(fused) if i in pool]
        for i in order:
            pool[i].rrf_score = round(fused[i], 6)
        stats["fused_candidates"] = len(order)

        # ---- near-duplicate removal (overlapping chunks / repeated boilerplate)
        kept: List[RetrievedChunk] = []
        for i in order:
            c = pool[i]
            if any(jaccard(c.text, k.text) >= self.dedupe_threshold for k in kept):
                continue
            kept.append(c)
        stats["deduplicated"] = len(order) - len(kept)

        # ---- optional rerank of a small head of the list
        head = kept[: max(top_k * 3, top_k)]
        if self.reranker.name != "none" and head:
            t0 = time.perf_counter()
            scores = self.reranker.score(q, [c.text for c in head])
            stats["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            for c, s in zip(head, scores):
                c.rerank_score = round(s, 4)
            head.sort(key=lambda c: -(c.rerank_score or 0.0))
        final = head[:top_k]
        for n, c in enumerate(final, start=1):
            c.citation = n
        stats["returned"] = len(final)
        stats["best_dense_score"] = max((c.score for c in final), default=0.0)
        stats["retrieval_ms"] = round((time.perf_counter() - t_all) * 1000, 2)
        return RetrievalResult(final, stats)


def format_evidence(chunks: List[RetrievedChunk], max_chars_per_chunk: int = 1200) -> str:
    """
    Evidence block for prompts. Each chunk is fenced and numbered; any fence-like text inside a
    document is neutralised so a document cannot 'close' its block and inject instructions.
    """
    parts = []
    for c in chunks:
        text = c.text[:max_chars_per_chunk].replace("<evidence", "<_evidence").replace("</evidence", "</_evidence")
        parts.append(f'<evidence id="{c.citation}" source="{c.label()}">\n{text}\n</evidence>')
    return "\n".join(parts)
