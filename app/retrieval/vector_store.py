"""
Vector store (ChromaDB)
=======================
The only module that talks to ChromaDB.

Safety rules:
* The collection remembers which embedding model filled it (`embedding_model_id`). Opening it
  with a different model raises `EmbeddingMismatchError` instead of silently mixing vector
  spaces. Nothing is ever deleted automatically -- use a new CHROMA_COLLECTION_NAME or re-index.
* Writes are upserts keyed by deterministic chunk IDs, so re-ingestion never duplicates chunks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb

log = logging.getLogger(__name__)


class EmbeddingMismatchError(RuntimeError):
    pass


def _clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool (no None)."""
    return {k: v for k, v in meta.items() if v is not None}


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_documents",
                 embedding_model_id: Optional[str] = None):
        # anonymized_telemetry=False: Chroma otherwise sends usage pings to its vendor.
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=chromadb.config.Settings(anonymized_telemetry=False))
        self.collection_name = collection_name
        meta = {"hnsw:space": "cosine"}
        if embedding_model_id:
            meta["embedding_model_id"] = embedding_model_id
        # NOTE: chromadb 0.5.x get_or_create_collection() OVERWRITES existing metadata, which would
        # erase the model record. So: get first, create only if missing.
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name, metadata=meta)
        stored = (self.collection.metadata or {}).get("embedding_model_id")
        self.legacy = stored is None
        if embedding_model_id and stored and stored != embedding_model_id:
            raise EmbeddingMismatchError(
                f"Collection '{collection_name}' was built with '{stored}' but the configured embedder is "
                f"'{embedding_model_id}'. Set a new CHROMA_COLLECTION_NAME (or re-index into a new one); "
                "existing data was left untouched."
            )
        if self.legacy and self.collection.count() > 0:
            # The original project only ever used sentence-transformers. Mixing another backend into
            # a legacy collection would silently corrupt retrieval, so refuse.
            if embedding_model_id and not embedding_model_id.startswith("sentence-transformers:"):
                raise EmbeddingMismatchError(
                    f"Collection '{collection_name}' contains legacy sentence-transformers vectors; the configured "
                    f"embedder is '{embedding_model_id}'. Use a new CHROMA_COLLECTION_NAME and re-upload documents."
                )
            log.warning("Collection '%s' predates model tracking (legacy, random chunk IDs). "
                        "Re-upload documents to get page citations and duplicate-safe IDs.", collection_name)

    # ------------------------------------------------------------------ writes
    def upsert(self, ids: List[str], embeddings: List[List[float]], documents: List[str],
               metadatas: List[Dict[str, Any]]) -> None:
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents,
                               metadatas=[_clean_meta(m) for m in metadatas])

    def add(self, ids, embeddings, documents, metadatas) -> None:  # original API
        self.upsert(ids, embeddings, documents, metadatas)

    def delete_document(self, doc_id: str) -> int:
        found = self.collection.get(where={"doc_id": doc_id}, include=[])
        if found["ids"]:
            self.collection.delete(ids=found["ids"])
        return len(found["ids"])

    def delete_legacy_source(self, source: str) -> int:
        """Remove chunks of `source` written by the ORIGINAL pipeline (no doc_id, random UUIDs)."""
        found = self.collection.get(where={"source": source}, include=["metadatas"])
        ids = [i for i, m in zip(found["ids"], found["metadatas"]) if not (m or {}).get("doc_id")]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        """Delete all chunks of THIS collection. Only called on explicit request (--force-reingest)."""
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)

    # ------------------------------------------------------------------ reads
    def query(self, query_embedding: List[float], top_k: int = 3,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n = self.count()
        if n == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.collection.query(query_embeddings=[query_embedding], n_results=min(top_k, n), where=where)

    def get_all(self, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.collection.get(where=where, include=["documents", "metadatas"])

    def doc_hashes(self) -> Dict[str, str]:
        """source -> content_hash for documents ingested by the new pipeline."""
        data = self.collection.get(include=["metadatas"])
        return {m["source"]: m["content_hash"] for m in data["metadatas"] if m and "content_hash" in m}

    def list_documents(self) -> List[Dict[str, Any]]:
        data = self.collection.get(include=["metadatas"])
        docs: Dict[str, Dict[str, Any]] = {}
        for m in data["metadatas"]:
            m = m or {}
            key = m.get("doc_id") or f"legacy:{m.get('source', 'unknown')}"
            d = docs.setdefault(key, {"document_id": key, "source": m.get("source", "unknown"),
                                      "chunks": 0, "pages": set(), "legacy": "doc_id" not in m})
            d["chunks"] += 1
            if m.get("page") is not None:
                d["pages"].add(m["page"])
        out = []
        for d in docs.values():
            d["pages"] = len(d["pages"]) or None
            out.append(d)
        return sorted(out, key=lambda d: d["source"])

    def count(self) -> int:
        return self.collection.count()
