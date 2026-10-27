"""
Ingestion service
=================
load -> clean -> chunk -> embed -> upsert, with:

* Deterministic document IDs derived from the file name: re-uploading `policy.pdf` replaces the
  old chunks of `policy.pdf` instead of adding a second copy.
* Content hashes: an unchanged file is skipped (no re-embedding); identical content uploaded
  under another name is reported as a duplicate and not embedded twice.
* Clear per-file statuses and errors (empty / corrupt / unsupported) instead of crashing a batch.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from app.ingestion.chunker import chunk_pages
from app.ingestion.embeddings import Embedder
from app.ingestion.loader import SUPPORTED_EXTENSIONS, DocumentLoadError, load_pages
from app.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


def document_id_for(source: str) -> str:
    return hashlib.sha1(f"doc|{source.lower()}".encode()).hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class IngestResult:
    source: str
    document_id: str
    status: str                   # indexed | reindexed | unchanged | duplicate | error
    chunks: int = 0
    pages: Optional[int] = None
    embedding_ms: float = 0.0
    message: str = ""
    duplicate_of: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class IngestionService:
    embedder: Embedder
    store: VectorStore
    chunk_size: int = 500
    chunk_overlap: int = 50
    on_change: Optional[Callable[[], None]] = None   # e.g. rebuild the BM25 index
    _last: List[IngestResult] = field(default_factory=list)

    def ingest_file(self, path: str, source: Optional[str] = None) -> IngestResult:
        source = source or os.path.basename(path)
        doc_id = document_id_for(source)
        try:
            content_hash = file_sha256(path)
            existing = self.store.doc_hashes()
            if existing.get(source) == content_hash:
                return IngestResult(source, doc_id, "unchanged", message="Identical content already indexed.")
            dup = next((s for s, h in existing.items() if h == content_hash and s != source), None)
            if dup:
                return IngestResult(source, doc_id, "duplicate", duplicate_of=dup,
                                    message=f"Same content is already indexed as '{dup}'.")

            pages = load_pages(path)
            chunks = chunk_pages(pages, source, doc_id, self.chunk_size, self.chunk_overlap)
            if not chunks:
                raise DocumentLoadError("No text chunks could be produced.")

            t0 = time.perf_counter()
            vectors = self.embedder.embed_batch([c.text for c in chunks])
            emb_ms = (time.perf_counter() - t0) * 1000

            # Replace older chunks of the same file (new-style ids and legacy random-UUID chunks).
            replaced = self.store.delete_document(doc_id) + self.store.delete_legacy_source(source)
            self.store.upsert(
                ids=[c.id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[{
                    "source": c.source, "doc_id": doc_id, "chunk_index": c.chunk_index,
                    "start_char": c.start_char, "page": c.page, "content_hash": content_hash,
                    "embedding_model": self.embedder.model_id,
                } for c in chunks],
            )
            if self.on_change:
                self.on_change()
            n_pages = len({p.page for p in pages if p.page is not None}) or None
            return IngestResult(source, doc_id, "reindexed" if replaced else "indexed", chunks=len(chunks),
                                pages=n_pages, embedding_ms=round(emb_ms, 2))
        except DocumentLoadError as e:
            return IngestResult(source, doc_id, "error", message=str(e))

    def ingest_directory(self, directory: str) -> List[IngestResult]:
        results = []
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
                results.append(IngestResult(name, document_id_for(name), "error", message="Unsupported file type."))
                continue
            results.append(self.ingest_file(path, name))
        return results

    def delete(self, doc_id: str) -> int:
        n = self.store.delete_document(doc_id)
        if n and self.on_change:
            self.on_change()
        return n
