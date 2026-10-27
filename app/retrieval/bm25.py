"""
BM25 sparse retrieval
=====================
Okapi BM25 over the chunk texts stored in ChromaDB. Keyword search catches exact terms that
embeddings blur (product names, numbers, error codes, "AES-256", "90%").

The index keeps only term frequencies and metadata, not the chunk texts, and is rebuilt
whenever documents are added or deleted. Memory grows with corpus size: fine for thousands of
chunks on 8 GB RAM; for very large corpora move this to SQLite FTS5 (same interface).
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.retrieval.text import tokenize


def _matches(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """Supports the simple equality filters used by the API ({"source": "x.pdf"}, {"$and": [...]})."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    for k, v in where.items():
        if isinstance(v, dict) and "$in" in v:
            if meta.get(k) not in v["$in"]:
                return False
        elif meta.get(k) != v:
            return False
    return True


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._lock = threading.Lock()
        self.ids: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.tfs: List[Counter] = []
        self.lengths: List[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def build(self, ids: List[str], texts: List[str], metas: List[Dict[str, Any]]) -> None:
        tfs = [Counter(tokenize(t)) for t in texts]
        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())
        lengths = [sum(tf.values()) for tf in tfs]
        with self._lock:  # swap atomically so concurrent searches never see a half-built index
            self.ids, self.metas, self.tfs, self.lengths, self.df = list(ids), [m or {} for m in metas], tfs, lengths, df
            self.avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

    def build_from_store(self, store) -> None:
        data = store.get_all()
        self.build(data["ids"], data["documents"], data["metadatas"])

    def __len__(self) -> int:
        return len(self.ids)

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        with self._lock:
            ids, metas, tfs, lengths, avgdl = self.ids, self.metas, self.tfs, self.lengths, self.avgdl
        scores: List[Tuple[str, float]] = []
        for i, tf in enumerate(tfs):
            if not _matches(metas[i], where):
                continue
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * lengths[i] / (avgdl or 1))
                    s += self.idf(t) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((ids[i], s))
        scores.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break
        return scores[:k]
