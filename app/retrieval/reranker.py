"""
Rerankers
=========
A reranker re-scores (query, chunk) pairs after fusion. Interface: `score(query, texts) -> list[float]`.

* NoReranker            keeps the fused order (default; zero cost)
* CrossEncoderReranker  local cross-encoder (default model cross-encoder/ms-marco-MiniLM-L-6-v2, ~90 MB).
                        Tries fastembed (ONNX, no torch) first, then sentence-transformers.
                        Adds roughly 50-300 ms per query on CPU for 10 candidates (measure it: the
                        latency is recorded in telemetry as rerank_ms).
"""

from __future__ import annotations

from typing import List, Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: List[str]) -> List[float]: ...


class NoReranker:
    name = "none"

    def score(self, query: str, texts: List[str]) -> List[float]:
        return [0.0] * len(texts)


class RerankerUnavailable(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.name = f"cross-encoder:{model_name}"
        self._impl = None
        errors = []
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            enc = TextCrossEncoder(model_name=model_name.replace("cross-encoder/", "Xenova/"))
            self._impl = lambda q, ts: [float(s) for s in enc.rerank(q, ts)]
        except Exception as e:
            errors.append(f"fastembed: {e}")
        if self._impl is None:
            try:
                from sentence_transformers import CrossEncoder

                enc2 = CrossEncoder(model_name)
                self._impl = lambda q, ts: [float(s) for s in enc2.predict([(q, t) for t in ts])]
            except Exception as e:
                errors.append(f"sentence-transformers: {e}")
        if self._impl is None:
            raise RerankerUnavailable("No cross-encoder backend could be loaded: " + " | ".join(errors))

    def score(self, query: str, texts: List[str]) -> List[float]:
        return self._impl(query, texts) if texts else []


def build_reranker(kind: str, model_name: str) -> Reranker:
    if kind == "cross-encoder":
        return CrossEncoderReranker(model_name)
    return NoReranker()
