"""
Reciprocal Rank Fusion (RRF)
============================
Combines several ranked lists without needing their scores to be comparable
(cosine similarity and BM25 live on different scales):

    rrf(d) = sum over lists L containing d of  1 / (k + rank_L(d))      (rank starts at 1)

k = 60 is the value from the original paper (Cormack et al., 2009); larger k flattens the
advantage of top ranks.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # an id repeated inside one list counts once
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def fused_order(scores: Dict[str, float]) -> List[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
