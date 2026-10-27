"""Small text utilities shared by BM25, deduplication, verification and evaluation."""

from __future__ import annotations

import re
from typing import List, Set

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "and", "or", "but", "if", "do", "does", "did", "have", "has",
    "had", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "what",
    "when", "where", "who", "how", "which", "can", "will", "would", "my", "your", "our", "their", "me",
    "us", "them", "there", "than", "then", "so", "any", "all", "about", "into", "per",
    # NOTE: "not"/"no" are deliberately NOT stopwords -- dropping them would make
    # "does not cover" match "does cover" in verification and evaluation.
}

_TOKEN = re.compile(r"[a-z0-9]+(?:[.%][a-z0-9]+)*")


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def tokenize(text: str, stem: bool = True) -> List[str]:
    """Lowercase word tokens without stopwords, crudely stemmed (refund/refunds/refunded -> refun)."""
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
    return [_stem(t) for t in toks] if stem else toks


def _stem(t: str) -> str:
    if t[0].isdigit():
        return t
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 2]
