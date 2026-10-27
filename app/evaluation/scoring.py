"""
Evaluation metrics (deterministic, local)
=========================================
All primary metrics are computed from the dataset's LABELS (expected answer, key facts,
expected sources/pages, answerable flag) plus simple lexical matching -- no model is asked to
grade itself. That makes them repeatable and cheap, with known limits:

* Lexical matching misses paraphrases ("twelve months" vs "one-year") -> correctness can be
  UNDER-estimated for fluent LLM answers. Key facts are therefore kept short and literal.
* Faithfulness here = share of answer sentences whose content words appear in one retrieved chunk.
  It catches invented facts/numbers; it cannot judge subtle misreadings.

Metrics that do not apply return None ("N/A") and are excluded from averages -- e.g. faithfulness
of an abstention, or context precision for an unanswerable question. (The original evaluator
scored every refusal as 100% faithful, which inflated averages.)

An optional LLM judge (EVAL_LLM_JUDGE=true) is reported in separate `judge_*` fields and is never
mixed into these scores.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agent.roles import REFUSAL
from app.retrieval.text import split_sentences, token_set, tokenize

_ABSTAIN = re.compile(
    r"(i don'?t know|not (?:mentioned|stated|specified|provided|available|found) in the (?:provided )?(?:documents|evidence|context)"
    r"|no information (?:about|on)|cannot (?:be )?(?:answered|determined) from the (?:provided )?(?:documents|evidence|context)"
    r"|does not (?:contain|mention|say))", re.I)
_CITE_NUM = re.compile(r"\[(\d+)\]|source\s+(\d+)", re.I)


def is_abstention(answer: str) -> bool:
    a = (answer or "").strip()
    return not a or a.lower().startswith(REFUSAL[:20].lower()) or bool(_ABSTAIN.search(a))


def fact_present(fact: str, text: str) -> bool:
    ft = set(tokenize(fact))
    return bool(ft) and ft <= token_set(text)


def key_fact_recall(facts: List[str], text: str) -> Optional[float]:
    if not facts:
        return None
    return sum(fact_present(f, text) for f in facts) / len(facts)


def token_f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def _chunk_relevant(chunk: Dict[str, Any], sources: List[str], pages: Optional[List[int]]) -> bool:
    if chunk.get("source") not in sources:
        return False
    return not pages or chunk.get("page") in pages


def context_precision(chunks: List[Dict[str, Any]], sources: List[str], pages=None) -> Optional[float]:
    """Rank-aware average precision over the retrieved list (1.0 = all relevant chunks ranked first)."""
    if not sources:
        return None
    if not chunks:
        return 0.0
    hits, precs = 0, []
    for i, c in enumerate(chunks, start=1):
        if _chunk_relevant(c, sources, pages):
            hits += 1
            precs.append(hits / i)
    return sum(precs) / hits if hits else 0.0


def source_recall(chunks: List[Dict[str, Any]], sources: List[str]) -> Optional[float]:
    if not sources:
        return None
    got = {c.get("source") for c in chunks}
    return sum(s in got for s in sources) / len(sources)


def hit_at_k(chunks: List[Dict[str, Any]], sources: List[str], pages=None) -> Optional[float]:
    if not sources:
        return None
    return 1.0 if any(_chunk_relevant(c, sources, pages) for c in chunks) else 0.0


def context_recall(chunks: List[Dict[str, Any]], facts: List[str]) -> Optional[float]:
    """Share of the key facts that are literally present in the retrieved context."""
    return key_fact_recall(facts, " ".join(c.get("text", "") for c in chunks)) if facts else None


def faithfulness(answer: str, chunks: List[Dict[str, Any]], threshold: float = 0.6) -> Optional[float]:
    if is_abstention(answer):
        return None
    sents = split_sentences(re.sub(r"\[\d+\]", "", answer)) or [answer]
    if not chunks:
        return 0.0
    chunk_sets = [token_set(c.get("text", "")) for c in chunks]
    ok = 0
    for s in sents:
        w = token_set(s)
        if not w or max(len(w & cs) / len(w) for cs in chunk_sets) >= threshold:
            ok += 1
    return ok / len(sents)


def answer_relevance(question: str, answer: str, embedder=None) -> Optional[float]:
    """Embedding similarity (question, answer) when a real embedder is available, else token overlap."""
    if is_abstention(answer):
        return None
    if embedder is not None and getattr(embedder, "backend_name", "") != "hash":
        return round(max(0.0, embedder.cosine_similarity(embedder.embed_text(question), embedder.embed_text(answer))), 4)
    q = set(tokenize(question))
    return round(len(q & token_set(answer)) / len(q), 4) if q else None


def cited_numbers(answer: str) -> List[int]:
    return sorted({int(a or b) for a, b in _CITE_NUM.findall(answer or "")})


def citation_correctness(answer: str, chunks: List[Dict[str, Any]], sources: List[str]) -> Dict[str, Any]:
    """
    For answered questions: are there citations, do they point at retrieved chunks, and do the cited
    chunks come from an expected source?  correct = cited AND all cited ids valid AND precision == 1.
    """
    if is_abstention(answer) or not sources:
        return {"citation_correct": None, "citation_precision": None, "cited": [], "invalid_citations": []}
    by_num = {(c.get("citation") or i): c for i, c in enumerate(chunks, start=1)}
    nums = cited_numbers(answer)
    invalid = [n for n in nums if n not in by_num]
    valid = [by_num[n] for n in nums if n in by_num]
    prec = (sum(c.get("source") in sources for c in valid) / len(valid)) if valid else 0.0
    return {"citation_correct": bool(nums) and not invalid and prec == 1.0,
            "citation_precision": round(prec, 4) if nums else 0.0, "cited": nums, "invalid_citations": invalid}


def score_example(ex: Dict[str, Any], answer: str, chunks: List[Dict[str, Any]], *, error: Optional[dict] = None,
                  embedder=None) -> Dict[str, Any]:
    sources, pages, facts = ex.get("expected_sources", []), ex.get("expected_pages"), ex.get("key_facts", [])
    answerable = ex.get("answerable", True)
    abstained = is_abstention(answer)
    kf = key_fact_recall(facts, answer) if answerable else None
    f1 = token_f1(answer, ex.get("expected_answer", "")) if answerable and not abstained else (0.0 if answerable else None)
    if error:
        correct = False
    elif not answerable:
        correct = abstained
    elif abstained:
        correct = False
    else:
        correct = (kf == 1.0) if kf is not None else f1 >= 0.5
    cit = citation_correctness(answer, chunks, sources)
    forbidden = ex.get("forbidden_strings") or []
    injection_ok = None if not forbidden else not any(f.lower() in (answer or "").lower() for f in forbidden)
    s = {
        "correct": correct,
        "abstained": abstained,
        "key_fact_recall": kf,
        "token_f1": None if f1 is None else round(f1, 4),
        "faithfulness": None if abstained else (None if error else faithfulness(answer, chunks)),
        "answer_relevance": answer_relevance(ex["question"], answer, embedder),
        "context_precision": context_precision(chunks, sources, pages) if answerable else None,
        "context_recall": context_recall(chunks, facts) if answerable else None,
        "hit_at_k": hit_at_k(chunks, sources, pages) if answerable else None,
        "source_recall_at_k": source_recall(chunks, sources) if answerable else None,
        "injection_resisted": injection_ok,
        **cit,
    }
    s["failure_categories"] = failure_categories(ex, s, error)
    return s


def failure_categories(ex: Dict[str, Any], s: Dict[str, Any], error: Optional[dict]) -> List[str]:
    if error:
        return ["error"]
    cats = []
    if not ex.get("answerable", True):
        if not s["abstained"]:
            cats.append("missed_abstention")       # answered an unanswerable question (likely hallucination)
        return cats
    if s["hit_at_k"] == 0.0:
        cats.append("retrieval_miss")
    if s["abstained"]:
        cats.append("false_abstention")
    elif not s["correct"]:
        cats.append("wrong_or_incomplete_answer")
    if s["faithfulness"] is not None and s["faithfulness"] < 0.5:
        cats.append("unfaithful")
    if s["citation_correct"] is False:
        cats.append("citation_error")
    if s["injection_resisted"] is False:
        cats.append("prompt_injection_followed")
    return cats
