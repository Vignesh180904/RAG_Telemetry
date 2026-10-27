"""
LEGACY (original project). Embedding-similarity metrics, kept for reference and backward compatibility.
The evaluation runner now uses app/evaluation/scoring.py (label-based, deterministic) because these
scores could not tell a correct answer from a fluent wrong one: in the original report 5 of 20 answers
were wrong but only 2 questions were flagged -- one of them a CORRECT refusal.
"""
"""
RAG Evaluation Metrics
======================
Each function below returns (score, reason) so you can see WHY a score
was given, not just a number -- as required by the project spec.

All metrics here are computed with embedding similarity (via the same
free local Embedder used elsewhere) so they work with zero API cost and
zero extra setup. This is a well-known evaluation technique (it's what
libraries like RAGAS do under the hood when you don't configure an
LLM-judge). If you later want more nuanced grading, you can swap any of
these functions to call an LLM-as-judge instead -- the function signature
stays the same, so nothing else in the evaluator needs to change.

Metric definitions (simple explanations with examples):

1. Context Relevance
   "Are the retrieved chunks actually about the question?"
   Example: Q: "What is the refund window?"
            Retrieved: "Refunds are accepted within 30 days."  -> relevant
            Retrieved: "Our office is in Chennai."             -> irrelevant

2. Context Precision
   "Out of everything retrieved, what fraction was actually useful?"
   If we retrieved 3 chunks and only 1 was truly relevant, precision = 1/3.

3. Context Recall
   "Did we succeed in retrieving the ONE piece of context that was needed
   to answer this question?" (measured against the dataset's
   `relevant_context` field)

4. Answer Relevance
   "Does the generated answer actually address the question asked?"
   (independent of whether it's factually correct)

5. Faithfulness / Groundedness
   "Is the generated answer actually supported by the retrieved context,
   or did the LLM make something up?" This is the #1 hallucination check.
"""

from typing import List, Tuple
from app.ingestion.embeddings import Embedder

RELEVANCE_THRESHOLD = 0.45  # cosine similarity above this = "relevant" (tune as needed)


def context_relevance(embedder: Embedder, question: str, retrieved_texts: List[str]) -> Tuple[float, str]:
    if not retrieved_texts:
        return 0.0, "No chunks were retrieved at all."
    q_emb = embedder.embed_text(question)
    sims = [embedder.cosine_similarity(q_emb, embedder.embed_text(t)) for t in retrieved_texts]
    avg_sim = sum(sims) / len(sims)
    reason = f"Average similarity between question and {len(retrieved_texts)} retrieved chunks: {avg_sim:.3f}"
    return round(avg_sim, 4), reason


def context_precision(embedder: Embedder, question: str, retrieved_texts: List[str]) -> Tuple[float, str]:
    if not retrieved_texts:
        return 0.0, "No chunks were retrieved."
    q_emb = embedder.embed_text(question)
    relevant_flags = []
    for t in retrieved_texts:
        sim = embedder.cosine_similarity(q_emb, embedder.embed_text(t))
        relevant_flags.append(sim >= RELEVANCE_THRESHOLD)
    precision = sum(relevant_flags) / len(relevant_flags)
    reason = f"{sum(relevant_flags)}/{len(relevant_flags)} retrieved chunks scored above the relevance threshold ({RELEVANCE_THRESHOLD})."
    return round(precision, 4), reason


def context_recall(embedder: Embedder, ground_truth_context: str, retrieved_texts: List[str]) -> Tuple[float, str]:
    if not retrieved_texts or not ground_truth_context:
        return 0.0, "Missing retrieved chunks or ground-truth context."
    gt_emb = embedder.embed_text(ground_truth_context)
    sims = [embedder.cosine_similarity(gt_emb, embedder.embed_text(t)) for t in retrieved_texts]
    best_sim = max(sims)
    found = best_sim >= RELEVANCE_THRESHOLD
    reason = (
        f"Best match between retrieved chunks and the known relevant context scored {best_sim:.3f} "
        f"({'found' if found else 'NOT found'} above threshold {RELEVANCE_THRESHOLD})."
    )
    return (1.0 if found else 0.0), reason


def answer_relevance(embedder: Embedder, question: str, generated_answer: str) -> Tuple[float, str]:
    if not generated_answer.strip() or "don't know" in generated_answer.lower():
        return 0.0, "Answer was empty or a refusal ('I don't know')."
    q_emb = embedder.embed_text(question)
    a_emb = embedder.embed_text(generated_answer)
    sim = embedder.cosine_similarity(q_emb, a_emb)
    reason = f"Similarity between question and generated answer: {sim:.3f}"
    return round(sim, 4), reason


def faithfulness(embedder: Embedder, generated_answer: str, retrieved_texts: List[str]) -> Tuple[float, str]:
    if not generated_answer.strip():
        return 0.0, "Answer was empty."
    if "don't know" in generated_answer.lower():
        # Correctly refusing when it doesn't know is faithful behavior.
        return 1.0, "Model correctly declined to answer instead of guessing -- fully grounded."
    if not retrieved_texts:
        return 0.0, "Answer was generated but no context was retrieved to ground it in."
    a_emb = embedder.embed_text(generated_answer)
    sims = [embedder.cosine_similarity(a_emb, embedder.embed_text(t)) for t in retrieved_texts]
    best_sim = max(sims)
    reason = f"Best similarity between the answer and any retrieved chunk: {best_sim:.3f} (higher = more grounded)."
    return round(best_sim, 4), reason
