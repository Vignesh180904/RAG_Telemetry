"""
Adaptive multi-LLM RAG workflow (LangGraph)
===========================================

    START -> router --math--> calculate --------------------------------------------> END
               |-----chat---> chat -------------------------------------------------> END
               '--retrieve--> retrieve -> grade --relevant--> generate -> verify --simple & grounded--> END
                                 ^          |                     |          |
                                 |          |--weak--> rewrite ---'          '--complex OR ungrounded--> evidence_check
                                 '----------'   (bounded)                                                 |   ^
                                            '--nothing relevant--> no_answer -> END     conflict (bounded)|   |
                                                                                  expand_evidence <-------'   |
                                                                                        '---------------------'
                                                                       evidence_check --> synthesize -> END
             any generator failure -> model_error -> END

Roles (configurable in .env, may all be the same physical model):
  generator   drafts an evidence-grounded, cited answer
  verifier    checks the draft against the evidence (structured JSON)
  synthesizer writes the final answer from draft + verification, or abstains

Models run strictly one after another (see model_manager: a process-wide lock).
"""

from __future__ import annotations

import functools
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.agent import roles
from app.agent.roles import REFUSAL, Draft, Verification
from app.agent.tools import calculator, extract_math_expression
from app.config import settings
from app.generation.llm import STOPWORDS, BaseLLM
from app.generation.model_manager import ModelError
from app.retrieval.retriever import RetrievedChunk
from app.telemetry.tracing import traced_span

CHAT_REPLY = "Hi! Ask me a question about your uploaded documents, or give me a calculation."
MODEL_ERROR_REPLY = "The answer could not be generated because the language model is unavailable."

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|good (morning|afternoon|evening)|how are you)\b", re.I)


class AgentState(TypedDict, total=False):
    question: str
    query: str
    top_k: Optional[int]
    source_filter: Optional[str]
    where: Optional[Dict[str, Any]]
    route: str                       # retrieve | calculate | chat
    complexity: str                  # simple | complex
    complexity_reasons: List[str]
    chunks: List[RetrievedChunk]
    retrieval_stats: List[Dict[str, Any]]
    attempts: int
    rewrote: bool
    relevant: bool
    relevance: Dict[str, Any]
    draft: Draft
    answer: str
    citations: List[int]
    grounded: bool
    grounding_score: float
    verification: Verification
    verifications: int
    final_from: str
    tool_used: Optional[str]
    llm_model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    model_calls: list
    warnings: List[str]
    error: Optional[Dict[str, str]]
    steps: List[Dict[str, Any]]


def _words(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def _node(label: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, state: AgentState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            with traced_span(f"Agent: {label}", attempt=state.get("attempts", 0)) as span:
                out = fn(self, state) or {}
                for k in ("route", "complexity", "relevant", "grounded", "query", "final_from"):
                    if k in out and isinstance(out[k], (str, bool, int, float)):
                        span.set_attribute(f"agent.{k}", out[k])
            ms = round((time.perf_counter() - t0) * 1000, 2)
            out["steps"] = list(state.get("steps", [])) + [{"node": label, "ms": ms}]
            return out
        return wrapper
    return decorator


def _docs_to_chunks(docs: List[Document]) -> List[RetrievedChunk]:
    """Accept any LangChain retriever (tests use a fake one)."""
    out = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        out.append(RetrievedChunk(id=str(m.get("id", i)), text=d.page_content, source=m.get("source", "unknown"),
                                  chunk_index=m.get("chunk_index", -1), score=float(m.get("score", 0.0)),
                                  page=m.get("page"), doc_id=m.get("doc_id", ""), found_by=["dense"], citation=i))
    return out


class RagAgent:
    def __init__(self, retriever, llm: BaseLLM, verifier_llm: Optional[BaseLLM] = None,
                 synthesizer_llm: Optional[BaseLLM] = None, router_llm: Optional[BaseLLM] = None):
        self.retriever = retriever
        self.llm = llm                                   # generator
        self.verifier_llm = verifier_llm or llm
        self.synthesizer_llm = synthesizer_llm or llm
        self.router_llm = router_llm or llm
        self.graph = self._build()

    # ------------------------------------------------------------------ helpers
    def _retrieve(self, query: str, state: AgentState, top_k: Optional[int] = None):
        k = top_k or state.get("top_k")
        if hasattr(self.retriever, "refresh_sparse_index"):        # HybridRetriever
            res = self.retriever.retrieve(query, top_k=k, source_filter=state.get("source_filter"),
                                          where=state.get("where"))
            return res.chunks, res.stats
        r = self.retriever
        overrides = {}
        if k and hasattr(r, "top_k"):
            overrides["top_k"] = k
        if state.get("source_filter") and hasattr(r, "source_filter"):
            overrides["source_filter"] = state["source_filter"]
        if overrides:
            r = r.model_copy(update=overrides)
        t0 = time.perf_counter()
        chunks = _docs_to_chunks(r.invoke(query))
        return chunks, {"mode": "langchain-retriever", "returned": len(chunks),
                        "retrieval_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @staticmethod
    def _calls(state: AgentState) -> list:
        return list(state.get("model_calls", []))

    # ------------------------------------------------------------------ nodes
    @_node("route")
    def router(self, state: AgentState):
        q = state["question"]
        cx = roles.classify_complexity(q)
        if settings.complexity_mode != "auto":
            cx = {"complexity": settings.complexity_mode, "reasons": [f"forced by COMPLEXITY_MODE={settings.complexity_mode}"]}
        base = {"complexity": cx["complexity"], "complexity_reasons": cx["reasons"]}
        if extract_math_expression(q):
            return {**base, "route": "calculate"}
        if _GREETING_RE.match(q) and len(q.split()) <= 6:
            return {**base, "route": "chat"}
        if settings.agent_router == "llm":
            calls = self._calls(state)
            prompt = ("Classify the user message. Reply with exactly one word: RETRIEVE (needs document lookup), "
                      f"CALCULATE (arithmetic) or CHAT (greeting/small talk).\nMessage: {q}\nAnswer:")
            try:
                choice = roles.call_llm(self.router_llm, "You are a request classifier.", prompt, calls=calls).text.strip().upper()
            except ModelError:
                choice = ""
            mapping = {"RETRIEVE": "retrieve", "CALCULATE": "calculate", "CHAT": "chat"}
            route = mapping.get(choice.split()[0] if choice else "", "retrieve")
            if route == "calculate" and not extract_math_expression(q):
                route = "retrieve"                      # never trust the LLM to invent an expression
            return {**base, "route": route, "model_calls": calls}
        return {**base, "route": "retrieve"}

    @_node("retrieve")
    def retrieve(self, state: AgentState):
        chunks, stats = self._retrieve(state["query"], state)
        return {"chunks": chunks, "retrieval_stats": list(state.get("retrieval_stats", [])) + [stats]}

    @_node("grade")
    def grade(self, state: AgentState):
        chunks = state.get("chunks", [])
        best = max((c.score for c in chunks), default=0.0)
        cov = max((roles.query_coverage(state["question"], c.text) for c in chunks), default=0.0)
        relevant = bool(chunks) and (best >= settings.agent_relevance_threshold or cov >= settings.agent_coverage_threshold)
        return {"relevant": relevant,
                "relevance": {"best_dense_score": round(best, 4), "best_query_coverage": round(cov, 3),
                              "dense_threshold": settings.agent_relevance_threshold,
                              "coverage_threshold": settings.agent_coverage_threshold}}

    @_node("rewrite")
    def rewrite(self, state: AgentState):
        question, old = state["question"], state["query"]
        calls = self._calls(state)
        if roles.is_mock(self.router_llm):
            new = " ".join(w for w in re.findall(r"[A-Za-z0-9']+", question.lower()) if w not in STOPWORDS)
        else:
            prompt = ("Rewrite this question as a short keyword search query for a document database. "
                      f"Reply with only the query.\nQuestion: {question}\nQuery:")
            try:
                new = roles.call_llm(self.router_llm, "You rewrite search queries.", prompt, calls=calls).text.strip().strip('"')
                new = new.splitlines()[0][:300] if new else ""
            except ModelError:
                new = " ".join(w for w in re.findall(r"[A-Za-z0-9']+", question.lower()) if w not in STOPWORDS)
        new = new or question
        return {"query": new, "attempts": state.get("attempts", 0) + 1, "rewrote": new != old, "model_calls": calls}

    @_node("generate")
    def generate(self, state: AgentState):
        calls = self._calls(state)
        try:
            d = roles.generate_answer(self.llm, state["question"], state["chunks"], calls=calls)
        except ModelError as e:
            return {"error": {"code": e.code, "message": str(e), "role": "generator"}, "model_calls": calls}
        warnings = list(state.get("warnings", []))
        if d.invalid_citations:
            warnings.append(f"Generator cited non-existent evidence {d.invalid_citations}.")
        return {"draft": d, "answer": d.answer, "citations": d.citations, "final_from": "generator",
                "llm_model": d.model, "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens,
                "model_calls": calls, "warnings": warnings}

    @_node("verify")
    def verify(self, state: AgentState):
        """Cheap lexical grounding gate on every answer (no model call)."""
        answer = state["answer"]
        if roles.is_refusal(answer):
            return {"grounded": True, "grounding_score": 1.0, "final_from": "refusal"}
        aw = _words(re.sub(r"\[\d+\]", "", answer))
        cw = _words(" ".join(c.text for c in state["chunks"]))
        score = len(aw & cw) / len(aw) if aw else 0.0
        return {"grounded": score >= settings.agent_grounding_threshold, "grounding_score": round(score, 3)}

    @_node("evidence_check")
    def evidence_check(self, state: AgentState):
        calls, warnings = self._calls(state), list(state.get("warnings", []))
        try:
            v = roles.verify_answer(self.verifier_llm, state["question"], state["answer"], state["chunks"], calls=calls)
        except ModelError as e:
            v = roles.lexical_verify(state["answer"], state["chunks"])
            v.method, v.note = "lexical_fallback", f"Verifier model failed ({e.code}); used lexical check."
            warnings.append(v.note)
        return {"verification": v, "verifications": state.get("verifications", 0) + 1,
                "model_calls": calls, "warnings": warnings}

    @_node("expand_evidence")
    def expand_evidence(self, state: AgentState):
        """Conflicting evidence: fetch a wider context so the next verification pass sees more of both sides."""
        k = (state.get("top_k") or settings.top_k) * 2
        chunks, stats = self._retrieve(state["query"], state, top_k=k)
        return {"chunks": chunks, "retrieval_stats": list(state.get("retrieval_stats", [])) + [stats]}

    @_node("synthesize")
    def synthesize(self, state: AgentState):
        calls, warnings = self._calls(state), list(state.get("warnings", []))
        v: Verification = state["verification"]
        try:
            d = roles.synthesize_answer(self.synthesizer_llm, state["question"], state["draft"], v, state["chunks"], calls=calls)
        except ModelError as e:
            warnings.append(f"Synthesizer model failed ({e.code}); returning the verified draft.")
            if v.verdict in ("unsupported", "insufficient_evidence"):
                return {"answer": REFUSAL, "citations": [], "final_from": "refusal", "model_calls": calls, "warnings": warnings}
            return {"model_calls": calls, "warnings": warnings}
        return {"answer": d.answer, "citations": d.citations,
                "final_from": "refusal" if roles.is_refusal(d.answer) else "synthesizer",
                "model_calls": calls, "warnings": warnings}

    @_node("calculate")
    def calculate(self, state: AgentState):
        expr = extract_math_expression(state["question"])
        result = calculator.invoke({"expression": expr})
        answer = result if result.startswith("Error") else f"{expr} = {result}"
        return {"answer": answer, "tool_used": "calculator", "chunks": [], "final_from": "tool"}

    @_node("chat")
    def chat(self, state: AgentState):
        return {"answer": CHAT_REPLY, "chunks": [], "final_from": "chat"}

    @_node("no_answer")
    def no_answer(self, state: AgentState):
        return {"answer": REFUSAL, "citations": [], "final_from": "refusal"}

    @_node("model_error")
    def model_error(self, state: AgentState):
        return {"answer": MODEL_ERROR_REPLY, "citations": [], "final_from": "error"}

    # ------------------------------------------------------------------ edges
    @staticmethod
    def _after_grade(state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("attempts", 0) < settings.agent_max_retries:
            return "rewrite"
        return "no_answer"

    @staticmethod
    def _after_generate(state: AgentState) -> str:
        return "model_error" if state.get("error") else "verify"

    @staticmethod
    def _after_verify(state: AgentState) -> str:
        if roles.is_refusal(state["answer"]):
            return "end"
        if state.get("complexity") == "complex" or not state.get("grounded"):
            return "evidence_check"
        return "end"

    @staticmethod
    def _after_evidence_check(state: AgentState) -> str:
        v: Verification = state["verification"]
        extra = state.get("verifications", 1) - 1
        if v.conflicting_evidence and extra < settings.max_extra_verifications:
            return "expand_evidence"
        return "synthesize"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("router", self.router)  # node name != state key "route"
        for name in ("retrieve", "grade", "rewrite", "generate", "verify", "evidence_check", "expand_evidence",
                     "synthesize", "calculate", "chat", "no_answer", "model_error"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "router")
        g.add_conditional_edges("router", lambda s: s["route"],
                                {"retrieve": "retrieve", "calculate": "calculate", "chat": "chat"})
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade,
                                {"generate": "generate", "rewrite": "rewrite", "no_answer": "no_answer"})
        g.add_conditional_edges("rewrite", lambda s: "retrieve" if s.get("rewrote") else "no_answer",
                                {"retrieve": "retrieve", "no_answer": "no_answer"})
        g.add_conditional_edges("generate", self._after_generate, {"verify": "verify", "model_error": "model_error"})
        g.add_conditional_edges("verify", self._after_verify, {"evidence_check": "evidence_check", "end": END})
        g.add_conditional_edges("evidence_check", self._after_evidence_check,
                                {"expand_evidence": "expand_evidence", "synthesize": "synthesize"})
        g.add_edge("expand_evidence", "evidence_check")
        for terminal in ("synthesize", "calculate", "chat", "no_answer", "model_error"):
            g.add_edge(terminal, END)
        return g.compile()

    # ------------------------------------------------------------------ API
    def run(self, question: str, top_k: Optional[int] = None, source_filter: Optional[str] = None,
            where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        final = self.graph.invoke({
            "question": question, "query": question, "top_k": top_k, "source_filter": source_filter, "where": where,
            "attempts": 0, "steps": [], "chunks": [], "model_calls": [], "warnings": [], "verifications": 0,
            "retrieval_stats": [],
        })
        chunks: List[RetrievedChunk] = final.get("chunks", [])
        cited = set(final.get("citations", []))
        v: Optional[Verification] = final.get("verification")
        calls = [c.to_dict() if hasattr(c, "to_dict") else c for c in final.get("model_calls", [])]
        return {
            "answer": final["answer"],
            "route": final["route"],
            "complexity": final.get("complexity"),
            "complexity_reasons": final.get("complexity_reasons", []),
            "query_used": final.get("query", question),
            "rewritten": final.get("query", question) != question,
            "attempts": final.get("attempts", 0),
            "relevance": final.get("relevance"),
            "grounded": final.get("grounded"),
            "grounding_score": final.get("grounding_score"),
            "verification": v.to_dict() if v else None,
            "final_from": final.get("final_from"),
            "abstained": roles.is_refusal(final["answer"]),
            "tool_used": final.get("tool_used"),
            "citations": [{"id": c.citation, "source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                          for c in chunks if c.citation in cited],
            "chunks": [{"id": c.id, "citation": c.citation, "source": c.source, "page": c.page,
                        "chunk_index": c.chunk_index, "score": c.score, "sparse_score": c.sparse_score,
                        "rrf_score": c.rrf_score, "rerank_score": c.rerank_score, "found_by": c.found_by,
                        "text": c.text} for c in chunks],
            "retrieval_stats": final.get("retrieval_stats", []),
            "steps": final["steps"],
            "model_calls": calls,
            "llm_model": final.get("llm_model", ""),
            "prompt_tokens": final.get("prompt_tokens"),
            "completion_tokens": final.get("completion_tokens"),
            "warnings": final.get("warnings", []),
            "error": final.get("error"),
        }

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
