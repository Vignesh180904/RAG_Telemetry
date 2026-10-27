"""
Tests for the agentic workflow. They need NO embedding model and NO ChromaDB:
a fake LangChain retriever stands in for the vector store, and the free
MockLLM (or a tiny stub) stands in for the LLM. So they run in about a second.
"""

from typing import Any, Callable, List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from app.agent.graph import CHAT_REPLY, REFUSAL, RagAgent
from app.agent.tools import calculator, extract_math_expression
from app.config import settings
from app.generation.llm import LLMResponse, MockLLM

POLICY = "Customers can return products within 30 days of purchase for a full refund."


class FakeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    responder: Any                      # Callable[[str], list[tuple[str, float]]]
    queries: List[str] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        self.queries.append(query)
        return [
            Document(page_content=t, metadata={"id": f"c{i}", "source": "policy.txt",
                                               "chunk_index": i, "score": s})
            for i, (t, s) in enumerate(self.responder(query))
        ]


class CountingLLM(MockLLM):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return super().generate(prompt)


class StubLLM:
    name = "stub"

    def __init__(self, text):
        self.text = text

    def generate(self, prompt):
        return LLMResponse(text=self.text, model="stub")


def nodes(result):
    return [s["node"] for s in result["steps"]]


# ---------------------------------------------------------------- tools
def test_calculator_basic():
    assert calculator.invoke({"expression": "(2 + 3) * 4"}) == "20"
    assert calculator.invoke({"expression": "7 / 2"}) == "3.5"


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "2 ** 1000", "abs(-1)"])
def test_calculator_rejects_unsafe(bad):
    assert calculator.invoke({"expression": bad}).startswith("Error")


def test_calculator_division_by_zero():
    assert "division by zero" in calculator.invoke({"expression": "1/0"})


def test_extract_math_expression():
    assert extract_math_expression("what is 15 * 4?") == "15 * 4"
    assert extract_math_expression("what is 6 x 7") == "6 * 7"
    assert extract_math_expression("What is the 30-day return policy?") is None
    assert extract_math_expression("how are you") is None


# ---------------------------------------------------------------- routing
def test_math_goes_to_calculator_not_retrieval():
    r = FakeRetriever(responder=lambda q: [])
    out = RagAgent(r, MockLLM()).run("what is 15 * 4?")
    assert out["route"] == "calculate" and out["tool_used"] == "calculator"
    assert out["answer"] == "15 * 4 = 60"
    assert r.queries == []                           # never touched the vector store


def test_greeting_goes_to_chat():
    r = FakeRetriever(responder=lambda q: [])
    out = RagAgent(r, MockLLM()).run("hello!")
    assert out["route"] == "chat" and out["answer"] == CHAT_REPLY and r.queries == []


def test_number_in_question_is_not_math():
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.8)])
    out = RagAgent(r, MockLLM()).run("What is the 30-day return policy?")
    assert out["route"] == "retrieve"


def test_llm_router_falls_back_to_retrieve_on_garbage(monkeypatch):
    monkeypatch.setattr(settings, "agent_router", "llm")
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.8)])
    out = RagAgent(r, StubLLM("banana")).run("How many days do I have to return a product?")
    assert out["route"] == "retrieve"


# ---------------------------------------------------------------- retrieval loop
def test_relevant_docs_go_straight_to_generation():
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.8)])
    out = RagAgent(r, MockLLM()).run("How many days do I have to return a product?")
    assert nodes(out) == ["route", "retrieve", "grade", "generate", "verify"]
    assert "30 days" in out["answer"] and out["grounded"] is True
    assert out["chunks"][0]["source"] == "policy.txt"


def test_poor_retrieval_triggers_rewrite_then_recovers():
    original = "How can I get my money back for a purchase?"
    # low similarity for the original wording, high once the agent rewrites it
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.8 if q != original else 0.1)])
    out = RagAgent(r, MockLLM()).run(original)
    assert nodes(out) == ["route", "retrieve", "grade", "rewrite", "retrieve", "grade", "generate", "verify"]
    assert out["rewritten"] and out["query_used"] != original and out["attempts"] == 1
    assert len(r.queries) == 2


def test_nothing_relevant_refuses_without_calling_llm():
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.05)])
    llm = CountingLLM()
    out = RagAgent(r, llm).run("Who won the football world cup in recent years?")
    assert out["answer"] == REFUSAL
    assert nodes(out)[-1] == "no_answer" and llm.calls == 0


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_retries", 0)
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.05)])
    out = RagAgent(r, MockLLM()).run("Who won the football world cup in recent years?")
    assert "rewrite" not in nodes(out) and len(r.queries) == 1


def test_unchanged_rewrite_does_not_loop():
    r = FakeRetriever(responder=lambda q: [])        # nothing ever found
    out = RagAgent(r, MockLLM()).run("What is it?")  # only stopwords -> rewrite can't change it
    assert nodes(out) == ["route", "retrieve", "grade", "rewrite", "no_answer"]
    assert len(r.queries) == 1


def test_empty_retrieval_is_not_relevant():
    r = FakeRetriever(responder=lambda q: [])
    out = RagAgent(r, MockLLM()).run("Tell me about warranty coverage details")
    assert out["answer"] == REFUSAL


# ---------------------------------------------------------------- verification
def test_verify_flags_ungrounded_answer():
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.8)])
    out = RagAgent(r, StubLLM("Bananas orbit Jupiter quickly every Tuesday")).run("How long do returns take?")
    assert out["grounded"] is False and out["grounding_score"] < settings.agent_grounding_threshold


def test_every_step_has_timing():
    r = FakeRetriever(responder=lambda q: [(POLICY, 0.8)])
    out = RagAgent(r, MockLLM()).run("How many days do I have to return a product?")
    assert all(s["ms"] >= 0 for s in out["steps"])
