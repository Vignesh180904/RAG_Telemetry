"""API-level test of POST /agent/query (no embedding model or ChromaDB data needed)."""

from fastapi.testclient import TestClient

from app import main
from app.agent.graph import RagAgent
from app.generation.llm import MockLLM
from tests.test_agent import POLICY, FakeRetriever


def _client(responder):
    # TestClient without `with` -> lifespan (model loading) is skipped; we inject the agent.
    main.state["agent"] = RagAgent(FakeRetriever(responder=responder), MockLLM())
    return TestClient(main.app)


def test_agent_query_returns_path_and_answer():
    r = _client(lambda q: [(POLICY, 0.8)]).post(
        "/agent/query", json={"question": "How many days do I have to return a product?"})
    body = r.json()
    assert r.status_code == 200
    assert "30 days" in body["answer"] and body["route"] == "retrieve"
    assert [s["node"] for s in body["steps"]] == ["route", "retrieve", "grade", "generate", "verify"]
    assert body["sources"] == ["policy.txt"] and body["grounded"] is True


def test_agent_query_calculator():
    body = _client(lambda q: []).post("/agent/query", json={"question": "what is 12 * 12?"}).json()
    assert body["answer"] == "12 * 12 = 144" and body["tool_used"] == "calculator"


def test_agent_query_rejects_empty_question():
    assert _client(lambda q: []).post("/agent/query", json={"question": "  "}).status_code == 400
