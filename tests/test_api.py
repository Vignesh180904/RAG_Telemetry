"""API tests: real FastAPI app + real ChromaDB (tmp) + hash embeddings + mock LLM."""

import io
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.generation.model_manager import ModelTimeout


@pytest.fixture
def client(services, test_settings, monkeypatch):
    for k in ("documents_dir", "max_upload_mb", "eval_output_dir"):
        monkeypatch.setattr(settings, k, getattr(test_settings, k))
    monkeypatch.setitem(main.state, "services", services)
    monkeypatch.setitem(main.state, "agent", services.agent)
    return TestClient(main.app)   # no `with`: lifespan (real model loading) is skipped


def err(r):
    return r.json()["error"]


def test_health_and_request_id(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123def"})
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.headers["X-Request-ID"] == "abc123def"
    assert len(client.get("/health").headers["X-Request-ID"]) == 16


def test_ready_reports_components(client):
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert set(body["components"]) == {"embeddings", "vector_store", "sparse_index", "llm"}
    assert body["components"]["vector_store"]["chunks"] > 0


def test_ready_is_503_when_a_component_failed(client, services):
    services.errors["embeddings"] = "boom"
    emb, services.embedder = services.embedder, None
    try:
        r = client.get("/ready")
        assert r.status_code == 503 and r.json()["components"]["embeddings"]["error"] == "boom"
    finally:
        services.embedder = emb
        services.errors.clear()


def test_models_endpoint_mock(client):
    assert client.get("/models").json()["provider"] == "mock"


def test_upload_list_query_delete_roundtrip(client):
    up = client.post("/documents/upload", files={"file": ("hr_notes.txt", b"The parking permit costs INR 300 per month for employees.", "text/plain")})
    assert up.status_code == 200 and up.json()["status"] == "indexed"
    doc_id = up.json()["document_id"]
    again = client.post("/documents/upload", files={"file": ("hr_notes.txt", b"The parking permit costs INR 300 per month for employees.", "text/plain")})
    assert again.json()["status"] == "unchanged"
    items = client.get("/documents").json()["items"]
    assert any(d["document_id"] == doc_id for d in items)

    q = client.post("/query", json={"question": "How much does the parking permit cost per month?",
                                    "source_filter": "hr_notes.txt"}).json()
    assert "300" in q["answer"] and q["citations"][0]["source"] == "hr_notes.txt"
    assert q["route"] == "retrieve" and q["request_id"] and q["latency_ms"]["total_ms"] > 0
    assert all(c["source"] == "hr_notes.txt" for c in q["retrieved_chunks"])

    d = client.delete(f"/documents/{doc_id}")
    assert d.status_code == 200 and d.json()["chunks_deleted"] >= 1
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_legacy_upload_path_still_works(client):
    r = client.post("/upload", files={"file": ("legacy.txt", b"Legacy route content for testing purposes.", "text/plain")})
    assert r.status_code == 200 and r.json()["chunks_created"] >= 1


@pytest.mark.parametrize("name,data,code", [
    ("evil.exe", b"MZ", "unsupported_file_type"),
    ("empty.txt", b"", "unreadable_document"),
    ("fake.pdf", b"hello", "unreadable_document"),
])
def test_upload_rejections(client, name, data, code):
    r = client.post("/documents/upload", files={"file": (name, data, "application/octet-stream")})
    assert r.status_code == 400 and err(r)["code"] == code


def test_upload_path_traversal_is_neutralised(client, test_settings):
    import os
    r = client.post("/documents/upload", files={"file": ("../../outside.txt", b"Traversal attempt content here.", "text/plain")})
    assert r.status_code == 200 and r.json()["filename"] == "outside.txt"
    assert os.path.exists(os.path.join(test_settings.documents_dir, "outside.txt"))


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    r = client.post("/documents/upload", files={"file": ("big.txt", io.BytesIO(b"a" * (1024 * 1024 + 10)), "text/plain")})
    assert r.status_code == 413 and err(r)["code"] == "file_too_large"


def test_invalid_requests_have_consistent_error_shape(client):
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400 and err(r)["code"] == "empty_question" and err(r)["request_id"]
    r = client.post("/query", json={"top_k": 3})
    assert r.status_code == 422 and err(r)["code"] == "invalid_request"
    r = client.post("/query", json={"question": "x", "top_k": 999})
    assert r.status_code == 422
    assert client.delete("/documents/../../etc").status_code in (400, 404)
    assert err(client.get("/no-such-route"))["code"] == "http_error"


def test_unanswerable_question_abstains(client):
    body = client.post("/query", json={"question": "What is the capital of France?"}).json()
    assert body["abstained"] is True and body["citations"] == []


def test_model_failure_is_503_without_internal_details(client, services, monkeypatch):
    class Broken:
        name = "broken"
        def generate(self, prompt):
            raise ModelTimeout("Model 'x' did not answer within 120s.")
    monkeypatch.setattr(services.agent, "llm", Broken())
    r = client.post("/query", json={"question": "How often do automatic backups run in CloudSync Pro?"})
    assert r.status_code == 503 and err(r)["code"] == "model_timeout"
    assert "Traceback" not in r.text and "/home" not in r.text and "\\Users" not in r.text


def test_unexpected_exception_hides_details(client, monkeypatch, services):
    def explode(*a, **k):
        raise RuntimeError("secret path C:\\Users\\me\\db")
    monkeypatch.setattr(services.agent, "run", explode)
    r = client.post("/query", json={"question": "anything at all"})
    assert r.status_code == 500 and "secret" not in r.text and err(r)["code"] == "query_failed"


def test_telemetry_summary_after_queries(client):
    client.post("/query", json={"question": "How often do automatic backups run in CloudSync Pro?"})
    s = client.get("/telemetry/summary").json()
    assert s["requests"] >= 1 and "latency_ms" in s
    line = json.loads(open(settings.telemetry_log_path).readlines()[-1])
    assert line["query_text"] is None and line["trace_id"]


def test_evals_results_404_then_available(client, monkeypatch, test_settings):
    assert client.get("/evals/results").status_code == 404
    from app.evaluation import evaluator

    def fake_run(s, mode, dataset_path=None):
        import os
        out = os.path.join(s.eval_output_dir, "evals")
        os.makedirs(out, exist_ok=True)
        latest = {"timestamp": "t", "mode": mode, "dataset": {"version": "2.0.0"}, "summary": {},
                  "paths": {mode: {"json": os.path.join(out, "x.json"), "csv": None}}, "ingestion_errors": []}
        json.dump(latest, open(os.path.join(out, "latest.json"), "w"))
        return {"latest": latest, "reports": {}}
    monkeypatch.setattr(evaluator, "run_evaluation", fake_run)
    r = client.post("/evals/run", json={"mode": "advanced"})
    assert r.status_code == 200 and r.json()["reports"]["advanced"]["json"] == "x.json"   # no server paths leaked
    assert client.get("/evals/results").json()["mode"] == "advanced"
    assert client.post("/evals/run", json={"dataset": "../../secrets.json"}).status_code == 400
