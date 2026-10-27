"""Telemetry: trace spans, request records (measured vs unavailable values), privacy, summary."""

import json

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agent.graph import RagAgent
from app.config import settings
from app.generation.llm import MockLLM
from app.telemetry import tracing
from app.telemetry.metrics import build_record, hash_query, latency_stats, record_request, summarize
from tests.test_agent import POLICY, FakeRetriever


def _run(q="How many days do I have to return a product?"):
    return RagAgent(FakeRetriever(responder=lambda _: [(POLICY, 0.8)]), MockLLM()).run(q)


def test_every_node_creates_a_span_in_one_trace():
    exporter = InMemorySpanExporter()
    tracing._provider = None
    tracing.setup_tracing(exporter=exporter)
    try:
        with tracing.traced_span("RAG Request"):
            trace_id = tracing.current_trace_id()
            _run()
        spans = exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert {"Agent: route", "Agent: retrieve", "Agent: grade", "Agent: generate", "Agent: verify", "RAG Request"} <= set(names)
        assert {format(s.context.trace_id, "032x") for s in spans} == {trace_id}
    finally:
        tracing._provider = None
        tracing.setup_tracing(kind="none")


def test_span_records_exceptions():
    exporter = InMemorySpanExporter()
    tracing._provider = None
    tracing.setup_tracing(exporter=exporter)
    try:
        try:
            with tracing.traced_span("boom"):
                raise ValueError("x")
        except ValueError:
            pass
        span = exporter.get_finished_spans()[0]
        assert span.attributes["error"] is True and span.attributes["error.type"] == "ValueError"
        assert span.events[0].name == "exception"
    finally:
        tracing._provider = None
        tracing.setup_tracing(kind="none")


def test_record_does_not_store_query_text_by_default():
    q = "my private question about salary"
    rec = build_record("r1", "/query", q, _run(), 12.3, "t1", "hash:hash-384")
    assert rec.query_text is None and rec.query_hash == hash_query(q) and rec.query_chars == len(q)


def test_query_text_can_be_enabled(monkeypatch):
    monkeypatch.setattr(settings, "telemetry_log_query_text", True)
    assert build_record("r1", "/query", "hello there", _run(), 1, None, "m").query_text == "hello there"


def test_mock_tokens_are_reported_as_given_and_cost_is_not_invented():
    rec = build_record("r1", "/query", "q", _run(), 5, None, "m")
    assert rec.estimated_cost_usd is None and "not estimated" in rec.cost_note
    assert rec.route == "retrieve" and rec.chunks_returned == 1


def test_missing_token_counts_stay_null():
    result = _run()
    result["model_calls"] = [{"role": "generator", "model": "x", "latency_ms": 5, "prompt_tokens": None,
                              "completion_tokens": None, "attempts": 2}]
    rec = build_record("r1", "/query", "q", result, 5, None, "m")
    assert rec.prompt_tokens is None and rec.completion_tokens is None and rec.tokens_complete is False
    assert rec.model_retries == 1 and rec.llm_ms == 5


def test_error_records_and_summary(tmp_path):
    path = str(tmp_path / "t.jsonl")
    record_request(build_record("r1", "/query", "q1", _run(), 10, None, "m"), path)
    record_request(build_record("r2", "/query", "q2", None, 3, None, "m", error_code="model_timeout",
                                error_message="slow"), path)
    lines = [json.loads(l) for l in open(path)]
    assert [l["request_id"] for l in lines] == ["r1", "r2"]
    s = summarize(path)
    assert s["requests"] == 2 and s["error_rate"] == 0.5 and s["errors_by_code"] == {"model_timeout": 1}
    assert s["latency_ms"]["total_ms"]["max"] == 10 and s["cost"]["estimated_usd"] is None


def test_summary_ignores_legacy_records(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"query": "old format", "total_latency_ms": 5}\n')
    assert summarize(str(p))["requests"] == 0


def test_latency_stats():
    s = latency_stats([10, 20, 30, 40, 1000])
    assert s["p50"] == 30 and s["max"] == 1000 and s["count"] == 5
    assert latency_stats([])["mean"] is None
