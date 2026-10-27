"""
Request telemetry and metrics
=============================
For every request:
1. OpenTelemetry instruments are updated (counters / histograms). They stay in memory unless
   METRICS_CONSOLE_EXPORT=true; an OTLP metrics exporter can be added the same way as for traces.
2. One JSON line is appended to TELEMETRY_LOG_PATH (default outputs/telemetry_log.jsonl).

Honesty rules:
* Only measured values are recorded. Token counts come from Ollama's prompt_eval_count/eval_count;
  if a provider does not return them they are null, never estimated.
* Cost: local Ollama inference has no per-token price, so `estimated_cost_usd` is null with a note.
  (Electricity/hardware cost is out of scope.)
* Query text is NOT stored by default (TELEMETRY_LOG_QUERY_TEXT=false): only a SHA-256 prefix and
  the length, so logs can correlate repeated questions without keeping private content.

Retention: the log rotates to `.1` when it exceeds TELEMETRY_LOG_MAX_MB (one backup kept).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, InMemoryMetricReader, PeriodicExportingMetricReader

from app.config import settings

TELEMETRY_LOG_MAX_MB = 50
_lock = threading.Lock()
_instruments: Dict[str, Any] = {}
_reader = None


def setup_metrics():
    global _reader
    if _instruments:
        return _instruments
    if settings.metrics_console_export:
        _reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60000)
    else:
        _reader = InMemoryMetricReader()   # no background thread, no console noise
    meter = MeterProvider(metric_readers=[_reader]).get_meter(settings.service_name)
    _instruments.update({
        "requests": meter.create_counter("rag_requests_total"),
        "errors": meter.create_counter("rag_errors_total"),
        "abstentions": meter.create_counter("rag_abstentions_total"),
        "retries": meter.create_counter("rag_model_retries_total"),
        "total_ms": meter.create_histogram("rag_request_latency_ms", unit="ms"),
        "retrieval_ms": meter.create_histogram("rag_retrieval_latency_ms", unit="ms"),
        "embedding_ms": meter.create_histogram("rag_embedding_latency_ms", unit="ms"),
        "rerank_ms": meter.create_histogram("rag_rerank_latency_ms", unit="ms"),
        "llm_ms": meter.create_histogram("rag_llm_latency_ms", unit="ms"),
    })
    return _instruments


def hash_query(q: str) -> str:
    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]


@dataclass
class RequestRecord:
    request_id: str
    endpoint: str
    trace_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    query_hash: str = ""
    query_chars: int = 0
    query_text: Optional[str] = None
    route: Optional[str] = None
    complexity: Optional[str] = None
    final_from: Optional[str] = None
    abstained: Optional[bool] = None
    embedding_model: str = ""
    retrieval_mode: Optional[str] = None
    top_k: Optional[int] = None
    embedding_ms: Optional[float] = None
    dense_ms: Optional[float] = None
    sparse_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    dense_hits: Optional[int] = None
    sparse_hits: Optional[int] = None
    chunks_returned: int = 0
    retrieved_doc_ids: List[str] = field(default_factory=list)
    rewrite_attempts: int = 0
    verifications: int = 0
    model_calls: List[Dict[str, Any]] = field(default_factory=list)
    llm_ms: float = 0.0
    model_retries: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_complete: bool = False            # True only if EVERY model call reported token counts
    estimated_cost_usd: Optional[float] = None
    cost_note: str = "Local inference: no per-token price; cost not estimated."
    total_ms: float = 0.0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def build_record(request_id: str, endpoint: str, question: str, result: Optional[Dict[str, Any]],
                 total_ms: float, trace_id: Optional[str], embedding_model: str,
                 error_code: Optional[str] = None, error_message: Optional[str] = None) -> RequestRecord:
    rec = RequestRecord(request_id=request_id, endpoint=endpoint, trace_id=trace_id,
                        query_hash=hash_query(question), query_chars=len(question),
                        query_text=question[:500] if settings.telemetry_log_query_text else None,
                        embedding_model=embedding_model, total_ms=round(total_ms, 2),
                        error_code=error_code, error_message=(error_message or None) and error_message[:300])
    if not result:
        return rec
    stats = result.get("retrieval_stats") or []

    def total(key):
        vals = [s.get(key) for s in stats if s.get(key) is not None]
        return round(sum(vals), 2) if vals else None

    calls = result.get("model_calls") or []
    rec.route, rec.complexity = result.get("route"), result.get("complexity")
    rec.final_from, rec.abstained = result.get("final_from"), result.get("abstained")
    rec.retrieval_mode = stats[-1].get("mode") if stats else None
    rec.top_k = stats[-1].get("top_k") if stats else None
    for k in ("embedding_ms", "dense_ms", "sparse_ms", "rerank_ms", "retrieval_ms"):
        setattr(rec, k, total(k))
    rec.dense_hits = stats[-1].get("dense_hits") if stats else None
    rec.sparse_hits = stats[-1].get("sparse_hits") if stats else None
    rec.chunks_returned = len(result.get("chunks") or [])
    rec.retrieved_doc_ids = [c.get("id") for c in result.get("chunks") or []]
    rec.rewrite_attempts = result.get("attempts", 0)
    rec.verifications = sum(1 for s in result.get("steps", []) if s["node"] == "evidence_check")
    rec.model_calls = calls
    rec.llm_ms = round(sum(c.get("latency_ms") or 0 for c in calls), 2)
    rec.model_retries = sum(max((c.get("attempts") or 1) - 1, 0) for c in calls)
    if calls:
        pt = [c.get("prompt_tokens") for c in calls]
        ct = [c.get("completion_tokens") for c in calls]
        rec.tokens_complete = all(v is not None for v in pt + ct)
        rec.prompt_tokens = sum(v for v in pt if v is not None) if any(v is not None for v in pt) else None
        rec.completion_tokens = sum(v for v in ct if v is not None) if any(v is not None for v in ct) else None
    rec.warnings = result.get("warnings") or []
    if result.get("error") and not rec.error_code:
        rec.error_code = result["error"].get("code")
        rec.error_message = (result["error"].get("message") or "")[:300]
    return rec


def record_request(rec: RequestRecord, path: Optional[str] = None) -> None:
    inst = setup_metrics()
    attrs = {"route": rec.route or "none", "endpoint": rec.endpoint}
    inst["requests"].add(1, attrs)
    if rec.error_code:
        inst["errors"].add(1, {**attrs, "code": rec.error_code})
    if rec.abstained:
        inst["abstentions"].add(1, attrs)
    if rec.model_retries:
        inst["retries"].add(rec.model_retries, attrs)
    inst["total_ms"].record(rec.total_ms, attrs)
    for k in ("retrieval_ms", "embedding_ms", "rerank_ms"):
        if getattr(rec, k) is not None:
            inst[k].record(getattr(rec, k), attrs)
    for c in rec.model_calls:
        inst["llm_ms"].record(c.get("latency_ms") or 0, {"role": c.get("role", "?"), "model": c.get("model", "?")})

    path = path or settings.telemetry_log_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(asdict(rec), ensure_ascii=False, default=str)
    with _lock:
        if os.path.exists(path) and os.path.getsize(path) > TELEMETRY_LOG_MAX_MB * 1024 * 1024:
            os.replace(path, path + ".1")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _pct(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round(p / 100 * (len(vals) - 1)))))
    return round(vals[idx], 2)


def latency_stats(vals: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": len(vals), "mean": round(statistics.fmean(vals), 2), "p50": _pct(vals, 50),
            "p95": _pct(vals, 95), "max": round(max(vals), 2)}


def summarize(path: Optional[str] = None, last_n: int = 1000) -> Dict[str, Any]:
    """Aggregate the most recent `last_n` JSONL records (reads the file; no external service needed)."""
    path = path or settings.telemetry_log_path
    if not os.path.exists(path):
        return {"requests": 0, "note": "No telemetry recorded yet."}
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "request_id" in r:          # skip records written by the original project's format
                recs.append(r)
    recs = recs[-last_n:]
    if not recs:
        return {"requests": 0, "note": "No records in the current format yet."}
    by = lambda key: {v: sum(1 for r in recs if r.get(key) == v) for v in sorted({str(r.get(key)) for r in recs})}
    models: Dict[str, List[float]] = {}
    timeouts = 0
    for r in recs:
        for c in r.get("model_calls", []):
            models.setdefault(f"{c.get('role')}:{c.get('model')}", []).append(c.get("latency_ms") or 0)
            timeouts += c.get("error_code") == "model_timeout"
    known_pt = [r["prompt_tokens"] for r in recs if r.get("prompt_tokens") is not None]
    known_ct = [r["completion_tokens"] for r in recs if r.get("completion_tokens") is not None]
    n = len(recs)
    return {
        "requests": n,
        "window": {"first": recs[0]["timestamp"], "last": recs[-1]["timestamp"]},
        "error_rate": round(sum(1 for r in recs if r.get("error_code")) / n, 4),
        "errors_by_code": {k: v for k, v in by("error_code").items() if k != "None"},
        "abstention_rate": round(sum(1 for r in recs if r.get("abstained")) / n, 4),
        "routes": by("route"),
        "complexity": by("complexity"),
        "final_from": by("final_from"),
        "latency_ms": {k: latency_stats([r.get(k) for r in recs]) for k in
                       ("total_ms", "retrieval_ms", "embedding_ms", "rerank_ms", "llm_ms")},
        "model_latency_ms": {k: latency_stats(v) for k, v in models.items()},
        "model_retries": sum(r.get("model_retries", 0) for r in recs),
        "model_timeouts": timeouts,
        "rewrite_attempts": sum(r.get("rewrite_attempts", 0) for r in recs),
        "tokens": {"prompt_total_reported": sum(known_pt), "completion_total_reported": sum(known_ct),
                   "requests_with_token_counts": len(known_pt),
                   "note": "Only provider-reported counts; requests without counts are excluded, not estimated."},
        "cost": {"estimated_usd": None, "note": "Local inference: no defensible per-token price."},
    }
