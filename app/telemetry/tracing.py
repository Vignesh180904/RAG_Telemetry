"""
Tracing and logging
===================
OpenTelemetry tracing with a pluggable exporter:

    TRACE_EXPORTER=none     spans are created (IDs, timing, attributes) but not exported (default)
    TRACE_EXPORTER=console  print spans to stdout (local debugging)
    TRACE_EXPORTER=otlp     send to an OTLP/HTTP collector (Jaeger, Grafana Tempo, Langfuse, ...)
                            needs: pip install opentelemetry-exporter-otlp-proto-http

Every request gets a request ID and the active trace ID; both are added to log lines and to the
JSONL telemetry record, so a slow or failed request can be followed from the API response to the
log to the trace.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

from app.config import settings

_tracer = None
_provider: Optional[TracerProvider] = None
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def current_trace_id() -> Optional[str]:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx and ctx.trace_id else None


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.trace_id = current_trace_id() or "-"
        return True


def configure_logging(level: Optional[str] = None) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_rag_handler", False) for h in root.handlers):
        root.setLevel(level or settings.log_level)
        return
    h = logging.StreamHandler()
    h._rag_handler = True  # type: ignore[attr-defined]
    h.addFilter(_ContextFilter())
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] req=%(request_id)s trace=%(trace_id)s %(message)s"))
    root.addHandler(h)
    root.setLevel(level or settings.log_level)


def setup_tracing(exporter: Optional[SpanExporter] = None, kind: Optional[str] = None):
    """Idempotent. Pass `exporter` (e.g. InMemorySpanExporter) in tests."""
    global _tracer, _provider
    if _provider is not None and exporter is None:
        return _tracer
    kind = kind or settings.trace_exporter
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif kind == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif kind == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise RuntimeError("TRACE_EXPORTER=otlp needs: pip install opentelemetry-exporter-otlp-proto-http") from e
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint)))
    _provider = provider
    _tracer = provider.get_tracer(settings.service_name)
    return _tracer


def get_tracer():
    return _tracer if _tracer is not None else setup_tracing()


@contextmanager
def traced_span(name: str, **attributes):
    """`with traced_span("Retrieval", top_k=3) as span:` -- records exceptions and re-raises them."""
    with get_tracer().start_as_current_span(name) as span:
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            raise
