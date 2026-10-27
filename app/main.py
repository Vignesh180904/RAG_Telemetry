"""
FastAPI application
===================
    GET    /                     simple UI (frontend/index.html)
    GET    /health               liveness: the process is up (never touches Ollama/Chroma)
    GET    /ready                readiness: embeddings, vector store, BM25 index, Ollama + role models
    GET    /models               configured roles -> models, installed models, model switching note
    POST   /documents/upload     ingest PDF/DOCX/TXT (also at legacy POST /upload)
    GET    /documents            indexed documents
    DELETE /documents/{id}       remove a document's chunks from the index (file kept unless delete_file=true)
    POST   /query                advanced RAG (hybrid retrieval + adaptive multi-LLM workflow)
    POST   /agent/query          legacy alias of /query
    POST   /evals/run            run the evaluation (isolated index; one run at a time)
    GET    /evals/results        latest evaluation summary
    GET    /telemetry/summary    aggregates over recent requests

Errors always look like {"error": {"code", "message", "request_id"}}; internal paths and stack
traces are logged server-side only.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.ingestion.loader import SUPPORTED_EXTENSIONS
from app.services import Services
from app.telemetry.metrics import build_record, record_request, summarize
from app.telemetry.tracing import configure_logging, current_trace_id, new_request_id, request_id_var, setup_tracing, traced_span

log = logging.getLogger("app.api")

# Module-level state kept for backward compatibility (tests inject state["agent"]).
state: Dict[str, Any] = {}
_ingest_lock = threading.Lock()
_eval_lock = threading.Lock()


def services() -> Optional[Services]:
    return state.get("services")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    setup_tracing()
    sv = Services(settings).init()
    state["services"] = sv
    if sv.agent is not None:
        state["agent"] = sv.agent
    os.makedirs(settings.documents_dir, exist_ok=True)
    log.info("startup complete; errors=%s warnings=%s", sv.errors, sv.warnings)
    yield


app = FastAPI(title="Advanced RAG: hybrid retrieval, multi-LLM orchestration, telemetry, evals", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


# ---------------------------------------------------------------- errors & request ids
class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "request_id": request_id_var.get()}})


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id", "")
    rid = rid if re.fullmatch(r"[A-Za-z0-9_-]{6,64}", rid or "") else new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(APIError)
async def _api_error(_: Request, e: APIError):
    return _err(e.status, e.code, e.message)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, e: StarletteHTTPException):
    return _err(e.status_code, "http_error", str(e.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, e: RequestValidationError):
    msgs = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:])}: {err['msg']}" for err in e.errors())
    return _err(422, "invalid_request", msgs[:500])


@app.exception_handler(Exception)
async def _unhandled(_: Request, e: Exception):
    log.exception("unhandled error")
    return _err(500, "internal_error", "Internal server error. See server logs with this request_id.")


# ---------------------------------------------------------------- schemas
class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    source_filter: Optional[str] = Field(None, max_length=255, description="only search this document (file name)")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class RetrievedChunkResponse(BaseModel):
    citation: int = 0
    source: str
    page: Optional[int] = None
    chunk_index: int
    score: float
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    found_by: List[str] = []
    text: str


class Citation(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    chunk_index: int


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # allows the field name 'model_calls'

    request_id: str
    trace_id: Optional[str]
    answer: str
    abstained: bool
    citations: List[Citation]
    route: str
    complexity: Optional[str]
    complexity_reasons: List[str]
    final_from: Optional[str]
    query_used: str
    rewritten: bool
    grounded: Optional[bool]
    grounding_score: Optional[float]
    verification: Optional[Dict[str, Any]]
    tool_used: Optional[str]
    retrieved_chunks: List[RetrievedChunkResponse]
    sources: List[str]
    steps: List[Dict[str, Any]]
    model_calls: List[Dict[str, Any]]
    warnings: List[str]
    latency_ms: Dict[str, Optional[float]]


class EvalRunRequest(BaseModel):
    mode: Literal["baseline", "advanced", "compare"] = "compare"
    dataset: Optional[str] = Field(None, description="file name inside data/evaluation/, e.g. eval_dataset_v2.json")


# ---------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def read_root():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - services().started_at, 1) if services() else None}


@app.get("/ready")
def ready():
    sv = services()
    if sv is None:
        return _err(503, "not_initialized", "Services not initialised.")
    r = sv.readiness()
    return JSONResponse(status_code=200 if r["ready"] else 503, content=r)


@app.get("/models")
def models():
    sv = services()
    if settings.llm_provider != "ollama":
        return {"provider": settings.llm_provider, "roles": {r: settings.llm_provider for r in
                                                             ("generator", "verifier", "synthesizer", "router")},
                "note": "Set LLM_PROVIDER=ollama and the *_MODEL variables to use local models."}
    return {"provider": "ollama", "execution": "sequential (one generation at a time)",
            "keep_alive": settings.llm_keep_alive, "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict, **sv.manager.status(settings.role_models())}


def _safe_filename(name: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:150]


@app.post("/documents/upload")
@app.post("/upload", include_in_schema=False)
def upload_document(file: UploadFile = File(...)):
    sv = services()
    if sv is None or sv.ingestion is None:
        raise APIError(503, "not_ready", "Ingestion unavailable; check GET /ready.")
    name = _safe_filename(file.filename)
    ext = os.path.splitext(name)[1].lower()
    if not name or ext not in SUPPORTED_EXTENSIONS:
        raise APIError(400, "unsupported_file_type", f"Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    limit = settings.max_upload_mb * 1024 * 1024
    os.makedirs(settings.documents_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=ext, dir=settings.documents_dir)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise APIError(413, "file_too_large", f"Maximum upload size is {settings.max_upload_mb} MB.")
                out.write(chunk)
        with _ingest_lock, traced_span("Document Upload", file_ext=ext, bytes=size):
            res = sv.ingestion.ingest_file(tmp, source=name)
            if res.status == "error":
                raise APIError(400, "unreadable_document", res.message)
            if res.status in ("indexed", "reindexed", "unchanged"):
                os.replace(tmp, os.path.join(settings.documents_dir, name))  # keep the file (replaces an older upload)
                tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    body = res.to_dict()
    body.update({"filename": name, "chunks_created": res.chunks, "total_chunks_in_store": sv.store.count()})
    return body


@app.get("/documents")
def list_documents():
    sv = services()
    if sv is None or sv.store is None:
        raise APIError(503, "not_ready", "Vector store unavailable; check GET /ready.")
    docs = sv.store.list_documents()
    return {"total_chunks": sv.store.count(), "documents": {d["source"]: d["chunks"] for d in docs},  # legacy shape
            "items": docs}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, delete_file: bool = Query(False)):
    sv = services()
    if sv is None or sv.ingestion is None:
        raise APIError(503, "not_ready", "Ingestion unavailable; check GET /ready.")
    if not re.fullmatch(r"[0-9a-f]{16}", document_id):
        raise APIError(400, "invalid_document_id", "Use a document_id from GET /documents.")
    match = next((d for d in sv.store.list_documents() if d["document_id"] == document_id), None)
    if not match:
        raise APIError(404, "document_not_found", "No indexed document with this id.")
    with _ingest_lock:
        n = sv.ingestion.delete(document_id)
        removed_file = False
        if delete_file:
            path = os.path.join(settings.documents_dir, _safe_filename(match["source"]))
            if os.path.isfile(path):
                os.remove(path)
                removed_file = True
    return {"document_id": document_id, "source": match["source"], "chunks_deleted": n, "file_deleted": removed_file}


def _run_query(req: QueryRequest, endpoint: str) -> QueryResponse:
    if not req.question:
        raise APIError(400, "empty_question", "Question must not be empty.")
    agent = state.get("agent")
    if agent is None:
        raise APIError(503, "not_ready", "Query pipeline unavailable; check GET /ready.")
    sv = services()
    emb = sv.embedder.model_id if sv and sv.embedder else settings.embedding_model
    rid = request_id_var.get()
    t0 = time.perf_counter()
    with traced_span("RAG Request", endpoint=endpoint, question_chars=len(req.question)) as span:
        trace_id = current_trace_id()
        try:
            result = agent.run(req.question, top_k=req.top_k, source_filter=req.source_filter)
        except Exception as e:
            total = (time.perf_counter() - t0) * 1000
            log.exception("query failed")
            record_request(build_record(rid, endpoint, req.question, None, total, trace_id, emb,
                                        error_code=type(e).__name__, error_message=str(e)))
            raise APIError(500, "query_failed", "The query failed. See server logs with this request_id.")
        span.set_attribute("rag.route", result["route"])
        span.set_attribute("rag.final_from", result.get("final_from") or "")
    total = (time.perf_counter() - t0) * 1000
    rec = build_record(rid, endpoint, req.question, result, total, trace_id, emb)
    record_request(rec)
    if result.get("error"):
        err = result["error"]
        raise APIError(503, err.get("code", "model_error"), f"{err.get('role', 'model')}: {err.get('message', '')}")
    chunks = result["chunks"]
    return QueryResponse(
        request_id=rid, trace_id=trace_id, answer=result["answer"], abstained=result["abstained"],
        citations=[Citation(**c) for c in result["citations"]], route=result["route"],
        complexity=result.get("complexity"), complexity_reasons=result.get("complexity_reasons", []),
        final_from=result.get("final_from"), query_used=result["query_used"], rewritten=result["rewritten"],
        grounded=result["grounded"], grounding_score=result["grounding_score"], verification=result["verification"],
        tool_used=result["tool_used"],
        retrieved_chunks=[RetrievedChunkResponse(**{k: c.get(k) for k in RetrievedChunkResponse.model_fields if k in c})
                          for c in chunks],
        sources=sorted({c["source"] for c in chunks}),
        steps=result["steps"], model_calls=result["model_calls"], warnings=result["warnings"],
        latency_ms={"retrieval_ms": rec.retrieval_ms, "embedding_ms": rec.embedding_ms, "rerank_ms": rec.rerank_ms,
                    "llm_ms": rec.llm_ms, "total_ms": rec.total_ms},
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return _run_query(req, "/query")


@app.post("/agent/query", response_model=QueryResponse)
def agent_query(req: QueryRequest):
    return _run_query(req, "/agent/query")


@app.post("/evals/run")
def evals_run(req: EvalRunRequest):
    from app.evaluation.evaluator import run_evaluation

    ds_path = None
    if req.dataset:
        name = _safe_filename(req.dataset)
        ds_path = os.path.join("data", "evaluation", name)
        if not name.endswith(".json") or not os.path.isfile(ds_path):
            raise APIError(400, "unknown_dataset", "Dataset must be a .json file inside data/evaluation/.")
    if not _eval_lock.acquire(blocking=False):
        raise APIError(409, "eval_in_progress", "An evaluation is already running.")
    try:
        out = run_evaluation(settings, mode=req.mode, dataset_path=ds_path)
    except RuntimeError as e:
        raise APIError(503, "eval_unavailable", str(e)[:300])
    finally:
        _eval_lock.release()
    latest = out["latest"]
    return {"mode": latest["mode"], "dataset": latest["dataset"], "summary": latest["summary"],
            "reports": {m: {k: os.path.basename(v) if v else None for k, v in p.items()} for m, p in latest["paths"].items()}}


@app.get("/evals/results")
def evals_results():
    import json

    path = os.path.join(settings.eval_output_dir, "evals", "latest.json")
    if not os.path.exists(path):
        raise APIError(404, "no_results", "No evaluation has been run yet. POST /evals/run or run evaluate.py.")
    with open(path, encoding="utf-8") as f:
        latest = json.load(f)
    latest["paths"] = {m: {k: os.path.basename(v) if v else None for k, v in p.items()} for m, p in latest["paths"].items()}
    return latest


@app.get("/telemetry/summary")
def telemetry_summary(last_n: int = Query(1000, ge=1, le=100000)):
    return summarize(last_n=last_n)
