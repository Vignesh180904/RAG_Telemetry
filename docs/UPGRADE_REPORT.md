# Upgrade report: original RAG project → advanced version

Date: 2026-10-24. The work was done on a copy of the project (from the zip) in a Linux sandbox with
Python 3.12.3. Nothing on the original Windows machine was touched.

## 1. Audit of the original project (Phase 1)

**Architecture found:** FastAPI (`/upload`, `/query`, `/agent/query`, `/documents`, `/health`) →
sentence-transformers embeddings → ChromaDB (dense only) → one prompt → one LLM (mock / Ollama /
OpenAI / Gemini). There was also a LangGraph agent (router → retrieve → grade → rewrite → generate →
lexical verify), OpenTelemetry console spans, a JSONL request log, and an evaluation script with 5
embedding-similarity metrics over 20 questions.

**Working components kept:** loaders, the safe AST calculator tool, the LangGraph agent skeleton
(node names, retry loop, loop guard, refusal without an LLM call), the provider abstraction, and
all 27 original tests (they still pass, unmodified).

**Problems found (with evidence):**

| # | Problem | Evidence / effect |
|---|---|---|
| 1 | Evaluation could not detect wrong answers | Original `outputs/eval_report.json`: 5 of 20 answers were wrong or missing although the documents contain them (encryption, cancellation, OS list, 90% alert, refund time). Only 2 questions were flagged, and one of those was a *correct* "I don't know". `ground_truth` and `relevant_source` were never used, and every refusal scored faithfulness 1.0. |
| 2 | No hybrid retrieval | Dense vectors only; no BM25, no fusion, no reranking |
| 3 | Re-uploading duplicated chunks | Random UUID chunk IDs (already noted in the old README) |
| 4 | No page numbers | PDF pages were joined; citations could not name a page |
| 5 | `import torch` at app import time | `app/main.py` → `embeddings.py` → `sentence_transformers`: on Windows with Application Control (`WinError 4551`) the whole API failed to start, and even the agent API tests could not be collected |
| 6 | Tests wrote to the real telemetry log | Running `test_agent_api.py` added 2 lines to `outputs/telemetry_log.jsonl` |
| 7 | Noisy shutdown error | `ConsoleMetricExporter` → "I/O operation on closed file" after pytest |
| 8 | XSS in the UI | `frontend/index.html` inserted document text with `innerHTML` |
| 9 | Error responses leaked internals | `HTTPException(500, f"Internal error: {e}")` returned raw exception text |
| 10 | CORS `*` with credentials | Any website could call the local API from a browser |
| 11 | Unpinned / incompatible deps | `langgraph`, `langchain-core` unpinned; numpy 1.26.4/pandas 2.2.2 have no Python 3.13 wheels; `pandas` was listed but unused |
| 12 | chromadb 0.5.5 + posthog ≥ 6 | "Failed to send telemetry event … capture() takes 1 positional argument" |
| 13 | Raw query text logged by default | Privacy risk for real documents |

## 2. What was built (file by file)

**New**
* `app/services.py`: builds all components and records failures for `/ready` instead of crashing.
* `app/ingestion/pipeline.py`: deterministic document IDs, content hashing (`unchanged` / `duplicate` / `reindexed`), per-file errors, replacement of legacy random-ID chunks.
* `app/retrieval/bm25.py`, `fusion.py` (RRF), `reranker.py` (none / cross-encoder), `text.py` (tokenizer; keeps "not"/"no" so negations survive).
* `app/generation/model_manager.py`: Ollama availability, timeouts, bounded retries (missing models are never retried), a process-wide sequential lock, measured latency, `load_duration`, model-switch flag, and token counts.
* `app/agent/roles.py`: generator / verifier (JSON) / synthesizer prompts with fenced, untrusted evidence; citation parsing; rule-based complexity classifier; lexical fallback verifier.
* `app/evaluation/scoring.py`: label-based metrics and failure categories.
* `data/evaluation/eval_dataset_v2.json` + `data/evaluation/corpus/` (PDF with 2 pages, DOCX with a table, a conflicting price update, a prompt-injection note).
* Tests: `conftest.py`, `test_ingestion.py`, `test_embeddings.py`, `test_retrieval.py`, `test_orchestration.py`, `test_model_manager.py`, `test_telemetry.py`, `test_eval_metrics.py`, `test_api.py`, `integration/test_live_local_models.py` (opt-in).
* `requirements-embeddings-st.txt`, `requirements-embeddings-onnx.txt`, `requirements-otlp.txt`.

**Changed**
* `app/config.py`: validated settings, role models, retrieval, telemetry and eval settings. An old `.env` still works: setting `OTLP_ENDPOINT` still enables OTLP export.
* `app/ingestion/embeddings.py`: 4 backends with lazy imports, a clear WinError 4551 message, an LRU query cache, `model_id`.
* `app/ingestion/loader.py`: page-aware loading, DOCX tables, encoding fallback, file signature checks, `DocumentLoadError`. Old functions are kept.
* `app/ingestion/chunker.py`: sentence-aware cuts, deterministic IDs, page metadata. The old signature is kept.
* `app/retrieval/vector_store.py`: upsert, per-document delete, the embedding-model guard (and a workaround for chromadb 0.5.x overwriting collection metadata in `get_or_create_collection`), and Chroma usage pings disabled.
* `app/retrieval/retriever.py`: the original `Retriever` is kept as the evaluation baseline, plus the new `HybridRetriever`.
* `app/agent/graph.py`: the adaptive multi-LLM graph (adds `evidence_check`, `expand_evidence`, `synthesize`, `model_error`). The original node names and simple path are unchanged.
* `app/telemetry/*`: request and trace IDs, per-stage and per-model latency, tokens only when reported, no invented cost, query hashing, rotation, summary.
* `app/evaluation/dataset.py`, `evaluator.py`, `evaluate.py`: versioned datasets, baseline vs advanced comparison, JSON/CSV reports, an isolated eval index, and an optional, separately reported LLM judge.
* `app/main.py`: new endpoints, a consistent error shape, upload limits and sanitising, CORS limited to localhost.
* `frontend/index.html`: XSS fix; now shows citations, route and verification.
* `app/generation/llm.py`: the mock splits on line breaks too, so headings aren't glued onto answers.

**Breaking changes**
1. `POST /query` now runs the advanced pipeline. Its response is a **superset** of the old one
   (`answer`, `retrieved_chunks`, `sources` and `latency_ms` are still there). `/agent/query` is an alias.
2. Error bodies changed from `{"detail": ...}` to `{"error": {"code", "message", "request_id"}}`
   (the bundled UI handles both).
3. `evaluate.py` modes: `baseline | advanced | compare` (default `compare`); `plain`/`agent` still work
   as aliases. Reports moved to `outputs/evals/`.
4. `sentence-transformers` moved out of `requirements.txt` into `requirements-embeddings-st.txt`.
5. The telemetry JSONL has a new schema; `/telemetry/summary` skips old-format lines.

## 3. Dependency changes and reasoning

| Change | Why |
|---|---|
| `sentence-transformers` → optional file | torch is the source of WinError 4551; the API must start without it |
| `chromadb` 0.5.5 → **0.5.4** | 0.5.5 pins `chroma-hnswlib==0.7.6`, which has no Windows wheels, so pip tries to compile it and fails without Microsoft C++ Build Tools. 0.5.4 uses 0.7.5, which has a cp312 `win_amd64` wheel. The full requirements set was resolved for Windows/Python 3.12 with wheels only (`uv pip compile --python-platform x86_64-pc-windows-msvc --only-binary :all:`) |
| `posthog>=2.4,<6` added | chromadb 0.5.5 is incompatible with posthog ≥ 6 (verified) |
| `langchain-core>=0.3,<2`, `langgraph>=0.2,<2` | Upper bounds; tested with 1.7.0 / 1.2.17 |
| `pandas` removed | Not imported anywhere (the CSV is written with the stdlib) |
| `fastembed` (optional) | Torch-free embeddings and reranker via ONNX Runtime |
| `opentelemetry-exporter-otlp-proto-http` (optional) | Only for `TRACE_EXPORTER=otlp` |
| No BM25 package | ~60-line implementation, fully unit-tested, one dependency fewer |

## 4. Verification status (honest)

| Item | Status |
|---|---|
| 149 offline tests (mocked models, hash embeddings, real ChromaDB in temp folders) | **Passed** in two venvs, including a fresh Python 3.12 venv built only from `requirements.txt` (no torch); `pip check` clean |
| Original 27 tests | **Passed, unmodified** |
| FastAPI server startup + all endpoints via HTTP (uvicorn) | **Passed** (mock LLM, hash embeddings) |
| Ollama HTTP code paths (tags / chat / embed, 3 roles on 2 models, switching, tokens) | **Passed against a fake Ollama HTTP server.** Real Ollama was not available here |
| Real embedding models (sentence-transformers, fastembed, Ollama all-minilm) | **Not verified here**: the sandbox cannot reach HuggingFace and has no Ollama. Run the smoke tests in README §2 |
| Real LLM answer quality, latency and RAM use on 8 GB | **Not measured.** Run `python evaluate.py` and the live tests with your models |
| Cross-encoder reranker with a real model | **Not verified** (code path tested with a fake reranker) |
| Windows-specific behaviour (paths, PowerShell, App Control) | **Not run on Windows.** The code uses `os.path`/`os.replace` and has no POSIX-only calls, but please run the test suite on your machine |

**Sample evaluation (mock LLM + hash embeddings, so it tests the pipeline, not quality):**

| metric | baseline | advanced |
|---|---|---|
| accuracy (answerable) | 0.654 | 0.692 |
| context precision | 0.907 | 0.958 |
| context recall | 0.885 | 0.962 |
| citation correctness | 0.000 | 1.000 |
| abstention on unanswerable | 0.500 | 0.500 |
| false-abstention rate | 0.038 | 0.077 |

The remaining errors are mostly the mock's limits: it can only copy one sentence, so it can't
combine facts or resolve conflicts. Real numbers will come from your Ollama run.

## 5. Recommended next steps on your machine
1. `python -m pytest -q`: expect 149 passed, 5 skipped.
2. Pick an embedding backend with the README §2 smoke test; set `EMBEDDING_BACKEND`.
3. `ollama list` → set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL` (and optionally the role models).
4. `Invoke-RestMethod http://127.0.0.1:8000/ready` must be `ready: true`.
5. `$env:RUN_LIVE_TESTS="1"; python -m pytest tests/integration -v`
6. `python evaluate.py -v`, then tune `AGENT_RELEVANCE_THRESHOLD` for your embedding model using the
   `false_abstention` / `missed_abstention` counts.
