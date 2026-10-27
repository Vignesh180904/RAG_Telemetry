"""
Central configuration
=====================
Every tunable value lives here and is read from environment variables / `.env`.
Values are validated at startup (pydantic), so a typo such as CHUNK_OVERLAP >= CHUNK_SIZE
fails fast with a clear message instead of misbehaving later.

Import the single instance everywhere:  `from app.config import settings`
"""

from typing import List, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------ embeddings
    # sentence-transformers : original backend (needs torch; may hit WinError 4551 on locked-down Windows)
    # ollama                : uses Ollama's /api/embed (e.g. `ollama pull all-minilm`) -- no torch needed
    # fastembed             : ONNX runtime, no torch (pip install fastembed)
    # hash                  : deterministic, dependency-free, LOW quality -- for tests/offline demos only
    embedding_backend: Literal["sentence-transformers", "ollama", "fastembed", "hash"] = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(32, ge=1, le=512)
    embedding_query_cache_size: int = Field(512, ge=0)

    # ------------------------------------------------------------------ ingestion
    chunk_size: int = Field(500, ge=50, le=8000)
    chunk_overlap: int = Field(50, ge=0)
    max_upload_mb: int = Field(20, ge=1, le=500)
    documents_dir: str = "./data/documents"

    # ------------------------------------------------------------------ retrieval
    top_k: int = Field(3, ge=1, le=50)
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    dense_candidates: int = Field(10, ge=1, le=200)     # how many dense hits enter fusion
    sparse_candidates: int = Field(10, ge=1, le=200)    # how many BM25 hits enter fusion
    rrf_k: int = Field(60, ge=1)                        # standard RRF constant
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    min_dense_score: float = 0.0                        # drop dense hits below this before fusion
    dedupe_threshold: float = Field(0.9, ge=0.0, le=1.0)  # token-Jaccard above this = duplicate chunk
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------------ vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # ------------------------------------------------------------------ LLM provider
    # mock   : deterministic, no model needed (tests / demos)
    # ollama : local models (recommended)
    # openai / gemini : optional legacy providers kept from the original project (never required)
    llm_provider: Literal["mock", "ollama", "openai", "gemini"] = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"            # default for any role not set below

    # Roles -- leave blank to reuse OLLAMA_MODEL. The same model may serve several roles.
    generator_model: str = ""
    verifier_model: str = ""
    synthesizer_model: str = ""
    router_model: str = ""                        # only used when AGENT_ROUTER=llm / query rewriting

    llm_timeout_s: float = Field(120.0, gt=0)
    llm_max_retries: int = Field(1, ge=0, le=5)   # retries AFTER the first attempt
    llm_retry_backoff_s: float = Field(1.0, ge=0)
    llm_num_ctx: int = Field(4096, ge=512)        # Ollama context window (memory grows with this)
    llm_num_predict: int = Field(512, ge=16)      # max tokens generated per call
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    # How long Ollama keeps a model in RAM after a call. "0" = unload immediately (lowest RAM,
    # slower when roles alternate between different models). Ollama decides when it actually frees memory.
    llm_keep_alive: str = "5m"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # ------------------------------------------------------------------ agent / routing
    agent_relevance_threshold: float = 0.30       # best dense similarity needed to call retrieval "relevant"
    agent_max_retries: int = Field(1, ge=0, le=5) # query rewrites on weak retrieval
    agent_router: Literal["rules", "llm"] = "rules"
    agent_grounding_threshold: float = 0.5        # lexical share of answer words found in evidence
    # Retrieval also counts as relevant when a chunk contains this share of the query's content words.
    # Embedding-independent, so the gate still works if similarity scores shift after a backend change.
    agent_coverage_threshold: float = Field(0.5, ge=0.0, le=1.0)
    complexity_mode: Literal["auto", "simple", "complex"] = "auto"  # force a route for experiments
    max_extra_verifications: int = Field(1, ge=0, le=3)  # extra verifier passes on conflicting evidence

    # ------------------------------------------------------------------ telemetry
    service_name: str = "rag-telemetry-evals"
    log_level: str = "INFO"
    trace_exporter: Literal["none", "console", "otlp"] = "none"
    otlp_endpoint: str = ""                       # e.g. http://localhost:4318/v1/traces (Jaeger / Langfuse / collector)
    metrics_console_export: bool = False          # print OTel metrics to console every 60 s
    telemetry_log_path: str = "./outputs/telemetry_log.jsonl"
    telemetry_log_query_text: bool = False        # False = store only a hash + length of each query (privacy)

    # ------------------------------------------------------------------ evaluation
    eval_dataset_path: str = "./data/evaluation/eval_dataset_v2.json"
    eval_output_dir: str = "./outputs"
    eval_llm_judge: bool = False                  # optional, uses the verifier model; reported separately

    # ------------------------------------------------------------------ API
    cors_origins: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        # Backward compatibility: the original project exported to OTLP whenever OTLP_ENDPOINT was set.
        if self.otlp_endpoint and self.trace_exporter == "none":
            self.trace_exporter = "otlp"
        if self.trace_exporter == "otlp" and not self.otlp_endpoint:
            raise ValueError("TRACE_EXPORTER=otlp requires OTLP_ENDPOINT")
        return self

    # ---- role helpers
    def model_for(self, role: str) -> str:
        explicit = {
            "generator": self.generator_model,
            "verifier": self.verifier_model,
            "synthesizer": self.synthesizer_model,
            "router": self.router_model,
        }.get(role, "")
        return explicit or self.ollama_model

    def role_models(self) -> dict:
        return {r: self.model_for(r) for r in ("generator", "verifier", "synthesizer", "router")}


settings = Settings()
