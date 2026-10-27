"""
Ollama model manager
====================
One place that talks to Ollama for chat models:

* model availability (`/api/tags`, cached for 30 s) and clear errors for missing models
* per-call timeout, bounded retries with backoff (never retries a missing model)
* conservative generation limits (num_ctx, num_predict, temperature) and keep_alive
* SEQUENTIAL execution: a process-wide lock guarantees only one generation runs at a time, so
  an 8 GB machine never has two requests competing for RAM. Ollama itself decides when a model
  is unloaded (keep_alive is a request, not a guarantee) -- check with `ollama ps`.
* measured latency, Ollama's own load_duration (model-switch cost) and token counts
  (prompt_eval_count / eval_count) -- reported as None when Ollama does not return them.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

from app.generation.llm import BaseLLM, LLMResponse

log = logging.getLogger(__name__)
_GENERATION_LOCK = threading.Lock()


class ModelError(RuntimeError):
    code = "model_error"


class OllamaUnavailable(ModelError):
    code = "ollama_unavailable"


class ModelNotInstalled(ModelError):
    code = "model_not_installed"


class ModelTimeout(ModelError):
    code = "model_timeout"


@dataclass
class ModelCall:
    role: str
    model: str
    latency_ms: float = 0.0
    load_ms: Optional[float] = None          # Ollama load_duration: >0 usually means the model was (re)loaded
    switched_model: bool = False             # previous call in this process used a different model
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OllamaModelManager:
    base_url: str = "http://localhost:11434"
    timeout_s: float = 120.0
    max_retries: int = 1
    backoff_s: float = 1.0
    num_ctx: int = 4096
    num_predict: int = 512
    temperature: float = 0.1
    keep_alive: str = "5m"
    _tags_cache: Optional[List[str]] = None
    _tags_at: float = 0.0
    _last_model: Optional[str] = None
    session: requests.Session = field(default_factory=requests.Session)

    # ------------------------------------------------------------------ availability
    def list_models(self, refresh: bool = False) -> List[str]:
        if not refresh and self._tags_cache is not None and time.time() - self._tags_at < 30:
            return self._tags_cache
        try:
            r = self.session.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaUnavailable(f"Ollama is not reachable at {self.base_url} ({type(e).__name__}).") from e
        self._tags_cache = [m.get("name", "") for m in r.json().get("models", [])]
        self._tags_at = time.time()
        return self._tags_cache

    def is_installed(self, model: str) -> bool:
        names = self.list_models()
        return model in names or f"{model}:latest" in names

    def status(self, role_models: Dict[str, str]) -> Dict[str, Any]:
        try:
            installed = self.list_models(refresh=True)
            ok = True
        except OllamaUnavailable as e:
            return {"ollama_reachable": False, "error": str(e), "roles": {
                r: {"model": m, "installed": None} for r, m in role_models.items()}}
        roles = {r: {"model": m, "installed": (m in installed or f"{m}:latest" in installed)}
                 for r, m in role_models.items()}
        distinct = sorted({m for m in role_models.values()})
        return {"ollama_reachable": ok, "installed_models": installed, "roles": roles,
                "distinct_models": distinct,
                "note": ("All roles share one physical model." if len(distinct) == 1 else
                         f"{len(distinct)} different models; they are run sequentially, never concurrently.")}

    # ------------------------------------------------------------------ generation
    def chat(self, role: str, model: str, system: str, user: str, json_mode: bool = False,
             calls: Optional[List[ModelCall]] = None) -> LLMResponse:
        call = ModelCall(role=role, model=model)
        if calls is not None:
            calls.append(call)
        try:
            if not self.is_installed(model):
                raise ModelNotInstalled(f"Model '{model}' (role: {role}) is not installed. Run: ollama pull {model}")
        except ModelError as e:
            call.error, call.error_code = str(e), e.code
            raise

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, "num_predict": self.num_predict, "temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        last_exc: Optional[ModelError] = None
        t_start = time.perf_counter()
        with _GENERATION_LOCK:  # sequential execution across the whole process
            call.switched_model = self._last_model is not None and self._last_model != model
            for attempt in range(1, self.max_retries + 2):
                call.attempts = attempt
                try:
                    r = self.session.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload, timeout=self.timeout_s)
                    if r.status_code == 404:
                        raise ModelNotInstalled(f"Ollama reports model '{model}' not found.")
                    r.raise_for_status()
                    data = r.json()
                    self._last_model = model
                    call.latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                    call.prompt_tokens = data.get("prompt_eval_count")
                    call.completion_tokens = data.get("eval_count")
                    if data.get("load_duration") is not None:
                        call.load_ms = round(data["load_duration"] / 1e6, 2)
                    text = (data.get("message") or {}).get("content", "").strip()
                    return LLMResponse(text=text, prompt_tokens=call.prompt_tokens,
                                       completion_tokens=call.completion_tokens, model=model)
                except ModelNotInstalled as e:
                    last_exc = e
                    break  # retrying cannot help
                except requests.Timeout:
                    last_exc = ModelTimeout(f"Model '{model}' did not answer within {self.timeout_s:.0f}s.")
                except requests.ConnectionError:
                    last_exc = OllamaUnavailable(f"Lost connection to Ollama at {self.base_url}.")
                except (requests.RequestException, ValueError) as e:
                    last_exc = ModelError(f"Ollama call failed ({type(e).__name__}).")
                if attempt <= self.max_retries:
                    log.warning("model call failed (role=%s model=%s attempt=%d): %s", role, model, attempt, last_exc)
                    time.sleep(self.backoff_s * attempt)
        call.latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        call.error, call.error_code = str(last_exc), last_exc.code
        raise last_exc


class OllamaRoleLLM(BaseLLM):
    """A BaseLLM bound to one role/model, backed by the shared manager (so the lock is shared)."""

    name = "ollama"

    def __init__(self, manager: OllamaModelManager, role: str, model: str):
        self.manager, self.role, self.model = manager, role, model

    def chat(self, system: str, user: str, json_mode: bool = False,
             calls: Optional[List[ModelCall]] = None) -> LLMResponse:
        return self.manager.chat(self.role, self.model, system, user, json_mode=json_mode, calls=calls)

    def generate(self, prompt: str) -> LLMResponse:
        return self.chat("You are a helpful assistant.", prompt)
