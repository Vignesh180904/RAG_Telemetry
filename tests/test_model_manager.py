"""Ollama model manager against a fake HTTP session (no Ollama needed)."""

import pytest
import requests

from app.generation.model_manager import (ModelNotInstalled, ModelTimeout, OllamaModelManager, OllamaRoleLLM,
                                          OllamaUnavailable)


class Resp:
    def __init__(self, data=None, status=200):
        self.data, self.status_code = data or {}, status

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, models=("llama3.2:3b",), post=None):
        self.models, self.post_script, self.posts = models, list(post or []), []

    def get(self, url, timeout=None):
        if self.models is None:
            raise requests.ConnectionError("refused")
        return Resp({"models": [{"name": m} for m in self.models]})

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        item = self.post_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


OK = Resp({"message": {"content": "hello"}, "prompt_eval_count": 12, "eval_count": 3, "load_duration": 2_500_000})


def mgr(session, **kw):
    return OllamaModelManager(session=session, backoff_s=0, **kw)


def test_successful_call_records_tokens_latency_and_load_time():
    s = FakeSession(post=[OK])
    calls = []
    r = mgr(s).chat("generator", "llama3.2:3b", "sys", "user", calls=calls)
    assert r.text == "hello" and r.prompt_tokens == 12 and r.completion_tokens == 3
    c = calls[0]
    assert c.attempts == 1 and c.load_ms == 2.5 and c.latency_ms >= 0 and c.error is None


def test_options_and_keep_alive_are_sent():
    s = FakeSession(post=[OK])
    mgr(s, num_ctx=2048, num_predict=128, temperature=0.0, keep_alive="0").chat("verifier", "llama3.2:3b", "s", "u",
                                                                               json_mode=True)
    p = s.posts[0]
    assert p["options"] == {"num_ctx": 2048, "num_predict": 128, "temperature": 0.0}
    assert p["keep_alive"] == "0" and p["format"] == "json" and p["stream"] is False
    assert p["messages"][0]["role"] == "system" and p["messages"][1]["role"] == "user"


def test_missing_model_fails_fast_without_calling_generate():
    s = FakeSession(models=("other:1b",))
    calls = []
    with pytest.raises(ModelNotInstalled, match="ollama pull llama3.2:3b"):
        mgr(s).chat("generator", "llama3.2:3b", "s", "u", calls=calls)
    assert s.posts == [] and calls[0].error_code == "model_not_installed"


def test_timeout_is_retried_then_reported():
    s = FakeSession(post=[requests.Timeout(), requests.Timeout()])
    calls = []
    with pytest.raises(ModelTimeout):
        mgr(s, max_retries=1).chat("generator", "llama3.2:3b", "s", "u", calls=calls)
    assert len(s.posts) == 2 and calls[0].attempts == 2 and calls[0].error_code == "model_timeout"


def test_transient_error_then_success():
    s = FakeSession(post=[Resp(status=500), OK])
    calls = []
    assert mgr(s, max_retries=2).chat("generator", "llama3.2:3b", "s", "u", calls=calls).text == "hello"
    assert calls[0].attempts == 2


def test_ollama_unavailable():
    with pytest.raises(OllamaUnavailable):
        mgr(FakeSession(models=None)).list_models()


def test_404_from_chat_is_not_retried():
    s = FakeSession(post=[Resp(status=404)])
    with pytest.raises(ModelNotInstalled):
        mgr(s, max_retries=3).chat("generator", "llama3.2:3b", "s", "u")
    assert len(s.posts) == 1


def test_model_switch_is_detected():
    s = FakeSession(models=("a:1b", "b:1b"), post=[OK, OK, OK])
    m, calls = mgr(s), []
    m.chat("generator", "a:1b", "s", "u", calls=calls)
    m.chat("verifier", "b:1b", "s", "u", calls=calls)
    m.chat("synthesizer", "b:1b", "s", "u", calls=calls)
    assert [c.switched_model for c in calls] == [False, True, False]


def test_status_reports_roles_and_shared_model():
    st = mgr(FakeSession(models=("llama3.2:3b",))).status({"generator": "llama3.2:3b", "verifier": "llama3.2:3b",
                                                           "synthesizer": "qwen2.5:3b"})
    assert st["roles"]["generator"]["installed"] and not st["roles"]["synthesizer"]["installed"]
    assert len(st["distinct_models"]) == 2


def test_status_when_ollama_down():
    st = mgr(FakeSession(models=None)).status({"generator": "x"})
    assert st["ollama_reachable"] is False and st["roles"]["generator"]["installed"] is None


def test_role_llm_binds_role_and_model():
    s = FakeSession(post=[OK])
    calls = []
    OllamaRoleLLM(mgr(s), "verifier", "llama3.2:3b").chat("s", "u", calls=calls)
    assert calls[0].role == "verifier" and s.posts[0]["model"] == "llama3.2:3b"
