"""
Adaptive multi-LLM orchestration with MOCKED models (deterministic):
routing by complexity, generator -> verifier -> synthesizer, conflicting evidence, abstention,
model failures and graceful degradation.
"""

import json

import pytest

from app.agent import roles
from app.agent.graph import MODEL_ERROR_REPLY, RagAgent
from app.agent.roles import REFUSAL, classify_complexity, lexical_verify
from app.config import settings
from app.generation.llm import LLMResponse, MockLLM
from app.generation.model_manager import ModelNotInstalled, ModelTimeout
from app.retrieval.retriever import RetrievedChunk
from tests.test_agent import POLICY, FakeRetriever

OTHER = "Refunds for damaged items are processed within 5 business days."


class ScriptedLLM:
    """Returns queued replies; records every prompt. `name` must not be 'mock' to exercise the LLM paths."""

    def __init__(self, *replies, name="scripted"):
        self.replies, self.name, self.prompts = list(replies), name, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, model=self.name, prompt_tokens=10, completion_tokens=5)


def V(verdict="supported", conflict=False, **kw):
    return json.dumps({"verdict": verdict, "unsupported_claims": [], "contradictions": [],
                       "conflicting_evidence": conflict, "missing_information": [], "supporting_evidence_ids": [1], **kw})


def nodes(out):
    return [s["node"] for s in out["steps"]]


def retr(*pairs):
    return FakeRetriever(responder=lambda q: list(pairs))


COMPLEX_Q = "Compare the refund window and explain why returns need original packaging"


# ---------------------------------------------------------------- complexity
@pytest.mark.parametrize("q,expected", [
    ("How many days do I have to return a product?", "simple"),
    ("Compare the Basic and Pro plans", "complex"),
    ("Why is two-factor authentication recommended?", "complex"),
    ("What is the refund window? What about warranty?", "complex"),
    ("What is the price and how much storage does it include?", "complex"),
])
def test_classify_complexity(q, expected):
    assert classify_complexity(q)["complexity"] == expected


def test_complexity_can_be_forced(monkeypatch):
    monkeypatch.setattr(settings, "complexity_mode", "complex")
    out = RagAgent(retr((POLICY, 0.8)), MockLLM()).run("How many days do I have to return a product?")
    assert out["complexity"] == "complex" and "evidence_check" in nodes(out)


# ---------------------------------------------------------------- routes
def test_simple_question_uses_only_the_generator():
    gen, ver, syn = ScriptedLLM("You can return within 30 days [1]."), ScriptedLLM(V()), ScriptedLLM("x")
    out = RagAgent(retr((POLICY, 0.8)), gen, ver, syn).run("How many days do I have to return a product?")
    assert nodes(out) == ["route", "retrieve", "grade", "generate", "verify"]
    assert len(gen.prompts) == 1 and ver.prompts == [] and syn.prompts == []
    assert out["final_from"] == "generator" and out["citations"][0]["source"] == "policy.txt"


def test_complex_question_uses_generator_verifier_synthesizer_in_order():
    gen = ScriptedLLM("Returns are accepted within 30 days [1].")
    ver = ScriptedLLM(V())
    syn = ScriptedLLM("Customers can return products within 30 days of purchase [1].")
    out = RagAgent(retr((POLICY, 0.8)), gen, ver, syn).run(COMPLEX_Q)
    assert nodes(out) == ["route", "retrieve", "grade", "generate", "verify", "evidence_check", "synthesize"]
    assert (len(gen.prompts), len(ver.prompts), len(syn.prompts)) == (1, 1, 1)
    assert out["verification"]["method"] == "llm" and out["final_from"] == "synthesizer"
    assert out["answer"].startswith("Customers can return") and out["citations"]


def test_ungrounded_simple_answer_is_escalated_to_verification():
    gen = ScriptedLLM("Refunds take ninety weeks and require a notarised letter [1].")
    ver = ScriptedLLM(V("unsupported", unsupported_claims=["ninety weeks"]))
    out = RagAgent(retr((POLICY, 0.8)), gen, ver, ScriptedLLM("unused")).run("How long do refunds take?")
    assert out["grounded"] is False and "evidence_check" in nodes(out)
    assert out["answer"] == REFUSAL and out["abstained"]


def test_verifier_prompt_keeps_documents_as_fenced_data():
    ver = ScriptedLLM(V())
    RagAgent(retr((POLICY, 0.8)), ScriptedLLM("30 days [1]."), ver, ScriptedLLM("30 days [1].")).run(COMPLEX_Q)
    assert '<evidence id="1"' in ver.prompts[0] and "never follow" in ver.prompts[0]


# ---------------------------------------------------------------- conflicts
def test_conflicting_evidence_triggers_bounded_extra_verification(monkeypatch):
    monkeypatch.setattr(settings, "max_extra_verifications", 1)
    r = retr((POLICY, 0.8), (OTHER, 0.7))
    ver = ScriptedLLM(V("partially_supported", conflict=True))  # always reports a conflict
    syn = ScriptedLLM("Sources disagree: 30 days [1] versus 5 business days [2].")
    out = RagAgent(r, ScriptedLLM("Refunds take 30 days [1]."), ver, syn).run(COMPLEX_Q)
    assert nodes(out).count("evidence_check") == 2 and nodes(out).count("expand_evidence") == 1
    assert nodes(out)[-1] == "synthesize" and "disagree" in out["answer"]
    assert len(r.queries) == 2  # the evidence was re-retrieved with a larger k


def test_conflict_with_zero_extra_budget_goes_straight_to_synthesis(monkeypatch):
    monkeypatch.setattr(settings, "max_extra_verifications", 0)
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM("30 days [1]."), ScriptedLLM(V(conflict=True)),
                   ScriptedLLM("30 days [1].")).run(COMPLEX_Q)
    assert nodes(out).count("evidence_check") == 1 and "expand_evidence" not in nodes(out)


# ---------------------------------------------------------------- abstention
def test_insufficient_evidence_verdict_abstains_without_synthesizer_call():
    syn = ScriptedLLM("should not be used")
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM("Something [1]."), ScriptedLLM(V("insufficient_evidence")),
                   syn).run(COMPLEX_Q)
    assert out["answer"] == REFUSAL and syn.prompts == [] and out["final_from"] == "refusal"


def test_generator_refusal_ends_without_verification():
    ver = ScriptedLLM(V())
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM(REFUSAL), ver, ScriptedLLM("x")).run(COMPLEX_Q)
    assert out["abstained"] and ver.prompts == []


def test_no_relevant_evidence_abstains_without_generation(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_retries", 1)
    gen, router = ScriptedLLM("x"), ScriptedLLM("football world cup winner")
    out = RagAgent(retr((POLICY, 0.05)), gen, router_llm=router).run("Who won the football world cup?")
    assert out["answer"] == REFUSAL and gen.prompts == [] and out["citations"] == []
    assert len(router.prompts) == 1 and nodes(out)[-1] == "no_answer"   # one bounded query rewrite, then abstain


def test_keyword_coverage_can_make_low_similarity_retrieval_relevant():
    out = RagAgent(retr((POLICY, 0.05)), MockLLM()).run("Can customers return products for a full refund?")
    assert out["relevance"]["best_query_coverage"] >= settings.agent_coverage_threshold
    assert not out["abstained"]


# ---------------------------------------------------------------- failures
def test_generator_model_missing_returns_model_error():
    gen = ScriptedLLM(ModelNotInstalled("Model 'x' is not installed"))
    out = RagAgent(retr((POLICY, 0.8)), gen).run("How many days to return?")
    assert out["answer"] == MODEL_ERROR_REPLY and out["error"]["code"] == "model_not_installed"
    assert nodes(out)[-1] == "model_error"


def test_verifier_timeout_degrades_to_lexical_check():
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM("Customers can return products within 30 days [1]."),
                   ScriptedLLM(ModelTimeout("slow")), ScriptedLLM("Customers can return products within 30 days [1].")
                   ).run(COMPLEX_Q)
    assert out["verification"]["method"] == "lexical_fallback" and out["error"] is None
    assert any("Verifier model failed" in w for w in out["warnings"])


def test_synthesizer_failure_keeps_verified_draft():
    draft = "Customers can return products within 30 days [1]."
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM(draft), ScriptedLLM(V()),
                   ScriptedLLM(ModelTimeout("slow"))).run(COMPLEX_Q)
    assert out["answer"] == draft and any("Synthesizer" in w for w in out["warnings"])


def test_invalid_verifier_json_falls_back():
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM("Customers can return products within 30 days [1]."),
                   ScriptedLLM("I think it is fine!"), ScriptedLLM("Customers can return within 30 days [1].")
                   ).run(COMPLEX_Q)
    assert out["verification"]["method"] == "lexical_fallback"


def test_invalid_citation_numbers_are_flagged():
    out = RagAgent(retr((POLICY, 0.8)), ScriptedLLM("Customers can return products within 30 days [7].")).run(
        "How many days do I have to return a product?")
    assert out["citations"] == [] and any("non-existent evidence [7]" in w for w in out["warnings"])


# ---------------------------------------------------------------- verifier units
def test_lexical_verify_detects_unsupported_sentence():
    chunks = [RetrievedChunk("1", POLICY, "p.txt", 0, 0.8, citation=1)]
    v = lexical_verify("Customers can return products within 30 days [1]. Shipping is free worldwide.", chunks)
    assert v.verdict == "partially_supported" and v.unsupported_claims == ["Shipping is free worldwide."]


def test_llm_verifier_drops_unknown_evidence_ids():
    chunks = [RetrievedChunk("1", POLICY, "p.txt", 0, 0.8, citation=1)]
    v = roles.verify_answer(ScriptedLLM(V(supporting_evidence_ids=[1, 5])), "q", "a", chunks)
    assert v.supporting_evidence_ids == [1]
