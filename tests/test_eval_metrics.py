"""Evaluation metrics, failure categories, dataset loading and the runner (mock + hash: pipeline test only)."""

import json

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.evaluator import EvalRunner, compare, write_reports
from app.evaluation.scoring import (citation_correctness, context_precision, context_recall, faithfulness,
                                    is_abstention, key_fact_recall, score_example, token_f1)

CH = [{"citation": 1, "source": "a.txt", "page": None, "text": "Refunds are processed within 5 to 7 business days."},
      {"citation": 2, "source": "b.txt", "page": None, "text": "The office is in Chennai."}]
EX = {"id": "q", "question": "How long do refunds take?", "expected_answer": "Refunds take 5 to 7 business days.",
      "expected_sources": ["a.txt"], "key_facts": ["5 to 7 business days"], "answerable": True, "category": "factoid"}


def test_abstention_detection():
    assert is_abstention("I don't know based on the provided documents.")
    assert is_abstention("This is not mentioned in the provided documents.")
    assert is_abstention("")
    assert not is_abstention("Refunds take 5 days.")


def test_key_facts_and_negation():
    assert key_fact_recall(["5 to 7 business days"], "processed in 5 to 7 business days") == 1.0
    assert key_fact_recall(["does not cover accidental damage"], "The warranty does cover accidental damage") == 0.0
    assert key_fact_recall([], "x") is None


def test_token_f1():
    assert token_f1("refunds take 5 days", "refunds take 5 days") == 1.0
    assert token_f1("totally different", "refunds take 5 days") == 0.0


def test_context_precision_is_rank_aware():
    rel_first = context_precision(CH, ["a.txt"])
    rel_second = context_precision(list(reversed(CH)), ["a.txt"])
    assert rel_first == 1.0 and rel_second == 0.5
    assert context_precision([], ["a.txt"]) == 0.0 and context_precision(CH, []) is None


def test_page_aware_relevance():
    chunks = [{"source": "g.pdf", "page": 1, "text": ""}, {"source": "g.pdf", "page": 2, "text": ""}]
    assert context_precision(chunks, ["g.pdf"], pages=[2]) == 0.5


def test_context_recall_from_key_facts():
    assert context_recall(CH, ["5 to 7 business days", "Chennai"]) == 1.0
    assert context_recall(CH, ["AES-256"]) == 0.0


def test_faithfulness_na_for_abstention_and_low_for_invention():
    assert faithfulness("I don't know based on the provided documents.", CH) is None
    assert faithfulness("Refunds are processed within 5 to 7 business days [1].", CH) == 1.0
    assert faithfulness("Refunds require a notarised letter from Mars.", CH) == 0.0


def test_citation_correctness():
    assert citation_correctness("5 to 7 days [1].", CH, ["a.txt"])["citation_correct"] is True
    assert citation_correctness("5 to 7 days [2].", CH, ["a.txt"])["citation_correct"] is False
    bad = citation_correctness("5 to 7 days [9].", CH, ["a.txt"])
    assert bad["citation_correct"] is False and bad["invalid_citations"] == [9]
    assert citation_correctness("5 to 7 days.", CH, ["a.txt"])["citation_correct"] is False   # uncited
    assert citation_correctness("5 to 7 days (Source 1).", CH, ["a.txt"])["citation_correct"] is True


def test_score_correct_answer():
    s = score_example(EX, "Refunds are processed within 5 to 7 business days [1].", CH)
    assert s["correct"] and s["failure_categories"] == []


def test_failure_categories():
    assert score_example(EX, "I don't know based on the provided documents.", CH)["failure_categories"] == ["false_abstention"]
    assert "wrong_or_incomplete_answer" in score_example(EX, "Refunds take 30 days [1].", CH)["failure_categories"]
    assert "retrieval_miss" in score_example(EX, "Refunds take 5 to 7 business days.", CH[1:])["failure_categories"]
    unans = {**EX, "answerable": False, "key_facts": [], "expected_sources": []}
    assert score_example(unans, "Paris.", CH)["failure_categories"] == ["missed_abstention"]
    assert score_example(unans, "I don't know based on the provided documents.", CH)["correct"] is True
    assert score_example(EX, "", [], error={"code": "x"})["failure_categories"] == ["error"]


def test_injection_check():
    ex = {**EX, "forbidden_strings": ["HACKED"]}
    s = score_example(ex, "HACKED", CH)
    assert s["injection_resisted"] is False and "prompt_injection_followed" in s["failure_categories"]


def test_load_v2_and_legacy_v1():
    v2 = load_dataset("data/evaluation/eval_dataset_v2.json")
    assert v2.version == "2.0.0" and len(v2) == 30 and sum(not e["answerable"] for e in v2.examples) == 4
    v1 = load_dataset("data/evaluation/eval_dataset.json")
    assert v1.version == "1.x-legacy" and len(v1) == 20 and v1.examples[-1]["answerable"] is False


def test_dataset_validation(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": "1", "examples": [{"id": "a"}]}))
    with pytest.raises(ValueError):
        load_dataset(str(p))


def test_runner_end_to_end_with_mock(services, tmp_path, test_settings):
    """MOCKED models + hash embeddings: checks the runner, reports and comparison -- NOT answer quality."""
    ds = load_dataset("data/evaluation/eval_dataset_v2.json")
    ds.examples = ds.examples[:4] + [e for e in ds.examples if not e["answerable"]][:1]
    runner = EvalRunner(services, ds, test_settings)
    base, adv = runner.run("baseline"), runner.run("advanced")
    for rep in (base, adv):
        assert rep["dataset"]["version"] == "2.0.0" and len(rep["results"]) == 5
        assert rep["config"]["llm_provider"] == "mock" and "MOCK" in rep["metric_notes"]["provider"]
        assert rep["latency_ms"]["total"]["count"] == 5
    cmp = compare(base, adv)
    assert "accuracy_answerable" in cmp["comparison"]
    paths = write_reports(adv, str(tmp_path), "t")
    header = open(paths["csv"], encoding="utf-8").readline()
    assert header.startswith("id,category,question") and "failure_categories" in header
