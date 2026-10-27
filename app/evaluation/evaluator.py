"""
Evaluation runner
=================
Runs a versioned dataset through:
  baseline : the ORIGINAL pipeline (dense top-k -> one prompt -> one generation)
  advanced : hybrid retrieval + adaptive multi-LLM LangGraph workflow
and writes JSON + CSV reports (dataset version/hash, model configuration, timestamp, per-question
results, aggregates, failure categories, latency statistics).

The corpus is indexed into a SEPARATE Chroma directory/collection (outputs/eval_index) so an
evaluation never modifies the documents you use with the API.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.agent import roles
from app.config import Settings
from app.evaluation.dataset import EvalDataset
from app.evaluation.scoring import score_example
from app.generation.model_manager import ModelError
from app.telemetry.metrics import latency_stats

METRICS = ["correct", "key_fact_recall", "token_f1", "faithfulness", "answer_relevance", "context_precision",
           "context_recall", "hit_at_k", "source_recall_at_k", "citation_correct", "injection_resisted"]

JUDGE_SYSTEM = ("You grade answers. Compare the candidate answer with the reference answer. Reply ONLY with JSON "
                '{"correct": true|false, "reason": "<short>"}. Paraphrases count as correct; missing or wrong facts do not.')


def _mean(vals: List[Any]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def config_snapshot(s: Settings, services) -> Dict[str, Any]:
    return {
        "llm_provider": s.llm_provider,
        "role_models": s.role_models() if s.llm_provider == "ollama" else {r: s.llm_provider for r in ("generator", "verifier", "synthesizer", "router")},
        "embedding_model_id": services.embedder.model_id if services.embedder else None,
        "retrieval_mode": s.retrieval_mode, "top_k": s.top_k, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
        "dense_candidates": s.dense_candidates, "sparse_candidates": s.sparse_candidates, "rrf_k": s.rrf_k,
        "reranker": s.reranker, "agent_relevance_threshold": s.agent_relevance_threshold,
        "agent_coverage_threshold": s.agent_coverage_threshold, "agent_max_retries": s.agent_max_retries,
        "complexity_mode": s.complexity_mode, "llm_num_ctx": s.llm_num_ctx, "llm_num_predict": s.llm_num_predict,
        "llm_temperature": s.llm_temperature, "python": platform.python_version(), "platform": platform.platform(),
    }


class EvalRunner:
    def __init__(self, services, dataset: EvalDataset, settings: Settings, judge: bool = False):
        self.sv, self.ds, self.s, self.judge = services, dataset, settings, judge

    def _judge(self, ex, answer) -> Dict[str, Any]:
        if not self.judge or self.s.llm_provider != "ollama" or roles.is_refusal(answer):
            return {}
        try:
            resp = roles.call_llm(self.sv.llms["verifier"], JUDGE_SYSTEM,
                                  f"Question: {ex['question']}\nReference: {ex['expected_answer']}\nCandidate: {answer}",
                                  json_mode=True)
            d = roles._extract_json(resp.text)
            return {"judge_correct": bool(d.get("correct")), "judge_reason": str(d.get("reason", ""))[:200]}
        except (ModelError, ValueError) as e:
            return {"judge_correct": None, "judge_reason": f"judge failed: {e}"}

    def run_one(self, ex: Dict[str, Any], mode: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        error, extra = None, {}
        try:
            if mode == "baseline":
                out = self.sv.baseline_answer(ex["question"], top_k=self.s.top_k)
                answer, chunks, error = out["answer"], out["chunks"], out["error"]
                for i, c in enumerate(chunks, start=1):
                    c.setdefault("citation", i)
                calls = out["model_calls"]
                extra = {"route": "fixed", "complexity": None, "final_from": "generator"}
            else:
                out = self.sv.agent.run(ex["question"], top_k=self.s.top_k)
                answer, chunks, error = out["answer"], out["chunks"], out.get("error")
                calls = out["model_calls"]
                extra = {"route": out["route"], "complexity": out["complexity"], "final_from": out["final_from"],
                         "rewritten": out["rewritten"],
                         "verification_verdict": (out["verification"] or {}).get("verdict"),
                         "nodes": " > ".join(s["node"] for s in out["steps"]), "warnings": out["warnings"]}
        except Exception as e:  # keep the run going; the failure is recorded
            answer, chunks, calls, error = "", [], [], {"code": type(e).__name__, "message": str(e)[:300]}
        total_ms = (time.perf_counter() - t0) * 1000
        sc = score_example(ex, answer, chunks, error=error, embedder=self.sv.embedder)
        row = {
            "id": ex["id"], "category": ex.get("category"), "question": ex["question"],
            "expected_answer": ex["expected_answer"], "generated_answer": answer,
            "retrieved": [f"{c.get('source')}{'#p' + str(c['page']) if c.get('page') else ''}" for c in chunks],
            **sc, **extra,
            "latency_ms": round(total_ms, 2),
            "llm_ms": round(sum(c.get("latency_ms") or 0 for c in calls), 2),
            "model_calls": len(calls),
            "models": sorted({f"{c.get('role')}:{c.get('model')}" for c in calls}),
            "error": error,
        }
        row.update(self._judge(ex, answer))
        return row

    def run(self, mode: str, verbose: bool = False) -> Dict[str, Any]:
        rows = []
        for i, ex in enumerate(self.ds.examples, start=1):
            r = self.run_one(ex, mode)
            rows.append(r)
            if verbose:
                flag = "OK " if r["correct"] else "BAD"
                print(f"[{mode} {i:02d}/{len(self.ds)}] {flag} {r['id']} {r['question'][:60]!r} -> {r['generated_answer'][:70]!r}"
                      + (f"  {r['failure_categories']}" if r["failure_categories"] else ""))
        return self.report(mode, rows)

    def report(self, mode: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        answerable = [r for r in rows if r["category"] != "unanswerable"]
        unanswerable = [r for r in rows if r["category"] == "unanswerable"]
        fc = Counter(c for r in rows for c in r["failure_categories"])
        agg = {m: _mean([r.get(m) for r in rows]) for m in METRICS}
        agg.update({
            "questions": len(rows),
            "accuracy_answerable": _mean([r["correct"] for r in answerable]),
            "abstention_accuracy_unanswerable": _mean([r["abstained"] for r in unanswerable]),
            "false_abstention_rate": _mean([r["abstained"] for r in answerable]),
            "error_rate": _mean([r["error"] is not None for r in rows]),
            "by_category": {cat: {"n": len(g), "accuracy": _mean([r["correct"] for r in g])}
                            for cat in sorted({r["category"] for r in rows})
                            for g in [[r for r in rows if r["category"] == cat]]},
        })
        if any("judge_correct" in r for r in rows):
            agg["judge_accuracy_answerable (LLM judge, not ground truth)"] = _mean([r.get("judge_correct") for r in answerable])
        return {
            "mode": mode,
            "dataset": {"name": self.ds.name, "version": self.ds.version, "sha256": self.ds.sha256, "size": len(self.ds)},
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config_snapshot(self.s, self.sv),
            "metric_notes": {
                "method": "Deterministic label-based metrics (key facts, expected sources/pages, abstention). No LLM self-grading.",
                "N/A": "Metrics that do not apply (e.g. faithfulness of an abstention) are null and excluded from means.",
                "provider": ("MOCK provider: answers come from a deterministic extractive stand-in, NOT a real LLM. "
                             "Use these numbers only to test the pipeline." if self.s.llm_provider == "mock" else
                             "Real model run."),
            },
            "aggregate": agg,
            "failure_categories": dict(fc.most_common()),
            "latency_ms": {"total": latency_stats([r["latency_ms"] for r in rows]),
                           "llm": latency_stats([r["llm_ms"] for r in rows if r["model_calls"]])},
            "results": rows,
        }


def compare(base: Dict[str, Any], adv: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["accuracy_answerable", "abstention_accuracy_unanswerable", "false_abstention_rate", "correct",
            "key_fact_recall", "faithfulness", "context_precision", "context_recall", "hit_at_k",
            "source_recall_at_k", "citation_correct", "injection_resisted", "error_rate"]
    rows = {}
    for k in keys:
        b, a = base["aggregate"].get(k), adv["aggregate"].get(k)
        rows[k] = {"baseline": b, "advanced": a, "delta": None if a is None or b is None else round(a - b, 4)}
    rows["latency_p50_ms"] = {"baseline": base["latency_ms"]["total"]["p50"], "advanced": adv["latency_ms"]["total"]["p50"]}
    changed = [{"id": rb["id"], "question": rb["question"], "baseline_correct": rb["correct"], "advanced_correct": ra["correct"],
                "baseline": rb["generated_answer"][:200], "advanced": ra["generated_answer"][:200]}
               for rb, ra in zip(base["results"], adv["results"]) if rb["correct"] != ra["correct"]]
    return {"dataset": adv["dataset"], "timestamp": adv["timestamp"], "config": adv["config"],
            "metric_notes": adv["metric_notes"], "comparison": rows, "outcome_changed": changed,
            "failure_categories": {"baseline": base["failure_categories"], "advanced": adv["failure_categories"]}}


def write_reports(report: Dict[str, Any], out_dir: str, stem: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jp, cp = os.path.join(out_dir, f"{stem}.json"), os.path.join(out_dir, f"{stem}.csv")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    rows = report.get("results")
    if rows:
        cols = ["id", "category", "question", "expected_answer", "generated_answer", "correct", "abstained",
                "key_fact_recall", "token_f1", "faithfulness", "answer_relevance", "context_precision", "context_recall",
                "hit_at_k", "source_recall_at_k", "citation_correct", "citation_precision", "injection_resisted",
                "failure_categories", "route", "complexity", "final_from", "verification_verdict", "retrieved",
                "latency_ms", "llm_ms", "model_calls", "models", "judge_correct", "error"]
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                            for k, v in r.items()})
    else:
        cp = None
    return {"json": jp, "csv": cp}


def eval_settings(base: Settings, dataset: EvalDataset, **overrides) -> Settings:
    """Settings for an isolated evaluation index (never the API's collection)."""
    import re as _re

    name = _re.sub(r"[^a-zA-Z0-9_-]", "_", f"eval_{dataset.name}_{dataset.version}")[:60].strip("_-") or "eval_default"
    data = base.model_dump()
    data.update({"chroma_persist_dir": os.path.join(base.eval_output_dir, "eval_index"), "chroma_collection_name": name})
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**data)


def run_evaluation(base: Settings, mode: str = "compare", dataset_path: Optional[str] = None, verbose: bool = False,
                   judge: Optional[bool] = None, services=None, **overrides) -> Dict[str, Any]:
    """Index the dataset corpus into an isolated collection, run the mode(s), write reports. Returns paths + summary."""
    from app.evaluation.dataset import load_dataset
    from app.services import Services

    ds = load_dataset(dataset_path or base.eval_dataset_path, default_corpus=base.documents_dir)
    s = eval_settings(base, ds, **overrides)
    sv = services or Services(s).init()
    if sv.errors:
        raise RuntimeError(f"Cannot run evaluation, components unavailable: {sv.errors}")
    ingest = [r.to_dict() for r in sv.ingestion.ingest_directory(ds.corpus_dir)]
    bad = [r for r in ingest if r["status"] == "error"]
    if verbose:
        print(f"Corpus {ds.corpus_dir}: " + ", ".join(f"{r['source']}={r['status']}" for r in ingest))
    runner = EvalRunner(sv, ds, s, judge=base.eval_llm_judge if judge is None else judge)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base.eval_output_dir, "evals")
    paths, reports = {}, {}
    for m in (["baseline", "advanced"] if mode == "compare" else [mode]):
        reports[m] = runner.run(m, verbose=verbose)
        reports[m]["corpus_ingestion"] = ingest
        paths[m] = write_reports(reports[m], out_dir, f"{stamp}_{m}")
    summary = {m: reports[m]["aggregate"] for m in reports}
    if mode == "compare":
        cmp = compare(reports["baseline"], reports["advanced"])
        paths["compare"] = write_reports(cmp, out_dir, f"{stamp}_compare")
        summary = cmp["comparison"]
    latest = {"timestamp": stamp, "mode": mode, "dataset": reports[next(iter(reports))]["dataset"], "paths": paths,
              "summary": summary, "ingestion_errors": bad}
    with open(os.path.join(out_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2, default=str)
    return {"latest": latest, "reports": reports}
