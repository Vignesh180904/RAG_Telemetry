"""
Evaluation dataset loader
=========================
v2 format (versioned object):
    {"dataset": "...", "version": "2.0.0", "corpus_dir": "...", "examples": [
        {"id", "category", "question", "expected_answer", "expected_sources": [...],
         "expected_pages": [..] (optional), "reference_context", "key_facts": [...],
         "answerable": true|false, "forbidden_strings": [...] (optional)}]}

The original v1 format (a bare list of {question, ground_truth, relevant_context, relevant_source})
is still accepted and converted (version reported as "1.x-legacy").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from app.agent.roles import REFUSAL


@dataclass
class EvalDataset:
    name: str
    version: str
    corpus_dir: str
    examples: List[Dict[str, Any]]
    sha256: str

    def __len__(self):
        return len(self.examples)


def load_dataset(path: str, default_corpus: str = "./data/documents") -> EvalDataset:
    raw_bytes = open(path, "rb").read()
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    raw = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(raw, list):  # v1
        exs = []
        for i, it in enumerate(raw, start=1):
            unanswerable = it["ground_truth"].strip().lower().startswith(REFUSAL[:20].lower())
            exs.append({"id": f"q{i:02d}", "category": "unanswerable" if unanswerable else "factoid",
                        "question": it["question"], "expected_answer": it["ground_truth"],
                        "expected_sources": [it["relevant_source"]] if it.get("relevant_source") else [],
                        "reference_context": it.get("relevant_context", ""), "key_facts": [],
                        "answerable": not unanswerable})
        return EvalDataset("legacy-v1", "1.x-legacy", default_corpus, exs, digest)
    for ex in raw["examples"]:
        for k in ("id", "question", "expected_answer"):
            if k not in ex:
                raise ValueError(f"dataset example missing '{k}': {ex}")
        ex.setdefault("expected_sources", [])
        ex.setdefault("key_facts", [])
        ex.setdefault("answerable", True)
        ex.setdefault("category", "factoid")
    return EvalDataset(raw.get("dataset", "unnamed"), raw.get("version", "0"), raw.get("corpus_dir", default_corpus),
                       raw["examples"], digest)


# ---- original API (kept for any external scripts)
@dataclass
class EvalExample:
    question: str
    ground_truth: str
    relevant_context: str
    relevant_source: str = ""


def load_eval_dataset(path: str) -> List[EvalExample]:
    ds = load_dataset(path)
    return [EvalExample(e["question"], e["expected_answer"], e.get("reference_context", ""),
                        (e.get("expected_sources") or [""])[0]) for e in ds.examples]
