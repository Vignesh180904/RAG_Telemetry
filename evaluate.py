"""
Evaluation CLI
==============
    python evaluate.py                         # baseline vs advanced comparison (default)
    python evaluate.py --mode advanced -v      # one mode, per-question output
    python evaluate.py --mode baseline
    python evaluate.py --dataset data/evaluation/eval_dataset.json   # original v1 dataset still works

Old mode names still work: --mode plain (= baseline), --mode agent (= advanced).

Reports go to outputs/evals/<timestamp>_<mode>.json|.csv and outputs/evals/latest.json.
The corpus is indexed into outputs/eval_index (separate from the API's chroma_db).
With LLM_PROVIDER=mock the answers come from a deterministic stand-in -- the scores then test
the pipeline, not answer quality.
"""

import argparse
import json
import sys

from app.config import settings
from app.evaluation.evaluator import run_evaluation
from app.telemetry.tracing import configure_logging

ALIASES = {"plain": "baseline", "agent": "advanced"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the RAG evaluation pipeline.")
    p.add_argument("--mode", default="compare", choices=["baseline", "advanced", "compare", "plain", "agent"])
    p.add_argument("--dataset", default=None, help=f"default: {settings.eval_dataset_path}")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--provider", choices=["mock", "ollama"], default=None, help="override LLM_PROVIDER")
    p.add_argument("--embedding-backend", choices=["sentence-transformers", "ollama", "fastembed", "hash"], default=None)
    p.add_argument("--judge", action="store_true", help="add an LLM-judge column (Ollama only; reported separately)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    configure_logging("WARNING")
    mode = ALIASES.get(a.mode, a.mode)
    out = run_evaluation(settings, mode=mode, dataset_path=a.dataset, verbose=a.verbose, judge=a.judge or None,
                         top_k=a.top_k, llm_provider=a.provider, embedding_backend=a.embedding_backend)
    latest = out["latest"]
    print("\n" + "=" * 72)
    print(f"Dataset {latest['dataset']['name']} v{latest['dataset']['version']} ({latest['dataset']['size']} questions)")
    if mode == "compare":
        print(f"{'metric':38s}{'baseline':>11s}{'advanced':>11s}{'delta':>10s}")
        for k, v in latest["summary"].items():
            fmt = lambda x: "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
            print(f"{k:38s}{fmt(v.get('baseline')):>11s}{fmt(v.get('advanced')):>11s}{fmt(v.get('delta')):>10s}")
    else:
        print(json.dumps(latest["summary"], indent=2, default=str))
    print("Reports:", json.dumps(latest["paths"], indent=2))
    if latest["ingestion_errors"]:
        print("Corpus files that failed to ingest:", latest["ingestion_errors"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
