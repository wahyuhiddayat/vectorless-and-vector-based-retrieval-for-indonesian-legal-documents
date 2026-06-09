"""Free BM25 k1 and b tuning for hybrid-tree, on candidate recall (no LLM).

Sweeps BM25 k1 and b over a grid and measures, for each cell, the fraction of
gold pasals that appear in the top-k BM25 node candidates within the gold
documents. This isolates the node-level BM25 quality that caps what the LLM
reranker can recover, at zero LLM cost. Step 1 of the vectorless tuning flow
(tune_vl_step1_bm25params.py) calls this to set HYBRID_BM25_K1 and
HYBRID_BM25_B for the later steps. This script can also be run standalone to
inspect the full grid.

Using the gold documents as an oracle for doc selection keeps this a clean
measure of the node-level BM25 alone, independent of the doc-pick stage.

Usage:
    python scripts/eval/tune_bm25_params.py
    python scripts/eval/tune_bm25_params.py --bm25-top-k 20 --split dev
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DATA_INDEX", "data/index_pasal")

from scripts.eval.core.io import load_testset, load_split_qids
from vectorless.retrieval.common import tokenize, load_doc

K1_GRID = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
B_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def node_corpus(doc: dict) -> tuple[list[list[str]], list[str]]:
    """Tokenized leaf corpus for one doc, matching hybrid-tree enrichment."""
    leaves: list[dict] = []

    def walk(nodes):
        """Collect every leaf node in the subtree into leaves."""
        for n in nodes:
            if n.get("nodes"):
                walk(n["nodes"])
            elif n.get("text"):
                leaves.append(n)

    walk(doc.get("structure", []))
    doc_title = doc.get("judul", "")
    corpus, ids = [], []
    for leaf in leaves:
        combined = doc_title + " " + leaf.get("navigation_path", "") + " " + leaf["text"]
        pj = leaf.get("penjelasan")
        if pj and pj != "Cukup jelas.":
            combined += " " + pj
        corpus.append(tokenize(combined))
        ids.append(leaf["node_id"])
    return corpus, ids


def build_queries(split: str) -> list[dict]:
    """Load split queries with gold pasals and pre-tokenized gold-doc corpora."""
    testset = load_testset(Path("data/validated_testset.pkl"))
    qids = set(load_split_qids(split))
    doc_cache: dict[str, tuple] = {}
    out = []
    for qid, item in testset.items():
        if qid not in qids:
            continue
        gold = [g for g in item.get("gold_pasal_node_ids", []) if g]
        docs = [d for d in item.get("gold_doc_ids", []) if d]
        if not gold or not docs:
            continue
        corpora = {}
        for did in set(docs):
            if did not in doc_cache:
                doc_cache[did] = node_corpus(load_doc(did))
            corpora[did] = doc_cache[did]
        out.append({
            "query_tokens": tokenize(item["query"]),
            "gold": set(gold),
            "docs": set(docs),
            "corpora": corpora,
        })
    return out


def candidate_recall(queries: list[dict], k1: float, b: float, top_k: int) -> float:
    """Mean fraction of gold pasals found in the top-k BM25 candidates per query."""
    recalls = []
    for q in queries:
        found = set()
        for did in q["docs"]:
            corpus, ids = q["corpora"][did]
            if not corpus:
                continue
            bm25 = BM25Okapi(corpus, k1=k1, b=b)
            scores = bm25.get_scores(q["query_tokens"])
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
            for i in top:
                if ids[i] in q["gold"]:
                    found.add(ids[i])
        recalls.append(len(found & q["gold"]) / len(q["gold"]))
    return sum(recalls) / len(recalls)


def tune_k1_b(split: str = "dev", bm25_top_k: int = 20) -> dict:
    """Grid-search k1 and b on candidate recall, return the winner and the grid.

    Ties at the top recall resolve to the cell closest to the library default
    (k1=1.5, b=0.75), so a null result reports the default rather than an
    arbitrary equal-scoring cell. No LLM is called.

    Returns a dict with best_k1, best_b, best_recall, default_recall, and the
    full results list of (k1, b, recall) tuples.
    """
    queries = build_queries(split)
    results = []
    for k1 in K1_GRID:
        for b in B_GRID:
            results.append((k1, b, candidate_recall(queries, k1, b, bm25_top_k)))
    best_r = max(r for _, _, r in results)
    winners = [(k1, b) for k1, b, r in results if r >= best_r - 1e-9]
    best_k1, best_b = min(winners, key=lambda kb: (abs(kb[0] - 1.5), abs(kb[1] - 0.75)))
    default_r = next(r for k1, b, r in results if k1 == 1.5 and b == 0.75)
    return {
        "best_k1": best_k1, "best_b": best_b, "best_recall": best_r,
        "default_recall": default_r, "results": results,
    }


def main() -> int:
    """Sweep the k1/b grid and report candidate recall, best cell last."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--bm25-top-k", type=int, default=20,
                    help="Candidate pool size to measure recall at (default 20).")
    args = ap.parse_args()

    print(f"Loading {args.split} split and gold-doc corpora...")
    out = tune_k1_b(args.split, args.bm25_top_k)
    print(f"BM25 candidate recall@{args.bm25_top_k}, {len(K1_GRID)}x{len(B_GRID)} grid.\n")
    print(f"{'k1':>5} {'b':>6} {'recall@'+str(args.bm25_top_k):>10}")
    for k1, b, r in sorted(out["results"], key=lambda x: x[2]):
        print(f"{k1:>5} {b:>6} {r:>10.4f}")

    print(f"\nBest:    k1={out['best_k1']}, b={out['best_b']}, "
          f"recall@{args.bm25_top_k}={out['best_recall']:.4f}")
    print(f"Default: k1=1.5,  b=0.75, recall@{args.bm25_top_k}={out['default_recall']:.4f}")
    print(f"Gain over default: {out['best_recall'] - out['default_recall']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
