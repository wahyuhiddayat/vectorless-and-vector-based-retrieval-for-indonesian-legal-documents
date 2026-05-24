"""BM25 flat retrieval for Indonesian legal QA.

Searches all leaf nodes across all documents in a single stage.
Each leaf is enriched with its document title and navigation path
before scoring.

Usage:
    python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?" --top_k 10
"""

import argparse
import time

from rank_bm25 import BM25Okapi

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_all_leaf_nodes, save_log,
)


def flat_search(query: str, leaves: list[dict], top_k: int = 10,
                verbose: bool = True) -> list[dict]:
    """Score all leaf nodes with BM25 and return the top results.

    Each leaf is enriched with doc_title, navigation_path, text, and
    penjelasan before tokenization.

    Args:
        query: Legal question in Indonesian.
        leaves: All leaf nodes loaded from the corpus.
        top_k: Maximum number of results to return.
        verbose: Print ranked results to stdout.

    Returns:
        List of result dicts sorted by BM25 score descending.
    """
    corpus = []
    for leaf in leaves:
        enriched = leaf["doc_title"] + " " + leaf["navigation_path"] + " " + leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            enriched += " " + leaf["penjelasan"]
        corpus.append(tokenize(enriched))

    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Collect up to `target` results, skipping zero-score leaves
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    target = max(top_k, 10)
    results = []
    for idx, score in ranked:
        if score <= 0:
            continue
        leaf = leaves[idx]
        results.append({
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "node_id": leaf["node_id"],
            "title": leaf["title"],
            "navigation_path": leaf["navigation_path"],
            "text": leaf["text"],
            "penjelasan": leaf.get("penjelasan"),
            "score": round(float(score), 4),
        })
        if len(results) >= target:
            break

    if verbose:
        print(f"\n[Flat BM25 Search] Top {len(results)} results:")
        for r in results:
            print(f"  {r['node_id']} {r['title']} (BM25: {r['score']:.4f})")
            print(f"    doc: {r['doc_id']}  path: {r['navigation_path']}")

    return results


def retrieve(query: str, top_k: int = 10, verbose: bool = True) -> dict:
    """Run the full flat BM25 retrieval pipeline.

    Loads all leaf nodes from the corpus, scores them with BM25,
    and returns the top results with timing and token metrics.

    Args:
        query: Legal question in Indonesian.
        top_k: Maximum number of results to return.
        verbose: Print progress to stdout.

    Returns:
        Dict with query, strategy, search rankings, sources, and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: bm25-flat (top_k={top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    results = flat_search(query, leaves, top_k=top_k, verbose=verbose)
    steps["bm25_search"] = step_metrics(t_step, snap)

    if not results:
        return {"query": query, "strategy": "bm25-flat",
                "error": "No results found"}

    sources = []
    for r in results:
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "bm25_score": r["score"],
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "bm25-flat",
        "corpus_size": len(leaves),
        "search": {"rankings": results},
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": steps},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


def main():
    """CLI entry point for BM25 flat retrieval."""
    ap = argparse.ArgumentParser(
        description="BM25 flat retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k", type=int, default=10, help="Number of results (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']} (BM25: {src['bm25_score']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
