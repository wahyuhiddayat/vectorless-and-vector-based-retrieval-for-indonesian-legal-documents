"""BM25 tree retrieval for Indonesian legal QA.

Navigates the document tree level by level using BM25 beam search.
Intermediate nodes are scored by title and summary, then all reached
leaves are re-ranked with full-text BM25 using the same enrichment
fields as bm25-flat. No LLM calls at query time.

Usage:
    python -m vectorless.retrieval.bm25.tree "Apa syarat penyadapan?"
    python -m vectorless.retrieval.bm25.tree "Apa syarat penyadapan?" --top_k_per_level 3 --top_k 10
"""

import argparse
import time

from rank_bm25 import BM25Okapi

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_catalog, load_doc, extract_nodes, save_log, doc_corpus_string,
    DOC_PICK_TOP_K,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K) -> list[dict]:
    """Rank catalog entries with BM25 over the doc corpus string.

    Each document is represented by its metadata and aggregated summary
    text from doc_corpus_string.

    Args:
        query: Legal question in Indonesian.
        catalog: List of document metadata dicts.
        top_k: Number of top documents to return.

    Returns:
        List of dicts with doc_id, judul, and bm25_score.
    """
    corpus = [tokenize(doc_corpus_string(doc)) for doc in catalog]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            results.append({
                "doc_id": catalog[idx]["doc_id"],
                "judul": catalog[idx]["judul"],
                "bm25_score": round(float(score), 4),
            })
    return results


def _bm25_level_search(query: str, nodes: list[dict], top_k: int = 3) -> list[dict]:
    """Score nodes at one tree level using BM25 on title and summary.

    Args:
        query: Legal question in Indonesian.
        nodes: List of nodes at the current tree level.
        top_k: Number of top nodes to select.

    Returns:
        List of selected nodes with their BM25 scores.
    """
    if not nodes:
        return []

    corpus = []
    for node in nodes:
        combined = node.get("title", "") + " " + node.get("summary", "")
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        node = nodes[idx]
        results.append({
            "node_id": node["node_id"],
            "title": node.get("title", ""),
            "summary": node.get("summary", ""),
            "bm25_score": round(float(score), 4),
            "has_children": "nodes" in node and bool(node.get("nodes")),
            "_node_ref": node,
        })

    return results


def _bm25_leaf_search(query: str, leaves: list[dict], doc_title: str,
                      top_k: int = 3) -> list[dict]:
    """Score leaf nodes with BM25 over doc_title, navigation_path, text, and penjelasan.

    Uses the same enrichment fields as bm25-flat for consistency.

    Args:
        query: Legal question in Indonesian.
        leaves: List of leaf nodes without children.
        doc_title: Document title from the parent document.
        top_k: Number of top leaves to select.

    Returns:
        List of selected leaves with their BM25 scores.
    """
    if not leaves:
        return []

    corpus = []
    for leaf in leaves:
        combined = doc_title + " " + leaf.get("navigation_path", "") + " " + leaf.get("text", "")
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            combined += " " + leaf["penjelasan"]
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        leaf = leaves[idx]
        results.append({
            "node_id": leaf["node_id"],
            "title": leaf.get("title", ""),
            "summary": leaf.get("summary", ""),
            "bm25_score": round(float(score), 4),
            "has_children": False,
            "_node_ref": leaf,
        })

    return results


def tree_search(query: str, doc: dict, top_k_per_level: int = 3,
                top_k: int = 10, verbose: bool = True) -> dict:
    """Navigate a document tree with BM25 beam search and re-rank leaves.

    Phase 1 walks the tree level by level, selecting top_k_per_level
    nodes by title and summary at each non-leaf level. Leaves found at
    any depth are accumulated into a candidate pool.

    Phase 2 re-scores the entire candidate pool with full-text BM25
    and returns the top results.

    Args:
        query: Legal question in Indonesian.
        doc: Loaded document dict with a structure field.
        top_k_per_level: Beam width at each navigation level.
        top_k: Number of leaves to return after final re-ranking.
        verbose: Print progress to stdout.

    Returns:
        Dict with steps (navigation trace), node_ids (ranked leaf ids),
        and pool_size (total leaves reached by the beam).
    """
    structure = doc["structure"]
    doc_title = doc.get("judul", "")
    steps = []
    max_rounds = 8

    candidate_pool: list[dict] = []
    seen_leaf_ids: set[str] = set()

    def _add_to_pool(leaves_to_add):
        """Append unseen leaves to the candidate pool."""
        for leaf in leaves_to_add:
            lid = leaf.get("node_id", "")
            if lid and lid not in seen_leaf_ids:
                candidate_pool.append(leaf)
                seen_leaf_ids.add(lid)

    current_nodes = structure
    round_num = 1

    while round_num <= max_rounds:
        all_leaves = all(not (n.get("nodes")) for n in current_nodes)

        if all_leaves:
            _add_to_pool(current_nodes)
            steps.append({
                "round": round_num,
                "level": f"level-{round_num}",
                "all_leaves": True,
                "options_shown": [n.get("title", "") for n in current_nodes],
                "added_to_pool": [n.get("node_id", "") for n in current_nodes],
            })
            if verbose:
                print(f"\n[BM25 Tree - Round {round_num}] All-leaves level reached; "
                      f"added {len(current_nodes)} leaves to candidate pool.")
            break

        selected = _bm25_level_search(query, current_nodes,
                                      top_k=top_k_per_level)
        if not selected:
            break

        steps.append({
            "round": round_num,
            "level": f"level-{round_num}",
            "all_leaves": False,
            "options_shown": [n.get("title", "") for n in current_nodes],
            "selected": [s["node_id"] for s in selected],
            "scores": {s["node_id"]: s["bm25_score"] for s in selected},
        })

        if verbose:
            print(f"\n[BM25 Tree - Round {round_num}] Beam selected:")
            for s in selected:
                print(f"  {s['node_id']} {s['title']} (BM25: {s['bm25_score']:.4f})")

        need_drill = []
        for s in selected:
            node_ref = s["_node_ref"]
            if s["has_children"]:
                need_drill.extend(node_ref.get("nodes", []))
            else:
                _add_to_pool([node_ref])

        if not need_drill:
            break

        current_nodes = need_drill
        round_num += 1

    if not candidate_pool:
        return {"steps": steps, "node_ids": [], "pool_size": 0}

    final_ranked = _bm25_leaf_search(query, candidate_pool, doc_title,
                                     top_k=top_k)
    final_ids = [s["node_id"] for s in final_ranked]

    if verbose:
        print(f"\n[BM25 Tree - Final Ranking] Re-ranked pool of "
              f"{len(candidate_pool)} leaves, returned top-{len(final_ids)}.")

    return {
        "steps": steps,
        "node_ids": final_ids,
        "pool_size": len(candidate_pool),
        "reached_leaves": candidate_pool,
    }


def retrieve(query: str, top_k_per_level: int = 3, top_k: int = 10,
             top_k_docs: int = DOC_PICK_TOP_K, verbose: bool = True) -> dict:
    """Run the full BM25 tree retrieval pipeline across multiple documents.

    Selects top documents from the catalog with BM25, runs beam-based
    tree search in each, merges all reached leaves, and re-ranks
    them globally with full-text BM25.

    Args:
        query: Legal question in Indonesian.
        top_k_per_level: Beam width during tree traversal.
        top_k: Final number of leaves to return.
        top_k_docs: Number of documents selected at stage 1.
        verbose: Print progress to stdout.

    Returns:
        Dict with query, strategy, search results, sources, and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: bm25-tree (top_k_docs={top_k_docs}, "
              f"top_k_per_level={top_k_per_level}, top_k={top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_results = _bm25_doc_search(query, catalog, top_k=top_k_docs)
    steps["doc_search"] = step_metrics(t_step, snap)

    if not doc_results:
        return {"query": query, "strategy": "bm25-tree", "picked_doc_ids": [],
                "error": "No relevant documents found"}

    picked_doc_ids = [r["doc_id"] for r in doc_results]

    if verbose:
        print(f"\n[Doc Search - BM25] Selected: {picked_doc_ids}")
        for r in doc_results:
            print(f"  {r['doc_id']} (BM25: {r['bm25_score']:.4f})")

    snap = snapshot_counters()
    t_step = time.time()

    merged_leaves: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    per_doc_pool_sizes: dict[str, int] = {}
    tree_search_per_doc: dict[str, dict] = {}

    for doc_meta in doc_results:
        doc_id = doc_meta["doc_id"]
        doc = load_doc(doc_id)
        tree_result = tree_search(query, doc, top_k_per_level=top_k_per_level,
                                  top_k=top_k, verbose=verbose)
        tree_search_per_doc[doc_id] = {
            "steps": tree_result.get("steps", []),
            "pool_size": tree_result.get("pool_size", 0),
        }
        per_doc_pool_sizes[doc_id] = tree_result.get("pool_size", 0)
        for leaf in tree_result.get("reached_leaves", []) or []:
            nid = leaf.get("node_id", "")
            key = (doc_id, nid)
            if not nid or key in seen_keys:
                continue
            tagged = dict(leaf)
            tagged["_doc_id"] = doc_id
            tagged["_doc_title"] = doc.get("judul", "")
            merged_leaves.append(tagged)
            seen_keys.add(key)

    steps["tree_search"] = step_metrics(t_step, snap)

    if not merged_leaves:
        return {"query": query, "strategy": "bm25-tree",
                "picked_doc_ids": picked_doc_ids,
                "error": "No leaves reached across picked docs"}

    corpus = []
    for leaf in merged_leaves:
        enriched = (leaf["_doc_title"] + " "
                    + (leaf.get("navigation_path") or "") + " "
                    + (leaf.get("text") or ""))
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            enriched += " " + leaf["penjelasan"]
        corpus.append(tokenize(enriched))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    final_picks: list[dict] = []
    for idx, score in ranked:
        if score <= 0:
            continue
        leaf = merged_leaves[idx]
        final_picks.append({
            "doc_id": leaf["_doc_id"],
            "node_id": leaf["node_id"],
            "title": leaf.get("title", ""),
            "navigation_path": leaf.get("navigation_path", ""),
            "bm25_score": round(float(score), 4),
        })
        if len(final_picks) >= top_k:
            break

    if not final_picks:
        return {"query": query, "strategy": "bm25-tree",
                "picked_doc_ids": picked_doc_ids,
                "error": "All merged leaves scored <= 0"}

    sources = []
    for pos, pick in enumerate(final_picks):
        sources.append({
            "doc_id": pick["doc_id"],
            "node_id": pick["node_id"],
            "title": pick["title"],
            "navigation_path": pick["navigation_path"],
            "bm25_score": pick["bm25_score"],
            "rerank_position": pos,
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "bm25-tree",
        "picked_doc_ids": picked_doc_ids,
        "doc_search": {"rankings": doc_results},
        "tree_search_per_doc": tree_search_per_doc,
        "per_doc_pool_sizes": per_doc_pool_sizes,
        "merged_pool_size": len(merged_leaves),
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
    """CLI entry point for BM25 tree retrieval."""
    ap = argparse.ArgumentParser(
        description="BM25 tree (hierarchical) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k_per_level", type=int, default=3,
                    help="Beam width during traversal (default: 3)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final number of leaves returned (default: 10)")
    ap.add_argument("--top_k_docs", type=int, default=DOC_PICK_TOP_K,
                    help=f"Number of docs picked at stage 1 (default: {DOC_PICK_TOP_K})")
    args = ap.parse_args()

    result = retrieve(args.query, top_k_per_level=args.top_k_per_level,
                      top_k=args.top_k, top_k_docs=args.top_k_docs)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
