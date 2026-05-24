"""Hybrid flat retrieval with RRF fusion of BM25 and LLM rankings.

BM25 and LLM each produce an independent ranking over the same
candidate set. The two rankings are fused via Reciprocal Rank Fusion
(Cormack et al. 2009) to produce the final ordering.

Usage:
    python -m vectorless.retrieval.hybrid.flat_rrf "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.flat_rrf "Apa syarat penyadapan?" --bm25_top_k 20 --k_rrf 60
"""

import argparse
import random
import time

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import load_all_leaf_nodes, save_log, validate_llm_ranking
from .flat import flat_bm25_candidates, llm_rerank


def rrf_fuse(rankings_a: list[str], rankings_b: list[str],
             k_rrf: int = 60, top_k: int = 10) -> tuple[list[str], dict[str, float]]:
    """Fuse two ranked id lists using Reciprocal Rank Fusion.

    Each id receives score 1/(k_rrf + rank) from each ranking it appears
    in. Items in only one ranking receive that ranking's contribution.
    Ties are broken by appearance order in rankings_a.

    Args:
        rankings_a: First ranked id list, rank 1 is best.
        rankings_b: Second ranked id list, rank 1 is best.
        k_rrf: RRF dampening constant (default 60, Cormack et al. 2009).
        top_k: Number of fused results to return.

    Returns:
        Tuple of (fused top-k ids, full score map).
    """
    scores: dict[str, float] = {}
    for rank, nid in enumerate(rankings_a, start=1):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rank)
    for rank, nid in enumerate(rankings_b, start=1):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rank)
    order_a = {nid: i for i, nid in enumerate(rankings_a)}
    ordered = sorted(
        scores.keys(),
        key=lambda x: (-scores[x], order_a.get(x, len(rankings_a))),
    )
    return ordered[:top_k], scores


def retrieve(query: str, bm25_top_k: int = 20, k_rrf: int = 60,
             top_k: int = 10, verbose: bool = True) -> dict:
    """Run the full hybrid flat RRF retrieval pipeline.

    BM25 scores all leaf nodes, the LLM reranks shuffled candidates
    independently, and the two rankings are fused with RRF.

    Args:
        query: Legal question in Indonesian.
        bm25_top_k: Number of BM25 candidates for reranking.
        k_rrf: RRF dampening constant.
        top_k: Number of final results to return.
        verbose: Print progress to stdout.

    Returns:
        Dict with query, strategy, candidates, rerank and RRF results,
        sources, and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid-flat-rrf "
              f"(bm25_top_k={bm25_top_k}, k_rrf={k_rrf}, top_k={top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    candidates = flat_bm25_candidates(query, leaves, top_k=bm25_top_k, verbose=verbose)
    steps["bm25_search"] = step_metrics(t_step, snap)

    if not candidates:
        return {"query": query, "strategy": "hybrid-flat-rrf",
                "error": "No BM25 candidates found"}

    def _ref(c: dict) -> str:
        """Build a compound ref that is unique across documents."""
        return f"{c['doc_id']}/{c['node_id']}"

    bm25_ranking_refs = [_ref(c) for c in candidates]

    snap = snapshot_counters()
    t_step = time.time()

    # Shuffle to avoid positional bias in the LLM rerank
    shuffled = list(candidates)
    random.shuffle(shuffled)

    rerank_result = llm_rerank(query, shuffled)

    raw_ranking = rerank_result.get("ranking", [])
    valid_ids = {c["node_id"] for c in candidates}
    n_hallucinated = sum(1 for nid in raw_ranking if nid not in valid_ids)
    llm_node_ranking = validate_llm_ranking(raw_ranking, candidates)

    # Map node_id ranking back to compound refs
    node_to_refs: dict[str, list[str]] = {}
    for c in candidates:
        node_to_refs.setdefault(c["node_id"], []).append(_ref(c))
    used_refs: set[str] = set()
    llm_ranking_refs: list[str] = []
    for nid in llm_node_ranking:
        for ref in node_to_refs.get(nid, []):
            if ref not in used_refs:
                llm_ranking_refs.append(ref)
                used_refs.add(ref)
                break
    # Append any refs missing from the LLM ranking in BM25 order
    for ref in bm25_ranking_refs:
        if ref not in used_refs:
            llm_ranking_refs.append(ref)
            used_refs.add(ref)

    rerank_result["validated_ranking"] = llm_ranking_refs
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(llm_ranking_refs)
    rerank_result["n_hallucinated"] = n_hallucinated

    steps["rerank"] = step_metrics(t_step, snap)

    fused_refs, rrf_scores = rrf_fuse(
        bm25_ranking_refs, llm_ranking_refs, k_rrf=k_rrf, top_k=top_k,
    )

    if verbose:
        print(f"\n[RRF Fusion] k_rrf={k_rrf}, top_k={top_k}")
        for pos, ref in enumerate(fused_refs[:top_k]):
            print(f"  rank {pos+1}: {ref}  rrf_score={rrf_scores.get(ref, 0):.6f}")

    candidate_map = {_ref(c): c for c in candidates}
    bm25_rank_map = {ref: i + 1 for i, ref in enumerate(bm25_ranking_refs)}
    llm_rank_map = {ref: i + 1 for i, ref in enumerate(llm_ranking_refs)}

    sources = []
    for pos, ref in enumerate(fused_refs):
        c = candidate_map.get(ref)
        if not c:
            continue
        sources.append({
            "doc_id": c["doc_id"],
            "node_id": c["node_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "bm25_score": c["bm25_score"],
            "bm25_rank": bm25_rank_map.get(ref),
            "llm_rank": llm_rank_map.get(ref),
            "rrf_score": round(rrf_scores.get(ref, 0.0), 6),
            "rerank_position": pos,
        })

    if not sources:
        return {"query": query, "strategy": "hybrid-flat-rrf",
                "error": "RRF fusion produced no sources"}

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid-flat-rrf",
        "corpus_size": len(leaves),
        "bm25_candidates": [{
            "node_id": c["node_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "bm25_score": c["bm25_score"],
        } for c in candidates],
        "rerank_result": rerank_result,
        "rrf": {
            "k_rrf": k_rrf,
            "top_k": top_k,
            "fused_refs": fused_refs,
            "score_map": {nid: round(s, 6) for nid, s in rrf_scores.items()},
        },
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": steps},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        for step_name, sm in steps.items():
            print(f"  {step_name}: {sm['elapsed_s']:.1f}s, {sm['llm_calls']} calls, "
                  f"{sm['input_tokens']+sm['output_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


def main():
    """CLI entry point for hybrid-flat-rrf retrieval."""
    ap = argparse.ArgumentParser(
        description="Hybrid-flat retrieval with RRF fusion of BM25 and LLM rankings.")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates for LLM reranking (default: 20)")
    ap.add_argument("--k_rrf", type=int, default=60,
                    help="RRF dampening constant (default: 60, Cormack et al.)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final number of leaves to return (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k,
                      k_rrf=args.k_rrf, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}  "
              f"(rrf={src['rrf_score']}, bm25_rank={src['bm25_rank']}, llm_rank={src['llm_rank']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
