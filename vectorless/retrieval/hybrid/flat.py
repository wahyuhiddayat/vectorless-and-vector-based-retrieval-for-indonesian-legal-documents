"""Hybrid flat retrieval for Indonesian legal QA.

BM25 global search across all leaf nodes, followed by LLM listwise
reranking. Candidates are passed to the LLM in BM25 order with an
explicit bm25_rank field.

Usage:
    python -m vectorless.retrieval.hybrid.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.flat "Apa syarat penyadapan?" --bm25_top_k 20
"""

import argparse
import json
import time

from rank_bm25 import BM25Okapi

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_all_leaf_nodes, save_log,
    validate_llm_ranking,
)


def flat_bm25_candidates(query: str, leaves: list[dict], top_k: int = 20,
                         verbose: bool = True) -> list[dict]:
    """Score all leaf nodes with BM25 and return top candidates.

    Each leaf is enriched with doc_title, navigation_path, text, and
    penjelasan before tokenization.

    Args:
        query: Legal question in Indonesian.
        leaves: All leaf nodes loaded from the corpus.
        top_k: Maximum number of candidates to return.
        verbose: Print ranked candidates to stdout.

    Returns:
        List of candidate dicts sorted by BM25 score descending.
    """
    corpus = []
    for leaf in leaves:
        enriched = leaf["doc_title"] + " " + leaf["navigation_path"] + " " + leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            enriched += " " + leaf["penjelasan"]
        corpus.append(tokenize(enriched))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    target = max(top_k, 10)  # ensure at least 10 candidates for stable evaluation
    candidates = []
    for idx, score in ranked:
        if score <= 0:
            continue
        leaf = leaves[idx]
        candidates.append({
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "node_id": leaf["node_id"],
            "title": leaf["title"],
            "navigation_path": leaf["navigation_path"],
            "text": leaf["text"],
            "penjelasan": leaf.get("penjelasan"),
            "summary": leaf.get("summary", ""),
            "bm25_score": round(float(score), 4),
        })
        if len(candidates) >= target:
            break

    if verbose:
        print(f"\n[Hybrid-Flat BM25] Top {len(candidates)} candidates:")
        for c in candidates:
            print(f"  {c['node_id']} {c['title']} (BM25: {c['bm25_score']:.4f})")
            print(f"    doc: {c['doc_id']}  path: {c['navigation_path']}")

    return candidates


def llm_rerank(query: str, candidates: list[dict]) -> dict:
    """Ask the LLM to rank all candidates from most to least relevant.

    Each candidate is identified by a compound ref (doc_id/node_id) to
    avoid collisions across documents. The LLM receives all candidates
    with full text in a single prompt and produces a complete ordering.

    Args:
        query: Legal question in Indonesian.
        candidates: BM25-ranked candidate dicts with text fields.

    Returns:
        LLM response dict with thinking and ranking fields.
    """
    candidates_for_prompt = []
    for idx, c in enumerate(candidates):
        ref = f"{c['doc_id']}/{c['node_id']}"
        entry = {
            "bm25_rank": idx + 1,
            "ref": ref,
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "text": c.get("text") or "",
        }
        penjelasan = c.get("penjelasan")
        if penjelasan and penjelasan != "Cukup jelas.":
            entry["penjelasan"] = penjelasan
        candidates_for_prompt.append(entry)

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)
    n_candidates = len(candidates)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan {n_candidates} Pasal kandidat dari berbagai Undang-Undang.
Setiap kandidat memiliki `ref` (format "doc_id/node_id" sebagai pengenal unik lintas UU),
isi teks (text), penjelasan resmi (jika ada), dan `bm25_rank` (peringkat dari tahap pertama
BM25, di mana rank 1 = paling cocok secara kata kunci).

Kandidat sudah diurutkan menaik berdasarkan bm25_rank (rank 1 muncul pertama). Urutan
ini adalah prior dari pencocokan term, bukan ground truth relevansi. Gunakan urutan ini
sebagai sinyal awal, lalu pertimbangkan isi text dan penjelasan untuk menentukan ranking
akhir yang benar.

Pertanyaan: {query}

Kandidat Pasal:
{candidates_text}

Tugas: Urutkan SELURUH {n_candidates} kandidat dari paling relevan ke paling tidak relevan
untuk menjawab pertanyaan. Output harus berisi SEMUA {n_candidates} ref dari input,
tanpa duplikat dan tanpa ref yang tidak ada di input.

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat tentang kriteria ranking>",
  "ranking": ["doc_id_1/node_id_1", "doc_id_2/node_id_2", "..."]
}}

Aturan:
- "ranking" HARUS berisi tepat {n_candidates} ref
- "ranking" tidak boleh ada duplikat
- Setiap ref harus muncul di input (tidak boleh hallucinate)
- Urutan menentukan ranking (index 0 = paling relevan)
- Pertimbangkan isi text, penjelasan, sumber UU (doc_title), navigation_path, dan bm25_rank sebagai prior
- Kembalikan HANYA JSON
"""

    return llm_call(prompt)


def retrieve(query: str, bm25_top_k: int = 20, verbose: bool = True) -> dict:
    """Run the full hybrid flat retrieval pipeline.

    Loads all leaf nodes, scores them with BM25, then reranks the
    top candidates with a single LLM listwise call.

    Args:
        query: Legal question in Indonesian.
        bm25_top_k: Number of BM25 candidates passed to LLM reranking.
        verbose: Print progress to stdout.

    Returns:
        Dict with query, strategy, candidates, rerank result, sources,
        and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid-flat (bm25_top_k={bm25_top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    candidates = flat_bm25_candidates(query, leaves, top_k=bm25_top_k, verbose=verbose)
    steps["bm25_search"] = step_metrics(t_step, snap)

    if not candidates:
        return {"query": query, "strategy": "hybrid-flat",
                "error": "No BM25 candidates found"}

    snap = snapshot_counters()
    t_step = time.time()

    rerank_result = llm_rerank(query, candidates)

    raw_ranking = rerank_result.get("ranking", [])
    valid_refs = {f"{c['doc_id']}/{c['node_id']}" for c in candidates}
    n_hallucinated = sum(1 for r in raw_ranking if r not in valid_refs)

    pseudo_candidates = [
        {"node_id": f"{c['doc_id']}/{c['node_id']}"} for c in candidates
    ]
    ranked_refs = validate_llm_ranking(raw_ranking, pseudo_candidates)
    rerank_result["validated_ranking"] = ranked_refs
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(ranked_refs)
    rerank_result["n_hallucinated"] = n_hallucinated

    steps["rerank"] = step_metrics(t_step, snap)

    if verbose:
        print(f"\n[Hybrid-Flat LLM Rerank] Ranked {len(ranked_refs)} candidates")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    candidate_map = {f"{c['doc_id']}/{c['node_id']}": c for c in candidates}
    ranked_results = [candidate_map[r] for r in ranked_refs if r in candidate_map]

    sources = []
    for pos, r in enumerate(ranked_results):
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "bm25_score": r["bm25_score"],
            "rerank_position": pos,
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid-flat",
        "corpus_size": len(leaves),
        "bm25_candidates": [{
            "node_id": c["node_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "bm25_score": c["bm25_score"],
        } for c in candidates],
        "rerank_result": rerank_result,
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
    """CLI entry point for hybrid flat retrieval."""
    ap = argparse.ArgumentParser(
        description="Hybrid flat retrieval (BM25 + LLM rerank) for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates for LLM reranking (default: 20)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']} (BM25: {src['bm25_score']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
