"""LLM flat retrieval via chunked elimination tournament.

Ranks all leaf nodes using listwise LLM calls in a multi-round
elimination tournament. Each round shuffles the pool, splits it into
chunks, and advances the top survivors per chunk. The final round
ranks survivors twice with different shuffles and merges via Borda
count. The LLM sees metadata only (title, summary, navigation_path),
not full leaf text.

Usage:
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?" --window_size 200
"""

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_all_leaf_nodes, save_log, validate_llm_ranking,
)


LISTWISE_WINDOW = 400      # candidates per LLM ranking call
LISTWISE_SURVIVORS = 20    # top-K kept from each chunk to advance to next round


def _query_seed(query: str) -> int:
    """Derive a deterministic int seed from a query string."""
    return int(hashlib.md5(query.encode("utf-8")).hexdigest()[:8], 16)


def _borda_merge(ranking_a: list[str], ranking_b: list[str], top_k: int) -> list[str]:
    """Merge two ranked id lists via Borda count, return top-k.

    Each id gets points equal to (list_len - position) in each ranking.
    Ids missing from a ranking get zero points from that ranking. Ties
    broken by first appearance in ranking_a for determinism.
    """
    scores: dict[str, int] = defaultdict(int)
    order_a: dict[str, int] = {}
    for i, nid in enumerate(ranking_a):
        scores[nid] += len(ranking_a) - i
        order_a.setdefault(nid, i)
    for i, nid in enumerate(ranking_b):
        scores[nid] += len(ranking_b) - i
    ordered = sorted(scores.keys(),
                     key=lambda x: (-scores[x], order_a.get(x, len(ranking_a))))
    return ordered[:top_k]


def _build_rank_prompt(query: str, candidates: list[dict], top_k: int) -> str:
    """Build the top-K selection prompt for one chunk of candidates.

    Candidates are identified by compound refs (doc_id/node_id). The
    prompt asks for a short top-K list only, not a full ranking.

    Args:
        query: Legal question in Indonesian.
        candidates: Candidate dicts with ref and metadata fields.
        top_k: Number of top candidates the LLM should select.

    Returns:
        Formatted prompt string.
    """
    n = len(candidates)
    candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"""\
Kamu diberi pertanyaan hukum dan {n} Pasal kandidat dari berbagai Undang-Undang Indonesia.
Setiap kandidat memiliki ringkasan isi (summary) dan lokasi dalam dokumen.
Setiap kandidat punya field "ref" berformat "doc_id/node_id" yang UNIK (gunakan ref
ini di jawaban karena node_id sendiri bisa sama antar dokumen).

Pertanyaan: {query}

Daftar Pasal kandidat:
{candidates_text}

Tugas: Pilih {top_k} Pasal PALING RELEVAN dari {n} kandidat di atas untuk menjawab
pertanyaan. Urutkan dari paling relevan ke kurang relevan.

Balas HANYA dengan JSON berikut, tanpa penjelasan atau teks lain:
{{"top": ["doc_id_1/node_id_paling_relevan", "doc_id_2/node_id_kedua", "...", "doc_id_N/node_id_ke_{top_k}"]}}

Aturan ketat:
- "top" HARUS berisi tepat {top_k} string ref (atau lebih sedikit kalau {n} < {top_k})
- Setiap nilai HARUS sama persis dengan field "ref" salah satu kandidat di input
  (format "doc_id/node_id", contoh "uu-3-2024/pasal_5")
- Tidak boleh ada duplikat
- JANGAN buat ref baru atau modifikasi format (no hallucination)
- Urutan menentukan ranking (index 0 = paling relevan)
- Pertimbangkan ISI summary, sumber UU (doc_title), dan navigation_path
- JANGAN tambahkan field lain (tidak boleh "thinking", "reasoning", "ranking" full, dll)
- Kembalikan HANYA satu objek JSON dengan field "top" saja
"""


def _select_top_from_chunk(query: str, chunk: list[dict], top_k: int) -> list[dict]:
    """Ask the LLM to pick the top candidates from one chunk.

    Each candidate is identified by a compound ref (doc_id/node_id).
    Invalid or missing refs are padded with input order so downstream
    slicing is stable.

    Args:
        query: Legal question in Indonesian.
        chunk: List of leaf node dicts for this chunk.
        top_k: Number of top candidates to select.

    Returns:
        Reordered list of leaf dicts, best first.
    """
    candidates_for_prompt = [
        {
            "ref": f"{leaf['doc_id']}/{leaf['node_id']}",
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "node_id": leaf["node_id"],
            "title": leaf.get("title", ""),
            "navigation_path": leaf.get("navigation_path", ""),
            "summary": leaf.get("summary", ""),
        }
        for leaf in chunk
    ]

    prompt = _build_rank_prompt(query, candidates_for_prompt, top_k)
    result = llm_call(prompt, max_completion_tokens=4096)

    raw_top = result.get("top", []) if isinstance(result, dict) else []
    if not raw_top and isinstance(result, dict):
        raw_top = result.get("ranking", []) or []

    valid_refs = {c["ref"] for c in candidates_for_prompt}
    seen: set[str] = set()
    cleaned_refs: list[str] = []
    for ref in raw_top:
        if isinstance(ref, str) and ref in valid_refs and ref not in seen:
            cleaned_refs.append(ref)
            seen.add(ref)
    # Pad with input order so downstream slice [:survivors] is stable.
    for c in candidates_for_prompt:
        if c["ref"] not in seen:
            cleaned_refs.append(c["ref"])
            seen.add(c["ref"])

    leaf_by_ref = {f"{leaf['doc_id']}/{leaf['node_id']}": leaf for leaf in chunk}
    return [leaf_by_ref[ref] for ref in cleaned_refs if ref in leaf_by_ref]


def flat_search(query: str, leaves: list[dict],
                window_size: int = LISTWISE_WINDOW,
                survivors_per_chunk: int = LISTWISE_SURVIVORS,
                top_k: int = 10, verbose: bool = True) -> dict:
    """Run a multi-round elimination tournament over all leaf nodes.

    Each round shuffles the pool, splits into chunks of window_size,
    and advances the top survivors_per_chunk from each chunk. The final
    round ranks survivors twice with different shuffles and merges via
    Borda count.

    Args:
        query: Legal question in Indonesian.
        leaves: All leaf nodes from all documents.
        window_size: Maximum candidates per LLM call.
        survivors_per_chunk: Number of candidates kept per chunk per round.
        top_k: Final number of leaves to return.
        verbose: Print progress per round.

    Returns:
        Dict with ranked_node_ids, candidates_shown, rounds info,
        and total_llm_calls.
    """
    candidates = list(leaves)
    rounds_info: list[dict] = []
    total_calls = 0
    rng = random.Random(_query_seed(query))

    if verbose:
        print(f"\n[LLM Flat] Starting with {len(candidates)} leaves")

    while len(candidates) > window_size:
        rng.shuffle(candidates)
        chunks = [candidates[i:i + window_size]
                  for i in range(0, len(candidates), window_size)]
        new_candidates: list[dict] = []
        for chunk in chunks:
            picked = _select_top_from_chunk(query, chunk, top_k=survivors_per_chunk)
            new_candidates.extend(picked[:survivors_per_chunk])
            total_calls += 1

        rounds_info.append({
            "round": len(rounds_info) + 1,
            "chunks": len(chunks),
            "input_size": sum(len(c) for c in chunks),
            "survivors": len(new_candidates),
            "calls": len(chunks),
        })
        if verbose:
            print(f"  Round {len(rounds_info)}: "
                  f"{len(chunks)} chunks of {window_size} -> "
                  f"{len(new_candidates)} survivors")

        candidates = new_candidates

    if len(candidates) > 1:
        pool_a = list(candidates)
        rng.shuffle(pool_a)
        picked_a = _select_top_from_chunk(query, pool_a, top_k=top_k)
        total_calls += 1

        pool_b = list(candidates)
        rng.shuffle(pool_b)
        picked_b = _select_top_from_chunk(query, pool_b, top_k=top_k)
        total_calls += 1

        refs_a = [f"{leaf['doc_id']}/{leaf['node_id']}" for leaf in picked_a]
        refs_b = [f"{leaf['doc_id']}/{leaf['node_id']}" for leaf in picked_b]
        merged_refs = _borda_merge(refs_a, refs_b, top_k=top_k)
        leaf_by_ref = {f"{leaf['doc_id']}/{leaf['node_id']}": leaf for leaf in candidates}
        candidates = [leaf_by_ref[ref] for ref in merged_refs if ref in leaf_by_ref]

        rounds_info.append({
            "round": len(rounds_info) + 1,
            "chunks": 2,
            "input_size": len(pool_a),
            "survivors": len(candidates),
            "calls": 2,
            "final": True,
            "merge": "borda",
        })
        if verbose:
            print(f"  Final round: 2 shuffled passes over {len(pool_a)} "
                  f"candidates -> Borda merge top-{top_k}")

    final = candidates[:top_k]

    return {
        "ranked_node_ids": [leaf["node_id"] for leaf in final],
        "candidates_shown": len(leaves),
        "validated_ranking_length": len(final),
        "rounds": rounds_info,
        "total_llm_calls": total_calls,
    }


def retrieve(query: str, window_size: int = LISTWISE_WINDOW,
             survivors_per_chunk: int = LISTWISE_SURVIVORS,
             top_k: int = 10, verbose: bool = True) -> dict:
    """Run the full LLM flat retrieval pipeline.

    Loads all leaf nodes and ranks them through a chunked elimination
    tournament.

    Args:
        query: Legal question in Indonesian.
        window_size: Maximum candidates per LLM call.
        survivors_per_chunk: Number of candidates kept per chunk per round.
        top_k: Final number of leaves to return.
        verbose: Print progress to stdout.

    Returns:
        Dict with query, strategy, search results, sources, and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'=' * 60}")
        print(f"Query: {query}")
        print(f"Strategy: llm-flat "
              f"(window_size={window_size}, survivors={survivors_per_chunk}, "
              f"top_k={top_k})")
        print(f"{'=' * 60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    search_result = flat_search(query, leaves,
                                window_size=window_size,
                                survivors_per_chunk=survivors_per_chunk,
                                top_k=top_k, verbose=verbose)
    ranked_ids = search_result.get("ranked_node_ids", [])
    steps["flat_search"] = step_metrics(t_step, snap)

    if not ranked_ids:
        return {"query": query, "strategy": "llm-flat",
                "error": "LLM returned no ranking"}

    leaf_map = {leaf["node_id"]: leaf for leaf in leaves}
    ranked_results = [leaf_map[nid] for nid in ranked_ids if nid in leaf_map]

    if not ranked_results:
        return {"query": query, "strategy": "llm-flat",
                "node_ids": ranked_ids, "error": "Ranked nodes not found in corpus"}

    sources = []
    for pos, r in enumerate(ranked_results):
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r.get("title", ""),
            "navigation_path": r.get("navigation_path", ""),
            "rerank_position": pos,
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "llm-flat",
        "corpus_size": len(leaves),
        "flat_search": search_result,
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": steps},
    }

    save_log(result)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        print(f"{'=' * 60}")

    return result


def main():
    """CLI entry point for LLM flat retrieval."""
    ap = argparse.ArgumentParser(
        description="LLM flat (chunked listwise) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--window_size", type=int, default=LISTWISE_WINDOW,
                    help=f"Max candidates per LLM call (default: {LISTWISE_WINDOW})")
    ap.add_argument("--survivors_per_chunk", type=int, default=LISTWISE_SURVIVORS,
                    help=f"Top-K kept per chunk per round (default: {LISTWISE_SURVIVORS})")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of final results to return (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query,
                      window_size=args.window_size,
                      survivors_per_chunk=args.survivors_per_chunk,
                      top_k=args.top_k)
    print(f"\n{'-' * 60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()
