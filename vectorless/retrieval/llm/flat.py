"""LLM flat retrieval via two-phase selection and ranking.

Phase 1 compresses every leaf node in the corpus into a compact metadata
line (ref, doc_title, navigation_path, truncated summary) and sends the
entire list to the LLM in a single call, asking it to select the top-N
most relevant candidates. If the corpus exceeds the context budget, it is
split into the minimum number of chunks (typically 2 for rincian).

Phase 2 loads full text for the survivors and ranks them in a single
listwise LLM call, identical in structure to hybrid-flat's rerank stage.

This two-phase design parallels hybrid-flat (BM25 filter + LLM rerank)
but replaces BM25 with LLM-on-metadata as the first-stage filter. The
comparison isolates whether LLM semantic reasoning over summaries is a
better flat-corpus filter than BM25 keyword matching over full text.

Usage:
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?" --phase1_survivors 50
"""

import argparse
import hashlib
import json
import math
import time

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_all_leaf_nodes, save_log, validate_llm_ranking,
)


SUMMARY_TRUNCATE = 100
PHASE1_SURVIVORS = 50
CONTEXT_BUDGET = 900_000
TOKENS_PER_CANDIDATE = 120


def _query_seed(query: str) -> int:
    """Derive a deterministic int seed from a query string."""
    return int(hashlib.md5(query.encode("utf-8")).hexdigest()[:8], 16)


def _compress_leaf(leaf: dict) -> str:
    """Build a compact one-line representation of a leaf for Phase 1.

    Format: ref | doc_title | navigation_path | summary (truncated).
    Approximately 30 tokens per leaf.
    """
    ref = f"{leaf['doc_id']}/{leaf['node_id']}"
    doc_title = leaf.get("doc_title", "")
    path = leaf.get("navigation_path", "")
    summary = (leaf.get("summary") or "")[:SUMMARY_TRUNCATE]
    return f"{ref} | {doc_title} | {path} | {summary}"


def _build_phase1_prompt(query: str, compressed_lines: list[str],
                         survivors: int) -> tuple[str, str]:
    """Build system and user messages for Phase 1 selection.

    Returns (system, prompt). System contains the candidate list (cached
    by DeepSeek across queries since the corpus is the same for all queries
    at one granularity). User contains the query (unique per call).
    """
    n = len(compressed_lines)
    candidates_block = "\n".join(
        f"{i+1}. {line}" for i, line in enumerate(compressed_lines)
    )

    system = f"""\
Kamu diberi daftar {n} Pasal dari berbagai Undang-Undang Indonesia.
Setiap baris berformat: ref | judul_UU | lokasi | ringkasan.

Daftar Pasal:
{candidates_block}

Aturan:
- Setiap ref HARUS sama persis dengan yang ada di input (format "doc_id/node_id")
- Tidak boleh ada duplikat
- Urutkan dari paling relevan ke kurang relevan
- Kembalikan HANYA JSON"""

    prompt = f"""\
Pertanyaan: {query}

Pilih {survivors} Pasal PALING RELEVAN untuk menjawab pertanyaan di atas (atau kurang jika tidak ada {survivors} yang relevan).

Balas dengan JSON:
{{"top": ["ref_paling_relevan", "ref_kedua", "..."]}}
"""
    return system, prompt


def _phase1_select(query: str, leaves: list[dict],
                   survivors: int = PHASE1_SURVIVORS,
                   verbose: bool = True) -> list[dict]:
    """Phase 1: select top candidates from the entire corpus using compressed metadata.

    If the corpus exceeds CONTEXT_BUDGET tokens, it is split into the
    minimum number of fixed chunks. Chunk composition is deterministic
    and query-independent (sequential split by leaf index) so DeepSeek
    prefix-caches each chunk's system message across all 357 queries at
    the same granularity.
    """
    compressed = [_compress_leaf(leaf) for leaf in leaves]
    total_tokens = len(leaves) * TOKENS_PER_CANDIDATE

    n_chunks = max(1, math.ceil(total_tokens / CONTEXT_BUDGET))

    if verbose:
        print(f"\n[LLM Flat Phase 1] {len(leaves)} leaves, "
              f"~{total_tokens:,} tokens, {n_chunks} chunk(s)")

    if n_chunks == 1:
        system, prompt = _build_phase1_prompt(query, compressed, survivors)
        result = llm_call(prompt, system=system, max_completion_tokens=4096)
        raw_top = result.get("top", []) if isinstance(result, dict) else []
        if not raw_top and isinstance(result, dict):
            raw_top = result.get("ranking", []) or []
    else:
        chunk_size = math.ceil(len(leaves) / n_chunks)
        raw_top = []
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, len(leaves))
            chunk_compressed = compressed[start:end]
            system, prompt = _build_phase1_prompt(
                query, chunk_compressed, survivors
            )
            result = llm_call(prompt, system=system, max_completion_tokens=4096)
            chunk_top = result.get("top", []) if isinstance(result, dict) else []
            if not chunk_top and isinstance(result, dict):
                chunk_top = result.get("ranking", []) or []
            raw_top.extend(chunk_top)
            if verbose:
                print(f"  Chunk {ci+1}/{n_chunks}: "
                      f"{len(chunk_compressed)} candidates, "
                      f"selected {len(chunk_top)}")

    leaf_by_ref = {f"{l['doc_id']}/{l['node_id']}": l for l in leaves}
    valid_refs = set(leaf_by_ref.keys())
    seen: set[str] = set()
    selected: list[dict] = []
    for ref in raw_top:
        if isinstance(ref, str) and ref in valid_refs and ref not in seen:
            selected.append(leaf_by_ref[ref])
            seen.add(ref)
            if len(selected) >= survivors * n_chunks:
                break

    if verbose:
        print(f"  Selected {len(selected)} survivors for Phase 2")

    return selected


def _build_phase2_prompt(query: str, candidates: list[dict]) -> tuple[str, str]:
    """Build system and user messages for Phase 2 full-text ranking.

    Same structure as hybrid-flat's llm_rerank but with selection_rank
    instead of bm25_rank.
    """
    n = len(candidates)
    candidates_for_prompt = []
    for idx, c in enumerate(candidates):
        ref = f"{c['doc_id']}/{c['node_id']}"
        entry = {
            "selection_rank": idx + 1,
            "ref": ref,
            "doc_id": c["doc_id"],
            "doc_title": c.get("doc_title", ""),
            "title": c.get("title", ""),
            "navigation_path": c.get("navigation_path", ""),
            "text": c.get("text") or "",
        }
        penjelasan = c.get("penjelasan")
        if penjelasan and penjelasan != "Cukup jelas.":
            entry["penjelasan"] = penjelasan
        candidates_for_prompt.append(entry)

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)

    system = f"""\
Kamu diberi pertanyaan hukum dan {n} Pasal kandidat dari berbagai Undang-Undang.
Setiap kandidat memiliki ref (format "doc_id/node_id"), isi teks lengkap (text),
penjelasan resmi (jika ada), dan selection_rank (peringkat dari tahap seleksi awal).

Tugas: Urutkan SELURUH {n} kandidat dari paling relevan ke paling tidak relevan
untuk menjawab pertanyaan.

Pertanyaan: {query}

Aturan:
- "ranking" HARUS berisi tepat {n} ref
- Tidak boleh ada duplikat
- Setiap ref harus muncul di input (tidak boleh hallucinate)
- Urutan menentukan ranking (index 0 = paling relevan)
- Pertimbangkan isi text, penjelasan, doc_title, navigation_path, dan selection_rank
- Kembalikan HANYA JSON"""

    prompt = f"""\
Kandidat Pasal:
{candidates_text}

Balas dalam format JSON:
{{"ranking": ["ref_paling_relevan", "ref_kedua", "..."]}}
"""
    return system, prompt


def _phase2_rank(query: str, survivors: list[dict],
                 top_k: int = 10,
                 verbose: bool = True) -> tuple[list[str], dict]:
    """Phase 2: rank survivors on full text and return top-k refs.

    Returns (ranked_refs, rerank_result_dict).
    """
    if not survivors:
        return [], {}

    system, prompt = _build_phase2_prompt(query, survivors)
    rerank_result = llm_call(prompt, system=system)

    raw_ranking = rerank_result.get("ranking", [])
    valid_refs = {f"{c['doc_id']}/{c['node_id']}" for c in survivors}
    n_hallucinated = sum(1 for r in raw_ranking if r not in valid_refs)

    pseudo_candidates = [
        {"node_id": f"{c['doc_id']}/{c['node_id']}"} for c in survivors
    ]
    ranked_refs = validate_llm_ranking(raw_ranking, pseudo_candidates)

    rerank_result["validated_ranking"] = ranked_refs
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(ranked_refs)
    rerank_result["n_hallucinated"] = n_hallucinated

    if verbose:
        print(f"\n[LLM Flat Phase 2] Ranked {len(ranked_refs)} candidates")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    return ranked_refs[:top_k], rerank_result


def retrieve(query: str,
             phase1_survivors: int = PHASE1_SURVIVORS,
             top_k: int = 10, verbose: bool = True) -> dict:
    """Run the full two-phase LLM flat retrieval pipeline.

    Phase 1 selects candidates from compressed metadata of the entire
    corpus. Phase 2 ranks survivors on full text.

    Args:
        query: Legal question in Indonesian.
        phase1_survivors: Number of candidates to advance from Phase 1.
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
              f"(phase1_survivors={phase1_survivors}, top_k={top_k})")
        print(f"{'=' * 60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    survivors = _phase1_select(query, leaves,
                               survivors=phase1_survivors,
                               verbose=verbose)
    steps["phase1_select"] = step_metrics(t_step, snap)

    if not survivors:
        return {"query": query, "strategy": "llm-flat",
                "error": "Phase 1 returned no candidates"}

    snap = snapshot_counters()
    t_step = time.time()

    ranked_refs, rerank_result = _phase2_rank(
        query, survivors, top_k=top_k, verbose=verbose
    )
    steps["phase2_rank"] = step_metrics(t_step, snap)

    if not ranked_refs:
        return {"query": query, "strategy": "llm-flat",
                "error": "Phase 2 returned no ranking"}

    survivor_map = {f"{s['doc_id']}/{s['node_id']}": s for s in survivors}
    ranked_results = [survivor_map[r] for r in ranked_refs if r in survivor_map]

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
        "phase1_survivors": len(survivors),
        "phase2_rerank": rerank_result,
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": steps},
    }

    save_log(result)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        for step_name, sm in steps.items():
            print(f"  {step_name}: {sm['elapsed_s']:.1f}s, {sm['llm_calls']} calls, "
                  f"{sm['input_tokens']+sm['output_tokens']:,} tokens")
        print(f"{'=' * 60}")

    return result


def main():
    """CLI entry point for LLM flat retrieval."""
    ap = argparse.ArgumentParser(
        description="LLM flat (two-phase selection + ranking) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--phase1_survivors", type=int, default=PHASE1_SURVIVORS,
                    help=f"Candidates advanced from Phase 1 (default: {PHASE1_SURVIVORS})")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of final results to return (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query,
                      phase1_survivors=args.phase1_survivors,
                      top_k=args.top_k)
    print(f"\n{'-' * 60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()
