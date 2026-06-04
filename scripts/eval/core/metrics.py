"""Pure retrieval scoring functions.

Zero I/O, zero side effects. Used by both the vectorless and vector eval
harnesses so metrics are computed identically across paradigms.

  N1. Gold sets may have one or multiple members. Single-gold queries
      collapse recall@k to hit@k and map@k to mrr@k. Multi-gold queries
      make recall@k a true partial-credit metric.

  N4. sibling_hit@k is a diagnostic, not a headline metric. It fires when a
      retrieved node shares the immediate parent of the gold anchor but is
      not the gold itself. Useful to distinguish "right paragraph, wrong
      sub-clause" failures from "completely wrong article" failures.
"""

from __future__ import annotations

import math


# ----------------------------------------------------------------------
# Constants shared across the harness
# ----------------------------------------------------------------------

# Cutoff 2 is included because multihop queries require exactly two gold
# pasals, so recall@2 measures whether both were retrieved at the top.
DEFAULT_CUTOFFS = [1, 2, 3, 5, 10]

GOLD_KEY_BY_GRANULARITY = {
    "pasal": "gold_pasal_node_ids",
    "ayat": "gold_ayat_node_ids",
    "rincian": "gold_rincian_node_ids",
}

SLICE_FIELDS = ["reference_mode", "query_style", "gold_doc_id", "query_type"]


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def safe_mean(values: list[float]) -> float:
    """Return the mean of values, or 0.0 if the list is empty."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def unique_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate a list of strings while preserving first-occurrence order."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


# ----------------------------------------------------------------------
# Hierarchical node-id helpers (for sibling_hit@k diagnostic, see N4)
# ----------------------------------------------------------------------

def parent_prefix(node_id: str) -> str:
    """Strip the trailing label-value pair from a node_id to get its parent.

    Node ids follow the LLM-parser readable format `pasal_3_ayat_2_huruf_a`
    where each level adds two segments (label, value). The immediate parent
    is obtained by removing the last two segments. Returns "" when the input
    is already at the top level (e.g. `pasal_3`) and has no parent.
    """
    if not node_id:
        return ""
    parts = node_id.split("_")
    if len(parts) < 4:
        return ""
    return "_".join(parts[:-2])


def sibling_hit_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return 1.0 if any node in top-k is a sibling of any gold node, else 0.0.

    A sibling shares the immediate parent of the gold but is not the gold
    itself. Diagnostic only, see metrics module docstring N4.
    """
    if not relevant_ids or not ranked_ids:
        return 0.0
    gold_parents = {p for g in relevant_ids if (p := parent_prefix(g))}
    if not gold_parents:
        return 0.0
    for retrieved in ranked_ids[:k]:
        if retrieved in relevant_ids:
            continue
        parent = parent_prefix(retrieved)
        if parent and parent in gold_parents:
            return 1.0
    return 0.0


def sibling_failure_stats(records: list[dict], cutoffs: list[int]) -> dict:
    """Count failure cases per cutoff and how many had a sibling near-miss."""
    out: dict[str, dict] = {}
    for k in cutoffs:
        failures = [
            r for r in records
            if not r.get("error") and float(r.get(f"hit@{k}", 0.0)) == 0.0
        ]
        with_sibling = [
            r for r in failures
            if float(r.get(f"sibling_hit@{k}", 0.0)) > 0.0
        ]
        out[f"k={k}"] = {
            "n_failures": len(failures),
            "n_with_sibling_near_miss": len(with_sibling),
            "near_miss_rate": (
                float(len(with_sibling)) / len(failures) if failures else 0.0
            ),
        }
    return out


# ----------------------------------------------------------------------
# Retrieval scoring
# ----------------------------------------------------------------------

def score_ranked_retrieval(
    ranked_ids: list[str],
    relevant_ids: set[str],
    cutoffs: list[int],
) -> dict:
    """Score one ranked retrieval result against the gold set.

    Outputs cover every cutoff plus a few rank-distribution descriptive stats.
    """
    ranked = unique_preserve_order(ranked_ids)
    relevant = set(relevant_ids)
    hit_positions = [idx for idx, node_id in enumerate(ranked, start=1) if node_id in relevant]
    first_rank = hit_positions[0] if hit_positions else None

    out = {
        "num_retrieved": len(ranked),
        "num_relevant": len(relevant),
        "first_relevant_rank": first_rank,
        "exact_top1_hit": bool(ranked) and ranked[0] in relevant,
        # Reciprocal rank without a cutoff.
        "full_reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
    }

    for k in cutoffs:
        top_k = ranked[:k]
        retrieved_relevant = [node_id for node_id in top_k if node_id in relevant]
        hit = bool(retrieved_relevant)
        n_relevant_in_top_k = len(set(retrieved_relevant))
        recall = (n_relevant_in_top_k / len(relevant)) if relevant else 0.0
        precision = (n_relevant_in_top_k / k) if k else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        dcg = 0.0
        for rank, node_id in enumerate(top_k, start=1):
            if node_id in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        num_hits = 0
        precision_sum = 0.0
        for rank, node_id in enumerate(top_k, start=1):
            if node_id in relevant:
                num_hits += 1
                precision_sum += num_hits / rank
        ap = precision_sum / len(relevant) if relevant else 0.0

        # Per-cutoff MRR.
        mrr_k = (1.0 / first_rank) if first_rank and first_rank <= k else 0.0

        out[f"hit@{k}"] = float(hit)
        out[f"recall@{k}"] = recall
        out[f"precision@{k}"] = precision
        out[f"f1@{k}"] = f1
        out[f"ndcg@{k}"] = ndcg
        out[f"dcg@{k}"] = dcg
        out[f"map@{k}"] = ap
        out[f"mrr@{k}"] = mrr_k
        out[f"sibling_hit@{k}"] = sibling_hit_at_k(ranked, relevant, k)

    return out


# ----------------------------------------------------------------------
# Tree-paradigm stage-1 vs stage-2 attribution diagnostics
# ----------------------------------------------------------------------

def compute_doc_pick_diagnostics(
    retrieved_sources: list[dict],
    picked_doc_ids: list[str],
    gold_doc_ids: list[str] | set[str],
    relevant_ids: set[str],
    cutoffs: list[int],
) -> dict:
    """Attribute tree-method failures to stage-1 (doc-pick) versus stage-2.

    For tree retrieval, a failed query can fail either because doc-pick
    missed the gold doc (stage-1) or because within-doc navigation missed
    the gold node even after doc-pick was correct (stage-2). Emits
    doc_pick_hit, doc_pick_hit_count, and within_doc_{hit,recall,mrr}@k.
    """
    gold_set = set(gold_doc_ids) if gold_doc_ids else set()
    picked = list(picked_doc_ids or [])
    doc_pick_hit_count = sum(1 for d in picked if d in gold_set)
    doc_pick_hit = 1.0 if doc_pick_hit_count > 0 else 0.0

    within_ids: list[str] = []
    if gold_set:
        for src in retrieved_sources:
            if src.get("doc_id") in gold_set:
                nid = src.get("node_id")
                if nid:
                    within_ids.append(nid)
    within_metrics = score_ranked_retrieval(within_ids, relevant_ids, cutoffs)

    out: dict = {
        "doc_pick_hit": doc_pick_hit,
        "doc_pick_hit_count": doc_pick_hit_count,
        "doc_pick_count": len(picked),
        "within_doc_num_retrieved": within_metrics["num_retrieved"],
        "within_doc_first_relevant_rank": within_metrics["first_relevant_rank"],
    }
    for k in cutoffs:
        out[f"within_doc_hit@{k}"] = within_metrics[f"hit@{k}"]
        out[f"within_doc_recall@{k}"] = within_metrics[f"recall@{k}"]
        out[f"within_doc_mrr@{k}"] = within_metrics[f"mrr@{k}"]
    return out


# ----------------------------------------------------------------------
# Rank-distribution descriptive stats over a population of records
# ----------------------------------------------------------------------

def rank_distribution_stats(records: list[dict]) -> dict:
    """Mean and median rank across records that hit, plus hit-rate context.

    MRR weighs top ranks heavily so the average can be misleading. Reporting
    mean and median rank on the hit-only subset gives a complementary view of
    "where does the gold land when we find it".
    """
    ranks = [
        int(row["first_relevant_rank"])
        for row in records
        if row.get("first_relevant_rank")
    ]
    if not ranks:
        return {
            "n_hits_anywhere": 0,
            "n_total": len(records),
            "mean_rank_on_hit": 0.0,
            "median_rank_on_hit": 0.0,
            "max_rank_on_hit": 0,
        }
    sorted_ranks = sorted(ranks)
    n = len(sorted_ranks)
    median = (
        sorted_ranks[n // 2]
        if n % 2 == 1
        else 0.5 * (sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2])
    )
    return {
        "n_hits_anywhere": n,
        "n_total": len(records),
        "mean_rank_on_hit": float(sum(sorted_ranks) / n),
        "median_rank_on_hit": float(median),
        "max_rank_on_hit": int(sorted_ranks[-1]),
    }


# ----------------------------------------------------------------------
# Self-test (called from CLI via --self-test-metrics)
# ----------------------------------------------------------------------

def run_self_test() -> None:
    """Run built-in correctness checks for all scoring functions."""
    def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
        if abs(actual - expected) > tol:
            raise AssertionError(f"expected {expected}, got {actual}")

    cutoffs = [1, 3, 5, 10]

    row = score_ranked_retrieval(["a", "b"], {"a"}, cutoffs)
    assert_close(row["hit@1"], 1.0)
    assert_close(row["recall@1"], 1.0)
    assert_close(row["mrr@1"], 1.0)
    assert_close(row["mrr@10"], 1.0)
    assert_close(row["map@10"], 1.0)
    assert_close(row["ndcg@10"], 1.0)
    assert_close(row["full_reciprocal_rank"], 1.0)

    row = score_ranked_retrieval(["a", "b", "c"], {"a", "c"}, cutoffs)
    assert_close(row["hit@1"], 1.0)
    assert_close(row["recall@1"], 0.5)
    assert_close(row["recall@3"], 1.0)
    assert_close(row["mrr@10"], 1.0)
    assert_close(row["map@10"], (1.0 + (2 / 3)) / 2)

    row = score_ranked_retrieval(["x", "y", "z"], {"a"}, cutoffs)
    assert_close(row["hit@10"], 0.0)
    assert_close(row["recall@10"], 0.0)
    assert_close(row["mrr@10"], 0.0)
    assert_close(row["map@10"], 0.0)
    assert_close(row["ndcg@10"], 0.0)
    assert_close(row["full_reciprocal_rank"], 0.0)

    row = score_ranked_retrieval(["x", "y", "a"], {"a"}, cutoffs)
    assert_close(row["hit@1"], 0.0)
    assert_close(row["hit@3"], 1.0)
    assert_close(row["mrr@1"], 0.0)
    assert_close(row["mrr@3"], 1 / 3)
    assert_close(row["mrr@10"], 1 / 3)
    assert_close(row["full_reciprocal_rank"], 1 / 3)

    # full_reciprocal_rank > 0 even when gold is past cutoff k=10
    long_list = [f"x{i}" for i in range(11)] + ["a"]
    row = score_ranked_retrieval(long_list, {"a"}, cutoffs)
    assert_close(row["mrr@10"], 0.0)
    assert_close(row["full_reciprocal_rank"], 1 / 12)

    # Rank distribution stats over a small population
    fake_records = [
        {"first_relevant_rank": 1},
        {"first_relevant_rank": 3},
        {"first_relevant_rank": 5},
        {"first_relevant_rank": None},
    ]
    rstat = rank_distribution_stats(fake_records)
    assert_close(rstat["mean_rank_on_hit"], 3.0)
    assert_close(rstat["median_rank_on_hit"], 3.0)
    if rstat["n_hits_anywhere"] != 3 or rstat["n_total"] != 4:
        raise AssertionError("rank_distribution_stats hit count mismatch")

    # parent_prefix correctness across all node-id shapes
    if parent_prefix("pasal_3_ayat_2_huruf_a") != "pasal_3_ayat_2":
        raise AssertionError("parent_prefix(huruf-leaf) failed")
    if parent_prefix("pasal_3_ayat_2") != "pasal_3":
        raise AssertionError("parent_prefix(ayat) failed")
    if parent_prefix("pasal_3") != "":
        raise AssertionError("parent_prefix(pasal-only) should return empty string")
    if parent_prefix("pasal_I_angka_2_pasal_7_ayat_1_huruf_a") != "pasal_I_angka_2_pasal_7_ayat_1":
        raise AssertionError("parent_prefix(amendment) failed")

    # sibling_hit@k semantics
    gold = {"pasal_3_ayat_2_huruf_a"}
    # exact gold in top-k, no sibling -> 0
    if sibling_hit_at_k(["pasal_3_ayat_2_huruf_a"], gold, 5) != 0.0:
        raise AssertionError("sibling_hit should be 0 when only gold present")
    # sibling in top-k -> 1
    if sibling_hit_at_k(["pasal_3_ayat_2_huruf_b", "pasal_4"], gold, 5) != 1.0:
        raise AssertionError("sibling_hit should fire on sibling huruf")
    # cousin (different parent) -> 0
    if sibling_hit_at_k(["pasal_3_ayat_3_huruf_a"], gold, 5) != 0.0:
        raise AssertionError("sibling_hit should not fire on cousin (different ayat)")
    # gold and sibling both present -> 1
    if sibling_hit_at_k(["pasal_3_ayat_2_huruf_a", "pasal_3_ayat_2_huruf_b"], gold, 5) != 1.0:
        raise AssertionError("sibling_hit should fire when sibling co-occurs with gold")
    # sibling beyond cutoff -> 0
    long_list = ["x"] * 10 + ["pasal_3_ayat_2_huruf_b"]
    if sibling_hit_at_k(long_list, gold, 10) != 0.0:
        raise AssertionError("sibling_hit should respect cutoff k=10")
    if sibling_hit_at_k(long_list, gold, 11) != 1.0:
        raise AssertionError("sibling_hit should fire at k=11 when sibling at rank 11")

    # sibling_hit@k integrated into score_ranked_retrieval
    row = score_ranked_retrieval(
        ["pasal_3_ayat_2_huruf_b", "pasal_3_ayat_2_huruf_a"], gold, [1, 3, 5, 10]
    )
    if row["sibling_hit@1"] != 1.0:
        raise AssertionError("sibling_hit@1 should be 1.0 when sibling at rank 1")
    if row["hit@1"] != 0.0:
        raise AssertionError("hit@1 should be 0.0 when sibling not gold at rank 1")
    if row["hit@5"] != 1.0:
        raise AssertionError("hit@5 should be 1.0 when gold at rank 2")

    # sibling_failure_stats aggregation
    fake = [
        {"hit@5": 0.0, "sibling_hit@5": 1.0},  # failure with sibling near-miss
        {"hit@5": 0.0, "sibling_hit@5": 0.0},  # failure no near-miss
        {"hit@5": 1.0, "sibling_hit@5": 0.0},  # not a failure
        {"hit@5": 0.0, "sibling_hit@5": 0.0, "error": "boom"},  # error excluded
    ]
    fstats = sibling_failure_stats(fake, [5])
    if fstats["k=5"]["n_failures"] != 2:
        raise AssertionError("sibling_failure_stats failure count wrong")
    if fstats["k=5"]["n_with_sibling_near_miss"] != 1:
        raise AssertionError("sibling_failure_stats near-miss count wrong")
    assert_close(fstats["k=5"]["near_miss_rate"], 0.5)
