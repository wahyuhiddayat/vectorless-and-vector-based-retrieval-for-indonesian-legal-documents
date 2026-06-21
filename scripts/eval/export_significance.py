"""Export the held-out hypothesis tests reported in the thesis to JSON.

Recomputes the eight pre-specified tests H1 to H8 on the sealed test partition
from the per-query records, applies the Holm-Bonferroni correction across the
family, and writes the result to disk. Also recomputes the Stage 1 tie tests
behind the development-partition readings. All tests are deterministic given the
fixed seed, so re-running this script reproduces the stored artifacts.

H1 to H7 are paired comparisons on per-query MAP@10. H8 is a two-sample
difference-of-differences contrast on the per-query paradigm margin, split by
the two-anchor multihop subset. The decision metric throughout is MAP@10.

Outputs:
    data/eval_runs/stage3_test/significance_hypotheses.json
    data/eval_runs/significance_dev_ties.json

Usage:
    python scripts/eval/export_significance.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core.significance import (
    compare_paired,
    holm_bonferroni,
    two_sample_bootstrap_ci,
    two_sample_randomization,
)

EVAL_RUNS = REPO_ROOT / "data" / "eval_runs"
B = 10_000
SEED = 42
METRIC = "map@10"

# Held-out runs keyed by the short labels used in the analysis plan.
HELD_OUT = {
    "A": "test_bm25/records/bm25-flat__pasal.jsonl",
    "B": "test_hybrid_tree_default/records/hybrid-tree__pasal.jsonl",
    "C": "test_bgem3/records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl",
    "D": "test_bgem3/records/vector-dense__pasal__bge-m3__none.jsonl",
    "E": "test_nusabert/records/vector-dense__pasal__all-nusabert-large-v4__bge-reranker-v2-m3.jsonl",
    "F": "stage3_test/rq4_test_hybrid_tree/records/hybrid-tree__pasal.jsonl",
    "G": "stage3_test/rq4_test_v2m3_qe/records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl",
}

# Paired hypotheses, left minus right, with the pre-registered alternative.
PAIRED_HYPOTHESES = [
    ("H1", "B", "A", "greater", "best vectorless vs BM25 baseline"),
    ("H2", "C", "A", "greater", "best vector vs BM25 baseline"),
    ("H3", "C", "D", "greater", "reranker vs no reranker, bge-m3 at pasal"),
    ("H4", "E", "C", "two-sided", "specialized NusaBERT vs multilingual bge-m3"),
    ("H5", "F", "B", "greater", "improved vs unimproved best vectorless"),
    ("H6", "G", "C", "greater", "improved vs unimproved best vector"),
    ("H7", "F", "G", "two-sided", "improved vectorless vs improved vector"),
]


def load_records(rel_path: str) -> dict[str, dict]:
    """Load one records file into a query_id keyed dict."""
    path = EVAL_RUNS / rel_path
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["query_id"]] = rec
    return out


def metric_value(rec: dict, metric: str = METRIC) -> float:
    """Per-query metric, with errors counted as zero."""
    return float(rec.get(metric) or 0.0)


def paired_hypothesis(records: dict[str, dict[str, dict]], left: str, right: str,
                      alternative: str) -> dict:
    """Run compare_paired on MAP@10 over the queries shared by two runs."""
    a, b = records[left], records[right]
    qids = sorted(set(a) & set(b))
    av = [metric_value(a[q]) for q in qids]
    bv = [metric_value(b[q]) for q in qids]
    result = compare_paired(av, bv, alternative=alternative, B=B, seed=SEED)
    result["map_left"] = sum(av) / len(av)
    result["map_right"] = sum(bv) / len(bv)
    result["n_queries"] = len(qids)
    return result


def multihop_interaction(records: dict[str, dict[str, dict]],
                         vectorless: str, vector: str) -> dict:
    """H8 contrast, paradigm margin on the two-anchor multihop subset vs the rest."""
    vl, ve = records[vectorless], records[vector]
    qids = sorted(set(vl) & set(ve))
    margins = {q: metric_value(vl[q]) - metric_value(ve[q]) for q in qids}
    is_multi = {
        q: vl[q].get("query_type") == "multihop" and (vl[q].get("num_relevant") or 0) == 2
        for q in qids
    }
    subset = [margins[q] for q in qids if is_multi[q]]
    complement = [margins[q] for q in qids if not is_multi[q]]
    ordered = subset + complement
    test = two_sample_randomization(ordered, len(subset), alternative="greater", B=B, seed=SEED)
    ci = two_sample_bootstrap_ci(subset, complement, seed=SEED)
    test["bootstrap_ci"] = ci
    test["margin_subset"] = sum(subset) / len(subset)
    test["margin_complement"] = sum(complement) / len(complement)
    test["n_subset"] = len(subset)
    test["n_complement"] = len(complement)
    return test


def build_held_out() -> dict:
    """Compute the eight hypothesis tests and the Holm correction."""
    records = {name: load_records(rel) for name, rel in HELD_OUT.items()}
    tests: dict[str, dict] = {}
    for hid, left, right, alt, desc in PAIRED_HYPOTHESES:
        entry = paired_hypothesis(records, left, right, alt)
        entry.update({"hypothesis": hid, "comparison": f"{left} vs {right}", "description": desc})
        tests[hid] = entry
    h8 = multihop_interaction(records, "F", "G")
    h8.update({"hypothesis": "H8", "comparison": "F-G margin, multihop vs single",
               "description": "paradigm gap larger on multihop"})
    tests["H8"] = h8

    pvalues = {hid: t["p_value"] if hid == "H8" else t["paired_randomization"]["p_value"]
               for hid, t in tests.items()}
    holm = holm_bonferroni(pvalues)
    for hid, t in tests.items():
        t["p_holm"] = holm[hid]["p_adjusted"]
        t["reject_holm"] = holm[hid]["reject"]
    return tests


def main() -> int:
    """Recompute and store the held-out hypothesis tests and the dev tie tests."""
    stamp = datetime.now().isoformat(timespec="seconds")

    print("Computing held-out hypothesis tests H1 to H8...")
    tests = build_held_out()
    out = {
        "completed_at": stamp,
        "note": "Held-out hypothesis tests on MAP@10, paired for H1 to H7 and a "
                "two-sample contrast for H8, Holm-Bonferroni across the family.",
        "metric": METRIC,
        "B": B,
        "seed": SEED,
        "tests": tests,
    }
    out_path = EVAL_RUNS / "stage3_test" / "significance_hypotheses.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    _export_dev_ties(stamp)

    print("\nHeld-out hypothesis summary (decision metric MAP@10):")
    header = f"  {'H':3} {'cmp':7} {'delta':>9} {'p_raw':>8} {'p_holm':>8} {'reject':>7}"
    print(header)
    for hid in ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]:
        t = tests[hid]
        delta = t["mean_diff"] if hid != "H8" else t["contrast"]
        p_raw = t["p_value"] if hid == "H8" else t["paired_randomization"]["p_value"]
        print(f"  {hid:3} {t['comparison']:7} {delta:+9.4f} {p_raw:8.4f} "
              f"{t['p_holm']:8.4f} {str(t['reject_holm']):>7}")
    return 0


def _export_dev_ties(stamp: str) -> None:
    """Recompute the Stage 1 tie tests behind the development-partition readings."""
    print("Computing development-partition tie tests...")
    s1vl = "stage1_vectorless"
    s1vec = "stage1_vector/run29_20260526_vector_dev_357q_full/records"
    ht = load_records(
        f"{s1vl}/run26_20260524_vectorless_dev_357q_hybrid_tree_deepseek/"
        "records/hybrid-tree__pasal.jsonl")
    dev_sides = {
        "hybrid-flat": load_records(
            f"{s1vl}/run25_20260524_vectorless_dev_357q_hybrid_flat_deepseek/"
            "records/hybrid-flat__pasal.jsonl"),
        "llm-tree": load_records(
            f"{s1vl}/run27_20260525_vectorless_dev_357q_llm_tree_deepseek/"
            "records/llm-tree__pasal.jsonl"),
        "llm-flat": load_records(
            f"{s1vl}/run28b_20260526_vectorless_dev_357q_llm_flat_pasal_deepseek/"
            "records/llm-flat__pasal.jsonl"),
    }
    bge = load_records(
        f"{s1vec}/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl")
    e5 = load_records(
        f"{s1vec}/vector-dense__pasal__multilingual-e5-large-instruct"
        "__bge-reranker-v2-m3.jsonl")
    e5_qwen = load_records(
        f"{s1vec}/vector-dense__pasal__multilingual-e5-large-instruct"
        "__qwen3-reranker-0.6b.jsonl")

    def dev_pair(a: dict, b: dict, label_a: str, label_b: str) -> dict:
        qids = sorted(set(a) & set(b))
        av = [metric_value(a[q]) for q in qids]
        bv = [metric_value(b[q]) for q in qids]
        entry = compare_paired(av, bv, B=B, seed=SEED)
        entry["side_a"] = label_a
        entry["side_b"] = label_b
        entry["n_queries"] = len(qids)
        return entry

    comparisons = [
        dev_pair(ht, side_b, "hybrid-tree, pasal, dev", f"{name}, pasal, dev")
        for name, side_b in dev_sides.items()
    ]
    comparisons.append(
        dev_pair(bge, e5, "BGE-M3 + BGE v2 M3, pasal, dev", "E5 + BGE v2 M3, pasal, dev"))
    comparisons.append(
        dev_pair(bge, e5_qwen, "BGE-M3 + BGE v2 M3, pasal, dev", "E5 + Qwen3 0.6B, pasal, dev"))

    dev_out = {
        "completed_at": stamp,
        "note": "Paired tests behind the tied-top-tier readings of the Stage 1 "
                "sections, errors counted as zero.",
        "B": B,
        "seed": SEED,
        "comparisons": comparisons,
    }
    dev_path = EVAL_RUNS / "significance_dev_ties.json"
    dev_path.write_text(json.dumps(dev_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dev_path}")


if __name__ == "__main__":
    raise SystemExit(main())
