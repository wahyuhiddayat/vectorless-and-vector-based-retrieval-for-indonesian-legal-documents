"""Export the held-out hypothesis tests reported in the thesis to JSON.

Recomputes the eight pre-specified tests H1 to H8 on the sealed test partition
from the per-query records, applies the Holm-Bonferroni correction across the
family, and writes one file per hypothesis. Also recomputes the Stage 1 tie
tests behind the development-partition readings. All tests are deterministic
given the fixed seed, so re-running this script reproduces the stored artifacts.

H1 to H7 are paired comparisons on per-query MAP@10, all two-sided, because an
opposite-direction result would be reportable rather than equivalent to a null.
H8 is a two-sample difference-of-differences contrast on the per-query paradigm
margin, split by the two-anchor multihop subset, computed on R@2 because that is
the metric the two-anchor ground truth was built for, and reported with the
contrast CI, Cliff's delta, and the paired margin within each group.

The per-hypothesis files share one Holm-Bonferroni correction, so they are
written together in a single run. A guard refuses to write unless all eight were
computed in the same pass, which keeps each file's adjusted p-value valid.

Outputs:
    data/eval_runs/stage3_test/hypotheses/H1.json ... H8.json
    data/eval_runs/stage3_test/hypotheses/_family.json
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
    cliffs_delta,
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
    "A": "stage3_test/test_bm25/records/bm25-flat__pasal.jsonl",
    "B": "stage3_test/test_hybrid_tree_default/records/hybrid-tree__pasal.jsonl",
    "C": "stage3_test/test_bgem3/records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl",
    "D": "stage3_test/test_bgem3/records/vector-dense__pasal__bge-m3__none.jsonl",
    "E": "stage3_test/test_nusabert/records/vector-dense__pasal__all-nusabert-large-v4__bge-reranker-v2-m3.jsonl",
    "F": "stage3_test/rq4_test_hybrid_tree/records/hybrid-tree__pasal.jsonl",
    "G": "stage3_test/rq4_test_v2m3_qe/records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl",
}

# Paired hypotheses, left minus right. All two-sided, see the module docstring,
# an opposite-direction result is reportable rather than equivalent to a null.
PAIRED_HYPOTHESES = [
    ("H1", "B", "A", "two-sided", "best vectorless vs BM25 baseline"),
    ("H2", "C", "A", "two-sided", "best vector vs BM25 baseline"),
    ("H3", "C", "D", "two-sided", "reranker vs no reranker, bge-m3 at pasal"),
    ("H4", "E", "C", "two-sided", "specialized NusaBERT vs multilingual bge-m3"),
    ("H5", "F", "B", "two-sided", "improved vs unimproved best vectorless"),
    ("H6", "G", "C", "two-sided", "improved vs unimproved best vector"),
    ("H7", "F", "G", "two-sided", "improved vectorless vs improved vector"),
]

# H8 is computed on R@2, not the MAP@10 used for the paired hypotheses.
H8_METRIC = "recall@2"


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


def component_pair(left: dict[str, dict], right: dict[str, dict],
                   qids: list[str], metric: str) -> dict:
    """Paired left-minus-right comparison on one query group, for a given metric."""
    av = [metric_value(left[q], metric) for q in qids]
    bv = [metric_value(right[q], metric) for q in qids]
    res = compare_paired(av, bv, alternative="two-sided", B=B, seed=SEED)
    res["metric_left"] = sum(av) / len(av)
    res["metric_right"] = sum(bv) / len(bv)
    res["n_queries"] = len(qids)
    return res


def multihop_interaction(records: dict[str, dict[str, dict]],
                         vectorless: str, vector: str, metric: str = H8_METRIC) -> dict:
    """H8 contrast, paradigm margin on the two-anchor multihop subset vs the rest.

    Computed on R@2, the metric the two-anchor ground truth was built for. The
    test is a two-sample randomization on the per-query margin, two-sided. Effect
    sizes, the contrast with a bootstrap CI, Cliff's delta between the subset and
    complement margins, and the paired vectorless-minus-vector comparison within
    each group so the reader sees where the gap lives.
    """
    vl, ve = records[vectorless], records[vector]
    qids = sorted(set(vl) & set(ve))
    margins = {q: metric_value(vl[q], metric) - metric_value(ve[q], metric) for q in qids}
    is_multi = {
        q: vl[q].get("query_type") == "multihop" and (vl[q].get("num_relevant") or 0) == 2
        for q in qids
    }
    subset_q = [q for q in qids if is_multi[q]]
    complement_q = [q for q in qids if not is_multi[q]]
    subset = [margins[q] for q in subset_q]
    complement = [margins[q] for q in complement_q]
    ordered = subset + complement
    test = two_sample_randomization(ordered, len(subset), alternative="two-sided", B=B, seed=SEED)
    test["metric"] = metric
    test["bootstrap_ci"] = two_sample_bootstrap_ci(subset, complement, seed=SEED)
    test["cliffs_delta"] = cliffs_delta(subset, complement)
    test["margin_subset"] = sum(subset) / len(subset)
    test["margin_complement"] = sum(complement) / len(complement)
    test["n_subset"] = len(subset)
    test["n_complement"] = len(complement)
    test["within_subset"] = component_pair(vl, ve, subset_q, metric)
    test["within_complement"] = component_pair(vl, ve, complement_q, metric)
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


EXPECTED_HYPOTHESES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]


def main() -> int:
    """Recompute and store the held-out hypothesis tests and the dev tie tests."""
    stamp = datetime.now().isoformat(timespec="seconds")

    print("Computing held-out hypothesis tests H1 to H8...")
    tests = build_held_out()

    missing = [h for h in EXPECTED_HYPOTHESES if h not in tests]
    if missing:
        raise SystemExit(
            f"refusing to write, the family is incomplete, missing {missing}. "
            "The Holm correction is only valid when all eight are computed together."
        )

    out_dir = EVAL_RUNS / "stage3_test" / "hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"completed_at": stamp, "B": B, "seed": SEED}
    for hid in EXPECTED_HYPOTHESES:
        # common goes first so a per-test "metric" key (H8 uses R@2) overrides it.
        payload = {**common, "metric": METRIC, **tests[hid]}
        (out_dir / f"{hid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw_p = {hid: raw_pvalue(tests[hid], hid) for hid in EXPECTED_HYPOTHESES}
    family = {
        **common,
        "note": "Per-hypothesis files corrected together with Holm-Bonferroni. "
                "H1 to H7 are paired randomization on MAP@10, two-sided. H8 is a "
                "two-sample contrast on R@2, two-sided. This file records the "
                "raw and adjusted p-values so the per-file Holm values are auditable.",
        "family_size": len(EXPECTED_HYPOTHESES),
        "raw_p_values": raw_p,
        "holm": {hid: {"p_adjusted": tests[hid]["p_holm"], "reject": tests[hid]["reject_holm"]}
                 for hid in EXPECTED_HYPOTHESES},
    }
    (out_dir / "_family.json").write_text(
        json.dumps(family, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'H1.json'} ... H8.json and _family.json")

    _export_dev_ties(stamp)

    print("\nHeld-out hypothesis summary (MAP@10 for H1 to H7, R@2 for H8, two-sided):")
    header = f"  {'H':3} {'cmp':7} {'delta':>9} {'p_raw':>8} {'p_holm':>8} {'reject':>7}"
    print(header)
    for hid in EXPECTED_HYPOTHESES:
        t = tests[hid]
        delta = t["mean_diff"] if hid != "H8" else t["contrast"]
        print(f"  {hid:3} {t['comparison']:7} {delta:+9.4f} {raw_p[hid]:8.4f} "
              f"{t['p_holm']:8.4f} {str(t['reject_holm']):>7}")
    return 0


def raw_pvalue(test: dict, hid: str) -> float:
    """Raw p-value of one hypothesis, nested for paired tests, top level for H8."""
    if hid == "H8":
        return test["p_value"]
    return test["paired_randomization"]["p_value"]


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
