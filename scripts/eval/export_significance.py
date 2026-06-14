"""Export the statistical comparisons reported in the thesis to JSON artifacts.

Recomputes every paired test cited in chapter 4 from the per-query records
and writes the full outputs to disk. All tests are deterministic given the
fixed seed, so re-running this script reproduces the stored artifacts.

Outputs:
    data/eval_runs/stage3_test/significance_test.json
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

from scripts.eval.core.significance import compare_paired

EVAL_RUNS = REPO_ROOT / "data" / "eval_runs"
B = 10_000
SEED = 42
TEST_METRICS = ["map@10", "recall@2", "recall@10", "mrr@10", "hit@1"]


def load_records(rel_path: str) -> dict[str, dict]:
    """Load one records file into a query_id keyed dict."""
    path = EVAL_RUNS / rel_path
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["query_id"]] = rec
    return out


def paired(a: dict[str, dict], b: dict[str, dict], metric: str,
           multi_gold: bool = False) -> dict:
    """Run compare_paired on the shared queries, errors counted as zero.

    With multi_gold, restrict to queries whose gold set spans at least two
    nodes, the multihop subset that did not collapse at this granularity.
    """
    qids = sorted(set(a) & set(b))
    if multi_gold:
        qids = [q for q in qids if (a[q].get("num_relevant") or 0) >= 2]
    av = [(a[q].get(metric) or 0) for q in qids]
    bv = [(b[q].get(metric) or 0) for q in qids]
    result = compare_paired(av, bv, B=B, seed=SEED)
    result["n_queries"] = len(qids)
    result["metric"] = metric
    if multi_gold:
        result["subset"] = "multi-gold (gold set spans two nodes)"
    return result


def main() -> int:
    """Recompute and store every statistical comparison cited in the thesis."""
    stamp = datetime.now().isoformat(timespec="seconds")

    print("Loading test-partition records...")
    vl_test = load_records(
        "stage3_test/rq4_test_hybrid_tree/records/hybrid-tree__pasal.jsonl")
    vec_test = load_records(
        "stage3_test/rq4_test_v2m3_qe/records/"
        "vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl")

    test_out = {
        "completed_at": stamp,
        "note": "Paired tests for the Stage 3 comparison, "
                "vectorless minus vector, errors counted as zero.",
        "side_a": "optimized hybrid-tree, pasal, test",
        "side_b": "optimized BGE-M3 + BGE v2 M3 + QE, pasal, test",
        "B": B,
        "seed": SEED,
        "comparisons": [paired(vl_test, vec_test, m) for m in TEST_METRICS],
    }
    test_out["comparisons"].append(
        paired(vl_test, vec_test, "recall@2", multi_gold=True))
    test_path = EVAL_RUNS / "stage3_test" / "significance_test.json"
    test_path.write_text(
        json.dumps(test_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Wrote {test_path}")

    print("Loading development-partition records...")
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

    dev_comparisons = []
    for name, side_b in dev_sides.items():
        entry = paired(ht, side_b, "map@10")
        entry["side_a"] = "hybrid-tree, pasal, dev"
        entry["side_b"] = f"{name}, pasal, dev"
        dev_comparisons.append(entry)
    pair = paired(bge, e5, "map@10")
    pair["side_a"] = "BGE-M3 + BGE v2 M3, pasal, dev"
    pair["side_b"] = "E5 + BGE v2 M3, pasal, dev"
    dev_comparisons.append(pair)
    third = paired(bge, e5_qwen, "map@10")
    third["side_a"] = "BGE-M3 + BGE v2 M3, pasal, dev"
    third["side_b"] = "E5 + Qwen3 0.6B, pasal, dev"
    dev_comparisons.append(third)

    dev_out = {
        "completed_at": stamp,
        "note": "Paired tests behind the tied-top-tier readings of the "
                "Stage 1 sections, errors counted as zero.",
        "B": B,
        "seed": SEED,
        "comparisons": dev_comparisons,
    }
    dev_path = EVAL_RUNS / "significance_dev_ties.json"
    dev_path.write_text(
        json.dumps(dev_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Wrote {dev_path}")

    print("\nSummary against thesis numbers:")
    for c in test_out["comparisons"]:
        label = c["metric"] + (", multi-gold" if c.get("subset") else "")
        print(f"  test {label}: diff={c['mean_diff']:+.4f} "
              f"p={c['paired_randomization']['p_value']:.4f} "
              f"d={c['cohens_d']['d']:+.3f} n={c['n_queries']}")
    for c in dev_comparisons:
        print(f"  dev {c['side_b']}: diff={c['mean_diff']:+.4f} "
              f"p={c['paired_randomization']['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
