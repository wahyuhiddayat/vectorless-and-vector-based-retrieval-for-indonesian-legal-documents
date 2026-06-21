"""Multihop contrast analysis between two eval runs on the test partition.

Tests whether the paired per-query metric margin on the two-anchor multihop
subset exceeds the margin on the remaining queries (a difference-of-differences
contrast), and characterizes the subset failures of each side as ordering
failures (all gold provisions retrieved but not ranked inside the cutoff) or
retrieval failures (at least one gold provision absent from the result list).

The permutation test shuffles the subset label across the paired per-query
differences. Errors count as zero, matching the thesis reporting convention.
Reported in Chapter 4 Section 4.5 (run a = optimized hybrid-tree, run b = optimized
BGE-M3 with reranker and query expansion).

Usage:
    # Reproduce the thesis numbers
    python scripts/eval/multihop_contrast.py

    # Explicit runs and metric
    python scripts/eval/multihop_contrast.py \\
        --run-a data/eval_runs/stage3_test/rq4_test_hybrid_tree \\
        --run-b data/eval_runs/stage3_test/rq4_test_v2m3_qe \\
        --metric recall@2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core import io as eval_io  # noqa: E402
from scripts.eval.core.significance import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_RANDOMIZATION_B,
    DEFAULT_SEED,
    two_sample_bootstrap_ci,
    two_sample_randomization,
)


DEFAULT_RUN_A = "data/eval_runs/stage3_test/rq4_test_hybrid_tree"
DEFAULT_RUN_B = "data/eval_runs/stage3_test/rq4_test_v2m3_qe"
DEFAULT_METRIC = "recall@2"


def load_run_records(run_dir: Path) -> dict[str, dict]:
    """Return {query_id: record} for all records in one run directory."""
    records_dir = run_dir / "records"
    if not records_dir.exists():
        raise SystemExit(f"records directory not found, {records_dir}")

    out: dict[str, dict] = {}
    for path in sorted(records_dir.glob("*.jsonl")):
        for row in eval_io.read_records_file(path, validate=True):
            qid = row.get("query_id")
            if qid:
                out[qid] = row
    return out


def metric_value(record: dict, metric: str) -> float:
    """Per-query metric with errors counted as zero."""
    if not record.get("worker_ok", True):
        return 0.0
    return float(record.get(metric, 0.0))


def gold_pairs(record: dict) -> list[tuple[str, str]]:
    """Pair each gold doc id with its relevant node id for this query."""
    return list(zip(record.get("gold_doc_ids", []), record.get("relevant_node_ids", [])))


def anchors_found(record: dict) -> int:
    """Number of gold (doc, node) pairs present anywhere in the retrieved list."""
    retrieved = {(s.get("doc_id"), s.get("node_id")) for s in record.get("retrieved_sources", [])}
    return sum(1 for pair in gold_pairs(record) if pair in retrieved)


def failure_breakdown(records: dict[str, dict], subset: list[str], metric: str) -> dict:
    """Count subset failures by how many of the two gold anchors were retrieved."""
    failures = [q for q in subset if metric_value(records[q], metric) < 1.0]
    counts = {2: 0, 1: 0, 0: 0}
    for qid in failures:
        counts[anchors_found(records[qid])] += 1
    return {
        "failures": len(failures),
        "all_anchors_retrieved": counts[2],
        "one_anchor_missing": counts[1],
        "all_anchors_missing": counts[0],
    }


def main() -> int:
    """Contrast the two-anchor multihop subset against its complement and test significance."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", default=DEFAULT_RUN_A, help="run directory of side a")
    parser.add_argument("--run-b", default=DEFAULT_RUN_B, help="run directory of side b")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--permutations", type=int, default=DEFAULT_RANDOMIZATION_B)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", help="optional path for a JSON summary")
    args = parser.parse_args()

    rec_a = load_run_records(Path(args.run_a))
    rec_b = load_run_records(Path(args.run_b))
    qids = sorted(set(rec_a) & set(rec_b))
    if not qids:
        raise SystemExit("no shared query ids between the two runs")

    subset = [
        q for q in qids
        if rec_a[q].get("query_type") == "multihop" and rec_a[q].get("num_relevant") == 2
    ]
    complement = [q for q in qids if q not in set(subset)]
    if not subset or not complement:
        raise SystemExit("subset or complement is empty, check the runs")

    metric = args.metric
    diffs = {q: metric_value(rec_a[q], metric) - metric_value(rec_b[q], metric) for q in qids}
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    subset_margin = mean([diffs[q] for q in subset])
    complement_margin = mean([diffs[q] for q in complement])
    observed = subset_margin - complement_margin

    print(f"Shared queries: {len(qids)}")
    print(f"Two-anchor multihop subset: {len(subset)}, complement: {len(complement)}")
    print(f"{metric} margin (a minus b): subset {subset_margin:.4f}, "
          f"complement {complement_margin:.4f}, overall {mean(list(diffs.values())):.4f}")
    print(f"Observed contrast (subset minus complement): {observed:.4f}")

    # Permutation test on the subset label, via the shared core engine. The
    # ordered values place the subset first so the observed contrast matches.
    subset_diffs = [diffs[q] for q in subset]
    complement_diffs = [diffs[q] for q in complement]
    ordered = subset_diffs + complement_diffs
    p_two = two_sample_randomization(
        ordered, len(subset), alternative="two-sided",
        B=args.permutations, seed=args.seed)["p_value"]
    p_one = two_sample_randomization(
        ordered, len(subset), alternative="greater",
        B=args.permutations, seed=args.seed)["p_value"]
    print(f"Permutation test ({args.permutations} permutations, seed {args.seed}): "
          f"p two-sided {p_two:.4f}, one-sided {p_one:.4f}")

    ci = two_sample_bootstrap_ci(
        subset_diffs, complement_diffs,
        resamples=args.bootstrap_resamples, seed=args.seed)
    lo, hi = ci["low"], ci["high"]
    print(f"Bootstrap 95 percent CI of the contrast "
          f"({args.bootstrap_resamples} resamples): [{lo:.4f}, {hi:.4f}]")

    breakdown = {}
    for name, records in (("a", rec_a), ("b", rec_b)):
        breakdown[name] = failure_breakdown(records, subset, metric)
        b = breakdown[name]
        print(f"Side {name} subset failures ({metric} below 1.0): {b['failures']} "
              f"(all anchors retrieved {b['all_anchors_retrieved']}, "
              f"one missing {b['one_anchor_missing']}, "
              f"all missing {b['all_anchors_missing']})")

    if args.out:
        eval_io.write_json(Path(args.out), {
            "run_a": str(args.run_a),
            "run_b": str(args.run_b),
            "metric": metric,
            "n_shared": len(qids),
            "n_subset": len(subset),
            "subset_margin": subset_margin,
            "complement_margin": complement_margin,
            "contrast": observed,
            "p_two_sided": p_two,
            "p_one_sided": p_one,
            "bootstrap_ci95": [lo, hi],
            "permutations": args.permutations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "failure_breakdown": breakdown,
        })
        print(f"Summary written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
