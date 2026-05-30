"""Sequential ablation tuning for the dense vector retrieval pipeline.

Runs 10 dev-split evaluations on the bge-m3 + bge-reranker pipeline,
carrying each step's winner forward into the next step. Ablation order
is candidate pool size, HNSW search depth, reranker model upgrade, and
query expansion. Decision rules are encoded as TIE_TOLERANCE
(within-noise threshold for tie-breaking) and INTERVENTION_THRESHOLD
(minimum lift to accept an upgrade).

Usage:
    python scripts/eval/tune_vector.py --qdrant-path ./qdrant_local
    python scripts/eval/tune_vector.py --qdrant-path ./qdrant_local --dry-run

Outputs:
    - 10 eval run directories under data/eval_runs (run38..run42 series)
    - data/eval_runs/tune_vector_log.json with the full decision history
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_RUNS_DIR = REPO_ROOT / "data" / "eval_runs"

TIE_TOLERANCE = 0.002
"""MRR@10 tolerance for tie-breaking. Ties resolve to the cheaper setting."""

INTERVENTION_THRESHOLD = 0.003
"""Minimum MRR@10 lift required to accept a model upgrade or query expansion."""


def run_eval(
    label: str,
    reranker: str,
    env_overrides: dict,
    qdrant_path: str,
    query_expansion: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Invoke scripts/eval/vector.py as a subprocess with the given config.

    Args:
        label: Eval run label, becomes the directory name under data/eval_runs.
        reranker: Reranker model key, used to build the --systems argument.
        env_overrides: Env vars to set for this run.
        qdrant_path: Path to the local Qdrant storage directory.
        query_expansion: Optional path to a cached query expansion JSON.
        dry_run: If True, print the command without executing.

    Returns:
        Path to the run's results directory.
    """
    cmd = [
        sys.executable, "scripts/eval/vector.py",
        "--label", label,
        "--systems", "vector-dense",
        "--embedding-models", "bge-m3",
        "--rerankers", reranker,
        "--granularities", "pasal",
        "--split", "dev",
        "--qdrant-path", qdrant_path,
    ]
    if query_expansion is not None:
        cmd.extend(["--query-expansion", str(query_expansion)])

    env = os.environ.copy()
    for k, v in env_overrides.items():
        env[k] = str(v)

    print()
    print("=" * 72)
    print(f"  Starting: {label}")
    print(f"  Reranker: {reranker}")
    print(f"  Env overrides: {env_overrides}")
    if query_expansion is not None:
        print(f"  Query expansion: {query_expansion}")
    print(f"  Started at: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 72)

    if dry_run:
        print("  [dry-run] would run:")
        print("  " + " ".join(cmd))
        return EVAL_RUNS_DIR / label

    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n!!! Eval failed: {label} (exit code {result.returncode}) !!!")
        print("Tuner aborting. Inspect the run, then resume by editing this script.")
        sys.exit(1)

    return EVAL_RUNS_DIR / label


def read_metrics(run_dir: Path) -> dict:
    """Read the four primary metrics from a run's summary file."""
    summary_path = run_dir / "summary_overall.json"
    if not summary_path.exists():
        print(f"!!! Missing summary at {summary_path}")
        sys.exit(1)
    with open(summary_path) as f:
        s = json.load(f)
    ov = s["overall"]
    return {
        "mrr@10": ov["mrr@10"],
        "hit@1": ov["hit@1"],
        "recall@10": ov["recall@10"],
        "errors": ov["error_count"],
    }


def pick_winner_smaller_tie(results: list[tuple]) -> tuple:
    """Pick the entry with highest MRR@10, ties within TIE_TOLERANCE go to smaller param.

    Args:
        results: list of (param_value, metrics_dict) tuples.

    Returns:
        Tuple of (winning_param_value, winning_metrics_dict).
    """
    best_mrr = max(m["mrr@10"] for _, m in results)
    contenders = [(v, m) for v, m in results if m["mrr@10"] >= best_mrr - TIE_TOLERANCE]
    return min(contenders, key=lambda x: x[0])


def print_table(name: str, results: list[tuple], winner_value) -> None:
    """Print a formatted comparison table for one sweep."""
    print(f"\n--- {name} results ---")
    print(f"{'Value':<12} {'MRR@10':<8} {'H@1':<6} {'R@10':<6} {'Errors':<6}")
    for v, m in sorted(results, key=lambda x: -x[1]["mrr@10"]):
        marker = "  <-- WINNER" if v == winner_value else ""
        print(
            f"{str(v):<12} {m['mrr@10']:<8.4f} "
            f"{m['hit@1']:<6.4f} {m['recall@10']:<6.4f} "
            f"{m['errors']:<6}{marker}"
        )


def main() -> int:
    """Run the full sequential ablation and write the decision log."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qdrant-path", default="./qdrant_local",
        help="Path to local Qdrant storage directory (default ./qdrant_local).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    args = ap.parse_args()

    # Carried-forward state. Starts at the un-tuned baseline.
    state = {
        "VECTOR_EMBEDDING_MODEL": "bge-m3",
        "VECTOR_RERANKER": "bge-reranker-v2-m3",
        "VECTOR_GRANULARITY": "pasal",
        "VECTOR_RERANKER_TOP_N": 50,
        "VECTOR_HNSW_EF_SEARCH": 128,
    }
    decision_log = []

    # Sweep 1, VECTOR_RERANKER_TOP_N
    print("\n" + "#" * 72)
    print("# SWEEP 1 of 2, VECTOR_RERANKER_TOP_N")
    print("#" * 72)
    sweep1_results = []
    for top_n in [20, 50, 100, 200]:
        label = f"run38_topn{top_n}"
        env = {**state, "VECTOR_RERANKER_TOP_N": top_n}
        run_dir = run_eval(label, "bge-reranker-v2-m3", env, args.qdrant_path, dry_run=args.dry_run)
        if not args.dry_run:
            sweep1_results.append((top_n, read_metrics(run_dir)))
    if args.dry_run:
        print("\n  [dry-run] skipping winner pick")
        return 0
    winner_top_n, _ = pick_winner_smaller_tie(sweep1_results)
    state["VECTOR_RERANKER_TOP_N"] = winner_top_n
    print_table("Sweep 1 VECTOR_RERANKER_TOP_N", sweep1_results, winner_top_n)
    decision_log.append({
        "step": "sweep_top_n",
        "winner": winner_top_n,
        "results": [{"value": v, **m} for v, m in sweep1_results],
    })

    # Sweep 2, VECTOR_HNSW_EF_SEARCH
    print("\n" + "#" * 72)
    print("# SWEEP 2 of 2, VECTOR_HNSW_EF_SEARCH")
    print("#" * 72)
    sweep2_results = []
    for ef in [64, 128, 256, 512]:
        label = f"run39_ef{ef}_topn{winner_top_n}"
        env = {**state, "VECTOR_HNSW_EF_SEARCH": ef}
        run_dir = run_eval(label, "bge-reranker-v2-m3", env, args.qdrant_path)
        sweep2_results.append((ef, read_metrics(run_dir)))
    winner_ef, _ = pick_winner_smaller_tie(sweep2_results)
    state["VECTOR_HNSW_EF_SEARCH"] = winner_ef
    print_table("Sweep 2 VECTOR_HNSW_EF_SEARCH", sweep2_results, winner_ef)
    decision_log.append({
        "step": "sweep_ef_search",
        "winner": winner_ef,
        "results": [{"value": v, **m} for v, m in sweep2_results],
    })

    # Use sweep 2 winner config as the tuned-v2m3 baseline for the upgrade comparison.
    tuned_v2m3_metrics = next(m for v, m in sweep2_results if v == winner_ef)

    # Reranker model upgrade
    print("\n" + "#" * 72)
    print("# RERANKER UPGRADE, bge-reranker-v2-m3 to bge-reranker-v2-gemma")
    print("#" * 72)
    label_upgrade = f"run41_v2gemma_topn{winner_top_n}_ef{winner_ef}"
    env_upgrade = {**state, "VECTOR_RERANKER": "bge-reranker-v2-gemma"}
    run_dir_upgrade = run_eval(label_upgrade, "bge-reranker-v2-gemma", env_upgrade, args.qdrant_path)
    gemma_metrics = read_metrics(run_dir_upgrade)

    lift_upgrade = gemma_metrics["mrr@10"] - tuned_v2m3_metrics["mrr@10"]
    print(f"\n  v2-m3 (tuned baseline): MRR={tuned_v2m3_metrics['mrr@10']:.4f}")
    print(f"  v2-gemma (upgraded):    MRR={gemma_metrics['mrr@10']:.4f}")
    print(f"  Lift over tuned v2-m3:  {lift_upgrade:+.4f}")

    if lift_upgrade > INTERVENTION_THRESHOLD:
        final_reranker = "bge-reranker-v2-gemma"
        baseline_for_qe = gemma_metrics
        state["VECTOR_RERANKER"] = "bge-reranker-v2-gemma"
        print(f"  Decision, UPGRADE ACCEPTED (lift exceeds +{INTERVENTION_THRESHOLD})")
    else:
        final_reranker = "bge-reranker-v2-m3"
        baseline_for_qe = tuned_v2m3_metrics
        print(f"  Decision, UPGRADE REJECTED, keeping v2-m3")
    decision_log.append({
        "step": "reranker_upgrade",
        "v2_m3_tuned": tuned_v2m3_metrics,
        "v2_gemma": gemma_metrics,
        "lift": lift_upgrade,
        "decision": final_reranker,
    })

    # Query expansion
    print("\n" + "#" * 72)
    print("# QUERY EXPANSION on tuned winner")
    print("#" * 72)
    short_rerank = final_reranker.replace("bge-reranker-", "").replace("-", "")
    label_qe = f"run42_qe_{short_rerank}_topn{winner_top_n}_ef{winner_ef}"
    qe_path = REPO_ROOT / "data" / "query_expansion" / "dev_expanded.json"
    run_dir_qe = run_eval(
        label_qe, final_reranker, state, args.qdrant_path,
        query_expansion=qe_path,
    )
    qe_metrics = read_metrics(run_dir_qe)

    lift_qe = qe_metrics["mrr@10"] - baseline_for_qe["mrr@10"]
    print(f"\n  Without QE: MRR={baseline_for_qe['mrr@10']:.4f}")
    print(f"  With QE:    MRR={qe_metrics['mrr@10']:.4f}")
    print(f"  Lift:       {lift_qe:+.4f}")

    if lift_qe > INTERVENTION_THRESHOLD:
        apply_qe = True
        final_metrics = qe_metrics
        print(f"  Decision, QE APPLIED (lift exceeds +{INTERVENTION_THRESHOLD})")
    else:
        apply_qe = False
        final_metrics = baseline_for_qe
        print(f"  Decision, QE REJECTED")
    decision_log.append({
        "step": "query_expansion",
        "without_qe": baseline_for_qe,
        "with_qe": qe_metrics,
        "lift": lift_qe,
        "decision": "applied" if apply_qe else "rejected",
    })

    # Final summary
    print("\n" + "#" * 72)
    print("# FINAL TUNED VECTOR WINNER")
    print("#" * 72)
    print(f"  Reranker:        {final_reranker}")
    print(f"  RERANKER_TOP_N:  {winner_top_n}")
    print(f"  HNSW_EF_SEARCH:  {winner_ef}")
    print(f"  Query expansion: {'applied' if apply_qe else 'rejected'}")
    print(f"  Final metrics:")
    print(f"    MRR@10 = {final_metrics['mrr@10']:.4f}")
    print(f"    H@1    = {final_metrics['hit@1']:.4f}")
    print(f"    R@10   = {final_metrics['recall@10']:.4f}")

    log_path = EVAL_RUNS_DIR / "tune_vector_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "final_config": {
                "reranker": final_reranker,
                "rerank_top_n": winner_top_n,
                "hnsw_ef_search": winner_ef,
                "query_expansion": apply_qe,
            },
            "final_metrics": final_metrics,
            "untuned_baseline": {
                "mrr@10": 0.9153,
                "hit@1": 0.8711,
                "recall@10": 0.9776,
            },
            "lift_over_baseline": {
                "mrr@10": round(final_metrics["mrr@10"] - 0.9153, 4),
                "hit@1": round(final_metrics["hit@1"] - 0.8711, 4),
                "recall@10": round(final_metrics["recall@10"] - 0.9776, 4),
            },
            "decision_log": decision_log,
        }, f, indent=2, default=str)
    print(f"\nDecision log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
