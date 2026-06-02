"""Sequential ablation tuning for the hybrid-tree vectorless pipeline.

Tunes the stage 1 vectorless winner, hybrid-tree at pasal, carrying each
step's winner forward. Decisions use MAP@10 as the primary metric, because
the ground truth is partly multi-gold (multihop queries have two required
pasals) and MAP credits retrieving all of them, while MRR ignores the second.

Ablation order is candidate pool size (HYBRID_BM25_TOP_K), document pool size
(HYBRID_DOC_PICK_TOP_K), retrieval-LLM model upgrade, then query expansion.
BM25 k1 and b are tuned separately on candidate recall (free, no LLM) and set
via HYBRID_BM25_K1 and HYBRID_BM25_B before running this orchestrator.

Usage:
    python scripts/eval/tune_vectorless.py
    python scripts/eval/tune_vectorless.py --dry-run
    python scripts/eval/tune_vectorless.py --label-suffix _v2

Outputs:
    - eval run directories under data/eval_runs
    - data/eval_runs/tune_vectorless_log.json with the decision history
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
"""MAP@10 tolerance for tie-breaking. Ties resolve to the cheaper setting."""

INTERVENTION_THRESHOLD = 0.003
"""Minimum MAP@10 lift required to accept a model upgrade or query expansion."""


def run_eval(
    label: str,
    env_overrides: dict,
    query_expansion: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Invoke scripts/eval/vectorless.py as a subprocess for hybrid-tree at pasal.

    Args:
        label: Eval run label, becomes the directory name under data/eval_runs.
        env_overrides: Env vars to set for this run (hybrid knobs, model override).
        query_expansion: Optional path to a cached query expansion JSON.
        dry_run: If True, print the command without executing.

    Returns:
        Path to the run's results directory.
    """
    cmd = [
        sys.executable, "scripts/eval/vectorless.py",
        "--label", label,
        "--systems", "hybrid-tree",
        "--granularities", "pasal",
        "--split", "dev",
    ]
    if query_expansion is not None:
        cmd.extend(["--query-expansion", str(query_expansion)])

    env = os.environ.copy()
    for k, v in env_overrides.items():
        env[k] = str(v)

    print()
    print("=" * 72)
    print(f"  Starting: {label}")
    print(f"  Env overrides: {env_overrides}")
    if query_expansion is not None:
        print(f"  Query expansion: {query_expansion}")
    print(f"  Started at: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 72)

    if dry_run:
        print("  [dry-run] would run:")
        print("  " + " ".join(cmd))
        print("  [dry-run] with env:", env_overrides)
        return EVAL_RUNS_DIR / label

    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n!!! Eval failed: {label} (exit code {result.returncode}) !!!")
        print("Tuner aborting. Inspect the run, then resume by editing this script.")
        sys.exit(1)

    return EVAL_RUNS_DIR / label


def read_metrics(run_dir: Path) -> dict:
    """Read MAP@10 and the secondary metrics from a run's summary file."""
    summary_path = run_dir / "summary_overall.json"
    if not summary_path.exists():
        print(f"!!! Missing summary at {summary_path}")
        sys.exit(1)
    with open(summary_path) as f:
        s = json.load(f)
    ov = s["overall"]
    return {
        "map@10": ov["map@10"],
        "mrr@10": ov["mrr@10"],
        "hit@1": ov["hit@1"],
        "recall@10": ov["recall@10"],
        "errors": ov["error_count"],
    }


def pick_winner_smaller_tie(results: list[tuple]) -> tuple:
    """Pick the entry with highest MAP@10, ties within TIE_TOLERANCE go to smaller param.

    Args:
        results: list of (param_value, metrics_dict) tuples.

    Returns:
        Tuple of (winning_param_value, winning_metrics_dict).
    """
    best = max(m["map@10"] for _, m in results)
    contenders = [(v, m) for v, m in results if m["map@10"] >= best - TIE_TOLERANCE]
    return min(contenders, key=lambda x: x[0])


def print_table(name: str, results: list[tuple], winner_value) -> None:
    """Print a formatted comparison table for one sweep, MAP@10 first."""
    print(f"\n--- {name} results ---")
    print(f"{'Value':<12} {'MAP@10':<8} {'MRR@10':<8} {'H@1':<6} {'R@10':<6} {'Errors':<6}")
    for v, m in sorted(results, key=lambda x: -x[1]["map@10"]):
        marker = "  <-- WINNER" if v == winner_value else ""
        print(
            f"{str(v):<12} {m['map@10']:<8.4f} {m['mrr@10']:<8.4f} "
            f"{m['hit@1']:<6.4f} {m['recall@10']:<6.4f} {m['errors']:<6}{marker}"
        )


def main() -> int:
    """Run the full sequential ablation on MAP@10 and write the decision log."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing.")
    ap.add_argument("--label-suffix", default="",
                    help="Suffix appended to every run label and the log, to "
                         "avoid collisions with earlier runs (e.g. _v2).")
    args = ap.parse_args()
    sfx = args.label_suffix

    # Carried-forward state. Starts at the stage 1 hybrid-tree defaults.
    # BM25 k1/b are inherited from the environment (tuned separately).
    state = {
        "HYBRID_BM25_TOP_K": 10,
        "HYBRID_DOC_PICK_TOP_K": 3,
    }
    decision_log = []

    # Sweep 1, HYBRID_BM25_TOP_K (candidate pool fed to the LLM reranker)
    print("\n" + "#" * 72)
    print("# SWEEP 1 of 2, HYBRID_BM25_TOP_K")
    print("#" * 72)
    sweep1 = []
    for v in [10, 20, 30, 50]:
        label = f"run50_bm25topk{v}{sfx}"
        env = {**state, "HYBRID_BM25_TOP_K": v}
        run_dir = run_eval(label, env, dry_run=args.dry_run)
        if not args.dry_run:
            sweep1.append((v, read_metrics(run_dir)))
    if args.dry_run:
        print("\n  [dry-run] skipping winner pick")
        return 0
    winner_topk, _ = pick_winner_smaller_tie(sweep1)
    state["HYBRID_BM25_TOP_K"] = winner_topk
    print_table("Sweep 1 HYBRID_BM25_TOP_K", sweep1, winner_topk)
    decision_log.append({
        "step": "sweep_bm25_top_k", "winner": winner_topk,
        "results": [{"value": v, **m} for v, m in sweep1],
    })

    # Sweep 2, HYBRID_DOC_PICK_TOP_K (document pool, drives multihop coverage)
    print("\n" + "#" * 72)
    print("# SWEEP 2 of 2, HYBRID_DOC_PICK_TOP_K")
    print("#" * 72)
    sweep2 = []
    for v in [1, 2, 3, 5]:
        label = f"run51_docpick{v}{sfx}"
        env = {**state, "HYBRID_DOC_PICK_TOP_K": v}
        run_dir = run_eval(label, env, dry_run=args.dry_run)
        sweep2.append((v, read_metrics(run_dir)))
    winner_docpick, _ = pick_winner_smaller_tie(sweep2)
    state["HYBRID_DOC_PICK_TOP_K"] = winner_docpick
    print_table("Sweep 2 HYBRID_DOC_PICK_TOP_K", sweep2, winner_docpick)
    decision_log.append({
        "step": "sweep_doc_pick_top_k", "winner": winner_docpick,
        "results": [{"value": v, **m} for v, m in sweep2],
    })

    tuned_flash = next(m for v, m in sweep2 if v == winner_docpick)

    # Model upgrade, deepseek-v4-flash to deepseek-v4-pro
    print("\n" + "#" * 72)
    print("# MODEL UPGRADE, deepseek-v4-flash to deepseek-v4-pro")
    print("#" * 72)
    label_pro = f"run52_v4pro_topk{winner_topk}_docpick{winner_docpick}{sfx}"
    env_pro = {**state, "RETRIEVAL_MODEL_OVERRIDE": "deepseek-v4-pro"}
    pro_metrics = read_metrics(run_eval(label_pro, env_pro))
    lift_pro = pro_metrics["map@10"] - tuned_flash["map@10"]
    print(f"\n  flash (tuned): MAP={tuned_flash['map@10']:.4f}")
    print(f"  v4-pro:        MAP={pro_metrics['map@10']:.4f}")
    print(f"  Lift:          {lift_pro:+.4f}")
    if lift_pro > INTERVENTION_THRESHOLD:
        state["RETRIEVAL_MODEL_OVERRIDE"] = "deepseek-v4-pro"
        baseline_for_qe = pro_metrics
        final_model = "deepseek-v4-pro"
        print(f"  Decision, UPGRADE ACCEPTED")
    else:
        baseline_for_qe = tuned_flash
        final_model = "deepseek-v4-flash"
        print(f"  Decision, UPGRADE REJECTED, keeping flash")
    decision_log.append({
        "step": "model_upgrade", "flash": tuned_flash, "v4_pro": pro_metrics,
        "lift": lift_pro, "decision": final_model,
    })

    # Query expansion on the tuned config
    print("\n" + "#" * 72)
    print("# QUERY EXPANSION on tuned winner")
    print("#" * 72)
    label_qe = f"run53_qe_topk{winner_topk}_docpick{winner_docpick}{sfx}"
    qe_path = REPO_ROOT / "data" / "query_expansion" / "dev_expanded.json"
    qe_metrics = read_metrics(run_eval(label_qe, state, query_expansion=qe_path))
    lift_qe = qe_metrics["map@10"] - baseline_for_qe["map@10"]
    print(f"\n  Without QE: MAP={baseline_for_qe['map@10']:.4f}")
    print(f"  With QE:    MAP={qe_metrics['map@10']:.4f}")
    print(f"  Lift:       {lift_qe:+.4f}")
    if lift_qe > INTERVENTION_THRESHOLD:
        apply_qe = True
        final_metrics = qe_metrics
        print(f"  Decision, QE APPLIED")
    else:
        apply_qe = False
        final_metrics = baseline_for_qe
        print(f"  Decision, QE REJECTED")
    decision_log.append({
        "step": "query_expansion", "without_qe": baseline_for_qe,
        "with_qe": qe_metrics, "lift": lift_qe,
        "decision": "applied" if apply_qe else "rejected",
    })

    # Final summary
    print("\n" + "#" * 72)
    print("# FINAL TUNED VECTORLESS (hybrid-tree) WINNER")
    print("#" * 72)
    print(f"  HYBRID_BM25_TOP_K:     {winner_topk}")
    print(f"  HYBRID_DOC_PICK_TOP_K: {winner_docpick}")
    print(f"  BM25 k1/b:             {os.environ.get('HYBRID_BM25_K1','1.5')}/{os.environ.get('HYBRID_BM25_B','0.75')}")
    print(f"  Retrieval LLM:         {final_model}")
    print(f"  Query expansion:       {'applied' if apply_qe else 'rejected'}")
    print(f"  Final MAP@10 = {final_metrics['map@10']:.4f}")
    print(f"  Final MRR@10 = {final_metrics['mrr@10']:.4f}")
    print(f"  Final H@1    = {final_metrics['hit@1']:.4f}")
    print(f"  Final R@10   = {final_metrics['recall@10']:.4f}")

    log_path = EVAL_RUNS_DIR / f"tune_vectorless_log{sfx}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "primary_metric": "map@10",
            "bm25_k1": os.environ.get("HYBRID_BM25_K1", "1.5"),
            "bm25_b": os.environ.get("HYBRID_BM25_B", "0.75"),
            "final_config": {
                "system": "hybrid-tree",
                "bm25_top_k": winner_topk,
                "doc_pick_top_k": winner_docpick,
                "retrieval_model": final_model,
                "query_expansion": apply_qe,
            },
            "final_metrics": final_metrics,
            "decision_log": decision_log,
        }, f, indent=2, default=str)
    print(f"\nDecision log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
