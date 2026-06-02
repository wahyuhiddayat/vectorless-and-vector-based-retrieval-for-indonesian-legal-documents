"""Step 5 of 5, query expansion on the tuned winner, then final summary.

Runs the tuned config with cached query expansion and compares MAP@10 against
the baseline from step 4. Query expansion is applied only if the lift exceeds
INTERVENTION_THRESHOLD. Writes the final tuned config and the full decision
log to data/eval_runs/tune_vectorless_log.json.

Usage:
    python scripts/eval/tune_vl_step5_qe.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.tune_vl_common import (
    load_state, save_state, require_step, run_eval, read_metrics,
    INTERVENTION_THRESHOLD, EVAL_RUNS_DIR,
)


def main() -> int:
    """Test query expansion, record the decision, and write the final log."""
    print("Step 5 of 5, query expansion on the tuned winner.")
    state = load_state()
    require_step(state, "model_upgrade")

    winner_topk = state["winner_topk"]
    winner_docpick = state["winner_docpick"]
    baseline = state["baseline_for_qe"]

    label = f"run53_qe_topk{winner_topk}_docpick{winner_docpick}"
    qe_path = REPO_ROOT / "data" / "query_expansion" / "dev_expanded.json"
    qe_metrics = read_metrics(run_eval(label, state["env"], query_expansion=qe_path))

    lift = qe_metrics["map@10"] - baseline["map@10"]
    print(f"\n  Without QE, MAP={baseline['map@10']:.4f}")
    print(f"  With QE,    MAP={qe_metrics['map@10']:.4f}")
    print(f"  Lift,       {lift:+.4f}")
    if lift > INTERVENTION_THRESHOLD:
        apply_qe = True
        final_metrics = qe_metrics
        print("  Decision, QE APPLIED")
    else:
        apply_qe = False
        final_metrics = baseline
        print("  Decision, QE REJECTED")

    state["apply_qe"] = apply_qe
    state["decision_log"].append({
        "step": "query_expansion", "without_qe": baseline,
        "with_qe": qe_metrics, "lift": lift,
        "decision": "applied" if apply_qe else "rejected",
    })
    state["steps_done"].append("query_expansion")
    save_state(state)

    print("\nFinal tuned vectorless winner, hybrid-tree.")
    print(f"  HYBRID_BM25_TOP_K,     {winner_topk}")
    print(f"  HYBRID_DOC_PICK_TOP_K, {winner_docpick}")
    print(f"  BM25 k1/b,             {state['env'].get('HYBRID_BM25_K1', 1.5)}/{state['env'].get('HYBRID_BM25_B', 0.75)}")
    print(f"  Retrieval LLM,         {state['final_model']}")
    print(f"  Query expansion,       {'applied' if apply_qe else 'rejected'}")
    print(f"  Final MAP@10 = {final_metrics['map@10']:.4f}")
    print(f"  Final MRR@10 = {final_metrics['mrr@10']:.4f}")
    print(f"  Final H@1    = {final_metrics['hit@1']:.4f}")
    print(f"  Final R@10   = {final_metrics['recall@10']:.4f}")

    log_path = EVAL_RUNS_DIR / "tune_vectorless_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "primary_metric": "map@10",
            "bm25_k1": state["env"].get("HYBRID_BM25_K1", 1.5),
            "bm25_b": state["env"].get("HYBRID_BM25_B", 0.75),
            "final_config": {
                "system": "hybrid-tree",
                "bm25_top_k": winner_topk,
                "doc_pick_top_k": winner_docpick,
                "retrieval_model": state["final_model"],
                "query_expansion": apply_qe,
            },
            "final_metrics": final_metrics,
            "decision_log": state["decision_log"],
        }, f, indent=2, default=str)
    print(f"\nDecision log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
