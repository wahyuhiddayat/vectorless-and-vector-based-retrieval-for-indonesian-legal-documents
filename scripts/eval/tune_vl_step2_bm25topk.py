"""Step 2 of 5, HYBRID_BM25_TOP_K sweep (candidate pool fed to the reranker).

Sweeps the BM25 candidate pool size over {10, 20, 30, 50} at the k1/b chosen
in step 1, decides the winner on MAP@10 (ties go to the smaller pool to keep
the prompt cheaper), and carries it forward in the shared tuning state.

Usage:
    python scripts/eval/tune_vl_step2_bm25topk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.tune_vl_common import (
    load_state, save_state, require_step, run_eval, read_metrics,
    pick_winner_smaller_tie, print_table,
)

SWEEP = [10, 20, 30, 50]


def main() -> int:
    """Sweep HYBRID_BM25_TOP_K and record the MAP@10 winner in the state."""
    print("Step 2 of 5, HYBRID_BM25_TOP_K sweep, MAP@10 decides the winner.")
    state = load_state()
    require_step(state, "bm25_k1_b")

    results = []
    for v in SWEEP:
        label = f"run50_bm25topk{v}"
        env = {**state["env"], "HYBRID_BM25_TOP_K": v}
        run_dir = run_eval(label, env)
        results.append((v, read_metrics(run_dir)))

    winner, _ = pick_winner_smaller_tie(results)
    state["env"]["HYBRID_BM25_TOP_K"] = winner
    state["winner_topk"] = winner
    print_table("Sweep HYBRID_BM25_TOP_K", results, winner)

    state["decision_log"].append({
        "step": "sweep_bm25_top_k", "winner": winner,
        "results": [{"value": v, **m} for v, m in results],
    })
    state["steps_done"].append("sweep_bm25_top_k")
    save_state(state)
    print(f"\nWinner HYBRID_BM25_TOP_K = {winner}")
    print("Next, run, python scripts/eval/tune_vl_step3_docpick.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
