"""Step 3 of 5, HYBRID_DOC_PICK_TOP_K sweep (document pool, multihop coverage).

Sweeps the number of documents selected at stage 1 over {1, 2, 3, 5} at the
candidate pool chosen in step 2, decides the winner on MAP@10 (ties go to the
smaller pool), and stores both the winner and its metrics as the flash
baseline for the model upgrade in step 4.

Usage:
    python scripts/eval/tune_vl_step3_docpick.py
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

SWEEP = [1, 2, 3, 5]


def main() -> int:
    """Sweep HYBRID_DOC_PICK_TOP_K and store the winner and its flash metrics."""
    print("Step 3 of 5, HYBRID_DOC_PICK_TOP_K sweep, MAP@10 decides the winner.")
    state = load_state()
    require_step(state, "sweep_bm25_top_k")

    results = []
    for v in SWEEP:
        label = f"run51_docpick{v}"
        env = {**state["env"], "HYBRID_DOC_PICK_TOP_K": v}
        run_dir = run_eval(label, env)
        results.append((v, read_metrics(run_dir)))

    winner, winner_metrics = pick_winner_smaller_tie(results)
    state["env"]["HYBRID_DOC_PICK_TOP_K"] = winner
    state["winner_docpick"] = winner
    state["tuned_flash"] = winner_metrics
    print_table("Sweep HYBRID_DOC_PICK_TOP_K", results, winner)

    state["decision_log"].append({
        "step": "sweep_doc_pick_top_k", "winner": winner,
        "results": [{"value": v, **m} for v, m in results],
    })
    state["steps_done"].append("sweep_doc_pick_top_k")
    save_state(state)
    print(f"\nWinner HYBRID_DOC_PICK_TOP_K = {winner}")
    print("Next, run, python scripts/eval/tune_vl_step4_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
