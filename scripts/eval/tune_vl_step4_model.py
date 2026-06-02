"""Step 4 of 5, retrieval-LLM model upgrade, deepseek-v4-flash to deepseek-v4-pro.

Runs the tuned config once on deepseek-v4-pro and compares MAP@10 against the
flash baseline from step 3. The upgrade is accepted only if the lift exceeds
INTERVENTION_THRESHOLD, otherwise flash is kept. The accepted model and the
baseline for query expansion are stored for step 5.

Usage:
    python scripts/eval/tune_vl_step4_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.tune_vl_common import (
    load_state, save_state, require_step, run_eval, read_metrics,
    INTERVENTION_THRESHOLD,
)


def main() -> int:
    """Test the pro upgrade against the flash baseline and record the decision."""
    print("Step 4 of 5, model upgrade, deepseek-v4-flash to deepseek-v4-pro.")
    state = load_state()
    require_step(state, "sweep_doc_pick_top_k")

    winner_topk = state["winner_topk"]
    winner_docpick = state["winner_docpick"]
    tuned_flash = state["tuned_flash"]

    label = f"run52_v4pro_topk{winner_topk}_docpick{winner_docpick}"
    env_pro = {**state["env"], "RETRIEVAL_MODEL_OVERRIDE": "deepseek-v4-pro"}
    pro_metrics = read_metrics(run_eval(label, env_pro))

    lift = pro_metrics["map@10"] - tuned_flash["map@10"]
    print(f"\n  flash (tuned), MAP={tuned_flash['map@10']:.4f}")
    print(f"  v4-pro,        MAP={pro_metrics['map@10']:.4f}")
    print(f"  Lift,          {lift:+.4f}")
    if lift > INTERVENTION_THRESHOLD:
        state["env"]["RETRIEVAL_MODEL_OVERRIDE"] = "deepseek-v4-pro"
        state["baseline_for_qe"] = pro_metrics
        state["final_model"] = "deepseek-v4-pro"
        print("  Decision, UPGRADE ACCEPTED")
    else:
        state["baseline_for_qe"] = tuned_flash
        state["final_model"] = "deepseek-v4-flash"
        print("  Decision, UPGRADE REJECTED, keeping flash")

    state["decision_log"].append({
        "step": "model_upgrade", "flash": tuned_flash, "v4_pro": pro_metrics,
        "lift": lift, "decision": state["final_model"],
    })
    state["steps_done"].append("model_upgrade")
    save_state(state)
    print("Next, run, python scripts/eval/tune_vl_step5_qe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
