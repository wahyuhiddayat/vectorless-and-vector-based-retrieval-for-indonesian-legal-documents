"""Step 1 of 5, BM25 k1/b tuning on candidate recall (free, no LLM).

Grid-searches BM25 k1 and b on candidate recall using the gold documents as
an oracle, then initializes the shared tuning state with the winning k1/b for
the later steps. This step is free because k1/b only change which candidates
BM25 surfaces, which is measurable without calling the reranker.

Running this step creates a fresh state file, so re-running it resets the
whole tuning back to the start.

Usage:
    python scripts/eval/tune_vl_step1_bm25params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.tune_vl_common import default_state, save_state, STATE_PATH
from scripts.eval.tune_bm25_params import tune_k1_b


def main() -> int:
    """Compute k1/b on candidate recall and initialize the tuning state."""
    print("#" * 72)
    print("# STEP 1 of 5, BM25 k1/b on candidate recall (free, no LLM)")
    print("#" * 72)
    if STATE_PATH.exists():
        print(f"  Note, an existing state at {STATE_PATH} will be overwritten.")

    state = default_state()
    bm = tune_k1_b(split="dev", bm25_top_k=20)
    state["env"]["HYBRID_BM25_K1"] = bm["best_k1"]
    state["env"]["HYBRID_BM25_B"] = bm["best_b"]
    gain = bm["best_recall"] - bm["default_recall"]
    print(f"  Winner k1={bm['best_k1']}  b={bm['best_b']}  recall@20={bm['best_recall']:.4f}")
    print(f"  (default 1.5/0.75 recall@20={bm['default_recall']:.4f}, gain {gain:+.4f})")

    state["decision_log"].append({
        "step": "bm25_k1_b", "k1": bm["best_k1"], "b": bm["best_b"],
        "recall@20": bm["best_recall"], "default_recall": bm["default_recall"],
    })
    state["steps_done"].append("bm25_k1_b")
    save_state(state)
    print("\nNext, run, python scripts/eval/tune_vl_step2_bm25topk.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
