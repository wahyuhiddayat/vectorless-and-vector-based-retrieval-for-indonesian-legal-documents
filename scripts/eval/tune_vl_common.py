"""Shared helpers for the split hybrid-tree vectorless tuning steps.

Holds the constants, the carried-forward state file, the subprocess eval
runner, the metric reader, the winner-picking rule, and the table printer
used by the five step scripts (tune_vl_step1_bm25params through
tune_vl_step5_qe). The tuning is split into one script per step so each step
can be launched and monitored on its own, and a crash in one step does not
discard the work done by the others.

The state file at data/eval_runs/tune_vectorless_state.json carries each
step's winner forward to the next step. Step 1 creates it, every later step
reads and updates it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_RUNS_DIR = REPO_ROOT / "data" / "eval_runs"
STATE_PATH = EVAL_RUNS_DIR / "tune_vectorless_state.json"

TIE_TOLERANCE = 0.002
"""MAP@10 tolerance for tie-breaking. Ties resolve to the cheaper setting."""

INTERVENTION_THRESHOLD = 0.003
"""Minimum MAP@10 lift required to accept a model upgrade or query expansion."""

ERROR_ABORT_THRESHOLD = 10
"""A run with more errored queries than this aborts, to protect the winner pick."""


def default_state() -> dict:
    """Return the initial carried-forward state at stage 1 hybrid-tree defaults."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env": {
            "HYBRID_BM25_TOP_K": 10,
            "HYBRID_DOC_PICK_TOP_K": 3,
        },
        "winner_topk": None,
        "winner_docpick": None,
        "tuned_flash": None,
        "baseline_for_qe": None,
        "final_model": "deepseek-v4-flash",
        "apply_qe": False,
        "decision_log": [],
        "steps_done": [],
    }


def load_state() -> dict:
    """Load the tuning state file, or exit if it is missing.

    A missing state means the steps are being run out of order, since step 1
    is responsible for creating it.
    """
    if not STATE_PATH.exists():
        print(f"!!! No state file at {STATE_PATH}")
        print("Run step 1 first, python scripts/eval/tune_vl_step1_bm25params.py")
        sys.exit(1)
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Write the tuning state file."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\nState saved to {STATE_PATH}")


def require_step(state: dict, step: str) -> None:
    """Exit if a prerequisite step has not been completed yet."""
    if step not in state["steps_done"]:
        print(f"!!! Prerequisite step '{step}' is not done yet.")
        print(f"    Completed so far, {state['steps_done']}")
        sys.exit(1)


def run_eval(label: str, env_overrides: dict, query_expansion: Path | None = None) -> Path:
    """Invoke scripts/eval/vectorless.py for hybrid-tree at pasal on dev.

    Resumes automatically if the run directory already exists, so a crash
    mid-step only re-runs the missing or errored queries on the next launch.

    Args:
        label: Eval run label, becomes the directory name under data/eval_runs.
        env_overrides: Env vars to set for this run (hybrid knobs, model override).
        query_expansion: Optional path to a cached query expansion JSON.

    Returns:
        Path to the run's results directory.
    """
    run_dir = EVAL_RUNS_DIR / label
    resume = run_dir.exists()
    cmd = [
        sys.executable, "scripts/eval/vectorless.py",
        "--label", label,
        "--systems", "hybrid-tree",
        "--granularities", "pasal",
        "--split", "dev",
    ]
    if query_expansion is not None:
        cmd.extend(["--query-expansion", str(query_expansion)])
    if resume:
        cmd.append("--resume")

    env = os.environ.copy()
    for k, v in env_overrides.items():
        env[k] = str(v)

    suffix = " (resume)" if resume else ""
    print(f"\nRunning {label}{suffix}, started {datetime.now().isoformat(timespec='seconds')}.")
    print(f"  Settings {env_overrides}")
    if query_expansion is not None:
        print(f"  Query expansion {query_expansion}")

    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n!!! Eval failed, {label} (exit code {result.returncode})")
        print("Fix the cause, then re-run this same step. It will resume where it stopped.")
        sys.exit(1)
    return run_dir


def read_metrics(run_dir: Path) -> dict:
    """Read MAP@10 and secondary metrics from a run's summary file.

    Aborts if the run has more than ERROR_ABORT_THRESHOLD errored queries,
    since a heavily errored run (for example after the API balance ran out)
    has a deflated MAP@10 that would poison the winner selection. Re-running
    the step resumes the errored queries.
    """
    summary_path = run_dir / "summary_overall.json"
    if not summary_path.exists():
        print(f"!!! Missing summary at {summary_path}")
        sys.exit(1)
    with open(summary_path) as f:
        s = json.load(f)
    ov = s["overall"]
    errors = ov["error_count"]
    if errors > ERROR_ABORT_THRESHOLD:
        print(f"!!! Run {run_dir.name} has {errors} errored queries, MAP@10 is unreliable.")
        print("Top up the API balance or fix the cause, then re-run this step to resume.")
        sys.exit(1)
    if errors > 0:
        print(f"  Warning, {run_dir.name} has {errors} errored queries (within tolerance).")
    return {
        "map@10": ov["map@10"],
        "mrr@10": ov["mrr@10"],
        "hit@1": ov["hit@1"],
        "recall@10": ov["recall@10"],
        "errors": errors,
    }


def pick_winner_smaller_tie(results: list[tuple]) -> tuple:
    """Pick highest MAP@10, ties within TIE_TOLERANCE go to the smaller param.

    Args:
        results: list of (param_value, metrics_dict) tuples.

    Returns:
        Tuple of (winning_param_value, winning_metrics_dict).
    """
    best = max(m["map@10"] for _, m in results)
    contenders = [(v, m) for v, m in results if m["map@10"] >= best - TIE_TOLERANCE]
    return min(contenders, key=lambda x: x[0])


def print_table(name: str, results: list[tuple], winner_value) -> None:
    """Print a plain comparison table for one sweep, sorted by MAP@10."""
    print(f"\n{name}")
    print(f"{'value':<12} {'MAP@10':<8} {'MRR@10':<8} {'H@1':<6} {'R@10':<6} {'errors':<6}")
    for v, m in sorted(results, key=lambda x: -x[1]["map@10"]):
        marker = "  winner" if v == winner_value else ""
        print(
            f"{str(v):<12} {m['map@10']:<8.4f} {m['mrr@10']:<8.4f} "
            f"{m['hit@1']:<6.4f} {m['recall@10']:<6.4f} {m['errors']:<6}{marker}"
        )
