"""Regenerate a run's summary files from its records at a wider cutoff set.

The development-partition runs were scored before cutoff 2 was added to the
metric set, so their per-query records were later augmented with the @2 fields
(see backfill_cutoff2.py) but their summary_*.csv files still expose only the
[1, 3, 5, 10] columns they were first written with. This script re-aggregates a
finished run from its existing records, using the same aggregation functions the
eval harness uses in EvalRunner.finalize, so the summaries gain the recall@2
columns without rerunning any retrieval.

The recomputation is read-only over the records. It rewrites only the summary
artifacts. By default it refuses to write a run unless every existing
non-@2 cell in summary_by_system_granularity.csv and summary_by_slice.csv is
reproduced exactly, which proves the change is purely additive. A run whose
stored summary disagrees on other columns (for example a run whose summary was
written from a partial, aborted finalize) is reported and left untouched unless
--allow-other-changes is passed, so a stale summary is never silently rewritten.

Usage:
    python scripts/eval/resummarize_from_records.py --dry-run data/eval_runs/stage1_vectorless/*
    python scripts/eval/resummarize_from_records.py data/eval_runs/stage1_vectorless/*
    python scripts/eval/resummarize_from_records.py --allow-other-changes data/eval_runs/stage1_vectorless/run27_20260525_vectorless_dev_357q_llm_tree_deepseek
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core import io as eval_io
from scripts.eval.core.aggregation import (
    aggregate_records,
    compute_combo_confidence_intervals,
    compute_combo_summaries,
    compute_reference_mode_breakdown,
    compute_slice_summaries,
)
from scripts.eval.core.metrics import DEFAULT_CUTOFFS, sibling_failure_stats

COMBO_KEY = ("system", "eval_granularity")
SLICE_KEY = ("slice_type", "slice_value", "system", "eval_granularity")


def _is_cutoff2_column(name: str) -> bool:
    """True for any metric column scoped to cutoff 2, which is the only addition."""
    return name.endswith("@2")


def _read_existing_csv(path: Path) -> list[dict]:
    """Read a summary CSV into a list of string-valued row dicts, empty if missing."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row_index(rows: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    """Index rows by the tuple of key-field values for row-to-row matching."""
    return {tuple(str(row.get(k, "")) for k in key_fields): row for row in rows}


def _csv_cell(value) -> str:
    """Render a value exactly as csv.DictWriter would, so comparisons are byte-exact."""
    if value is None:
        return ""
    return str(value)


def check_additive(existing: list[dict], computed: list[dict], key_fields: tuple[str, ...]) -> list[str]:
    """Return a list of human-readable differences in non-@2 cells, empty if purely additive.

    Confirms every existing row has a computed counterpart and that all of its
    non-@2 columns are reproduced byte-for-byte. New @2 columns are expected and
    ignored. A non-empty result means the rewrite would change more than the
    cutoff-2 columns and must not be applied blindly.
    """
    diffs: list[str] = []
    existing_idx = _row_index(existing, key_fields)
    computed_idx = _row_index(computed, key_fields)

    for key, old_row in existing_idx.items():
        new_row = computed_idx.get(key)
        if new_row is None:
            diffs.append(f"row {key} present in existing summary but not recomputed")
            continue
        for col, old_val in old_row.items():
            if _is_cutoff2_column(col):
                continue
            new_val = _csv_cell(new_row.get(col))
            if new_val != old_val:
                diffs.append(f"row {key} col {col}: existing={old_val!r} recomputed={new_val!r}")

    for key in computed_idx:
        if key not in existing_idx:
            diffs.append(f"row {key} recomputed but absent from existing summary (new combo/slice)")
    return diffs


def resummarize_run(run_dir: Path, cutoffs: list[int], dry_run: bool, allow_other_changes: bool) -> dict:
    """Recompute one run's summaries from its records. Returns a status dict."""
    records_dir = run_dir / "records"
    records = eval_io.read_all_records(records_dir)
    if not records:
        return {"run": run_dir.name, "status": "skip", "reason": "no records"}

    # Vector runs key each (embedding, reranker) as its own synthetic system, the
    # same relabeling vector.py applies before aggregation. Vectorless records
    # carry no embedding_model, so this leaves their system field unchanged.
    for r in records:
        if r.get("system") and r.get("embedding_model"):
            rerank_part = r.get("reranker", "none") or "none"
            r["system"] = f"{r['system']}:{r['embedding_model']}:{rerank_part}"

    systems = sorted({row["system"] for row in records})
    granularities = sorted({row["eval_granularity"] for row in records})

    combos = compute_combo_summaries(records, systems, granularities, cutoffs)
    slices = compute_slice_summaries(records, systems, granularities, cutoffs)
    ref_modes = compute_reference_mode_breakdown(records, systems, granularities, cutoffs)
    bootstrap_ci = compute_combo_confidence_intervals(records, systems, granularities, cutoffs)

    combo_path = run_dir / "summary_by_system_granularity.csv"
    slice_path = run_dir / "summary_by_slice.csv"
    ref_path = run_dir / "summary_by_reference_mode.csv"
    overall_path = run_dir / "summary_overall.json"

    combo_diffs = check_additive(_read_existing_csv(combo_path), combos, COMBO_KEY)
    slice_diffs = check_additive(_read_existing_csv(slice_path), slices, SLICE_KEY)
    other_changes = combo_diffs + slice_diffs

    status = {
        "run": run_dir.name,
        "rows": len(combos),
        "slice_rows": len(slices),
        "added_cols": sorted({c for row in combos for c in row if _is_cutoff2_column(c)}),
        "other_changes": other_changes,
    }

    if other_changes and not allow_other_changes:
        status["status"] = "blocked"
        return status

    if dry_run:
        status["status"] = "dry"
        return status

    eval_io.write_csv(combo_path, combos)
    eval_io.write_csv(slice_path, slices)
    eval_io.write_csv(ref_path, ref_modes)

    # Rebuild summary_overall.json, preserving original run metadata and config,
    # only refreshing the metric blocks and the embedded cutoff list.
    failure_analysis: dict[str, dict] = {}
    for system in systems:
        for granularity in granularities:
            rows = [
                r for r in records
                if r["system"] == system and r["eval_granularity"] == granularity
            ]
            if rows:
                failure_analysis[f"{system}__{granularity}"] = sibling_failure_stats(rows, cutoffs)

    existing_overall = {}
    if overall_path.exists():
        with open(overall_path, encoding="utf-8") as f:
            existing_overall = json.load(f)
    config = dict(existing_overall.get("config", {}))
    config["cutoffs"] = list(cutoffs)
    overall = {
        "generated_at": existing_overall.get("generated_at"),
        "started_at": existing_overall.get("started_at"),
        "completed_at": existing_overall.get("completed_at"),
        "wall_elapsed_s": existing_overall.get("wall_elapsed_s"),
        "config": config,
        "overall": aggregate_records(records, cutoffs),
        "by_system_granularity": combos,
        "by_reference_mode": ref_modes,
        "bootstrap_ci": bootstrap_ci,
        "failure_analysis": failure_analysis,
        "error_categories": existing_overall.get("error_categories", {}),
    }
    eval_io.write_json(overall_path, overall)

    status["status"] = "wrote" + ("-with-other-changes" if other_changes else "")
    return status


def main() -> int:
    """Re-summarize the given run directories, additive by default."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="Run directories or globs.")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS,
                    help="Cutoffs to write (default 1 2 3 5 10).")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing.")
    ap.add_argument("--allow-other-changes", action="store_true",
                    help="Write even if non-@2 columns change (use to repair a stale summary).")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for pattern in args.runs:
        matches = [Path(p) for p in glob.glob(pattern)]
        run_dirs.extend(matches if matches else [Path(pattern)])

    blocked = 0
    for run_dir in sorted(run_dirs):
        if not (run_dir / "records").exists():
            continue
        r = resummarize_run(run_dir, args.cutoffs, args.dry_run, args.allow_other_changes)
        tag = r["status"].upper()
        print(f"[{tag}] {r['run']}  rows={r.get('rows', 0)} slices={r.get('slice_rows', 0)} "
              f"added={r.get('added_cols', [])}")
        for d in r.get("other_changes", []):
            print(f"    other-change: {d}")
        if r["status"] == "blocked":
            blocked += 1

    if blocked:
        print(f"\n{blocked} run(s) blocked because non-@2 columns would change. "
              f"Inspect the listed changes, re-run with --allow-other-changes to repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
