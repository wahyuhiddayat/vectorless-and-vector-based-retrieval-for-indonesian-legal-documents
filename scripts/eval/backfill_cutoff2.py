"""Backfill cutoff-2 metrics into existing eval records, additive and verified.

Adds the @2 metric fields (recall@2, precision@2, f1@2, ndcg@2, dcg@2, hit@2,
map@2, mrr@2, sibling_hit@2) to records that were scored before cutoff 2
existed. The values are recomputed from each record's stored ranking and gold
with the same score_ranked_retrieval function the harness uses, so the new
fields are consistent with the existing ones.

The change is strictly additive and verified per record. Before writing, the
script recomputes the existing cutoff metrics and asserts they match the stored
values exactly, then confirms that the new record contains every original
key and value unchanged and only adds @2 keys. It proves this by hashing the
original-keys projection of the record before and after. If any check fails on
any record, the file is left untouched and the run aborts.

Usage:
    python scripts/eval/backfill_cutoff2.py --dry-run data/eval_runs/run52_v4pro_topk20_docpick5
    python scripts/eval/backfill_cutoff2.py data/eval_runs/run52_v4pro_topk20_docpick5
    python scripts/eval/backfill_cutoff2.py data/eval_runs/stage1_vectorless/*
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core.metrics import score_ranked_retrieval

NEW_CUTOFF = 2
EXISTING_CUTOFFS = [1, 3, 5, 10]
TOL = 1e-9


def projection_hash(record: dict, keys: set) -> str:
    """SHA256 of the record restricted to the given keys, order-independent."""
    subset = {k: record[k] for k in keys if k in record}
    blob = json.dumps(subset, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def augment_record(record: dict) -> tuple[dict, list[str]]:
    """Return a copy of record with @2 keys added, plus the list of added keys.

    Verifies that recomputing the existing cutoffs reproduces the stored values
    before adding anything. Raises ValueError on any mismatch so a changed
    scoring function cannot silently corrupt the records.
    """
    if record.get("error"):
        return record, []  # error rows carry no ranking, leave untouched
    ranked = record.get("retrieved_node_ids") or []
    relevant = set(record.get("relevant_node_ids") or [])

    check = score_ranked_retrieval(ranked, relevant, EXISTING_CUTOFFS)
    for key, value in check.items():
        if key in record and isinstance(value, (int, float)):
            if abs(value - record[key]) > TOL:
                raise ValueError(
                    f"existing key {key} would change ({record[key]} -> {value}) "
                    f"for {record.get('query_id')}, aborting"
                )

    scored = score_ranked_retrieval(ranked, relevant, [NEW_CUTOFF])
    new_keys = [k for k in scored if k.endswith(f"@{NEW_CUTOFF}") and k not in record]
    augmented = dict(record)
    for k in new_keys:
        augmented[k] = scored[k]

    original_keys = set(record.keys())
    if projection_hash(augmented, original_keys) != projection_hash(record, original_keys):
        raise ValueError(f"original fields changed for {record.get('query_id')}, aborting")
    return augmented, new_keys


def process_file(path: Path, dry_run: bool) -> dict:
    """Backfill one combo file. Returns a verification summary dict."""
    lines_out = []
    n_records = 0
    n_augmented = 0
    added_keys: set = set()
    pre_hashes = []
    post_hashes = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            n_records += 1
            original_keys = set(record.keys())
            pre_hashes.append(projection_hash(record, original_keys))
            augmented, new_keys = augment_record(record)
            if new_keys:
                n_augmented += 1
                added_keys.update(new_keys)
            post_hashes.append(projection_hash(augmented, original_keys))
            lines_out.append(json.dumps(augmented, ensure_ascii=False))

    # File-level proof, the original-keys projection is identical for every record.
    projections_unchanged = pre_hashes == post_hashes

    if not dry_run and n_augmented > 0 and projections_unchanged:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + "\n")
        os.replace(tmp, path)

    return {
        "file": str(path),
        "records": n_records,
        "augmented": n_augmented,
        "added_keys": sorted(added_keys),
        "original_fields_unchanged": projections_unchanged,
        "written": (not dry_run and n_augmented > 0 and projections_unchanged),
    }


def main() -> int:
    """Backfill @2 metrics into the given run directories, with verification."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="Run directories or globs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify and report without writing.")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for pattern in args.runs:
        matches = [Path(p) for p in glob.glob(pattern)]
        run_dirs.extend(matches if matches else [Path(pattern)])

    any_written = False
    for run_dir in run_dirs:
        records_dir = run_dir / "records"
        if not records_dir.exists():
            continue
        for combo_path in sorted(records_dir.glob("*.jsonl")):
            try:
                r = process_file(combo_path, args.dry_run)
            except ValueError as exc:
                print(f"ABORT {combo_path}: {exc}")
                return 1
            status = "WROTE" if r["written"] else ("DRY" if args.dry_run else "skip")
            flag = "ok" if r["original_fields_unchanged"] else "FIELDS CHANGED"
            print(f"[{status}] {combo_path.parent.parent.name}/{combo_path.stem}  "
                  f"records={r['records']} augmented={r['augmented']} "
                  f"existing_fields={flag}  new={r['added_keys']}")
            any_written = any_written or r["written"]

    if args.dry_run:
        print("\nDry run, no files written. Re-run without --dry-run to apply.")
    elif any_written:
        print("\nBackfill complete. Existing fields verified unchanged on every record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
