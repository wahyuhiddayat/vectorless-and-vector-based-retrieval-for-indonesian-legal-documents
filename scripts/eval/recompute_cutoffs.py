"""Recompute cutoff metrics from stored eval records, no retrieval rerun.

Reads the per-query records of finished eval runs and recomputes ranking
metrics at any cutoff directly from the stored ranked list and gold set, using
the same score_ranked_retrieval function the harness uses at eval time. This
adds metrics like recall@2 to runs that were scored before cutoff 2 existed,
without spending any API budget.

Recall@2 is the headline multihop metric, because multihop queries require two
gold pasals, so recall@2 measures whether both were retrieved at the top. The
report splits it by multihop (two or more gold) versus single-gold queries,
since single-gold recall sits near the ceiling and the paradigms separate on
the multihop subset.

Usage:
    python scripts/eval/recompute_cutoffs.py data/eval_runs/run52_v4pro_topk20_docpick5
    python scripts/eval/recompute_cutoffs.py data/eval_runs/stage1_vectorless/*
    python scripts/eval/recompute_cutoffs.py --cutoffs 2 10 data/eval_runs/run52_v4pro_topk20_docpick5
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core.metrics import score_ranked_retrieval, safe_mean

METRIC_KEYS = ["recall", "precision", "hit", "map", "mrr"]


def load_combo_records(path: Path) -> list[dict]:
    """Load records from one combo file, deduplicated by query_id, errors dropped.

    Each file holds one (system, granularity) combo. Granularities are kept in
    separate files because their gold sets differ, so they must not be merged.
    Keeps the last record per query_id so a healed re-run replaces its earlier
    error row, matching the harness resume semantics.
    """
    records: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("query_id")
            if qid:
                records[qid] = row
    return [r for r in records.values() if not r.get("error")]


def recompute(records: list[dict], cutoffs: list[int]) -> list[dict]:
    """Recompute cutoff metrics per record from the stored ranking and gold."""
    out = []
    for r in records:
        ranked = r.get("retrieved_node_ids") or []
        relevant = set(r.get("relevant_node_ids") or [])
        scores = score_ranked_retrieval(ranked, relevant, cutoffs)
        scores["query_id"] = r.get("query_id")
        scores["num_gold"] = len(relevant)
        out.append(scores)
    return out


def aggregate(scored: list[dict], cutoffs: list[int]) -> dict:
    """Mean of each cutoff metric over a set of scored records."""
    agg = {"n": len(scored)}
    for k in cutoffs:
        for m in METRIC_KEYS:
            key = f"{m}@{k}"
            agg[key] = safe_mean([s.get(key, 0.0) for s in scored])
    return agg


def report_combo(path: Path, cutoffs: list[int], focus_k: int) -> None:
    """Print recompute results for one combo file, overall and by multihop split."""
    records = load_combo_records(path)
    if not records:
        print(f"{path.name}: no records")
        return
    scored = recompute(records, cutoffs)
    multihop = [s for s in scored if s["num_gold"] >= 2]
    single = [s for s in scored if s["num_gold"] == 1]

    overall = aggregate(scored, cutoffs)
    combo = path.stem
    print(f"\n{path.parent.parent.name} / {combo}  "
          f"(n={overall['n']}, multihop={len(multihop)}, single={len(single)})")
    header = "subset       n    " + "".join(f"recall@{k:<4}" for k in cutoffs)
    print(header)
    for name, subset in [("overall", scored), ("multihop", multihop), ("single", single)]:
        if not subset:
            continue
        a = aggregate(subset, cutoffs)
        cells = "".join(f"{a[f'recall@{k}']:<11.4f}" for k in cutoffs)
        print(f"{name:<12} {a['n']:<4} {cells}")

    # Focused multihop view at the headline cutoff, recall equals precision here.
    if multihop:
        a = aggregate(multihop, [focus_k])
        print(f"  multihop recall@{focus_k} = {a[f'recall@{focus_k}']:.4f}  "
              f"(precision@{focus_k} = {a[f'precision@{focus_k}']:.4f}, "
              f"equal on 2-gold queries)")


def main() -> int:
    """Recompute and print cutoff metrics for the given run directories."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="Run directories or globs.")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                    help="Cutoffs to compute (default 1 2 3 5 10).")
    ap.add_argument("--focus-k", type=int, default=2,
                    help="Headline cutoff for the multihop view (default 2).")
    ap.add_argument("--granularity", default=None,
                    help="Only report this granularity (e.g. pasal). Default all.")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for pattern in args.runs:
        matches = [Path(p) for p in glob.glob(pattern)]
        run_dirs.extend(matches if matches else [Path(pattern)])

    for run_dir in run_dirs:
        records_dir = run_dir / "records"
        if not records_dir.exists():
            print(f"{run_dir}: no records directory, skipping")
            continue
        for combo_path in sorted(records_dir.glob("*.jsonl")):
            if args.granularity and not combo_path.stem.endswith(f"__{args.granularity}"):
                continue
            report_combo(combo_path, args.cutoffs, args.focus_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
