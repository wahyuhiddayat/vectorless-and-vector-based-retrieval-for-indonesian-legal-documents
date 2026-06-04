"""Regenerate every number in the thesis corpus-statistics table (Chapter 3).

Runs leaf_text_lengths.py and tree_depth_stats.py, then assembles their
outputs into the exact rows of the indexed-corpus statistics table, namely
leaf counts, mean leaves per document, text-length distribution, tree depth,
and branching factor per granularity. Use this to recheck the thesis table
whenever the index changes.

Usage:
    python scripts/analysis/corpus_stats_table.py
"""

import json
import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path("scripts/analysis")
LEAF_OUTPUT = ANALYSIS_DIR / "leaf_text_lengths_output.json"
TREE_OUTPUT = ANALYSIS_DIR / "tree_depth_stats_output.json"
TABLE_OUTPUT = ANALYSIS_DIR / "corpus_stats_table_output.json"
GRANULARITIES = ("pasal", "ayat", "rincian")


def run_component(script: str) -> None:
    """Run one underlying analysis script so its JSON output is fresh."""
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, str(ANALYSIS_DIR / script)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed:\n{result.stderr}")


def main() -> None:
    """Regenerate component outputs and print the corpus-statistics table."""
    run_component("leaf_text_lengths.py")
    run_component("tree_depth_stats.py")

    leaf = json.loads(LEAF_OUTPUT.read_text(encoding="utf-8"))
    tree = json.loads(TREE_OUTPUT.read_text(encoding="utf-8"))

    table = {}
    for g in GRANULARITIES:
        n_docs = tree[g]["n_docs"]
        n_leaves = leaf[g]["n_leaves"]
        table[g] = {
            "n_docs": n_docs,
            "leaf_count": n_leaves,
            "mean_leaves_per_doc": round(n_leaves / n_docs, 1),
            "mean_text_length": leaf[g]["avg"],
            "median_text_length": leaf[g]["median"],
            "p95_text_length": leaf[g]["p95"],
            "p99_text_length": leaf[g]["p99"],
            "max_text_length": leaf[g]["max"],
            "mean_tree_depth": tree[g]["avg_depth"],
            "p95_tree_depth": tree[g]["p95_depth"],
            "max_tree_depth": tree[g]["max_depth"],
            "mean_branching": tree[g]["avg_branching"],
            "p95_branching": tree[g]["p95_branching"],
            "max_branching": tree[g]["max_branching"],
        }

    TABLE_OUTPUT.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    rows = [
        ("Documents", "n_docs"),
        ("Leaf count", "leaf_count"),
        ("Mean leaves per document", "mean_leaves_per_doc"),
        ("Mean text length (chars)", "mean_text_length"),
        ("Median text length (chars)", "median_text_length"),
        ("p95 text length (chars)", "p95_text_length"),
        ("p99 text length (chars)", "p99_text_length"),
        ("Max text length (chars)", "max_text_length"),
        ("Mean tree depth", "mean_tree_depth"),
        ("p95 tree depth", "p95_tree_depth"),
        ("Max tree depth", "max_tree_depth"),
        ("Mean branching factor", "mean_branching"),
        ("p95 branching factor", "p95_branching"),
        ("Max branching factor", "max_branching"),
    ]
    header = f"{'Statistic':<28}" + "".join(f"{g.capitalize():>10}" for g in GRANULARITIES)
    print()
    print(header)
    print("-" * len(header))
    for label, key in rows:
        print(f"{label:<28}" + "".join(f"{table[g][key]:>10}" for g in GRANULARITIES))
    print()
    print(f"Written to {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
