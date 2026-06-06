"""Generate the chapter 4 result plots from the per-query eval records.

Produces three PDF figures for the thesis, recomputed from the records so
they stay consistent with the reported tables. Errored queries count as
zero, matching the table convention.

Usage:
    python scripts/eval/plots_chapter4.py --out "../laporan-skripsi/assets/figures/bab4"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "data" / "eval_runs"

UIBLUE = "#284887"
UIRED = "#962D30"
GRAY = "#8C8C8C"

VL_RUNS = {
    "bm25-flat": "stage1_vectorless/run23_20260520_vectorless_dev_357q_bm25_flat",
    "bm25-tree": "stage1_vectorless/run24_20260520_vectorless_dev_357q_bm25_tree",
    "hybrid-flat": "stage1_vectorless/run25_20260524_vectorless_dev_357q_hybrid_flat_deepseek",
    "hybrid-tree": "stage1_vectorless/run26_20260524_vectorless_dev_357q_hybrid_tree_deepseek",
    "llm-tree": "stage1_vectorless/run27_20260525_vectorless_dev_357q_llm_tree_deepseek",
}
LLM_FLAT_RUNS = {
    "ayat": "stage1_vectorless/run28a_20260526_vectorless_dev_357q_llm_flat_ayat_deepseek",
    "pasal": "stage1_vectorless/run28b_20260526_vectorless_dev_357q_llm_flat_pasal_deepseek",
    "rincian": "stage1_vectorless/run28c_20260527_vectorless_dev_357q_llm_flat_rincian_deepseek",
}
VECTOR_RUN = "stage1_vector/run29_20260526_vector_dev_357q_full"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
    })


def load_records(rel: str) -> list[dict]:
    path = RUNS / rel
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean(rows: list[dict], key: str) -> float:
    return sum((r.get(key) or 0) for r in rows) / len(rows)


def vl_records(method: str, gran: str) -> list[dict]:
    if method == "llm-flat":
        rel = f"{LLM_FLAT_RUNS[gran]}/records/llm-flat__{gran}.jsonl"
    else:
        rel = f"{VL_RUNS[method]}/records/{method}__{gran}.jsonl"
    return load_records(rel)


def vec_records(gran: str, emb: str, rer: str) -> list[dict]:
    rel = f"{VECTOR_RUN}/records/vector-dense__{gran}__{emb}__{rer}.jsonl"
    return load_records(rel)


def plot_granularity(out: Path) -> None:
    """Grouped bars, MAP@10 by vectorless method and granularity."""
    methods = ["bm25-flat", "bm25-tree", "hybrid-flat", "hybrid-tree", "llm-flat", "llm-tree"]
    grans = ["pasal", "ayat", "rincian"]
    colors = {"pasal": UIBLUE, "ayat": "#7C93BE", "rincian": "#C5CFE2"}

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    width = 0.26
    for gi, gran in enumerate(grans):
        vals = [mean(vl_records(m, gran), "map@10") for m in methods]
        xs = [i + (gi - 1) * width for i in range(len(methods))]
        ax.bar(xs, vals, width=width, color=colors[gran], label=gran.capitalize(),
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "vl-granularity.pdf")
    plt.close(fig)


def plot_bytype(out: Path) -> None:
    """Grouped bars, MAP@10 by query type for the leading pasal configurations."""
    configs = [
        ("bm25-flat", lambda: vl_records("bm25-flat", "pasal")),
        ("hybrid-flat", lambda: vl_records("hybrid-flat", "pasal")),
        ("hybrid-tree", lambda: vl_records("hybrid-tree", "pasal")),
        ("llm-flat", lambda: vl_records("llm-flat", "pasal")),
        ("llm-tree", lambda: vl_records("llm-tree", "pasal")),
        ("vector", lambda: vec_records("pasal", "bge-m3", "bge-reranker-v2-m3")),
    ]
    types = ["factual", "paraphrased", "multihop"]
    colors = {"factual": UIBLUE, "paraphrased": GRAY, "multihop": UIRED}

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    width = 0.26
    for ti, qt in enumerate(types):
        vals = []
        for _, loader in configs:
            rows = [r for r in loader() if r.get("query_type") == qt]
            vals.append(mean(rows, "map@10"))
        xs = [i + (ti - 1) * width for i in range(len(configs))]
        ax.bar(xs, vals, width=width, color=colors[qt], label=qt.capitalize(),
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c for c, _ in configs])
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(out / "bytype.pdf")
    plt.close(fig)


def plot_cost(out: Path) -> None:
    """Scatter, MAP@10 against mean per-query latency on a log axis."""
    points = []
    for method in ["bm25-flat", "bm25-tree", "hybrid-flat", "hybrid-tree", "llm-flat", "llm-tree"]:
        rows = vl_records(method, "pasal")
        points.append((method, mean(rows, "elapsed_s"), mean(rows, "map@10"), UIRED))
    for rer, label in [("none", "vector, no reranker"), ("bge-reranker-v2-m3", "vector + reranker")]:
        rows = vec_records("pasal", "bge-m3", rer)
        points.append((label, mean(rows, "elapsed_s"), mean(rows, "map@10"), UIBLUE))

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    offsets = {
        "bm25-flat": (0, 9, "center"), "bm25-tree": (0, 9, "center"),
        "hybrid-flat": (-6, 9, "center"), "hybrid-tree": (8, 9, "left"),
        "llm-flat": (4, -15, "left"), "llm-tree": (-8, -4, "right"),
        "vector, no reranker": (0, 9, "center"), "vector + reranker": (0, 9, "center"),
    }
    for name, x, y, color in points:
        ax.scatter(x, y, s=42, color=color, zorder=3)
        dx, dy, ha = offsets[name]
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per query (s, log scale)")
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.05, 40)
    fig.tight_layout()
    fig.savefig(out / "cost-scatter.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT.parents[1] / "laporan-skripsi" / "assets" / "figures" / "bab4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_granularity(out)
    plot_bytype(out)
    plot_cost(out)
    print(f"Wrote 3 figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
