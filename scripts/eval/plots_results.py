"""Generate the thesis result plots from the per-query eval records.

Produces three PDF figures for the thesis, recomputed from the records so
they stay consistent with the reported tables. Errored queries count as
zero, matching the table convention.

Usage:
    python scripts/eval/plots_results.py --out "../laporan-skripsi/assets/figures/bab4"
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
    # Extra horizontal gap between the lexical, hybrid, and LLM-based pairs
    # so the three family bands are visible in the figure itself.
    method_x = [0.0, 1.0, 2.45, 3.45, 4.9, 5.9]
    grans = ["pasal", "ayat", "rincian"]
    colors = {"pasal": UIBLUE, "ayat": "#6E87B7", "rincian": "#C9D3E6"}

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    width = 0.26
    for gi, gran in enumerate(grans):
        vals = [mean(vl_records(m, gran), "map@10") for m in methods]
        xs = [x + (gi - 1) * width for x in method_x]
        bars = ax.bar(xs, vals, width=width, color=colors[gran], label=gran.capitalize(),
                      edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", rotation=90, padding=2, fontsize=7, color="#444444")
    ax.set_xticks(method_x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
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
    # Family gaps within the vectorless side and a wider gap before the
    # vector configuration to separate the two paradigms.
    config_x = [0.0, 1.45, 2.45, 3.9, 4.9, 6.55]

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    width = 0.26
    for ti, qt in enumerate(types):
        vals = []
        for _, loader in configs:
            rows = [r for r in loader() if r.get("query_type") == qt]
            vals.append(mean(rows, "map@10"))
        xs = [x + (ti - 1) * width for x in config_x]
        bars = ax.bar(xs, vals, width=width, color=colors[qt], label=qt.capitalize(),
                      edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", rotation=90, padding=2, fontsize=7, color="#444444")
    ax.set_xticks(config_x)
    ax.set_xticklabels([c for c, _ in configs])
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(out / "bytype.pdf")
    plt.close(fig)


def plot_cost(out: Path) -> None:
    """Scatter, MAP@10 against mean per-query latency on a log axis.

    Vectorless methods are red circles with name labels. Vector
    configurations are blue markers grouped by reranker, since the
    reranker rather than the embedding determines their cost.
    """
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    vl_offsets = {
        "bm25-flat": (0, 9, "center"), "bm25-tree": (0, 9, "center"),
        "hybrid-flat": (-6, 9, "center"), "hybrid-tree": (8, 9, "left"),
        "llm-flat": (4, -16, "left"), "llm-tree": (-9, -5, "right"),
    }
    first = True
    for method in ["bm25-flat", "bm25-tree", "hybrid-flat", "hybrid-tree", "llm-flat", "llm-tree"]:
        rows = vl_records(method, "pasal")
        x, y = mean(rows, "elapsed_s"), mean(rows, "map@10")
        ax.scatter(x, y, s=58, color=UIRED, marker="o", edgecolor="white",
                   linewidth=0.6, zorder=3, label="Vectorless" if first else None)
        dx, dy, ha = vl_offsets[method]
        ax.annotate(method, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8, color="#333333")
        first = False

    embeds = ["bge-m3", "multilingual-e5-large-instruct", "all-nusabert-large-v4"]
    reranker_classes = [
        ("none", "Vector, no reranker", "D"),
        ("bge-reranker-v2-m3", "Vector + BGE v2 M3", "s"),
        ("qwen3-reranker-0.6b", "Vector + Qwen3 0.6B", "^"),
    ]
    for rer_key, rer_label, marker in reranker_classes:
        first = True
        for emb in embeds:
            rows = vec_records("pasal", emb, rer_key)
            ax.scatter(mean(rows, "elapsed_s"), mean(rows, "map@10"), s=58,
                       color=UIBLUE, marker=marker, edgecolor="white",
                       linewidth=0.6, zorder=3, label=rer_label if first else None)
            first = False

    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_xlabel("Mean latency per query (s, log scale)")
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0.05, 40)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
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
