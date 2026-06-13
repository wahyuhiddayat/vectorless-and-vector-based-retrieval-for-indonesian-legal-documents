"""Generate the thesis result plots from the per-query eval records.

Produces four PDF figures for the thesis, recomputed from the records so
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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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
    """Apply the shared matplotlib style used for every thesis figure."""
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
    """Load a records JSONL file relative to the eval runs directory."""
    path = RUNS / rel
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean(rows: list[dict], key: str) -> float:
    """Mean of one metric over records, treating missing values as zero."""
    return sum((r.get(key) or 0) for r in rows) / len(rows)


def vl_records(method: str, gran: str) -> list[dict]:
    """Load vectorless records for one method and granularity.

    llm-flat runs are stored per granularity, so they resolve to a separate
    run path from the other methods.
    """
    if method == "llm-flat":
        rel = f"{LLM_FLAT_RUNS[gran]}/records/llm-flat__{gran}.jsonl"
    else:
        rel = f"{VL_RUNS[method]}/records/{method}__{gran}.jsonl"
    return load_records(rel)


def vec_records(gran: str, emb: str, rer: str) -> list[dict]:
    """Load vector records for one granularity, embedding, and reranker combination."""
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
    """Grouped bars, MAP@10 by query type for the leading config of each paradigm."""
    configs = [
        ("Vectorless\n(hybrid-tree)", lambda: vl_records("hybrid-tree", "pasal")),
        ("Vector\n(BGE-M3 + BGE v2 M3)", lambda: vec_records("pasal", "bge-m3", "bge-reranker-v2-m3")),
    ]
    types = ["factual", "paraphrased", "multihop"]
    colors = {"factual": UIBLUE, "paraphrased": GRAY, "multihop": UIRED}
    config_x = [0.0, 1.25]

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    width = 0.32
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
    ax.set_xlim(-0.7, 1.95)
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(out / "bytype.pdf")
    plt.close(fig)


def plot_cost(out: Path) -> None:
    """Two-panel cost-effectiveness scatter, MAP@10 against latency and tokens.

    Effectiveness shares the vertical axis across both panels. The left panel
    plots mean per-query latency on a log axis, the right panel plots mean
    per-query LLM tokens on a symlog axis so the token-free configurations sit
    honestly at zero. Vectorless methods are red circles, vector configurations
    are blue markers grouped by reranker, since the reranker rather than the
    embedding determines their cost. Only the token-spending methods are
    labeled, in the token panel, and the rest are carried by the legend.
    """
    vl_methods = ["bm25-flat", "bm25-tree", "hybrid-flat", "hybrid-tree", "llm-flat", "llm-tree"]
    embeds = ["bge-m3", "multilingual-e5-large-instruct", "all-nusabert-large-v4"]
    reranker_classes = [
        ("none", "Vector, no reranker", "D"),
        ("bge-reranker-v2-m3", "Vector + BGE v2 M3", "s"),
        ("qwen3-reranker-0.6b", "Vector + Qwen3 0.6B", "^"),
    ]
    # Labels go in the token panel, where the four token-spending methods spread out.
    token_labels = {
        "hybrid-flat": (0, -15, "center"),
        "hybrid-tree": (0, 9, "center"),
        "llm-tree": (10, -15, "left"),
        "llm-flat": (0, 9, "center"),
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)

    def draw(ax, xkey, do_labels):
        """Plot one cost-versus-MAP@10 panel on ax, using xkey as the x-axis metric."""
        first = True
        for method in vl_methods:
            rows = vl_records(method, "pasal")
            x, y = mean(rows, xkey), mean(rows, "map@10")
            ax.scatter(x, y, s=52, color=UIRED, marker="o", edgecolor="white",
                       linewidth=0.6, zorder=3, label="Vectorless" if first else None)
            first = False
            if do_labels and method in token_labels:
                dx, dy, ha = token_labels[method]
                ax.annotate(method, (x, y), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, fontsize=7.5, color="#333333")
        for rer_key, rer_label, marker in reranker_classes:
            first = True
            for emb in embeds:
                rows = vec_records("pasal", emb, rer_key)
                ax.scatter(mean(rows, xkey), mean(rows, "map@10"), s=52,
                           color=UIBLUE, marker=marker, edgecolor="white",
                           linewidth=0.6, zorder=3, label=rer_label if first else None)
                first = False

    draw(axL, "elapsed_s", do_labels=False)
    draw(axR, "total_tokens", do_labels=True)

    axL.set_xscale("log")
    axL.set_xticks([0.1, 1, 10])
    axL.set_xticklabels(["0.1", "1", "10"])
    axL.set_xlim(0.05, 40)
    axL.set_xlabel("Mean latency per query (s, log scale)")
    axL.set_ylabel("MAP@10")
    axL.set_ylim(0.5, 0.95)

    axR.set_xscale("symlog", linthresh=1000)
    axR.set_xticks([0, 1e4, 1e5, 1e6])
    axR.set_xticklabels(["0", "10k", "100k", "1M"])
    axR.set_xlim(-300, 3e6)
    axR.set_xlabel("Mean LLM tokens per query")

    handles, labels = axR.get_legend_handles_labels()
    axR.legend(handles, labels, frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cost-scatter.pdf")
    plt.close(fig)


def plot_tuning(out: Path) -> None:
    """Line plot, best-so-far MAP@10 across the four aligned Stage 2 tuning stages.

    Each paradigm follows the same four-stage shape, baseline, hyperparameter
    tuning, main-model swap, and query expansion. A rejected step reuses the
    previous run, so the line tracks the value actually carried forward.
    """
    stages = ["Baseline", "Hyperparameters", "Model swap", "Query expansion"]
    vl_runs = [
        "stage2_vectorless/run50_bm25topk10",
        "stage2_vectorless/run51_docpick5",
        "stage2_vectorless/run52_v4pro_topk20_docpick5",
        "stage2_vectorless/run52_v4pro_topk20_docpick5",
    ]
    vec_runs = [
        "stage2_vector/run38_topn50",
        "stage2_vector/run39_ef64_topn100",
        "stage2_vector/run39_ef64_topn100",
        "stage2_vector/run42_qe_v2m3_topn100_ef64",
    ]
    vl_rec = "records/hybrid-tree__pasal.jsonl"
    vec_rec = "records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl"
    vl_vals = [mean(load_records(f"{r}/{vl_rec}"), "map@10") for r in vl_runs]
    vec_vals = [mean(load_records(f"{r}/{vec_rec}"), "map@10") for r in vec_runs]

    x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.plot(x, vl_vals, marker="o", color=UIRED, linewidth=2,
            label="Vectorless (hybrid-tree)")
    ax.plot(x, vec_vals, marker="s", color=UIBLUE, linewidth=2,
            label="Vector (BGE-M3 + BGE v2 M3)")
    for xi, v in zip(x, vl_vals):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=UIRED)
    for xi, v in zip(x, vec_vals):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color=UIBLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0.84, 0.98)
    ax.set_xlim(-0.35, len(stages) - 0.65)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tuning-trajectory.pdf")
    plt.close(fig)


def plot_emb_reranker_heatmap(out: Path, fmt: str = "pdf") -> None:
    """Heatmap, pasal MAP@10 by embedding model and reranker for the vector paradigm.

    Rows are embedding models and columns are rerankers, recomputed from the
    Stage 1 vector records so the cells match the reported table. The fill uses
    the same blue family as the granularity bars, so the darkest cell is the
    leading configuration.
    """
    embeds = [
        ("bge-m3", "BGE-M3"),
        ("multilingual-e5-large-instruct", "Multilingual E5"),
        ("all-nusabert-large-v4", "NusaBERT"),
    ]
    rerankers = [
        ("none", "No reranker"),
        ("qwen3-reranker-0.6b", "Qwen3-Reranker\n0.6B"),
        ("bge-reranker-v2-m3", "BGE-Reranker\nv2-M3"),
    ]
    grid = [[mean(vec_records("pasal", emb, rer), "map@10") for rer, _ in rerankers]
            for emb, _ in embeds]
    flat = [v for row in grid for v in row]

    # Light to UIBLUE ramp, the same blue family the granularity bars use.
    cmap = LinearSegmentedColormap.from_list("uiblue", ["#EEF3FA", "#7E97C4", UIBLUE])

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.grid(False)
    sns.heatmap(
        grid, ax=ax, cmap=cmap, vmin=min(flat), vmax=max(flat),
        annot=True, fmt=".4f", annot_kws={"fontsize": 10}, square=True,
        linewidths=0, cbar_kws={"label": "MAP@10", "shrink": 0.85},
        xticklabels=[lab for _, lab in rerankers],
        yticklabels=[lab for _, lab in embeds],
    )

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    ax.set_xlabel("Reranker", fontweight="bold", labelpad=8)
    ax.set_ylabel("Embedding", fontweight="bold", labelpad=8)
    fig.tight_layout()
    fig.savefig(out / f"emb-reranker-heatmap.{fmt}")
    plt.close(fig)


def plot_vec_granularity(out: Path, fmt: str = "pdf") -> None:
    """Grouped bars, MAP@10 by embedding and granularity at the BGE reranker.

    The reranker is fixed to BGE-Reranker-v2-M3, the strongest of the three, so
    the figure isolates the granularity effect for each embedding. Pasal leads at
    every embedding, which mirrors the vectorless side and is why the comparison
    centers on the pasal level.
    """
    embeds = [
        ("bge-m3", "BGE-M3"),
        ("multilingual-e5-large-instruct", "Multilingual E5"),
        ("all-nusabert-large-v4", "NusaBERT"),
    ]
    reranker = "bge-reranker-v2-m3"
    grans = ["pasal", "ayat", "rincian"]
    colors = {"pasal": UIBLUE, "ayat": "#6E87B7", "rincian": "#C9D3E6"}

    emb_x = [0.0, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    width = 0.26
    for gi, gran in enumerate(grans):
        vals = [mean(vec_records(gran, emb, reranker), "map@10") for emb, _ in embeds]
        xs = [x + (gi - 1) * width for x in emb_x]
        bars = ax.bar(xs, vals, width=width, color=colors[gran], label=gran.capitalize(),
                      edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", rotation=90, padding=2, fontsize=7, color="#444444")
    ax.set_xticks(emb_x)
    ax.set_xticklabels([lab for _, lab in embeds])
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(out / f"vec-granularity.{fmt}")
    plt.close(fig)


def main() -> int:
    """Render every thesis figure into the --out directory.

    When --svg-dir is given, the two RQ2 slide figures are also written there as
    SVG for the defense deck, which uses SVG assets rather than PDF.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT.parents[1] / "laporan-skripsi" / "assets" / "figures" / "bab4"))
    ap.add_argument("--svg-dir", default=None,
                    help="If set, also write the RQ2 slide figures as SVG into this directory.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_granularity(out)
    plot_bytype(out)
    plot_cost(out)
    plot_tuning(out)
    plot_emb_reranker_heatmap(out)
    plot_vec_granularity(out)
    print(f"Wrote 6 figures to {out}")
    if args.svg_dir:
        svg_dir = Path(args.svg_dir)
        svg_dir.mkdir(parents=True, exist_ok=True)
        plot_emb_reranker_heatmap(svg_dir, fmt="svg")
        plot_vec_granularity(svg_dir, fmt="svg")
        print(f"Wrote 2 SVG slide figures to {svg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
