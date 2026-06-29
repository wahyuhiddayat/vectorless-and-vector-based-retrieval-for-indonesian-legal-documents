"""Generate the thesis result plots from the per-query eval records.

Produces six PDF figures for the thesis, recomputed from the records so
they stay consistent with the reported tables. Errored queries count as
zero, matching the table convention.

Usage:
    python scripts/figures/plots_results.py --out "../laporan-skripsi/assets/figures/bab4"
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

# Palette ported from the Bab 3 TikZ figures (config/diagram-style.tex) so the
# Python plots share one visual language with the LaTeX diagrams. Role colors:
# blue = lexical/BM25, red = LLM-driven, teal = neural encoder, slate = ink.
C_BM25 = "#2E60A0"   # cBm25, RGB(46,96,160)
C_LLM = "#B0343A"    # cLlm, RGB(176,52,58)
C_ENC = "#268C6E"    # cEnc, RGB(38,140,110)
C_PROC = "#7A5A9E"   # cProc, RGB(122,90,158)
C_STORE = "#C48C2E"  # cStore, RGB(196,140,46)
C_INK = "#2D3440"    # cInk, RGB(45,52,64)
INK_SOFT = "#5A6473"  # cInk at reduced strength, for spines and secondary text
GRID_C = "#E4E6EA"   # cInk at low opacity, for gridlines

# Paradigm convention, kept from the earlier figures but recolored to the exact
# Bab 3 hues. Vectorless is the LLM-driven paradigm, so it takes the red role.
UIBLUE = C_BM25      # vector paradigm and lexical/ordinal blue ramps
UIRED = C_LLM        # vectorless paradigm
GRAY = INK_SOFT


def tint(hex_color: str, frac: float) -> str:
    """Mix a hex color toward white by frac, where 0 keeps it and 1 is white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * frac) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


# The Bab 3 diagrams and these plots share a palette, fonts, and restraint, but
# data plots follow their own grammar. Bars and markers use solid role-color
# fills so magnitude reads on the strongest visual channel, with a thin white
# separator for crisp edges.
def bar_style(role: str) -> dict:
    """Solid-fill bar styling with a thin white separator."""
    return dict(color=role, edgecolor="white", linewidth=0.6)


# Ordinal granularity ramp as a solid sequential blue, dark to light, so the
# three levels read as one ordered scale.
GRAN_FILL = {"pasal": 0.0, "ayat": 0.34, "rincian": 0.60}


def gran_style(level: str, role: str = C_BM25) -> dict:
    """Solid bar styling for one granularity level on the ordinal blue ramp."""
    return dict(color=tint(role, GRAN_FILL[level]), edgecolor="white", linewidth=0.6)


def pt_style(role: str, marker: str = "o") -> dict:
    """Solid line and marker styling with a thin white marker edge."""
    return dict(marker=marker, color=role, markerfacecolor=role,
                markeredgecolor="white", markeredgewidth=0.8)

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
        "text.color": C_INK,
        "axes.labelcolor": C_INK,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.7,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelcolor": C_INK,
        "ytick.labelcolor": C_INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": tint(C_INK, 0.90),
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.fontsize": 8,
        "figure.dpi": 120,
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


def plot_granularity(out: Path, fmt: str = "pdf") -> None:
    """Line plots, MAP@10 by granularity for the BM25 baseline and vectorless methods.

    The methods come in a flat and a hierarchical form, so the figure splits into
    two panels by form, each comparing the families across granularity, the
    bm25-flat lexical baseline in gray, the hybrid methods in red, and the LLM-based
    methods in violet. BM25 is the baseline, so its single carried-forward
    configuration, bm25-flat, is shown as the reference floor in both panels rather
    than its per-form variant. Every family peaks at pasal and declines toward
    rincian, with the gray baseline falling the most steeply.
    """
    families = [
        ("bm25", "BM25 baseline", INK_SOFT),
        ("hybrid", "Hybrid", C_LLM),
        ("llm", "LLM-based", C_PROC),
    ]
    forms = [("flat", "Flat"), ("tree", "Hierarchical (tree)")]
    grans = ["pasal", "ayat", "rincian"]
    x = list(range(len(grans)))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True)
    for ax, (form, ftitle) in zip(axes, forms):
        for fam, flabel, color in families:
            # BM25 is the baseline, shown as the single bm25-flat reference in both panels.
            method = "bm25-flat" if fam == "bm25" else f"{fam}-{form}"
            vals = [mean(vl_records(method, g), "map@10") for g in grans]
            ax.plot(x, vals, linewidth=1.8, markersize=6, label=flabel, **pt_style(color, "o"))
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in grans])
        ax.set_xlabel("Granularity")
        ax.set_title(ftitle, fontsize=9, color=C_INK)
        ax.set_xlim(-0.25, 2.25)
    axes[0].set_ylabel("MAP@10")
    axes[0].set_ylim(0.30, 0.97)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"vl-granularity.{fmt}")
    plt.close(fig)


def plot_bytype(out: Path, fmt: str = "pdf") -> None:
    """Dumbbell plot, MAP@10 by query type for the leading config of each paradigm.

    Each query type is one row with the two paradigms as connected dots, so the
    complementary pattern is read directly. The paradigms tie on factual queries,
    the vector dot leads on paraphrased, and the vectorless dot leads on multihop,
    a crossing that grouped bars of these near-equal values would flatten.
    """
    configs = [
        ("Vectorless (hybrid-tree)", lambda: vl_records("hybrid-tree", "pasal"), UIRED, 9),
        ("Vector (BGE-M3 + BGE v2 M3)", lambda: vec_records("pasal", "bge-m3", "bge-reranker-v2-m3"), UIBLUE, -15),
    ]
    types = ["factual", "paraphrased", "multihop"]
    ys = [2, 1, 0]

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for y, qt in zip(ys, types):
        vl_v, vec_v = (mean([r for r in loader() if r.get("query_type") == qt], "map@10")
                       for _, loader, _, _ in configs)
        ax.plot([vl_v, vec_v], [y, y], color=tint(C_INK, 0.65), linewidth=1.6, zorder=1)
        ax.scatter(vl_v, y, s=80, color=UIRED, edgecolor="white", linewidth=0.9, zorder=3)
        ax.scatter(vec_v, y, s=80, color=UIBLUE, edgecolor="white", linewidth=0.9, zorder=3)
        delta = vl_v - vec_v
        if abs(delta) < 0.005:
            # Near-tie, the dots overlap, so one centered label avoids duplication.
            ax.annotate(f"{vl_v:.3f}", ((vl_v + vec_v) / 2, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.5, color=C_INK)
            dcolor, dtext = INK_SOFT, "tie"
        else:
            ax.annotate(f"{vl_v:.3f}", (vl_v, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=UIRED)
            ax.annotate(f"{vec_v:.3f}", (vec_v, y), textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=7.5, color=UIBLUE)
            dcolor = UIRED if delta > 0 else UIBLUE
            dtext = f"{delta:+.3f}"
        ax.annotate(dtext, (max(vl_v, vec_v), y), textcoords="offset points", xytext=(12, 0),
                    ha="left", va="center", fontsize=8, color=dcolor, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([t.capitalize() for t in types])
    ax.set_ylabel("Query type")
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(0.80, 1.0)
    ax.set_xlabel("MAP@10")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8, color=c,
                          markeredgecolor="white", label=lbl) for lbl, _, c, _ in configs]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"bytype.{fmt}")
    plt.close(fig)


def plot_cost(out: Path, fmt: str = "pdf") -> None:
    """Two-panel cost-effectiveness scatter, MAP@10 against latency and tokens.

    Effectiveness shares the vertical axis across both panels. The left panel
    plots mean per-query latency on a log axis, the right panel plots mean
    per-query LLM tokens on a symlog axis so the token-free configurations sit
    honestly at zero. Marker shape separates the groups, the lexical baseline by
    BM25 (v), the vectorless paradigm by Hybrid (o) and LLM-based (P), and the
    vector paradigm by reranker class (D, s, ^). Color separates the three,
    gray for the lexical baseline, red for vectorless, blue for vector, matching
    the cost tiers of the text. The BM25 baseline and the four vectorless methods
    are labeled in the latency panel, and only the four token-spending vectorless
    methods are labeled in the token panel where they spread out horizontally.
    """
    point_groups = [
        (["bm25-flat", "bm25-tree"], "v", INK_SOFT, "Lexical baseline (BM25)"),
        (["hybrid-flat", "hybrid-tree"], "o", UIRED, "Vectorless (Hybrid)"),
        (["llm-flat", "llm-tree"], "P", UIRED, "Vectorless (LLM-based)"),
    ]
    embeds = ["bge-m3", "multilingual-e5-large-instruct", "all-nusabert-large-v4"]
    reranker_classes = [
        ("none", "Vector, no reranker", "D"),
        ("bge-reranker-v2-m3", "Vector + BGE v2 M3", "s"),
        ("qwen3-reranker-0.6b", "Vector + Qwen3 0.6B", "^"),
    ]
    # Labels for the latency panel: all 6 vectorless methods identified by name.
    latency_labels = {
        "bm25-flat":   (0, -15, "center"),
        "bm25-tree":   (0,   9, "center"),
        "hybrid-flat": (-8,  6, "right"),
        "hybrid-tree": ( 8,  6, "left"),
        "llm-tree":    (-8, -15, "right"),
        "llm-flat":    ( 8, -15, "left"),
    }
    # Labels for the token panel: only the 4 token-spending methods spread out.
    token_labels = {
        "hybrid-flat": (0, -15, "center"),
        "hybrid-tree": (0,   9, "center"),
        "llm-tree":    (10, -15, "left"),
        "llm-flat":    (0,   9, "center"),
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)

    def draw(ax, xkey, labels=None):
        """Plot one cost-versus-MAP@10 panel on ax, using xkey as the x-axis metric."""
        for methods, marker, color, legend_label in point_groups:
            first = True
            for method in methods:
                rows = vl_records(method, "pasal")
                x, y = mean(rows, xkey), mean(rows, "map@10")
                ax.scatter(x, y, s=58, color=color, marker=marker, edgecolor="white",
                           linewidth=0.8, zorder=3, label=legend_label if first else None)
                first = False
                if labels and method in labels:
                    dx, dy, ha = labels[method]
                    ax.annotate(method, (x, y), textcoords="offset points", xytext=(dx, dy),
                                ha=ha, fontsize=7.5, color=C_INK)
        for rer_key, rer_label, marker in reranker_classes:
            first = True
            for emb in embeds:
                rows = vec_records("pasal", emb, rer_key)
                ax.scatter(mean(rows, xkey), mean(rows, "map@10"), s=58,
                           color=UIBLUE, marker=marker, edgecolor="white",
                           linewidth=0.8, zorder=3, label=rer_label if first else None)
                first = False

    draw(axL, "elapsed_s", labels=latency_labels)
    draw(axR, "total_tokens", labels=token_labels)

    axL.set_xscale("log")
    axL.set_xticks([0.1, 1, 10])
    axL.set_xticklabels(["0.1", "1", "10"])
    axL.set_xlim(0.05, 40)
    axL.set_xlabel("Mean latency per query (seconds, log scale)")
    axL.set_ylabel("MAP@10")
    axL.set_ylim(0.5, 0.95)

    axR.set_xscale("symlog", linthresh=1000)
    axR.set_xticks([0, 1e4, 1e5, 1e6])
    axR.set_xticklabels(["0", "10k", "100k", "1M"])
    axR.set_xlim(-300, 3e6)
    axR.set_xlabel("Mean LLM tokens per query (symlog; 0 = no LLM call)")

    handles, labels = axR.get_legend_handles_labels()
    axR.legend(handles, labels, frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"cost-scatter.{fmt}")
    plt.close(fig)


def plot_tuning(out: Path, fmt: str = "pdf") -> None:
    """Line plot, best-so-far MAP@10 across the four aligned Stage 2 tuning stages.

    Each paradigm follows the same four-stage shape, baseline, hyperparameter
    tuning, main-model swap, and query expansion. A rejected step reuses the
    previous run, so the line tracks the value actually carried forward.
    """
    stages = ["Default", "Hyperparameter\ntuning", "Model swap", "Query expansion"]
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
    ax.plot(x, vl_vals, linewidth=1.8, markersize=7, label="Vectorless (hybrid-tree)",
            **pt_style(UIRED, "o"))
    ax.plot(x, vec_vals, linewidth=1.8, markersize=7, label="Vector (BGE-M3 + BGE v2 M3)",
            **pt_style(UIBLUE, "s"))
    for xi, v in zip(x, vl_vals):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=UIRED)
    for xi, v in zip(x, vec_vals):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color=UIBLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlabel("Improvement step")
    ax.set_ylabel("MAP@10")
    ax.set_ylim(0.84, 0.98)
    ax.set_xlim(-0.35, len(stages) - 0.65)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"tuning-trajectory.{fmt}")
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
        ("qwen3-reranker-0.6b", "Qwen3 0.6B"),
        ("bge-reranker-v2-m3", "BGE v2 M3"),
    ]
    grid = [[mean(vec_records("pasal", emb, rer), "map@10") for rer, _ in rerankers]
            for emb, _ in embeds]
    flat = [v for row in grid for v in row]

    # Light to UIBLUE ramp, the same blue family the granularity bars use.
    cmap = LinearSegmentedColormap.from_list("uiblue", [tint(UIBLUE, 0.93), tint(UIBLUE, 0.50), UIBLUE])

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
    """Line plot, MAP@10 by granularity for each embedding at the BGE reranker.

    The reranker is fixed to BGE-Reranker-v2-M3, the strongest of the three, so
    the figure isolates the granularity effect for each embedding. Every embedding
    peaks at pasal and declines through ayat to rincian, mirroring the vectorless
    side and motivating the pasal-level comparison. A line encodes the ordered
    granularity axis more directly than grouped bars.
    """
    embeds = [
        ("bge-m3", "BGE-M3", C_BM25, "o"),
        ("multilingual-e5-large-instruct", "Multilingual E5", C_ENC, "s"),
        ("all-nusabert-large-v4", "NusaBERT", C_PROC, "^"),
    ]
    reranker = "bge-reranker-v2-m3"
    grans = ["pasal", "ayat", "rincian"]
    x = list(range(len(grans)))

    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for key, label, color, marker in embeds:
        vals = [mean(vec_records(g, key, reranker), "map@10") for g in grans]
        ax.plot(x, vals, linewidth=1.8, markersize=7, label=label, **pt_style(color, marker))
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in grans])
    ax.set_xlabel("Granularity")
    ax.set_ylabel("MAP@10")
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(0.45, 0.95)
    ax.legend(frameon=False, loc="upper right", fontsize=8, title="Embedding", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"vec-granularity.{fmt}")
    plt.close(fig)


STAGE3 = "stage3_test"
TEST_VL = f"{STAGE3}/rq4_test_hybrid_tree/records/hybrid-tree__pasal.jsonl"
TEST_VEC = f"{STAGE3}/rq4_test_v2m3_qe/records/vector-dense__pasal__bge-m3__bge-reranker-v2-m3.jsonl"
TEST_BM25 = f"{STAGE3}/test_bm25/records/bm25-flat__pasal.jsonl"


def load_json(rel: str) -> dict:
    """Load a JSON file relative to the eval runs directory."""
    with open(RUNS / rel, encoding="utf-8") as f:
        return json.load(f)


def plot_hypothesis_forest(out: Path, fmt: str = "pdf") -> None:
    """Forest plot of the eight held-out hypothesis tests.

    Each row shows the effect of one pre-registered comparison as the metric
    difference with its 95 percent bootstrap interval, with the Holm-adjusted
    decision encoded by color and fill. Hypotheses 1 to 7 are MAP@10 differences
    and Hypothesis 8 is the multihop R@2 difference of differences, all on a
    metric-probability scale. Supported tests are blue, the refuted Hypothesis 4
    is red, and the unsupported Hypothesis 6 is hollow gray.
    """
    labels = {
        1: "Vectorless vs BM25", 2: "Vector vs BM25", 3: "Reranker vs none",
        4: "Specialized vs multilingual", 5: "Improved vs default (vectorless)",
        6: "Improved vs default (vector)", 7: "Vectorless vs vector",
        8: "Multihop margin (R@2)",
    }
    family = load_json(f"{STAGE3}/hypotheses/_family.json")["holm"]
    rows = []
    for i in range(1, 9):
        h = load_json(f"{STAGE3}/hypotheses/H{i}.json")
        ci = h["bootstrap_ci"]
        if i == 8:
            delta, eff = h["contrast"], f"$\\delta$={h['cliffs_delta']['delta']:.2f}"
        else:
            delta, eff = h["mean_diff"], f"d={h['cohens_d']['d']:.2f}"
        reject = family[f"H{i}"]["reject"]
        if not reject:
            outcome = "Not supported"
        elif delta > 0:
            outcome = "Supported"
        else:
            outcome = "Refuted"
        rows.append((i, delta, ci["low"], ci["high"], eff, outcome))

    style = {
        "Supported": dict(color=C_BM25, mfc=C_BM25),
        "Refuted": dict(color=C_LLM, mfc=C_LLM),
        "Not supported": dict(color=INK_SOFT, mfc="white"),
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK_SOFT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ys = list(range(len(rows)))[::-1]
    # H1 to H7 compare MAP@10; H8 is an R@2 difference of differences. Separate it
    # so the shared x-axis is not read as one metric.
    ax.axhline(0.5, color=INK_SOFT, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.annotate("MAP@10", xy=(-0.2, 7.3), va="center", ha="left", fontsize=7,
                color=INK_SOFT, style="italic")
    ax.annotate("R@2", xy=(-0.2, 0.32), va="center", ha="left", fontsize=7,
                color=INK_SOFT, style="italic")
    for y, (i, delta, lo, hi, eff, outcome) in zip(ys, rows):
        st = style[outcome]
        ax.plot([lo, hi], [y, y], color=st["color"], linewidth=1.6, zorder=2,
                solid_capstyle="round")
        ax.plot([delta], [y], marker="o", markersize=7, color=st["color"],
                markerfacecolor=st["mfc"], markeredgewidth=1.4, zorder=3)
        ax.annotate(eff, (hi, y), textcoords="offset points", xytext=(8, 0),
                    va="center", ha="left", fontsize=7.5, color=C_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"H{i}  {labels[i]}" for i, *_ in rows])
    ax.set_xlabel("Difference in metric, 95% CI (MAP@10; H8 in R@2)")
    ax.set_xlim(-0.2, 0.42)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                          color=st["color"], markerfacecolor=st["mfc"],
                          markeredgewidth=1.4, label=name)
               for name, st in style.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.5,
              title="Outcome", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / f"hypothesis-forest.{fmt}")
    plt.close(fig)


def plot_rank_distribution(out: Path, fmt: str = "pdf") -> None:
    """Cumulative Hit@k curve of the gold node on the test partition.

    For each rank cutoff k, the curve gives the fraction of queries whose first
    relevant node falls within the top k. The two improved configurations start
    apart at k equals one and converge to the same value by k equals ten, so the
    vectorless lead is one of ordering at the top of the ranking rather than of
    coverage within the top ten. BM25 trails throughout and tops out lower.
    """
    series = [
        ("Vectorless (improved hybrid-tree)", TEST_VL, UIRED, "o"),
        ("Vector (improved BGE-M3 + reranker)", TEST_VEC, UIBLUE, "s"),
        ("BM25 baseline", TEST_BM25, INK_SOFT, "^"),
    ]
    ks = list(range(1, 11))

    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    hit1 = {}
    for label, path, color, marker in series:
        rows = load_records(path)
        n = len(rows)
        vals = [sum(1 for r in rows if (r.get("first_relevant_rank") or 99) <= k) / n
                for k in ks]
        hit1[label] = vals[0]
        ax.plot(ks, vals, linewidth=1.8, markersize=6, label=label, **pt_style(color, marker))
    # The two paradigms converge by k=10, so annotate the k=1 gap where they differ.
    top = hit1["Vectorless (improved hybrid-tree)"]
    bot = hit1["Vector (improved BGE-M3 + reranker)"]
    ax.annotate("", xy=(1, top), xytext=(1, bot),
                arrowprops=dict(arrowstyle="<->", color=C_INK, lw=0.8))
    ax.annotate(f"+{top - bot:.3f} at k=1", xy=(1, (top + bot) / 2),
                textcoords="offset points", xytext=(8, 0), va="center", ha="left",
                fontsize=7.5, color=C_INK)
    ax.set_xticks(ks)
    ax.set_xlabel("Rank cutoff $k$")
    ax.set_ylabel("Hit@$k$ (fraction of queries)")
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"rank-distribution.{fmt}")
    plt.close(fig)


def plot_multihop_breakdown(out: Path, fmt: str = "pdf") -> None:
    """Stacked bars of the multihop outcome breakdown on the test partition.

    The 78 multihop test queries are split into both anchors placed in the top
    two, both anchors retrieved within the top ten but ranked below the top two,
    and at least one anchor missed. The vector gap is mostly the middle band,
    where both provisions are found but not ranked together, which isolates the
    advantage as ordering rather than retrieval.
    """
    mc = load_json(f"{STAGE3}/multihop_contrast.json")
    n = mc["n_subset"]
    fb = mc["failure_breakdown"]

    def split(side: dict) -> list[int]:
        success = n - side["failures"]
        mid = side["all_anchors_retrieved"]
        missed = side["one_anchor_missing"] + side["all_anchors_missing"]
        return [success, mid, missed]

    rows = [("Vectorless\n(improved hybrid-tree)", split(fb["a"])),
            ("Vector\n(improved BGE-M3 + reranker)", split(fb["b"]))]
    seg_labels = ["Both in top 2", "Both found, below top 2", "At least one missing from top 10"]
    seg_colors = [C_ENC, C_STORE, C_LLM]

    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ys = [1, 0]
    for yi, (_, parts) in zip(ys, rows):
        left = 0
        for val, color in zip(parts, seg_colors):
            if val > 0:
                ax.barh(yi, val, left=left, height=0.62, **bar_style(color))
                ax.annotate(f"{val}\n{val / n * 100:.0f}%", (left + val / 2, yi),
                            ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold", linespacing=0.95)
            left += val
        if parts[2] == 0:
            ax.annotate("0 missing", (n, yi), textcoords="offset points", xytext=(4, 0),
                        ha="left", va="center", fontsize=7.5, color=INK_SOFT)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_xlabel(f"Multihop test queries (n = {n})")
    ax.set_xlim(0, n * 1.12)
    ax.set_ylim(-0.6, 1.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seg_colors]
    ax.legend(handles, seg_labels, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.22), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / f"multihop-breakdown.{fmt}")
    plt.close(fig)


def plot_tree_funnel(out: Path, fmt: str = "pdf") -> None:
    """Slope chart of the two-stage decomposition for the hierarchical methods.

    Each method is traced across three quantities, how often the gold document
    survived selection, how often the gold node was then ranked first inside that
    document, and the final H@1 after the per-document lists were merged. The
    second value is conditional on the document being found, matching the reported
    table, so the drop to the final value reflects the cross-document merge. The
    methods separate at the within-document step rather than at document selection.
    """
    # dy offsets keep the two near-identical vectorless lines from colliding,
    # with hybrid-tree labeled above its markers and llm-tree below.
    methods = [
        ("bm25-tree", "bm25-tree", INK_SOFT, "o", -16),
        ("hybrid-tree", "hybrid-tree", C_LLM, "o", 9),
        ("llm-tree", "llm-tree", C_PROC, "s", -16),
    ]
    stages = ["Correct document\nretrieved", "Correct node first\nin that document",
              "Correct node first\noverall (H@1)"]

    def decomp(method: str) -> list[float]:
        rows = vl_records(method, "pasal")
        n = len(rows)
        doc = sum(r.get("doc_pick_hit") or 0 for r in rows) / n
        found = [r for r in rows if (r.get("doc_pick_hit") or 0) == 1]
        within = sum(r.get("within_doc_hit@1") or 0 for r in found) / len(found)
        h1 = sum(r.get("hit@1") or 0 for r in rows) / n
        return [doc, within, h1]

    x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    for method, label, color, marker, dy in methods:
        vals = decomp(method)
        ax.plot(x, vals, linewidth=1.8, markersize=7, label=label, zorder=3,
                **pt_style(color, marker))
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=7.5, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlim(-0.35, len(stages) - 0.65)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"tree-funnel.{fmt}")
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
    plot_hypothesis_forest(out)
    plot_rank_distribution(out)
    plot_multihop_breakdown(out)
    plot_tree_funnel(out)
    print(f"Wrote 10 figures to {out}")
    if args.svg_dir:
        svg_dir = Path(args.svg_dir)
        svg_dir.mkdir(parents=True, exist_ok=True)
        plot_granularity(svg_dir, fmt="svg")
        plot_bytype(svg_dir, fmt="svg")
        plot_cost(svg_dir, fmt="svg")
        plot_tuning(svg_dir, fmt="svg")
        plot_emb_reranker_heatmap(svg_dir, fmt="svg")
        plot_vec_granularity(svg_dir, fmt="svg")
        print(f"Wrote 6 SVG slide figures to {svg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
