"""Render the thesis-defense result tables as styled SVGs.

The defense slides reuse condensed versions of the Chapter 3 and Chapter 4
tables. Building them by hand in the slide editor is slow and the default
tables are bulky, so this script renders each one as an SVG that matches the
deck style, a dark navy header with white text over white body rows. Headers
are kept on a single line and column widths are measured from the text, so
nothing is clipped. Values are taken from the verified thesis tables, and each
spec notes its source table.

Usage:
    python scripts/figures/slide_tables.py
    python scripts/figures/slide_tables.py --out "../../05 Thesis Defense/Assets/svg"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

REPO_ROOT = Path(__file__).resolve().parents[2]
DECK_SVG = REPO_ROOT.parents[1] / "05 Thesis Defense" / "Assets" / "svg"

HEADER_COLOR = "#0D2D44"
BORDER_COLOR = "#D9D9D9"
TEXT_COLOR = "#1A1A1A"
HIGHLIGHT_COLOR = "#EAF0F8"  # light tint marking the chosen or carried-forward row
PAD_IN = 0.34  # horizontal breathing room added to each column, in inches


def _line_width(text: str, font: FontProperties, renderer, dpi: float) -> float:
    """Width in inches of the widest line in text, measured with the given font."""
    return max(renderer.get_text_width_height_descent(line, font, False)[0]
               for line in text.split("\n")) / dpi


def _column_widths(headers: list[str], rows: list[list[str]], fontsize: float) -> list[float]:
    """Inch width per column, sized to the widest header or body line plus padding."""
    probe = plt.figure()
    renderer = probe.canvas.get_renderer()
    header_font = FontProperties(family="Arial", size=fontsize, weight="bold")
    body_font = FontProperties(family="Arial", size=fontsize, weight="normal")
    widths = []
    for c in range(len(headers)):
        header_w = _line_width(headers[c], header_font, renderer, probe.dpi)
        body_w = max((_line_width(row[c], body_font, renderer, probe.dpi) for row in rows),
                     default=0.0)
        widths.append(max(header_w, body_w) + PAD_IN)
    plt.close(probe)
    return widths


def render_table(headers: list[str], rows: list[list[str]], out_path: Path,
                 fontsize: float = 12.0, bold_rows: list[int] | None = None) -> None:
    """Render one table to out_path, styled like the deck with a navy header over white rows.

    bold_rows holds zero-based body row indices to emphasize, drawn in bold over
    a light tint to mark a chosen or carried-forward configuration. The output
    format follows the file extension of out_path, so an .svg path writes SVG
    with selectable text and a .png path writes a raster preview.
    """
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})
    bold = set(bold_rows or [])

    col_widths = _column_widths(headers, rows, fontsize)
    header_lines = max(h.count("\n") + 1 for h in headers)
    body_lines = max((c.count("\n") + 1 for row in rows for c in row), default=1)
    header_in = 0.26 * header_lines + 0.22
    body_in = 0.26 * body_lines + 0.16
    fig_w = sum(col_widths)
    fig_h = header_in + body_in * len(rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Let the axes fill the figure so colWidths map to figure inches, not the
    # smaller default subplot area, otherwise wide headers clip.
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers, cellLoc="center", loc="center",
        colWidths=[w / fig_w for w in col_widths],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER_COLOR)
        cell.set_linewidth(0.8)
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_height(header_in / fig_h)
            text = cell.get_text()
            text.set_color("white")
            text.set_fontweight("bold")
        else:
            cell.set_height(body_in / fig_h)
            cell.get_text().set_color(TEXT_COLOR)
            if (r - 1) in bold:
                cell.set_facecolor(HIGHLIGHT_COLOR)
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# Each spec lists its source thesis table. Headers stay on one line. Long body
# cells wrap with an explicit newline. Values match the verified thesis tables.
TABLES = [
    # Source: tab:vectorless-methods (bab3.tex).
    {
        "stem": "tab-vectorless-methods",
        "headers": ["Method", "Paradigm", "First Stage", "Final Scorer", "LLM Calls per Query"],
        "rows": [
            ["bm25-flat", "Flat", "BM25", "BM25", "0"],
            ["bm25-tree", "Hierarchical", "BM25", "BM25", "0"],
            ["hybrid-flat", "Flat", "BM25", "LLM listwise", "1"],
            ["hybrid-tree", "Hierarchical", "BM25 and LLM", "LLM listwise", "2"],
            ["llm-flat", "Flat", "LLM", "LLM listwise", "2 to 7"],
            ["llm-tree", "Hierarchical", "LLM", "LLM agent", "2 to 31"],
        ],
    },
    # Source: tab:vector-config (bab3.tex).
    {
        "stem": "tab-vector-config",
        "headers": ["Axis", "Levels"],
        "rows": [
            ["Embedding model", "BGE-M3, Multilingual E5 Large Instruct,\nAll-Nusabert-Large-V4"],
            ["Neural reranker", "none, BGE-Reranker-v2-M3,\nQwen3 Reranker 0.6B"],
            ["Index granularity", "Pasal, Ayat, Rincian"],
        ],
    },
    # Source: tab:stages (bab3.tex).
    {
        "stem": "tab-stages",
        "headers": ["Stage", "Evaluated Configurations", "Partition", "Objective"],
        "rows": [
            ["1", "All 18 vectorless and\n27 vector configurations", "Dev (357)",
             "Identify the best configuration\nfrom each paradigm"],
            ["2", "The two best Stage 1\nconfigurations", "Dev (357)",
             "Optimize hyperparameters, model\ncapacity, and query expansion"],
            ["3", "The two tuned configurations", "Test (356)",
             "Compare the paradigms on unseen\nqueries and measure cost"],
        ],
    },
    # Source: tab:vl-stage1 (bab4.tex), condensed to the top six rows plus the two BM25 pasal rows.
    {
        "stem": "tab-vl-stage1",
        "bold_rows": [0],
        "headers": ["Method", "Granularity", "MAP@10", "R@2", "R@10", "MRR@10", "H@1", "Avg s"],
        "rows": [
            ["hybrid-tree", "Pasal", "0.8974", "0.9034", "0.9244", "0.9074", "0.8880", "13.84"],
            ["hybrid-flat", "Pasal", "0.8954", "0.8908", "0.9230", "0.9060", "0.8908", "6.71"],
            ["llm-tree", "Pasal", "0.8868", "0.8922", "0.9090", "0.9085", "0.8880", "11.09"],
            ["llm-flat", "Pasal", "0.8840", "0.8768", "0.9216", "0.8994", "0.8824", "13.13"],
            ["llm-flat", "Rincian", "0.8609", "0.8515", "0.9328", "0.8955", "0.8571", "62.52"],
            ["hybrid-tree", "Ayat", "0.8327", "0.8445", "0.8782", "0.8610", "0.8235", "15.83"],
            ["bm25-flat", "Pasal", "0.6948", "0.6933", "0.8613", "0.7427", "0.6667", "0.49"],
            ["bm25-tree", "Pasal", "0.5582", "0.5392", "0.7129", "0.5989", "0.5266", "0.09"],
        ],
    },
    # Source: tab:tree-decomp (bab4.tex), two-stage breakdown of the hierarchical methods.
    {
        "stem": "tab-tree-decomp",
        "headers": ["Method", "Gold Document Found", "Gold Node First When Found", "Overall H@1"],
        "rows": [
            ["bm25-tree", "0.8964", "0.6344", "0.5266"],
            ["hybrid-tree", "0.9608", "0.9563", "0.8880"],
            ["llm-tree", "0.9496", "0.9469", "0.8880"],
        ],
    },
    # Source: tab:vec-stage1 (bab4.tex), condensed to the top six pasal rows plus one NusaBERT row.
    {
        "stem": "tab-vec-stage1",
        "bold_rows": [0],
        "headers": ["Embedding", "Reranker", "MAP@10", "R@2", "R@10", "MRR@10", "H@1", "Avg s"],
        "rows": [
            ["BGE-M3", "BGE v2 M3", "0.8898", "0.8768", "0.9776", "0.9153", "0.8711", "1.70"],
            ["E5", "BGE v2 M3", "0.8870", "0.8711", "0.9678", "0.9142", "0.8739", "1.35"],
            ["E5", "Qwen3 0.6B", "0.8280", "0.8375", "0.9608", "0.8604", "0.7787", "3.08"],
            ["BGE-M3", "Qwen3 0.6B", "0.8257", "0.8347", "0.9650", "0.8552", "0.7675", "3.69"],
            ["NusaBERT", "BGE v2 M3", "0.8057", "0.8137", "0.8880", "0.8533", "0.8011", "1.02"],
            ["BGE-M3", "none", "0.7948", "0.7927", "0.9342", "0.8368", "0.7591", "0.12"],
            ["NusaBERT", "none", "0.6017", "0.5896", "0.8221", "0.6550", "0.5462", "0.11"],
        ],
    },
    # Source: tab:winners (bab4.tex), the two best Stage 1 configurations side by side.
    {
        "stem": "tab-winners",
        "headers": ["Metric", "Vectorless (hybrid-tree)", "Vector (BGE-M3 + BGE v2 M3)"],
        "rows": [
            ["MAP@10", "0.8974", "0.8898"],
            ["R@2", "0.9034", "0.8768"],
            ["R@10", "0.9244", "0.9776"],
            ["MRR@10", "0.9074", "0.9153"],
            ["H@1", "0.8880", "0.8711"],
            ["MAP@10 factual", "0.9553", "0.9546"],
            ["MAP@10 paraphrased", "0.8287", "0.8533"],
            ["MAP@10 multihop", "0.9078", "0.8583"],
            ["Mean latency (s)", "13.84", "1.70"],
            ["LLM calls", "2.0", "0"],
            ["LLM tokens", "138,055", "0"],
        ],
    },
    # Source: tab:vl-tuning (bab4.tex), sequential optimization of the best vectorless configuration.
    {
        "stem": "tab-vl-tuning",
        "bold_rows": [1, 7, 9, 10],
        "headers": ["Step", "Value", "MAP@10", "R@10", "LLM tokens", "Avg s"],
        "rows": [
            ["Candidate count", "10 (default)", "0.9077", "0.9342", "138,578", "11.18"],
            ["", "20", "0.9434", "0.9748", "146,291", "12.04"],
            ["", "30", "0.9396", "0.9762", "148,694", "13.26"],
            ["", "50", "0.9395", "0.9734", "156,621", "16.70"],
            ["Document-pick count", "1", "0.8949", "0.9244", "131,220", "9.63"],
            ["", "2", "0.9345", "0.9622", "138,528", "9.92"],
            ["", "3 (default)", "0.9326", "0.9650", "146,923", "12.00"],
            ["", "5", "0.9445", "0.9846", "164,321", "18.77"],
            ["Retrieval model", "deepseek-v4-flash (default)", "0.9445", "0.9846", "164,321", "18.77"],
            ["", "deepseek-v4-pro", "0.9516", "0.9790", "162,818", "39.84"],
            ["Query expansion", "original query (default)", "0.9516", "0.9790", "162,818", "39.84"],
            ["", "expanded query", "0.9445", "0.9734", "162,331", "41.37"],
        ],
    },
    # Source: tab:vec-tuning (bab4.tex), sequential optimization of the best vector configuration.
    {
        "stem": "tab-vec-tuning",
        "bold_rows": [2, 4, 8, 11],
        "headers": ["Step", "Value", "MAP@10", "R@10", "Avg s"],
        "rows": [
            ["First-stage depth", "20", "0.8801", "0.9566", "0.23"],
            ["", "50 (default)", "0.8898", "0.9776", "0.95"],
            ["", "100", "0.8933", "0.9818", "2.65"],
            ["", "200", "0.8920", "0.9790", "5.21"],
            ["HNSW ef", "64 (default)", "0.8933", "0.9818", "2.57"],
            ["", "128", "0.8933", "0.9818", "2.66"],
            ["", "256", "0.8933", "0.9818", "2.73"],
            ["", "512", "0.8933", "0.9818", "2.52"],
            ["Reranker", "BGE v2 M3 (default)", "0.8933", "0.9818", "2.57"],
            ["", "BGE v2 Gemma", "0.8661", "0.9734", "44.38"],
            ["Query expansion", "original query (default)", "0.8933", "0.9818", "2.57"],
            ["", "expanded query", "0.8974", "0.9804", "2.68"],
        ],
    },
    # Source: tab:test-winners (bab4.tex), the two tuned configurations on the test partition.
    {
        "stem": "tab-test-winners",
        "headers": ["Metric", "Vectorless (tuned hybrid-tree)", "Vector (tuned BGE-M3 + reranker + QE)"],
        "rows": [
            ["MAP@10", "0.9489", "0.8750"],
            ["R@2", "0.9466", "0.8834"],
            ["R@10", "0.9888", "0.9789"],
            ["MRR@10", "0.9552", "0.9047"],
            ["H@1", "0.9326", "0.8455"],
        ],
    },
    # Source: tab:test-sig (bab4.tex), paired significance tests on the test partition.
    {
        "stem": "tab-test-sig",
        "headers": ["Metric", "Difference", "p-value", "Significant"],
        "rows": [
            ["MAP@10", "+0.0738", "0.0001", "yes"],
            ["R@2", "+0.0632", "0.0001", "yes"],
            ["MRR@10", "+0.0505", "0.0008", "yes"],
            ["H@1", "+0.0871", "0.0002", "yes"],
            ["R@10", "+0.0098", "0.29", "no"],
            ["R@2, multihop", "+0.1603", "0.0001", "yes"],
        ],
    },
    # Source: tab:sibling (bab4.tex), sibling-hit rate at rank 1 for the two best configurations.
    {
        "stem": "tab-sibling",
        "headers": ["Granularity", "Vectorless (hybrid-tree)", "Vector (BGE-M3 + BGE v2 M3)"],
        "rows": [
            ["Pasal", "0.0056", "0.0028"],
            ["Ayat", "0.0588", "0.0924"],
            ["Rincian", "0.0756", "0.1092"],
        ],
    },
    # Source: tab:cost-test (bab4.tex), per-query cost of the tuned configurations.
    {
        "stem": "tab-cost-test",
        "headers": ["Metric", "Vectorless (tuned hybrid-tree)", "Vector (tuned BGE-M3 + reranker + QE)"],
        "rows": [
            ["MAP@10", "0.9489", "0.8750"],
            ["LLM calls", "2.0", "1.0"],
            ["LLM input tokens", "160,843", "1,018"],
            ["LLM output tokens", "2,488", "47"],
            ["LLM tokens, total", "163,331", "1,065"],
            ["Mean latency (s)", "42.43", "2.69"],
            ["p95 latency (s)", "66.99", "5.12"],
        ],
    },
    # Source: tab:vl-stage1 (bab4.tex), full 18-configuration table.
    {
        "stem": "tab-vl-stage1-full",
        "bold_rows": [0],
        "headers": ["Method", "Granularity", "MAP@10", "R@2", "R@10", "MRR@10", "H@1",
                    "LLM calls", "LLM tokens", "Avg s"],
        "rows": [
            ["hybrid-tree", "Pasal", "0.8974", "0.9034", "0.9244", "0.9074", "0.8880", "2.0", "138,055", "13.84"],
            ["hybrid-flat", "Pasal", "0.8954", "0.8908", "0.9230", "0.9060", "0.8908", "1.0", "10,684", "6.71"],
            ["llm-tree", "Pasal", "0.8868", "0.8922", "0.9090", "0.9085", "0.8880", "4.2", "143,741", "11.09"],
            ["llm-flat", "Pasal", "0.8840", "0.8768", "0.9216", "0.8994", "0.8824", "2.0", "877,530", "13.13"],
            ["llm-flat", "Rincian", "0.8609", "0.8515", "0.9328", "0.8955", "0.8571", "7.0", "5,383,840", "62.52"],
            ["hybrid-tree", "Ayat", "0.8327", "0.8445", "0.8782", "0.8610", "0.8235", "2.0", "136,528", "15.83"],
            ["llm-flat", "Ayat", "0.8198", "0.8193", "0.8922", "0.8526", "0.8067", "4.0", "2,445,032", "25.13"],
            ["llm-tree", "Rincian", "0.7761", "0.7815", "0.8067", "0.8004", "0.7759", "5.4", "222,662", "17.85"],
            ["hybrid-flat", "Ayat", "0.7741", "0.7759", "0.8011", "0.8147", "0.7899", "1.0", "6,925", "7.53"],
            ["llm-tree", "Ayat", "0.7698", "0.7703", "0.7913", "0.8025", "0.7815", "5.3", "174,985", "15.74"],
            ["hybrid-tree", "Rincian", "0.7631", "0.7787", "0.8095", "0.8121", "0.7703", "2.0", "130,236", "15.35"],
            ["hybrid-flat", "Rincian", "0.7535", "0.7577", "0.7927", "0.8053", "0.7759", "1.0", "5,595", "7.70"],
            ["bm25-flat", "Pasal", "0.6948", "0.6933", "0.8613", "0.7427", "0.6667", "0", "0", "0.49"],
            ["bm25-tree", "Pasal", "0.5582", "0.5392", "0.7129", "0.5989", "0.5266", "0", "0", "0.09"],
            ["bm25-flat", "Ayat", "0.5267", "0.5224", "0.7409", "0.5957", "0.4902", "0", "0", "0.76"],
            ["bm25-flat", "Rincian", "0.5157", "0.5014", "0.7283", "0.5869", "0.4874", "0", "0", "1.23"],
            ["bm25-tree", "Ayat", "0.4295", "0.4244", "0.5938", "0.4805", "0.3950", "0", "0", "0.14"],
            ["bm25-tree", "Rincian", "0.3585", "0.3585", "0.4888", "0.4151", "0.3361", "0", "0", "0.23"],
        ],
    },
    # Source: tab:vec-stage1 (bab4.tex), full 27-configuration table.
    {
        "stem": "tab-vec-stage1-full",
        "bold_rows": [0],
        "headers": ["Embedding", "Reranker", "Granularity", "MAP@10", "R@2", "R@10", "MRR@10", "H@1", "Avg s"],
        "rows": [
            ["BGE-M3", "BGE v2 M3", "Pasal", "0.8898", "0.8768", "0.9776", "0.9153", "0.8711", "1.70"],
            ["E5", "BGE v2 M3", "Pasal", "0.8870", "0.8711", "0.9678", "0.9142", "0.8739", "1.35"],
            ["E5", "Qwen3 0.6B", "Pasal", "0.8280", "0.8375", "0.9608", "0.8604", "0.7787", "3.08"],
            ["BGE-M3", "Qwen3 0.6B", "Pasal", "0.8257", "0.8347", "0.9650", "0.8552", "0.7675", "3.69"],
            ["NusaBERT", "BGE v2 M3", "Pasal", "0.8057", "0.8137", "0.8880", "0.8533", "0.8011", "1.02"],
            ["BGE-M3", "none", "Pasal", "0.7948", "0.7927", "0.9342", "0.8368", "0.7591", "0.12"],
            ["E5", "none", "Pasal", "0.7788", "0.7997", "0.9384", "0.8270", "0.7283", "0.11"],
            ["BGE-M3", "BGE v2 M3", "Ayat", "0.7726", "0.7661", "0.9034", "0.8134", "0.7395", "1.35"],
            ["E5", "BGE v2 M3", "Ayat", "0.7617", "0.7521", "0.8754", "0.8045", "0.7395", "1.29"],
            ["NusaBERT", "Qwen3 0.6B", "Pasal", "0.7515", "0.7689", "0.8796", "0.8035", "0.7227", "2.23"],
            ["NusaBERT", "BGE v2 M3", "Ayat", "0.7045", "0.7101", "0.8291", "0.7679", "0.6947", "0.62"],
            ["BGE-M3", "Qwen3 0.6B", "Ayat", "0.7039", "0.7059", "0.8908", "0.7547", "0.6415", "2.61"],
            ["BGE-M3", "BGE v2 M3", "Rincian", "0.7032", "0.6905", "0.8515", "0.7491", "0.6667", "0.41"],
            ["E5", "Qwen3 0.6B", "Ayat", "0.7010", "0.7003", "0.8641", "0.7504", "0.6499", "2.60"],
            ["E5", "BGE v2 M3", "Rincian", "0.6907", "0.6723", "0.8207", "0.7448", "0.6723", "0.55"],
            ["NusaBERT", "BGE v2 M3", "Rincian", "0.6794", "0.6709", "0.8165", "0.7369", "0.6611", "0.40"],
            ["BGE-M3", "Qwen3 0.6B", "Rincian", "0.6650", "0.6513", "0.8347", "0.7232", "0.6331", "0.80"],
            ["BGE-M3", "none", "Ayat", "0.6506", "0.6541", "0.8487", "0.7081", "0.5966", "0.18"],
            ["E5", "Qwen3 0.6B", "Rincian", "0.6448", "0.6373", "0.8109", "0.7063", "0.6106", "1.05"],
            ["NusaBERT", "Qwen3 0.6B", "Ayat", "0.6412", "0.6625", "0.8109", "0.6994", "0.5910", "1.22"],
            ["NusaBERT", "Qwen3 0.6B", "Rincian", "0.6327", "0.6289", "0.7983", "0.6930", "0.5966", "0.74"],
            ["E5", "none", "Ayat", "0.6140", "0.6036", "0.8123", "0.6694", "0.5602", "0.20"],
            ["NusaBERT", "none", "Pasal", "0.6017", "0.5896", "0.8221", "0.6550", "0.5462", "0.11"],
            ["BGE-M3", "none", "Rincian", "0.5831", "0.5952", "0.7913", "0.6417", "0.5238", "0.26"],
            ["E5", "none", "Rincian", "0.5633", "0.5616", "0.7577", "0.6257", "0.5238", "0.27"],
            ["NusaBERT", "none", "Ayat", "0.4908", "0.4776", "0.7045", "0.5484", "0.4398", "0.17"],
            ["NusaBERT", "none", "Rincian", "0.4751", "0.4524", "0.6849", "0.5294", "0.4230", "0.26"],
        ],
    },
]


def main() -> int:
    """Render every deck table spec into the --out directory as SVG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DECK_SVG))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in TABLES:
        render_table(spec["headers"], spec["rows"], out / f"{spec['stem']}.svg",
                     bold_rows=spec.get("bold_rows"))
        print(f"Wrote {spec['stem']}.svg")
    print(f"Wrote {len(TABLES)} tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
