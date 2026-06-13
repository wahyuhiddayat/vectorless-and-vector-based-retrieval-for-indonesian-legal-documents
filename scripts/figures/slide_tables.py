"""Render curated result tables as styled SVGs for the defense deck.

The defense slides reuse condensed versions of the Chapter 4 tables. Building
them by hand in the slide editor is slow and the default tables are bulky, so
this script renders each one as a compact SVG that matches the deck style, a
dark navy header with white text over white body rows. Values are taken from
the verified thesis tables, and each spec notes its source table.

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

REPO_ROOT = Path(__file__).resolve().parents[2]
DECK_SVG = REPO_ROOT.parents[1] / "05 Thesis Defense" / "Assets" / "svg"

HEADER_COLOR = "#0D2D44"
BORDER_COLOR = "#D9D9D9"
TEXT_COLOR = "#1A1A1A"


def render_table(headers: list[str], rows: list[list[str]], col_widths: list[float],
                 out_path: Path, fontsize: float = 12.0) -> None:
    """Render one table to out_path, styled like the deck with a navy header over white rows.

    The output format follows the file extension of out_path, so an .svg path
    writes SVG with selectable text and a .png path writes a raster preview.
    """
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})
    total = sum(col_widths)
    fig, ax = plt.subplots(figsize=(total, 0.66 * (len(rows) + 1.6)))
    ax.axis("off")

    table = ax.table(
        cellText=rows, colLabels=headers, cellLoc="center", loc="center",
        colWidths=[w / total for w in col_widths],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER_COLOR)
        cell.set_linewidth(0.8)
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_height(0.30)
            text = cell.get_text()
            text.set_color("white")
            text.set_fontweight("bold")
        else:
            cell.set_facecolor("white")
            cell.set_height(0.22)
            cell.get_text().set_color(TEXT_COLOR)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# Source: tab:tree-decomp (bab4.tex), two-stage breakdown of the hierarchical methods.
TREE_DECOMP = {
    "stem": "tab-tree-decomp",
    "headers": ["Method", "Gold Document\nFound", "Gold Node First\nWhen Found", "Overall H@1"],
    "rows": [
        ["bm25-tree", "0.8964", "0.6344", "0.5266"],
        ["hybrid-tree", "0.9608", "0.9563", "0.8880"],
        ["llm-tree", "0.9496", "0.9469", "0.8880"],
    ],
    "col_widths": [1.25, 1.75, 2.0, 1.4],
}

TABLES = [TREE_DECOMP]


def main() -> int:
    """Render every deck table spec into the --out directory as SVG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DECK_SVG))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in TABLES:
        render_table(spec["headers"], spec["rows"], spec["col_widths"], out / f"{spec['stem']}.svg")
        print(f"Wrote {spec['stem']}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
