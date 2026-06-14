"""Render custom-metric definition assets for the thesis and defense deck.

Sibling-hit is a custom diagnostic rather than a standard metric, so it needs
its definition spelled out. This renders two assets, a compact formula card for
the slide and a worked-example diagram for the appendix that shows, on a small
granularity tree, when the rank-1 result counts as a sibling hit.

Usage:
    python scripts/figures/metric_cards.py \
        --out "../../laporan-skripsi/assets/figures/bab4" \
        --svg-dir "../../05 Thesis Defense/Assets/svg"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY = "#0D2D44"
UIBLUE = "#284887"
UIRED = "#962D30"
GREEN = "#2E7D32"
GOLD = "#C9A227"
GRAY = "#8C8C8C"
INK = "#1A1A1A"

FORMULA = (
    r"$\mathrm{Sibling\ hit}@1 \;=\; \frac{1}{|Q|}\sum_{q \in Q}\mathbf{1}"
    r"\left[\, r_1(q)\notin G(q)\ \wedge\ \exists\, g \in G(q):\ "
    r"\mathrm{par}(r_1(q))=\mathrm{par}(g) \,\right]$"
)
LEGEND = (
    r"$r_1(q)$ rank-1 retrieved node" "      "
    r"$G(q)$ gold set at the granularity" "      "
    r"$\mathrm{par}(\cdot)$ parent in the granularity tree"
)


def render_formula_card(svg_dir: Path) -> None:
    """Write the sibling-hit definition card to metric-sibling-hit.svg."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(9.0, 1.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.012, 0.06), 0.976, 0.88, facecolor="white",
                               edgecolor="#D9D9D9", linewidth=1.1, zorder=0))
    ax.text(0.5, 0.80, "Sibling-hit at rank 1, near-miss diagnostic (per granularity)",
            ha="center", va="center", color=NAVY, fontsize=12, fontweight="bold", zorder=2)
    ax.text(0.5, 0.45, FORMULA, ha="center", va="center", fontsize=17, color=INK, zorder=2)
    ax.text(0.5, 0.16, LEGEND, ha="center", va="center", fontsize=9.5, color=GRAY, zorder=2)

    out = svg_dir / "metric-sibling-hit.svg"
    fig.savefig(out, format="svg", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {out}")


def _box(ax, x, y, w, h, text, face, edge, tcolor, fontsize=8.5, lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.014",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=tcolor, zorder=4, fontweight="bold" if bold else "normal")


def _line(ax, x1, y1, x2, y2, color="#AAB0B8", lw=1.0):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1)


def render_sibling_example(out_dir: Path, fmt: str = "pdf") -> None:
    """Write the sibling-hit worked example to sibling-hit-example.<fmt>."""
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    root = (0.50, 0.93)
    p5, p9 = (0.28, 0.68), (0.76, 0.68)
    leaf = {"p5a1": (0.10, 0.44), "p5a2": (0.28, 0.44), "p5a3": (0.46, 0.44),
            "p9a1": (0.64, 0.44), "p9a2": (0.88, 0.44)}
    lw_, lh_ = 0.155, 0.072

    _line(ax, *root, *p5)
    _line(ax, *root, *p9)
    for k in ("p5a1", "p5a2", "p5a3"):
        _line(ax, *p5, *leaf[k])
    for k in ("p9a1", "p9a2"):
        _line(ax, *p9, *leaf[k])

    _box(ax, *root, 0.24, 0.085, "UU 12/2011", "#F2F4F8", UIBLUE, INK, 9.5, bold=True)
    _box(ax, *p5, 0.16, 0.075, "Pasal 5", "#F2F4F8", UIBLUE, INK, 9, bold=True)
    _box(ax, *p9, 0.16, 0.075, "Pasal 9", "#F2F4F8", UIBLUE, INK, 9, bold=True)
    _box(ax, *leaf["p5a1"], lw_, lh_, "Ayat (1)", "white", "#B7BDC6", INK)
    _box(ax, *leaf["p5a2"], lw_, lh_, "Ayat (2)", GOLD, "#9C7D12", "white", bold=True)
    _box(ax, *leaf["p5a3"], lw_, lh_, "Ayat (3)", "white", UIRED, INK, lw=1.8)
    _box(ax, *leaf["p9a1"], lw_, lh_, "Ayat (1)", "white", GRAY, INK, lw=1.8)
    _box(ax, *leaf["p9a2"], lw_, lh_, "Ayat (2)", "white", "#B7BDC6", INK)
    ax.text(0.28, 0.44 - lh_ / 2 - 0.018, "gold node", ha="center", va="top",
            fontsize=7.5, color="#9C7D12")

    cy = 0.115
    _box(ax, 0.21, cy, 0.275, 0.15,
         "Case B  rank-1 = Ayat (2)\nit is the gold node\n-> correct hit, contributes 0",
         "#EAF3EA", GREEN, INK, 7.8)
    _box(ax, 0.50, cy, 0.275, 0.15,
         "Case A  rank-1 = Ayat (3)\nsame Pasal 5, not gold\n-> sibling hit, contributes 1",
         "#F7E9E9", UIRED, INK, 7.8)
    _box(ax, 0.79, cy, 0.275, 0.15,
         "Case C  rank-1 = Ayat (1), Pasal 9\ndifferent parent\n-> unrelated, contributes 0",
         "#EEF0F2", GRAY, INK, 7.8)
    _line(ax, 0.28, 0.44 - lh_ / 2, 0.21, cy + 0.075, GREEN, 1.0)
    _line(ax, 0.46, 0.44 - lh_ / 2, 0.50, cy + 0.075, UIRED, 1.0)
    _line(ax, 0.64, 0.44 - lh_ / 2, 0.79, cy + 0.075, GRAY, 1.0)

    out = out_dir / f"sibling-hit-example.{fmt}"
    fig.savefig(out, format=fmt, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="thesis figures directory for the PDF example")
    ap.add_argument("--svg-dir", help="deck Assets/svg directory for the SVG assets")
    args = ap.parse_args()
    if args.svg_dir:
        svg = Path(args.svg_dir)
        render_formula_card(svg)
        render_sibling_example(svg, "svg")
    if args.out:
        render_sibling_example(Path(args.out), "pdf")
    if not args.out and not args.svg_dir:
        ap.error("provide --out and/or --svg-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
