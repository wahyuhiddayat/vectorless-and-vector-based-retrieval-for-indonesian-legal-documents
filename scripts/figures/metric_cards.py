"""Render custom-metric definition cards as SVG for the defense deck.

Sibling-hit is a custom diagnostic rather than a standard metric, so the
failure-analysis slide needs its definition and formula spelled out. This
renders that as a self-contained SVG so the formula does not have to be typed
by hand in the slide editor.

Usage:
    python scripts/figures/metric_cards.py --svg-dir "../../05 Thesis Defense/Assets/svg"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#0D2D44"
INK = "#1A1A1A"
GRAY = "#5A5A5A"

# Sibling-hit at rank 1, computed per granularity. The top result counts when it
# shares a gold node's parent in the granularity tree but is not itself gold.
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


def render(svg_dir: Path) -> None:
    """Write the sibling-hit definition card to metric-sibling-hit.svg."""
    plt.rcParams["font.family"] = "Arial"

    fig = plt.figure(figsize=(9.4, 1.9))
    fig.text(0.5, 0.82, "Sibling-hit at rank 1, a near-miss diagnostic (per granularity)",
             ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.50, FORMULA, ha="center", va="center", fontsize=15, color=INK)
    fig.text(0.5, 0.15, LEGEND, ha="center", va="center", fontsize=9.5, color=GRAY)

    out = svg_dir / "metric-sibling-hit.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg-dir", required=True, help="deck Assets/svg directory")
    args = ap.parse_args()
    render(Path(args.svg_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
