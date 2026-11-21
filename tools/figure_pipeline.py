"""Draw the data-pipeline figure: small labelled blocks, left to right, the big picture only.

    python tools/figure_pipeline.py   -> results/figures/figure1_pipeline.{pdf,png}

Also called by results/paper_tables.ipynb (draw()).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "results", "figures")

INK, EDGE, ARROW = "#1f2328", "#4d5560", "#4d5560"
SOURCE, STEP, NEG, OUT = "#f9d7e0", "#fde9c4", "#cfe3f7", "#d9eec8"
FS = 7.2


def draw(out_dir=FIGURES):
    """Draw Figure 1 and save it as figure1_pipeline.pdf / .png in out_dir; returns the figure."""
    fig, ax = plt.subplots(figsize=(7.0, 1.95))
    ax.set_xlim(-0.05, 16.05)
    ax.set_ylim(-0.15, 3.25)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=FS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=EDGE, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=0.9, mutation_scale=8, shrinkA=0, shrinkB=0))

    def group(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="#f6f7f9", ec="#9aa3ad", lw=0.8, ls=(0, (3, 2))))
        ax.text(x + w / 2, y + h + 0.12, title, ha="center", va="bottom", fontsize=FS, color=INK, fontweight="bold")


    MID = 1.45
    # sources -> bridge match
    box(0.0, 1.95, 1.55, 0.8, "Overture\nPlaces", SOURCE)
    box(0.0, 0.15, 1.55, 0.8, "Foursquare\nPlaces", SOURCE)
    box(2.05, MID - 0.45, 1.5, 0.9, "Bridge-file\nmatch", STEP)
    arrow(1.55, 2.35, 2.05, MID + 0.15)
    arrow(1.55, 0.55, 2.05, MID - 0.15)

    # positives
    group(3.95, 0.45, 6.71, 2.0, "Positive pairs")
    xs = [4.15, 5.77, 7.39, 9.01]
    labels = ["Clean ZIP\n+ address", "Split\nby ZIP", "Set aside\nnumber\nmismatches", "Remove\nduplicates"]
    arrow(3.55, MID, 4.15, MID)
    for i, (x, lab) in enumerate(zip(xs, labels)):
        box(x, MID - 0.5, 1.45, 1.0, lab, STEP, fs=6.6)
        if i < len(xs) - 1:
            arrow(x + 1.45, MID, xs[i + 1], MID)

    # non-matches
    group(11.15, 0.0, 2.6, 2.9, "Non-matches")
    negs = ["Different number", "Real look-alike", "Look-alike street", "Random"]
    for i, lab in enumerate(negs):
        box(11.3, 2.25 - i * 0.68, 2.3, 0.52, lab, NEG, fs=6.6)
    arrow(10.46, MID, 11.3, MID)

    # output
    box(14.35, MID - 0.55, 1.65, 1.1, "Train / val /\ntest pairs\n1 : 9", OUT)
    arrow(13.6, MID, 14.35, MID)

    fig.tight_layout(pad=0.05)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "figure1_pipeline.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figure1_pipeline.png"), bbox_inches="tight", facecolor="white", dpi=300)
    return fig


if __name__ == "__main__":
    draw()
    print("written to", FIGURES)
