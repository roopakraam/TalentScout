"""Draw the study-design figure for the report (results/figures/fig00_study_design.png).

    python report/design_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.patches as patches  # noqa: E402

from src.plotting import BLUE, INK, INK_SECONDARY, plt, save, set_style  # noqa: E402

LIGHT = "#cde2fb"
VERSIONS = list(range(15, 25))
LABELS = [f"FIFA\n{v}" if v < 24 else "FC\n24" for v in VERSIONS]

LANES = [  # name, feature versions, label window, note
    ("Train", (15, 18), (16, 21), "12,148 rows · 344 future elite"),
    ("Validation", (19, 20), (20, 23), "3,974 rows · 113 future elite"),
    ("Test", (21, 21), (22, 24), "3,288 rows · 72 future elite"),
]


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for row, (name, (f0, f1), (l0, l1), note) in enumerate(LANES):
        y = len(LANES) - 1 - row
        ax.add_patch(patches.FancyBboxPatch((l0 - 0.45, y - 0.28), l1 - l0 + 0.9, 0.56,
                                            boxstyle="round,pad=0,rounding_size=0.08", linewidth=0, facecolor=LIGHT))
        ax.add_patch(patches.FancyBboxPatch((f0 - 0.45, y - 0.28), f1 - f0 + 0.9, 0.56,
                                            boxstyle="round,pad=0,rounding_size=0.08", linewidth=0, facecolor=BLUE))
        ax.text(13.85, y, name, ha="right", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(24.6, y, note, ha="left", va="center", fontsize=7.5, color=INK_SECONDARY)
    ax.add_patch(patches.Rectangle((0, 0), 0, 0, facecolor=BLUE, label="feature rows (age ≤ 21, overall < 80)"))
    ax.add_patch(patches.Rectangle((0, 0), 0, 0, facecolor=LIGHT, label="label window: reaches 80+ within 3 versions?"))
    ax.set_xticks(VERSIONS, LABELS, fontsize=7.2, rotation=0)
    ax.set_yticks([])
    ax.set_xlim(12.2, 27.9)
    ax.set_ylim(-0.75, len(LANES) - 0.35)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ecebe5", linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncols=2, fontsize=7.5, handlelength=1.2)
    ax.text(12.2, len(LANES) - 0.32, "Every player in the test set is removed from train and validation.",
            fontsize=7.5, color=INK_SECONDARY, va="bottom")
    save(fig, "fig00_study_design")
    print("saved results/figures/fig00_study_design.png")


if __name__ == "__main__":
    main()
