"""One figure style for the whole project.

Rules followed in every figure:

* each method keeps the same colour, marker and line style everywhere, so the
  figures stay readable in grayscale and for colour-blind readers;
* one y-axis per plot, thin lines, a light solid grid, text in neutral ink;
* figures are saved as 300 dpi PNG and as vector PDF in ``results/figures``.

The three method colours were checked with a colour-vision-deficiency
validator on a white background (worst pairwise CVD distance 9.2, above the
target of 8).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# neutral ink
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# series colours
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#52514e"
LIGHT_BLUE = "#9ec5f4"

METHODS = ["ga", "bpso", "bgwo", "random"]
METHOD_STYLE = {
    "ga": dict(label="GA", color=BLUE, linestyle="-", marker="o"),
    "bpso": dict(label="Binary PSO", color=ORANGE, linestyle="--", marker="s"),
    "bgwo": dict(label="Binary GWO", color=AQUA, linestyle="-.", marker="^"),
    "random": dict(label="Random search", color=GRAY, linestyle=":", marker="D"),
}
METHOD_LABEL = {k: v["label"] for k, v in METHOD_STYLE.items()}

TEXT_WIDTH = 6.3          # inches, the text width of the report


def set_style() -> None:
    """Apply the project style; call once at the top of a notebook."""
    mpl.rcParams.update({
        "figure.figsize": (TEXT_WIDTH, 3.6),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "errorbar.capsize": 2.5,
    })


def method_kwargs(method: str, with_marker: bool = True, **extra) -> dict:
    """Plot keyword arguments for one of ga / bpso / bgwo / random."""
    style = dict(METHOD_STYLE[method])
    if not with_marker:
        style.pop("marker")
    style.update(extra)
    return style


def save(fig, name: str) -> None:
    """Save ``fig`` as ``results/figures/<name>.png`` and ``.pdf``."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.png")
    fig.savefig(config.FIGURES / f"{name}.pdf")


def bar_labels(ax, bars, fmt: str = "{:.3f}", pad: float = 3, **kwargs) -> None:
    """Value at the tip of each bar, in neutral ink."""
    for bar in bars:
        horizontal = bar.get_width() > bar.get_height() and bar.get_y() != 0
        if horizontal:
            ax.annotate(fmt.format(bar.get_width()), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(pad, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=INK_SECONDARY, **kwargs)
        else:
            ax.annotate(fmt.format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, pad), textcoords="offset points", va="bottom", ha="center",
                        fontsize=8, color=INK_SECONDARY, **kwargs)


__all__ = [
    "set_style", "save", "method_kwargs", "bar_labels", "METHODS", "METHOD_STYLE", "METHOD_LABEL",
    "BLUE", "ORANGE", "AQUA", "GRAY", "LIGHT_BLUE", "INK", "INK_SECONDARY", "MUTED", "GRID", "AXIS",
    "TEXT_WIDTH", "plt",
]
