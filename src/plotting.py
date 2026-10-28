"""Shared plot style so every figure in the project looks the same."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIG_DIR

# Fixed categorical order: colour follows the entity, never its rank.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
GREY = "#9a9994"
INK = "#52514e"

MODEL_COLORS = {"Mean baseline": GREY, "Elastic Net": AQUA,
                "Random Forest": ORANGE, "XGBoost": BLUE}

DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", [BLUE, "#f0efec", ORANGE])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#eaf2fb", BLUE, "#0d3a73"])


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cfcdc6",
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ecebe7",
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "font.family": "sans-serif",
    })


def save(fig, name):
    """Save a figure to outputs/figures/<name>.png (used by the report)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
