"""One shared look for every figure, so the deck feels like one project."""
import matplotlib.pyplot as plt

def use_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
        "font.size": 10.5, "font.family": "DejaVu Sans",
        "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "legend.frameon": False, "lines.linewidth": 1.8,
        "axes.prop_cycle": plt.cycler(color=[
            "#1f5f8b", "#c1403d", "#e08a00", "#157a3a", "#6b4fa1", "#555555"]),
    })
