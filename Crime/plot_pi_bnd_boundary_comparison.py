"""Plot original/adjacent boundary knowledge against observed 2022 responses."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "figures" / "pi_bnd_boundary_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km_adjacent.csv")

required_source = {"cell_id", "ix", "iy", "y2"}
required_target = {"cell_id", "ix", "iy", "violent_ratio", "y2", "is_classification_boundary"}
required_knowledge = {"cell_id", "pi_bnd", "pi_bnd_adjacent", "is_classification_boundary_2021"}
assert required_source.issubset(source.columns)
assert required_target.issubset(target.columns)
assert required_knowledge.issubset(knowledge.columns)
assert source["cell_id"].is_unique and target["cell_id"].is_unique and knowledge["cell_id"].is_unique

boundary_2022 = target[target["is_classification_boundary"] == 1]
boundary_2021 = knowledge[knowledge["is_classification_boundary_2021"] == 1]

with plt.rc_context({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")

    def base_panel(ax, frame, value, title, cmap, vmin, vmax, colorbar_label):
        artist = ax.scatter(
            frame["ix"], frame["iy"], c=frame[value], marker="s", s=48,
            cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("Grid x index (1 km cells)")
        ax.set_ylabel("Grid y index (1 km cells)")
        ax.set_aspect("equal")
        ax.set_xlim(target["ix"].min() - 0.8, target["ix"].max() + 0.8)
        ax.set_ylim(target["iy"].min() - 0.8, target["iy"].max() + 0.8)
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.84)
        colorbar.set_label(colorbar_label)
        return artist

    def overlay_2022_boundary(ax):
        ax.scatter(
            boundary_2022["ix"], boundary_2022["iy"], marker="s", s=82,
            facecolors="none", edgecolors="white", linewidths=1.6,
        )
        ax.scatter(
            boundary_2022["ix"], boundary_2022["iy"], marker="s", s=76,
            facecolors="none", edgecolors="black", linewidths=0.65,
        )

    base_panel(
        axes[0, 0], knowledge, "pi_bnd",
        "A  Original 2021 $\\pi_{bnd}$: threshold proximity",
        "cividis", 0, 1, "Knowledge intensity",
    )
    overlay_2022_boundary(axes[0, 0])

    base_panel(
        axes[0, 1], knowledge, "pi_bnd_adjacent",
        "B  Adjacent 2021 $\\pi_{bnd}$: distance to class boundary",
        "cividis", 0, 1, "Knowledge intensity",
    )
    axes[0, 1].scatter(
        boundary_2021["ix"], boundary_2021["iy"], marker="o", s=9,
        c="#00c8ff", edgecolors="none",
    )
    overlay_2022_boundary(axes[0, 1])

    base_panel(
        axes[1, 0], target, "violent_ratio",
        "C  Observed 2022 violent ratio",
        "magma", 0, 1, "Violent incidents / all incidents",
    )
    overlay_2022_boundary(axes[1, 0])

    categorical = ListedColormap(["#4575b4", "#d73027"])
    base_panel(
        axes[1, 1], target, "y2",
        "D  Observed 2022 $y_2$ (fixed threshold = 1/3)",
        categorical, 0, 1, "$y_2$: 0 = non-high, 1 = high",
    )
    overlay_2022_boundary(axes[1, 1])

    legend_items = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=9, label="2022 true class boundary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#00c8ff",
               markeredgecolor="none", markersize=6, label="2021 source class boundary (panel B)"),
    ]
    fig.legend(handles=legend_items, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Boundary knowledge versus observed 2022 responses (1 km grid)", fontsize=16)
    fig.text(
        0.5, 0.005,
        "Empty locations are unoccupied cells, not zero-valued observations. "
        "Knowledge panels contain 2021 occupied cells only; no spatial interpolation is shown.",
        ha="center", fontsize=9,
    )

    output = OUT_DIR / "pi_bnd_vs_2022_boundary.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)

provenance = {
    "figure": str(output.relative_to(ROOT)),
    "source_files": [
        "data/units/units_2021_1km.csv",
        "data/units/units_2022_1km.csv",
        "data/knowledge/knowledge_2021_1km_adjacent.csv",
    ],
    "plot_script": "plot_pi_bnd_boundary_comparison.py",
    "encodings": {
        "knowledge_panels": "cividis, fixed [0, 1]",
        "violent_ratio": "magma, fixed [0, 1]",
        "y2": "blue = 0, red = 1",
        "2022_boundary": "open black squares with white outer outline",
        "2021_boundary": "cyan dots in panel B only",
    },
    "missing_data": "Unoccupied cells are left blank; no interpolation or zero imputation.",
    "figure_size_inches": [14, 11],
    "dpi": 180,
    "matplotlib_version": matplotlib.__version__,
    "pandas_version": pd.__version__,
}
(OUT_DIR / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"FIGURE={output}")
print(f"SIZE_BYTES={output.stat().st_size}")
