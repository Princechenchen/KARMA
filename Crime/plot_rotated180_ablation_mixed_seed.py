"""Plot user-specified mixed-seed spatial illustrations for the four arms.

A/B/C use final seed 7416 and ROI-D uses seed 34. This is deliberately an
illustrative mixed-seed figure, not a paired same-split method comparison.
No model is refitted and no sampling result is changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from finalize_rotated180_ablation_10seed import collect_runs
from plot_rotated180_ablation_spatial import METHODS, draw_region, region_legend

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "figures" / "rotated180_ablation_mixed_seed"
METHOD_SEEDS = {
    "KARMA-Aligned": 7416,
    "KARMA-Misleading-Adaptive": 7416,
    "KARMA-Misleading-Fixed": 7416,
    "ROI-D": 34,
}
FILE_TAG = "ABC7416_D34"
DISPLAY = {
    "KARMA-Aligned": "KARMA-A",
    "KARMA-Misleading-Adaptive": "KARMA-R",
    "KARMA-Misleading-Fixed": "KARMA-wo arbitration",
    "ROI-D": "KARMA-wo knowledge",
}


def tile_size_for(ax, units, margin=1.18):
    """Compute a scatter-marker area that makes cell squares tile without seams.

    Measures the panel's final drawing extent, converts the median grid step
    to points, and adds a margin so antialiasing and later colorbar shrink
    cannot reopen gaps (squares overlap invisibly at alpha=1).
    """
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    pts_per_unit = min(bbox.width, bbox.height) / 1.03  
    xs = np.sort(units["x"].unique())
    step = float(np.median(np.diff(xs)))
    return (step * pts_per_unit * margin) ** 2


def draw_clean_region(ax, units, run, density_norm, title, point_size=52,
                      tile_size=None):
    """Draw a panel with the paper-facing variant name as a single-line title."""
    image = draw_region(ax, units, run, density_norm, title,
                        point_size=point_size, tile_size=tile_size)
    ax.set_title(title, fontsize=26, pad=10)
    ax.xaxis.label.set_size(21)
    ax.yaxis.label.set_size(21)
    ax.tick_params(labelsize=18)
    for text in ax.texts:
        text.set_fontsize(17)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data/units/units_2022_1km_hotspot.csv")
    runs, _ = collect_runs()
    selected = {}
    for method, _, _ in METHODS:
        seed = METHOD_SEEDS[method]
        row = runs[(runs["method"] == method) & (runs["seed"] == seed)]
        assert len(row) == 1
        selected[method] = row.iloc[0]

    density_cap = float(units["y1"].quantile(0.99))
    density_norm = Normalize(vmin=0, vmax=density_cap)
    paths = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.8), layout="constrained")
        image = None
        tile = tile_size_for(axes.flat[0], units)
        for ax, (method, _, _) in zip(axes.flat, METHODS):
            run = selected[method]
            image = draw_clean_region(ax, units, run, density_norm,
                                      DISPLAY[method], tile_size=tile)
        colorbar = fig.colorbar(
            image, ax=axes, shrink=0.82, label="Observed incidents/km²",
        )
        colorbar.set_label("Observed incidents/km²", fontsize=21)
        colorbar.ax.tick_params(labelsize=18)
        legend_handles = region_legend()
        legend_handles[2].set_label("Recommended regions R1–R4")
        fig.legend(
            handles=legend_handles, loc="outside lower center", ncol=3,
            frameon=False, fontsize=19, handlelength=2.2, columnspacing=1.2,
        )
        combined = OUT / f"four_arm_regions_mixed_{FILE_TAG}.png"
        fig.savefig(combined, dpi=220, transparent=False)
        plt.close(fig)
        paths.append(combined)

        for method, _, slug in METHODS:
            seed = METHOD_SEEDS[method]
            run = selected[method]
            fig, ax = plt.subplots(figsize=(7.2, 6.7), layout="constrained")
            tile = tile_size_for(ax, units)
            image = draw_clean_region(ax, units, run, density_norm,
                                      DISPLAY[method], point_size=68,
                                      tile_size=tile)
            colorbar = fig.colorbar(
                image, ax=ax, shrink=0.86, label="Observed incidents/km²",
            )
            colorbar.set_label("Observed incidents/km²", fontsize=13)
            colorbar.ax.tick_params(labelsize=11)
            ax.legend(handles=region_legend(), loc="upper right", fontsize=11, framealpha=0.92)
            path = OUT / f"{slug}_regions_seed{seed}.png"
            fig.savefig(path, dpi=220, transparent=False)
            plt.close(fig)
            paths.append(path)

    box_rows, sample_rows = [], []
    for method, _, _ in METHODS:
        run = selected[method]
        seed = METHOD_SEEDS[method]
        for box_index, (cx, cy, lx, ly) in enumerate(np.asarray(run["boxes"], dtype=float), 1):
            box_rows.append({"method": method, "seed": seed, "box": box_index,
                             "cx": cx, "cy": cy, "lx": lx, "ly": ly})
        for order, index in enumerate(map(int, run["training_indices"])):
            sample_rows.append({"method": method, "seed": seed, "training_order": order,
                                "unit_index_python_zero_based": index,
                                "cell_id": units.iloc[index]["cell_id"],
                                "sample_stage": "initial" if order < 30 else "acquired"})
    pd.DataFrame(box_rows).to_csv(OUT / f"mixed_{FILE_TAG}_region_boxes.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUT / f"mixed_{FILE_TAG}_training_samples.csv", index=False)

    provenance = {
        "method_seeds": METHOD_SEEDS,
        "selection": "User-specified mixed-seed illustration: A/B/C=7416, D=34.",
        "comparison_status": "Not a paired same-split comparison; do not use as a standalone fairness ranking.",
        "methods": [row[0] for row in METHODS],
        "final_training_size": 50,
        "initial_samples": 30,
        "acquired_samples": 20,
        "source_data": [
            "data/units/units_2022_1km_hotspot.csv",
            "results/final_10seed_rotated180_ablation/ablation_per_run_metrics.csv",
            "formal histories and valid recommendations collected by finalize_rotated180_ablation_10seed.collect_runs",
        ],
        "region_background": "Observed 2022 y1 incident count per occupied 1 km cell; identical for all panels.",
        "density_color_cap_99th_percentile": density_cap,
        "density_values_above_cap": "Retained and drawn at the cap color; not removed.",
        "missing_data": "Unoccupied cells are blank; no interpolation or zero imputation.",
        "model_refit_or_sampling_rerun": False,
        "figures": [str(path.relative_to(ROOT)) for path in paths],
        "matplotlib_version": matplotlib.__version__,
        "pandas_version": pd.__version__,
        "dpi": 220,
    }
    (OUT / f"provenance_mixed_{FILE_TAG}.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for path in paths:
        print(f"FIGURE={path}")
        print(f"SIZE_BYTES={path.stat().st_size}")


if __name__ == "__main__":
    main()
