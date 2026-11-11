"""Plot final regions for the four ablation arms and aligned/rotated priors.

The current figure uses illustrative final seed 7416, selected because the
aligned arm is strong and the fixed misleading-prior arm is weak. It does not
refit models, rerun sampling, or alter any result. All boxes and sample indices
come from the finalized ablation runs; backgrounds use observed 2022 values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from finalize_rotated180_ablation_10seed import collect_runs, prior_diagnostics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "figures" / "rotated180_ablation_spatial"
SEED = 7416
METHODS = [
    ("KARMA-Aligned", "A  KARMA-Aligned\nCorrect prior + adaptive arbitration", "karma_aligned"),
    ("KARMA-Misleading-Adaptive", "B  Rotated-Adaptive\nMisleading prior + adaptive arbitration", "rotated_adaptive"),
    ("KARMA-Misleading-Fixed", "C  Rotated-Fixed\nMisleading prior + fixed arbitration", "rotated_fixed"),
    ("ROI-D", "D  ROI-D\nNo effective prior", "roi_d"),
]


def style_axes(ax):
    ax.set(
        xlim=(-0.015, 1.015),
        ylim=(-0.015, 1.015),
        xlabel="Normalized longitude",
        ylabel="Normalized latitude",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)


def draw_region(ax, units, run, density_norm, title, point_size=48, tile_size=None):
    image = ax.scatter(
        units["x"], units["y"], c=units["y1"], marker="s",
        s=point_size if tile_size is None else tile_size,
        cmap="viridis", norm=density_norm, linewidths=0, edgecolors="none",
    )
    training = units.iloc[list(map(int, run["training_indices"]))]
    initial = training.iloc[:30]
    acquired = training.iloc[30:]
    ax.scatter(
        initial["x"], initial["y"], s=point_size * 0.62, marker="o",
        facecolors="none", edgecolors="white", linewidths=0.9,
    )
    ax.scatter(
        acquired["x"], acquired["y"], s=point_size * 0.72, marker="D",
        c="#FF9F1C", edgecolors="black", linewidths=0.45,
    )
    for index, (cx, cy, lx, ly) in enumerate(np.asarray(run["boxes"], dtype=float), 1):
        ax.add_patch(Rectangle(
            (cx - lx / 2, cy - ly / 2), lx, ly,
            fill=False, edgecolor="#D62728", linewidth=1.9,
        ))
        ax.text(
            cx, cy, f"R{index}", color="#B2182B", fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.3},
        )
    ax.set_title(
        f"{title}\n"
        f"RP={run['region_pass']:.2f}  CRP={run['conditional_region_pass']:.2f}  "
        f"DL={run['density_lift']:.3f}  TopN={run['topn_grid_precision']:.3f}",
        fontsize=11,
    )
    style_axes(ax)
    return image


def region_legend():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="white", markeredgewidth=1.2, markersize=8,
               label="Initial 30 samples"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#FF9F1C",
               markeredgecolor="black", markersize=7, label="Acquired 20 samples"),
        Line2D([0], [0], color="#D62728", linewidth=1.9,
               label="Final strict-valid regions R1–R4"),
    ]


def save_region_figures(selected, units, density_norm):
    paths = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.0), layout="constrained")
        image = None
        for ax, (method, title, _) in zip(axes.flat, METHODS):
            run = selected.loc[method]
            image = draw_region(ax, units, run, density_norm, title)
        fig.suptitle(
            f"Four-arm region recommendation comparison — seed {SEED}, final 50 samples\n"
            "Background: observed 2022 total incidents per 1 km cell; identical color scale in all panels",
            fontsize=15,
        )
        fig.colorbar(
            image, ax=axes, shrink=0.82,
            label=f"Observed incidents/km² (color capped at 99th percentile = {density_norm.vmax:.1f})",
        )
        fig.legend(handles=region_legend(), loc="outside lower center", ncol=3, frameon=False)
        combined = OUT / f"four_arm_regions_seed{SEED}.png"
        fig.savefig(combined, dpi=220, transparent=False)
        plt.close(fig)
        paths.append(combined)

        for method, title, slug in METHODS:
            fig, ax = plt.subplots(figsize=(7.2, 6.5), layout="constrained")
            run = selected.loc[method]
            image = draw_region(ax, units, run, density_norm, title, point_size=68)
            fig.colorbar(
                image, ax=ax, shrink=0.86,
                label=f"Observed incidents/km² (cap = {density_norm.vmax:.1f})",
            )
            ax.legend(handles=region_legend(), loc="upper right", fontsize=8, framealpha=0.92)
            path = OUT / f"{slug}_regions_seed{SEED}.png"
            fig.savefig(path, dpi=220, transparent=False)
            plt.close(fig)
            paths.append(path)
    return paths


def draw_prior(ax, knowledge, value, title, norm):
    image = ax.scatter(
        knowledge["x"], knowledge["y"], c=knowledge[value], marker="s", s=66,
        cmap="cividis", norm=norm, linewidths=0, edgecolors="none",
    )
    ax.set_title(title, fontsize=16)
    style_axes(ax)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(labelsize=12)
    return image


def save_prior_figures(knowledge, units, diagnostics):
    norm = Normalize(0, 1)
    panels = [
        ("pi_bnd_highside_adjacent_2022", "A  Correct prior: 2022 high-side adjacent boundary", "aligned_correct_prior"),
        ("pi_bnd_highside_rotated180_2022", "B  Misleading prior: spatially rotated 180°", "rotated180_misleading_prior"),
    ]
    paths = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), layout="constrained")
        image = None
        for ax, (value, title, _) in zip(axes, panels):
            image = draw_prior(ax, knowledge, value, title, norm)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.86, label="Knowledge intensity π_bnd")
        colorbar.set_label("Knowledge intensity π_bnd", fontsize=14)
        colorbar.ax.tick_params(labelsize=12)
        combined = OUT / "aligned_vs_rotated180_priors.png"
        fig.savefig(combined, dpi=220, transparent=False)
        plt.close(fig)
        paths.append(combined)

        for value, title, slug in panels:
            fig, ax = plt.subplots(figsize=(7.2, 6.2), layout="constrained")
            image = draw_prior(ax, knowledge, value, title, norm)
            colorbar = fig.colorbar(image, ax=ax, shrink=0.86, label="Knowledge intensity π_bnd")
            colorbar.set_label("Knowledge intensity π_bnd", fontsize=13)
            colorbar.ax.tick_params(labelsize=11)
            path = OUT / f"{slug}.png"
            fig.savefig(path, dpi=220, transparent=False)
            plt.close(fig)
            paths.append(path)
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data/units/units_2022_1km_hotspot.csv")
    knowledge = pd.read_csv(ROOT / "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv")
    runs, _ = collect_runs()
    selected = runs[runs["seed"] == SEED].set_index("method")
    assert len(selected) == 4 and set(selected.index) == {row[0] for row in METHODS}
    assert len(units) == 659 and len(knowledge) == 657
    assert np.isfinite(knowledge[["pi_bnd_highside_adjacent_2022", "pi_bnd_highside_rotated180_2022"]]).all().all()

    density_cap = float(units["y1"].quantile(0.99))
    density_norm = Normalize(vmin=0, vmax=density_cap)
    diagnostics = prior_diagnostics(units, knowledge)
    region_paths = save_region_figures(selected, units, density_norm)
    prior_paths = save_prior_figures(knowledge, units, diagnostics)

    box_rows, sample_rows = [], []
    for method, _, _ in METHODS:
        run = selected.loc[method]
        for box_index, (cx, cy, lx, ly) in enumerate(np.asarray(run["boxes"], dtype=float), 1):
            box_rows.append({"method": method, "seed": SEED, "box": box_index,
                             "cx": cx, "cy": cy, "lx": lx, "ly": ly})
        for order, index in enumerate(map(int, run["training_indices"])):
            sample_rows.append({"method": method, "seed": SEED, "training_order": order,
                                "unit_index_python_zero_based": index,
                                "cell_id": units.iloc[index]["cell_id"],
                                "sample_stage": "initial" if order < 30 else "acquired"})
    pd.DataFrame(box_rows).to_csv(OUT / f"seed{SEED}_region_boxes.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUT / f"seed{SEED}_training_samples.csv", index=False)

    provenance = {
        "seed": SEED,
        "seed_choice": "Illustrative final seed selected for strong aligned-arm and weak fixed-misleading-arm regional metrics; not an average result.",
        "methods": [row[0] for row in METHODS],
        "final_training_size": 50,
        "initial_samples": 30,
        "acquired_samples": 20,
        "source_data": [
            "data/units/units_2022_1km_hotspot.csv",
            "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv",
            "results/final_10seed_rotated180_ablation/ablation_per_run_metrics.csv",
            "formal histories and valid recommendations collected by finalize_rotated180_ablation_10seed.collect_runs",
        ],
        "region_background": "Observed 2022 y1 incident count per occupied 1 km cell; identical for all four arms.",
        "density_color_cap_99th_percentile": density_cap,
        "density_values_above_cap": "Retained in data and drawn at the cap color; not removed.",
        "prior_fields": ["pi_bnd_highside_adjacent_2022", "pi_bnd_highside_rotated180_2022"],
        "prior_scale": [0, 1],
        "prior_diagnostics": diagnostics,
        "missing_data": "Unoccupied cells are blank; no interpolation or zero imputation in the figures.",
        "model_refit_or_sampling_rerun": False,
        "figures": [str(path.relative_to(ROOT)) for path in region_paths + prior_paths],
        "matplotlib_version": matplotlib.__version__,
        "pandas_version": pd.__version__,
        "dpi": 220,
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for path in region_paths + prior_paths:
        print(f"FIGURE={path}")
        print(f"SIZE_BYTES={path.stat().st_size}")


if __name__ == "__main__":
    main()
