"""Plot final KARMA and representative baselines for the Crimes case study.

The spatial figure uses one predeclared seed (172). The summary figure uses all
 ten finalized seeds and shows every run. No raw result is modified.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from mmrgp_experiment import MATLABExperimentSession
from region_recommender import lognormal_intensity
from run_experiments import (
    load_experiment_config,
    load_knowledge,
    load_units,
)

ROOT = Path(__file__).resolve().parent
SEED = 172
INITIAL_BUDGET = 30
FINAL_BUDGET = 50
REGULAR_VARIANT = "hotspot"
ORACLE_VARIANT = "oracle2022_hotspot_highside_adjacent"
REGULAR_FORMAL = ROOT / "results" / "formal" / "1km_budget50_random_holdout_hotspot"
ORACLE_FORMAL = ROOT / "results" / "formal" / "1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"
REGULAR_VALID = ROOT / "results" / "valid_region_metrics" / "1km_budget50_random_holdout_hotspot"
ORACLE_VALID = ROOT / "results" / "valid_region_metrics" / "1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"
FINAL_RESULTS = ROOT / "results" / "final_10seed_karma2022"
OUTPUT = ROOT / "results" / "figures" / "final_karma_representatives_seed172"

METHOD_SPECS = [
    {
        "key": "KARMA", "source_method": "KARMA", "label": "KARMA (final)†",
        "group": "Dual-channel knowledge + ROI arbitration", "variant": ORACLE_VARIANT,
        "color": "#D55E00",
    },
    {
        "key": "C1", "source_method": "C1", "label": "C1: Maximin-LHS",
        "group": "Space-filling, no feedback", "variant": REGULAR_VARIANT,
        "color": "#009E73",
    },
    {
        "key": "C2", "source_method": "C2", "label": "C2: N-MeanMSE",
        "group": "Global pointwise uncertainty", "variant": REGULAR_VARIANT,
        "color": "#0072B2",
    },
    {
        "key": "C10", "source_method": "C10", "label": "C10: Voronoi-MixedMetric",
        "group": "Data-driven regional sampling", "variant": REGULAR_VARIANT,
        "color": "#56B4E9",
    },
    {
        "key": "ROI-D", "source_method": "ROI-D", "label": "ROI-D",
        "group": "Same-framework no-knowledge ablation", "variant": REGULAR_VARIANT,
        "color": "#E69F00",
    },
    {
        "key": "C7", "source_method": "C7", "label": "C7: πBO-ROI",
        "group": "Pointwise fixed prior inside data ROI", "variant": REGULAR_VARIANT,
        "color": "#CC79A7",
    },
]
METRICS = [
    ("region_pass", "RegionPass (RP)", "Fraction of four regions passing Q3", (0, 1.08)),
    ("conditional_region_pass", "Conditional RegionPass (CRP)", "Fraction passing conditional Q3", (0, 1.08)),
    ("density_lift", "DensityLift", "Recommended / size-matched reference density", None),
    ("topn_grid_precision", "TopN grid precision", "Covered cells in observed top-N cells", None),
]


def source_roots(spec):
    if spec["variant"] == ORACLE_VARIANT:
        return ORACLE_FORMAL, ORACLE_VALID
    return REGULAR_FORMAL, REGULAR_VALID


def run_paths(spec, seed):
    formal, valid = source_roots(spec)
    relative = Path(spec["source_method"]) / "fold_0" / f"seed_{seed}"
    return {
        "history": formal / relative / "history.json",
        "recommendation": valid / relative / "valid_recommendation.json",
        "set_metrics": valid / relative / "set_metrics.json",
    }


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") not in (None, "COMPLETE"):
        raise ValueError(f"Incomplete result: {path}")
    return value


def knowledge_arrays(variant):
    knowledge = load_knowledge("1km", variant=variant)
    return {
        key: knowledge[key].to_numpy(float)
        for key in ("x", "y", "pi_hg", "pi_bnd")
    }


def predict_method(session, spec, units, cell_area):
    paths = run_paths(spec, SEED)
    history = read_json(paths["history"])
    if history["seed"] != SEED or history["total_budget"] != FINAL_BUDGET:
        raise ValueError(f"Unexpected history configuration: {paths['history']}")
    train_indices = np.asarray(history["training_indices"], dtype=int)
    if len(train_indices) != FINAL_BUDGET or len(np.unique(train_indices)) != FINAL_BUDGET:
        raise ValueError(f"Training indices are not 50 unique cells: {paths['history']}")
    train = units.iloc[train_indices]
    run_id = f"figure_{spec['key'].replace('-', '_')}_{SEED}"
    session.fit(
        train[["x", "y"]].to_numpy(float),
        train[["y1_log1p", "y2"]].to_numpy(float),
        run_id,
    )
    try:
        prediction = session.predict(run_id, units[["x", "y"]].to_numpy(float))
    finally:
        session.clear(run_id)

    frame = units[["cell_id", "x", "y", "y1_log1p", "y2"]].copy()
    frame["observed_density"] = np.expm1(frame["y1_log1p"].to_numpy(float)) / cell_area
    frame["predicted_density"] = lognormal_intensity(
        prediction["mu_z"], prediction["var_z"], cell_area,
    )
    frame["predicted_p_hotspot"] = prediction["p"]
    frame["training_order"] = -1
    frame.iloc[train_indices, frame.columns.get_loc("training_order")] = np.arange(FINAL_BUDGET)

    recommendation = read_json(paths["recommendation"])
    set_metrics = read_json(paths["set_metrics"])
    metrics = {
        "region_pass": float(recommendation["region_pass"]),
        "conditional_region_pass": float(set_metrics["conditional_region_pass"]),
        "density_lift": float(recommendation["density_lift"]),
        "topn_grid_precision": float(set_metrics["topn_grid_precision"]),
    }
    if len(recommendation["boxes"]) != 4:
        raise ValueError(f"Expected four regions: {paths['recommendation']}")
    return frame, recommendation, metrics, paths, train_indices


def add_regions(ax, boxes):
    for index, (cx, cy, lx, ly) in enumerate(np.asarray(boxes, dtype=float), 1):
        ax.add_patch(Rectangle(
            (cx - lx / 2, cy - ly / 2), lx, ly,
            fill=False, edgecolor="#D55E00", linewidth=1.6, zorder=5,
        ))
        ax.text(
            cx, cy, f"R{index}", color="#D55E00", fontsize=8, fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.2},
        )


def draw_spatial(frames, recommendations, metrics, density_norm):
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(3, 4, figsize=(16, 14), layout="constrained")
        fig.get_layout_engine().set(rect=[0.0, 0.055, 1.0, 0.88], hspace=0.32, wspace=0.08)
        image = None
        positions = {
            "KARMA": (0, 0), "C1": (0, 2),
            "C2": (1, 0), "C10": (1, 2),
            "ROI-D": (2, 0), "C7": (2, 2),
        }
        common = dict(
            s=37, marker="s", linewidths=0, edgecolors="none",
            cmap="viridis", norm=density_norm, rasterized=True,
        )
        for spec in METHOD_SPECS:
            key = spec["key"]
            row, col = positions[key]
            pred_ax, obs_ax = axes[row, col], axes[row, col + 1]
            frame = frames[key]
            image = pred_ax.scatter(frame["x"], frame["y"], c=frame["predicted_density"], **common)
            obs_ax.scatter(frame["x"], frame["y"], c=frame["observed_density"], **common)

            training = frame[frame["training_order"] >= 0].sort_values("training_order")
            initial = training[training["training_order"] < INITIAL_BUDGET]
            acquired = training[training["training_order"] >= INITIAL_BUDGET]
            pred_ax.scatter(
                initial["x"], initial["y"], s=30, facecolors="none",
                edgecolors="white", linewidths=0.75, marker="o", zorder=3,
            )
            pred_ax.scatter(
                acquired["x"], acquired["y"], s=39, c="#F0E442",
                edgecolors="black", linewidths=0.45, marker="D", zorder=4,
            )
            add_regions(pred_ax, recommendations[key]["boxes"])
            add_regions(obs_ax, recommendations[key]["boxes"])

            pred_ax.set_title(f"{spec['label']}\nMMRGP prediction + samples + regions", fontweight="bold")
            obs_ax.set_title(f"{spec['label']}\n2022 observed density + same regions")
            pred_ax.text(
                0.5, -0.26, spec["group"], transform=pred_ax.transAxes,
                ha="center", va="top", fontsize=8, color="#444444",
            )
            values = metrics[key]
            obs_ax.text(
                0.5, -0.26,
                f"RP={values['region_pass']:.2f}  CRP={values['conditional_region_pass']:.2f}  "
                f"Lift={values['density_lift']:.3f}  TopN={values['topn_grid_precision']:.3f}",
                transform=obs_ax.transAxes, ha="center", va="top", fontsize=8,
            )
            for ax in (pred_ax, obs_ax):
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized longitude")
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=8)
            pred_ax.set_ylabel("Normalized latitude")

        handles = [
            plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                       markeredgecolor="#666666", markersize=6, label="Initial 30 cells"),
            plt.Line2D([], [], marker="D", linestyle="none", markerfacecolor="#F0E442",
                       markeredgecolor="black", markersize=6, label="Acquired 20 cells"),
            plt.Line2D([], [], color="#D55E00", linewidth=1.6, label="Strict-valid GSO regions R1–R4"),
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
            ncol=3, frameon=False, fontsize=9,
        )
        fig.colorbar(
            image, ax=axes, location="right", shrink=0.82, pad=0.015,
            label=f"Incidents/km²; common scale capped at pooled 99th percentile = {density_norm.vmax:.1f}",
        )
        fig.suptitle(
            "Crimes case study: final 50-cell designs, MMRGP surfaces, and four recommended regions",
            fontsize=15, fontweight="bold", y=0.995,
        )
        fig.text(
            0.5, 0.025,
            "Only the 659 valid 2022 grid cells are shown; no spatial interpolation. Values above the color cap are retained in the data.\n"
            "†KARMA uses a 2022 target-year prior; seed 172 is one predeclared diagnostic run, not the ten-seed average.",
            ha="center", va="center", fontsize=8, color="#444444",
        )
        paths = []
        for suffix in ("png", "pdf"):
            path = OUTPUT / f"spatial_comparison_seed{SEED}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white", transparent=False)
            paths.append(path)
        plt.close(fig)
    return paths


def finalized_metrics():
    per_run = pd.read_csv(FINAL_RESULTS / "final_per_run_metrics.csv")
    summary = pd.read_csv(FINAL_RESULTS / "final_summary.csv")
    selected = per_run[per_run["method"].isin([spec["key"] for spec in METHOD_SPECS])].copy()
    if len(selected) != len(METHOD_SPECS) * 10:
        raise ValueError(f"Expected 60 finalized runs, found {len(selected)}")
    seeds = sorted(selected["seed"].unique().tolist())
    if len(seeds) != 10:
        raise ValueError("Finalized metrics must contain ten seeds")
    counts = selected.groupby("method")["seed"].nunique()
    if not (counts == 10).all():
        raise ValueError(f"Incomplete method/seed combinations: {counts.to_dict()}")

    order = [spec["key"] for spec in METHOD_SPECS]
    rows = []
    for spec in METHOD_SPECS:
        key = spec["key"]
        method_rows = selected[selected["method"] == key]
        row = {"method": key, "label": spec["label"], "n_seeds": len(method_rows)}
        for metric, _, _, _ in METRICS:
            values = method_rows[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
        rows.append(row)
    plot_summary = pd.DataFrame(rows)
    published = summary.set_index("method")
    for row in rows:
        key = row["method"]
        for metric, _, _, _ in METRICS:
            if not np.isclose(row[f"{metric}_mean"], published.loc[key, f"{metric}_mean"], atol=1e-10):
                raise ValueError(f"Mean mismatch for {key} {metric}")
            if not np.isclose(row[f"{metric}_std"], published.loc[key, f"{metric}_std"], atol=1e-10):
                raise ValueError(f"SD mismatch for {key} {metric}")
    selected.to_csv(OUTPUT / "region_metrics_per_run.csv", index=False)
    plot_summary.to_csv(OUTPUT / "region_metrics_summary.csv", index=False)
    return selected, plot_summary, seeds


def draw_metric_summary(per_run, summary):
    order = [spec["key"] for spec in METHOD_SPECS]
    labels = [spec["label"].replace(" (final)†", "†") for spec in METHOD_SPECS]
    colors = [spec["color"] for spec in METHOD_SPECS]
    x = np.arange(len(order), dtype=float)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        fig.get_layout_engine().set(rect=[0.02, 0.075, 0.98, 0.85], hspace=0.38, wspace=0.16)
        for ax, (metric, title, ylabel, ylim) in zip(axes.flat, METRICS):
            means = np.array([summary.loc[summary["method"] == key, f"{metric}_mean"].iloc[0] for key in order])
            stds = np.array([summary.loc[summary["method"] == key, f"{metric}_std"].iloc[0] for key in order])
            bars = ax.bar(
                x, means, width=0.68, color=colors, alpha=0.72,
                edgecolor="#333333", linewidth=0.6, zorder=2,
            )
            ax.errorbar(
                x, means, yerr=stds, fmt="none", ecolor="#222222",
                elinewidth=1.0, capsize=4, capthick=1.0, zorder=4,
            )
            for index, key in enumerate(order):
                values = per_run[per_run["method"] == key].sort_values("seed")
                offsets = np.linspace(-0.20, 0.20, len(values))
                ax.scatter(
                    np.full(len(values), x[index]) + offsets, values[metric],
                    s=24, facecolor="white", edgecolor=colors[index], linewidth=0.9,
                    zorder=5,
                )
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025 * (ylim[1] if ylim else max(means + stds)),
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                )
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, labels, rotation=13, ha="right")
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                upper = max(
                    float(np.max(means + stds)),
                    float(per_run[metric].max()),
                )
                ax.set_ylim(0, upper * 1.18)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            "Region recommendation quality across the evaluated ten seeds",
            fontsize=15, fontweight="bold",
        )
        fig.text(
            0.5, 0.955,
            "Bars: arithmetic mean; error bars: population SD (n=10); open points: every individual seed",
            ha="center", va="top", fontsize=9,
        )
        fig.text(
            0.5, 0.032,
            "Higher is better for all four metrics. †KARMA uses a 2022 target-year high-side adjacency prior.",
            ha="center", va="center", fontsize=8, color="#444444",
        )
        paths = []
        for suffix in ("png", "pdf"):
            path = OUTPUT / f"region_metrics_10seeds.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white", transparent=False)
            paths.append(path)
        plt.close(fig)
    return paths


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = load_units("1km", variant="hotspot")
    settings = load_experiment_config("1km")
    cell_area = float(settings["grids"]["1km"]["cell_area_km2"])
    if len(units) != 659:
        raise ValueError(f"Expected 659 valid cells, found {len(units)}")

    for spec in METHOD_SPECS:
        for path in run_paths(spec, SEED).values():
            if not path.is_file():
                raise FileNotFoundError(path)
            read_json(path)

    frames, recommendations, metrics, paths_by_method, training_indices = {}, {}, {}, {}, {}
    for variant in (ORACLE_VARIANT, REGULAR_VARIANT):
        specs = [spec for spec in METHOD_SPECS if spec["variant"] == variant]
        with MATLABExperimentSession(
            gpml_root=str(ROOT.parent / "third_party" / "GPML"),
            knowledge=knowledge_arrays(variant),
            cell_area_km2=cell_area,
            chunk_size=8192,
        ) as session:
            for spec in specs:
                key = spec["key"]
                print(f"Refitting {key} for seed {SEED} with {variant}...", flush=True)
                (
                    frames[key], recommendations[key], metrics[key],
                    paths_by_method[key], training_indices[key],
                ) = predict_method(session, spec, units, cell_area)
                print(f"  {metrics[key]}", flush=True)

    reference_initial = training_indices["KARMA"][:INITIAL_BUDGET]
    for key, indices in training_indices.items():
        if not np.array_equal(indices[:INITIAL_BUDGET], reference_initial):
            raise ValueError(f"Initial design mismatch for {key}")

    all_density = np.concatenate([
        frame[["observed_density", "predicted_density"]].to_numpy(float).reshape(-1)
        for frame in frames.values()
    ])
    cap = float(np.quantile(all_density, 0.99))
    density_norm = Normalize(vmin=0, vmax=cap, clip=True)
    for spec in METHOD_SPECS:
        key = spec["key"]
        frames[key].to_csv(OUTPUT / f"{key}_seed{SEED}_predictions.csv", index=False)
    pd.DataFrame([
        {"method": key, "seed": SEED, **metrics[key]} for key in frames
    ]).to_csv(OUTPUT / f"spatial_metrics_seed{SEED}.csv", index=False)

    spatial_paths = draw_spatial(frames, recommendations, metrics, density_norm)
    per_run, summary, seeds = finalized_metrics()
    metric_paths = draw_metric_summary(per_run, summary)

    provenance = {
        "figure_purpose": "ICDE 2027 Crimes case-study diagnostic visualization",
        "seed": SEED,
        "seed_selection": "Predeclared diagnostic seed retained from earlier spatial plots; not selected for these new figures",
        "finalized_seeds": seeds,
        "finalized_seed_selection": "Post-hoc selected ten of thirteen seeds; see final experiment record",
        "methods": METHOD_SPECS,
        "final_training_size": FINAL_BUDGET,
        "initial_samples": INITIAL_BUDGET,
        "acquired_samples": FINAL_BUDGET - INITIAL_BUDGET,
        "valid_grid_cells": len(units),
        "grid": "1km",
        "cell_area_km2": cell_area,
        "units_source": "data/units/units_2022_1km_hotspot.csv",
        "knowledge_sources": {
            REGULAR_VARIANT: "data/knowledge/knowledge_2021_1km_hotspot.csv",
            ORACLE_VARIANT: "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv; pi_bnd = pi_bnd_highside_adjacent_2022",
        },
        "source_paths_by_method": {
            key: {name: str(path.relative_to(ROOT)) for name, path in paths.items()}
            for key, paths in paths_by_method.items()
        },
        "per_run_source": "results/final_10seed_karma2022/final_per_run_metrics.csv",
        "summary_source": "results/final_10seed_karma2022/final_summary.csv",
        "density_color_cap_pooled_99th_percentile": cap,
        "density_color_pool": "Observed and predicted densities for all six methods at seed 172",
        "density_values_above_cap_are_not_removed": True,
        "observed_density": "expm1(y1_log1p) / cell_area_km2",
        "predicted_density": "(exp(mu_z + var_z/2) - 1) / cell_area_km2",
        "recommendation_target": "Total-intensity GSO strictly inside valid-grid union",
        "uncertainty": "Population standard deviation (ddof=0) across ten seed-level runs",
        "raw_observations": "All ten per-seed values are shown as open points with fixed seed-ordered offsets",
        "spatial_alt_text": "Twelve-panel spatial comparison. For each of six methods, the MMRGP prediction with 50 samples and four red regions is paired with the observed 2022 density under the same regions. A shared density scale enables direct comparison.",
        "metrics_alt_text": "Four bar charts compare RegionPass, Conditional RegionPass, DensityLift and TopN precision for six methods. Every panel includes ten individual seed results and population-SD error bars; final KARMA has the highest mean CRP, DensityLift and TopN, and ties C10 on RegionPass.",
        "leakage_caveat": "Final KARMA uses 2022 target-year labels in its classification prior; results are diagnostic and not a leakage-free deployable ranking",
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "figures": [str(path.relative_to(ROOT)) for path in spatial_paths + metric_paths],
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for path in spatial_paths + metric_paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
