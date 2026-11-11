"""补充材料版: 全部 15 方法的区域推荐图 (seed 172 诊断种子, 与主文 Fig5 同款绘图方式)。
4x8 = 32 面板(15 方法 x 预测/实况双面板 + KARMA 占 oracle 变体); KARMA-HG/KARMA-BND 无
hotspot 正式轮或无推荐数据的跳过并提示。输出: results/figures/fig5_suppmat/。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

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
OUTPUT = ROOT / "results" / "figures" / "fig5_suppmat"

METHOD_SPECS = [
    {"key": "KARMA", "label": "KARMA", "variant": ORACLE_VARIANT},
    {"key": "C1", "label": "MaximinLHS", "variant": REGULAR_VARIANT},
    {"key": "C2", "label": "MSE", "variant": REGULAR_VARIANT},
    {"key": "C3", "label": "IMSE", "variant": REGULAR_VARIANT},
    {"key": "C4", "label": "PALC", "variant": REGULAR_VARIANT},
    {"key": "C5", "label": "AK-MCS-IRS", "variant": REGULAR_VARIANT},
    {"key": "C6", "label": "piBO-global", "variant": REGULAR_VARIANT},
    {"key": "C7", "label": "piBO-ROI", "variant": REGULAR_VARIANT},
    {"key": "C9", "label": "KeyRegion", "variant": REGULAR_VARIANT},
    {"key": "C10", "label": "Voronoi", "variant": REGULAR_VARIANT},
    {"key": "C11", "label": "HHs", "variant": REGULAR_VARIANT},
    {"key": "C12", "label": "QBC-global", "variant": REGULAR_VARIANT},
]


def source_roots(spec):
    if spec["variant"] == ORACLE_VARIANT:
        return ORACLE_FORMAL, ORACLE_VALID
    return REGULAR_FORMAL, REGULAR_VALID


def run_paths(spec, seed):
    formal, valid = source_roots(spec)
    src = spec.get("source_method", spec["key"])
    relative = Path(src) / "fold_0" / f"seed_{seed}"
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
    return {key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")}


def predict_method(session, spec, units, cell_area):
    paths = run_paths(spec, SEED)
    history = read_json(paths["history"])
    train_indices = np.asarray(history["training_indices"], dtype=int)
    train = units.iloc[train_indices]
    run_id = f"fig5s_{spec['key'].replace('-', '_')}_{SEED}"
    session.fit(train[["x", "y"]].to_numpy(float), train[["y1_log1p", "y2"]].to_numpy(float), run_id)
    try:
        prediction = session.predict(run_id, units[["x", "y"]].to_numpy(float))
    finally:
        session.clear(run_id)

    frame = units[["cell_id", "x", "y", "y1_log1p", "y2"]].copy()
    frame["observed_density"] = np.expm1(frame["y1_log1p"].to_numpy(float)) / cell_area
    frame["predicted_density"] = lognormal_intensity(prediction["mu_z"], prediction["var_z"], cell_area)
    frame["training_order"] = -1
    frame.iloc[train_indices, frame.columns.get_loc("training_order")] = np.arange(FINAL_BUDGET)

    recommendation = read_json(paths["recommendation"])
    set_metrics = read_json(paths["set_metrics"])
    metrics = {"crp": float(set_metrics["conditional_region_pass"]),
               "lift": float(recommendation["density_lift"])}
    return frame, recommendation, metrics


def add_regions(ax, boxes):
    for index, (cx, cy, lx, ly) in enumerate(np.asarray(boxes, dtype=float), 1):
        ax.add_patch(Rectangle((cx - lx / 2, cy - ly / 2), lx, ly,
                               fill=False, edgecolor="#D55E00", linewidth=1.6, zorder=5))
        ax.text(cx, cy, f"R{index}", color="#D55E00", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 0.9})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = load_units("1km", variant="hotspot")
    settings = load_experiment_config("1km")
    cell_area = float(settings["grids"]["1km"]["cell_area_km2"])

    available = []
    for spec in METHOD_SPECS:
        try:
            for path in run_paths(spec, SEED).values():
                read_json(path)
            available.append(spec)
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP {spec['key']}: {type(e).__name__}")
    print(f"{len(available)}/{len(METHOD_SPECS)} methods available")

    frames, recommendations, metrics = {}, {}, {}
    for variant in (ORACLE_VARIANT, REGULAR_VARIANT):
        specs = [sp for sp in available if sp["variant"] == variant]
        if not specs:
            continue
        with MATLABExperimentSession(
            gpml_root=str(ROOT.parent / "third_party" / "GPML"),
            knowledge=knowledge_arrays(variant),
            cell_area_km2=cell_area, chunk_size=8192,
        ) as session:
            for spec in specs:
                print(f"Refitting {spec['key']} ({variant})...", flush=True)
                frames[spec["key"]], recommendations[spec["key"]], metrics[spec["key"]] = \
                    predict_method(session, spec, units, cell_area)

    all_density = np.concatenate([
        f[["observed_density", "predicted_density"]].to_numpy(float).reshape(-1)
        for f in frames.values()
    ])
    cap = float(np.quantile(all_density, 0.99))
    density_norm = Normalize(vmin=0, vmax=cap, clip=True)

    n = len(available)
    ncols = 4
    nrows = int(np.ceil(n / (ncols // 2)))   
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.4 * nrows), layout="constrained")
        fig.get_layout_engine().set(rect=[0, 0.015, 1.0, 0.925], hspace=0.10, wspace=0.04)
        axes = np.atleast_2d(axes)
        image = None
        for idx, spec in enumerate(available):
            key = spec["key"]
            row, col = idx // (ncols // 2), (idx % (ncols // 2)) * 2
            pred_ax, obs_ax = axes[row, col], axes[row, col + 1]
            frame = frames[key]
            common = dict(s=30, marker="s", linewidths=0, edgecolors="none",
                          cmap="viridis", norm=density_norm, rasterized=True)
            image = pred_ax.scatter(frame["x"], frame["y"], c=frame["predicted_density"], **common)
            obs_ax.scatter(frame["x"], frame["y"], c=frame["observed_density"], **common)
            training = frame[frame["training_order"] >= 0].sort_values("training_order")
            initial = training[training["training_order"] < INITIAL_BUDGET]
            acquired = training[training["training_order"] >= INITIAL_BUDGET]
            pred_ax.scatter(initial["x"], initial["y"], s=26, facecolors="none",
                            edgecolors="white", linewidths=0.7, marker="o", zorder=3)
            pred_ax.scatter(acquired["x"], acquired["y"], s=34, c="#F0E442",
                            edgecolors="black", linewidths=0.4, marker="D", zorder=4)
            add_regions(pred_ax, recommendations[key]["boxes"])
            add_regions(obs_ax, recommendations[key]["boxes"])
            pred_ax.set_title(spec["label"], fontweight="bold", pad=4)
            obs_ax.set_title(f"{spec['label']} (observed)", fontweight="bold", pad=4)
            v = metrics[key]
            obs_ax.text(0.5, -0.22, f"CRP={v['crp']:.2f}  Lift={v['lift']:.2f}",
                        transform=obs_ax.transAxes, ha="center", va="top", fontsize=9.5)
            for ax in (pred_ax, obs_ax):
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized longitude")
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=8.5)
            pred_ax.set_ylabel("Normalized latitude")
        for flat in range(n * 2, nrows * ncols):
            r, c = flat // ncols, flat % ncols
            axes[r, c].axis("off")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                       markeredgecolor="#666666", markersize=6.5, label="Initial 30 cells"),
            plt.Line2D([], [], marker="D", linestyle="none", markerfacecolor="#F0E442",
                       markeredgecolor="black", markersize=6.5, label="Acquired 20 cells"),
            plt.Line2D([], [], color="#D55E00", linewidth=1.7, label="Recommended regions R1–R4"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=3, frameon=False, fontsize=11)
        fig.colorbar(image, ax=axes, location="right", shrink=0.85, pad=0.012,
                     label="Incidents/km$^2$")
        for suffix in ("png", "pdf"):
            path = OUTPUT / f"fig5_suppmat_12methods.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white", transparent=False)
            print("saved", path)


if __name__ == "__main__":
    main()
