"""Fig5 论文版重绘 (2026-11-08, 用户指令):
① ROI-D 子图替换为 piBO-global(C6)——用其正式 hotspot 轮的 history/valid_recommendation;
② 子图标题去编号(C7: πBO-ROI -> πBO-ROI), 方法名与 Table V 主文一致;
③ 子图下指标只保留 CRP 与 Lift;
④ 字号统一放大, 子图排布紧凑(缩小画幅+收紧间距)。
只读既有结果, 不改任何正式数据; 输出新文件名 fig5_paper_*.png 不覆盖原图。"""
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
OUTPUT = ROOT / "results" / "figures" / "fig5_paper"

METHOD_SPECS = [
    {"key": "KARMA", "source_method": "KARMA", "label": "KARMA", "variant": ORACLE_VARIANT},
    {"key": "C1", "source_method": "C1", "label": "Maximin-LHS", "variant": REGULAR_VARIANT},
    {"key": "C2", "source_method": "C2", "label": "MeanMSE", "variant": REGULAR_VARIANT},
    {"key": "C10", "source_method": "C10", "label": "Voronoi", "variant": REGULAR_VARIANT},
    {"key": "C6", "source_method": "C6", "label": "piBO-global", "variant": REGULAR_VARIANT},
    {"key": "C7", "source_method": "C7", "label": "piBO-ROI", "variant": REGULAR_VARIANT},
]
POSITIONS = {"KARMA": (0, 0), "C1": (0, 2), "C2": (1, 0), "C10": (1, 2), "C6": (2, 0), "C7": (2, 2)}


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
    return {key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")}


def predict_method(session, spec, units, cell_area):
    paths = run_paths(spec, SEED)
    history = read_json(paths["history"])
    train_indices = np.asarray(history["training_indices"], dtype=int)
    train = units.iloc[train_indices]
    run_id = f"fig5_{spec['key']}_{SEED}"
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
    metrics = {
        "crp": float(set_metrics["conditional_region_pass"]),
        "lift": float(recommendation["density_lift"]),
    }
    return frame, recommendation, metrics


def add_regions(ax, boxes):
    for index, (cx, cy, lx, ly) in enumerate(np.asarray(boxes, dtype=float), 1):
        ax.add_patch(Rectangle((cx - lx / 2, cy - ly / 2), lx, ly,
                               fill=False, edgecolor="#D55E00", linewidth=1.8, zorder=5))
        ax.text(cx, cy, f"R{index}", color="#D55E00", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.0})


def draw_spatial(frames, recommendations, metrics, density_norm):
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(3, 4, figsize=(16, 12.6), layout="constrained")
        fig.get_layout_engine().set(rect=[0, 0.055, 1.0, 0.945], hspace=0.02, wspace=0.04)
        image = None
        for spec in METHOD_SPECS:
            key = spec["key"]
            row, col = POSITIONS[key]
            pred_ax, obs_ax = axes[row, col], axes[row, col + 1]
            frame = frames[key]
            common = dict(s=37, marker="s", linewidths=0, edgecolors="none",
                          cmap="viridis", norm=density_norm, rasterized=True)
            image = pred_ax.scatter(frame["x"], frame["y"], c=frame["predicted_density"], **common)
            obs_ax.scatter(frame["x"], frame["y"], c=frame["observed_density"], **common)

            training = frame[frame["training_order"] >= 0].sort_values("training_order")
            initial = training[training["training_order"] < INITIAL_BUDGET]
            acquired = training[training["training_order"] >= INITIAL_BUDGET]
            pred_ax.scatter(initial["x"], initial["y"], s=30, facecolors="none",
                            edgecolors="white", linewidths=0.75, marker="o", zorder=3)
            pred_ax.scatter(acquired["x"], acquired["y"], s=39, c="#F0E442",
                            edgecolors="black", linewidths=0.45, marker="D", zorder=4)
            add_regions(pred_ax, recommendations[key]["boxes"])
            add_regions(obs_ax, recommendations[key]["boxes"])

            pred_ax.set_title(spec["label"], fontweight="bold", pad=5)
            obs_ax.set_title(f"{spec['label']} (observed)", fontweight="bold", pad=5)
            for ax in (pred_ax, obs_ax):
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized longitude")
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=13)
            pred_ax.set_ylabel("Normalized latitude")

        handles = [
            plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                       markeredgecolor="#666666", markersize=7, label="Initial 30 cells"),
            plt.Line2D([], [], marker="D", linestyle="none", markerfacecolor="#F0E442",
                       markeredgecolor="black", markersize=7, label="Acquired 20 cells"),
            plt.Line2D([], [], color="#D55E00", linewidth=1.8, label="Recommended regions R1–R4"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=3, frameon=False, fontsize=15)
        fig.colorbar(image, ax=axes, location="right", shrink=0.85, pad=0.012,
                     label="Incidents/km$^2$")
        paths = []
        for suffix in ("png", "pdf"):
            path = OUTPUT / f"fig5_paper_seed{SEED}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white", transparent=False)
            paths.append(path)
        plt.close(fig)
    return paths


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = load_units("1km", variant="hotspot")
    settings = load_experiment_config("1km")
    cell_area = float(settings["grids"]["1km"]["cell_area_km2"])

    for spec in METHOD_SPECS:
        for path in run_paths(spec, SEED).values():
            if not path.is_file():
                raise FileNotFoundError(path)
            read_json(path)

    frames, recommendations, metrics = {}, {}, {}
    for variant in (ORACLE_VARIANT, REGULAR_VARIANT):
        specs = [sp for sp in METHOD_SPECS if sp["variant"] == variant]
        with MATLABExperimentSession(
            gpml_root=str(ROOT.parent / "third_party" / "GPML"),
            knowledge=knowledge_arrays(variant),
            cell_area_km2=cell_area, chunk_size=8192,
        ) as session:
            for spec in specs:
                key = spec["key"]
                print(f"Refitting {key} ({variant})...", flush=True)
                frames[key], recommendations[key], metrics[key] = predict_method(session, spec, units, cell_area)
                print(" ", metrics[key], flush=True)

    all_density = np.concatenate([
        f[["observed_density", "predicted_density"]].to_numpy(float).reshape(-1)
        for f in frames.values()
    ])
    cap = float(np.quantile(all_density, 0.99))
    density_norm = Normalize(vmin=0, vmax=cap, clip=True)
    paths = draw_spatial(frames, recommendations, metrics, density_norm)
    for path in paths:
        print(path, flush=True)


if __name__ == "__main__":
    main()
