"""Plot matched KARMA/ROI-D spatial diagnostics for formal seed 172."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from mmrgp_experiment import MATLABExperimentSession
from recommend_violent_regions import _final_evaluation
from region_recommender import lognormal_intensity
from run_experiments import load_experiment_config

ROOT = Path(__file__).resolve().parent
SEED = 172
METHODS = ["KARMA", "ROI-D"]


def predict_method(session, method, units, cell_area):
    history_path = ROOT / "results" / "formal" / "1km_budget50_random_holdout" / method / "fold_0" / f"seed_{SEED}" / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train = units.iloc[list(map(int, history["training_indices"]))]
    run_id = str(history["run_id"])
    session.fit(
        train[["x", "y"]].to_numpy(float),
        train[["y1_log1p", "y2"]].to_numpy(float), run_id,
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
    frame["predicted_p_violent"] = prediction["p"]
    frame["training_order"] = -1
    frame.loc[train.index, "training_order"] = np.arange(len(train))
    evaluation = _final_evaluation(history)
    return frame, evaluation


def draw_method(method, frame, evaluation, density_norm, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle(f"{method} | formal seed {SEED} | final 50 samples", fontsize=16)
    common = dict(s=48, marker="s", linewidths=0, edgecolors="none")
    top_images = []
    for ax, column, title in [
        (axes[0, 0], "observed_density", "Observed 2022 total-crime density"),
        (axes[0, 1], "predicted_density", "MMRGP predicted total-crime density"),
    ]:
        image = ax.scatter(
            frame["x"], frame["y"], c=frame[column], cmap="viridis",
            norm=density_norm, **common,
        )
        top_images.append(image)
        ax.set_title(title)
    probability_image = None
    for ax, column, title, cmap, norm in [
        (axes[1, 0], "y2", "Observed high-violence label $y_2$", ListedColormap(["#31688E", "#D1495B"]), Normalize(0, 1)),
        (axes[1, 1], "predicted_p_violent", "Predicted violent probability", "cividis", Normalize(0, 1)),
    ]:
        image = ax.scatter(
            frame["x"], frame["y"], c=frame[column], cmap=cmap, norm=norm, **common,
        )
        ax.set_title(title)
        if column == "y2":
            fig.colorbar(image, ax=ax, ticks=[0, 1], label="0 = non-high, 1 = high")
        else:
            probability_image = image

    training = frame[frame["training_order"] >= 0]
    initial = training[training["training_order"] < 30]
    acquired = training[training["training_order"] >= 30]
    for ax in [axes[0, 1], axes[1, 1]]:
        ax.scatter(
            initial["x"], initial["y"], s=34, facecolors="none", edgecolors="white",
            linewidths=0.8, marker="o", label="Initial 30",
        )
        ax.scatter(
            acquired["x"], acquired["y"], s=46, c="#FF9F1C", edgecolors="black",
            linewidths=0.4, marker="D", label="Acquired 20",
        )
    recommendation = evaluation["recommendation"]
    for cx, cy, lx, ly in np.asarray(recommendation["boxes"], dtype=float):
        axes[0, 1].add_patch(Rectangle(
            (cx - lx / 2, cy - ly / 2), lx, ly,
            fill=False, edgecolor="#E63946", linewidth=1.8,
        ))
    axes[0, 1].scatter([], [], s=34, facecolors="none", edgecolors="white", marker="o", label="Initial 30")
    axes[0, 1].scatter([], [], s=46, c="#FF9F1C", edgecolors="black", marker="D", label="Acquired 20")
    axes[0, 1].plot([], [], color="#E63946", linewidth=1.8, label="Old-target GSO boxes")
    axes[0, 1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[0, 1].set_title(
        f"Predicted density + samples + regions\n"
        f"RegionPass={evaluation['region_pass']:.2f}, DensityLift={evaluation['density_lift']:.2f}"
    )
    for ax in axes.flat:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized longitude", ylabel="Normalized latitude")
        ax.set_aspect("equal", adjustable="box")
    cap_label = f"incidents/km$^2$ (color capped at pooled 99th percentile = {density_norm.vmax:.1f})"
    fig.colorbar(top_images[0], ax=axes[0, :], label=cap_label)
    fig.colorbar(probability_image, ax=axes[1, 1], label="P(high violence)")
    path = output / f"{method}_seed{SEED}_spatial.png"
    fig.savefig(path, dpi=200, facecolor="white", transparent=False)
    plt.close(fig)
    return path


def main():
    output = ROOT / "results" / "figures" / "karma_roi_spatial"
    output.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    settings = load_experiment_config("1km")
    cell_area = float(settings["grids"]["1km"]["cell_area_km2"])
    frames, evaluations = {}, {}
    with MATLABExperimentSession(
        gpml_root=str(ROOT.parent / "third_party" / "GPML"),
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
        cell_area_km2=cell_area,
        chunk_size=8192,
    ) as session:
        for method in METHODS:
            frames[method], evaluations[method] = predict_method(session, method, units, cell_area)
    all_density = np.concatenate([
        frame[["observed_density", "predicted_density"]].to_numpy(float).reshape(-1)
        for frame in frames.values()
    ])
    cap = float(np.quantile(all_density, 0.99))
    density_norm = Normalize(vmin=0, vmax=cap)
    paths = []
    for method in METHODS:
        frames[method].to_csv(output / f"{method}_seed{SEED}_predictions.csv", index=False)
        paths.append(draw_method(method, frames[method], evaluations[method], density_norm, output))
    provenance = {
        "seed": SEED, "methods": METHODS, "final_training_size": 50,
        "initial_samples": 30, "acquired_samples": 20,
        "grid": "1km", "cell_area_km2": cell_area,
        "density_color_cap_pooled_99th_percentile": cap,
        "density_values_above_cap_are_not_removed": True,
        "observed_density": "expm1(y1_log1p) / cell_area_km2",
        "predicted_density": "(exp(mu_z + var_z/2) - 1) / cell_area_km2",
        "recommendation_target": "original total-intensity GSO",
        "figures": [str(path) for path in paths],
        "note": "One predeclared formal seed for visual diagnosis; not an eight-seed average.",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
