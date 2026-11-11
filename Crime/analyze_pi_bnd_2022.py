"""Evaluate two boundary-prior forms with 2022 data and build Oracle diagnostics.

The 2021 fields are evaluated externally on common occupied cells. The 2022
fields are target-year Oracle constructs for mechanism tests, not deployable
priors for the formal cross-year comparison. Raw input files are read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "pi_bnd_2022_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def ranking_and_correlation_metrics(
    frame: pd.DataFrame, score_column: str, mask: np.ndarray
) -> dict:
    evaluated = frame.loc[mask]
    actual = evaluated["is_classification_boundary"].to_numpy(dtype=int)
    score = evaluated[score_column].to_numpy(dtype=float)
    uncertainty = evaluated["uncertainty_2022"].to_numpy(dtype=float)
    assert np.isfinite(score).all() and 0 < actual.sum() < len(actual)
    return {
        "cells": int(len(evaluated)),
        "boundary_cells": int(actual.sum()),
        "knowledge_mean": float(score.mean()),
        "knowledge_std": float(score.std(ddof=1)),
        "knowledge_median": float(np.median(score)),
        "boundary_cell_knowledge_mean": float(score[actual == 1].mean()),
        "nonboundary_cell_knowledge_mean": float(score[actual == 0].mean()),
        "boundary_roc_auc": float(roc_auc_score(actual, score)),
        "boundary_average_precision": float(average_precision_score(actual, score)),
        "uncertainty_pearson_r": float(pearsonr(score, uncertainty).statistic),
        "uncertainty_spearman_rho": float(spearmanr(score, uncertainty).statistic),
    }


def main() -> None:
    source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km_adjacent.csv")
    for frame in (source, target, knowledge):
        assert frame["cell_id"].is_unique
    assert not target[["violent_ratio", "y2", "is_classification_boundary"]].isna().any().any()

    threshold = float(source["violent_ratio"].median())
    mad = float(np.median(np.abs(source["violent_ratio"] - threshold)))
    tau = max(1.4826 * mad, 0.05)
    assert np.isclose(threshold, 1.0 / 3.0)
    assert np.allclose(target["y2_threshold"], threshold)

    fields = target[[
        "cell_id", "ix", "iy", "cell_lat", "cell_lon", "x", "y",
        "violent_ratio", "y2", "is_classification_boundary",
    ]].copy()
    fields["uncertainty_2022"] = 4 * fields["violent_ratio"] * (1 - fields["violent_ratio"])
    fields = fields.merge(
        knowledge[["cell_id", "pi_bnd", "pi_bnd_adjacent"]],
        on="cell_id", how="left", validate="one_to_one",
    )
    fields = fields.rename(columns={
        "pi_bnd": "threshold_2021",
        "pi_bnd_adjacent": "adjacent_2021",
    })

    ratio = fields["violent_ratio"].to_numpy(dtype=float)
    fields["threshold_2022_oracle"] = np.clip(
        np.exp(-((ratio - threshold) ** 2) / (2 * tau**2)), 0.01, 1.0
    )
    adjacent_oracle = build_adjacent_boundary_knowledge(target, sigma_cells=1.0)
    assert np.array_equal(
        adjacent_oracle["is_classification_boundary_2021"].to_numpy(dtype=int),
        target["is_classification_boundary"].to_numpy(dtype=int),
    )
    fields["adjacent_2022_oracle"] = adjacent_oracle["pi_bnd_adjacent"].to_numpy(dtype=float)

    field_columns = [
        "threshold_2021", "adjacent_2021",
        "threshold_2022_oracle", "adjacent_2022_oracle",
    ]
    metrics_rows = []
    for column in field_columns:
        mask = fields[column].notna().to_numpy()
        row = {"field": column}
        row.update(ranking_and_correlation_metrics(fields, column, mask))
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    common = fields["threshold_2021"].notna() & fields["adjacent_2021"].notna()
    assert int(common.sum()) == 649
    controlled = {}
    common_rows = []
    for column in field_columns:
        controlled[column] = ranking_and_correlation_metrics(
            fields, column, common.to_numpy()
        )
        common_rows.append({"field": column, **controlled[column]})
    common_metrics = pd.DataFrame(common_rows)
    for column in ("threshold_2021", "adjacent_2021"):
        assert controlled[column] == metrics.loc[metrics["field"] == column].iloc[0].drop("field").to_dict()

    cell_path = OUT / "cell_level_fields.csv"
    metrics_path = OUT / "metrics.csv"
    common_metrics_path = OUT / "metrics_common_649.csv"
    fields.to_csv(cell_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    common_metrics.to_csv(common_metrics_path, index=False)

    result = {
        "scope": "Exploratory spatial association on occupied 1 km cells; no causal or confirmatory claim.",
        "external_evaluation": "2021 fields versus observed 2022 boundary and empirical ratio uncertainty.",
        "oracle_warning": "2022 fields use target-year responses and are only for mechanism/upper-bound experiments; they leak target information if presented as deployable priors.",
        "construction": {
            "classification_threshold_from_2021": threshold,
            "tau_from_2021": tau,
            "threshold_form": "clip(exp(-(violent_ratio-threshold)^2/(2*tau^2)),0.01,1)",
            "adjacent_form": "exp(-distance_to_nearest_four_neighbor_y2_boundary^2/(2*1^2))",
            "uncertainty_diagnostic": "4*r_2022*(1-r_2022), where r_2022 is the observed violent ratio, not an MMRGP predictive probability",
        },
        "support": {
            "cells_2021": int(len(source)),
            "cells_2022": int(len(target)),
            "common_cells": int(common.sum()),
            "target_only_cells_excluded_for_2021_fields": int((~common).sum()),
            "boundary_cells_all_2022": int(target["is_classification_boundary"].sum()),
            "boundary_cells_common": int(fields.loc[common, "is_classification_boundary"].sum()),
            "boundary_prevalence_all_2022": float(target["is_classification_boundary"].mean()),
            "boundary_prevalence_common": float(fields.loc[common, "is_classification_boundary"].mean()),
        },
        "metrics": metrics.to_dict(orient="records"),
        "controlled_common_649": controlled,
    }
    json_path = OUT / "metrics.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {
        "threshold_2021": "A  2021 threshold proximity (external)",
        "adjacent_2021": "B  2021 adjacent boundary (external)",
        "threshold_2022_oracle": "C  2022 threshold proximity (Oracle)",
        "adjacent_2022_oracle": "D  2022 adjacent boundary (Oracle)",
    }
    metric_lookup = metrics.set_index("field")
    boundary = fields[fields["is_classification_boundary"] == 1]

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
    }):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.10, hspace=0.34, wspace=0.24)

        def overlay_boundary(ax):
            ax.scatter(
                boundary["ix"], boundary["iy"], marker="s", s=50,
                facecolors="none", edgecolors="white", linewidths=1.25,
            )
            ax.scatter(
                boundary["ix"], boundary["iy"], marker="s", s=46,
                facecolors="none", edgecolors="black", linewidths=0.45,
            )

        def finish_panel(ax, artist, title, colorbar_label):
            ax.set_title(title)
            ax.set_xlabel("Grid x index (1 km cells)")
            ax.set_ylabel("Grid y index (1 km cells)")
            ax.set_aspect("equal")
            ax.set_xlim(fields["ix"].min() - 0.8, fields["ix"].max() + 0.8)
            ax.set_ylim(fields["iy"].min() - 0.8, fields["iy"].max() + 0.8)
            colorbar = fig.colorbar(artist, ax=ax, shrink=0.76)
            colorbar.set_label(colorbar_label)
            overlay_boundary(ax)

        positions = [(0, 0), (0, 1), (0, 2), (1, 0)]
        for column, (row, col) in zip(field_columns, positions):
            ax = axes[row, col]
            valid = fields[column].notna()
            artist = ax.scatter(
                fields.loc[valid, "ix"], fields.loc[valid, "iy"],
                c=fields.loc[valid, column], marker="s", s=32,
                cmap="cividis", vmin=0, vmax=1, linewidths=0,
            )
            stats = metric_lookup.loc[column]
            title = (
                f"{labels[column]}\n"
                f"Boundary AUC={stats['boundary_roc_auc']:.3f}; "
                f"r with 2022 uncertainty={stats['uncertainty_pearson_r']:.3f}"
            )
            finish_panel(ax, artist, title, r"$\pi_{bnd}$ intensity")

        ax = axes[1, 1]
        artist = ax.scatter(
            fields["ix"], fields["iy"], c=fields["uncertainty_2022"],
            marker="s", s=32, cmap="magma", vmin=0, vmax=1, linewidths=0,
        )
        finish_panel(
            ax, artist,
            "E  Observed 2022 ratio uncertainty\n$4r(1-r)$; open squares = true class boundary",
            "Uncertainty diagnostic",
        )

        ax = axes[1, 2]
        categorical = ListedColormap(["#4575b4", "#d73027"])
        artist = ax.scatter(
            fields["ix"], fields["iy"], c=fields["y2"], marker="s", s=32,
            cmap=categorical, vmin=0, vmax=1, linewidths=0,
        )
        finish_panel(
            ax, artist,
            "F  Observed 2022 $y_2$ and true boundary\nFixed threshold = 1/3; four-neighbor rule",
            "$y_2$: 0 = non-high, 1 = high",
        )
        fig.suptitle("Two forms of boundary knowledge evaluated with 2022 data", fontsize=16, y=0.975)
        fig.text(
            0.5, 0.925,
            "2021 fields = external cross-year validation  |  2022 fields = target-year Oracle, not deployable formal priors",
            ha="center", fontsize=10,
        )
        fig.text(
            0.5, 0.025,
            "Open black squares mark the observed 2022 class boundary. Empty locations are unoccupied or unavailable 2021 support; no zero imputation.",
            ha="center", fontsize=9,
        )
        figure_path = OUT / "pi_bnd_two_forms_2022_analysis.png"
        fig.savefig(figure_path, dpi=190, facecolor="white")
        plt.close(fig)

    provenance = {
        "figure": str(figure_path.relative_to(ROOT)),
        "analysis_script": "analyze_pi_bnd_2022.py",
        "source_files": [
            "data/units/units_2021_1km.csv",
            "data/units/units_2022_1km.csv",
            "data/knowledge/knowledge_2021_1km_adjacent.csv",
        ],
        "outputs": [str(path.relative_to(ROOT)) for path in (cell_path, metrics_path, common_metrics_path, json_path, figure_path)],
        "missing_data": "Ten 2022-only cells are excluded for 2021-field metrics and the controlled comparison; all 659 cells are used for the main 2022 Oracle metrics.",
        "software": {
            "python_numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        },
        "figure_size_inches": [15, 9.5], "dpi": 190,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MAIN METRICS")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("CONTROLLED COMMON-CELL METRICS")
    print(common_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"METRICS={json_path}")
    print(f"FIGURE={figure_path}")
    print(f"SIZE_BYTES={figure_path.stat().st_size}")


if __name__ == "__main__":
    main()
