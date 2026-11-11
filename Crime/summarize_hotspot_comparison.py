"""Summarize the three-seed hotspot run and same-seed original-region controls."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ["KARMA", "ROI-D", "C9", "C10"]
SEEDS = [172, 136, 34]
REGION_METRICS = ["region_pass", "conditional_region_pass", "density_lift", "topn_grid_precision"]


def region_means(path):
    frame = pd.read_csv(path)
    frame = frame[frame["method"].isin(METHODS) & frame["seed"].isin(SEEDS)]
    assert len(frame) == 12
    return frame.groupby("method")[REGION_METRICS].mean()


def main():
    original = region_means(ROOT / "results" / "valid_region_metrics" / "1km_budget50_random_holdout" / "per_run_set_metrics.csv")
    hotspot = region_means(ROOT / "results" / "valid_region_metrics" / "1km_budget50_random_holdout_hotspot" / "per_run_set_metrics.csv")
    comparison = hotspot.join(original, lsuffix="_hotspot", rsuffix="_original")
    for metric in REGION_METRICS:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_hotspot"] - comparison[f"{metric}_original"]

    model_path = ROOT / "results" / "aggregated" / "1km_budget50_random_holdout_hotspot_3seeds" / "summary_budget_50.csv"
    model = pd.read_csv(model_path).set_index("method")
    comparison = comparison.join(model[["rmae_mean", "acc_mean", "balanced_acc_mean", "f1_mean"]])
    comparison = comparison.reindex(METHODS).rename_axis("method").reset_index()
    output = ROOT / "results" / "aggregated" / "hotspot_vs_original_3seeds.csv"
    comparison.to_csv(output, index=False)
    columns = [
        "method", "rmae_mean", "acc_mean", "balanced_acc_mean", "f1_mean",
        *[f"{metric}_hotspot" for metric in REGION_METRICS],
        *[f"{metric}_original" for metric in REGION_METRICS],
    ]
    print(comparison[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
