"""Merge the frozen 10-seed model and strict-valid-region summaries."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = [
    "C1", "C2", "C12", "C10", "ROI-D", "C6", "C7", "KARMA",
    "C3", "C4", "C5", "C9", "C11", "KARMA-HG",
]
OUTPUT = ROOT / "results/aggregated/hotspot_all_methods_10seeds/summary.csv"


def one_row(frame, method):
    selected = frame[(frame["method"] == method) & (frame["n_seeds"] == 10)]
    if "budget" in frame:
        selected = selected[selected["budget"] == 50]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one 10-seed row for {method}, found {len(selected)}")
    return selected.iloc[0]


def merged_row(model, region, display_name, role):
    return {
        "method": display_name,
        "role": role,
        "n_seeds": 10,
        "rmae_mean": model["rmae_mean"],
        "rmae_std": model["rmae_std"],
        "acc_mean": model["acc_mean"],
        "acc_std": model["acc_std"],
        "balanced_acc_mean": model["balanced_acc_mean"],
        "balanced_acc_std": model["balanced_acc_std"],
        "region_pass_mean": region["region_pass_mean"],
        "region_pass_std": region["region_pass_std"],
        "conditional_region_pass_mean": region["conditional_region_pass_mean"],
        "conditional_region_pass_std": region["conditional_region_pass_std"],
        "density_lift_mean": region["density_lift_mean"],
        "density_lift_std": region["density_lift_std"],
        "topn_grid_precision_mean": region["topn_grid_precision_mean"],
        "topn_grid_precision_std": region["topn_grid_precision_std"],
    }


def main():
    model = pd.read_csv(ROOT / "results/aggregated/hotspot_10seeds/summary_budget_50.csv")
    region = pd.read_csv(ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot/summary_set_metrics.csv")
    rows = [
        merged_row(one_row(model, method), one_row(region, method), method, "deployable")
        for method in METHODS
    ]
    oracle_model = pd.read_csv(ROOT / "results/aggregated/hotspot_oracle2022_highside_adjacent_10seeds/summary_budget_50.csv")
    oracle_region = pd.read_csv(ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent/summary_set_metrics.csv")
    rows.append(merged_row(
        one_row(oracle_model, "KARMA"), one_row(oracle_region, "KARMA"),
        "KARMA-2022-HighSide-Oracle", "oracle_diagnostic",
    ))
    result = pd.DataFrame(rows)
    if len(result) != 15 or result["method"].nunique() != 15:
        raise RuntimeError("Final summary must contain 15 unique methods")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, float_format="%.10f")
    print(result.to_string(index=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
