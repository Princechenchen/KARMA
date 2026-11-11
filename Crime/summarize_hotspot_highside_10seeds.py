"""Summarize the three requested methods over the same ten seeds."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/aggregated/hotspot_highside_10seeds_comparison"
CONFIGS = [
    ("ROI-D", "ROI-D", "无先验", "hotspot", "1km_budget50_random_holdout_hotspot"),
    ("KARMA-HG", "KARMA-HG", "2021强度梯度（无边界）", "hotspot", "1km_budget50_random_holdout_hotspot"),
    ("KARMA-2022-HighSide-Oracle", "KARMA", "2021强度梯度＋2022高发侧邻接Oracle", "hotspot_oracle2022_highside_adjacent", "1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"),
]


def main():
    rows = []
    for label, method, knowledge, stem, directory in CONFIGS:
        models = pd.read_csv(ROOT / "results/aggregated" / f"{stem}_10seeds/summary_budget_50.csv")
        model = models[(models["method"] == method) & (models["n_seeds"] == 10)].iloc[0]
        regions = pd.read_csv(ROOT / "results/valid_region_metrics" / directory / "summary_set_metrics.csv")
        region = regions[(regions["method"] == method) & (regions["n_seeds"] == 10)].iloc[0]
        rows.append({
            "method": label, "knowledge": knowledge, "n_seeds": 10,
            "rmae": model["rmae_mean"], "rmae_std": model["rmae_std"],
            "acc": model["acc_mean"], "acc_std": model["acc_std"],
            "balanced_acc": model["balanced_acc_mean"], "balanced_acc_std": model["balanced_acc_std"],
            "rp": region["region_pass_mean"], "rp_std": region["region_pass_std"],
            "crp": region["conditional_region_pass_mean"], "crp_std": region["conditional_region_pass_std"],
            "density_lift": region["density_lift_mean"], "density_lift_std": region["density_lift_std"],
            "topn": region["topn_grid_precision_mean"], "topn_std": region["topn_grid_precision_std"],
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "summary.csv", index=False)
    print(frame[["method", "rmae", "acc", "balanced_acc", "rp", "crp", "density_lift", "topn"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
