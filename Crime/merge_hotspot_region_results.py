"""Validate hotspot summaries and merge them with the original region metrics."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HOTSPOT = ROOT / "results" / "hotspot_metrics" / "1km_budget50_random_holdout"
OLD = ROOT / "results" / "aggregated" / "1km_budget50_random_holdout_13methods_8seeds" / "summary_budget_50.csv"

per_run = pd.read_csv(HOTSPOT / "per_run_metrics.csv")
hotspot = pd.read_csv(HOTSPOT / "summary_metrics.csv")
old = pd.read_csv(OLD)
assert len(per_run) == 104 and len(hotspot) == 13 and len(old) == 13
assert not per_run.duplicated(["method", "seed"]).any()
assert per_run.groupby("method").size().eq(8).all()
assert per_run["n_test"].eq(132).all() and per_run["n_hotspot"].eq(27).all()
assert per_run["budget"].eq(50).all()
assert np.isfinite(per_run[["ndcg_at_20", "h_rmae_at_20"]].to_numpy()).all()
assert per_run["ndcg_at_20"].between(0, 1).all()
assert per_run["h_rmae_at_20"].ge(0).all()
for _, row in hotspot.iterrows():
    group = per_run[per_run["method"] == row["method"]]
    for metric in ["ndcg_at_20", "h_rmae_at_20"]:
        np.testing.assert_allclose(row[f"{metric}_mean"], group[metric].mean(), rtol=1e-12)
        np.testing.assert_allclose(row[f"{metric}_std"], group[metric].std(ddof=0), rtol=1e-12)

region_columns = [
    "method", "region_pass_mean", "region_pass_std",
    "density_lift_mean", "density_lift_std",
]
combined = hotspot.merge(old[region_columns], on="method", validate="one_to_one")
combined.to_csv(HOTSPOT / "summary_with_old_region_metrics.csv", index=False)
for _, row in combined.iterrows():
    print(
        f"{row['method']}: "
        f"{row['ndcg_at_20_mean']:.3f} ± {row['ndcg_at_20_std']:.3f}; "
        f"{row['h_rmae_at_20_mean']:.3f} ± {row['h_rmae_at_20_std']:.3f}; "
        f"{row['region_pass_mean']:.3f} ± {row['region_pass_std']:.3f}; "
        f"{row['density_lift_mean']:.3f} ± {row['density_lift_std']:.3f}"
    )
print("PASS: 104 hotspot rows and 13 merged summaries verified")
