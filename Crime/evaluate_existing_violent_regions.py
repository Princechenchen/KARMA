"""Evaluate existing recommended boxes against violent incidents only.

This is an offline post-hoc evaluation. It does not alter the fitted models,
selected samples, recommended boxes, or the total-crime recommendation objective.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acquisition import METHODS
from region_recommender import evaluate_recommendations
from smoke_lhs_oracle import load_domain, load_incidents

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
INPUT = ROOT / "results" / "formal" / "1km_budget50_random_holdout"
OUTPUT = ROOT / "results" / "aggregated" / "1km_budget50_random_holdout_13methods_8seeds"


def main():
    config, domain_width = load_domain()
    incidents, incidents_xy = load_incidents(config)
    violent_xy = incidents_xy[incidents["is_violent"].to_numpy(bool)]
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            path = INPUT / method / "fold_0" / f"seed_{seed}" / "history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            assert history["status"] == "COMPLETE"
            evaluations = [row for row in history["evaluations"] if row["budget"] == 50]
            assert len(evaluations) == 1
            evaluation = evaluations[0]
            boxes = np.asarray(evaluation["recommendation"]["boxes"], dtype=float)
            total = evaluate_recommendations(boxes, incidents_xy, domain_width, 2000, 20261024)
            assert np.isclose(total["region_pass"], evaluation["region_pass"], atol=1e-12)
            assert np.isclose(total["density_lift"], evaluation["density_lift"], atol=1e-12)
            violent = evaluate_recommendations(boxes, violent_xy, domain_width, 2000, 20261024)
            rows.append({
                "method": method,
                "seed": seed,
                "budget": 50,
                "region_pass": evaluation["region_pass"],
                "density_lift": evaluation["density_lift"],
                "violent_region_pass": violent["region_pass"],
                "violent_density_lift": violent["density_lift"],
                "violent_incident_share": evaluation["violent_incident_share"],
                "violent_incident_uplift": evaluation["violent_incident_uplift"],
                "violent_label_share": evaluation["violent_label_share"],
                "violent_label_uplift": evaluation["violent_label_uplift"],
                "violent_counts": json.dumps(violent["counts"]),
                "violent_reference_q3": json.dumps(violent["reference_q3"]),
                "violent_pass_flags": json.dumps(violent["pass_flags"]),
                "n_reference": violent["n_reference"],
                "reference_seed": violent["reference_seed"],
            })
    frame = pd.DataFrame(rows)
    assert len(frame) == 104
    assert np.isfinite(frame[["violent_region_pass", "violent_density_lift"]]).all().all()
    metrics = [
        "region_pass", "density_lift", "violent_region_pass", "violent_density_lift",
        "violent_incident_share", "violent_incident_uplift",
        "violent_label_share", "violent_label_uplift",
    ]
    grouped = frame.groupby("method")[metrics]
    summary = pd.concat([
        grouped.mean().add_suffix("_mean"),
        grouped.std(ddof=0).add_suffix("_std"),
    ], axis=1)
    summary.insert(0, "n_seeds", grouped.size())
    summary.insert(0, "budget", 50)
    summary = summary.reset_index()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "per_run_violent_region_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "summary_violent_region_budget_50.csv", index=False)
    print(summary[[
        "method", "n_seeds", "violent_region_pass_mean", "violent_region_pass_std",
        "violent_density_lift_mean", "violent_density_lift_std",
    ]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
