"""Validate production GSO/Oracle recommendation across ten fixed seeds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from region_recommender import RecommendationConfig, evaluate_recommendations, recommend_regions
from smoke_lhs_oracle import OracleGridIntensity, load_domain, load_incidents

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158, 180, 1]


def main():
    config, domain_width = load_domain()
    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    _, incidents_xy = load_incidents(config)
    field = OracleGridIntensity(units)
    runs = []
    for seed in SEEDS:
        recommendation = recommend_regions(field, domain_width, RecommendationConfig(), seed=seed)
        metrics = evaluate_recommendations(
            recommendation["boxes"], incidents_xy, domain_width,
            n_reference=2000, seed=20261024,
        )
        runs.append({
            "seed": seed,
            "success": recommendation["success"],
            "region_pass": metrics["region_pass"],
            "density_lift": metrics["density_lift"],
            "counts": metrics["counts"],
        })
    summary = {
        "scope": "Production GSO with corrected Oracle xy adapter; no historical-elite extraction",
        "seeds": SEEDS,
        "mean_region_pass": float(np.mean([run["region_pass"] for run in runs])),
        "mean_density_lift": float(np.mean([run["density_lift"] for run in runs])),
        "zero_count_regions": int(sum(np.sum(np.asarray(run["counts"]) == 0) for run in runs)),
        "all_returned_k": all(run["success"] for run in runs),
        "runs": runs,
    }
    destination = ROOT / "results" / "smoke_lhs_oracle" / "oracle_seed_stability.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
