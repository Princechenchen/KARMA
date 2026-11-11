"""Evaluation-only Oracle-GSO upper reference; never an MMRGP experiment result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_metrics import region_supplements
from mmrgp_experiment import _json_safe
from region_recommender import RecommendationConfig, evaluate_recommendations, recommend_regions
from smoke_lhs_oracle import OracleGridIntensity, load_domain, load_incidents

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158, 180, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    output = args.output or ROOT / "results" / "oracle" / args.grid
    output.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data" / "units" / f"units_2022_{args.grid}.csv")
    construct, domain_width = load_domain()
    incidents, incidents_xy = load_incidents(construct)
    field = OracleGridIntensity(units)
    rows = []
    for seed in seeds:
        recommendation = recommend_regions(field, domain_width, RecommendationConfig(), seed=seed)
        if not recommendation["success"]:
            raise RuntimeError(f"Oracle GSO failed for seed {seed}")
        metrics = evaluate_recommendations(recommendation["boxes"], incidents_xy, domain_width)
        supplements = region_supplements(
            recommendation["boxes"], units, float(units["y2"].mean())
        )
        rows.append({
            "seed": seed, "scope": "evaluation_only_oracle_not_mmr_gp",
            "recommendation": {
                "boxes": recommendation["boxes"].tolist(),
                "scores": recommendation["scores"].tolist(),
                "seconds": recommendation["seconds"],
            },
            **metrics, **supplements,
        })
    payload = {
        "status": "COMPLETE", "scope": "Oracle full-target-grid upper reference only",
        "grid": args.grid, "seeds": seeds, "results": rows,
    }
    (output / "oracle_history.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": "COMPLETE", "grid": args.grid, "n_seeds": len(seeds),
        "region_pass_mean": float(np.mean([row["region_pass"] for row in rows])),
        "density_lift_mean": float(np.mean([row["density_lift"] for row in rows])),
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
