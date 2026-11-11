"""Add Conditional RegionPass and TopNGridPrecision to saved valid-region runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acquisition import METHODS, parse_methods
from recompute_valid_regions import SEEDS, _parse_seeds
from region_set_metrics import conditional_region_metrics, topn_grid_precision
from run_experiments import atomic_write_json, load_experiment_config
from smoke_lhs_oracle import load_domain, load_incidents
from spatial_validity import GridBoxValidity

ROOT = Path(__file__).resolve().parent
NAME = "1km_budget50_random_holdout"
INPUT = ROOT / "results" / "valid_region_metrics" / NAME
METRICS = ["region_pass", "conditional_region_pass", "density_lift", "topn_grid_precision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    methods = parse_methods(args.methods)
    seeds = _parse_seeds(args.seeds)

    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    validity = GridBoxValidity(units)
    construct, domain_width = load_domain()
    _, incidents_xy = load_incidents(construct)
    settings = load_experiment_config("1km")
    cell_area = float(settings["grids"]["1km"]["cell_area_km2"])

    rows = []
    for method in methods:
        for seed in seeds:
            directory = INPUT / method / "fold_0" / f"seed_{seed}"
            result = json.loads((directory / "valid_recommendation.json").read_text(encoding="utf-8"))
            if result.get("status") != "COMPLETE":
                raise RuntimeError(f"incomplete valid-region result: {method} seed={seed}")
            boxes = np.asarray(result["boxes"], dtype=float)
            set_path = directory / "set_metrics.json"
            if set_path.exists():
                saved = json.loads(set_path.read_text(encoding="utf-8"))
                if saved.get("status") == "COMPLETE":
                    conditional = {
                        key: saved[key] for key in (
                            "conditional_region_pass", "conditional_reference_q3",
                            "conditional_pass_flags", "conditional_reference_boxes",
                        )
                    }
                    topn = {
                        key: saved[key] for key in (
                            "area_equivalent_n", "covered_grid_count",
                            "topn_hit_count", "topn_grid_precision",
                        )
                    }
                    rows.append({
                        "method": method, "seed": seed,
                        "region_pass": float(result["region_pass"]),
                        "density_lift": float(result["density_lift"]),
                        "counts": json.dumps(result["counts"]),
                        **conditional, **topn,
                    })
                    print(f"[SetMetrics] skip {method} seed={seed}", flush=True)
                    continue
            conditional = conditional_region_metrics(
                boxes, incidents_xy, domain_width,
                int(result["n_reference"]), int(result["reference_seed"]), validity,
            )
            topn = topn_grid_precision(boxes, units, domain_width, cell_area)
            row = {
                "method": method,
                "seed": seed,
                "region_pass": float(result["region_pass"]),
                "density_lift": float(result["density_lift"]),
                "counts": json.dumps(result["counts"]),
                **conditional,
                **topn,
            }
            atomic_write_json(directory / "set_metrics.json", {
                "status": "COMPLETE", "method": method, "seed": seed,
                "conditional_reference_seed": int(result["reference_seed"]),
                **conditional, **topn,
            })
            rows.append(row)
            print(
                f"[SetMetrics] {method} seed={seed} "
                f"RP={row['region_pass']:.3f} CRP={row['conditional_region_pass']:.3f} "
                f"TopN={row['topn_grid_precision']:.3f}", flush=True,
            )

    frame = pd.DataFrame(rows)
    order = {method: index for index, method in enumerate(METHODS)}
    frame = frame.sort_values(
        ["method", "seed"],
        key=lambda values: values.map(order) if values.name == "method" else values,
    )
    frame.to_csv(INPUT / "per_run_set_metrics.csv", index=False)
    grouped = frame.groupby("method", sort=False)[METRICS]
    summary = pd.concat([
        grouped.mean().add_suffix("_mean"),
        grouped.std(ddof=0).add_suffix("_std"),
    ], axis=1)
    summary.insert(0, "n_seeds", grouped.size())
    summary = summary.reset_index().sort_values(
        "method", key=lambda values: values.map(order),
    )
    summary.to_csv(INPUT / "summary_set_metrics.csv", index=False)
    print(summary[["method", *[f"{metric}_mean" for metric in METRICS]]].to_string(index=False))


if __name__ == "__main__":
    main()
