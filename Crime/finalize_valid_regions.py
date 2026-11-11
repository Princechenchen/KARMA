"""Consolidate and independently validate the strict-valid-grid region results."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from acquisition import METHODS
from recompute_valid_regions import SEEDS, _result_path, _save_aggregates
from run_experiments import atomic_write_json
from spatial_validity import GridBoxValidity

ROOT = Path(__file__).resolve().parent
NAME = "1km_budget50_random_holdout"
FORMAL = ROOT / "results" / "formal" / NAME
OUTPUT = ROOT / "results" / "valid_region_metrics" / NAME
PARTS = ROOT / "results" / "valid_region_metrics" / "parts"


def load_results():
    records = {}
    sources = [OUTPUT, *sorted(PARTS.glob("*"))]
    for source in sources:
        for path in source.glob("*/fold_0/seed_*/valid_recommendation.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "COMPLETE":
                continue
            key = (result["method"], int(result["seed"]))
            if key in records:
                raise RuntimeError(f"duplicate completed result: {key}")
            records[key] = result
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(records) != expected:
        raise RuntimeError(
            f"result coverage mismatch: missing={sorted(expected - set(records))}, "
            f"unexpected={sorted(set(records) - expected)}"
        )
    return records


def main():
    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    validity = GridBoxValidity(units)
    records = load_results()
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            result = records[(method, seed)]
            boxes = np.asarray(result["boxes"], dtype=float)
            if boxes.shape != (4, 4) or not np.isfinite(boxes).all():
                raise RuntimeError(f"invalid box array: {method} seed={seed}")
            if not validity(boxes).all():
                raise RuntimeError(f"box outside valid grid: {method} seed={seed}")
            if result["valid_recommended_boxes"] != 4:
                raise RuntimeError(f"incomplete recommendations: {method} seed={seed}")
            if result["valid_reference_boxes"] != 8000 or result["n_reference"] != 2000:
                raise RuntimeError(f"incomplete matched references: {method} seed={seed}")
            if result["budget"] != 50 or result["n_train"] != 50:
                raise RuntimeError(f"wrong training budget: {method} seed={seed}")
            if len(result["counts"]) != 4 or any(int(x) < 0 for x in result["counts"]):
                raise RuntimeError(f"invalid incident counts: {method} seed={seed}")
            destination = _result_path(OUTPUT, method, seed)
            atomic_write_json(destination, result)
            if destination.with_name("gso_state.npz").exists():
                raise RuntimeError(f"stale checkpoint: {method} seed={seed}")

            history = json.loads(
                (FORMAL / method / "fold_0" / f"seed_{seed}" / "history.json").read_text(
                    encoding="utf-8"
                )
            )
            original = history["evaluations"][-1]
            rows.append({
                "method": method,
                "seed": seed,
                "original_region_pass": float(original["region_pass"]),
                "valid_region_pass": float(result["region_pass"]),
                "delta_region_pass": float(result["region_pass"] - original["region_pass"]),
                "original_density_lift": float(original["density_lift"]),
                "valid_density_lift": float(result["density_lift"]),
                "delta_density_lift": float(result["density_lift"] - original["density_lift"]),
                "original_counts": json.dumps(original["counts"]),
                "valid_counts": json.dumps(result["counts"]),
                "original_boxes": json.dumps(original["recommendation"]["boxes"]),
                "valid_boxes": json.dumps(result["boxes"]),
            })

    paired = pd.DataFrame(rows)
    paired.to_csv(OUTPUT / "comparison_with_original.csv", index=False)
    metrics = [
        "original_region_pass", "valid_region_pass", "delta_region_pass",
        "original_density_lift", "valid_density_lift", "delta_density_lift",
    ]
    grouped = paired.groupby("method", sort=False)[metrics]
    summary = pd.concat(
        [grouped.mean().add_suffix("_mean"), grouped.std(ddof=0).add_suffix("_std")],
        axis=1,
    )
    summary.insert(0, "n_seeds", grouped.size())
    summary.reset_index().to_csv(OUTPUT / "summary_comparison.csv", index=False)
    _save_aggregates(OUTPUT, METHODS, SEEDS)
    report = {
        "status": "COMPLETE",
        "n_methods": len(METHODS),
        "n_seeds_per_method": len(SEEDS),
        "n_complete": len(records),
        "all_recommended_boxes_strictly_valid": True,
        "n_recommended_boxes_validated": 4 * len(records),
        "n_matched_references_per_run": 8000,
        "budget": 50,
        "old_results_overwritten": False,
    }
    atomic_write_json(OUTPUT / "validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    columns = [
        "method", "original_region_pass_mean", "valid_region_pass_mean",
        "original_density_lift_mean", "valid_density_lift_mean",
    ]
    print(summary.reset_index()[columns].to_string(index=False))


if __name__ == "__main__":
    main()
