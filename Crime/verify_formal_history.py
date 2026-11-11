"""Independently validate one completed formal Crimes history."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / "results/formal/1km/KARMA/fold_0/seed_172/history.json"


def close(actual, expected, tolerance=1e-12):
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def verify(history_path, units_path):
    history_path = Path(history_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    units = pd.read_csv(units_path)
    checks = {}

    checks["complete"] = history.get("status") == "COMPLETE"
    checks["identity"] = (
        history.get("method") in {"C10", "KARMA"}
        and (history.get("fold"), history.get("seed"), history.get("grid")) == (0, 172, "1km")
    )
    initial = int(history["initial_budget"])
    final_budget = int(history["total_budget"])
    steps = int(history["sequential_steps"])
    checks["frozen_b40_protocol"] = (initial, steps, final_budget) == (20, 20, 40)

    evaluations = history["evaluations"]
    budgets = [int(row["budget"]) for row in evaluations]
    checks["all_budgets_20_through_40"] = budgets == list(range(20, 41))
    train_indices = list(map(int, history["training_indices"]))
    train_ids = list(history["training_cell_ids"])
    checks["forty_unique_training_cells"] = (
        len(train_indices) == 40 and len(set(train_indices)) == 40
        and len(train_ids) == 40 and len(set(train_ids)) == 40
    )
    checks["training_indices_match_ids"] = (
        units.iloc[train_indices]["cell_id"].tolist() == train_ids
    )
    checks["held_out_fold_never_sampled"] = bool(
        np.all(units.iloc[train_indices]["spatial_block"].to_numpy(int) != 0)
    )

    acquisitions = history["acquisitions"]
    checks["twenty_sequential_acquisitions"] = (
        len(acquisitions) == 20
        and [int(row["from_budget"]) for row in acquisitions] == list(range(20, 40))
        and [int(row["step"]) for row in acquisitions] == list(range(1, 21))
        and [int(row["selected_index"]) for row in acquisitions] == train_indices[20:]
    )
    if history["method"] == "KARMA":
        checks["karma_has_twenty_roi_steps"] = (
            len(acquisitions[-1]["diagnostics"].get("roi_selection_hist", [])) == 20
        )
    else:
        checks["c10_uses_exact_loo_at_every_step"] = all(
            row["diagnostics"].get("C10", {}).get("loo_exact") is True
            and row["diagnostics"]["C10"].get("loo_cap") is None
            for row in acquisitions
        )

    by_budget = {int(row["budget"]): row for row in evaluations}
    checks["recommendations_at_b20_and_b40"] = all(
        budget in by_budget and "recommendation" in by_budget[budget] for budget in (20, 40)
    )
    final = by_budget[40]
    recommendation = final["recommendation"]
    config = recommendation["config"]
    checks["four_recommendation_boxes"] = (
        len(recommendation["boxes"]) == 4 and len(recommendation["scores"]) == 4
        and all(len(box) == 4 for box in recommendation["boxes"])
    )
    checks["gso_100_particles_100_iterations"] = (
        int(config["particles"]) == 100 and int(config["iterations"]) == 100
        and int(recommendation["evaluations"]) == 10200
    )
    checks["two_thousand_matched_references"] = (
        int(final["n_reference"]) == 2000 and int(recommendation["n_reference"]) == 2000
        and int(final["reference_seed"]) == 20261024
        and int(recommendation["reference_seed"]) == 20261024
    )

    counts = np.asarray(final["counts"], dtype=float)
    q3 = np.asarray(final["reference_q3"], dtype=float)
    pass_flags = np.asarray(final["pass_flags"], dtype=int)
    densities = np.asarray(final["densities_per_km2"], dtype=float)
    reference_means = np.asarray(final["reference_mean_density"], dtype=float)
    checks["region_pass_recomputed"] = (
        np.array_equal(pass_flags, counts > q3)
        and close(final["region_pass"], np.mean(counts > q3))
    )
    expected_lift = float(np.mean(densities) / np.mean(reference_means))
    checks["density_lift_recomputed"] = close(final["density_lift"], expected_lift)

    auc_values = {}
    for key, output in (("hit", "hit_auc"), ("rmae", "rmae_aulc"), ("acc", "acc_auc")):
        expected = float(np.mean([float(by_budget[b][key]) for b in range(21, 41)]))
        auc_values[output] = expected
        checks[f"{output}_recomputed"] = close(final[output], expected)

    core_names = (
        "rmae", "acc", "balanced_acc", "f1", "hit", "region_pass", "density_lift",
        "high_crime_coverage", "component_coverage", "violent_incident_share",
        "violent_incident_uplift", "hit_auc", "rmae_aulc", "acc_auc",
    )
    checks["core_metrics_finite"] = all(math.isfinite(float(final[name])) for name in core_names)
    checks["no_matlab_checkpoint_left"] = not (history_path.parent / "matlab_checkpoint.mat").exists()

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "history": str(history_path.resolve()),
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "final_metrics": {name: final[name] for name in core_names},
        "recomputed_auc": auc_values,
        "recommendation_seconds_last_process": recommendation["seconds"],
        "total_seconds_last_process": history["total_seconds"],
    }
    if failed:
        raise AssertionError(f"Formal history verification failed: {failed}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--units", type=Path, default=ROOT / "data/units/units_2022_1km.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.history, args.units)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
