"""Validate the isolated KARMA initial-20 plus sequential-40 experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "results/formal/1km_budget60/KARMA/fold_0/seed_172/history.json"
UNITS = ROOT / "data/units/units_2022_1km.csv"
OUTPUT = HISTORY.with_name("verification.json")


def main():
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    units = pd.read_csv(UNITS)
    evaluations = history["evaluations"]
    by_budget = {int(row["budget"]): row for row in evaluations}
    acquisitions = history["acquisitions"]
    indices = list(map(int, history["training_indices"]))
    final = by_budget[60]
    recommendation = final["recommendation"]
    checks = {
        "complete_b60_identity": (
            history["status"] == "COMPLETE" and history["method"] == "KARMA"
            and history["fold"] == 0 and history["seed"] == 172 and history["grid"] == "1km"
            and history["initial_budget"] == 20 and history["sequential_steps"] == 40
            and history["total_budget"] == 60
        ),
        "all_budgets_20_through_60": [row["budget"] for row in evaluations] == list(range(20, 61)),
        "sixty_unique_training_cells": (
            len(indices) == 60 and len(set(indices)) == 60
            and len(set(history["training_cell_ids"])) == 60
        ),
        "training_ids_match": units.iloc[indices]["cell_id"].tolist() == history["training_cell_ids"],
        "held_out_fold_never_sampled": bool(np.all(units.iloc[indices]["spatial_block"].to_numpy(int) != 0)),
        "forty_sequential_acquisitions": (
            len(acquisitions) == 40
            and [row["from_budget"] for row in acquisitions] == list(range(20, 60))
            and [row["step"] for row in acquisitions] == list(range(1, 41))
            and [row["selected_index"] for row in acquisitions] == indices[20:]
        ),
        "karma_has_forty_roi_steps": len(acquisitions[-1]["diagnostics"]["roi_selection_hist"]) == 40,
        "recommendations_at_b20_b40_b60": all("recommendation" in by_budget[b] for b in (20, 40, 60)),
        "four_boxes_and_gso_100_by_100": (
            len(recommendation["boxes"]) == 4 and recommendation["evaluations"] == 10200
            and recommendation["config"]["particles"] == 100
            and recommendation["config"]["iterations"] == 100
        ),
        "two_thousand_matched_references": final["n_reference"] == recommendation["n_reference"] == 2000,
        "region_pass_recomputed": (
            np.array_equal(np.asarray(final["pass_flags"]), np.asarray(final["counts"]) > np.asarray(final["reference_q3"]))
            and math.isclose(final["region_pass"], float(np.mean(np.asarray(final["counts"]) > np.asarray(final["reference_q3"]))))
        ),
        "density_lift_recomputed": math.isclose(
            final["density_lift"],
            float(np.mean(final["densities_per_km2"]) / np.mean(final["reference_mean_density"])),
            rel_tol=1e-12,
        ),
        "b40_auc_not_reused_for_b60": all(
            key not in final for key in ("hit_auc", "rmae_aulc", "acc_auc")
        ),
        "core_metrics_finite": all(math.isfinite(float(final[key])) for key in (
            "rmae", "acc", "balanced_acc", "f1", "hit", "region_pass", "density_lift",
            "high_crime_coverage", "component_coverage", "violent_incident_share",
            "violent_incident_uplift",
        )),
        "no_checkpoint_left": not HISTORY.with_name("matlab_checkpoint.mat").exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "b40_within_this_run": {key: by_budget[40][key] for key in (
            "rmae", "acc", "balanced_acc", "f1", "hit", "region_pass", "density_lift",
            "high_crime_coverage", "component_coverage",
        )},
        "b60_final": {key: final[key] for key in (
            "rmae", "acc", "balanced_acc", "f1", "hit", "region_pass", "density_lift",
            "high_crime_coverage", "component_coverage", "counts", "reference_q3", "pass_flags",
        )},
        "recommendation_seconds_last_process": final["recommendation_seconds"],
        "total_seconds_last_process": history["total_seconds"],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if failed:
        raise SystemExit(f"Verification failed: {failed}")


if __name__ == "__main__":
    main()
