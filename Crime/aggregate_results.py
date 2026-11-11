"""Aggregate formal histories without treating spatial folds as independent seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def seed_fold_summary(frame, metrics):
    frame = frame.copy()
    if "method" not in frame:
        frame["method"] = "ALL"
    if "budget" not in frame:
        frame["budget"] = 100
    grouped = frame.groupby(["method", "budget", "seed"], as_index=False)[metrics].mean()
    rows = []
    for (method, budget), group in grouped.groupby(["method", "budget"]):
        row = {"method": method, "budget": int(budget), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "budget"]).reset_index(drop=True)


def final_budget_summary(frame, metrics):
    final_budget = int(frame["budget"].max())
    final = frame[frame["budget"] == final_budget]
    return final_budget, seed_fold_summary(final, metrics)


def load_histories(input_root):
    rows = []
    for path in sorted(Path(input_root).glob("*/fold_*/seed_*/history.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        if history.get("status") != "COMPLETE":
            continue
        for evaluation in history["evaluations"]:
            row = {
                "method": history["method"], "fold": history["fold"], "seed": history["seed"],
                "grid": history["grid"], **evaluation,
            }
            row.pop("recommendation", None)
            rows.append(row)
    if not rows:
        raise RuntimeError("no complete formal histories found")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "formal" / "1km")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "aggregated" / "1km")
    args = parser.parse_args()
    frame = load_histories(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "per_run_budget_metrics.csv", index=False)
    metrics = [
        "region_pass", "density_lift", "high_crime_coverage", "component_coverage",
        "violent_incident_share", "violent_incident_uplift", "violent_label_share",
        "violent_label_uplift", "hit", "component_hit_coverage", "boundary_hit",
        "rmae", "acc", "balanced_acc", "f1", "hit_auc", "rmae_aulc", "acc_auc",
        "fit_seconds", "selection_seconds", "recommendation_seconds",
    ]
    metrics = [name for name in metrics if name in frame]
    seed_fold_summary(frame, metrics).to_csv(args.output / "summary_by_budget.csv", index=False)
    final_budget, final = final_budget_summary(frame, metrics)
    final.to_csv(args.output / f"summary_budget_{final_budget}.csv", index=False)


if __name__ == "__main__":
    main()
