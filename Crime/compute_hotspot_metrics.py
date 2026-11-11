"""Compute top-20% NDCG and hotspot RMSE from saved final MMRGP training sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from acquisition import parse_methods
from hotspot_metrics import hotspot_metrics
from mmrgp_experiment import MATLABExperimentSession
from region_recommender import lognormal_intensity
from run_experiments import atomic_write_json, load_experiment_config

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
METRICS = ["ndcg_at_20", "h_rmae_at_20"]


def evaluate_saved_run(session, history, units, cell_area_km2, fraction=0.2):
    """Refit one final model and evaluate its saved random-holdout test cells."""
    if history.get("status") != "COMPLETE":
        raise ValueError("only COMPLETE formal histories can be evaluated")
    train = units.iloc[list(map(int, history["training_indices"]))]
    test = units.iloc[list(map(int, history["test_indices"]))]
    if len(train) != int(history["total_budget"]):
        raise ValueError("training indices do not match the final budget")
    run_id = str(history["run_id"])
    session.fit(
        train[["x", "y"]].to_numpy(float),
        train[["y1_log1p", "y2"]].to_numpy(float),
        run_id,
    )
    try:
        prediction = session.predict(run_id, test[["x", "y"]].to_numpy(float))
        intensity = lognormal_intensity(
            prediction["mu_z"], prediction["var_z"], cell_area_km2,
        )
        metrics = hotspot_metrics(
            test["y1_log1p"].to_numpy(float), prediction["mu_z"], intensity, fraction,
        )
        return {
            "status": "COMPLETE", "method": str(history["method"]),
            "fold": int(history["fold"]), "seed": int(history["seed"]),
            "budget": int(history["total_budget"]), "run_id": run_id,
            "n_test": int(len(test)), "fraction": float(fraction), **metrics,
        }
    finally:
        session.clear(run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    methods = parse_methods(args.methods)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    settings = load_experiment_config(args.grid)
    budget = int(settings["protocol"]["total_budget"])
    split_scheme = settings["protocol"]["split_scheme"]
    input_root = ROOT / "results" / "formal" / f"{args.grid}_budget{budget}_{split_scheme}"
    output = ROOT / "results" / "hotspot_metrics" / f"{args.grid}_budget{budget}_{split_scheme}"
    output.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data" / "units" / f"units_2022_{args.grid}.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / f"knowledge_2021_{args.grid}.csv")
    cell_area = float(settings["grids"][args.grid]["cell_area_km2"])

    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
        cell_area_km2=cell_area,
        chunk_size=8192,
    ) as session:
        for method in methods:
            for seed in seeds:
                path = output / method / "fold_0" / f"seed_{seed}" / "hotspot_metrics.json"
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old.get("status") == "COMPLETE" and old.get("fraction") == args.fraction:
                        print(f"[Hotspot] skip {method} seed={seed}", flush=True)
                        continue
                history_path = input_root / method / "fold_0" / f"seed_{seed}" / "history.json"
                history = json.loads(history_path.read_text(encoding="utf-8"))
                started = time.perf_counter()
                result = evaluate_saved_run(session, history, units, cell_area, args.fraction)
                result["seconds"] = float(time.perf_counter() - started)
                atomic_write_json(path, result)
                print(
                    f"[Hotspot] done {method} seed={seed} "
                    f"NDCG={result['ndcg_at_20']:.4f} "
                    f"H-RMAE={result['h_rmae_at_20']:.4f} "
                    f"seconds={result['seconds']:.2f}",
                    flush=True,
                )

    rows = []
    for method in methods:
        for seed in seeds:
            path = output / method / "fold_0" / f"seed_{seed}" / "hotspot_metrics.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("status") == "COMPLETE" and result.get("fraction") == args.fraction:
                    rows.append(result)
    if rows:
        frame = pd.DataFrame(rows)
        columns = ["method", "fold", "seed", "budget", "n_test", "n_hotspot", *METRICS, "seconds"]
        frame[columns].sort_values(["method", "seed"]).to_csv(
            output / "per_run_metrics.csv", index=False,
        )
        grouped = frame.groupby("method", sort=False)[METRICS]
        summary = pd.concat([
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=0).add_suffix("_std"),
        ], axis=1)
        summary.insert(0, "n_seeds", grouped.size())
        summary.insert(0, "budget", frame.groupby("method", sort=False)["budget"].first())
        summary = summary.reset_index()
        order = {method: index for index, method in enumerate(parse_methods("all"))}
        summary.sort_values("method", key=lambda values: values.map(order)).to_csv(
            output / "summary_metrics.csv", index=False,
        )


if __name__ == "__main__":
    main()
