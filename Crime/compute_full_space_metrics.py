"""Evaluate final saved MMRGP models on every grid cell, including training cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from acquisition import parse_methods
from experiment_metrics import classification_metrics
from mmrgp_experiment import MATLABExperimentSession
from run_experiments import atomic_write_json, load_experiment_config, load_knowledge

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
METRICS = ["rmae", "acc", "balanced_acc", "f1"]


def load_and_validate_history(path, method, seed, units, budget):
    """Require one intact, original-knowledge formal history before refitting."""
    history = json.loads(path.read_text(encoding="utf-8"))
    if history.get("status") != "COMPLETE" or history.get("error") is not None:
        raise ValueError(f"incomplete or failed formal history: {path}")
    if history.get("knowledge_variant", "original") != "original":
        raise ValueError(f"non-original knowledge history is not eligible: {path}")
    if str(history.get("method")) != method or int(history.get("seed")) != seed:
        raise ValueError(f"method or seed mismatch: {path}")
    if int(history.get("fold", -1)) != 0 or int(history.get("total_budget")) != budget:
        raise ValueError(f"fold or budget mismatch: {path}")
    indices = np.asarray(history.get("training_indices", []), dtype=int)
    if len(indices) != budget or len(np.unique(indices)) != budget:
        raise ValueError(f"final training set does not contain {budget} unique cells: {path}")
    if np.any(indices < 0) or np.any(indices >= len(units)):
        raise ValueError(f"training cell index out of bounds: {path}")
    return history


def evaluate_saved_run(session, history, units, budget):
    """Refit the final training set and reconstruct all observed 2022 grid cells."""
    train = units.iloc[list(map(int, history["training_indices"]))]
    run_id = f"{history['run_id']}_fullspace"
    session.fit(
        train[["x", "y"]].to_numpy(float),
        train[["y1_log1p", "y2"]].to_numpy(float),
        run_id,
    )
    try:
        prediction = session.predict(run_id, units[["x", "y"]].to_numpy(float))
        rmae = float(np.mean(np.abs(
            units["y1_log1p"].to_numpy(float) - prediction["mu_z"]
        )))
        return {
            "status": "COMPLETE",
            "method": str(history["method"]),
            "fold": int(history["fold"]),
            "seed": int(history["seed"]),
            "budget": int(budget),
            "run_id": run_id,
            "n_train": int(len(train)),
            "n_eval": int(len(units)),
            "evaluation_scope": "all_cells_including_training",
            "rmae": rmae,
            **classification_metrics(units["y2"].to_numpy(int), prediction["p"]),
        }
    finally:
        session.clear(run_id)


def valid_saved_result(path, method, seed, budget, n_eval):
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    return (
        result.get("status") == "COMPLETE"
        and result.get("method") == method
        and result.get("seed") == seed
        and result.get("budget") == budget
        and result.get("n_eval") == n_eval
        and result.get("evaluation_scope") == "all_cells_including_training"
        and all(np.isfinite(float(result.get(key, np.nan))) for key in METRICS)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    methods = parse_methods(args.methods)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    settings = load_experiment_config(args.grid)
    budget = int(settings["protocol"]["total_budget"])
    split_scheme = settings["protocol"]["split_scheme"]
    tag = f"{args.grid}_budget{budget}_{split_scheme}"
    input_root = ROOT / "results" / "formal" / tag
    output = ROOT / "results" / "full_space_metrics" / tag
    output.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data" / "units" / f"units_2022_{args.grid}.csv")
    knowledge = load_knowledge(args.grid, variant="original")
    cell_area = float(settings["grids"][args.grid]["cell_area_km2"])

    histories = {}
    for method in methods:
        for seed in seeds:
            path = input_root / method / "fold_0" / f"seed_{seed}" / "history.json"
            histories[method, seed] = load_and_validate_history(
                path, method, seed, units, budget,
            )
    print(
        f"[Full-space] validated {len(histories)} histories; "
        f"budget={budget}, evaluation cells={len(units)}",
        flush=True,
    )

    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
        cell_area_km2=cell_area,
        chunk_size=8192,
    ) as session:
        for method in methods:
            for seed in seeds:
                path = output / method / "fold_0" / f"seed_{seed}" / "full_space_metrics.json"
                if valid_saved_result(path, method, seed, budget, len(units)):
                    print(f"[Full-space] skip {method} seed={seed}", flush=True)
                    continue
                started = time.perf_counter()
                result = evaluate_saved_run(session, histories[method, seed], units, budget)
                result["seconds"] = float(time.perf_counter() - started)
                atomic_write_json(path, result)
                print(
                    f"[Full-space] done {method} seed={seed} "
                    f"RMAE={result['rmae']:.4f} ACC={result['acc']:.4f} "
                    f"seconds={result['seconds']:.2f}",
                    flush=True,
                )

    rows = []
    for method in methods:
        for seed in seeds:
            path = output / method / "fold_0" / f"seed_{seed}" / "full_space_metrics.json"
            if valid_saved_result(path, method, seed, budget, len(units)):
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    if rows:
        frame = pd.DataFrame(rows)
        columns = [
            "method", "fold", "seed", "budget", "n_train", "n_eval",
            *METRICS, "tp", "tn", "fp", "fn", "seconds",
        ]
        order = {method: index for index, method in enumerate(methods)}
        frame[columns].sort_values(
            ["method", "seed"], key=lambda values: values.map(order) if values.name == "method" else values,
        ).to_csv(output / "per_run_metrics.csv", index=False)
        grouped = frame.groupby("method", sort=False)[METRICS]
        summary = pd.concat([
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=0).add_suffix("_std"),
        ], axis=1)
        summary.insert(0, "n_eval", frame.groupby("method", sort=False)["n_eval"].first())
        summary.insert(0, "n_seeds", grouped.size())
        summary.insert(0, "budget", frame.groupby("method", sort=False)["budget"].first())
        summary = summary.reset_index()
        summary.sort_values("method", key=lambda values: values.map(order)).to_csv(
            output / "summary_metrics.csv", index=False,
        )


if __name__ == "__main__":
    main()
