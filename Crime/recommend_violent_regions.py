"""Re-recommend regions with the joint violent intensity lambda_all * p_violent.

The final 50-sample MMRGP for every completed formal run is reconstructed from
its saved training indices. No acquisition is repeated. GSO settings and seeds
are paired with the original total-intensity recommendation; only the objective
changes. Both total-crime and violent-crime metrics are evaluated on the new boxes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from acquisition import parse_methods
from mmrgp_experiment import MATLABExperimentSession
from region_recommender import (
    RecommendationConfig,
    evaluate_recommendations,
    lognormal_intensity,
)
from resumable_gso import ResumableGSO
from run_experiments import atomic_write_json, load_experiment_config
from smoke_lhs_oracle import load_domain, load_incidents

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
METRICS = ["region_pass", "density_lift", "violent_region_pass", "violent_density_lift"]


def violent_intensity(session, run_id, query_xy, cell_area_km2):
    """Return E[count|xy] / cell_area multiplied by P(violent|xy)."""
    prediction = session.predict(run_id, query_xy)
    total = lognormal_intensity(
        prediction["mu_z"], prediction["var_z"], cell_area_km2,
    )
    return total * prediction["p"]


def _final_evaluation(history):
    if history.get("status") != "COMPLETE":
        raise ValueError("only COMPLETE formal histories can be re-recommended")
    budget = int(history["total_budget"])
    evaluations = [row for row in history["evaluations"] if int(row["budget"]) == budget]
    if len(evaluations) != 1 or "recommendation" not in evaluations[0]:
        raise ValueError("formal history lacks its final recommendation")
    return evaluations[0]


def _save_gso_state(path, search, seconds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    state = search.state_dict()
    state["rng_state"] = json.dumps(state["rng_state"])
    state["seconds"] = float(seconds)
    np.savez(temporary, **state)
    temporary.replace(path)


def _load_gso_state(path):
    with np.load(path, allow_pickle=False) as data:
        state = {key: data[key] for key in data.files}
    state["iteration"] = int(state["iteration"])
    state["evaluations"] = int(state["evaluations"])
    state["rng_state"] = json.loads(str(state["rng_state"]))
    return state


def run_recommendation(
    session, history, units, incidents, incidents_xy,
    domain_width_km, cell_area_km2, *, state_path=None, max_steps=None,
):
    """Refit one saved final training set, re-search, and evaluate both responses."""
    evaluation = _final_evaluation(history)
    old_recommendation = evaluation["recommendation"]
    run_id = str(history["run_id"])
    train = units.iloc[list(map(int, history["training_indices"]))]
    if len(train) != int(history["total_budget"]):
        raise ValueError("training indices do not match the final budget")
    config = RecommendationConfig(**old_recommendation["config"])
    started = time.perf_counter()
    session.fit(
        train[["x", "y"]].to_numpy(float),
        train[["y1_log1p", "y2"]].to_numpy(float),
        run_id,
    )
    try:
        field = lambda points: violent_intensity(session, run_id, points, cell_area_km2)
        if state_path is not None and Path(state_path).exists():
            state = _load_gso_state(state_path)
            search = ResumableGSO.from_state(field, domain_width_km, config, state)
            elapsed = float(state["seconds"])
        else:
            search = ResumableGSO(
                field, domain_width_km, config, seed=int(old_recommendation["seed"]),
            )
            elapsed = 0.0
        steps = 0
        while search.iteration < config.iterations and (max_steps is None or steps < max_steps):
            search.step()
            steps += 1
            if state_path is not None:
                _save_gso_state(state_path, search, elapsed + time.perf_counter() - started)
        elapsed += time.perf_counter() - started
        if search.iteration < config.iterations:
            return {
                "status": "RUNNING", "method": str(history["method"]),
                "seed": int(history["seed"]), "iteration": int(search.iteration),
                "iterations": int(config.iterations), "seconds": float(elapsed),
            }
        final_started = time.perf_counter()
        recommendation = search.finalize()
        if not recommendation["success"]:
            raise RuntimeError("joint-intensity GSO returned fewer than K regions")
        boxes = np.asarray(recommendation["boxes"], dtype=float)
        n_reference = int(old_recommendation["n_reference"])
        reference_seed = int(old_recommendation["reference_seed"])
        total = evaluate_recommendations(
            boxes, incidents_xy, domain_width_km, n_reference, reference_seed,
        )
        violent_xy = incidents_xy[incidents["is_violent"].to_numpy(bool)]
        violent = evaluate_recommendations(
            boxes, violent_xy, domain_width_km, n_reference, reference_seed,
        )
        if state_path is not None and Path(state_path).exists():
            Path(state_path).unlink()
        return {
            "status": "COMPLETE",
            "method": str(history["method"]),
            "fold": int(history["fold"]),
            "seed": int(history["seed"]),
            "budget": int(history["total_budget"]),
            "run_id": run_id,
            "objective": "lambda_all_times_p_violent",
            "recommendation_seed": int(old_recommendation["seed"]),
            "recommendation_config": config.to_dict(),
            "n_reference": n_reference,
            "reference_seed": reference_seed,
            "boxes": boxes,
            "scores": np.asarray(recommendation["scores"]),
            "gso_evaluations": int(recommendation["evaluations"]),
            "region_pass": total["region_pass"],
            "density_lift": total["density_lift"],
            "violent_region_pass": violent["region_pass"],
            "violent_density_lift": violent["density_lift"],
            "total_evaluation": total,
            "violent_evaluation": violent,
            "seconds": float(elapsed + time.perf_counter() - final_started),
        }
    finally:
        session.clear(run_id)


def aggregate_metrics(rows):
    """Aggregate per-run metrics across seeds with population standard deviations."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["method", "budget", "n_seeds", *[
            f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")
        ]])
    grouped = frame.groupby("method", sort=False)[METRICS]
    summary = pd.concat([
        grouped.mean().add_suffix("_mean"),
        grouped.std(ddof=0).add_suffix("_std"),
    ], axis=1)
    summary.insert(0, "n_seeds", grouped.size())
    if "budget" in frame:
        summary.insert(0, "budget", frame.groupby("method", sort=False)["budget"].first())
    summary = summary.reset_index()
    order = {method: index for index, method in enumerate(parse_methods("all"))}
    return summary.sort_values("method", key=lambda values: values.map(order)).reset_index(drop=True)


def _result_path(output, method, fold, seed):
    return output / method / f"fold_{fold}" / f"seed_{seed}" / "violent_recommendation.json"


def _save_aggregates(output, methods, seeds):
    rows = []
    for method in methods:
        for seed in seeds:
            path = _result_path(output, method, 0, seed)
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") == "COMPLETE":
                rows.append(result)
    if not rows:
        return
    columns = ["method", "fold", "seed", "budget", "recommendation_seed", *METRICS, "seconds"]
    frame = pd.DataFrame(rows)[columns].sort_values(["method", "seed"])
    frame.to_csv(output / "per_run_metrics.csv", index=False)
    aggregate_metrics(rows).to_csv(output / "summary_metrics.csv", index=False)


def _parse_seeds(text):
    return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, help="stop cleanly after this execution window")
    parser.add_argument("--max-steps", type=int, default=10, help="GSO iterations between progress checks")
    return parser


def main():
    args = build_parser().parse_args()
    if args.max_steps < 1 or (args.max_seconds is not None and args.max_seconds <= 0):
        raise ValueError("max-steps and max-seconds must be positive")
    deadline = None if args.max_seconds is None else time.perf_counter() + args.max_seconds
    methods = parse_methods(args.methods)
    seeds = _parse_seeds(args.seeds)
    settings = load_experiment_config(args.grid)
    budget = int(settings["protocol"]["total_budget"])
    split_scheme = settings["protocol"]["split_scheme"]
    input_root = args.input or ROOT / "results" / "formal" / f"{args.grid}_budget{budget}_{split_scheme}"
    output = args.output or ROOT / "results" / "violent_recommendation" / f"{args.grid}_budget{budget}_{split_scheme}"
    output.mkdir(parents=True, exist_ok=True)

    units = pd.read_csv(ROOT / "data" / "units" / f"units_2022_{args.grid}.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / f"knowledge_2021_{args.grid}.csv")
    construct, domain_width = load_domain()
    incidents, incidents_xy = load_incidents(construct)
    cell_area = float(settings["grids"][args.grid]["cell_area_km2"])
    stop = False

    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
        cell_area_km2=cell_area,
        chunk_size=8192,
    ) as session:
        for method in methods:
            if stop:
                break
            for seed in seeds:
                if stop:
                    break
                path = _result_path(output, method, 0, seed)
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old.get("status") == "COMPLETE":
                        print(f"[ViolentGSO] skip {method} seed={seed}", flush=True)
                        continue
                history_path = input_root / method / "fold_0" / f"seed_{seed}" / "history.json"
                history = json.loads(history_path.read_text(encoding="utf-8"))
                state_path = path.with_name("gso_state.npz")
                while True:
                    print(f"[ViolentGSO] run {method} seed={seed}", flush=True)
                    try:
                        result = run_recommendation(
                            session, history, units, incidents, incidents_xy,
                            domain_width, cell_area, state_path=state_path,
                            max_steps=args.max_steps,
                        )
                    except Exception as exc:
                        atomic_write_json(path, {
                            "status": "FAILED", "method": method, "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise
                    if result["status"] == "RUNNING":
                        print(
                            f"[ViolentGSO] checkpoint {method} seed={seed} "
                            f"iteration={result['iteration']}/{result['iterations']}",
                            flush=True,
                        )
                        if deadline is not None and time.perf_counter() >= deadline:
                            stop = True
                            break
                        continue
                    atomic_write_json(path, result)
                    _save_aggregates(output, methods, seeds)
                    print(
                        f"[ViolentGSO] done {method} seed={seed} "
                        f"RP={result['region_pass']:.3f} DL={result['density_lift']:.3f} "
                        f"VRP={result['violent_region_pass']:.3f} "
                        f"VDL={result['violent_density_lift']:.3f} "
                        f"seconds={result['seconds']:.1f}",
                        flush=True,
                    )
                    if deadline is not None and time.perf_counter() >= deadline:
                        stop = True
                    break
    _save_aggregates(output, methods, seeds)
    if stop:
        print("[ViolentGSO] execution window reached; checkpoints saved", flush=True)


if __name__ == "__main__":
    main()
