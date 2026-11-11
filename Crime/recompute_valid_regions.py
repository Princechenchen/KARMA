"""Recompute total-intensity GSO regions strictly inside the valid grid union."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from acquisition import parse_methods
from mmrgp_experiment import MATLABExperimentSession
from recommend_violent_regions import (
    _final_evaluation, _load_gso_state, _parse_seeds, _save_gso_state,
)
from region_recommender import RecommendationConfig, evaluate_recommendations
from resumable_gso import ResumableGSO
from run_experiments import (
    KNOWLEDGE_VARIANTS, UNITS_VARIANTS, atomic_write_json,
    load_experiment_config, load_knowledge, load_units,
)
from smoke_lhs_oracle import load_domain, load_incidents
from spatial_validity import GridBoxValidity

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
METRICS = ["region_pass", "density_lift"]


def run_valid_recommendation(
    session, history, units, incidents_xy, domain_width_km, box_valid_fn,
    *, state_path=None, max_steps=None, max_seconds=None,
):
    evaluation = _final_evaluation(history)
    old = evaluation["recommendation"]
    run_id = str(history["run_id"])
    train = units.iloc[list(map(int, history["training_indices"]))]
    if len(train) != int(history["total_budget"]):
        raise ValueError("training indices do not match the final budget")
    config = RecommendationConfig(**old["config"])
    started = time.perf_counter()
    print(f"[ValidGSO] fitting final MMRGP for {run_id}", flush=True)
    session.fit(
        train[["x", "y"]].to_numpy(float),
        train[["y1_log1p", "y2"]].to_numpy(float), run_id,
    )
    print(f"[ValidGSO] fitted {run_id} in {time.perf_counter() - started:.1f}s", flush=True)
    try:
        field = lambda points: session.intensity(run_id, points)
        if state_path is not None and Path(state_path).exists():
            state = _load_gso_state(state_path)
            search = ResumableGSO.from_state(
                field, domain_width_km, config, state, box_valid_fn,
            )
            elapsed = float(state["seconds"])
        else:
            search = ResumableGSO(
                field, domain_width_km, config, int(old["seed"]), box_valid_fn,
            )
            elapsed = 0.0
        steps = 0
        while search.iteration < config.iterations:
            search.step()
            steps += 1
            elapsed_now = elapsed + time.perf_counter() - started
            if state_path is not None:
                _save_gso_state(state_path, search, elapsed_now)
            if (max_steps is not None and steps >= max_steps) or (
                max_seconds is not None and elapsed_now >= max_seconds
            ):
                break
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
            raise RuntimeError("valid-grid GSO returned fewer than K regions")
        boxes = np.asarray(recommendation["boxes"], dtype=float)
        if not np.all(box_valid_fn(boxes)):
            raise RuntimeError("valid-grid GSO returned an invalid box")
        n_reference = int(old["n_reference"])
        reference_seed = int(old["reference_seed"])
        metrics = evaluate_recommendations(
            boxes, incidents_xy, domain_width_km, n_reference, reference_seed,
            box_valid_fn,
        )
        if state_path is not None and Path(state_path).exists():
            Path(state_path).unlink()
        return {
            "status": "COMPLETE", "method": str(history["method"]),
            "fold": int(history["fold"]), "seed": int(history["seed"]),
            "budget": int(history["total_budget"]), "run_id": run_id,
            "n_train": int(len(train)), "objective": "total_intensity_strict_valid_grid",
            "recommendation_seed": int(old["seed"]),
            "recommendation_config": config.to_dict(),
            "n_reference": n_reference, "reference_seed": reference_seed,
            "valid_recommended_boxes": int(len(boxes)),
            "valid_reference_boxes": int(len(boxes) * n_reference),
            "boxes": boxes, "scores": np.asarray(recommendation["scores"]),
            "gso_evaluations": int(recommendation["evaluations"]),
            **metrics,
            "seconds": float(elapsed + time.perf_counter() - final_started),
        }
    finally:
        session.clear(run_id)


def _result_path(output, method, seed):
    return output / method / "fold_0" / f"seed_{seed}" / "valid_recommendation.json"


def _save_aggregates(output, methods, seeds):
    rows = []
    for method in methods:
        for seed in seeds:
            path = _result_path(output, method, seed)
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("status") == "COMPLETE":
                    rows.append(result)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    columns = [
        "method", "fold", "seed", "budget", "n_train", "recommendation_seed",
        "valid_recommended_boxes", "valid_reference_boxes", *METRICS,
        "counts", "pass_flags", "seconds",
    ]
    frame[columns].sort_values(["method", "seed"]).to_csv(output / "per_run_metrics.csv", index=False)
    grouped = frame.groupby("method", sort=False)[METRICS]
    summary = pd.concat([
        grouped.mean().add_suffix("_mean"), grouped.std(ddof=0).add_suffix("_std"),
    ], axis=1)
    summary.insert(0, "n_seeds", grouped.size())
    summary.insert(0, "budget", frame.groupby("method", sort=False)["budget"].first())
    summary.reset_index().to_csv(output / "summary_metrics.csv", index=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--knowledge-variant", default="original", choices=KNOWLEDGE_VARIANTS)
    parser.add_argument("--units-variant", default="original", choices=UNITS_VARIANTS)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    parser.add_argument("--max-steps", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    if args.max_steps < 1 or args.max_seconds <= 0:
        raise ValueError("max-steps and max-seconds must be positive")
    deadline = time.perf_counter() + args.max_seconds
    methods = parse_methods(args.methods)
    seeds = _parse_seeds(args.seeds)
    settings = load_experiment_config(args.grid)
    budget = int(settings["protocol"]["total_budget"])
    split_scheme = settings["protocol"]["split_scheme"]
    input_root = args.input or ROOT / "results" / "formal" / f"{args.grid}_budget{budget}_{split_scheme}"
    output = args.output or ROOT / "results" / "valid_region_metrics" / f"{args.grid}_budget{budget}_{split_scheme}"
    output.mkdir(parents=True, exist_ok=True)
    units = load_units(args.grid, variant=args.units_variant)
    validity = GridBoxValidity(units)
    knowledge = load_knowledge(args.grid, variant=args.knowledge_variant)
    construct, domain_width = load_domain()
    _, incidents_xy = load_incidents(construct)
    cell_area = float(settings["grids"][args.grid]["cell_area_km2"])
    stop = False
    session_started = time.perf_counter()
    print("[ValidGSO] starting MATLAB session", flush=True)
    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
        cell_area_km2=cell_area, chunk_size=51200,
    ) as session:
        print(f"[ValidGSO] MATLAB session ready in {time.perf_counter() - session_started:.1f}s", flush=True)
        for method in methods:
            if stop:
                break
            for seed in seeds:
                if stop:
                    break
                path = _result_path(output, method, seed)
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old.get("status") == "COMPLETE":
                        print(f"[ValidGSO] skip {method} seed={seed}", flush=True)
                        continue
                history = json.loads((input_root / method / "fold_0" / f"seed_{seed}" / "history.json").read_text(encoding="utf-8"))
                state_path = path.with_name("gso_state.npz")
                while True:
                    print(f"[ValidGSO] run {method} seed={seed}", flush=True)
                    try:
                        result = run_valid_recommendation(
                            session, history, units, incidents_xy, domain_width, validity,
                            state_path=state_path, max_steps=args.max_steps,
                            max_seconds=max(1.0, deadline - time.perf_counter()),
                        )
                    except Exception as exc:
                        atomic_write_json(path, {
                            "status": "FAILED", "method": method, "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise
                    if result["status"] == "RUNNING":
                        print(
                            f"[ValidGSO] checkpoint {method} seed={seed} "
                            f"iteration={result['iteration']}/{result['iterations']}", flush=True,
                        )
                        if time.perf_counter() >= deadline:
                            stop = True
                            break
                        continue
                    atomic_write_json(path, result)
                    _save_aggregates(output, methods, seeds)
                    print(
                        f"[ValidGSO] done {method} seed={seed} "
                        f"RP={result['region_pass']:.3f} DL={result['density_lift']:.3f} "
                        f"valid={result['valid_recommended_boxes']}/4", flush=True,
                    )
                    if time.perf_counter() >= deadline:
                        stop = True
                    break
    _save_aggregates(output, methods, seeds)
    if stop:
        print("[ValidGSO] execution window reached; checkpoints saved", flush=True)


if __name__ == "__main__":
    main()
