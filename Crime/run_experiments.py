"""Resumable folded offline-replay experiments using real MATLAB MMRGP only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml

from acquisition import lhs_sequence, parse_methods
from experiment_metrics import classification_metrics, region_supplements, sequential_auc
from mmrgp_experiment import MATLABExperimentSession
from region_recommender import (
    RecommendationConfig,
    evaluate_recommendations,
    recommend_regions,
    rectangle_counts,
)
from smoke_lhs_oracle import load_domain, load_incidents

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158, 180, 1]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path, payload):
    """Atomically write strict JSON, never leaving a partial history file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_experiment_config(grid="1km"):
    path = ROOT / "config" / "experiment.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if grid not in config["grids"]:
        raise ValueError(f"unsupported grid {grid}; choose from {list(config['grids'])}")
    return config


KNOWLEDGE_VARIANTS = (
    "original", "adjacent", "hotspot", "oracle2022_threshold", "oracle2022_adjacent",
    "oracle2022_hotspot_threshold", "oracle2022_hotspot_adjacent",
    "hotspot_highside_adjacent", "oracle2022_hotspot_highside_adjacent",
    "misleading2022_hotspot_highside_inverted",
    "misleading2022_hotspot_highside_rotated180",
)
UNITS_VARIANTS = ("original", "hotspot")


def load_units(grid="1km", variant="original"):
    """Load an isolated response variant without modifying source units."""
    if variant not in UNITS_VARIANTS:
        raise ValueError(f"unsupported units variant {variant}; choose from {UNITS_VARIANTS}")
    suffix = "" if variant == "original" else "_hotspot"
    return pd.read_csv(ROOT / "data" / "units" / f"units_2022_{grid}{suffix}.csv")


def load_knowledge(grid="1km", variant="original"):
    """Load a knowledge variant without modifying the source artifacts."""
    if variant not in KNOWLEDGE_VARIANTS:
        raise ValueError(f"unsupported knowledge variant {variant}; choose from {KNOWLEDGE_VARIANTS}")
    if variant.startswith(("oracle2022_hotspot_", "misleading2022_hotspot_")):
        path = ROOT / "data" / "knowledge" / f"knowledge_2022_{grid}_hotspot_oracle.csv"
        knowledge = pd.read_csv(path)
        columns = {
            "oracle2022_hotspot_threshold": "pi_bnd_hotspot_threshold_2022",
            "oracle2022_hotspot_adjacent": "pi_bnd_hotspot_adjacent_2022",
            "oracle2022_hotspot_highside_adjacent": "pi_bnd_highside_adjacent_2022",
            "misleading2022_hotspot_highside_inverted": "pi_bnd_highside_inverted_2022",
            "misleading2022_hotspot_highside_rotated180": "pi_bnd_highside_rotated180_2022",
        }
        column = columns[variant]
        if column not in knowledge:
            raise ValueError(f"{variant} knowledge is missing {column}: {path}")
        knowledge["pi_bnd"] = knowledge[column]
        return knowledge
    if variant.startswith("oracle2022_"):
        path = ROOT / "data" / "knowledge" / f"knowledge_2022_{grid}_oracle.csv"
        knowledge = pd.read_csv(path)
        column = (
            "pi_bnd_threshold_2022" if variant == "oracle2022_threshold"
            else "pi_bnd_adjacent_2022"
        )
        if column not in knowledge:
            raise ValueError(f"{variant} knowledge is missing {column}: {path}")
        knowledge["pi_bnd"] = knowledge[column]
        return knowledge

    suffixes = {
        "original": "", "adjacent": "_adjacent", "hotspot": "_hotspot",
        "hotspot_highside_adjacent": "_hotspot_highside_adjacent",
    }
    path = ROOT / "data" / "knowledge" / f"knowledge_2021_{grid}{suffixes[variant]}.csv"
    knowledge = pd.read_csv(path)
    if variant == "adjacent":
        if "pi_bnd_adjacent" not in knowledge:
            raise ValueError(f"adjacent knowledge is missing pi_bnd_adjacent: {path}")
        knowledge["pi_bnd"] = knowledge["pi_bnd_adjacent"]
    if variant == "hotspot_highside_adjacent":
        if "pi_bnd_highside_adjacent" not in knowledge:
            raise ValueError(f"high-side knowledge is missing pi_bnd_highside_adjacent: {path}")
        knowledge["pi_bnd"] = knowledge["pi_bnd_highside_adjacent"]
    return knowledge


def random_holdout_indices(n_units, seed, test_fraction=0.2):
    """Return a sorted, response-independent random test split for one seed."""
    n_units = int(n_units)
    fraction = float(test_fraction)
    if n_units < 2 or not 0.0 < fraction < 1.0:
        raise ValueError("invalid random holdout dimensions")
    count = int(round(n_units * fraction))
    if not 1 <= count < n_units:
        raise ValueError("random holdout leaves no train or test cells")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_units, size=count, replace=False)).astype(int)


class ExperimentRunner:
    def __init__(
        self, session, units, incidents, incidents_xy, domain_width_km,
        output_root, *, grid="1km", initial_budget=20, sequential_steps=20,
        evaluation_budgets=(20, 40), reference_points=None, auc_steps=20,
        recommendation_config=None, n_reference=2000, reference_seed=20261024,
        split_scheme="random_holdout", test_fraction=0.2, knowledge_variant="original",
        units_variant="original",
    ):
        self.session = session
        self.units = units.reset_index(drop=True).copy()
        self.incidents = incidents
        self.incidents_xy = np.asarray(incidents_xy, dtype=float)
        self.domain_width_km = tuple(domain_width_km)
        self.output_root = Path(output_root)
        self.grid = grid
        self.knowledge_variant = str(knowledge_variant)
        self.units_variant = str(units_variant)
        self.split_scheme = str(split_scheme)
        self.test_fraction = float(test_fraction)
        if self.split_scheme != "random_holdout":
            raise ValueError("only random_holdout is supported")
        self.initial_budget = int(initial_budget)
        self.sequential_steps = int(sequential_steps)
        self.auc_steps = int(auc_steps)
        self.total_budget = self.initial_budget + self.sequential_steps
        self.evaluation_budgets = set(int(value) for value in evaluation_budgets)
        self.evaluation_budgets.add(self.total_budget)
        self.recommendation_config = recommendation_config or RecommendationConfig()
        self.n_reference = int(n_reference)
        self.reference_seed = int(reference_seed)
        self.reference_points = np.asarray(reference_points, dtype=float)
        if self.reference_points.ndim != 2 or self.reference_points.shape[1] != 2:
            raise ValueError("reference_points must be an N-by-2 array")
        required = {"x", "y", "y1_log1p", "y2", "spatial_block", "cell_id", "is_key_q3",
                    "key_component_id", "is_classification_boundary"}
        missing = required.difference(self.units.columns)
        if missing:
            raise ValueError(f"units are missing columns: {sorted(missing)}")

    def _run_id(self, method, fold, seed):
        return f"crimes_{self.grid.replace('.', 'p')}_{method.replace('-', '_')}_{fold}_{seed}"

    def _history_path(self, method, fold, seed):
        return self.output_root / method / f"fold_{fold}" / f"seed_{seed}" / "history.json"

    def _surrogate_metrics(self, run_id, train_indices, test_indices, budget):
        train = self.units.iloc[train_indices]
        test = self.units.iloc[test_indices]
        prediction = self.session.predict(run_id, test[["x", "y"]].to_numpy(float))
        rmae = float(np.mean(np.abs(test["y1_log1p"].to_numpy(float) - prediction["mu_z"])))
        result = {
            "budget": int(budget), "rmae": rmae,
            **classification_metrics(test["y2"].to_numpy(int), prediction["p"]),
            "hit": float(train["is_key_q3"].mean()),
            "component_hit_coverage": float(
                train.loc[train["key_component_id"] >= 0, "key_component_id"].nunique()
                / max(self.units.loc[self.units["key_component_id"] >= 0, "key_component_id"].nunique(), 1)
            ),
            "boundary_hit": float(train["is_classification_boundary"].mean()),
            "n_train": int(len(train)), "n_test": int(len(test)),
            "selection_seconds": 0.0, "recommendation_seconds": 0.0,
        }
        return result

    def _region_metrics(self, run_id, fold, seed, budget):
        recommendation_seed = int(seed + 1000 * fold + budget)
        recommendation = recommend_regions(
            lambda points: self.session.intensity(run_id, points),
            self.domain_width_km, self.recommendation_config, seed=recommendation_seed,
        )
        if not recommendation["success"]:
            raise RuntimeError(f"GSO returned fewer than K non-overlapping regions at budget {budget}")
        boxes = np.asarray(recommendation["boxes"], dtype=float)
        metrics = evaluate_recommendations(
            boxes, self.incidents_xy, self.domain_width_km,
            n_reference=self.n_reference, seed=self.reference_seed,
        )
        supplements = region_supplements(boxes, self.units, float(self.units["y2"].mean()))
        violent_xy = self.incidents_xy[self.incidents["is_violent"].to_numpy(bool)]
        violent_counts = rectangle_counts(violent_xy, boxes)
        total_counts = np.asarray(metrics["counts"], dtype=float)
        violent_share = float(violent_counts.sum() / total_counts.sum()) if total_counts.sum() else float("nan")
        supplements.update({
            "violent_incident_share": violent_share,
            "violent_incident_uplift": float(violent_share - self.incidents["is_violent"].mean()),
            "violent_incident_counts": violent_counts.tolist(),
        })
        serializable_recommendation = {
            "boxes": boxes.tolist(), "scores": np.asarray(recommendation["scores"]).tolist(),
            "seed": recommendation_seed, "evaluations": recommendation["evaluations"],
            "seconds": recommendation["seconds"], "config": recommendation["config"],
            "reference_seed": self.reference_seed, "n_reference": self.n_reference,
        }
        return {**metrics, **supplements}, serializable_recommendation

    def _checkpoint_path(self, method, fold, seed):
        return self._history_path(method, fold, seed).with_name("matlab_checkpoint.mat")

    def run(self, method, fold, seed):
        method, fold, seed = str(method), int(fold), int(seed)
        path = self._history_path(method, fold, seed)
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if old and old.get("knowledge_variant", "original") != getattr(self, "knowledge_variant", "original"):
            raise RuntimeError("knowledge variant mismatch between history and current run")
        if old and old.get("units_variant", "original") != getattr(self, "units_variant", "original"):
            raise RuntimeError("units variant mismatch between history and current run")
        if old and old.get("status") == "COMPLETE":
            return old
        run_id = self._run_id(method, fold, seed)
        checkpoint = self._checkpoint_path(method, fold, seed)
        test_indices = random_holdout_indices(len(self.units), seed, self.test_fraction)
        if old and (
            old.get("split_scheme") != self.split_scheme
            or list(map(int, old.get("test_indices", []))) != test_indices.tolist()
        ):
            raise RuntimeError("split mismatch between history and current seed protocol")
        eligible = np.setdiff1d(np.arange(len(self.units), dtype=int), test_indices)
        lhs = lhs_sequence(self.units, eligible, seed, total=self.total_budget)
        resume = bool(
            old and checkpoint.exists()
            and hasattr(self.session, "restore_checkpoint")
            and old.get("run_id") == run_id
            and old.get("total_budget") == self.total_budget
        )
        if resume:
            history = old
            history["status"] = "RUNNING"
            train_indices = list(map(int, history["training_indices"]))
            train = self.units.iloc[train_indices]
            checkpoint_budget = self.session.restore_checkpoint(
                run_id, checkpoint,
                train[["x", "y"]].to_numpy(float),
                train[["y1_log1p", "y2"]].to_numpy(float),
            )
            if checkpoint_budget not in {len(train_indices), len(train_indices) - 1}:
                raise RuntimeError("checkpoint and history are more than one acquisition apart")
            need_fit = len(train_indices) > checkpoint_budget
        else:
            self.session.clear(run_id)
            train_indices = list(map(int, lhs[:self.initial_budget]))
            history = {
                "status": "RUNNING", "method": method, "fold": fold, "seed": seed, "grid": self.grid,
                "run_id": run_id, "initial_budget": self.initial_budget,
                "sequential_steps": self.sequential_steps, "total_budget": self.total_budget,
                "training_indices": train_indices.copy(),
                "training_cell_ids": self.units.iloc[train_indices]["cell_id"].tolist(),
                "split_scheme": self.split_scheme, "test_fraction": self.test_fraction,
                "knowledge_variant": self.knowledge_variant, "units_variant": self.units_variant,
                "test_indices": test_indices.tolist(),
                "test_cell_ids": self.units.iloc[test_indices]["cell_id"].tolist(),
                "evaluations": [], "acquisitions": [],
                "recommendation_config": self.recommendation_config.to_dict(),
            }
            atomic_write_json(path, history)
            need_fit = True
        started = time.perf_counter()
        try:
            if resume and len(train_indices) > checkpoint_budget:
                budget = checkpoint_budget
                observed = set(train_indices[:budget])
                candidates = np.asarray([i for i in eligible if i not in observed], dtype=int)
                if method != "C1":
                    candidate_xy = self.units.iloc[candidates][["x", "y"]].to_numpy(float)
                    local = self.session.select(
                        method, candidate_xy, self.reference_points, run_id,
                        budget - self.initial_budget + 1, seed,
                    )
                    selected = int(candidates[local])
                    if selected != train_indices[-1]:
                        raise RuntimeError("replayed acquisition differs from saved history")

            while len(train_indices) <= self.total_budget:
                budget = len(train_indices)
                train = self.units.iloc[train_indices]
                train_xy = train[["x", "y"]].to_numpy(float)
                train_y = train[["y1_log1p", "y2"]].to_numpy(float)
                fit_seconds = 0.0
                if need_fit:
                    fit_started = time.perf_counter()
                    self.session.fit(train_xy, train_y, run_id)
                    fit_seconds = time.perf_counter() - fit_started
                    if budget == self.total_budget and hasattr(self.session, "save_checkpoint"):
                        self.session.save_checkpoint(run_id, checkpoint, budget)
                evaluated_budgets = {int(row["budget"]) for row in history["evaluations"]}
                if budget not in evaluated_budgets:
                    evaluation = self._surrogate_metrics(run_id, train_indices, test_indices, budget)
                    evaluation["fit_seconds"] = float(fit_seconds)
                    if budget in self.evaluation_budgets:
                        region, recommendation = self._region_metrics(run_id, fold, seed, budget)
                        evaluation.update(region)
                        evaluation["recommendation_seconds"] = float(recommendation["seconds"])
                        evaluation["recommendation"] = recommendation
                    history["evaluations"].append(evaluation)
                    atomic_write_json(path, history)
                if budget == self.total_budget:
                    break
                if hasattr(self.session, "save_checkpoint"):
                    self.session.save_checkpoint(run_id, checkpoint, budget)

                candidates = np.asarray([index for index in eligible if index not in set(train_indices)], dtype=int)
                if not len(candidates):
                    raise RuntimeError("candidate pool exhausted before total budget")
                select_started = time.perf_counter()
                if method == "C1":
                    selected = int(lhs[budget])
                    diagnostics = {"design": "shared_maximin_lhs"}
                else:
                    candidate_xy = self.units.iloc[candidates][["x", "y"]].to_numpy(float)
                    local = self.session.select(
                        method, candidate_xy, self.reference_points, run_id,
                        budget - self.initial_budget + 1, seed,
                    )
                    selected = int(candidates[local])
                    diagnostics = self.session.last_diagnostics.get(run_id, {})
                selection_seconds = time.perf_counter() - select_started
                if selected in train_indices or selected not in eligible:
                    raise RuntimeError("acquisition returned an observed or held-out cell")
                train_indices.append(selected)
                history["training_indices"] = train_indices.copy()
                history["training_cell_ids"] = self.units.iloc[train_indices]["cell_id"].tolist()
                history["acquisitions"].append({
                    "from_budget": budget, "step": budget - self.initial_budget + 1,
                    "selected_index": selected, "selected_cell_id": self.units.iloc[selected]["cell_id"],
                    "seconds": float(selection_seconds), "diagnostics": diagnostics,
                })
                history["evaluations"][-1]["selection_seconds"] = float(selection_seconds)
                atomic_write_json(path, history)
                need_fit = True

            if self.sequential_steps == self.auc_steps:
                final = history["evaluations"][-1]
                auc_kwargs = {"initial_budget": self.initial_budget, "steps": self.auc_steps}
                final["hit_auc"] = sequential_auc(history["evaluations"], "hit", **auc_kwargs)
                final["rmae_aulc"] = sequential_auc(history["evaluations"], "rmae", **auc_kwargs)
                final["acc_auc"] = sequential_auc(history["evaluations"], "acc", **auc_kwargs)
            history["status"] = "COMPLETE"
            history["total_seconds"] = float(time.perf_counter() - started)
            atomic_write_json(path, history)
            self.session.clear(run_id)
            if checkpoint.exists():
                checkpoint.unlink()
            return history
        except Exception as exc:
            history["status"] = "FAILED"
            history["error"] = f"{type(exc).__name__}: {exc}"
            history["total_seconds"] = float(time.perf_counter() - started)
            atomic_write_json(path, history)
            self.session.clear(run_id)
            raise


def _parse_int_list(text):
    return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--methods", default="core", help="core, all, or comma-separated method names")
    parser.add_argument("--folds", default="0", help="single split identifier; random test set varies by seed")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--initial-budget", type=int)
    parser.add_argument("--sequential-steps", type=int)
    parser.add_argument("--evaluation-budgets")
    parser.add_argument("--knowledge-variant", default="original", choices=KNOWLEDGE_VARIANTS)
    parser.add_argument("--units-variant", default="original", choices=UNITS_VARIANTS)
    parser.add_argument("--output", type=Path)
    return parser


def resolve_budget_options(args, settings):
    initial_budget = (
        int(settings["protocol"]["initial_budget"])
        if args.initial_budget is None else int(args.initial_budget)
    )
    evaluation_budgets = (
        list(map(int, settings["evaluation_budgets"]))
        if args.evaluation_budgets is None else _parse_int_list(args.evaluation_budgets)
    )
    if initial_budget < 1:
        raise ValueError("initial budget must be positive")
    if not evaluation_budgets or any(value < 1 for value in evaluation_budgets):
        raise ValueError("evaluation budgets must be positive")
    return initial_budget, evaluation_budgets


def main():
    parser = build_parser()
    args = parser.parse_args()
    settings = load_experiment_config(args.grid)
    sequential_steps = (
        int(settings["protocol"]["sequential_steps"])
        if args.sequential_steps is None else args.sequential_steps
    )
    if sequential_steps < 1:
        parser.error("--sequential-steps must be positive")
    try:
        initial_budget, evaluation_budgets = resolve_budget_options(args, settings)
    except ValueError as exc:
        parser.error(str(exc))
    units = load_units(args.grid, variant=args.units_variant)
    knowledge = load_knowledge(args.grid, variant=args.knowledge_variant)
    construct, domain_width = load_domain()
    incidents, incidents_xy = load_incidents(construct)
    reference = qmc.Sobol(d=2, scramble=True, seed=20261024).random_base2(8)
    split_scheme = settings["protocol"]["split_scheme"]
    variant_suffixes = {
        "original": "", "adjacent": "_adjacent_bnd", "hotspot": "_hotspot_bnd",
        "oracle2022_threshold": "_oracle2022_threshold_bnd",
        "oracle2022_adjacent": "_oracle2022_adjacent_bnd",
        "oracle2022_hotspot_threshold": "_hotspot_oracle2022_threshold",
        "oracle2022_hotspot_adjacent": "_hotspot_oracle2022_adjacent",
        "hotspot_highside_adjacent": "_hotspot_highside_adjacent",
        "oracle2022_hotspot_highside_adjacent": "_hotspot_oracle2022_highside_adjacent",
        "misleading2022_hotspot_highside_inverted": "_hotspot_misleading2022_highside_inverted",
        "misleading2022_hotspot_highside_rotated180": "_hotspot_misleading2022_highside_rotated180",
    }
    variant_suffix = variant_suffixes[args.knowledge_variant]
    if args.units_variant != "original":
        variant_suffix += f"_{args.units_variant}_y2"
    output = args.output or ROOT / "results" / "formal" / (
        f"{args.grid}_budget{initial_budget + sequential_steps}_{split_scheme}{variant_suffix}"
    )

    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
        cell_area_km2=settings["grids"][args.grid]["cell_area_km2"],
    ) as session:
        runner = ExperimentRunner(
            session, units, incidents, incidents_xy, domain_width, output, grid=args.grid,
            initial_budget=initial_budget,
            sequential_steps=sequential_steps,
            evaluation_budgets=evaluation_budgets, reference_points=reference,
            auc_steps=settings["aggregation"]["sequential_auc_steps"],
            n_reference=settings["region_evaluation"]["n_reference"],
            reference_seed=settings["region_evaluation"]["seed"],
            split_scheme=split_scheme,
            test_fraction=settings["protocol"]["test_fraction"],
            knowledge_variant=args.knowledge_variant, units_variant=args.units_variant,
        )
        for fold in _parse_int_list(args.folds):
            for seed in _parse_int_list(args.seeds):
                for method in parse_methods(args.methods):
                    print(f"[Crimes] {method} fold={fold} seed={seed}", flush=True)
                    runner.run(method, fold, seed)


if __name__ == "__main__":
    main()
