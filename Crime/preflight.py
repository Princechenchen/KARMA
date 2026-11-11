"""Validate frozen Crimes data/code readiness without fabricating MATLAB readiness."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from acquisition import METHODS, lhs_sequence
from run_experiments import SEEDS, _json_safe, atomic_write_json, load_experiment_config

ROOT = Path(__file__).resolve().parent
REQUIRED_MATLAB = [
    "MMRGP0_crimes.m", "mmrgp_fit_crimes.m", "mmrgp_predict_all_latest.m",
    "mmrgp_select_next.m", "mmrgp_get_diagnostics.m", "mmrgp_clear_run.m",
    "mmrgp_save_checkpoint.m", "mmrgp_load_checkpoint.m",
    "mmrgp_set_knowledge.m", "crimes_knowledge_eval.m", "PALROI_crimes.m",
    "pal_region_classifier.py",
]
REQUIRED_PROJECT_MATLAB = [
    "numeric_sample/mMeanMSE_norm.m",
    "numeric_sample/mIMSE_refset.m",
    "numeric_sample/mAK_MCS_IRS_unified.m",
    "numeric_sample/mGlobal_SUR2w_varcs02lPD_fast_QBC.m",
    "sampling_Funciton/Voronoi_MixedMetric.m",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight(grid="1km", gpml_root=None):
    settings = load_experiment_config(grid)
    summary = json.loads((ROOT / "data" / "dataset_summary.json").read_text(encoding="utf-8"))
    units_path = ROOT / settings["grids"][grid]["units"]
    knowledge_path = ROOT / settings["grids"][grid]["knowledge"]
    units = pd.read_csv(units_path)
    knowledge = pd.read_csv(knowledge_path)
    required = {
        "cell_id", "x", "y", "y1", "y1_log1p", "y2", "spatial_block", "is_key_q3",
        "key_component_id", "is_classification_boundary", "ix", "iy",
    }
    issues, warnings = [], []
    missing = required.difference(units.columns)
    if missing:
        issues.append(f"missing unit columns: {sorted(missing)}")
    if not units["cell_id"].is_unique:
        issues.append("cell_id is not unique")
    if not np.isin(units["y2"], [0, 1]).all():
        issues.append("y2 is not binary")
    if not np.allclose(units["y1_log1p"], np.log1p(units["y1"])):
        issues.append("y1_log1p does not match log1p(y1)")
    if set(units["spatial_block"].unique()) != set(range(5)):
        issues.append("spatial blocks are not exactly 0..4")
    q3 = float(units["y1"].quantile(0.75))
    if not np.array_equal(units["is_key_q3"].to_numpy(bool), units["y1"].to_numpy() > q3):
        issues.append("is_key_q3 differs from strict target-year Q3 rule")
    scale_summary = summary["scales"][grid]
    if q3 != scale_summary["high_crime_q3_2022"]:
        issues.append("target Q3 differs from dataset_summary.json")
    if len(knowledge) == 0 or not {"x", "y", "pi_hg", "pi_bnd"}.issubset(knowledge.columns):
        issues.append("knowledge table is empty or missing required columns")
    else:
        values = knowledge[["x", "y", "pi_hg", "pi_bnd"]].to_numpy(float)
        if not np.isfinite(values).all():
            issues.append("knowledge contains non-finite values")
        if np.any((knowledge[["pi_hg", "pi_bnd"]].to_numpy(float) < 0) |
                  (knowledge[["pi_hg", "pi_bnd"]].to_numpy(float) > 1)):
            issues.append("knowledge values are outside [0,1]")

    initial_budget = int(settings["protocol"]["initial_budget"])
    total_budget = int(settings["protocol"]["total_budget"])
    fold_checks, single_class = [], []
    for fold in range(5):
        eligible = np.flatnonzero(units["spatial_block"].to_numpy(int) != fold)
        for seed in SEEDS:
            try:
                sequence = lhs_sequence(units, eligible, seed, total=total_budget)
                initial = sequence[:initial_budget]
                valid = (
                    len(sequence) == total_budget
                    and len(np.unique(sequence)) == total_budget
                    and np.all(units.iloc[sequence]["spatial_block"].to_numpy(int) != fold)
                )
                classes = sorted(units.iloc[initial]["y2"].unique().tolist())
                if classes != [0, 1]:
                    single_class.append({"fold": fold, "seed": seed, "classes": classes})
                fold_checks.append({"fold": fold, "seed": seed, "valid": bool(valid), "initial_classes": classes})
                if not valid:
                    issues.append(f"invalid LHS design fold={fold} seed={seed}")
            except Exception as exc:
                issues.append(f"LHS design failed fold={fold} seed={seed}: {exc}")
    if single_class:
        warnings.append(f"{len(single_class)} initial designs contain only one y2 class; frozen protocol was not altered")

    missing_matlab = [name for name in REQUIRED_MATLAB if not (ROOT / "matlab" / name).exists()]
    if missing_matlab:
        issues.append(f"missing MATLAB adapters: {missing_matlab}")
    missing_original = [name for name in REQUIRED_PROJECT_MATLAB if not (ROOT.parent / name).exists()]
    if missing_original:
        issues.append(f"missing original project samplers: {missing_original}")
    for year in (2021, 2022):
        record = summary["years"][str(year)]
        raw_path = ROOT / "data" / "raw" / Path(record["raw_file"]).name
        if not raw_path.exists():
            issues.append(f"missing raw data: {raw_path.name}")
        elif sha256(raw_path) != record["raw_sha256"]:
            issues.append(f"raw SHA256 mismatch: {raw_path.name}")
        processed = ROOT / "data" / "processed" / f"incidents_{year}_clean.csv.gz"
        if not processed.exists():
            issues.append(f"missing processed incidents: {processed.name}")
        else:
            rows = len(pd.read_csv(processed, usecols=["is_violent"]))
            if rows != record["valid_rows"]:
                issues.append(f"processed row count mismatch for {year}")

    matlab_executable = shutil.which("matlab") or shutil.which("matlab.exe")
    try:
        engine_available = importlib.util.find_spec("matlab.engine") is not None
    except ModuleNotFoundError:
        engine_available = False
    gpml_valid = None
    if gpml_root is not None:
        gpml_valid = Path(gpml_root).is_dir() and (Path(gpml_root) / "gp.m").exists()
        if not gpml_valid:
            issues.append("provided GPML root is invalid")
    engine_start_verified = False
    if engine_available and gpml_valid:
        try:
            import matlab.engine
            engine = matlab.engine.start_matlab()
            try:
                engine_start_verified = engine.eval("1+2", nargout=1) == 3
            finally:
                engine.quit()
        except Exception as exc:
            issues.append(f"MATLAB Engine startup failed: {exc}")
    numerical_ready = bool(engine_start_verified and gpml_valid)
    if gpml_root is None:
        warnings.append("GPML root was not supplied; numerical readiness remains unconfirmed")
    if not matlab_executable:
        warnings.append("MATLAB is not on PATH; this does not prevent Engine use")
    if not engine_available:
        warnings.append("MATLAB Engine for Python is unavailable")
    return {
        "grid": grid, "data_code_ready": not issues,
        "real_matlab_numerical_ready": numerical_ready,
        "matlab_executable": matlab_executable, "matlab_engine_available": engine_available,
        "engine_start_verified": engine_start_verified,
        "gpml_root_valid": gpml_valid, "methods": METHODS,
        "n_units": len(units), "target_q3": q3,
        "initial_budget": initial_budget, "total_budget": total_budget,
        "classification_threshold_2021": scale_summary["classification_threshold"],
        "target_positive_cells": int(units["y2"].sum()),
        "target_negative_cells": int((1 - units["y2"]).sum()),
        "fold_seed_designs_checked": len(fold_checks),
        "single_class_initial_designs": single_class,
        "issues": issues, "warnings": warnings,
        "numerical_boundary": "No RegionPass/DensityLift from MMRGP is inferred by this preflight.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="1km", choices=["1km", "0.5km"])
    parser.add_argument("--gpml-root")
    parser.add_argument("--require-matlab", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.grid, args.gpml_root)
    output = args.output or ROOT / "results" / f"preflight_{args.grid}.json"
    atomic_write_json(output, _json_safe(report))
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    if not report["data_code_ready"]:
        raise SystemExit(2)
    if args.require_matlab and not report["real_matlab_numerical_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
