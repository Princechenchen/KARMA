"""Numerical equivalence and timing check for continuous-only MMRGP prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matlab
import numpy as np
import pandas as pd

from mmrgp_experiment import MATLABExperimentSession

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-query", type=int, default=51200)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    indices = np.asarray(history["training_indices"], dtype=int)
    train = units.iloc[indices]
    query = np.random.default_rng(20261028).random((args.n_query, 2))
    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
    ) as session:
        session.engine.addpath(session.engine.genpath(args.gpml_root), nargout=0)
        session.engine.addpath(str(ROOT.parent / "numeric_sample"), nargout=0)
        session.engine.addpath(str(ROOT.parent / "MRSM_subprograms1"), nargout=0)
        fit_budget = session.restore_checkpoint(
            "continuous_validation", args.checkpoint,
            train[["x", "y"]].to_numpy(float),
            train[["y1_log1p", "y2"]].to_numpy(float),
        )
        original = {"mu_z": [], "var_z": []}
        optimized = {"mu_z": [], "var_z": []}
        original_seconds = 0.0
        optimized_seconds = 0.0
        for start in range(0, len(query), session.chunk_size):
            chunk = query[start:start + session.chunk_size]
            values = matlab.double(chunk.tolist())
            started = time.perf_counter()
            full = session.engine.mmrgp_predict_all_latest(
                values, "continuous_validation", nargout=5,
            )
            original_seconds += time.perf_counter() - started
            started = time.perf_counter()
            reduced = session.engine.mmrgp_predict_latest(
                values, "continuous_validation", nargout=2,
            )
            optimized_seconds += time.perf_counter() - started
            original["mu_z"].append(np.asarray(full[0]).reshape(-1))
            original["var_z"].append(np.asarray(full[1]).reshape(-1))
            optimized["mu_z"].append(np.asarray(reduced[0]).reshape(-1))
            optimized["var_z"].append(np.asarray(reduced[1]).reshape(-1))
    original = {key: np.concatenate(value) for key, value in original.items()}
    optimized = {key: np.concatenate(value) for key, value in optimized.items()}
    result = {
        "status": "PASS",
        "fit_budget": fit_budget,
        "n_query": len(query),
        "chunk_size": session.chunk_size,
        "original_seconds": original_seconds,
        "optimized_seconds": optimized_seconds,
        "speedup": original_seconds / optimized_seconds,
        "max_abs_difference": {
            key: float(np.max(np.abs(original[key] - optimized[key])))
            for key in original
        },
        "all_finite": bool(all(np.isfinite(value).all() for value in optimized.values())),
    }
    assert result["all_finite"]
    assert all(value <= 1e-10 for value in result["max_abs_difference"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
