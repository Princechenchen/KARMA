"""Historical-year spatial-kernel diagnostic for the Crimes continuous response.

This is a univariate GP smoothness diagnostic, not a replacement for MMRGP and
not a target-year result. It uses 2021 only, so no 2022 evaluation information
participates in kernel selection.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "kernel_check_2021.json"


def main():
    df = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    x = df[["x", "y"]].to_numpy(float)
    y = df["y1_log1p"].to_numpy(float)
    blocks = df["spatial_block"].to_numpy(int)
    specifications = {
        "gaussian_rbf": RBF(length_scale=0.15, length_scale_bounds=(0.02, 2.0)),
        "exponential_matern_0.5": Matern(length_scale=0.15, length_scale_bounds=(0.02, 2.0), nu=0.5),
    }
    results = []
    for name, base_kernel in specifications.items():
        records = []
        for fold in sorted(np.unique(blocks)):
            test = blocks == fold
            train = ~test
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(
                noise_level=0.05, noise_level_bounds=(1e-6, 2.0)
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                optimizer="fmin_l_bfgs_b",
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=20261024,
            )
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x[train], y[train])
            elapsed = time.perf_counter() - start
            pred = model.predict(x[test])
            error = pred - y[test]
            records.append({
                "fold": int(fold),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "mae_log1p": float(np.mean(np.abs(error))),
                "rmse_log1p": float(np.sqrt(np.mean(error**2))),
                "seconds": elapsed,
                "fitted_kernel": str(model.kernel_),
            })
            print(name, records[-1], flush=True)
        results.append({
            "kernel": name,
            "mae_mean_folds": float(np.mean([r["mae_log1p"] for r in records])),
            "mae_pooled": float(np.sum([r["mae_log1p"] * r["n_test"] for r in records]) / len(df)),
            "rmse_pooled": float(np.sqrt(np.sum([r["rmse_log1p"] ** 2 * r["n_test"] for r in records]) / len(df))),
            "seconds_total": float(np.sum([r["seconds"] for r in records])),
            "folds": records,
        })
    payload = {
        "scope": "2021-only univariate GP spatial-block diagnostic; not an MMRGP or 2022 result",
        "n_cells": len(df),
        "response": "log1p annual incident count",
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
