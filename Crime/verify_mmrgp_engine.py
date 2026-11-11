"""Minimal real-MATLAB MMRGP fit/predict/select check without region optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from acquisition import initial_design
from mmrgp_experiment import MATLABExperimentSession

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpml-root", required=True)
    args = parser.parse_args()
    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    eligible = np.flatnonzero(units["spatial_block"].to_numpy(int) != 0)
    selected = initial_design(units, eligible, 172)
    train = units.iloc[selected]
    test = units.iloc[eligible[:8]]
    candidates = units.iloc[eligible[8:24]][["x", "y"]].to_numpy(float)
    reference = qmc.Sobol(d=2, scramble=True, seed=20261024).random_base2(4)
    with MATLABExperimentSession(
        gpml_root=args.gpml_root,
        knowledge={key: knowledge[key].to_numpy(float) for key in ("x", "y", "pi_hg", "pi_bnd")},
    ) as session:
        session.fit(
            train[["x", "y"]].to_numpy(float),
            train[["y1_log1p", "y2"]].to_numpy(float),
            "verify_mmrgp",
        )
        prediction = session.predict("verify_mmrgp", test[["x", "y"]].to_numpy(float))
        intensity = session.intensity("verify_mmrgp", candidates[:3])
        picked = session.select("C2", candidates, reference, "verify_mmrgp", 1, 172)
    result = {
        "status": "PASS", "matlab_mmr_gp": True, "train_rows": len(train),
        "prediction_rows": len(prediction["mu_z"]),
        "all_predictions_finite": all(np.isfinite(value).all() for value in prediction.values()),
        "all_variances_nonnegative": bool(np.all(prediction["var_z"] >= 0) and np.all(prediction["var_f"] >= 0)),
        "probabilities_in_unit_interval": bool(np.all((prediction["p"] >= 0) & (prediction["p"] <= 1))),
        "intensity_finite_nonnegative": bool(np.all(np.isfinite(intensity)) and np.all(intensity >= 0)),
        "c2_selected_zero_based_candidate": int(picked),
        "mu_z_first8": prediction["mu_z"].tolist(),
        "var_z_first8": prediction["var_z"].tolist(),
        "p_first8": prediction["p"].tolist(),
    }
    assert all((result["all_predictions_finite"], result["all_variances_nonnegative"],
                result["probabilities_in_unit_interval"], result["intensity_finite_nonnegative"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
