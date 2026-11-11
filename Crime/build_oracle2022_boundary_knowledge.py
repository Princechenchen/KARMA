"""Build controlled 2022 Oracle boundary fields on the original 2021 support.

This target-year knowledge is only for mechanism/upper-bound experiments. The
original coordinates, pi_hg channel, and all source artifacts remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "knowledge" / "knowledge_2022_1km_oracle.csv"


def build_oracle2022_knowledge() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    original = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    for frame in (source, target, original):
        if not frame["cell_id"].is_unique:
            raise ValueError("cell_id must be unique")

    threshold = float(source["violent_ratio"].median())
    mad = float(np.median(np.abs(source["violent_ratio"] - threshold)))
    tau = max(1.4826 * mad, 0.05)
    if not np.isclose(threshold, 1.0 / 3.0):
        raise ValueError("unexpected 2021 classification threshold")

    adjacent = build_adjacent_boundary_knowledge(target, sigma_cells=1.0)
    if not np.array_equal(
        adjacent["is_classification_boundary_2021"].to_numpy(dtype=int),
        target["is_classification_boundary"].to_numpy(dtype=int),
    ):
        raise ValueError("2022 adjacent boundary does not match the frozen ground truth")

    target_values = target[["cell_id", "violent_ratio", "x", "y"]].copy()
    target_values["pi_bnd_threshold_2022"] = np.clip(
        np.exp(-((target_values["violent_ratio"] - threshold) ** 2) / (2 * tau**2)),
        0.01, 1.0,
    )
    target_values["pi_bnd_adjacent_2022"] = adjacent["pi_bnd_adjacent"].to_numpy(dtype=float)

    value_columns = ["pi_bnd_threshold_2022", "pi_bnd_adjacent_2022"]
    result = original.merge(
        target_values[["cell_id", *value_columns]], on="cell_id", how="left",
        validate="one_to_one",
    )
    missing = result["pi_bnd_threshold_2022"].isna().to_numpy()
    if missing.sum() != 8:
        raise ValueError(f"expected 8 source-only support cells, found {missing.sum()}")
    if result.loc[~missing, value_columns].isna().any().any():
        raise ValueError("incomplete Oracle values on common cells")

    nearest = cKDTree(target[["x", "y"]].to_numpy(dtype=float)).query(
        result.loc[missing, ["x", "y"]].to_numpy(dtype=float), k=1
    )[1]
    for column in value_columns:
        result.loc[missing, column] = target_values.iloc[nearest][column].to_numpy(dtype=float)
    result["oracle2022_nearest_filled"] = missing.astype(np.int8)

    if len(result) != 657 or not np.isfinite(result[value_columns].to_numpy()).all():
        raise ValueError("invalid Oracle knowledge support or non-finite values")
    if not result[value_columns].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Oracle knowledge outside [0,1]")
    if not np.allclose(original["pi_hg"], result["pi_hg"]):
        raise ValueError("pi_hg changed during Oracle construction")
    return result


def main() -> None:
    result = build_oracle2022_knowledge()
    result.to_csv(OUTPUT, index=False)
    report = {
        "warning": "Target-year Oracle knowledge; not a deployable cross-year prior.",
        "support_cells": int(len(result)),
        "direct_2022_cells": int((result["oracle2022_nearest_filled"] == 0).sum()),
        "nearest_filled_2021_only_cells": int(result["oracle2022_nearest_filled"].sum()),
        "pi_hg_and_support_unchanged": True,
        "output": str(OUTPUT),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
