"""Build target-year Oracle boundary fields for the hotspot-response diagnosis.

The 2022 fields are for mechanism/upper-bound experiments only. The support,
pi_hg channel, source data, and all existing knowledge artifacts stay unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "knowledge" / "knowledge_2022_1km_hotspot_oracle.csv"
THRESHOLD = 445.0
TAU = 1.2171857664987304
VALUE_COLUMNS = ["pi_bnd_hotspot_threshold_2022", "pi_bnd_hotspot_adjacent_2022"]


def build_hotspot_oracle2022_knowledge() -> pd.DataFrame:
    target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km_hotspot.csv")
    original = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km_hotspot.csv")
    for frame in (target, original):
        if not frame["cell_id"].is_unique:
            raise ValueError("cell_id must be unique")
    if not np.array_equal(
        target["y2"].to_numpy(dtype=int),
        (target["y1"].to_numpy(dtype=float) > THRESHOLD).astype(int),
    ):
        raise ValueError("hotspot labels do not match the frozen 2021 threshold")

    adjacent = build_adjacent_boundary_knowledge(target, sigma_cells=1.0)
    if not np.array_equal(
        adjacent["is_classification_boundary_2021"].to_numpy(dtype=int),
        target["is_classification_boundary"].to_numpy(dtype=int),
    ):
        raise ValueError("2022 hotspot adjacency does not match the frozen boundary")

    values = target[["cell_id", "y1", "x", "y"]].copy()
    z = np.log1p(values["y1"].to_numpy(dtype=float))
    values["pi_bnd_hotspot_threshold_2022"] = np.clip(
        np.exp(-((z - np.log1p(THRESHOLD)) ** 2) / (2 * TAU**2)), 0.01, 1.0,
    )
    values["pi_bnd_hotspot_adjacent_2022"] = adjacent["pi_bnd_adjacent"].to_numpy(dtype=float)

    result = original.merge(
        values[["cell_id", *VALUE_COLUMNS]], on="cell_id", how="left",
        validate="one_to_one",
    )
    missing = result[VALUE_COLUMNS[0]].isna().to_numpy()
    if missing.sum() != 8:
        raise ValueError(f"expected 8 source-only support cells, found {missing.sum()}")
    if result.loc[~missing, VALUE_COLUMNS].isna().any().any():
        raise ValueError("incomplete Oracle values on common cells")
    nearest = cKDTree(target[["x", "y"]].to_numpy(dtype=float)).query(
        result.loc[missing, ["x", "y"]].to_numpy(dtype=float), k=1,
    )[1]
    for column in VALUE_COLUMNS:
        result.loc[missing, column] = values.iloc[nearest][column].to_numpy(dtype=float)
    result["oracle2022_nearest_filled"] = missing.astype(np.int8)
    result["oracle_hotspot_threshold"] = THRESHOLD
    result["oracle_hotspot_tau"] = TAU

    if len(result) != 657 or not np.isfinite(result[VALUE_COLUMNS].to_numpy()).all():
        raise ValueError("invalid hotspot Oracle support or non-finite values")
    if not result[VALUE_COLUMNS].apply(lambda series: series.between(0, 1)).all().all():
        raise ValueError("hotspot Oracle knowledge outside [0,1]")
    if not np.allclose(original["pi_hg"], result["pi_hg"]):
        raise ValueError("pi_hg changed during hotspot Oracle construction")
    return result


def main() -> None:
    result = build_hotspot_oracle2022_knowledge()
    result.to_csv(OUTPUT, index=False)
    print(json.dumps({
        "warning": "Target-year Oracle knowledge; not a deployable cross-year prior.",
        "support_cells": int(len(result)),
        "nearest_filled_cells": int(result["oracle2022_nearest_filled"].sum()),
        "threshold": THRESHOLD,
        "tau": TAU,
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
