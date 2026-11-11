"""Build a deployable 2021 high-side adjacency prior for hotspot responses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "knowledge" / "knowledge_2021_1km_hotspot_highside_adjacent.csv"
THRESHOLD = 445.0
SIGMA_CELLS = 1.0


def build_hotspot_highside_adjacent_knowledge() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    original = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km_hotspot.csv")
    if not source["cell_id"].is_unique or not original["cell_id"].is_unique:
        raise ValueError("cell_id must be unique")
    source = source.copy()
    source["y2"] = (source["y1"].to_numpy(dtype=float) > THRESHOLD).astype("int8")
    adjacent = build_adjacent_boundary_knowledge(source, sigma_cells=SIGMA_CELLS)

    coordinates = source[["ix", "iy"]].to_numpy(dtype=float)
    weights = np.exp(-(cdist(coordinates, coordinates) ** 2) / (2 * SIGMA_CELLS**2))
    probability = weights @ source["y2"].to_numpy(dtype=float) / weights.sum(axis=1)
    boundary = adjacent["pi_bnd_adjacent"].to_numpy(dtype=float)
    highside = boundary * probability

    derived = source[["cell_id"]].copy()
    derived["hotspot_side_probability"] = probability
    derived["pi_bnd_adjacent_hotspot"] = boundary
    derived["pi_bnd_highside_adjacent"] = highside
    derived["is_hotspot_boundary_2021"] = adjacent["is_classification_boundary_2021"]
    result = original.merge(derived, on="cell_id", how="left", validate="one_to_one")
    if len(result) != 657 or result[["hotspot_side_probability", "pi_bnd_highside_adjacent"]].isna().any().any():
        raise ValueError("incomplete high-side adjacency support")
    if not np.isfinite(result["pi_bnd_highside_adjacent"]).all():
        raise ValueError("non-finite high-side adjacency values")
    if not result["pi_bnd_highside_adjacent"].between(0, 1).all():
        raise ValueError("high-side adjacency values outside [0,1]")
    if not np.allclose(original["pi_hg"], result["pi_hg"]):
        raise ValueError("pi_hg changed during high-side prior construction")
    return result


def main() -> None:
    result = build_hotspot_highside_adjacent_knowledge()
    result.to_csv(OUTPUT, index=False)
    print(json.dumps({
        "source_year": 2021,
        "threshold": THRESHOLD,
        "sigma_cells": SIGMA_CELLS,
        "support_cells": int(len(result)),
        "boundary_cells": int(result["is_hotspot_boundary_2021"].sum()),
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
