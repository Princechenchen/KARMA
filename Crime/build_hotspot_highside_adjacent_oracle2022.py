"""Build the 2022 high-side adjacency Oracle for hotspot-response diagnosis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "knowledge" / "knowledge_2022_1km_hotspot_oracle.csv"
SIGMA_CELLS = 1.0


def build_hotspot_highside_adjacent_oracle2022() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km_hotspot.csv")
    original = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2022_1km_hotspot_oracle.csv")
    original = original.drop(columns=[
        "hotspot_side_probability_2022", "pi_bnd_highside_adjacent_2022",
        "pi_bnd_highside_inverted_2022", "pi_bnd_highside_rotated180_2022",
        "is_hotspot_boundary_2022",
    ], errors="ignore")
    if not source["cell_id"].is_unique or not original["cell_id"].is_unique:
        raise ValueError("cell_id must be unique")
    adjacent = build_adjacent_boundary_knowledge(source, sigma_cells=SIGMA_CELLS)
    coordinates = source[["ix", "iy"]].to_numpy(dtype=float)
    weights = np.exp(-(cdist(coordinates, coordinates) ** 2) / (2 * SIGMA_CELLS**2))
    probability = weights @ source["y2"].to_numpy(dtype=float) / weights.sum(axis=1)
    boundary_decay = adjacent["pi_bnd_adjacent"].to_numpy(dtype=float)

    derived = source[["cell_id"]].copy()
    derived["hotspot_side_probability_2022"] = probability
    derived["pi_bnd_highside_adjacent_2022"] = boundary_decay * probability
    derived["pi_bnd_highside_inverted_2022"] = boundary_decay * (1.0 - probability)
    derived["is_hotspot_boundary_2022"] = adjacent["is_classification_boundary_2021"]
    result = original.merge(derived, on="cell_id", how="left", validate="one_to_one")
    columns = [
        "hotspot_side_probability_2022", "pi_bnd_highside_adjacent_2022",
        "pi_bnd_highside_inverted_2022",
    ]
    missing = result[columns[0]].isna().to_numpy()
    if missing.sum() != 8 or result.loc[~missing, columns].isna().any().any():
        raise ValueError("unexpected 2022 high-side adjacency support")
    nearest = cKDTree(source[["x", "y"]].to_numpy(dtype=float)).query(
        result.loc[missing, ["x", "y"]].to_numpy(dtype=float), k=1,
    )[1]
    for column in columns:
        result.loc[missing, column] = derived.iloc[nearest][column].to_numpy(dtype=float)
    result.loc[missing, "is_hotspot_boundary_2022"] = derived.iloc[nearest]["is_hotspot_boundary_2022"].to_numpy(dtype=float)
    if len(result) != 657 or not np.isfinite(result[columns].to_numpy()).all():
        raise ValueError("incomplete or non-finite 2022 high-side adjacency support")
    for column in columns[1:]:
        if not result[column].between(0, 1).all():
            raise ValueError(f"2022 high-side knowledge values outside [0,1]: {column}")
    if not np.allclose(original["pi_hg"], result["pi_hg"]):
        raise ValueError("pi_hg changed during Oracle construction")

    support_xy = result[["x", "y"]].to_numpy(dtype=float)
    rotated_nearest = cKDTree(support_xy).query(1.0 - support_xy, k=1)[1]
    result["pi_bnd_highside_rotated180_2022"] = result[
        "pi_bnd_highside_adjacent_2022"
    ].to_numpy(dtype=float)[rotated_nearest]
    return result


def main() -> None:
    result = build_hotspot_highside_adjacent_oracle2022()
    result.to_csv(OUTPUT, index=False)
    print(json.dumps({
        "source_year": 2022,
        "oracle": True,
        "sigma_cells": SIGMA_CELLS,
        "support_cells": int(len(result)),
        "boundary_cells": int(result["is_hotspot_boundary_2022"].sum()),
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
