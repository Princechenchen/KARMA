"""Build a derived 2021 adjacency-boundary field and evaluate it against 2022.

The original knowledge artifacts are read-only. The derived field uses 2021 y2
only; 2022 is loaded exclusively for post hoc evaluation on common cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent


def build_adjacent_boundary_knowledge(source: pd.DataFrame, sigma_cells: float = 1.0) -> pd.DataFrame:
    """Return 2021 four-neighbor boundaries and a Gaussian distance field."""
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive")
    required = {"cell_id", "ix", "iy", "y2"}
    if not required.issubset(source.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(source.columns))}")
    if not source["cell_id"].is_unique:
        raise ValueError("cell_id must be unique")
    if source[list(required)].isna().any().any():
        raise ValueError("Boundary inputs must be complete")

    result = source.copy()
    lookup = {(row.ix, row.iy): int(row.y2) for row in result.itertuples()}
    boundary = np.zeros(len(result), dtype=np.int8)
    for row_index, row in enumerate(result.itertuples()):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_label = lookup.get((row.ix + dx, row.iy + dy))
            if neighbor_label is not None and neighbor_label != int(row.y2):
                boundary[row_index] = 1
                break

    coordinates = result[["ix", "iy"]].to_numpy(dtype=float)
    boundary_coordinates = coordinates[boundary == 1]
    if not len(boundary_coordinates):
        raise ValueError("No 2021 classification boundary exists")
    distance, _ = cKDTree(boundary_coordinates).query(coordinates, k=1)
    result["is_classification_boundary_2021"] = boundary
    result["boundary_distance_2021"] = distance
    result["pi_bnd_adjacent"] = np.exp(-(distance**2) / (2 * sigma_cells**2))
    return result


def ranking_metrics(actual: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(actual, score)),
        "average_precision": float(average_precision_score(actual, score)),
    }


def tolerant_boundary_metrics(
    coordinates: np.ndarray, actual: np.ndarray, predicted: np.ndarray, tolerance: float = 1.0
) -> dict:
    actual_coordinates = coordinates[np.asarray(actual, dtype=bool)]
    predicted_coordinates = coordinates[np.asarray(predicted, dtype=bool)]
    if not len(actual_coordinates) or not len(predicted_coordinates):
        raise ValueError("Boundary-F1 requires nonempty actual and predicted sets")
    predicted_hits = cKDTree(actual_coordinates).query(predicted_coordinates, k=1)[0] <= tolerance
    actual_hits = cKDTree(predicted_coordinates).query(actual_coordinates, k=1)[0] <= tolerance
    precision = float(predicted_hits.mean())
    recall = float(actual_hits.mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_grid_cells": tolerance,
        "predicted_cells": int(len(predicted_coordinates)),
        "actual_cells": int(len(actual_coordinates)),
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
    }


def main() -> None:
    source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    original_knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    adjacent = build_adjacent_boundary_knowledge(source, sigma_cells=1.0)
    added_columns = ["cell_id", "is_classification_boundary_2021", "boundary_distance_2021", "pi_bnd_adjacent"]
    derived = original_knowledge.merge(adjacent[added_columns], on="cell_id", how="left", validate="one_to_one")
    assert not derived[added_columns[1:]].isna().any().any()
    derived_path = ROOT / "data" / "knowledge" / "knowledge_2021_1km_adjacent.csv"
    derived.to_csv(derived_path, index=False)

    evaluated = target[["cell_id", "ix", "iy", "is_classification_boundary"]].merge(
        derived[["cell_id", "pi_bnd", "pi_bnd_adjacent", "boundary_distance_2021"]],
        on="cell_id", how="inner", validate="one_to_one",
    )
    actual = evaluated["is_classification_boundary"].to_numpy(dtype=int)
    coordinates = evaluated[["ix", "iy"]].to_numpy(dtype=float)
    assert 0 < actual.sum() < len(actual)
    predicted = evaluated["boundary_distance_2021"].to_numpy() <= 1.0

    result = {
        "scope": "Post hoc exploratory evaluation on common occupied cells; 2022 was not used to construct the field.",
        "construction": {
            "source_year": 2021,
            "threshold": "unchanged 2021 violent-ratio median",
            "boundary_rule": "any existing four-neighbor has different 2021 y2",
            "field": "exp(-distance_to_nearest_2021_boundary^2 / (2 * sigma_cells^2))",
            "sigma_cells": 1.0,
            "source_cells": int(len(source)),
            "source_boundary_cells": int(adjacent["is_classification_boundary_2021"].sum()),
        },
        "evaluation": {
            "common_cells": int(len(evaluated)),
            "excluded_target_only_cells": int(len(target) - len(evaluated)),
            "actual_2022_boundary_cells": int(actual.sum()),
            "actual_2022_boundary_share": float(actual.mean()),
            "original_pi_bnd": ranking_metrics(actual, evaluated["pi_bnd"].to_numpy()),
            "adjacent_pi_bnd": ranking_metrics(actual, evaluated["pi_bnd_adjacent"].to_numpy()),
            "adjacent_within_one_cell_boundary_match": tolerant_boundary_metrics(
                coordinates, actual, predicted, tolerance=1.0
            ),
        },
        "derived_file": str(derived_path.relative_to(ROOT)),
    }
    output = ROOT / "results" / "knowledge_adjacent_boundary_agreement_1km.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"DERIVED={derived_path}")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
