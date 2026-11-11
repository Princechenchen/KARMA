import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_adjacent_boundary_knowledge import (
    build_adjacent_boundary_knowledge,
    tolerant_boundary_metrics,
)


class AdjacentBoundaryKnowledgeTest(unittest.TestCase):
    def test_marks_four_neighbor_changes_and_decays_with_grid_distance(self):
        rows = []
        labels = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ])
        for iy in range(3):
            for ix in range(3):
                rows.append({"cell_id": f"g_{ix}_{iy}", "ix": ix, "iy": iy, "y2": labels[iy, ix]})
        source = pd.DataFrame(rows)

        result = build_adjacent_boundary_knowledge(source, sigma_cells=1.0)

        expected_boundary = {
            (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2),
        }
        actual_boundary = set(
            result.loc[result["is_classification_boundary_2021"] == 1, ["ix", "iy"]]
            .itertuples(index=False, name=None)
        )
        self.assertEqual(actual_boundary, expected_boundary)
        self.assertTrue(np.allclose(
            result.loc[result["is_classification_boundary_2021"] == 1, "pi_bnd_adjacent"], 1.0
        ))
        non_boundary = result["is_classification_boundary_2021"] == 0
        self.assertTrue((result.loc[non_boundary, "pi_bnd_adjacent"] < 1.0).all())
        corner = result[(result["ix"] == 0) & (result["iy"] == 0)].iloc[0]
        self.assertEqual(corner["boundary_distance_2021"], 1.0)
        self.assertAlmostEqual(corner["pi_bnd_adjacent"], np.exp(-0.5))
        self.assertNotIn("is_classification_boundary_2021", source.columns)

    def test_boundary_match_treats_integer_labels_as_masks(self):
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]])
        actual = np.array([1, 1, 0, 0], dtype=int)
        predicted = np.array([0, 0, 1, 1], dtype=bool)

        result = tolerant_boundary_metrics(coordinates, actual, predicted, tolerance=1.0)

        self.assertEqual(result["actual_cells"], 2)
        self.assertEqual(result["predicted_cells"], 2)
        self.assertEqual(result["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
