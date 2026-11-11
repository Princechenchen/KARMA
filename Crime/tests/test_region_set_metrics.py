from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RegionSetMetricTests(unittest.TestCase):
    def setUp(self):
        self.units = pd.DataFrame([
            {
                "cell_id": f"g1000_x{ix}_y{iy}",
                "x": (ix + 0.5) / 3,
                "y": (iy + 0.5) / 3,
                "y1": 9 - (3 * iy + ix),
            }
            for iy in range(3) for ix in range(3)
        ])
        self.boxes = np.array([
            [1 / 6, 1 / 6, 1 / 3, 1 / 3],
            [0.5, 1 / 6, 1 / 3, 1 / 3],
        ])

    def test_conditional_references_are_valid_and_avoid_previous_boxes(self):
        from region_recommender import rectangles_overlap
        from region_set_metrics import conditional_matched_references
        from spatial_validity import GridBoxValidity
        validity = GridBoxValidity(self.units)
        references = conditional_matched_references(
            self.boxes, 20, 172, validity,
        )
        self.assertEqual(references.shape, (2, 20, 4))
        self.assertTrue(validity(references.reshape(-1, 4)).all())
        np.testing.assert_allclose(references[0, 0, 2:], self.boxes[0, 2:])
        np.testing.assert_allclose(references[1, 0, 2:], self.boxes[1, 2:])
        previous = np.repeat(self.boxes[0][None, :], 20, axis=0)
        self.assertFalse(rectangles_overlap(references[1], previous).any())

    def test_conditional_region_pass_uses_conditional_reference_quantiles(self):
        from region_recommender import rectangle_counts
        from region_set_metrics import conditional_matched_references, conditional_region_metrics
        from spatial_validity import GridBoxValidity
        validity = GridBoxValidity(self.units)
        incidents = self.units[["x", "y"]].to_numpy(float)
        references = conditional_matched_references(self.boxes, 30, 172, validity)
        counts = rectangle_counts(incidents, references.reshape(-1, 4)).reshape(2, 30)
        expected_q3 = np.quantile(counts, 0.75, axis=1, method="linear")
        result = conditional_region_metrics(
            self.boxes, incidents, (3.0, 3.0), 30, 172, validity,
        )
        np.testing.assert_allclose(result["conditional_reference_q3"], expected_q3)
        recommended_counts = rectangle_counts(incidents, self.boxes)
        self.assertEqual(
            result["conditional_region_pass"],
            float(np.mean(recommended_counts > expected_q3)),
        )

    def test_topn_precision_uses_area_equivalent_n_and_average_ranks(self):
        from region_set_metrics import topn_grid_precision
        ranked = self.units.copy()
        ranked["y1"] = [9, 1, 8, 7, 6, 5, 4, 3, 2]
        result = topn_grid_precision(
            self.boxes, ranked, (3.0, 3.0), 1.0,
        )
        self.assertEqual(result["area_equivalent_n"], 2)
        self.assertEqual(result["covered_grid_count"], 2)
        self.assertEqual(result["topn_hit_count"], 1)
        self.assertEqual(result["topn_grid_precision"], 0.5)

        tied = self.units.copy()
        tied["y1"] = [5, 5, 4, 3, 2, 1, 0, 0, 0]
        tied_result = topn_grid_precision(
            self.boxes[:1], tied, (3.0, 3.0), 1.0,
        )
        self.assertEqual(tied_result["area_equivalent_n"], 1)
        self.assertEqual(tied_result["topn_hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
