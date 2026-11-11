from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SpatialValidityTests(unittest.TestCase):
    def setUp(self):
        from spatial_validity import GridBoxValidity
        rows = []
        for ix in range(3):
            for iy in range(3):
                if (ix, iy) == (1, 1):
                    continue
                rows.append({"cell_id": f"g1000_x{ix}_y{iy}", "x": 0.2 + 0.2 * ix, "y": 0.2 + 0.2 * iy})
        self.validity = GridBoxValidity(pd.DataFrame(rows))

    def test_box_rejects_every_intersection_with_invalid_cell(self):
        boxes = np.array([
            [0.2, 0.2, 0.1, 0.1],       
            [0.3, 0.3, 0.2, 0.2],       
            [0.3, 0.2, 0.4, 0.2],       
            [0.05, 0.2, 0.1, 0.1],      
        ])
        np.testing.assert_array_equal(self.validity(boxes), [True, False, True, False])

    def test_gso_and_matched_references_return_only_valid_boxes(self):
        from region_recommender import RecommendationConfig, make_matched_references, recommend_regions
        config = RecommendationConfig(k=2, particles=30, iterations=10, n_sobol=32, length_max=0.2)
        result = recommend_regions(
            lambda points: np.ones(len(points)), (3.0, 3.0), config,
            seed=172, box_valid_fn=self.validity,
        )
        self.assertTrue(result["success"])
        self.assertTrue(np.all(self.validity(result["boxes"])))
        references = make_matched_references(
            result["boxes"], n_reference=20, seed=20261024, box_valid_fn=self.validity,
        )
        self.assertEqual(references.shape, (2, 20, 4))
        self.assertTrue(np.all(self.validity(references.reshape(-1, 4))))

    def test_resumable_gso_preserves_mask_and_checkpoint_trajectory(self):
        from region_recommender import RecommendationConfig, recommend_regions
        from resumable_gso import ResumableGSO
        field = lambda points: np.ones(len(points))
        config = RecommendationConfig(k=2, particles=20, iterations=5, n_sobol=32, length_max=0.2)
        expected = recommend_regions(field, (3.0, 3.0), config, 172, self.validity)
        search = ResumableGSO(field, (3.0, 3.0), config, 172, self.validity)
        search.step()
        resumed = ResumableGSO.from_state(field, (3.0, 3.0), config, search.state_dict(), self.validity)
        while resumed.iteration < config.iterations:
            resumed.step()
        actual = resumed.finalize()
        np.testing.assert_allclose(actual["boxes"], expected["boxes"])
        self.assertTrue(np.all(self.validity(actual["boxes"])))

    def test_evaluation_rejects_invalid_recommended_box(self):
        from region_recommender import evaluate_recommendations
        with self.assertRaisesRegex(ValueError, "valid grid"):
            evaluate_recommendations(
                np.array([[0.3, 0.3, 0.2, 0.2]]), np.array([[0.2, 0.2]]),
                (3.0, 3.0), n_reference=10, box_valid_fn=self.validity,
            )


if __name__ == "__main__":
    unittest.main()
