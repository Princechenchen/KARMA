from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from region_recommender import RecommendationConfig, recommend_regions  
from resumable_gso import ResumableGSO  


def field(points):
    points = np.asarray(points)
    return 1.0 + 20.0 * np.exp(-((points[:, 0] - 0.3) ** 2 + (points[:, 1] - 0.7) ** 2) / 0.01)


class ResumableGSOTests(unittest.TestCase):
    def test_checkpoint_resume_matches_uninterrupted_search(self):
        config = RecommendationConfig(k=2, particles=12, iterations=6, n_sobol=32)
        expected = recommend_regions(field, (30.0, 40.0), config, seed=123)
        search = ResumableGSO(field, (30.0, 40.0), config, seed=123)
        for _ in range(2):
            search.step()
        state = search.state_dict()
        resumed = ResumableGSO.from_state(field, (30.0, 40.0), config, state)
        while resumed.iteration < config.iterations:
            resumed.step()
        actual = resumed.finalize()
        np.testing.assert_allclose(actual["boxes"], expected["boxes"])
        np.testing.assert_allclose(actual["scores"], expected["scores"])
        self.assertEqual(actual["evaluations"], expected["evaluations"])

    def test_checkpoint_captures_mid_iteration_progress(self):
        config = RecommendationConfig(k=2, particles=12, iterations=6, n_sobol=32)
        search = ResumableGSO(field, (30.0, 40.0), config, seed=123)
        search.step()
        state = search.state_dict()
        self.assertEqual(state["iteration"], 1)
        self.assertIn("rng_state", state)
        self.assertEqual(state["particles"].shape, (12, 4))
        self.assertEqual(state["luciferin"].shape, (12,))
        self.assertEqual(state["radius"].shape, (12,))


if __name__ == "__main__":
    unittest.main()
