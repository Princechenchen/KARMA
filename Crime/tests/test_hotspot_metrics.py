from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute_hotspot_metrics import evaluate_saved_run
from hotspot_metrics import hotspot_metrics, ndcg_at_fraction


class HotspotMetricTests(unittest.TestCase):
    def test_ndcg_is_one_for_perfect_high_intensity_ranking(self):
        truth = np.array([0.0, 1.0, 2.0, 3.0])
        intensity = np.array([0.1, 0.4, 0.8, 1.6])
        self.assertAlmostEqual(ndcg_at_fraction(truth, intensity, 0.5), 1.0)

    def test_hotspot_metrics_use_predicted_intensity_and_true_top_fraction(self):
        truth = np.array([0.0, 1.0, 2.0, 3.0])
        mu_z = np.array([3.0, 2.0, 1.0, 0.0])
        intensity = np.array([0.1, 0.4, 0.8, 1.6])
        result = hotspot_metrics(truth, mu_z, intensity, fraction=0.5)
        self.assertEqual(result["n_hotspot"], 2)
        self.assertAlmostEqual(result["ndcg_at_20"], 1.0)
        self.assertAlmostEqual(result["h_rmae_at_20"], np.sqrt(5.0))

    def test_metrics_reject_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            hotspot_metrics([1.0, 2.0], [1.0], [1.0, 2.0])

    def test_saved_run_uses_final_training_and_saved_test_indices(self):
        class FakeSession:
            def fit(self, xy, y, run_id):
                self.fitted = (np.asarray(xy), np.asarray(y), run_id)

            def predict(self, run_id, xy):
                return {
                    "mu_z": np.array([1.0, 0.0]),
                    "var_z": np.zeros(2),
                }

            def clear(self, run_id):
                self.cleared = run_id

        history = {
            "status": "COMPLETE", "method": "KARMA", "fold": 0, "seed": 172,
            "run_id": "run", "total_budget": 2,
            "training_indices": [0, 1], "test_indices": [2, 3],
        }
        units = pd.DataFrame({
            "x": [0.1, 0.2, 0.3, 0.4], "y": [0.2, 0.3, 0.4, 0.5],
            "y1_log1p": [0.0, 0.0, 1.0, 3.0], "y2": [0, 0, 0, 1],
        })
        session = FakeSession()
        result = evaluate_saved_run(session, history, units, 1.0, fraction=0.5)
        self.assertEqual(result["budget"], 2)
        self.assertEqual(result["n_test"], 2)
        self.assertEqual(result["n_hotspot"], 1)
        self.assertEqual(session.cleared, "run")
        np.testing.assert_array_equal(session.fitted[0], [[0.1, 0.2], [0.2, 0.3]])
        self.assertTrue(np.isfinite(result["ndcg_at_20"]))
        self.assertTrue(np.isfinite(result["h_rmae_at_20"]))


if __name__ == "__main__":
    unittest.main()
