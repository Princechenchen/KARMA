from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommend_violent_regions import (  
    aggregate_metrics,
    run_recommendation,
    violent_intensity,
)


class FakeSession:
    def __init__(self):
        self.fitted = None
        self.cleared = None

    def fit(self, xy, y, run_id):
        self.fitted = (np.asarray(xy), np.asarray(y), run_id)

    def predict(self, run_id, xy):
        xy = np.asarray(xy, dtype=float)
        return {
            "mu_z": np.log1p(3.0 + xy[:, 0]),
            "var_z": np.full(len(xy), 0.2),
            "p": np.full(len(xy), 0.25),
            "fmu": np.zeros(len(xy)),
            "var_f": np.zeros(len(xy)),
        }

    def clear(self, run_id):
        self.cleared = run_id


class ViolentRecommendationTests(unittest.TestCase):
    def test_violent_intensity_multiplies_total_intensity_by_probability(self):
        session = FakeSession()
        points = np.array([[0.2, 0.4], [0.8, 0.6]])
        actual = violent_intensity(session, "run", points, cell_area_km2=2.0)
        expected = 0.25 * (np.exp(np.log1p(3.0 + points[:, 0]) + 0.1) - 1.0) / 2.0
        np.testing.assert_allclose(actual, expected)

    def test_run_recommendation_refits_saved_training_and_returns_four_metrics(self):
        history = {
            "status": "COMPLETE",
            "method": "KARMA",
            "fold": 0,
            "seed": 172,
            "run_id": "run",
            "total_budget": 2,
            "training_indices": [1, 0],
            "evaluations": [{
                "budget": 2,
                "recommendation": {
                    "seed": 222,
                    "n_reference": 20,
                    "reference_seed": 20261024,
                    "config": {
                        "k": 2, "particles": 8, "iterations": 2, "n_sobol": 16,
                    },
                },
            }],
        }
        units = pd.DataFrame({
            "x": [0.2, 0.8], "y": [0.3, 0.7],
            "y1_log1p": [1.0, 2.0], "y2": [0, 1],
        })
        incidents = pd.DataFrame({"is_violent": [True, False, True]})
        incidents_xy = np.array([[0.2, 0.3], [0.7, 0.8], [0.8, 0.7]])
        session = FakeSession()
        result = run_recommendation(
            session, history, units, incidents, incidents_xy,
            domain_width_km=(30.0, 40.0), cell_area_km2=1.0,
        )
        np.testing.assert_array_equal(session.fitted[0], [[0.8, 0.7], [0.2, 0.3]])
        np.testing.assert_array_equal(session.fitted[1], [[2.0, 1], [1.0, 0]])
        self.assertEqual(session.fitted[2], "run")
        self.assertEqual(session.cleared, "run")
        self.assertEqual(result["objective"], "lambda_all_times_p_violent")
        self.assertEqual(np.asarray(result["boxes"]).shape, (2, 4))
        for key in ("region_pass", "density_lift", "violent_region_pass", "violent_density_lift"):
            self.assertTrue(np.isfinite(result[key]))
        self.assertEqual(result["recommendation_seed"], 222)

        state_path = Path(__file__).resolve().parent / "tmp_gso_state.npz"
        partial = run_recommendation(
            session, history, units, incidents, incidents_xy,
            domain_width_km=(30.0, 40.0), cell_area_km2=1.0,
            state_path=state_path, max_steps=1,
        )
        self.assertEqual(partial["status"], "RUNNING")
        self.assertEqual(partial["iteration"], 1)
        resumed = run_recommendation(
            session, history, units, incidents, incidents_xy,
            domain_width_km=(30.0, 40.0), cell_area_km2=1.0,
            state_path=state_path,
        )
        self.assertEqual(resumed["status"], "COMPLETE")
        np.testing.assert_allclose(resumed["boxes"], result["boxes"])
        self.assertFalse(state_path.exists())

    def test_aggregate_metrics_uses_population_std(self):
        rows = [
            {"method": "KARMA", "region_pass": 0.0, "density_lift": 1.0,
             "violent_region_pass": 0.5, "violent_density_lift": 2.0},
            {"method": "KARMA", "region_pass": 1.0, "density_lift": 3.0,
             "violent_region_pass": 1.0, "violent_density_lift": 4.0},
        ]
        summary = aggregate_metrics(rows).iloc[0]
        self.assertEqual(summary["n_seeds"], 2)
        self.assertEqual(summary["region_pass_mean"], 0.5)
        self.assertEqual(summary["region_pass_std"], 0.5)
        self.assertEqual(summary["violent_density_lift_mean"], 3.0)
        self.assertEqual(summary["violent_density_lift_std"], 1.0)


if __name__ == "__main__":
    unittest.main()
