import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_hotspot_y2_candidate import (
    build_hotspot_boundary_prior,
    build_hotspot_candidate,
    hotspot_threshold,
    label_agreement_metrics,
)


class HotspotY2CandidateTest(unittest.TestCase):
    def test_uses_2021_q3_strictly_and_marks_candidate_four_neighbor_boundaries(self):
        source = pd.DataFrame({"y1": [0, 0, 0, 100]})
        target = pd.DataFrame([
            {"cell_id": "a", "ix": 0, "iy": 0, "y1": 20},
            {"cell_id": "b", "ix": 1, "iy": 0, "y1": 31},
            {"cell_id": "c", "ix": 2, "iy": 0, "y1": 40},
        ])

        threshold = hotspot_threshold(source)
        result = build_hotspot_candidate(target, threshold)

        self.assertEqual(threshold, 25)
        self.assertEqual(result["y2_hot"].tolist(), [0, 1, 1])
        self.assertEqual(result["is_hotspot_boundary"].tolist(), [1, 1, 0])
        self.assertNotIn("y2_hot", target.columns)

    def test_threshold_prior_uses_log_scale_robust_spread_and_clipping(self):
        source = pd.DataFrame({
            "cell_id": ["low", "threshold", "high"],
            "ix": [0, 1, 2],
            "iy": [0, 0, 0],
            "y1": [0.0, 10.0, 100.0],
        })

        result = build_hotspot_boundary_prior(source, threshold=10.0)

        z = np.log1p(source["y1"].to_numpy())
        center = np.log1p(10.0)
        tau = max(1.4826 * np.median(np.abs(z - np.median(z))), 0.05)
        expected = np.clip(np.exp(-((z - center) ** 2) / (2 * tau**2)), 0.01, 1.0)
        self.assertTrue(np.allclose(result["pi_bnd_hot"], expected))
        self.assertAlmostEqual(result.loc[result["cell_id"] == "threshold", "pi_bnd_hot"].iloc[0], 1.0)
        self.assertEqual(result["prior_threshold_2021"].unique().tolist(), [10.0])
        self.assertTrue(np.allclose(result["prior_tau_2021"], tau))

    def test_label_agreement_reports_confusion_and_association(self):
        actual = np.array([1, 1, 0, 0])
        predicted = np.array([1, 0, 1, 0])

        result = label_agreement_metrics(actual, predicted)

        self.assertEqual(result["confusion_using_candidate_as_prediction"], {
            "tn": 1, "fp": 1, "fn": 1, "tp": 1,
        })
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["positive_f1"], 0.5)
        self.assertAlmostEqual(result["cohen_kappa"], 0.0)
        self.assertAlmostEqual(result["matthews_correlation"], 0.0)
        self.assertAlmostEqual(result["jaccard_positive"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
