from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


class ExperimentMetricTests(unittest.TestCase):
    def test_classification_metrics_handle_imbalance(self):
        from experiment_metrics import classification_metrics
        m = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.4]))
        self.assertAlmostEqual(m["acc"], 0.5)
        self.assertAlmostEqual(m["balanced_acc"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_sequential_auc_excludes_initial_budget(self):
        from experiment_metrics import sequential_auc
        history = [{"budget": 20, "value": 100.0}] + [
            {"budget": 20 + i, "value": 1.0} for i in range(1, 21)
        ]
        self.assertEqual(sequential_auc(history, "value", initial_budget=20, steps=20), 1.0)
        with self.assertRaises(ValueError):
            sequential_auc(history[:-1], "value", initial_budget=20, steps=20)

    def test_region_supplements_use_grid_truth_only_offline(self):
        from experiment_metrics import region_supplements
        units = pd.DataFrame({
            "x": [0.25, 0.75], "y": [0.25, 0.75],
            "is_key_q3": [1, 0], "key_component_id": [3, -1], "y2": [1, 0],
        })
        boxes = np.array([[0.25, 0.25, 0.5, 0.5]])
        result = region_supplements(boxes, units, violent_share=0.25)
        self.assertEqual(result["high_crime_coverage"], 1.0)
        self.assertEqual(result["component_coverage"], 1.0)
        self.assertEqual(result["violent_label_share"], 1.0)
        self.assertEqual(result["violent_label_uplift"], 0.75)


if __name__ == "__main__":
    unittest.main()
