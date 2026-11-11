import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_hotspot_experiment_data import build_hotspot_knowledge, build_hotspot_units


class HotspotExperimentDataTest(unittest.TestCase):
    def test_units_replace_response_and_boundary_but_preserve_original_columns(self):
        units = pd.DataFrame([
            {"cell_id": "a", "ix": 0, "iy": 0, "y1": 400, "y2": 1,
             "is_classification_boundary": 0, "y2_threshold": 1 / 3},
            {"cell_id": "b", "ix": 1, "iy": 0, "y1": 500, "y2": 0,
             "is_classification_boundary": 1, "y2_threshold": 1 / 3},
        ])

        result = build_hotspot_units(units, threshold=445.0)

        self.assertEqual(result["y2"].tolist(), [0, 1])
        self.assertEqual(result["is_classification_boundary"].tolist(), [1, 1])
        self.assertEqual(result["y2_violent_original"].tolist(), [1, 0])
        self.assertEqual(result["is_violent_boundary_original"].tolist(), [0, 1])
        self.assertEqual(result["y2_threshold"].unique().tolist(), [445.0])
        self.assertEqual(units["y2"].tolist(), [1, 0])

    def test_knowledge_replaces_only_boundary_channel(self):
        knowledge = pd.DataFrame({
            "cell_id": ["a", "b"], "pi_hg": [0.2, 0.4], "pi_bnd": [0.8, 0.9],
        })
        prior = pd.DataFrame({"cell_id": ["a", "b"], "pi_bnd_hot": [0.1, 0.7]})

        result = build_hotspot_knowledge(knowledge, prior)

        self.assertTrue(np.allclose(result["pi_hg"], knowledge["pi_hg"]))
        self.assertTrue(np.allclose(result["pi_bnd"], prior["pi_bnd_hot"]))
        self.assertTrue(np.allclose(result["pi_bnd_violent_original"], [0.8, 0.9]))
        self.assertTrue(np.allclose(knowledge["pi_bnd"], [0.8, 0.9]))


if __name__ == "__main__":
    unittest.main()
