from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RecomputeValidRegionTests(unittest.TestCase):
    def test_parser_accepts_hotspot_data_variants(self):
        from recompute_valid_regions import build_parser
        args = build_parser().parse_args([
            "--gpml-root", "dummy", "--knowledge-variant", "hotspot",
            "--units-variant", "hotspot",
        ])
        self.assertEqual(args.knowledge_variant, "hotspot")
        self.assertEqual(args.units_variant, "hotspot")

    def test_saved_run_recommendation_is_strictly_valid(self):
        from recompute_valid_regions import run_valid_recommendation
        from spatial_validity import GridBoxValidity
        rows = []
        for ix in range(3):
            for iy in range(3):
                if (ix, iy) != (1, 1):
                    rows.append({
                        "cell_id": f"g1000_x{ix}_y{iy}", "x": 0.2 + 0.2 * ix,
                        "y": 0.2 + 0.2 * iy, "y1_log1p": 1.0, "y2": 0,
                    })
        units = pd.DataFrame(rows)
        validity = GridBoxValidity(units)
        history = {
            "status": "COMPLETE", "method": "KARMA", "fold": 0, "seed": 172,
            "run_id": "test", "total_budget": 2, "training_indices": [0, 1],
            "evaluations": [{"budget": 2, "recommendation": {
                "seed": 222, "n_reference": 20, "reference_seed": 20261024,
                "config": {"k": 2, "particles": 12, "iterations": 3, "n_sobol": 32, "length_max": 0.2},
            }}],
        }

        class Session:
            def fit(self, xy, y, run_id):
                self.train_size = len(xy)

            def intensity(self, run_id, points):
                return np.ones(len(points))

            def clear(self, run_id):
                pass

        result = run_valid_recommendation(
            Session(), history, units, np.array([[0.2, 0.2]]), (3.0, 3.0), validity,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(np.all(validity(result["boxes"])))
        self.assertEqual(result["n_train"], 2)
        self.assertEqual(result["valid_reference_boxes"], 40)


if __name__ == "__main__":
    unittest.main()
