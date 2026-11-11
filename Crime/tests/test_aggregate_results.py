from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


class AggregationTests(unittest.TestCase):
    def test_std_is_over_seed_fold_means(self):
        from aggregate_results import seed_fold_summary
        frame = pd.DataFrame({
            "seed": [1, 1, 2, 2], "fold": [0, 1, 0, 1],
            "region_pass": [0.0, 1.0, 0.0, 0.0],
        })
        result = seed_fold_summary(frame, ["region_pass"])
        self.assertAlmostEqual(result.loc[0, "region_pass_mean"], 0.25)
        self.assertAlmostEqual(result.loc[0, "region_pass_std"], 0.25)
        self.assertEqual(result.loc[0, "n_seeds"], 2)

    def test_final_budget_uses_latest_common_budget(self):
        from aggregate_results import final_budget_summary
        frame = pd.DataFrame({
            "method": ["C1", "C1", "C1"],
            "seed": [1, 1, 1],
            "budget": [20, 40, 40],
            "region_pass": [0.0, 0.5, 1.0],
        })
        final_budget, result = final_budget_summary(frame, ["region_pass"])
        self.assertEqual(final_budget, 40)
        self.assertEqual(result.loc[0, "region_pass_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
