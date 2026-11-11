from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OracleComparisonTests(unittest.TestCase):
    def test_summary_uses_population_std_over_eight_seeds(self):
        from compare_oracle2022_prior_variants import summarize
        frame = pd.DataFrame({
            "method": ["KARMA"] * 8,
            "knowledge_variant": ["oracle2022_threshold"] * 8,
            "seed": list(range(8)),
            "region_pass": [0.0] * 7 + [1.0],
        })
        result = summarize(frame, ["region_pass"])
        self.assertEqual(result.loc[0, "n_seeds"], 8)
        self.assertAlmostEqual(result.loc[0, "region_pass_mean"], 0.125)
        self.assertAlmostEqual(result.loc[0, "region_pass_std"], (7 / 64) ** 0.5)

    def test_karma_diagnostics_reads_alpha_and_gate(self):
        from compare_oracle2022_prior_variants import karma_diagnostics
        history = {"acquisitions": [
            {"diagnostics": {"pi_comp_alpha": [[0.8, 0.4]], "pi3_hist": [[0.8, 0.4, 0.8, 1.0]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.6, 0.2]], "pi3_hist": [[0.6, 0.2, 0.0, 0.0]]}},
        ]}
        result = karma_diagnostics(history)
        self.assertAlmostEqual(result["alpha_hg_mean"], 0.7)
        self.assertAlmostEqual(result["alpha_bnd_mean"], 0.3)
        self.assertAlmostEqual(result["alpha_bnd_final"], 0.2)
        self.assertEqual(result["gate_off_count"], 1)


if __name__ == "__main__":
    unittest.main()
