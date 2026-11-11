import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import summarize_misleading_highside_3seeds as summary_module
from summarize_misleading_highside_3seeds import suppression_step


class MisleadingHighsideSummaryTests(unittest.TestCase):
    def test_rotated_study_configuration_uses_isolated_paths(self):
        self.assertTrue(hasattr(summary_module, "study_configuration"))
        config = summary_module.study_configuration("rotated180")
        self.assertIn("highside_rotated180", config["formal"].name)
        self.assertIn("highside_rotated180", config["valid"].name)
        self.assertIn("highside_rotated180", config["output"].name)

    def test_final10_configuration_uses_exact_final_seed_set(self):
        self.assertTrue(hasattr(summary_module, "FINAL_SEEDS"))
        config = summary_module.study_configuration("rotated180", seed_set="final10")
        self.assertEqual(
            config["seeds"], [34, 46, 77, 136, 160, 172, 180, 2159, 6790, 7416],
        )
        self.assertTrue(config["output"].name.endswith("_10seeds"))

    def test_suppression_step_returns_third_consecutive_low_weight_step(self):
        history = {"acquisitions": [
            {"diagnostics": {"pi_comp_alpha": [[0.8, 0.6]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.7, 0.2]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.6, 0.2]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.5, 0.1]]}},
        ]}
        self.assertEqual(suppression_step(history), 4)

    def test_suppression_step_rejects_short_or_interrupted_dips(self):
        history = {"acquisitions": [
            {"diagnostics": {"pi_comp_alpha": [[0.8, 0.2]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.7, 0.2]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.6, 0.4]]}},
            {"diagnostics": {"pi_comp_alpha": [[0.5, 0.1]]}},
        ]}
        self.assertIsNone(suppression_step(history))


if __name__ == "__main__":
    unittest.main()
