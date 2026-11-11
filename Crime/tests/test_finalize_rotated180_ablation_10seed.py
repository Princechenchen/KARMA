import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "finalize_rotated180_ablation_10seed.py"


class FinalizeRotated180AblationTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(MODULE_PATH.exists(), "ablation finalizer is not implemented")
        return importlib.import_module("finalize_rotated180_ablation_10seed")

    def test_collects_four_paired_arms_over_exact_final_seeds(self):
        module = self.load_module()
        runs, alpha = module.collect_runs()
        self.assertEqual(len(runs), 40)
        self.assertEqual(len(alpha), 400)
        self.assertEqual(set(runs["method"]), {
            "KARMA-Aligned", "KARMA-Misleading-Adaptive",
            "KARMA-Misleading-Fixed", "ROI-D",
        })
        for seed, group in runs.groupby("seed"):
            self.assertEqual(len(group), 4)
            tests = np.stack(group["test_indices"].to_numpy())
            initial = np.stack(group["initial_indices"].to_numpy())
            self.assertTrue(np.all(tests == tests[0]))
            self.assertTrue(np.all(initial == initial[0]))

    def test_summary_matches_verified_aggregated_results(self):
        module = self.load_module()
        runs, _ = module.collect_runs()
        summary = module.summarize(runs)
        existing = pd.read_csv(
            ROOT / "results" / "aggregated" /
            "hotspot_misleading2022_highside_rotated180_10seeds" / "summary.csv",
        ).set_index("method")
        for row in summary.itertuples():
            for metric in module.METRICS:
                self.assertTrue(np.isclose(
                    getattr(row, f"{metric}_mean"), existing.loc[row.method, f"{metric}_mean"],
                    atol=1e-10, rtol=0,
                ))
                self.assertTrue(np.isclose(
                    getattr(row, f"{metric}_std"), existing.loc[row.method, f"{metric}_std"],
                    atol=1e-10, rtol=0,
                ))

    def test_saved_mat_and_markdown_are_complete_and_readable(self):
        module = self.load_module()
        self.assertTrue(module.MAT_PATH.exists(), "MAT bundle has not been generated")
        self.assertTrue(module.MD_PATH.exists(), "Markdown record has not been generated")
        validation = module.verify_saved_bundle()
        self.assertEqual(validation["runs"], 40)
        self.assertEqual(validation["alpha_rows"], 400)
        self.assertEqual(validation["target_cells"], 659)
        self.assertEqual(validation["prior_cells"], 657)
        self.assertEqual(validation["incidents"], 233049)
        text = module.MD_PATH.read_text(encoding="utf-8")
        for heading in ("实验背景", "实验设置", "对比方法", "先验设置", "实验结果", "结论"):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
