import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_oracle2022_boundary_knowledge import build_oracle2022_knowledge


class Oracle2022KnowledgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
        cls.oracle = build_oracle2022_knowledge()

    def test_support_pi_hg_and_fill_rule_are_controlled(self):
        self.assertEqual(len(self.oracle), 657)
        self.assertTrue(np.array_equal(self.original["cell_id"], self.oracle["cell_id"]))
        self.assertTrue(np.allclose(self.original[["x", "y"]], self.oracle[["x", "y"]]))
        self.assertTrue(np.allclose(self.original["pi_hg"], self.oracle["pi_hg"]))
        self.assertEqual(int(self.oracle["oracle2022_nearest_filled"].sum()), 8)
        columns = ["pi_bnd_threshold_2022", "pi_bnd_adjacent_2022"]
        self.assertTrue(np.isfinite(self.oracle[columns].to_numpy()).all())
        self.assertTrue(self.oracle[columns].apply(lambda s: s.between(0, 1)).all().all())

    def test_common_cells_match_the_reviewed_oracle_analysis(self):
        analysis = pd.read_csv(ROOT / "results" / "pi_bnd_2022_analysis" / "cell_level_fields.csv")
        paired = self.oracle.merge(
            analysis[["cell_id", "threshold_2022_oracle", "adjacent_2022_oracle"]],
            on="cell_id", how="inner", validate="one_to_one",
        )
        self.assertEqual(len(paired), 649)
        self.assertTrue(np.allclose(
            paired["pi_bnd_threshold_2022"], paired["threshold_2022_oracle"]
        ))
        self.assertTrue(np.allclose(
            paired["pi_bnd_adjacent_2022"], paired["adjacent_2022_oracle"]
        ))
        boundary = analysis["is_classification_boundary"].to_numpy(dtype=int)
        self.assertTrue(np.allclose(
            analysis.loc[boundary == 1, "adjacent_2022_oracle"], 1.0
        ))


if __name__ == "__main__":
    unittest.main()
