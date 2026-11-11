from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smoke_mmrgp import select_mmrgp_training  


class SmokeMMRGPTests(unittest.TestCase):
    def test_training_selection_preserves_unique_cells_and_response_layout(self):
        units = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(20)],
            "x": np.linspace(0.05, 0.95, 20),
            "y": np.linspace(0.95, 0.05, 20),
            "y1_log1p": np.log1p(np.arange(20)),
            "y2": np.arange(20) % 2,
            "spatial_block": np.arange(20) % 5,
        })
        train = select_mmrgp_training(units, budget=8, seed=172)
        self.assertEqual(len(train), 8)
        self.assertTrue(train["cell_id"].is_unique)
        merged = train[["cell_id", "y1_log1p", "y2"]].merge(
            units[["cell_id", "y1_log1p", "y2"]], on="cell_id", suffixes=("", "_source"),
        )
        np.testing.assert_allclose(merged["y1_log1p"], merged["y1_log1p_source"])
        np.testing.assert_array_equal(merged["y2"], merged["y2_source"])


if __name__ == "__main__":
    unittest.main()
