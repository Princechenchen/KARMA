from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smoke_lhs_oracle import OracleGridIntensity  


class OracleGridIntensityTests(unittest.TestCase):
    def test_cell_centers_use_xy_query_order(self):
        units = pd.DataFrame({
            "ix": [0, 1, 0, 1, 0, 1],
            "iy": [0, 0, 1, 1, 2, 2],
            "y1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })
        field = OracleGridIntensity(units)
        xy = np.column_stack([
            (units["ix"].to_numpy() + 0.5) / 2,
            (units["iy"].to_numpy() + 0.5) / 3,
        ])
        np.testing.assert_allclose(field(xy), units["y1"].to_numpy())


if __name__ == "__main__":
    unittest.main()
