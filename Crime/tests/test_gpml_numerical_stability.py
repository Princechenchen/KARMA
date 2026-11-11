import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GPML_ROOT = ROOT.parent / "third_party" / "GPML"


class GPMLNumericalStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import matlab
            import matlab.engine
        except ImportError as exc:
            raise unittest.SkipTest("MATLAB Engine Python package is unavailable") from exc
        cls.matlab = matlab
        cls.engine = matlab.engine.start_matlab()
        cls.engine.addpath(str(GPML_ROOT / "cov"), nargout=0)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.quit()

    def test_extreme_ard_lengthscale_preserves_covariance_positivity(self):
        history = json.loads((ROOT / "results/formal/1km_budget50_random_holdout_hotspot/C3/fold_0/seed_158/history.json").read_text(encoding="utf-8"))
        units = np.genfromtxt(ROOT / "data/units/units_2022_1km_hotspot.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
        train = units[history["training_indices"]]
        train_x = np.column_stack([train["x"], train["y"]]).astype(float)
        hyp_cov = [-24.401061359231974, -4.932245720572075, 1.29194631568468]
        covariance = np.asarray(self.engine.covSEard(
            self.matlab.double(hyp_cov), self.matlab.double(train_x.tolist()), nargout=1,
        ), dtype=float)
        min_eigenvalue = np.linalg.eigvalsh((covariance + covariance.T) / 2).min()
        self.assertGreater(min_eigenvalue, -1e-8)


if __name__ == "__main__":
    unittest.main()
