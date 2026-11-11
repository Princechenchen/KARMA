from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


class FakeMatlab:
    @staticmethod
    def double(value):
        return np.asarray(value, dtype=float)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.quit_calls = 0

    def addpath(self, path, nargout=0):
        self.calls.append(("addpath", str(path)))

    def mmrgp_set_knowledge(self, x, y, hg, bnd, nargout=0):
        self.calls.append(("knowledge", len(x)))

    def mmrgp_fit_crimes(self, x, y, class_pos, gpml, run_id, nargout=0):
        self.calls.append(("fit", np.asarray(x).shape[0], run_id))

    def mmrgp_select_next(self, method, x, y, candidate, reference, run_id, step, seed, gpml, nargout=1):
        self.calls.append(("select", method, np.asarray(candidate).shape[0], step, seed))
        return 2.0

    def mmrgp_get_diagnostics(self, run_id, nargout=1):
        self.calls.append(("diagnostics", run_id))
        return {"pi3_hist": np.array([[1.0, 1.0, 1.0, 1.0]])}

    def mmrgp_predict_all_latest(self, query, run_id, nargout=5):
        self.calls.append(("predict", run_id))
        n = len(query)
        return (
            np.full((n, 1), 4.0), np.full((n, 1), 0.2),
            np.full((n, 1), 0.7), np.full((n, 1), 0.3), np.full((n, 1), 0.1),
        )

    def mmrgp_predict_latest(self, query, run_id, nargout=2):
        self.calls.append(("predict_continuous", run_id))
        n = len(query)
        return np.full((n, 1), 4.0), np.full((n, 1), 0.2)

    def mmrgp_clear_run(self, run_id, nargout=0):
        self.calls.append(("clear", run_id))

    def mmrgp_load_checkpoint(self, run_id, path, nargout=1):
        self.calls.append(("load", run_id))
        return 2.0

    def quit(self):
        self.quit_calls += 1


class MMRGPExperimentTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        from mmrgp_experiment import MATLABExperimentSession
        self.session = MATLABExperimentSession(
            engine=self.engine,
            matlab_module=FakeMatlab(),
            gpml_root="gpml",
            knowledge={
                "x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]),
                "pi_hg": np.array([0.2, 0.4]), "pi_bnd": np.array([0.3, 0.5]),
            },
            cell_area_km2=1.0,
        )
        self.train_xy = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.train_y = np.array([[3.0, 0.0], [4.0, 1.0]])
        self.candidate = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        self.reference = np.array([[0.0, 0.0], [1.0, 1.0]])

    def test_fit_select_predict_use_real_matlab_contract(self):
        self.session.fit(self.train_xy, self.train_y, "run")
        picked = self.session.select("C2", self.candidate, self.reference, "run", 1, 172)
        self.assertEqual(picked, 1)
        pred = self.session.predict("run", self.candidate)
        self.assertEqual(set(pred), {"mu_z", "var_z", "p", "fmu", "var_f"})
        self.assertTrue(all(len(value) == 3 for value in pred.values()))
        self.assertEqual(
            [call[0] for call in self.engine.calls],
            ["addpath", "knowledge", "fit", "select", "diagnostics", "predict"],
        )
        self.assertIn("pi3_hist", self.session.last_diagnostics["run"])

    def test_matlab_double_diagnostics_become_builtin_lists(self):
        try:
            import matlab
        except ImportError:
            self.skipTest("MATLAB Engine Python package is unavailable")
        diagnostics = {"pi3_hist": matlab.double([[1.0, 0.5], [0.8, 0.25]])}
        converted = self.session._to_builtin(diagnostics)
        self.assertEqual(converted, {"pi3_hist": [[1.0, 0.5], [0.8, 0.25]]})
        self.assertIsInstance(converted["pi3_hist"], list)

    def test_intensity_uses_lognormal_moments(self):
        self.session.fit(self.train_xy, self.train_y, "run")
        actual = self.session.intensity("run", self.candidate[:1])
        self.assertAlmostEqual(actual[0], np.exp(4.1) - 1.0)
        self.assertIn(("predict_continuous", "run"), self.engine.calls)

    def test_restore_checkpoint_enables_prediction_dependencies(self):
        budget = self.session.restore_checkpoint(
            "run", "checkpoint.mat", self.train_xy, self.train_y,
        )
        self.assertEqual(budget, 2)
        self.session.predict("run", self.candidate[:1])
        addpaths = [call[1] for call in self.engine.calls if call[0] == "addpath"]
        self.assertTrue(any(path.endswith("MRSM_subprograms1") for path in addpaths))
        self.assertTrue(any(path.endswith("numeric_sample") for path in addpaths))
        self.assertIn(str(Path("gpml") / "util"), addpaths)

    def test_external_engine_is_not_quit(self):
        self.session.close()
        self.assertEqual(self.engine.quit_calls, 0)

    def test_invalid_training_is_rejected_before_fit(self):
        with self.assertRaises(ValueError):
            self.session.fit(self.train_xy, np.array([[3.0, 2.0], [4.0, 1.0]]), "run")

    def test_invalid_run_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.session.fit(self.train_xy, self.train_y, "bad-id")


if __name__ == "__main__":
    unittest.main()
