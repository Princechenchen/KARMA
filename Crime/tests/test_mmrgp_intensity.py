from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmrgp_intensity import MMRGPIntensity  
from region_recommender import lognormal_intensity  


class FakeEngine:
    def __init__(self, variance=0.25):
        self.variance = variance
        self.chunks = []
        self.closed = False
        self.invalid_length = False
        self.nonfinite_mu = False
        self.fit_raises = False

    def addpath(self, path, nargout=0):
        self.path = path

    def mmrgp_fit_crimes(self, x, y, class_pos, gpml_root, model_id, nargout=0):
        if self.fit_raises:
            raise RuntimeError("fit failed")
        self.fit = (
            np.asarray(x), np.asarray(y), np.asarray(class_pos), gpml_root, model_id,
        )

    def mmrgp_predict_latest(self, x, model_id, nargout=2):
        x = np.asarray(x, float)
        self.chunks.append(len(x))
        mu = x[:, 0] + 2 * x[:, 1]
        if self.nonfinite_mu:
            mu[0] = np.nan
        if self.invalid_length:
            mu = mu[:-1]
        return mu, np.full(len(x), self.variance)

    def mmrgp_clear_model(self, model_id, nargout=0):
        self.cleared = model_id

    def quit(self):
        self.closed = True


class MMRGPIntensityTests(unittest.TestCase):
    def setUp(self):
        self.external_engine = FakeEngine()
        self.owned_engine = FakeEngine()
        self.matlab = SimpleNamespace(
            double=lambda values: np.asarray(values, float),
            engine=SimpleNamespace(start_matlab=lambda: self.owned_engine),
        )
        self.patch = patch.dict(sys.modules, {
            "matlab": self.matlab, "matlab.engine": self.matlab.engine,
        })
        self.patch.start()
        self.train_x = np.array([[0.1, 0.2], [0.4, 0.5], [0.8, 0.7]])
        self.train_y = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    def tearDown(self):
        self.patch.stop()

    def make_field(self, engine="default", **kwargs):
        parameters = dict(chunk_size=2)
        parameters.update(kwargs)
        if engine == "default":
            engine = self.external_engine
        return MMRGPIntensity(
            self.train_x, self.train_y, 2.0, "fake-gpml",
            engine=engine, **parameters,
        )

    def test_fit_contract_chunking_order_and_intensity(self):
        field = self.make_field()
        x, y, class_pos, gpml_root, model_id = self.external_engine.fit
        np.testing.assert_allclose(x, self.train_x)
        np.testing.assert_allclose(y, self.train_y)
        np.testing.assert_array_equal(class_pos, [1, 2])
        self.assertEqual(gpml_root, "fake-gpml")
        self.assertRegex(model_id, r"^[0-9a-f]{32}$")
        self.assertTrue(field.is_mmr_gp)
        self.assertEqual(field.training_rows, 3)

        points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
        mu, variance = field.predict_moments(points)
        np.testing.assert_allclose(mu, points[:, 0] + 2 * points[:, 1])
        np.testing.assert_allclose(variance, 0.25)
        self.assertEqual(self.external_engine.chunks, [2, 2, 1])
        self.assertEqual(field.query_count, 5)
        np.testing.assert_allclose(field(points), lognormal_intensity(mu, variance, 2.0))

    def test_empty_query_returns_empty_without_matlab_call(self):
        field = self.make_field()
        mu, variance = field.predict_moments(np.empty((0, 2)))
        self.assertEqual(mu.shape, (0,))
        self.assertEqual(variance.shape, (0,))
        self.assertEqual(field(np.empty((0, 2))).shape, (0,))
        self.assertEqual(self.external_engine.chunks, [])

    def test_invalid_training_and_query_inputs_fail(self):
        bad_y = self.train_y.copy()
        bad_y[0, 1] = 2
        with self.assertRaises(ValueError):
            MMRGPIntensity(
                self.train_x, bad_y, 1.0, "fake", engine=self.external_engine,
            )
        field = self.make_field()
        with self.assertRaises(ValueError):
            field(np.array([0.1, 0.2]))
        with self.assertRaises(ValueError):
            field(np.array([[np.nan, 0.2]]))

    def test_invalid_matlab_outputs_fail_without_fallback(self):
        self.external_engine.invalid_length = True
        with self.assertRaisesRegex(RuntimeError, "length"):
            self.make_field()(np.array([[0.1, 0.2]]))
        self.external_engine.invalid_length = False
        self.external_engine.nonfinite_mu = True
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.make_field()(np.array([[0.1, 0.2]]))
        self.external_engine.nonfinite_mu = False
        self.external_engine.variance = -1e-8
        with self.assertRaisesRegex(RuntimeError, "negative"):
            self.make_field()(np.array([[0.1, 0.2]]))

    def test_tiny_negative_variance_is_clipped_to_zero(self):
        self.external_engine.variance = -1e-12
        _, variance = self.make_field().predict_moments(np.array([[0.1, 0.2]]))
        np.testing.assert_array_equal(variance, [0.0])

    def test_external_engine_is_cleared_but_not_quit(self):
        field = self.make_field()
        model_id = field.model_id
        field.close()
        field.close()
        self.assertEqual(self.external_engine.cleared, model_id)
        self.assertFalse(self.external_engine.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            field(np.array([[0.1, 0.2]]))

    def test_missing_engine_raises_clear_dependency_error(self):
        with patch.dict(sys.modules, {
            "matlab": SimpleNamespace(double=np.asarray), "matlab.engine": None,
        }):
            with self.assertRaisesRegex(ImportError, "MATLAB Engine"):
                self.make_field(engine=None)

    def test_owned_engine_is_quit_and_fit_failure_cleans_up(self):
        field = self.make_field(engine=None)
        field.close()
        self.assertTrue(self.owned_engine.closed)
        self.owned_engine.fit_raises = True
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.make_field(engine=None)
        self.assertTrue(self.owned_engine.closed)


if __name__ == "__main__":
    unittest.main()
