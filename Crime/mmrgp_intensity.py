"""Persistent MATLAB Engine adapter for real mixed-response MMRGP intensity.

The model is trained once in MATLAB. Arbitrary xy queries are predicted in
bounded chunks and converted from log1p(count) moments to incidents/km² in
Python. MATLAB failures are propagated; no surrogate fallback is provided.
"""
from __future__ import annotations

import importlib
import operator
from pathlib import Path
import uuid

import numpy as np

from region_recommender import lognormal_intensity


class MMRGPIntensity:
    """Real MMRGP intensity on normalized two-dimensional spatial coordinates."""

    def __init__(
        self,
        train_xy: np.ndarray,
        train_y: np.ndarray,
        cell_area_km2: float,
        gpml_root: str | Path,
        *,
        chunk_size: int = 512,
        engine=None,
        project_root: str | Path | None = None,
    ):
        train_xy = np.asarray(train_xy, dtype=float)
        train_y = np.asarray(train_y, dtype=float)
        if train_xy.ndim != 2 or train_xy.shape[1] != 2:
            raise ValueError("train_xy must be a B-by-2 array")
        if train_y.shape != train_xy.shape:
            raise ValueError("train_y must be B-by-2 and aligned with train_xy")
        if len(train_xy) == 0 or not np.isfinite(train_xy).all() or not np.isfinite(train_y).all():
            raise ValueError("training arrays must be nonempty and finite")
        if not np.isin(train_y[:, 1], [0.0, 1.0]).all():
            raise ValueError("the second training response must be binary y2")
        if not np.isfinite(cell_area_km2) or cell_area_km2 <= 0:
            raise ValueError("cell_area_km2 must be positive and finite")
        try:
            chunk_size = operator.index(chunk_size)
        except TypeError as exc:
            raise TypeError("chunk_size must be an integer") from exc
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.cell_area_km2 = float(cell_area_km2)
        self.chunk_size = chunk_size
        self.training_rows = int(len(train_xy))
        self.model_id = uuid.uuid4().hex
        self.kernel = "correxp continuous/joint residual; covSEard classification latent"
        self.response_layout = "train_y=[log1p(y1), y2]; ClassPos=[1,2]"
        self.query_count = 0
        self.is_mmr_gp = False
        self._closed = False
        self._owns_engine = engine is None
        self.engine = engine
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        matlab_dir = root / "crimes-mixed-response" / "matlab"

        try:
            try:
                engine_api = importlib.import_module("matlab.engine")
                self._matlab = importlib.import_module("matlab")
            except ImportError as exc:
                raise ImportError(
                    "MATLAB Engine for Python is required; install a version compatible "
                    "with MATLAB and Python >= 3.10"
                ) from exc
            if self.engine is None:
                self.engine = engine_api.start_matlab()
            self.engine.addpath(str(matlab_dir), nargout=0)
            self.engine.mmrgp_fit_crimes(
                self._matlab.double(train_xy.tolist()),
                self._matlab.double(train_y.tolist()),
                self._matlab.double([1.0, 2.0]),
                str(gpml_root),
                self.model_id,
                nargout=0,
            )
            self.is_mmr_gp = True
        except Exception:
            self._closed = True
            if self._owns_engine and self.engine is not None:
                self.engine.quit()
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError("MMRGPIntensity is closed")
        if not self.is_mmr_gp:
            raise RuntimeError("MMRGP model is not fitted")

    def predict_moments(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and variance on the original log1p(y1) scale."""
        points = np.asarray(xy, dtype=float)
        self._require_open()
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("xy must be a finite N-by-2 array")
        if len(points) == 0:
            return np.empty(0), np.empty(0)

        mu_parts, var_parts = [], []
        for start in range(0, len(points), self.chunk_size):
            chunk = points[start:start + self.chunk_size]
            mu, variance = self.engine.mmrgp_predict_latest(
                self._matlab.double(chunk.tolist()), self.model_id, nargout=2,
            )
            mu = np.asarray(mu, dtype=float).reshape(-1)
            variance = np.asarray(variance, dtype=float).reshape(-1)
            if len(mu) != len(chunk) or len(variance) != len(chunk):
                raise RuntimeError("MATLAB returned an invalid moment length")
            if not np.isfinite(mu).all() or not np.isfinite(variance).all():
                raise RuntimeError("MATLAB returned non-finite moments")
            if np.any(variance < -1e-10):
                raise RuntimeError("MATLAB returned negative predictive variance")
            mu_parts.append(mu)
            var_parts.append(np.maximum(variance, 0.0))
        self.query_count += len(points)
        return np.concatenate(mu_parts), np.concatenate(var_parts)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        mu, variance = self.predict_moments(xy)
        return lognormal_intensity(mu, variance, self.cell_area_km2)

    def close(self):
        """Clear this model and quit only an Engine owned by this instance."""
        if self._closed:
            return
        try:
            self.engine.mmrgp_clear_model(self.model_id, nargout=0)
        finally:
            self._closed = True
            self.is_mmr_gp = False
            if self._owns_engine:
                self.engine.quit()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
