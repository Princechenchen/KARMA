"""Persistent real-MATLAB session shared by all formal Crimes methods."""
from __future__ import annotations

import importlib
from pathlib import Path
import re

import numpy as np

from region_recommender import lognormal_intensity

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r"^[A-Za-z0-9_]+$")


class MATLABExperimentSession:
    """Own fitted MMRGP models and sampler state without numerical fallback."""

    def __init__(
        self, gpml_root, knowledge, cell_area_km2=1.0, *, engine=None,
        matlab_module=None, chunk_size=1024,
    ):
        self.gpml_root = str(gpml_root)
        self.cell_area_km2 = float(cell_area_km2)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0 or self.cell_area_km2 <= 0:
            raise ValueError("chunk_size and cell_area_km2 must be positive")
        self._owns_engine = engine is None
        self._closed = False
        self._training = {}
        self.last_diagnostics = {}
        if engine is None or matlab_module is None:
            try:
                engine_api = importlib.import_module("matlab.engine")
                matlab_module = matlab_module or importlib.import_module("matlab")
            except ImportError as exc:
                raise ImportError(
                    "MATLAB Engine for Python is required; install a version compatible "
                    "with MATLAB and Python >= 3.10"
                ) from exc
            engine = engine or engine_api.start_matlab()
        self.engine = engine
        self._matlab = matlab_module
        self.engine.addpath(str(ROOT / "matlab"), nargout=0)
        arrays = [np.asarray(knowledge[key], dtype=float).reshape(-1, 1)
                  for key in ("x", "y", "pi_hg", "pi_bnd")]
        if not all(len(arrays[0]) == len(a) for a in arrays[1:]):
            raise ValueError("knowledge arrays must have equal lengths")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("knowledge arrays must be finite")
        self.engine.mmrgp_set_knowledge(
            *(self._matlab.double(a.tolist()) for a in arrays), nargout=0,
        )

    @staticmethod
    def _run_id(run_id):
        if not RUN_ID.fullmatch(str(run_id)):
            raise ValueError("run_id must contain only letters, digits, and underscores")
        return str(run_id)

    def fit(self, train_xy, train_y, run_id):
        run_id = self._run_id(run_id)
        xy = np.asarray(train_xy, dtype=float)
        y = np.asarray(train_y, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or y.shape != xy.shape or len(xy) == 0:
            raise ValueError("train arrays must be nonempty B-by-2 arrays")
        if not np.isfinite(xy).all() or not np.isfinite(y).all():
            raise ValueError("training arrays must be finite")
        if not np.isin(y[:, 1], [0, 1]).all():
            raise ValueError("the second response must be binary")
        self.engine.mmrgp_fit_crimes(
            self._matlab.double(xy.tolist()), self._matlab.double(y.tolist()),
            self._matlab.double([1, 2]), self.gpml_root, run_id, nargout=0,
        )
        self._training[run_id] = (xy.copy(), y.copy())

    def select(self, method, candidate_xy, reference_xy, run_id, step, seed):
        run_id = self._run_id(run_id)
        if run_id not in self._training:
            raise RuntimeError("fit must be called before select")
        candidate = np.asarray(candidate_xy, dtype=float)
        reference = np.asarray(reference_xy, dtype=float)
        if candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) == 0:
            raise ValueError("candidate_xy must be a nonempty N-by-2 array")
        train_x, train_y = self._training[run_id]
        picked = int(self.engine.mmrgp_select_next(
            str(method), self._matlab.double(train_x.tolist()), self._matlab.double(train_y.tolist()),
            self._matlab.double(candidate.tolist()), self._matlab.double(reference.tolist()),
            run_id, int(step), int(seed), self.gpml_root, nargout=1,
        )) - 1
        if not 0 <= picked < len(candidate):
            raise RuntimeError(f"MATLAB returned invalid candidate index {picked + 1}")
        diagnostics = self.engine.mmrgp_get_diagnostics(run_id, nargout=1)
        self.last_diagnostics[run_id] = self._to_builtin(diagnostics)
        return picked

    @classmethod
    def _to_builtin(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._to_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_builtin(item) for item in value]
        if isinstance(value, np.ndarray):
            return cls._to_builtin(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if type(value).__module__ == "matlab":
            return cls._to_builtin(np.asarray(value).tolist())
        return value

    def predict(self, run_id, query_xy):
        run_id = self._run_id(run_id)
        query = np.asarray(query_xy, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2 or not np.isfinite(query).all():
            raise ValueError("query_xy must be a finite N-by-2 array")
        if len(query) == 0:
            return {key: np.empty(0) for key in ("mu_z", "var_z", "p", "fmu", "var_f")}
        parts = {key: [] for key in ("mu_z", "var_z", "p", "fmu", "var_f")}
        for start in range(0, len(query), self.chunk_size):
            chunk = query[start:start + self.chunk_size]
            values = self.engine.mmrgp_predict_all_latest(
                self._matlab.double(chunk.tolist()), run_id, nargout=5,
            )
            for key, value in zip(parts, values):
                array = np.asarray(value, dtype=float).reshape(-1)
                if len(array) != len(chunk) or not np.isfinite(array).all():
                    raise RuntimeError(f"MATLAB returned invalid {key} predictions")
                parts[key].append(array)
        result = {key: np.concatenate(value) for key, value in parts.items()}
        if np.any(result["var_z"] < -1e-10) or np.any(result["var_f"] < -1e-10):
            raise RuntimeError("MATLAB returned negative predictive variance")
        result["var_z"] = np.maximum(result["var_z"], 0)
        result["var_f"] = np.maximum(result["var_f"], 0)
        return result

    def intensity(self, run_id, query_xy):
        run_id = self._run_id(run_id)
        query = np.asarray(query_xy, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2 or not np.isfinite(query).all():
            raise ValueError("query_xy must be a finite N-by-2 array")
        if len(query) == 0:
            return np.empty(0)
        parts = {"mu_z": [], "var_z": []}
        for start in range(0, len(query), self.chunk_size):
            chunk = query[start:start + self.chunk_size]
            values = self.engine.mmrgp_predict_latest(
                self._matlab.double(chunk.tolist()), run_id, nargout=2,
            )
            for key, value in zip(parts, values):
                array = np.asarray(value, dtype=float).reshape(-1)
                if len(array) != len(chunk) or not np.isfinite(array).all():
                    raise RuntimeError(f"MATLAB returned invalid {key} predictions")
                parts[key].append(array)
        prediction = {key: np.concatenate(value) for key, value in parts.items()}
        if np.any(prediction["var_z"] < -1e-10):
            raise RuntimeError("MATLAB returned negative predictive variance")
        prediction["var_z"] = np.maximum(prediction["var_z"], 0)
        return lognormal_intensity(
            prediction["mu_z"], prediction["var_z"], self.cell_area_km2,
        )

    def clear(self, run_id):
        run_id = self._run_id(run_id)
        self.engine.mmrgp_clear_run(run_id, nargout=0)
        self._training.pop(run_id, None)
        self.last_diagnostics.pop(run_id, None)

    def save_checkpoint(self, run_id, path, fit_budget):
        """Persist model and sampler state; replace only after MATLAB save succeeds."""
        run_id = self._run_id(run_id)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp.mat")
        try:
            self.engine.mmrgp_save_checkpoint(
                run_id, str(temporary), int(fit_budget), nargout=0,
            )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def restore_checkpoint(self, run_id, path, train_xy, train_y):
        """Restore state and return its fitted budget; training is the paired prefix."""
        run_id = self._run_id(run_id)
        xy = np.asarray(train_xy, dtype=float)
        y = np.asarray(train_y, dtype=float)
        fit_budget = int(self.engine.mmrgp_load_checkpoint(run_id, str(path), nargout=1))
        self.engine.addpath(str(ROOT.parent / "MRSM_subprograms1"), nargout=0)
        self.engine.addpath(str(ROOT.parent / "numeric_sample"), nargout=0)
        self.engine.addpath(str(Path(self.gpml_root) / "util"), nargout=0)
        if not 1 <= fit_budget <= len(xy) or y.shape != xy.shape:
            self.clear(run_id)
            raise RuntimeError("checkpoint budget is incompatible with saved training data")
        self._training[run_id] = (xy[:fit_budget].copy(), y[:fit_budget].copy())
        return fit_budget

    def close(self):
        if self._closed:
            return
        for run_id in list(self._training):
            self.clear(run_id)
        self._closed = True
        if self._owns_engine:
            self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
