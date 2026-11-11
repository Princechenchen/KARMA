"""Exact rectangle feasibility against the union of valid regular grid cells."""
from __future__ import annotations

import re

import numpy as np


class GridBoxValidity:
    """Return True only when a box has positive-area overlap with valid cells alone."""

    def __init__(self, units, tolerance=1e-10):
        pattern = re.compile(r"^g\d+_x(\d+)_y(\d+)$")
        parsed = [pattern.fullmatch(str(value)) for value in units["cell_id"]]
        if not all(parsed):
            raise ValueError("cell_id does not encode regular grid coordinates")
        self.ix = np.array([int(value.group(1)) for value in parsed], dtype=int)
        self.iy = np.array([int(value.group(2)) for value in parsed], dtype=int)
        self.x = np.asarray(units["x"], dtype=float)
        self.y = np.asarray(units["y"], dtype=float)
        self.tolerance = float(tolerance)
        self.x0, self.dx = self._axis(self.ix, self.x, "x")
        self.y0, self.dy = self._axis(self.iy, self.y, "y")
        self.valid = set(zip(self.ix.tolist(), self.iy.tolist()))

    @staticmethod
    def _axis(indices, coordinates, name):
        unique = np.unique(indices)
        centers = np.array([coordinates[indices == index].mean() for index in unique])
        if len(unique) < 2:
            raise ValueError(f"at least two {name} grid coordinates are required")
        steps = np.diff(centers)
        step = float(np.median(steps))
        if step <= 0 or not np.allclose(steps, step, rtol=0, atol=1e-8):
            raise ValueError(f"{name} grid centers must be regularly spaced")
        origin = float(centers[0] - unique[0] * step)
        return origin, step

    def _overlap_indices(self, lower, upper, origin, step):
        first = int(np.floor((lower - origin) / step - 0.5 + self.tolerance) + 1)
        last = int(np.ceil((upper - origin) / step + 0.5 - self.tolerance) - 1)
        return range(first, last + 1)

    def __call__(self, boxes):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        result = np.zeros(len(boxes), dtype=bool)
        for index, (cx, cy, lx, ly) in enumerate(boxes):
            if not np.all(np.isfinite([cx, cy, lx, ly])) or lx <= 0 or ly <= 0:
                continue
            xmin, xmax = cx - lx / 2, cx + lx / 2
            ymin, ymax = cy - ly / 2, cy + ly / 2
            if xmin < -self.tolerance or xmax > 1 + self.tolerance or ymin < -self.tolerance or ymax > 1 + self.tolerance:
                continue
            xs = self._overlap_indices(xmin, xmax, self.x0, self.dx)
            ys = self._overlap_indices(ymin, ymax, self.y0, self.dy)
            result[index] = all((ix, iy) in self.valid for ix in xs for iy in ys)
        return result
