"""Unified offline Top-K region recommendation and matched-reference evaluation.

Boxes are arrays ``[center_x, center_y, length_x, length_y]`` in the normalized
[0, 1]^2 data domain. Intensity callables return incidents/km² at xy points.
The module is surrogate-agnostic; MMRGP or another model can be connected by an
intensity adapter without changing GSO or the final evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

IntensityFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RecommendationConfig:
    k: int = 4
    c: float = 4.0
    epsilon: float = 1e-12
    length_min: float = 0.01
    length_max: float = 0.15
    n_sobol: int = 512
    sobol_seed: int = 20261024
    particles: int = 100
    iterations: int = 100
    rho: float = 0.4
    gamma: float = 0.6
    step: float = 0.03
    radius_init: float = 0.30
    target_neighbors: int = 5
    radius_beta: float = 0.08
    radius_min: float = 0.05
    radius_max: float = 0.60

    def to_dict(self):
        return asdict(self)


def lognormal_intensity(mu: np.ndarray, variance: np.ndarray, cell_area_km2: float) -> np.ndarray:
    """Invert log1p(count) predictions and convert counts to incidents/km²."""
    mu = np.asarray(mu, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if cell_area_km2 <= 0:
        raise ValueError("cell_area_km2 must be positive")
    if np.any(variance < 0):
        raise ValueError("predictive variance must be nonnegative")
    return (np.exp(mu + 0.5 * variance) - 1.0) / cell_area_km2


def rectangles_overlap(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) != len(b):
        raise ValueError("overlap inputs must contain the same number of boxes")
    dx = np.minimum(a[:, 0] + a[:, 2] / 2, b[:, 0] + b[:, 2] / 2) - np.maximum(
        a[:, 0] - a[:, 2] / 2, b[:, 0] - b[:, 2] / 2
    )
    dy = np.minimum(a[:, 1] + a[:, 3] / 2, b[:, 1] + b[:, 3] / 2) - np.maximum(
        a[:, 1] - a[:, 3] / 2, b[:, 1] - b[:, 3] / 2
    )
    return (dx > tol) & (dy > tol)


def make_matched_references(
    boxes: np.ndarray, n_reference: int = 2000, seed: int = 20261024,
    box_valid_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Translate each box while preserving its side lengths and spatial feasibility."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if n_reference <= 0:
        raise ValueError("n_reference must be positive")
    rng = np.random.default_rng(seed)
    references = np.empty((len(boxes), n_reference, 4), dtype=float)
    references[:, :, 2:] = boxes[:, None, 2:]
    for index, box in enumerate(boxes):
        accepted = []
        n_accepted = 0
        attempts = 0
        while n_accepted < n_reference:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("could not generate enough valid matched references")
            uniforms = rng.random((n_reference, 2))
            candidates = np.empty((n_reference, 4), dtype=float)
            candidates[:, 2:] = box[2:]
            candidates[:, :2] = box[2:] / 2 + uniforms * (1 - box[2:])
            if box_valid_fn is not None:
                candidates = candidates[np.asarray(box_valid_fn(candidates), dtype=bool)]
            accepted.append(candidates)
            n_accepted += len(candidates)
        references[index, :, :2] = np.concatenate(accepted, axis=0)[:n_reference, :2]
    return references


def rectangle_counts(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Exact closed-rectangle counts using offline 2D prefix dominance queries.

    A static Fenwick tree over y processes all four inclusion-exclusion queries
    in one x sweep. This avoids a boxes-by-incidents comparison matrix.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    if len(points) == 0:
        return np.zeros(len(boxes), dtype=np.int64)
    xmin = np.nextafter(boxes[:, 0] - boxes[:, 2] / 2, -np.inf)
    xmax = boxes[:, 0] + boxes[:, 2] / 2
    ymin = np.nextafter(boxes[:, 1] - boxes[:, 3] / 2, -np.inf)
    ymax = boxes[:, 1] + boxes[:, 3] / 2
    query_x = np.concatenate([xmax, xmin, xmax, xmin])
    query_y = np.concatenate([ymax, ymax, ymin, ymin])
    signs = np.concatenate([
        np.ones(len(boxes)), -np.ones(len(boxes)),
        -np.ones(len(boxes)), np.ones(len(boxes)),
    ])

    point_order = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[point_order]
    y_values = np.unique(points[:, 1])
    point_y_rank = np.searchsorted(y_values, sorted_points[:, 1])
    bit = np.zeros(len(y_values) + 1, dtype=np.int64)

    def update(index):
        index += 1
        while index < len(bit):
            bit[index] += 1
            index += index & -index

    def prefix(index):
        total = 0
        index += 1
        while index > 0:
            total += bit[index]
            index -= index & -index
        return total

    query_order = np.argsort(query_x, kind="mergesort")
    dominance = np.empty(len(query_x), dtype=np.int64)
    cursor = 0
    for query_index in query_order:
        x_limit = query_x[query_index]
        while cursor < len(sorted_points) and sorted_points[cursor, 0] <= x_limit:
            update(point_y_rank[cursor])
            cursor += 1
        y_rank = np.searchsorted(y_values, query_y[query_index], side="right") - 1
        dominance[query_index] = prefix(y_rank) if y_rank >= 0 else 0
    grouped = dominance.reshape(4, len(boxes))
    signed = signs.reshape(4, len(boxes))
    return np.sum(grouped * signed, axis=0, dtype=np.int64)


def evaluate_recommendations(
    boxes: np.ndarray,
    incidents_xy: np.ndarray,
    domain_width_km: tuple[float, float],
    n_reference: int = 2000,
    seed: int = 20261024,
    box_valid_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    incidents_xy = np.asarray(incidents_xy, dtype=float).reshape(-1, 2)
    if len(boxes) == 0:
        raise ValueError("at least one recommended box is required")
    if box_valid_fn is not None and not np.all(box_valid_fn(boxes)):
        raise ValueError("recommended boxes must lie entirely within the valid grid")
    widths = np.asarray(domain_width_km, dtype=float)
    if widths.shape != (2,) or np.any(widths <= 0):
        raise ValueError("domain_width_km must contain two positive values")
    references = make_matched_references(boxes, n_reference, seed, box_valid_fn)
    counts = rectangle_counts(incidents_xy, boxes)
    reference_counts = rectangle_counts(incidents_xy, references.reshape(-1, 4)).reshape(
        len(boxes), n_reference
    )
    areas = boxes[:, 2] * widths[0] * boxes[:, 3] * widths[1]
    densities = counts / areas
    reference_densities = reference_counts / areas[:, None]
    reference_q3 = np.quantile(reference_counts, 0.75, axis=1, method="linear")
    pass_flags = counts > reference_q3
    denominator = float(np.mean(reference_densities))
    density_lift = float(np.mean(densities) / denominator) if denominator > 0 else float("nan")
    return {
        "region_pass": float(np.mean(pass_flags)),
        "density_lift": density_lift,
        "counts": counts.tolist(),
        "densities_per_km2": densities.tolist(),
        "areas_km2": areas.tolist(),
        "reference_q3": reference_q3.tolist(),
        "reference_mean_density": np.mean(reference_densities, axis=1).tolist(),
        "pass_flags": pass_flags.astype(int).tolist(),
        "n_reference": int(n_reference),
        "reference_seed": int(seed),
    }


class _SobolObjective:
    def __init__(
        self, intensity_fn: IntensityFn, domain_width_km, config: RecommendationConfig,
        box_valid_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.intensity_fn = intensity_fn
        self.widths = np.asarray(domain_width_km, dtype=float)
        self.config = config
        self.box_valid_fn = box_valid_fn
        power = int(np.log2(config.n_sobol))
        if 2**power != config.n_sobol:
            raise ValueError("n_sobol must be a power of two")
        self.unit = qmc.Sobol(d=2, scramble=True, seed=config.sobol_seed).random_base2(power)
        self.evaluations = 0

    def valid_mask(self, particles):
        if self.box_valid_fn is None:
            return np.ones(len(particles), dtype=bool)
        return np.asarray(self.box_valid_fn(particles), dtype=bool)

    def repair(self, particles):
        particles = np.asarray(particles, dtype=float).copy()
        particles[:, 2:] = np.clip(particles[:, 2:], self.config.length_min, self.config.length_max)
        particles[:, 0] = np.clip(particles[:, 0], particles[:, 2] / 2, 1 - particles[:, 2] / 2)
        particles[:, 1] = np.clip(particles[:, 1], particles[:, 3] / 2, 1 - particles[:, 3] / 2)
        return particles

    def _random_particles(self, rng, n):
        particles = np.empty((n, 4))
        particles[:, 2:] = rng.uniform(
            self.config.length_min, self.config.length_max, size=(n, 2)
        )
        particles[:, 0] = rng.uniform(particles[:, 2] / 2, 1 - particles[:, 2] / 2)
        particles[:, 1] = rng.uniform(particles[:, 3] / 2, 1 - particles[:, 3] / 2)
        return particles

    def initialize(self, rng):
        particles = self._random_particles(rng, self.config.particles)
        if self.box_valid_fn is not None:
            for _ in range(1000):
                invalid = ~self.valid_mask(particles)
                if not invalid.any():
                    break
                particles[invalid] = self._random_particles(rng, int(invalid.sum()))
        return particles

    def values(self, particles):
        particles = np.asarray(particles, dtype=float)
        n = len(particles)
        valid = self.valid_mask(particles)
        values = np.full(n, -1e12, dtype=float)
        if valid.any():
            feasible = particles[valid]
            points = np.empty((len(feasible), self.config.n_sobol, 2))
            points[:, :, 0] = feasible[:, None, 0] + (self.unit[None, :, 0] - 0.5) * feasible[:, None, 2]
            points[:, :, 1] = feasible[:, None, 1] + (self.unit[None, :, 1] - 0.5) * feasible[:, None, 3]
            intensity = np.asarray(self.intensity_fn(points.reshape(-1, 2)), dtype=float).reshape(
                len(feasible), self.config.n_sobol
            )
            if not np.all(np.isfinite(intensity)):
                raise ValueError("intensity function returned non-finite values")
            areas = feasible[:, 2] * self.widths[0] * feasible[:, 3] * self.widths[1]
            numbers = areas * np.maximum(intensity.mean(axis=1), 0.0)
            values[valid] = np.log(numbers + self.config.epsilon) + self.config.c * np.log(
                feasible[:, 2] * feasible[:, 3]
            )
        self.evaluations += n
        return values


def _select_nonoverlapping(particles, values, k, valid_mask=None):
    chosen, scores = [], []
    if valid_mask is None:
        valid_mask = np.ones(len(particles), dtype=bool)
    for index in np.argsort(-values):
        if not valid_mask[index]:
            continue
        candidate = particles[index]
        if chosen:
            old = np.asarray(chosen)
            if rectangles_overlap(np.repeat(candidate[None, :], len(old), axis=0), old).any():
                continue
        chosen.append(candidate.copy())
        scores.append(float(values[index]))
        if len(chosen) == k:
            break
    return np.asarray(chosen).reshape(-1, 4), np.asarray(scores)


def recommend_regions(
    intensity_fn: IntensityFn,
    domain_width_km: tuple[float, float],
    config: RecommendationConfig | None = None,
    seed: int = 20261024,
    box_valid_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Run one multimodal GSO search and select K non-overlapping boxes."""
    config = config or RecommendationConfig()
    rng = np.random.default_rng(seed)
    objective = _SobolObjective(intensity_fn, domain_width_km, config, box_valid_fn)
    start = time.perf_counter()
    particles = objective.initialize(rng)
    values = objective.values(particles)
    initial_particles = particles.copy()
    initial_values = values.copy()
    luciferin = config.gamma * values
    radius = np.full(config.particles, config.radius_init)
    for _ in range(config.iterations):
        values = objective.values(particles)
        luciferin = (1 - config.rho) * luciferin + config.gamma * values
        distances = cdist(particles, particles)
        for i in range(config.particles):
            neighbors = np.flatnonzero(
                (distances[i] <= radius[i]) & (luciferin > luciferin[i])
            )
            radius[i] = np.clip(
                radius[i] + config.radius_beta * (config.target_neighbors - len(neighbors)),
                config.radius_min,
                config.radius_max,
            )
            if len(neighbors):
                weights = np.maximum(luciferin[neighbors] - luciferin[i], 0)
                if weights.sum() > 0:
                    selected = rng.choice(neighbors, p=weights / weights.sum())
                    particles[i] += config.step * (particles[selected] - particles[i]) / max(
                        distances[i, selected], 1e-12
                    )
        particles = objective.repair(particles)
    values = objective.values(particles)
    boxes, scores = _select_nonoverlapping(
        particles, values, config.k, objective.valid_mask(particles),
    )
    fallback_used = len(boxes) < config.k
    if fallback_used:
        combined = np.concatenate([particles, initial_particles])
        boxes, scores = _select_nonoverlapping(
            combined,
            np.concatenate([values, initial_values]),
            config.k,
            objective.valid_mask(combined),
        )
    elapsed = time.perf_counter() - start
    return {
        "boxes": boxes,
        "scores": scores,
        "success": bool(len(boxes) == config.k),
        "initial_candidate_fallback": fallback_used,
        "evaluations": int(objective.evaluations),
        "seconds": float(elapsed),
        "seed": int(seed),
        "config": config.to_dict(),
    }


def maximin_lhs_indices(xy: np.ndarray, n: int, seed: int, n_designs: int = 32) -> np.ndarray:
    """Select unique discrete cells using the best of several mapped LHS designs."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not 1 <= n <= len(xy):
        raise ValueError("n must be between 1 and the number of candidate cells")
    best_indices, best_distance = None, -np.inf
    for design_index in range(n_designs):
        design = qmc.LatinHypercube(d=2, seed=seed + design_index).random(n)
        distances = cdist(design, xy)
        rows, columns = linear_sum_assignment(distances)
        indices = columns[np.argsort(rows)]
        minimum = float(np.min(pdist(xy[indices]))) if n > 1 else 0.0
        if minimum > best_distance:
            best_indices, best_distance = indices, minimum
    return np.asarray(best_indices, dtype=int)
