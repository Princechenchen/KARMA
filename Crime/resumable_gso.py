"""Checkpointable equivalent of region_recommender.recommend_regions."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from region_recommender import _SobolObjective, _select_nonoverlapping


class ResumableGSO:
    """Run the frozen GSO trajectory one iteration at a time."""

    def __init__(self, intensity_fn, domain_width_km, config, seed, box_valid_fn=None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.objective = _SobolObjective(intensity_fn, domain_width_km, config, box_valid_fn)
        self.particles = self.objective.initialize(self.rng)
        values = self.objective.values(self.particles)
        self.initial_particles = self.particles.copy()
        self.initial_values = values.copy()
        self.luciferin = config.gamma * values
        self.radius = np.full(config.particles, config.radius_init)
        self.iteration = 0

    def step(self):
        if self.iteration >= self.config.iterations:
            raise RuntimeError("GSO has already completed all iterations")
        config = self.config
        particles = self.particles
        values = self.objective.values(particles)
        self.luciferin = (1 - config.rho) * self.luciferin + config.gamma * values
        distances = cdist(particles, particles)
        for i in range(config.particles):
            neighbors = np.flatnonzero(
                (distances[i] <= self.radius[i]) & (self.luciferin > self.luciferin[i])
            )
            self.radius[i] = np.clip(
                self.radius[i] + config.radius_beta * (config.target_neighbors - len(neighbors)),
                config.radius_min,
                config.radius_max,
            )
            if len(neighbors):
                weights = np.maximum(self.luciferin[neighbors] - self.luciferin[i], 0)
                if weights.sum() > 0:
                    selected = self.rng.choice(neighbors, p=weights / weights.sum())
                    particles[i] += config.step * (particles[selected] - particles[i]) / max(
                        distances[i, selected], 1e-12
                    )
        self.particles = self.objective.repair(particles)
        self.iteration += 1

    def finalize(self):
        if self.iteration != self.config.iterations:
            raise RuntimeError("GSO cannot finalize before all iterations complete")
        values = self.objective.values(self.particles)
        boxes, scores = _select_nonoverlapping(
            self.particles, values, self.config.k, self.objective.valid_mask(self.particles),
        )
        fallback_used = len(boxes) < self.config.k
        if fallback_used:
            combined = np.concatenate([self.particles, self.initial_particles])
            boxes, scores = _select_nonoverlapping(
                combined,
                np.concatenate([values, self.initial_values]),
                self.config.k,
                self.objective.valid_mask(combined),
            )
        return {
            "boxes": boxes,
            "scores": scores,
            "success": bool(len(boxes) == self.config.k),
            "initial_candidate_fallback": fallback_used,
            "evaluations": int(self.objective.evaluations),
        }

    def state_dict(self):
        return {
            "iteration": int(self.iteration),
            "particles": self.particles.copy(),
            "luciferin": self.luciferin.copy(),
            "radius": self.radius.copy(),
            "initial_particles": self.initial_particles.copy(),
            "initial_values": self.initial_values.copy(),
            "evaluations": int(self.objective.evaluations),
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state(cls, intensity_fn, domain_width_km, config, state, box_valid_fn=None):
        self = cls.__new__(cls)
        self.config = config
        self.objective = _SobolObjective(intensity_fn, domain_width_km, config, box_valid_fn)
        self.particles = np.asarray(state["particles"], dtype=float).copy()
        self.luciferin = np.asarray(state["luciferin"], dtype=float).copy()
        self.radius = np.asarray(state["radius"], dtype=float).copy()
        self.initial_particles = np.asarray(state["initial_particles"], dtype=float).copy()
        self.initial_values = np.asarray(state["initial_values"], dtype=float).copy()
        self.iteration = int(state["iteration"])
        self.objective.evaluations = int(state["evaluations"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]
        if not 0 <= self.iteration <= config.iterations:
            raise ValueError("invalid checkpoint iteration")
        return self
