"""Benchmark GSO and MOPSO-style search on the frozen Crimes region objective.

This is an offline optimizer benchmark, not a sampling experiment. It uses the
2022 field only to provide a realistic, fixed oracle landscape. No method or
region-evaluation parameter is tuned from benchmark quality scores.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "optimizer_benchmark.json"
K = 4
C = 4.0
EPS = 1e-12
L_MIN, L_MAX = 0.01, 0.15
N_SOBOL = 512
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158, 180, 1]


def oracle_field():
    df = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    n = 241
    gx = np.linspace(0.0, 1.0, n)
    gy = np.linspace(0.0, 1.0, n)
    ix = np.clip(np.rint(df["x"].to_numpy() * (n - 1)).astype(int), 0, n - 1)
    iy = np.clip(np.rint(df["y"].to_numpy() * (n - 1)).astype(int), 0, n - 1)
    counts = np.zeros((n, n), dtype=float)
    np.add.at(counts, (iy, ix), df["y1"].to_numpy(dtype=float))
    smooth = gaussian_filter(counts, sigma=1.0, mode="nearest")
    interp = RegularGridInterpolator((gy, gx), smooth, bounds_error=False, fill_value=0.0)
    return interp


class Objective:
    def __init__(self):
        self.interp = oracle_field()
        self.unit = qmc.Sobol(d=2, scramble=True, seed=20261024).random_base2(
            m=int(np.log2(N_SOBOL))
        )
        self.evaluations = 0

    def repair(self, p):
        p = np.asarray(p, dtype=float).copy()
        p[:, 2:4] = np.clip(p[:, 2:4], L_MIN, L_MAX)
        p[:, 0] = np.clip(p[:, 0], p[:, 2] / 2, 1 - p[:, 2] / 2)
        p[:, 1] = np.clip(p[:, 1], p[:, 3] / 2, 1 - p[:, 3] / 2)
        return p

    def initialize(self, n, rng):
        p = np.empty((n, 4))
        p[:, 2:4] = rng.uniform(L_MIN, L_MAX, size=(n, 2))
        p[:, 0] = rng.uniform(p[:, 2] / 2, 1 - p[:, 2] / 2)
        p[:, 1] = rng.uniform(p[:, 3] / 2, 1 - p[:, 3] / 2)
        return p

    def values(self, p, occupied=None):
        p = np.asarray(p, dtype=float)
        n = len(p)
        pts = np.empty((n, N_SOBOL, 2))
        pts[:, :, 0] = p[:, None, 0] + (self.unit[None, :, 0] - 0.5) * p[:, None, 2]
        pts[:, :, 1] = p[:, None, 1] + (self.unit[None, :, 1] - 0.5) * p[:, None, 3]
        intensity = self.interp(pts.reshape(-1, 2)).reshape(n, N_SOBOL)
        area = p[:, 2] * p[:, 3]
        number = area * np.maximum(intensity.mean(axis=1), 0.0)
        value = np.log(number + EPS) + C * np.log(area)
        if occupied is not None and len(occupied):
            bad = np.zeros(n, dtype=bool)
            for old in occupied:
                bad |= overlap(p, np.repeat(old[None, :], n, axis=0))
            value[bad] = -1e6
        self.evaluations += n
        return value


def overlap(a, b, tol=1e-12):
    dx = np.minimum(a[:, 0] + a[:, 2] / 2, b[:, 0] + b[:, 2] / 2) - np.maximum(
        a[:, 0] - a[:, 2] / 2, b[:, 0] - b[:, 2] / 2
    )
    dy = np.minimum(a[:, 1] + a[:, 3] / 2, b[:, 1] + b[:, 3] / 2) - np.maximum(
        a[:, 1] - a[:, 3] / 2, b[:, 1] - b[:, 3] / 2
    )
    return (dx > tol) & (dy > tol)


def select_nonoverlapping(p, values, k=K):
    order = np.argsort(-values)
    chosen, scores = [], []
    for i in order:
        if not chosen:
            chosen.append(p[i].copy())
            scores.append(float(values[i]))
        else:
            old = np.asarray(chosen)
            if not overlap(np.repeat(p[i][None, :], len(old), axis=0), old).any():
                chosen.append(p[i].copy())
                scores.append(float(values[i]))
        if len(chosen) == k:
            break
    return np.asarray(chosen), np.asarray(scores)


def run_gso(seed, n_particles=100, iterations=100):
    rng = np.random.default_rng(seed)
    obj = Objective()
    rho, gamma, step = 0.4, 0.6, 0.03
    p = obj.initialize(n_particles, rng)
    values = obj.values(p)
    luciferin = gamma * values
    radius = np.full(n_particles, 0.30)
    for _ in range(iterations):
        values = obj.values(p)
        luciferin = (1 - rho) * luciferin + gamma * values
        distances = cdist(p, p)
        for i in range(n_particles):
            neighbors = np.flatnonzero((distances[i] <= radius[i]) & (luciferin > luciferin[i]))
            radius[i] = min(0.60, max(0.05, radius[i] + 0.08 * (5 - len(neighbors))))
            if len(neighbors):
                weights = luciferin[neighbors] - luciferin[i]
                weights = np.maximum(weights, 0)
                if weights.sum() > 0:
                    j = rng.choice(neighbors, p=weights / weights.sum())
                    p[i] += step * (p[j] - p[i]) / max(distances[i, j], 1e-12)
        p = obj.repair(p)
    values = obj.values(p)
    boxes, scores = select_nonoverlapping(p, values)
    return boxes, scores, obj.evaluations


def run_mopso(seed):
    rng = np.random.default_rng(seed)
    obj = Objective()
    n_particles, iterations_per_region = 50, 50
    occupied = []
    all_boxes, all_scores = [], []
    for _ in range(K):
        p = obj.initialize(n_particles, rng)
        velocity = rng.uniform(-0.02, 0.02, size=p.shape)
        values = obj.values(p, occupied)
        personal = p.copy()
        personal_values = values.copy()
        best = int(np.argmax(values))
        global_position = p[best].copy()
        global_value = float(values[best])
        for t in range(iterations_per_region):
            ratio = t / max(iterations_per_region - 1, 1)
            w = 0.9 - 0.5 * ratio
            c1 = 2.5 - 2.0 * ratio
            c2 = 0.5 + 2.0 * ratio
            velocity = (
                w * velocity
                + c1 * rng.random(p.shape) * (personal - p)
                + c2 * rng.random(p.shape) * (global_position[None, :] - p)
            )
            velocity = np.clip(velocity, -0.05, 0.05)
            p = obj.repair(p + velocity)
            values = obj.values(p, occupied)
            improved = values > personal_values
            personal[improved] = p[improved]
            personal_values[improved] = values[improved]
            best = int(np.argmax(values))
            if values[best] > global_value:
                global_value = float(values[best])
                global_position = p[best].copy()
        occupied.append(global_position)
        all_boxes.append(global_position)
        all_scores.append(global_value)
    return np.asarray(all_boxes), np.asarray(all_scores), obj.evaluations


def summarize(name, records):
    times = np.array([r["seconds"] for r in records])
    counts = np.array([r["n_regions"] for r in records])
    scores = np.array([r["score_sum"] for r in records])
    evals = np.array([r["evaluations"] for r in records])
    return {
        "method": name,
        "runs": len(records),
        "seconds_mean": float(times.mean()),
        "seconds_std": float(times.std(ddof=1)),
        "seconds_median": float(np.median(times)),
        "evaluations_per_run": int(evals[0]),
        "regions_returned_mean": float(counts.mean()),
        "runs_returning_k": int(np.sum(counts == K)),
        "objective_sum_mean_over_complete_runs": (
            float(scores[counts == K].mean()) if np.any(counts == K) else None
        ),
        "raw": records,
    }


def main():
    results = []
    configurations = (
        ("GSO-50x200", lambda seed: run_gso(seed, 50, 200)),
        ("GSO-100x100", lambda seed: run_gso(seed, 100, 100)),
        ("MOPSO-style-50x200", run_mopso),
    )
    for name, fn in configurations:
        fn(999)  
        records = []
        for seed in SEEDS:
            start = time.perf_counter()
            boxes, scores, evaluations = fn(seed)
            elapsed = time.perf_counter() - start
            records.append({
                "seed": seed,
                "seconds": elapsed,
                "evaluations": int(evaluations),
                "n_regions": int(len(boxes)),
                "score_sum": float(scores.sum()),
                "scores": scores.tolist(),
                "boxes": boxes.tolist(),
            })
        summary = summarize(name, records)
        results.append(summary)
        print(name, json.dumps({k: v for k, v in summary.items() if k != "raw"}, ensure_ascii=False))
    faster = min(results, key=lambda r: r["seconds_median"])
    payload = {
        "purpose": "Offline speed benchmark on a fixed Crimes 2022 oracle landscape; not a method comparison.",
        "settings": {
            "K": K, "c": C, "epsilon": EPS, "length_bounds": [L_MIN, L_MAX],
            "sobol_points_per_region": N_SOBOL, "seeds": SEEDS,
            "GSO": {"rho": 0.4, "gamma": 0.6, "budgets": [[50, 200], [100, 100]]},
            "MOPSO-style": {"particles": 50, "iterations_total": 200, "regions": 4},
        },
        "faster_by_median_seconds": faster["method"],
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
