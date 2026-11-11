"""Top-fraction surrogate diagnostics aligned with total-intensity region search."""
from __future__ import annotations

import numpy as np


def _validated_arrays(*arrays):
    converted = [np.asarray(values, dtype=float).reshape(-1) for values in arrays]
    if not converted[0].size or any(len(values) != len(converted[0]) for values in converted):
        raise ValueError("metric inputs must be nonempty and have equal lengths")
    if any(not np.isfinite(values).all() for values in converted):
        raise ValueError("metric inputs must be finite")
    return converted


def _cutoff(n_samples, fraction):
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return max(1, int(np.ceil(n_samples * fraction)))


def ndcg_at_fraction(y_true_log1p, predicted_intensity, fraction=0.2):
    """NDCG of predicted total intensity, using true log1p counts as gains."""
    truth, intensity = _validated_arrays(y_true_log1p, predicted_intensity)
    if np.any(truth < 0):
        raise ValueError("log1p responses must be nonnegative")
    k = _cutoff(len(truth), fraction)
    predicted_order = np.argsort(-intensity, kind="mergesort")[:k]
    ideal_order = np.argsort(-truth, kind="mergesort")[:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(truth[predicted_order] / discounts))
    ideal = float(np.sum(truth[ideal_order] / discounts))
    return dcg / ideal if ideal > 0 else 1.0


def hotspot_metrics(y_true_log1p, mu_z, predicted_intensity, fraction=0.2):
    """Return top-fraction NDCG and RMSE on the truly hottest test cells."""
    truth, mu_z, intensity = _validated_arrays(y_true_log1p, mu_z, predicted_intensity)
    k = _cutoff(len(truth), fraction)
    hottest = np.argsort(-truth, kind="mergesort")[:k]
    return {
        "ndcg_at_20": ndcg_at_fraction(truth, intensity, fraction),
        "h_rmae_at_20": float(np.sqrt(np.mean((truth[hottest] - mu_z[hottest]) ** 2))),
        "n_hotspot": int(k),
    }
