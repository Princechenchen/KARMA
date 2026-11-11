"""Offline-only surrogate and recommended-region metrics."""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("y_true and probability must be aligned, nonempty, and binary")
    pred = (p >= 0.5).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "acc": float(np.mean(pred == y)),
        "balanced_acc": float((tpr + tnr) / 2),
        "f1": float(f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def sequential_auc(history, key, initial_budget=20, steps=20):
    final_budget = int(initial_budget) + int(steps)
    values = [
        float(row[key]) for row in history
        if int(initial_budget) < int(row["budget"]) <= final_budget
    ]
    if len(values) != int(steps):
        raise ValueError(f"sequential AUC requires exactly {steps} steps, got {len(values)}")
    return float(np.mean(values))


def region_supplements(boxes, units, violent_share):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    xy = units[["x", "y"]].to_numpy(float)
    inside = np.zeros(len(units), dtype=bool)
    for cx, cy, lx, ly in boxes:
        inside |= (
            (xy[:, 0] >= cx - lx / 2) & (xy[:, 0] <= cx + lx / 2)
            & (xy[:, 1] >= cy - ly / 2) & (xy[:, 1] <= cy + ly / 2)
        )
    key = units["is_key_q3"].to_numpy(bool)
    components = units["key_component_id"].to_numpy(int)
    all_components = set(components[components >= 0].tolist())
    covered_components = set(components[inside & (components >= 0)].tolist())
    label_share = float(np.mean(units.loc[inside, "y2"])) if inside.any() else float("nan")
    return {
        "high_crime_coverage": float(np.sum(inside & key) / max(np.sum(key), 1)),
        "component_coverage": float(len(covered_components) / max(len(all_components), 1)),
        "n_components_covered": len(covered_components),
        "n_components_total": len(all_components),
        "violent_label_share": label_share,
        "violent_label_uplift": float(label_share - violent_share),
        "n_cells_inside": int(np.sum(inside)),
    }
