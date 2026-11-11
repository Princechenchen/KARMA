"""Pre-check a count-based hotspot response before any model experiment.

The 2022 candidate label uses a threshold learned only from 2021 counts. The
2021 threshold-proximity field is evaluated against the candidate's 2022
four-neighbor class boundary on common occupied cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "hotspot_y2_precheck_1km"


def hotspot_threshold(source: pd.DataFrame) -> float:
    """Return the third quartile of source-year incident counts."""
    return float(source["y1"].quantile(0.75))


def four_neighbor_boundary(frame: pd.DataFrame, label_column: str) -> np.ndarray:
    """Mark cells with an existing four-neighbor carrying another label."""
    lookup = {(row.ix, row.iy): int(getattr(row, label_column)) for row in frame.itertuples()}
    boundary = np.zeros(len(frame), dtype=np.int8)
    for row_index, row in enumerate(frame.itertuples()):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = lookup.get((row.ix + dx, row.iy + dy))
            if neighbor is not None and neighbor != int(getattr(row, label_column)):
                boundary[row_index] = 1
                break
    return boundary


def build_hotspot_candidate(target: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build candidate hotspot labels without using the target-year quartile."""
    result = target.copy()
    result["hotspot_threshold_2021"] = float(threshold)
    result["y2_hot"] = (result["y1"] > threshold).astype(np.int8)
    result["is_hotspot_boundary"] = four_neighbor_boundary(result, "y2_hot")
    return result


def build_hotspot_boundary_prior(source: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build a 2021 log-count threshold-proximity boundary field."""
    result = source[["cell_id", "ix", "iy", "y1"]].copy()
    z = np.log1p(result["y1"].to_numpy(dtype=float))
    center = float(np.log1p(threshold))
    tau = max(1.4826 * float(np.median(np.abs(z - np.median(z)))), 0.05)
    result["prior_threshold_2021"] = float(threshold)
    result["prior_tau_2021"] = tau
    result["pi_bnd_hot"] = np.clip(
        np.exp(-((z - center) ** 2) / (2 * tau**2)), 0.01, 1.0
    )
    return result


def label_agreement_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Describe candidate hotspot labels against the existing 2022 Q3 labels."""
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    tp = int(np.sum((predicted == 1) & (actual == 1)))
    fp = int(np.sum((predicted == 1) & (actual == 0)))
    fn = int(np.sum((predicted == 0) & (actual == 1)))
    tn = int(np.sum((predicted == 0) & (actual == 0)))
    n = len(actual)
    accuracy = (tp + tn) / n
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    phi_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "cells": n,
        "candidate_positive_cells": int(predicted.sum()),
        "candidate_positive_share": float(predicted.mean()),
        "existing_q3_positive_cells": int(actual.sum()),
        "existing_q3_positive_share": float(actual.mean()),
        "confusion_using_candidate_as_prediction": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2),
        "positive_precision": float(precision),
        "positive_recall": float(recall),
        "positive_f1": float(2 * precision * recall / (precision + recall)),
        "cohen_kappa": float((accuracy - p_e) / (1 - p_e)),
        "matthews_correlation": float((tp * tn - fp * fn) / phi_denominator),
        "jaccard_positive": float(tp / (tp + fp + fn)),
    }


def main() -> None:
    source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    for frame in (source, target):
        assert frame["cell_id"].is_unique
        assert not frame[["cell_id", "ix", "iy", "y1"]].isna().any().any()
    assert not target["is_key_q3"].isna().any()

    threshold = hotspot_threshold(source)
    candidate = build_hotspot_candidate(target, threshold)
    prior = build_hotspot_boundary_prior(source, threshold)
    evaluated = candidate[[
        "cell_id", "y2_hot", "is_hotspot_boundary",
    ]].merge(prior, on="cell_id", how="inner", validate="one_to_one")
    actual_boundary = evaluated["is_hotspot_boundary"].to_numpy(dtype=int)
    score = evaluated["pi_bnd_hot"].to_numpy(dtype=float)
    assert 0 < actual_boundary.sum() < len(actual_boundary)

    labels = label_agreement_metrics(
        candidate["is_key_q3"].to_numpy(dtype=int), candidate["y2_hot"].to_numpy(dtype=int)
    )
    boundary = {
        "common_cells": int(len(evaluated)),
        "target_only_cells_excluded": int(len(candidate) - len(evaluated)),
        "candidate_boundary_cells": int(actual_boundary.sum()),
        "candidate_boundary_share": float(actual_boundary.mean()),
        "prior_mean": float(score.mean()),
        "boundary_cell_prior_mean": float(score[actual_boundary == 1].mean()),
        "nonboundary_cell_prior_mean": float(score[actual_boundary == 0].mean()),
        "roc_auc": float(roc_auc_score(actual_boundary, score)),
        "average_precision": float(average_precision_score(actual_boundary, score)),
        "random_ranking_average_precision_baseline": float(actual_boundary.mean()),
    }
    result = {
        "scope": "Exploratory pre-check only; no model fitting, sampling, or region recommendation.",
        "construction": {
            "candidate_label": "y2_hot = 1[y1_2022 > Q3(y1_2021)]",
            "source_q3_threshold": threshold,
            "target_q3_for_reference_only": hotspot_threshold(target),
            "target_q3_is_used_for_candidate": False,
            "boundary_rule": "an existing four-neighbor has a different y2_hot label",
            "prior": "clip(exp(-(log1p(y1_2021)-log1p(threshold))^2/(2*tau^2)),0.01,1)",
            "prior_tau": float(prior["prior_tau_2021"].iloc[0]),
        },
        "support": {
            "cells_2021": int(len(source)),
            "cells_2022": int(len(target)),
            "common_cells": int(len(evaluated)),
            "source_only_cells": int(len(source) - len(evaluated)),
            "target_only_cells": int(len(target) - len(evaluated)),
        },
        "label_agreement_all_2022_cells": labels,
        "boundary_prior_on_common_cells": boundary,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    candidate_columns = [
        "cell_id", "ix", "iy", "y1", "hotspot_threshold_2021", "y2_hot",
        "y1_q3", "is_key_q3", "is_hotspot_boundary",
    ]
    candidate[candidate_columns].to_csv(OUT / "candidate_labels_2022.csv", index=False)
    prior.to_csv(OUT / "boundary_prior_2021.csv", index=False)
    (OUT / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
