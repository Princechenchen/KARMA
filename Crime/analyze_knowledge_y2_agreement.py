"""Quantify exploratory 2021-2022 agreement on occupied 1 km grid cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")

for frame in (source, target, knowledge):
    assert frame["cell_id"].is_unique

paired = source[["cell_id", "y1", "violent_ratio", "y2"]].merge(
    target[["cell_id", "y1", "violent_ratio", "y2", "is_classification_boundary"]],
    on="cell_id", how="inner", suffixes=("_2021", "_2022"), validate="one_to_one",
)
assert not paired.isna().any().any()

actual = paired["y2_2022"].to_numpy(dtype=int)
predicted = paired["y2_2021"].to_numpy(dtype=int)
tp = int(np.sum((predicted == 1) & (actual == 1)))
fp = int(np.sum((predicted == 1) & (actual == 0)))
fn = int(np.sum((predicted == 0) & (actual == 1)))
tn = int(np.sum((predicted == 0) & (actual == 0)))
n = len(paired)
accuracy = (tp + tn) / n
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * precision * sensitivity / (precision + sensitivity)
p_o = accuracy
p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
kappa = (p_o - p_e) / (1 - p_e)
phi_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc = (tp * tn - fp * fn) / phi_denominator

ratio_2021 = paired["violent_ratio_2021"].to_numpy()
ratio_2022 = paired["violent_ratio_2022"].to_numpy()
pearson = pearsonr(ratio_2021, ratio_2022)
spearman = spearmanr(ratio_2021, ratio_2022)

boundary_paired = paired.merge(
    knowledge[["cell_id", "pi_bnd", "pi_hg"]], on="cell_id", how="left", validate="one_to_one"
)
assert not boundary_paired[["pi_bnd", "pi_hg"]].isna().any().any()
boundary = boundary_paired["is_classification_boundary"].to_numpy(dtype=int)
assert 0 < boundary.sum() < len(boundary)

stable = paired[(paired["y1_2021"] >= 10) & (paired["y1_2022"] >= 10)]
stable_accuracy = float((stable["y2_2021"] == stable["y2_2022"]).mean())

result = {
    "scope": "Exploratory descriptive analysis; occupied cells only; no causal inference.",
    "grid_size_km": 1,
    "support": {
        "cells_2021": int(len(source)),
        "cells_2022": int(len(target)),
        "common_cells": int(n),
        "target_only_cells": int(len(target) - n),
        "source_only_cells": int(len(source) - n),
        "target_coverage_by_2021": float(n / len(target)),
        "jaccard_support": float(n / (len(source) + len(target) - n)),
    },
    "y2_on_common_cells": {
        "positive_share_2021": float(predicted.mean()),
        "positive_share_2022": float(actual.mean()),
        "confusion_using_2021_as_prediction": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "unchanged_share": float(accuracy),
        "changed_share": float(1 - accuracy),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "positive_precision": float(precision),
        "positive_recall": float(sensitivity),
        "positive_f1": float(f1),
        "cohen_kappa": float(kappa),
        "matthews_correlation": float(mcc),
        "all_negative_accuracy_baseline": float(1 - actual.mean()),
        "stable_cells_both_years_at_least_10_incidents": int(len(stable)),
        "unchanged_share_on_stable_cells": stable_accuracy,
    },
    "violent_ratio_on_common_cells": {
        "pearson_r": float(pearson.statistic),
        "pearson_p_exploratory": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_exploratory": float(spearman.pvalue),
        "median_absolute_difference": float(np.median(np.abs(ratio_2021 - ratio_2022))),
        "mean_absolute_difference": float(np.mean(np.abs(ratio_2021 - ratio_2022))),
    },
    "knowledge_vs_2022_boundary_on_common_cells": {
        "boundary_cells": int(boundary.sum()),
        "boundary_share": float(boundary.mean()),
        "pi_bnd_roc_auc": float(roc_auc_score(boundary, boundary_paired["pi_bnd"])),
        "pi_bnd_average_precision": float(average_precision_score(boundary, boundary_paired["pi_bnd"])),
        "pi_hg_roc_auc": float(roc_auc_score(boundary, boundary_paired["pi_hg"])),
        "note": "pi_bnd is a 2021 threshold-proximity field, not a 2021 y2 label.",
    },
}

output = ROOT / "results" / "knowledge_y2_agreement_1km.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"OUTPUT={output}")
