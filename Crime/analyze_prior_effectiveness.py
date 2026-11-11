"""Diagnose whether 2021 knowledge changes KARMA acquisition and downstream utility."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from build_crimes_dataset import build_knowledge

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
FORMAL = ROOT / "results" / "formal" / "1km_budget50_random_holdout"
HOTSPOT = ROOT / "results" / "hotspot_metrics" / "1km_budget50_random_holdout"
OUTPUT = ROOT / "results" / "prior_effectiveness"


def history(method, seed):
    path = FORMAL / method / "fold_0" / f"seed_{seed}" / "history.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETE"
    assert result["total_budget"] == 50 and result["initial_budget"] == 30
    assert len(result["acquisitions"]) == 20
    return result


def final_metrics(result):
    row = next(item for item in result["evaluations"] if item["budget"] == 50)
    return {key: float(row[key]) for key in (
        "region_pass", "density_lift", "hit", "high_crime_coverage", "rmae", "acc",
    )}


def hotspot_metrics(method, seed):
    path = HOTSPOT / method / "fold_0" / f"seed_{seed}" / "hotspot_metrics.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETE" and result["n_test"] == 132
    return {key: float(result[key]) for key in ("ndcg_at_20", "h_rmae_at_20")}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km.csv")
    source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
    knowledge = pd.read_csv(ROOT / "data" / "knowledge" / "knowledge_2021_1km.csv")
    joined = units[["cell_id", "x", "y", "y1_log1p", "is_key_q3", "is_classification_boundary"]].copy()
    source_xy = knowledge[["x", "y"]].to_numpy(float)
    query_xy = joined[["x", "y"]].to_numpy(float)
    for channel in ("pi_hg", "pi_bnd"):
        source_values = knowledge[channel].to_numpy(float)
        linear = griddata(source_xy, source_values, query_xy, method="linear")
        nearest = griddata(source_xy, source_values, query_xy, method="nearest")
        joined[channel] = np.where(np.isfinite(linear), linear, nearest)
    assert len(joined) == len(units) == 659 and joined[["pi_hg", "pi_bnd"]].notna().all().all()

    mechanism_rows = []
    paired_rows = []
    for seed in SEEDS:
        karma = history("KARMA", seed)
        roi = history("ROI-D", seed)
        assert karma["training_indices"][:30] == roi["training_indices"][:30]
        assert karma["test_indices"] == roi["test_indices"]
        kd = karma["acquisitions"][-1]["diagnostics"]
        rd = roi["acquisitions"][-1]["diagnostics"]
        alpha = np.asarray(kd["pi_comp_alpha_hist"], dtype=float)
        consistency = np.asarray(kd["pi_comp_cons_hist"], dtype=float)
        gate = np.asarray(kd["pi3_hist"], dtype=float)
        assert alpha.shape == consistency.shape == (20, 2) and gate.shape == (20, 4)
        k_seq = np.asarray(karma["training_indices"][30:], dtype=int)
        r_seq = np.asarray(roi["training_indices"][30:], dtype=int)
        candidates = np.setdiff1d(np.arange(len(units)), np.asarray(karma["test_indices"], dtype=int))
        k_roi = np.asarray(kd["roi_selection_hist"], dtype=float)
        r_roi = np.asarray(rd["roi_selection_hist"], dtype=float)
        row = {
            "seed": seed,
            "alpha_hg_initial": alpha[0, 0], "alpha_bnd_initial": alpha[0, 1],
            "alpha_hg_final": alpha[-1, 0], "alpha_bnd_final": alpha[-1, 1],
            "alpha_hg_mean": alpha[:, 0].mean(), "alpha_bnd_mean": alpha[:, 1].mean(),
            "cons_hg_mean": consistency[:, 0].mean(), "cons_bnd_mean": consistency[:, 1].mean(),
            "knowledge_gate_active_steps": int(gate[:, 3].sum()),
            "knowledge_gate_closed_steps": int(20 - gate[:, 3].sum()),
            "karma_roi_selected_steps": int(k_roi[:, 1].sum()),
            "roi_d_roi_selected_steps": int(r_roi[:, 1].sum()),
            "ordered_same_selections": int(np.sum(k_seq == r_seq)),
            "unordered_selection_overlap": int(len(set(k_seq).intersection(r_seq))),
        }
        for channel in ("pi_hg", "pi_bnd"):
            values = joined[channel].to_numpy(float)
            row[f"karma_seq_{channel}_mean"] = float(values[k_seq].mean())
            row[f"roi_d_seq_{channel}_mean"] = float(values[r_seq].mean())
            row[f"candidate_{channel}_mean"] = float(values[candidates].mean())
        mechanism_rows.append(row)

        km = {**final_metrics(karma), **hotspot_metrics("KARMA", seed)}
        rm = {**final_metrics(roi), **hotspot_metrics("ROI-D", seed)}
        for metric in km:
            paired_rows.append({
                "seed": seed, "metric": metric,
                "karma": km[metric], "roi_d": rm[metric], "delta": km[metric] - rm[metric],
            })

    mechanism = pd.DataFrame(mechanism_rows)
    paired = pd.DataFrame(paired_rows)
    mechanism.to_csv(OUTPUT / "mechanism_by_seed.csv", index=False)
    paired.to_csv(OUTPUT / "paired_karma_minus_roi_d.csv", index=False)

    outcome_summary = []
    for metric, group in paired.groupby("metric", sort=False):
        delta = group["delta"].to_numpy(float)
        outcome_summary.append({
            "metric": metric, "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(ddof=0)),
            "karma_wins": int(np.sum(delta > 1e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "roi_d_wins": int(np.sum(delta < -1e-12)),
        })
    outcome_summary = pd.DataFrame(outcome_summary)
    outcome_summary.to_csv(OUTPUT / "paired_outcome_summary.csv", index=False)

    target_knowledge, _, _ = build_knowledge(units, 1.0)
    aligned = knowledge[["cell_id", "pi_hg"]].merge(
        target_knowledge[["cell_id", "pi_hg"]], on="cell_id",
        suffixes=("_2021", "_2022"), validate="one_to_one",
    )
    common = source[["cell_id", "y1_log1p"]].merge(
        units[["cell_id", "y1_log1p"]], on="cell_id",
        suffixes=("_2021", "_2022"), validate="one_to_one",
    )
    agreement_path = ROOT / "results" / "knowledge_y2_agreement_1km.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    cross_year = {
        "common_cells": int(len(common)),
        "intensity_pearson": float(pearsonr(common["y1_log1p_2021"], common["y1_log1p_2022"]).statistic),
        "intensity_spearman": float(spearmanr(common["y1_log1p_2021"], common["y1_log1p_2022"]).statistic),
        "pi_hg_pearson": float(pearsonr(aligned["pi_hg_2021"], aligned["pi_hg_2022"]).statistic),
        "pi_hg_spearman": float(spearmanr(aligned["pi_hg_2021"], aligned["pi_hg_2022"]).statistic),
        "pi_bnd_vs_2022_boundary_auc": agreement["knowledge_vs_2022_boundary_on_common_cells"]["pi_bnd_roc_auc"],
        "pi_bnd_vs_2022_boundary_ap": agreement["knowledge_vs_2022_boundary_on_common_cells"]["pi_bnd_average_precision"],
        "pi_hg_vs_2022_high_crime_auc": float(roc_auc_score(joined["is_key_q3"], joined["pi_hg"])),
        "pi_hg_vs_2022_intensity_spearman": float(spearmanr(joined["pi_hg"], joined["y1_log1p"]).statistic),
    }
    summary = {
        "scope": "Existing eight paired formal runs; descriptive diagnostics, not a new confirmatory experiment.",
        "n_seeds": len(SEEDS),
        "mechanism_means": mechanism.drop(columns="seed").mean().to_dict(),
        "totals": {
            "knowledge_gate_active_steps": int(mechanism["knowledge_gate_active_steps"].sum()),
            "knowledge_gate_closed_steps": int(mechanism["knowledge_gate_closed_steps"].sum()),
            "ordered_same_selections": int(mechanism["ordered_same_selections"].sum()),
            "unordered_selection_overlap": int(mechanism["unordered_selection_overlap"].sum()),
            "karma_roi_selected_steps": int(mechanism["karma_roi_selected_steps"].sum()),
            "roi_d_roi_selected_steps": int(mechanism["roi_d_roi_selected_steps"].sum()),
        },
        "cross_year_alignment": cross_year,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(mechanism.to_string(index=False))
    print("\nPaired KARMA - ROI-D:\n", outcome_summary.to_string(index=False))
    print("\nCross-year alignment:\n", json.dumps(cross_year, ensure_ascii=False, indent=2))
    print("\nTotals:\n", json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
