"""Paired eight-seed comparison of 2021 priors and target-year Oracle mechanisms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = [172, 136, 34, 46, 77, 160, 91, 158]
METHODS = ["C6", "C7", "KARMA"]
VARIANTS = {
    "original": "1km_budget50_random_holdout",
    "oracle2022_threshold": "1km_budget50_random_holdout_oracle2022_threshold_bnd",
    "oracle2022_adjacent": "1km_budget50_random_holdout_oracle2022_adjacent_bnd",
}
METRICS = [
    "region_pass", "density_lift", "rmae", "acc", "balanced_acc", "f1",
    "hit", "boundary_hit", "hit_auc", "rmae_aulc", "acc_auc",
]
KARMA_METRICS = ["alpha_hg_mean", "alpha_bnd_mean", "gate_off_count", "alpha_bnd_final"]


def karma_diagnostics(history):
    alphas, gates = [], []
    for acquisition in history["acquisitions"]:
        diagnostics = acquisition.get("diagnostics", {})
        alpha = np.asarray(diagnostics.get("pi_comp_alpha", []), dtype=float).reshape(-1, 2)
        gate = np.asarray(diagnostics.get("pi3_hist", []), dtype=float)
        if not len(alpha) or not gate.size:
            raise ValueError("KARMA history lacks arbitration diagnostics")
        alphas.append(alpha[-1])
        gates.append(float(gate.reshape(-1, 4)[-1, 3]))
    alphas = np.asarray(alphas)
    return {
        "alpha_hg_mean": float(alphas[:, 0].mean()),
        "alpha_bnd_mean": float(alphas[:, 1].mean()),
        "gate_off_count": int(sum(value < 0.5 for value in gates)),
        "alpha_bnd_final": float(alphas[-1, 1]),
    }


def summarize(frame, metrics):
    rows = []
    for (method, variant), group in frame.groupby(["method", "knowledge_variant"], sort=False):
        row = {"method": method, "knowledge_variant": variant, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def load_runs():
    rows = []
    for variant, directory in VARIANTS.items():
        for method in METHODS:
            for seed in SEEDS:
                path = ROOT / "results" / "formal" / directory / method / "fold_0" / f"seed_{seed}" / "history.json"
                history = json.loads(path.read_text(encoding="utf-8"))
                if history.get("status") != "COMPLETE" or len(history.get("acquisitions", [])) != 20:
                    raise RuntimeError(f"incomplete formal run: {path}")
                evaluation = next(row for row in history["evaluations"] if int(row["budget"]) == 50)
                row = {
                    "method": method, "knowledge_variant": variant, "seed": seed,
                    "run_id": history["run_id"], "total_seconds": float(history["total_seconds"]),
                    **{metric: float(evaluation[metric]) for metric in METRICS},
                }
                if method == "KARMA":
                    row.update(karma_diagnostics(history))
                rows.append(row)
    frame = pd.DataFrame(rows)
    counts = frame.groupby(["method", "knowledge_variant"])["seed"].nunique()
    if len(frame) != 72 or not (counts == 8).all():
        raise RuntimeError("expected exactly eight seeds for every method and knowledge variant")
    return frame


def main():
    output = ROOT / "results" / "aggregated" / "oracle2022_prior_paired_8seeds"
    output.mkdir(parents=True, exist_ok=True)
    frame = load_runs()
    frame.to_csv(output / "per_run_metrics.csv", index=False)
    metrics = METRICS + ["total_seconds"] + KARMA_METRICS
    summary = summarize(frame, metrics)
    summary.to_csv(output / "summary_metrics.csv", index=False)

    baseline = frame[frame["knowledge_variant"] == "original"].set_index(["method", "seed"])
    deltas = []
    for variant in ["oracle2022_threshold", "oracle2022_adjacent"]:
        current = frame[frame["knowledge_variant"] == variant].set_index(["method", "seed"])
        for (method, seed), row in current.iterrows():
            delta = {"method": method, "knowledge_variant": variant, "seed": seed}
            for metric in METRICS:
                delta[metric] = float(row[metric] - baseline.loc[(method, seed), metric])
            deltas.append(delta)
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(output / "paired_delta_vs_2021.csv", index=False)
    summarize(delta_frame, METRICS).to_csv(output / "paired_delta_summary.csv", index=False)
    print(summary[["method", "knowledge_variant", "n_seeds", "region_pass_mean", "density_lift_mean", "rmae_mean", "acc_mean", "alpha_bnd_mean_mean", "gate_off_count_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
