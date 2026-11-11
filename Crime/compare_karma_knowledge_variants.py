"""Compare completed KARMA runs under original and adjacent boundary knowledge."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = (172, 136, 34)
METRICS = (
    "region_pass", "density_lift", "rmae", "acc", "balanced_acc", "f1",
    "hit", "boundary_hit", "hit_auc", "rmae_aulc", "acc_auc",
    "violent_label_share", "violent_incident_share",
)
ORIGINAL = ROOT / "results" / "formal" / "1km_budget50_random_holdout"
ADJACENT = ROOT / "results" / "formal" / "1km_budget50_random_holdout_adjacent_bnd"
OUTPUT = ROOT / "results" / "aggregated" / "karma_knowledge_variant_paired_3seeds"


def load_history(root: Path, seed: int, variant: str) -> dict:
    path = root / "KARMA" / "fold_0" / f"seed_{seed}" / "history.json"
    history = json.loads(path.read_text(encoding="utf-8"))
    assert history["status"] == "COMPLETE" and not history.get("error")
    assert history["method"] == "KARMA" and history["seed"] == seed
    assert history.get("knowledge_variant", "original") == variant
    assert history["initial_budget"] == 30 and history["total_budget"] == 50
    assert history["split_scheme"] == "random_holdout"
    assert len(history["training_indices"]) == 50 and len(history["test_indices"]) == 132
    assert not path.with_name("matlab_checkpoint.mat").exists()
    return history


def main() -> None:
    rows = []
    for seed in SEEDS:
        original = load_history(ORIGINAL, seed, "original")
        adjacent = load_history(ADJACENT, seed, "adjacent")
        assert original["test_indices"] == adjacent["test_indices"]
        assert original["training_indices"][:30] == adjacent["training_indices"][:30]
        original_final = next(row for row in original["evaluations"] if row["budget"] == 50)
        adjacent_final = next(row for row in adjacent["evaluations"] if row["budget"] == 50)
        row = {"seed": seed}
        for metric in METRICS:
            row[f"original_{metric}"] = original_final[metric]
            row[f"adjacent_{metric}"] = adjacent_final[metric]
            row[f"delta_{metric}"] = adjacent_final[metric] - original_final[metric]
        rows.append(row)

    paired = pd.DataFrame(rows)
    summary_rows = []
    for metric in METRICS:
        summary_rows.append({
            "metric": metric,
            "original_mean": paired[f"original_{metric}"].mean(),
            "adjacent_mean": paired[f"adjacent_{metric}"].mean(),
            "delta_adjacent_minus_original": paired[f"delta_{metric}"].mean(),
            "original_std": paired[f"original_{metric}"].std(ddof=0),
            "adjacent_std": paired[f"adjacent_{metric}"].std(ddof=0),
        })
    summary = pd.DataFrame(summary_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUTPUT / "paired_per_seed.csv", index=False)
    summary.to_csv(OUTPUT / "paired_summary.csv", index=False)
    verification = {
        "status": "PASS", "seeds": list(SEEDS), "n_pairs": len(paired),
        "checks": [
            "all histories COMPLETE", "knowledge variants original/adjacent",
            "same test_indices within each seed", "same first 30 training indices within each seed",
            "50 total training samples", "132 test samples", "no residual MATLAB checkpoint",
        ],
    }
    (OUTPUT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
