"""Validate all saved joint-intensity recommendations against raw incidents."""
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from acquisition import parse_methods
from region_recommender import rectangle_counts
from smoke_lhs_oracle import load_domain, load_incidents

OUTPUT = ROOT / "results" / "violent_recommendation" / "1km_budget50_random_holdout"
METHODS = parse_methods(os.environ.get("VALIDATE_METHODS", "all"))
SEEDS = [int(value) for value in os.environ.get(
    "VALIDATE_SEEDS", "172,136,34,46,77,160,91,158",
).split(",")]
METRICS = ["region_pass", "density_lift", "violent_region_pass", "violent_density_lift"]


def direct_counts(points, boxes, chunk=100):
    """Closed-rectangle counts by direct point-in-box comparisons."""
    result = np.empty(len(boxes), dtype=np.int64)
    for start in range(0, len(boxes), chunk):
        current = boxes[start:start + chunk]
        inside = (
            (points[:, None, 0] >= current[None, :, 0] - current[None, :, 2] / 2)
            & (points[:, None, 0] <= current[None, :, 0] + current[None, :, 2] / 2)
            & (points[:, None, 1] >= current[None, :, 1] - current[None, :, 3] / 2)
            & (points[:, None, 1] <= current[None, :, 1] + current[None, :, 3] / 2)
        )
        result[start:start + chunk] = inside.sum(axis=0)
    return result


def direct_evaluation(boxes, points, widths, direct=False):
    rng = np.random.default_rng(20261024)
    uniforms = rng.random((4, 2000, 2))
    references = np.empty((4, 2000, 4))
    references[:, :, :2] = boxes[:, None, 2:] / 2 + uniforms * (1 - boxes[:, None, 2:])
    references[:, :, 2:] = boxes[:, None, 2:]
    count_fn = direct_counts if direct else rectangle_counts
    counts = count_fn(points, boxes)
    reference_counts = count_fn(points, references.reshape(-1, 4)).reshape(4, 2000)
    areas = boxes[:, 2] * widths[0] * boxes[:, 3] * widths[1]
    densities = counts / areas
    q3 = np.quantile(reference_counts, 0.75, axis=1, method="linear")
    lift = np.mean(densities) / np.mean(reference_counts / areas[:, None])
    return counts, q3, float(np.mean(counts > q3)), float(lift)


def main():
    construct, widths = load_domain()
    incidents, xy = load_incidents(construct)
    violent_xy = xy[incidents["is_violent"].to_numpy(bool)]
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            path = OUTPUT / method / "fold_0" / f"seed_{seed}" / "violent_recommendation.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            assert result["status"] == "COMPLETE" and result["budget"] == 50
            assert result["objective"] == "lambda_all_times_p_violent"
            assert result["n_reference"] == 2000 and result["reference_seed"] == 20261024
            assert result["gso_evaluations"] == 10200
            config = result["recommendation_config"]
            assert config["k"] == 4 and config["particles"] == 100 and config["iterations"] == 100
            assert config["n_sobol"] == 512
            boxes = np.asarray(result["boxes"], dtype=float)
            assert boxes.shape == (4, 4) and np.isfinite(boxes).all()
            assert (boxes[:, 2:] > 0).all()
            assert (boxes[:, :2] - boxes[:, 2:] / 2 >= -1e-12).all()
            assert (boxes[:, :2] + boxes[:, 2:] / 2 <= 1 + 1e-12).all()
            for i in range(4):
                for j in range(i):
                    overlap_x = min(boxes[i, 0] + boxes[i, 2] / 2, boxes[j, 0] + boxes[j, 2] / 2) - max(boxes[i, 0] - boxes[i, 2] / 2, boxes[j, 0] - boxes[j, 2] / 2)
                    overlap_y = min(boxes[i, 1] + boxes[i, 3] / 2, boxes[j, 1] + boxes[j, 3] / 2) - max(boxes[i, 1] - boxes[i, 3] / 2, boxes[j, 1] - boxes[j, 3] / 2)
                    assert not (overlap_x > 1e-12 and overlap_y > 1e-12)
            for prefix, points, rp, dl in [
                ("total", xy, "region_pass", "density_lift"),
                ("violent", violent_xy, "violent_region_pass", "violent_density_lift"),
            ]:
                counts, q3, region_pass, density_lift = direct_evaluation(
                    boxes, points, widths, direct=(method == "KARMA" and seed == 172),
                )
                evaluation = result[f"{prefix}_evaluation"]
                np.testing.assert_array_equal(counts, evaluation["counts"])
                np.testing.assert_allclose(q3, evaluation["reference_q3"], rtol=0, atol=1e-10)
                assert region_pass == result[rp]
                np.testing.assert_allclose(density_lift, result[dl], rtol=1e-12, atol=1e-12)
            rows.append(result)
            print(f"validated {method} seed={seed}", flush=True)

    assert len(rows) == len(METHODS) * len(SEEDS)
    assert not list(OUTPUT.glob("**/gso_state.npz"))
    per_run = pd.read_csv(OUTPUT / "per_run_metrics.csv")
    summary = pd.read_csv(OUTPUT / "summary_metrics.csv")
    assert len(per_run) == 104 and len(summary) == 13
    assert not per_run.duplicated(["method", "seed"]).any()
    assert per_run.groupby("method").size().eq(8).all()
    assert summary["n_seeds"].eq(8).all()
    all_results = [json.loads(path.read_text(encoding="utf-8")) for path in OUTPUT.glob(
        "*/fold_0/seed_*/violent_recommendation.json",
    )]
    assert len(all_results) == 104
    frame = pd.DataFrame(all_results)
    for _, row in summary.iterrows():
        group = frame[frame["method"] == row["method"]]
        for metric in METRICS:
            np.testing.assert_allclose(row[f"{metric}_mean"], group[metric].mean(), rtol=1e-12)
            np.testing.assert_allclose(row[f"{metric}_std"], group[metric].std(ddof=0), rtol=1e-12)
    print("PASS: 104 results; all boxes, counts, metrics and summaries verified")


if __name__ == "__main__":
    main()
