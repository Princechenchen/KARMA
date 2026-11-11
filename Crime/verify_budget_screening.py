"""Verify and summarize paired KARMA/ROI-D budget-screening histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUDGETS = [(20, 20), (20, 40), (30, 20), (30, 40), (40, 20), (40, 40)]
METRICS = ["region_pass", "density_lift", "rmae", "acc", "hit"]


def verify_history(path):
    path = Path(path)
    history = json.loads(path.read_text(encoding="utf-8"))
    initial = int(history["initial_budget"])
    steps = int(history["sequential_steps"])
    total = int(history["total_budget"])
    indices = list(map(int, history["training_indices"]))
    evaluations = history["evaluations"]
    assert history.get("status") == "COMPLETE", f"incomplete history: {path}"
    assert not history.get("error"), f"failed history: {path}"
    assert initial + steps == total, f"budget mismatch: {path}"
    assert len(indices) == total and len(set(indices)) == total, f"training cells: {path}"
    assert len(history["acquisitions"]) == steps, f"acquisitions: {path}"
    assert len(evaluations) == steps + 1, f"evaluations: {path}"
    assert int(evaluations[-1]["budget"]) == total, f"final budget: {path}"
    assert not path.with_name("matlab_checkpoint.mat").exists(), f"checkpoint remains: {path}"
    final = evaluations[-1]
    for metric in METRICS:
        assert metric in final and final[metric] is not None, f"missing {metric}: {path}"
    return {
        "path": str(path.resolve()), "method": history["method"],
        "fold": int(history["fold"]), "seed": int(history["seed"]),
        "grid": history["grid"], "initial_budget": initial,
        "sequential_steps": steps, "total_budget": total,
        "pass_flags": final.get("pass_flags"),
        **{metric: float(final[metric]) for metric in METRICS},
    }


def verify_pair(karma_path, roi_path):
    karma = verify_history(karma_path)
    roi = verify_history(roi_path)
    assert karma["method"] == "KARMA" and roi["method"] == "ROI-D", "pair methods"
    for key in ("initial_budget", "sequential_steps", "total_budget", "fold", "seed", "grid"):
        assert karma[key] == roi[key], f"pair {key} mismatch"
    initial = karma["initial_budget"]
    karma_history = json.loads(Path(karma_path).read_text(encoding="utf-8"))
    roi_history = json.loads(Path(roi_path).read_text(encoding="utf-8"))
    assert (karma_history["training_indices"][:initial]
            == roi_history["training_indices"][:initial]), "initial design mismatch"
    return {
        "initial_budget": initial, "sequential_steps": karma["sequential_steps"],
        "total_budget": karma["total_budget"], "fold": karma["fold"],
        "seed": karma["seed"], "grid": karma["grid"],
        "delta_region_pass": karma["region_pass"] - roi["region_pass"],
        "delta_density_lift": karma["density_lift"] - roi["density_lift"],
        "karma": karma, "roi_d": roi,
    }


def history_path(root, initial, steps, method, seed):
    if root == ROOT / "results" / "budget_screening" and seed == 172 and method == "KARMA":
        if (initial, steps) == (20, 20):
            return ROOT / "results/formal/1km/KARMA/fold_0/seed_172/history.json"
        if (initial, steps) == (20, 40):
            return ROOT / "results/formal/1km_budget60/KARMA/fold_0/seed_172/history.json"
    return (root / f"{initial}_plus_{steps}" / method / "fold_0" /
            f"seed_{seed}" / "history.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "budget_screening")
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    missing = []
    for initial, steps in BUDGETS:
        karma_path = history_path(root, initial, steps, "KARMA", args.seed)
        roi_path = history_path(root, initial, steps, "ROI-D", args.seed)
        if not karma_path.exists() or not roi_path.exists():
            missing.extend(str(path) for path in (karma_path, roi_path) if not path.exists())
            continue
        rows.append(verify_pair(karma_path, roi_path))
    if missing:
        raise FileNotFoundError("Missing paired histories:\n" + "\n".join(missing))
    rows.sort(key=lambda row: (
        -row["delta_region_pass"], -row["delta_density_lift"], row["total_budget"],
    ))
    report = {"status": "PASS", "seed": args.seed, "pairs": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
