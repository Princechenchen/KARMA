"""Verify paired formal histories using per-seed random holdout splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE_METHODS = ("C1", "C2", "C12", "C10", "ROI-D", "C6", "C7", "KARMA")


def verify_random_holdout_group(
    root, methods=CORE_METHODS, seeds=(172, 136, 34),
    initial_budget=30, sequential_steps=20, test_fraction=0.2, n_units=659,
):
    root = Path(root)
    total_budget = int(initial_budget) + int(sequential_steps)
    expected_test = int(round(int(n_units) * float(test_fraction)))
    histories = {}
    for seed in seeds:
        reference_test = None
        reference_initial = None
        for method in methods:
            path = root / method / "fold_0" / f"seed_{seed}" / "history.json"
            assert path.exists(), f"missing history: {path}"
            history = json.loads(path.read_text(encoding="utf-8"))
            assert history.get("status") == "COMPLETE", f"incomplete history: {path}"
            assert not history.get("error"), f"failed history: {path}"
            assert history.get("split_scheme") == "random_holdout", f"split scheme: {path}"
            assert (history["method"], history["fold"], history["seed"]) == (method, 0, seed)
            assert int(history["initial_budget"]) == int(initial_budget)
            assert int(history["sequential_steps"]) == int(sequential_steps)
            assert int(history["total_budget"]) == total_budget
            train = list(map(int, history["training_indices"]))
            test = list(map(int, history["test_indices"]))
            assert len(train) == total_budget and len(set(train)) == total_budget
            assert len(test) == expected_test and len(set(test)) == expected_test
            assert all(0 <= index < int(n_units) for index in train + test)
            assert not set(train) & set(test), f"train/test overlap: {path}"
            assert len(history["test_cell_ids"]) == expected_test
            assert len(history["acquisitions"]) == int(sequential_steps)
            assert len(history["evaluations"]) == int(sequential_steps) + 1
            assert int(history["evaluations"][-1]["budget"]) == total_budget
            assert not path.with_name("matlab_checkpoint.mat").exists(), f"checkpoint: {path}"
            if reference_test is None:
                reference_test = test
                reference_initial = train[:int(initial_budget)]
            else:
                assert test == reference_test, f"test split mismatch: {path}"
                assert train[:int(initial_budget)] == reference_initial, f"initial design mismatch: {path}"
            histories[(method, seed)] = history
    for position, seed in enumerate(seeds):
        current = histories[(methods[0], seed)]["test_indices"]
        for other in seeds[position + 1:]:
            assert current != histories[(methods[0], other)]["test_indices"], (
                f"test splits are identical across seeds {seed} and {other}"
            )
    return {
        "status": "PASS", "root": str(root.resolve()),
        "methods": list(methods), "seeds": list(map(int, seeds)),
        "n_histories": len(histories), "initial_budget": int(initial_budget),
        "sequential_steps": int(sequential_steps), "total_budget": total_budget,
        "test_size_per_seed": expected_test,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "results/formal/1km_budget50_random_holdout")
    parser.add_argument("--methods", default=",".join(CORE_METHODS))
    parser.add_argument("--seeds", default="172,136,34")
    parser.add_argument("--initial-budget", type=int, default=30)
    parser.add_argument("--sequential-steps", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-units", type=int, default=659)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_random_holdout_group(
        args.root,
        methods=tuple(value.strip() for value in args.methods.split(",") if value.strip()),
        seeds=tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip()),
        initial_budget=args.initial_budget,
        sequential_steps=args.sequential_steps,
        test_fraction=args.test_fraction,
        n_units=args.n_units,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
