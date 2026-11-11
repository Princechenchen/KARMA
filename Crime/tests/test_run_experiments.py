import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeCompletedSession:
    def __init__(self):
        self.fit_calls = 0


class RunnerTests(unittest.TestCase):
    def test_completed_run_is_not_repeated(self):
        from run_experiments import ExperimentRunner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "C1" / "fold_0" / "seed_172"
            run_dir.mkdir(parents=True)
            history = {"status": "COMPLETE", "method": "C1", "fold": 0, "seed": 172,
                       "training_indices": list(range(100)), "evaluations": [], "acquisitions": []}
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
            session = FakeCompletedSession()
            runner = object.__new__(ExperimentRunner)
            runner.output_root = root
            runner.session = session
            result = runner.run("C1", 0, 172)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(session.fit_calls, 0)

    def test_atomic_history_write_leaves_valid_json(self):
        from run_experiments import atomic_write_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "history.json"
            atomic_write_json(path, {"status": "RUNNING", "value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 1)

    def test_short_c1_run_evaluates_then_acquires_and_completes(self):
        from run_experiments import ExperimentRunner

        class Session:
            def __init__(self):
                self.fit_calls = 0
                self.clear_calls = 0
                self.last_diagnostics = {}

            def clear(self, run_id):
                self.clear_calls += 1

            def fit(self, xy, y, run_id):
                self.fit_calls += 1
                self.train_size = len(xy)

            def predict(self, run_id, query):
                n = len(query)
                return {"mu_z": np.zeros(n), "var_z": np.ones(n), "p": np.full(n, 0.7),
                        "fmu": np.zeros(n), "var_f": np.ones(n)}

            def intensity(self, run_id, query):
                return np.ones(len(query))

            def save_checkpoint(self, run_id, checkpoint, budget):
                self.checkpoint_budgets = getattr(self, "checkpoint_budgets", []) + [budget]

        units = pd.DataFrame({
            "cell_id": [f"g1000_x{index}_y0" for index in range(30)],
            "x": np.linspace(0.01, 0.99, 30),
            "y": np.resize(np.linspace(0.05, 0.95, 6), 30),
            "y1_log1p": np.log1p(np.arange(30)), "y2": np.arange(30) % 2,
            "spatial_block": np.arange(30) % 5, "is_key_q3": np.arange(30) >= 24,
            "key_component_id": np.where(np.arange(30) >= 24, 0, -1),
            "is_classification_boundary": np.arange(30) % 3 == 0,
        })
        incidents = pd.DataFrame({"is_violent": [1, 0, 1, 0]})
        incidents_xy = np.array([[0.1, 0.1], [0.4, 0.4], [0.8, 0.8], [0.2, 0.7]])
        recommendation = {
            "success": True, "boxes": np.array([[0.2, 0.2, 0.1, 0.1]] * 4),
            "scores": np.arange(4), "seconds": 0.1, "evaluations": 4,
            "config": {"k": 4},
        }
        region_metrics = {"region_pass": 0.5, "density_lift": 2.0, "counts": [1, 1, 0, 0]}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("run_experiments.recommend_regions", return_value=recommendation), \
                patch("run_experiments.evaluate_recommendations", return_value=region_metrics):
            session = Session()
            runner = ExperimentRunner(
                session, units, incidents, incidents_xy, (1.0, 1.0), tmp,
                initial_budget=2, sequential_steps=2, evaluation_budgets=[2, 4],
                reference_points=np.array([[0.0, 0.0], [1.0, 1.0]]), n_reference=2,
            )
            history = runner.run("C1", 0, 172)
            self.assertEqual(history["status"], "COMPLETE")
            self.assertEqual([row["budget"] for row in history["evaluations"]], [2, 3, 4])
            self.assertEqual(len(history["acquisitions"]), 2)
            self.assertEqual(len(history["training_indices"]), 4)
            self.assertEqual(
                history["training_cell_ids"],
                units.iloc[history["training_indices"]]["cell_id"].tolist(),
            )
            self.assertEqual(
                [row["selected_cell_id"] for row in history["acquisitions"]],
                units.iloc[history["training_indices"][2:]]["cell_id"].tolist(),
            )
            self.assertNotIn("hit_auc", history["evaluations"][-1])
            self.assertEqual(session.fit_calls, 3)
            self.assertEqual(session.checkpoint_budgets, [2, 3, 4])
            self.assertEqual(session.clear_calls, 2)
            self.assertEqual(history["split_scheme"], "random_holdout")
            self.assertEqual(history["test_fraction"], 0.2)
            self.assertEqual(len(history["test_indices"]), 6)
            self.assertFalse(set(history["training_indices"]) & set(history["test_indices"]))
            self.assertEqual(
                history["test_cell_ids"],
                units.iloc[history["test_indices"]]["cell_id"].tolist(),
            )

    def test_random_holdout_is_deterministic_and_seed_specific(self):
        from run_experiments import random_holdout_indices
        first = random_holdout_indices(659, 172)
        second = random_holdout_indices(659, 172)
        other = random_holdout_indices(659, 136)
        self.assertEqual(len(first), 132)
        self.assertEqual(len(np.unique(first)), 132)
        self.assertTrue(np.all(first[1:] > first[:-1]))
        self.assertTrue(np.all((first >= 0) & (first < 659)))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_incomplete_history_rejects_split_mismatch(self):
        from run_experiments import ExperimentRunner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "C1" / "fold_0" / "seed_172"
            run_dir.mkdir(parents=True)
            history = {
                "status": "RUNNING", "method": "C1", "fold": 0, "seed": 172,
                "split_scheme": "random_holdout", "test_fraction": 0.2,
                "test_indices": [0, 1, 2, 3, 4, 5], "training_indices": [6, 7],
            }
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
            runner = object.__new__(ExperimentRunner)
            runner.output_root = root
            runner.units = pd.DataFrame({"cell_id": [str(i) for i in range(30)]})
            runner.grid = "1km"
            runner.split_scheme = "random_holdout"
            runner.test_fraction = 0.2
            with self.assertRaisesRegex(RuntimeError, "split mismatch"):
                runner.run("C1", 0, 172)

    def test_budget_cli_defaults_and_overrides(self):
        from run_experiments import build_parser, resolve_budget_options
        parser = build_parser()
        settings = {
            "protocol": {"initial_budget": 20, "sequential_steps": 20},
            "evaluation_budgets": [20, 40],
        }
        default_args = parser.parse_args(["--gpml-root", "gpml"])
        self.assertEqual(resolve_budget_options(default_args, settings), (20, [20, 40]))
        custom_args = parser.parse_args([
            "--gpml-root", "gpml", "--initial-budget", "40",
            "--evaluation-budgets", "80",
        ])
        self.assertEqual(resolve_budget_options(custom_args, settings), (40, [80]))

    def test_budget_cli_rejects_invalid_values(self):
        from run_experiments import build_parser, resolve_budget_options
        parser = build_parser()
        settings = {
            "protocol": {"initial_budget": 20, "sequential_steps": 20},
            "evaluation_budgets": [20, 40],
        }
        args = parser.parse_args([
            "--gpml-root", "gpml", "--initial-budget", "0",
        ])
        with self.assertRaisesRegex(ValueError, "initial budget"):
            resolve_budget_options(args, settings)


if __name__ == "__main__":
    unittest.main()
