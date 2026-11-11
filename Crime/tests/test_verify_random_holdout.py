import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_experiments import random_holdout_indices


class RandomHoldoutVerificationTests(unittest.TestCase):
    def write_history(self, root, method, seed):
        test = random_holdout_indices(659, seed).tolist()
        eligible = np.setdiff1d(np.arange(659), np.asarray(test))
        train = eligible[:50].tolist()
        history = {
            "status": "COMPLETE", "method": method, "fold": 0, "seed": seed,
            "grid": "1km", "initial_budget": 30, "sequential_steps": 20,
            "total_budget": 50, "split_scheme": "random_holdout",
            "test_fraction": 0.2, "test_indices": test,
            "test_cell_ids": [str(i) for i in test],
            "training_indices": train,
            "training_cell_ids": [str(i) for i in train],
            "acquisitions": [{"from_budget": b} for b in range(30, 50)],
            "evaluations": [{"budget": b} for b in range(30, 51)],
        }
        path = root / method / "fold_0" / f"seed_{seed}" / "history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history), encoding="utf-8")
        return path

    def test_group_verifies_shared_and_seed_specific_splits(self):
        from verify_random_holdout import verify_random_holdout_group
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (172, 136):
                for method in ("C1", "KARMA"):
                    self.write_history(root, method, seed)
            report = verify_random_holdout_group(root, ("C1", "KARMA"), (172, 136))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["n_histories"], 4)

    def test_group_rejects_cross_method_split_mismatch(self):
        from verify_random_holdout import verify_random_holdout_group
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (172, 136):
                for method in ("C1", "KARMA"):
                    path = self.write_history(root, method, seed)
            history = json.loads(path.read_text(encoding="utf-8"))
            history["test_indices"] = history["test_indices"][::-1]
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "test split mismatch"):
                verify_random_holdout_group(root, ("C1", "KARMA"), (172, 136))


if __name__ == "__main__":
    unittest.main()
