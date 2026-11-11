import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_budget_screening import verify_history, verify_pair


class BudgetVerificationTests(unittest.TestCase):
    def make_history(self, indices, method="KARMA"):
        return {
            "status": "COMPLETE", "method": method, "fold": 0, "seed": 172,
            "grid": "1km", "initial_budget": 30, "sequential_steps": 2,
            "total_budget": 32, "training_indices": list(range(30)) + indices,
            "acquisitions": [{"from_budget": 30}, {"from_budget": 31}],
            "evaluations": [
                {"budget": 30}, {"budget": 31},
                {"budget": 32, "region_pass": 0.5, "density_lift": 2.0,
                 "rmae": 0.8, "acc": 0.7, "hit": 0.25},
            ],
        }

    def write(self, root, name, payload):
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_history_and_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            karma = self.write(root, "karma.json", self.make_history([30, 31]))
            roi = self.write(root, "roi.json", self.make_history([30, 31], method="ROI-D"))
            self.assertEqual(verify_history(karma)["total_budget"], 32)
            self.assertEqual(verify_pair(karma, roi)["delta_region_pass"], 0.0)

    def test_pair_rejects_different_initial_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            karma_payload = self.make_history([30, 31])
            roi_payload = self.make_history([30, 31], method="ROI-D")
            roi_payload["training_indices"][1] = 99
            karma = self.write(root, "karma.json", karma_payload)
            roi = self.write(root, "roi.json", roi_payload)
            with self.assertRaisesRegex(AssertionError, "initial design"):
                verify_pair(karma, roi)


if __name__ == "__main__":
    unittest.main()
