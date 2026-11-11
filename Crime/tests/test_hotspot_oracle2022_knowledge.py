import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiments import build_parser, load_knowledge


class HotspotOracle2022KnowledgeTests(unittest.TestCase):
    def test_builder_exists(self):
        self.assertTrue((ROOT / "build_hotspot_oracle2022_knowledge.py").exists())

    def test_hotspot_oracle_fields_preserve_support_and_intensity_prior(self):
        base = load_knowledge("1km", variant="hotspot")
        for variant, column in (
            ("oracle2022_hotspot_threshold", "pi_bnd_hotspot_threshold_2022"),
            ("oracle2022_hotspot_adjacent", "pi_bnd_hotspot_adjacent_2022"),
        ):
            with self.subTest(variant=variant):
                knowledge = load_knowledge("1km", variant=variant)
                self.assertEqual(len(knowledge), len(base))
                self.assertTrue(np.array_equal(base["cell_id"], knowledge["cell_id"]))
                self.assertTrue(np.allclose(base["pi_hg"], knowledge["pi_hg"]))
                self.assertTrue(np.allclose(knowledge["pi_bnd"], knowledge[column]))
                self.assertTrue(knowledge[column].between(0, 1).all())
                self.assertTrue(np.isfinite(knowledge[column]).all())

    def test_threshold_and_adjacent_fields_match_frozen_definitions(self):
        self.test_builder_exists()
        module = importlib.import_module("build_hotspot_oracle2022_knowledge")
        result = module.build_hotspot_oracle2022_knowledge()
        target = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km_hotspot.csv")
        common = result.merge(target[["cell_id", "y1", "is_classification_boundary"]], on="cell_id")
        z = np.log1p(common["y1"].to_numpy(dtype=float))
        expected = np.clip(
            np.exp(-((z - np.log1p(445.0)) ** 2) / (2 * 1.2171857664987304**2)),
            0.01, 1.0,
        )
        np.testing.assert_allclose(common["pi_bnd_hotspot_threshold_2022"], expected)
        boundary = common["is_classification_boundary"].to_numpy(dtype=bool)
        self.assertTrue(np.allclose(
            common.loc[boundary, "pi_bnd_hotspot_adjacent_2022"], 1.0,
        ))
        self.assertTrue(np.all(
            common.loc[~boundary, "pi_bnd_hotspot_adjacent_2022"] < 1.0,
        ))
        self.assertEqual(int(result["oracle2022_nearest_filled"].sum()), 8)

    def test_parser_accepts_hotspot_oracle_variants(self):
        parser = build_parser()
        for variant in ("oracle2022_hotspot_threshold", "oracle2022_hotspot_adjacent"):
            with self.subTest(variant=variant):
                args = parser.parse_args(["--gpml-root", "dummy", "--knowledge-variant", variant])
                self.assertEqual(args.knowledge_variant, variant)


if __name__ == "__main__":
    unittest.main()
