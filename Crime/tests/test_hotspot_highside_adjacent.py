import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge
from run_experiments import build_parser, load_knowledge


class HotspotHighsideAdjacentTests(unittest.TestCase):
    def test_builder_exists(self):
        self.assertTrue((ROOT / "build_hotspot_highside_adjacent_knowledge.py").exists())

    def test_highside_field_matches_frozen_formula(self):
        self.test_builder_exists()
        module = importlib.import_module("build_hotspot_highside_adjacent_knowledge")
        result = module.build_hotspot_highside_adjacent_knowledge()
        source = pd.read_csv(ROOT / "data" / "units" / "units_2021_1km.csv")
        source["y2"] = (source["y1"].to_numpy(dtype=float) > 445.0).astype("int8")
        adjacent = build_adjacent_boundary_knowledge(source, sigma_cells=1.0)
        coordinates = source[["ix", "iy"]].to_numpy(dtype=float)
        weights = np.exp(-(cdist(coordinates, coordinates) ** 2) / 2.0)
        probability = weights @ source["y2"].to_numpy(dtype=float) / weights.sum(axis=1)
        expected = adjacent["pi_bnd_adjacent"].to_numpy(dtype=float) * probability
        np.testing.assert_allclose(result["hotspot_side_probability"], probability)
        np.testing.assert_allclose(result["pi_bnd_highside_adjacent"], expected)
        self.assertTrue(result["pi_bnd_highside_adjacent"].between(0, 1).all())
        self.assertTrue(np.isfinite(result["pi_bnd_highside_adjacent"]).all())

    def test_variant_replaces_only_boundary_channel(self):
        base = load_knowledge("1km", variant="hotspot")
        knowledge = load_knowledge("1km", variant="hotspot_highside_adjacent")
        self.assertTrue(np.array_equal(base["cell_id"], knowledge["cell_id"]))
        self.assertTrue(np.allclose(base["pi_hg"], knowledge["pi_hg"]))
        self.assertTrue(np.allclose(
            knowledge["pi_bnd"], knowledge["pi_bnd_highside_adjacent"],
        ))
        self.assertFalse(np.allclose(base["pi_bnd"], knowledge["pi_bnd"]))

    def test_parser_accepts_highside_variant(self):
        args = build_parser().parse_args([
            "--gpml-root", "dummy", "--knowledge-variant", "hotspot_highside_adjacent",
        ])
        self.assertEqual(args.knowledge_variant, "hotspot_highside_adjacent")


if __name__ == "__main__":
    unittest.main()
