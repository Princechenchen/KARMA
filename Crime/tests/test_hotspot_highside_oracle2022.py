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


class HotspotHighsideOracle2022Tests(unittest.TestCase):
    def test_oracle_highside_field_matches_frozen_formula(self):
        path = ROOT / "build_hotspot_highside_adjacent_oracle2022.py"
        self.assertTrue(path.exists())
        module = importlib.import_module("build_hotspot_highside_adjacent_oracle2022")
        result = module.build_hotspot_highside_adjacent_oracle2022()
        source = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km_hotspot.csv")
        adjacent = build_adjacent_boundary_knowledge(source, sigma_cells=1.0)
        coordinates = source[["ix", "iy"]].to_numpy(dtype=float)
        weights = np.exp(-(cdist(coordinates, coordinates) ** 2) / 2.0)
        probability = weights @ source["y2"].to_numpy(dtype=float) / weights.sum(axis=1)
        expected = pd.DataFrame({
            "cell_id": source["cell_id"],
            "probability": probability,
            "boundary": adjacent["pi_bnd_adjacent"].to_numpy(dtype=float) * probability,
        })
        common = result.merge(expected, on="cell_id", how="inner")
        self.assertEqual(len(common), 649)
        np.testing.assert_allclose(common["hotspot_side_probability_2022"], common["probability"])
        np.testing.assert_allclose(common["pi_bnd_highside_adjacent_2022"], common["boundary"])
        self.assertEqual(int(result["oracle2022_nearest_filled"].sum()), 8)
        self.assertTrue(result["pi_bnd_highside_adjacent_2022"].between(0, 1).all())

    def test_oracle_variant_replaces_only_boundary_channel(self):
        base = load_knowledge("1km", variant="hotspot")
        knowledge = load_knowledge("1km", variant="oracle2022_hotspot_highside_adjacent")
        self.assertTrue(np.array_equal(base["cell_id"], knowledge["cell_id"]))
        self.assertTrue(np.allclose(base["pi_hg"], knowledge["pi_hg"]))
        self.assertTrue(np.allclose(
            knowledge["pi_bnd"], knowledge["pi_bnd_highside_adjacent_2022"],
        ))

    def test_parser_accepts_oracle_highside_variant(self):
        args = build_parser().parse_args([
            "--gpml-root", "dummy",
            "--knowledge-variant", "oracle2022_hotspot_highside_adjacent",
        ])
        self.assertEqual(args.knowledge_variant, "oracle2022_hotspot_highside_adjacent")


if __name__ == "__main__":
    unittest.main()
