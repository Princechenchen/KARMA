import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition import parse_methods
from build_hotspot_highside_adjacent_oracle2022 import (
    build_hotspot_highside_adjacent_oracle2022,
)
from run_experiments import KNOWLEDGE_VARIANTS, build_parser, load_knowledge

VARIANT = "misleading2022_hotspot_highside_rotated180"
COLUMN = "pi_bnd_highside_rotated180_2022"


class HotspotHighsideRotated180Misleading2022Tests(unittest.TestCase):
    def test_rotated_field_nearest_samples_center_reflection_only(self):
        saved = pd.read_csv(
            ROOT / "data" / "knowledge" / "knowledge_2022_1km_hotspot_oracle.csv",
        )
        result = build_hotspot_highside_adjacent_oracle2022()
        self.assertIn(COLUMN, result.columns)

        coordinates = result[["x", "y"]].to_numpy(dtype=float)
        nearest = cKDTree(coordinates).query(1.0 - coordinates, k=1)[1]
        expected = result["pi_bnd_highside_adjacent_2022"].to_numpy()[nearest]
        np.testing.assert_allclose(result[COLUMN], expected)
        self.assertTrue(np.isfinite(result[COLUMN]).all())
        self.assertTrue(result[COLUMN].between(0, 1).all())
        self.assertFalse(np.allclose(result[COLUMN], expected * 0.0))
        for column in (
            "pi_hg", "pi_bnd_highside_adjacent_2022",
            "pi_bnd_highside_inverted_2022",
        ):
            np.testing.assert_allclose(result[column], saved[column])

    def test_misleading_variant_replaces_only_boundary_channel(self):
        self.assertIn(VARIANT, KNOWLEDGE_VARIANTS)
        base = load_knowledge("1km", variant="hotspot")
        knowledge = load_knowledge("1km", variant=VARIANT)
        self.assertTrue(np.array_equal(base["cell_id"], knowledge["cell_id"]))
        np.testing.assert_allclose(base["pi_hg"], knowledge["pi_hg"])
        np.testing.assert_allclose(knowledge["pi_bnd"], knowledge[COLUMN])

    def test_parser_and_registry_accept_rotated_smoke_arms(self):
        self.assertIn(VARIANT, KNOWLEDGE_VARIANTS)
        args = build_parser().parse_args([
            "--gpml-root", "dummy", "--knowledge-variant", VARIANT,
            "--methods", "KARMA,KARMA-Fixed",
        ])
        self.assertEqual(args.knowledge_variant, VARIANT)
        self.assertEqual(parse_methods(args.methods), ["KARMA", "KARMA-Fixed"])

    def test_rotated_variant_has_isolated_output_suffix(self):
        source = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
        self.assertIn(
            f'"{VARIANT}": "_hotspot_misleading2022_highside_rotated180"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
