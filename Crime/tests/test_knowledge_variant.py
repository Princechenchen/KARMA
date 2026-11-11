import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiments import build_parser, load_knowledge


class KnowledgeVariantTest(unittest.TestCase):
    def test_adjacent_variant_replaces_only_the_boundary_channel(self):
        original = load_knowledge("1km", variant="original")
        adjacent = load_knowledge("1km", variant="adjacent")

        self.assertEqual(len(original), len(adjacent))
        self.assertTrue(np.array_equal(original["cell_id"], adjacent["cell_id"]))
        self.assertTrue(np.allclose(original["pi_hg"], adjacent["pi_hg"]))
        self.assertTrue(np.allclose(adjacent["pi_bnd"], adjacent["pi_bnd_adjacent"]))
        self.assertFalse(np.allclose(original["pi_bnd"], adjacent["pi_bnd"]))

    def test_oracle2022_variants_replace_only_the_boundary_channel(self):
        original = load_knowledge("1km", variant="original")
        for variant, column in (
            ("oracle2022_threshold", "pi_bnd_threshold_2022"),
            ("oracle2022_adjacent", "pi_bnd_adjacent_2022"),
        ):
            with self.subTest(variant=variant):
                knowledge = load_knowledge("1km", variant=variant)
                self.assertEqual(len(knowledge), 657)
                self.assertTrue(np.array_equal(original["cell_id"], knowledge["cell_id"]))
                self.assertTrue(np.allclose(original["pi_hg"], knowledge["pi_hg"]))
                self.assertTrue(np.allclose(knowledge["pi_bnd"], knowledge[column]))
                self.assertFalse(np.allclose(original["pi_bnd"], knowledge["pi_bnd"]))

    def test_parser_accepts_both_oracle2022_variants(self):
        parser = build_parser()
        for variant in ("oracle2022_threshold", "oracle2022_adjacent"):
            with self.subTest(variant=variant):
                args = parser.parse_args([
                    "--gpml-root", "dummy", "--knowledge-variant", variant,
                ])
                self.assertEqual(args.knowledge_variant, variant)

    def test_hotspot_variant_loads_new_response_and_only_replaces_boundary_prior(self):
        from run_experiments import load_units
        original = load_knowledge("1km", variant="original")
        hotspot = load_knowledge("1km", variant="hotspot")
        units = load_units("1km", variant="hotspot")

        self.assertTrue(np.allclose(original["pi_hg"], hotspot["pi_hg"]))
        self.assertTrue(np.allclose(hotspot["pi_bnd"], hotspot["pi_bnd_hot"]))
        self.assertEqual(int(units["y2"].sum()), 198)
        self.assertEqual(units["y2_threshold"].unique().tolist(), [445.0])

    def test_parser_defaults_to_original_and_accepts_adjacent(self):
        parser = build_parser()
        default_args = parser.parse_args(["--gpml-root", "dummy"])
        adjacent_args = parser.parse_args(["--gpml-root", "dummy", "--knowledge-variant", "adjacent"])
        hotspot_args = parser.parse_args([
            "--gpml-root", "dummy", "--knowledge-variant", "hotspot",
            "--units-variant", "hotspot",
        ])

        self.assertEqual(default_args.knowledge_variant, "original")
        self.assertEqual(adjacent_args.knowledge_variant, "adjacent")
        self.assertEqual(hotspot_args.knowledge_variant, "hotspot")
        self.assertEqual(hotspot_args.units_variant, "hotspot")

    def test_hotspot_oracle_variants_replace_only_the_boundary_channel(self):
        base = load_knowledge("1km", variant="hotspot")
        for variant, column in (
            ("oracle2022_hotspot_threshold", "pi_bnd_hotspot_threshold_2022"),
            ("oracle2022_hotspot_adjacent", "pi_bnd_hotspot_adjacent_2022"),
        ):
            with self.subTest(variant=variant):
                knowledge = load_knowledge("1km", variant=variant)
                self.assertTrue(np.array_equal(base["cell_id"], knowledge["cell_id"]))
                self.assertTrue(np.allclose(base["pi_hg"], knowledge["pi_hg"]))
                self.assertTrue(np.allclose(knowledge["pi_bnd"], knowledge[column]))

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            load_knowledge("1km", variant="unknown")


if __name__ == "__main__":
    unittest.main()
