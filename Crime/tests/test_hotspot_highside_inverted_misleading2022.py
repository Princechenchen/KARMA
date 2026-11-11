import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition import parse_methods
from build_adjacent_boundary_knowledge import build_adjacent_boundary_knowledge
from run_experiments import build_parser, load_knowledge


class HotspotHighsideInvertedMisleading2022Tests(unittest.TestCase):
    def test_inverted_field_reverses_only_highside_semantics(self):
        module = importlib.import_module("build_hotspot_highside_adjacent_oracle2022")
        result = module.build_hotspot_highside_adjacent_oracle2022()
        source = pd.read_csv(ROOT / "data" / "units" / "units_2022_1km_hotspot.csv")
        adjacent = build_adjacent_boundary_knowledge(source, sigma_cells=1.0)
        coordinates = source[["ix", "iy"]].to_numpy(dtype=float)
        weights = np.exp(-(cdist(coordinates, coordinates) ** 2) / 2.0)
        probability = weights @ source["y2"].to_numpy(dtype=float) / weights.sum(axis=1)
        boundary = adjacent["pi_bnd_adjacent"].to_numpy(dtype=float)
        expected = pd.DataFrame({
            "cell_id": source["cell_id"],
            "probability": probability,
            "aligned": boundary * probability,
            "inverted": boundary * (1.0 - probability),
            "decay": boundary,
        })
        common = result.merge(expected, on="cell_id", how="inner")
        self.assertEqual(len(common), 649)
        np.testing.assert_allclose(
            common["pi_bnd_highside_adjacent_2022"], common["aligned"],
        )
        np.testing.assert_allclose(
            common["pi_bnd_highside_inverted_2022"], common["inverted"],
        )
        np.testing.assert_allclose(
            common["pi_bnd_highside_adjacent_2022"]
            + common["pi_bnd_highside_inverted_2022"],
            common["decay"],
        )
        self.assertTrue(result["pi_bnd_highside_inverted_2022"].between(0, 1).all())

    def test_misleading_variant_replaces_only_boundary_channel(self):
        base = load_knowledge("1km", variant="hotspot")
        knowledge = load_knowledge(
            "1km", variant="misleading2022_hotspot_highside_inverted",
        )
        self.assertTrue(np.array_equal(base["cell_id"], knowledge["cell_id"]))
        np.testing.assert_allclose(base["pi_hg"], knowledge["pi_hg"])
        np.testing.assert_allclose(
            knowledge["pi_bnd"], knowledge["pi_bnd_highside_inverted_2022"],
        )

    def test_parser_and_registry_accept_misleading_smoke_arms(self):
        args = build_parser().parse_args([
            "--gpml-root", "dummy",
            "--knowledge-variant", "misleading2022_hotspot_highside_inverted",
            "--methods", "KARMA,KARMA-Fixed",
        ])
        self.assertEqual(
            args.knowledge_variant, "misleading2022_hotspot_highside_inverted",
        )
        self.assertEqual(parse_methods(args.methods), ["KARMA", "KARMA-Fixed"])

    def test_fixed_arm_keeps_full_misleading_prior_without_arbitration(self):
        source = (ROOT / "matlab" / "mmrgp_select_next.m").read_text(encoding="utf-8")
        block = source.split("case 'KARMA-FIXED'", 1)[1].split("case '", 1)[0]
        self.assertIn("model.pi_comp_adaptive=false", block)
        self.assertIn("model.pi_comp_alpha=[1 1]", block)
        self.assertIn(
            "mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf(model)", block,
        )


if __name__ == "__main__":
    unittest.main()
