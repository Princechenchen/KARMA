from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


MATLAB_DIR = Path(__file__).resolve().parents[1] / "matlab"


class AcquisitionTests(unittest.TestCase):
    def test_registry_has_thirteen_unique_methods(self):
        from acquisition import CORE_METHODS, METHODS, OPTIONAL_METHODS
        self.assertEqual(len(METHODS), 13)
        self.assertEqual(len(set(METHODS)), 13)
        self.assertEqual(set(CORE_METHODS) | set(OPTIONAL_METHODS), set(METHODS))
        self.assertFalse(set(CORE_METHODS) & set(OPTIONAL_METHODS))

    def test_c2_c3_c5_c10_c12_dispatch_to_original_implementations(self):
        source = (MATLAB_DIR / "mmrgp_select_next.m").read_text(encoding="utf-8")
        expected = {
            "C2": "mMeanMSE_norm",
            "C3": "mIMSE_refset",
            "C4": "PALROI_crimes",
            "C5": "mAK_MCS_IRS_unified",
            "C10": "Voronoi_MixedMetric",
            "C10-APPROX": "Voronoi_MixedMetric",
            "C12": "mGlobal_SUR2w_varcs02lPD_fast_QBC",
        }
        for method, function_name in expected.items():
            with self.subTest(method=method):
                block = source.split(f"case '{method}'", 1)[1].split("case '", 1)[0]
                self.assertIn(f"[x_star,model]={function_name}(model);", block)

    def test_c10_and_approx_use_original_approximate_loo_branch(self):
        from acquisition import parse_methods
        self.assertEqual(parse_methods("C10-Approx"), ["C10-Approx"])
        self.assertNotIn("C10-Approx", parse_methods("all"))
        source = (MATLAB_DIR / "mmrgp_select_next.m").read_text(encoding="utf-8")
        self.assertIn("model.c10_loo_exact=false", source)
        self.assertNotIn("model.c10_loo_exact=true", source)

    def test_retired_crimes_reimplementations_are_removed(self):
        retired = [
            "NMeanMSE_crimes.m", "IMSE_crimes.m", "AKMCSIRS_crimes.m",
            "QBCGlobal_crimes.m", "Voronoi_MixedMetric_crimes.m",
        ]
        self.assertEqual([name for name in retired if (MATLAB_DIR / name).exists()], [])

    def test_parse_core_and_all(self):
        from acquisition import CORE_METHODS, METHODS, parse_methods
        self.assertEqual(parse_methods("core"), CORE_METHODS)
        self.assertEqual(parse_methods("all"), METHODS)
        self.assertEqual(parse_methods("C1,KARMA"), ["C1", "KARMA"])

    def test_shared_lhs_is_unique_and_excludes_test_fold(self):
        from acquisition import initial_design, lhs_sequence
        units = pd.DataFrame({
            "x": np.linspace(0, 1, 125), "y": np.resize(np.linspace(0, 1, 12), 125),
            "spatial_block": np.arange(125) % 5,
        })
        eligible = np.flatnonzero(units["spatial_block"].to_numpy() != 0)
        first = initial_design(units, eligible, 172)
        second = initial_design(units, eligible, 172)
        sequence = lhs_sequence(units, eligible, 172)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, sequence[:20])
        self.assertEqual(len(np.unique(sequence)), 100)
        self.assertTrue(np.all(units.iloc[sequence]["spatial_block"].to_numpy() != 0))

    def test_roi_d_uses_no_knowledge(self):
        source = (MATLAB_DIR / "mmrgp_select_next.m").read_text(encoding="utf-8")
        self.assertIn("roi_d_constant_knowledge", source)
        self.assertIn("model.pi_comp_alpha=[0 0]", source)

    def test_single_channel_karma_ablations_have_explicit_active_masks(self):
        from acquisition import parse_methods
        self.assertEqual(parse_methods("KARMA-HG,KARMA-BND"), ["KARMA-HG", "KARMA-BND"])
        source = (MATLAB_DIR / "mmrgp_select_next.m").read_text(encoding="utf-8")
        for method, mask in (("KARMA-HG", "[1 0]"), ("KARMA-BND", "[0 1]")):
            with self.subTest(method=method):
                block = source.split(f"case '{method}'", 1)[1].split("case '", 1)[0]
                self.assertIn(f"model.pi_comp_active_mask={mask}", block)
                self.assertIn("model.pi_comp_adaptive=true", block)
        sampler = (MATLAB_DIR.parents[1] / "numeric_sample" /
                   "mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf.m").read_text(encoding="utf-8")
        self.assertIn("a_new(~active_mask)=0", sampler)
        self.assertIn("fused_fn=fn_hg", sampler)
        self.assertIn("fused_fn=fn_bnd", sampler)

    def test_pro_lhs_takes_precedence_over_legacy_two_input_version(self):
        source = (MATLAB_DIR / "mmrgp_select_next.m").read_text(encoding="utf-8")
        legacy = source.index("'MS-PSO'")
        pro = source.index("'MS-PSO-pro'")
        self.assertLess(
            legacy, pro,
            "MS-PSO-pro must be added last so its three-input LHS is not shadowed",
        )

    def test_qbc_reuses_original_with_portable_fit_and_generic_diagnostics(self):
        project_root = Path(__file__).resolve().parents[2]
        original = (project_root / "numeric_sample" / "mGlobal_SUR2w_varcs02lPD_fast_QBC.m").read_text(encoding="utf-8")
        loop = original.index("for k = 1:K")
        self.assertIn("MMRGP0(", original[loop:])
        self.assertIn("randperm", original[:loop])
        self.assertNotIn("MMRGP0_crimes", original)
        self.assertIn("size(zsd,2)>1", original)
        model_fit = (project_root / "numeric_sample" / "MMRGP0.m").read_text(encoding="utf-8")
        guarded = model_fit.split("[parent_dir, ~, ~] = fileparts(pwd);", 1)[0]
        self.assertIn("exist(''crimes_gpml_root'',''var'')", guarded)
        self.assertIn("MMRGP0_crimes(trainy,trainx,ClassPos,dm,gpml_root)", guarded)
        self.assertIn("return", guarded)
        self.assertFalse((MATLAB_DIR / "MMRGP0.m").exists())
        dispatcher = (MATLAB_DIR / "mmrgp_select_next.m").read_text(encoding="utf-8")
        self.assertIn("assignin('base','crimes_gpml_root',gpml_root)", dispatcher)


if __name__ == "__main__":
    unittest.main()
