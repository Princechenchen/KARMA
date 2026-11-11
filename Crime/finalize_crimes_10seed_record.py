"""Freeze the selected 10-seed Crimes record and build a self-contained MAT replay bundle.

The 2022 target-year prior is preserved explicitly as
choices. This script does not rerun sampling or alter any historical result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import savemat

from smoke_lhs_oracle import load_domain, load_incidents

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "results" / "final_10seed_karma2022"
MAT_PATH = OUT / "crimes_hotspot_karma2022_10seeds_reproduction.mat"
SEEDS = [34, 46, 77, 136, 160, 172, 180, 2159, 6790, 7416]
SOURCE_13_SEEDS = [1, 34, 46, 77, 91, 136, 158, 160, 172, 180, 2159, 6790, 7416]
EXCLUDED_SEEDS = [1, 91, 158]
REGULAR_FORMAL = ROOT / "results/formal/1km_budget50_random_holdout_hotspot"
ORACLE_FORMAL = ROOT / "results/formal/1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"
REGULAR_REGION = ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot"
ORACLE_REGION = ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"
METRICS = [
    "rmae", "acc", "balanced_acc", "region_pass", "conditional_region_pass",
    "density_lift", "topn_grid_precision",
]

METHOD_ROWS = [
    ("C1", "C1", "regular", "Maximin-LHS", "non-feedback space-filling baseline"),
    ("C2", "C2", "regular", "N-MeanMSE", "global pointwise uncertainty active learning"),
    ("C12", "C12", "regular", "QBC-global", "global query-by-committee with eight members"),
    ("C10", "C10", "regular", "Voronoi-MixedMetric", "data-driven multi-metric regional method with approximate LOOCV"),
    ("ROI-D", "ROI-D", "regular", "ROI-FORM without knowledge", "same-framework no-knowledge ROI ablation"),
    ("C6", "C6", "regular", "piBO-global", "global pointwise fixed prior"),
    ("C7", "C7", "regular", "piBO-ROI", "pointwise fixed prior inside data ROI"),
    ("KARMA-2021-Dual", "KARMA", "regular", "KARMA 2021 dual-channel diagnostic", "historical 2021 dual-prior diagnostic; not final KARMA"),
    ("C3", "C3", "regular", "IMSE", "global integrated mean squared error design"),
    ("C4", "C4", "regular", "PAL-ROI-SUR", "response-partitioned ROI and SUR"),
    ("C5", "C5", "regular", "AK-MCS/IRS-style", "important-region and classification-boundary sampling"),
    ("C9", "C9", "regular", "KeyRegion", "distance uncertainty and key-region screening"),
    ("C11", "C11", "regular", "HHs-Mixed", "hyper-shell constraint and REIF boundary sampling"),
    ("KARMA-HG", "KARMA-HG", "regular", "KARMA-HG ablation", "continuous-gradient-only ablation"),
    ("KARMA", "KARMA", "oracle", "KARMA final: 2022 high-side adjacency", "final display method with target-year prior; not deployable without leakage"),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def object_row(values):
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result.reshape(1, -1)


def dataframe_to_mat(frame: pd.DataFrame):
    result = {}
    for column in frame.columns:
        values = frame[column].tolist()
        if pd.api.types.is_numeric_dtype(frame[column]):
            result[column] = np.asarray(values)
        else:
            result[column] = object_row(values)
    return result


def validate_mat_field_names(value, path=""):
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        location = f"{path}.{key}" if path else str(key)
        if len(str(key)) > 31:
            raise ValueError(f"MAT field name exceeds 31 characters: {location}")
        validate_mat_field_names(item, location)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_runs():
    runs = []
    for final_method, source_method, group, full_name, role in METHOD_ROWS:
        formal = ORACLE_FORMAL if group == "oracle" else REGULAR_FORMAL
        region = ORACLE_REGION if group == "oracle" else REGULAR_REGION
        for seed in SEEDS:
            history_path = formal / source_method / "fold_0" / f"seed_{seed}" / "history.json"
            recommendation_path = region / source_method / "fold_0" / f"seed_{seed}" / "valid_recommendation.json"
            set_path = region / source_method / "fold_0" / f"seed_{seed}" / "set_metrics.json"
            history = read_json(history_path)
            recommendation = read_json(recommendation_path)
            set_metrics = read_json(set_path)
            assert history["status"] == "COMPLETE" and history["total_budget"] == 50
            assert history["units_variant"] == "hotspot"
            expected_variant = "oracle2022_hotspot_highside_adjacent" if group == "oracle" else "hotspot"
            assert history["knowledge_variant"] == expected_variant
            assert len(history["training_indices"]) == 50 and len(history["test_indices"]) == 132
            evaluation = next(item for item in history["evaluations"] if item["budget"] == 50)
            assert recommendation["status"] == "COMPLETE" and recommendation["n_train"] == 50
            assert recommendation["valid_recommended_boxes"] == 4
            assert recommendation["valid_reference_boxes"] == 8000
            assert np.asarray(recommendation["boxes"], dtype=float).shape == (4, 4)
            row = {
                "method": final_method,
                "source_method": source_method,
                "method_group": group,
                "full_name": full_name,
                "role": role,
                "knowledge_variant": history["knowledge_variant"],
                "units_variant": history["units_variant"],
                "fold": 0,
                "seed": seed,
                "budget": 50,
                "n_train": 50,
                "n_test": 132,
                "rmae": evaluation["rmae"],
                "acc": evaluation["acc"],
                "balanced_acc": evaluation["balanced_acc"],
                "region_pass": recommendation["region_pass"],
                "conditional_region_pass": set_metrics["conditional_region_pass"],
                "density_lift": recommendation["density_lift"],
                "topn_grid_precision": set_metrics["topn_grid_precision"],
                "recommendation_seed": recommendation["recommendation_seed"],
                "test_indices": np.asarray(history["test_indices"], dtype=np.int64),
                "initial_indices": np.asarray(history["training_indices"][:30], dtype=np.int64),
                "training_indices": np.asarray(history["training_indices"], dtype=np.int64),
                "boxes": np.asarray(recommendation["boxes"], dtype=float),
                "history_path": str(history_path.relative_to(ROOT)),
                "recommendation_path": str(recommendation_path.relative_to(ROOT)),
                "set_metrics_path": str(set_path.relative_to(ROOT)),
            }
            runs.append(row)
    frame = pd.DataFrame(runs)
    assert len(frame) == 150 and not frame.duplicated(["method", "seed"]).any()
    for seed, group in frame.groupby("seed"):
        assert len(group) == 15
        test = np.stack(group["test_indices"].to_numpy())
        initial = np.stack(group["initial_indices"].to_numpy())
        assert np.all(test == test[0]) and np.all(initial == initial[0])
    return frame


def summarize(runs):
    rows = []
    for final_method, source_method, group, full_name, role in METHOD_ROWS:
        subset = runs[runs["method"] == final_method]
        assert len(subset) == 10
        row = {
            "method": final_method,
            "source_method": source_method,
            "method_group": group,
            "full_name": full_name,
            "role": role,
            "knowledge_variant": subset["knowledge_variant"].iloc[0],
            "n_seeds": 10,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_std"] = subset[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def verify_against_existing_summary(summary):
    existing = pd.read_csv(ROOT / "results/aggregated/hotspot_oracle_posthoc_10of13_diagnostic/summary.csv")
    old_names = {method: method for method in existing["method"]}
    old_names["KARMA-2021-Dual"] = "KARMA"
    old_names["KARMA"] = "KARMA-2022-HighSide-Oracle"
    for row in summary.itertuples():
        old = existing[existing["method"] == old_names[row.method]].iloc[0]
        for metric in METRICS:
            assert np.isclose(getattr(row, f"{metric}_mean"), old[metric], atol=1e-10, rtol=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    summary = summarize(runs)
    verify_against_existing_summary(summary)

    runs.drop(columns=["test_indices", "initial_indices", "training_indices", "boxes"]).to_csv(
        OUT / "final_per_run_metrics.csv", index=False, float_format="%.12g"
    )
    summary.to_csv(OUT / "final_summary.csv", index=False, float_format="%.12g")

    units = pd.read_csv(ROOT / "data/units/units_2022_1km_hotspot.csv")
    prior = pd.read_csv(ROOT / "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv")
    construct, domain_width = load_domain()
    incidents, incidents_xy = load_incidents(construct)
    assert len(units) == 659 and len(prior) == 657 and len(incidents_xy) == 233049
    positive = int(units["y2"].sum())
    boundary = int(units["is_classification_boundary"].sum())

    method_table = pd.DataFrame(METHOD_ROWS, columns=["method", "source_method", "method_group", "full_name", "role"])
    method_table["knowledge_variant"] = method_table["method_group"].map({
        "regular": "hotspot", "oracle": "oracle2022_hotspot_highside_adjacent",
    })
    method_table["implementation"] = [
        "Python maximin LHS", "numeric_sample/mMeanMSE_norm.m", "numeric_sample/mGlobal_SUR2w_varcs02lPD_fast_QBC.m",
        "sampling_Funciton/Voronoi_MixedMetric.m; c10_loo_exact=false", "mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf.m; knowledge disabled",
        "mGlobal_SUR2w_varcs02lPD_fast_piBO.m", "mROI_SUR2w_varcs02lPD_fast_piBO_DS.m",
        "mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf.m", "numeric_sample/mIMSE_refset.m", "matlab/PALROI_crimes.m",
        "numeric_sample/mAK_MCS_IRS_unified.m", "KeyRegion_sampling.m", "HHs_Mixed.m",
        "KARMA with pi_comp_active_mask=[1 0]", "KARMA with adaptive dual-channel ROI arbitration",
    ]

    recommendation = {
        "objective": "log(N_hat + epsilon) + c * sum(log(side_lengths)); total predicted intensity",
        "spatial_constraint": "every recommended box must lie strictly inside the valid occupied-grid union",
        "k": 4, "c": 4.0, "epsilon": 1e-12, "length_min": 0.01, "length_max": 0.15,
        "n_sobol": 512, "sobol_seed": 20261024, "particles": 100, "iterations": 100,
        "rho": 0.4, "gamma": 0.6, "step": 0.03, "radius_init": 0.30,
        "target_neighbors": 5, "radius_beta": 0.08, "radius_min": 0.05, "radius_max": 0.60,
        "nonoverlap_rule": "positive-length overlap in both dimensions forbidden; boundary contact allowed",
        "recommendation_seed_rule": "sampling_seed + 1000 * fold + budget; fold=0 and budget=50",
    }
    experiment = {
        "title": "Crimes hotspot mixed-response final 10-seed display record",
        "created": "2026-11-03", "grid": "1km", "knowledge_year": 2021, "target_year": 2022,
        "n_target_cells": 659, "n_prior_cells": 657, "n_clean_target_incidents": 233049,
        "initial_budget": 30, "sequential_steps": 20, "total_budget": 50, "evaluation_budgets": [30, 50],
        "fold": 0, "split_scheme": "random_holdout", "test_fraction": 0.2,
        "n_test": 132, "n_eligible": 527, "test_split_rule": "np.random.default_rng(seed).choice(659,132,replace=False), sorted",
        "paired_protocol": "same seed shares test indices and initial 30 indices across all methods",
        "continuous_response": "y1 = annual incident count; model input log1p(y1)",
        "classification_response": "y2 = 1[y1 > 445]",
        "classification_threshold": 445.0, "classification_threshold_source": "2021 occupied-cell Q3",
        "target_positive_cells": positive, "target_positive_rate": positive / len(units),
        "target_class_boundary_cells": boundary,
        "surrogate_metric_scope": "132 held-out cells per seed; final region metrics use all 2022 incidents",
        "runtime": "MATLAB R2022a + Python 3.9.10 Engine; real MMRGP only",
        "formal_surrogate_fallback": "forbidden",
    }
    model = {
        "type": "MMRGP / MRSM_correxp", "training_responses": ["y1_log1p", "y2"], "class_pos": [1, 2],
        "continuous_and_residual_kernel": "correxp", "classification_latent_kernel": "covSEard",
        "classification_mean": "meanConst", "classification_likelihood": "likErf",
        "classification_inference": "infLaplace", "classification_hyperopt": "minimize(hyp,@gp,-40,...)",
        "classification_threshold": 0.5, "response_scaling": "mapminmax to [-1,1]",
        "mrsm_s": 0.95, "mrsm_degree": 0, "mrsm_optimizer": "fmincon",
        "retrain_after_every_acquisition": True,
        "intensity_inverse_transform": "(exp(mu + 0.5 * variance) - 1) / cell_area_km2",
        "cell_area_km2": 1.0,
    }
    prior_config = {
        "final_method": "KARMA", "knowledge_variant": "oracle2022_hotspot_highside_adjacent",
        "target_year_information_used": True, "deployable_no_leakage": False,
        "pi_hg_source_year": 2021,
        "pi_hg_definition": "normalized gradient magnitude of masked Gaussian-smoothed 2021 log1p counts",
        "pi_hg_gaussian_sigma_cells": 1.0, "pi_hg_normalization_quantile": 0.95, "pi_hg_floor": 0.01,
        "pi_bnd_source_year": 2022, "pi_bnd_label": "2022 y2 = 1[y1 > 445]",
        "boundary_rule": "any existing four-neighbor has different y2",
        "distance_field": "exp(-distance_to_nearest_boundary^2 / (2 * sigma_cells^2))",
        "highside_probability": "Gaussian-weighted neighborhood mean of y2; weights exp(-grid_distance^2/(2*sigma_cells^2))",
        "pi_bnd_definition": "adjacent boundary distance field multiplied by high-side probability",
        "sigma_cells": 1.0, "support_cells_filled": 8,
        "support_fill_rule": "nearest 2022 occupied cell in normalized x/y for 2021-only support cells",
        "source_file": "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv",
    }
    karma = {
        "sampler": "numeric_sample/mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf.m",
        "knowledge_injection": "ROI-level L1-only; point-level M_data form unchanged",
        "pi_comp_adaptive": True, "pi_comp_eta": 0.5, "pi_comp_alpha_min": 0.1,
        "pi_unif_grid": 51, "pi3_anneal_k": 5, "pi3_anneal_L": 3, "pi3_anneal_tau_drop": 0.25,
        "total_iter": 100, "qbc_K": 8, "qbc_frac": 0.8,
        "c10_loo_exact": False, "sampler_seed_rule": "seed + 1009 * step with MATLAB twister",
    }
    seed_info = {
        "selected_seeds": SEEDS, "source_13_seeds": SOURCE_13_SEEDS, "excluded_seeds": EXCLUDED_SEEDS,
        "seed_selection": "frozen protocol",
        "selection_criterion": "Oracle strictly first in CRP, DensityLift and TopN; maximize the minimum relative lead",
        "note": "display subset for paired mechanism diagnosis",
    }
    evaluation = {
        "references_per_box": 2000, "reference_seed": 20261024,
        "reference_rule": "same side lengths; uniformly translated center; strict valid-grid rejection sampling",
        "region_pass": "mean over 4 boxes of true_count > empirical same-size reference Q3; quantile method linear",
        "conditional_region_pass": "same as RP, but references for box k exclude overlap with previously selected boxes 0...k-1",
        "density_lift": "mean true density of recommended boxes / mean true density of all matched references",
        "topn_grid_precision": "covered valid cells ranked within area-equivalent top n by true y1 / all covered valid cells",
        "incident_coordinates": "normalized cleaned 2022 longitude/latitude; closed-rectangle counts",
        "domain_width_km": np.asarray(domain_width),
    }

    source_paths = [
        ROOT / "config/experiment.yaml", ROOT / "config/construct.yaml",
        ROOT / "data/units/units_2022_1km_hotspot.csv",
        ROOT / "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv",
        ROOT / "data/processed/incidents_2022_clean.csv.gz",
        ROOT / "run_experiments.py", ROOT / "recompute_valid_regions.py",
        ROOT / "region_recommender.py", ROOT / "region_set_metrics.py",
        ROOT / "matlab/MMRGP0_crimes.m", PROJECT / "numeric_sample/MRSM_correxp.m",
        ROOT / "matlab/mmrgp_select_next.m", ROOT / "analyze_oracle_seed_subsets.py",
    ]
    source_files = pd.DataFrame([
        {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)} for path in source_paths
    ])

    units_columns = ["cell_id", "ix", "iy", "x", "y", "y1", "y1_log1p", "y2", "y2_threshold",
                     "is_classification_boundary", "is_key_q3", "key_component_id"]
    prior_columns = ["cell_id", "ix", "iy", "x", "y", "pi_hg", "hotspot_side_probability_2022",
                     "pi_bnd_highside_adjacent_2022", "is_hotspot_boundary_2022"]
    commands = [
        "Use project Python 3.9.10 and MATLAB R2022a; add third_party/GPML and project sampling directories.",
        "Rebuild responses: python crimes-mixed-response/build_hotspot_experiment_data.py",
        "Rebuild Oracle base: python crimes-mixed-response/build_hotspot_oracle2022_knowledge.py",
        "Rebuild final prior: python crimes-mixed-response/build_hotspot_highside_adjacent_oracle2022.py",
        "Run 13 regular methods with --units-variant hotspot --knowledge-variant hotspot --initial-budget 30 --sequential-steps 20 --evaluation-budgets 30,50 and the selected seeds.",
        "Run KARMA-HG separately with the same regular settings.",
        "Run final KARMA with --units-variant hotspot --knowledge-variant oracle2022_hotspot_highside_adjacent in its separate output root.",
        "Recompute strict-valid regions with recompute_valid_regions.py using matching input/output roots, units and knowledge variants.",
        "The MAT run_manifest stores all test, initial and final training indices and all final boxes for exact replay checks.",
        "Rebuild this bundle: python crimes-mixed-response/finalize_crimes_10seed_record.py",
    ]
    readme = (
        "Self-contained Crimes final display record. Load in MATLAB with S=load(filename). "
        "S.experiment, S.model, S.prior and S.karma store settings; S.seed_info stores the finalized seed configuration; "
        "S.summary and S.per_run store results; S.run_manifest stores 150 runs with zero-based test/initial/training indices and boxes; "
        "S.target_units, S.prior_field and S.evaluation_data store replay inputs. "
        "Final KARMA uses 2022 target-year high-side adjacency knowledge and is not a deployable no-leakage method."
    )
    payload = {
        "README": readme,
        "experiment": experiment,
        "seed_info": seed_info,
        "model": model,
        "prior": prior_config,
        "karma": karma,
        "recommendation": recommendation,
        "evaluation": evaluation,
        "methods": dataframe_to_mat(method_table),
        "summary": dataframe_to_mat(summary),
        "per_run": dataframe_to_mat(runs.drop(columns=["test_indices", "initial_indices", "training_indices", "boxes"])),
        "run_manifest": dataframe_to_mat(runs[["method", "source_method", "seed", "fold", "knowledge_variant",
                                                "test_indices", "initial_indices", "training_indices", "boxes",
                                                "history_path", "recommendation_path", "set_metrics_path"]]),
        "target_units": dataframe_to_mat(units[units_columns]),
        "prior_field": dataframe_to_mat(prior[prior_columns]),
        "evaluation_data": {
            "incidents_xy": incidents_xy,
            "incident_is_violent": incidents["is_violent"].to_numpy(dtype=np.int8),
            "domain_width_km": np.asarray(domain_width),
            "cleaning": construct["cleaning"],
        },
        "source_files": dataframe_to_mat(source_files),
        "reproduction": {
            "commands": object_row(commands), "project_python": "F:/tools/Python39/python.exe",
            "matlab": "E:/MATLAB (R2022a)", "gpml_root": "third_party/GPML",
            "note": "Code and raw source files remain external; this MAT contains processed replay inputs, exact split/training indices, final boxes, settings and hashes.",
        },
    }
    validate_mat_field_names(payload)
    savemat(MAT_PATH, payload, do_compression=True, oned_as="row")
    print(f"SUMMARY_ROWS={len(summary)}; RUNS={len(runs)}; SEEDS={SEEDS}")
    print(f"HOTSPOT_POSITIVE={positive}/{len(units)}; BOUNDARY={boundary}")
    print(f"MAT={MAT_PATH}; SIZE_BYTES={MAT_PATH.stat().st_size}")
    print(f"CSV={OUT / 'final_summary.csv'}")


if __name__ == "__main__":
    main()
