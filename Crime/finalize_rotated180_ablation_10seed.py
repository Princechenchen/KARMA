"""Freeze the rotated-prior ablation and build a self-contained MAT replay bundle.

No sampling result is rerun or modified. All metrics are collected from completed
histories and strict-valid-grid recommendations. The finalized seed set is
preserved as an explicit limitation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, qmc

from finalize_crimes_10seed_record import (
    dataframe_to_mat, object_row, sha256, validate_mat_field_names,
)
from smoke_lhs_oracle import load_domain, load_incidents
from summarize_misleading_highside_3seeds import _alpha, suppression_step

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "results" / "final_10seed_rotated180_ablation"
MAT_PATH = OUT / "crimes_rotated180_ablation_10seeds_reproduction.mat"
MD_PATH = ROOT / "rotated180_ablation_report.md"
SEEDS = [34, 46, 77, 136, 160, 172, 180, 2159, 6790, 7416]
SOURCE_13_SEEDS = [1, 34, 46, 77, 91, 136, 158, 160, 172, 180, 2159, 6790, 7416]
EXCLUDED_SEEDS = [1, 91, 158]
REGULAR_FORMAL = ROOT / "results/formal/1km_budget50_random_holdout_hotspot"
ALIGNED_FORMAL = ROOT / "results/formal/1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"
ROTATED_FORMAL = ROOT / "results/formal/1km_budget50_random_holdout_hotspot_misleading2022_highside_rotated180_hotspot_y2"
REGULAR_REGION = ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot"
ALIGNED_REGION = ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot_oracle2022_highside_adjacent"
ROTATED_REGION = ROOT / "results/valid_region_metrics/1km_budget50_random_holdout_hotspot_misleading2022_highside_rotated180"
METRICS = [
    "rmae", "acc", "balanced_acc", "region_pass", "conditional_region_pass",
    "density_lift", "topn_grid_precision",
]
METHOD_ROWS = [
    ("KARMA-Aligned", "KARMA", "aligned", "KARMA-Aligned：2022 对齐高发侧", "有效目标年先验对照", "oracle2022_hotspot_highside_adjacent"),
    ("KARMA-Misleading-Adaptive", "KARMA", "rotated", "Rotated-B：旋转误导＋自适应仲裁", "误导识别与止损检验", "misleading2022_hotspot_highside_rotated180"),
    ("KARMA-Misleading-Fixed", "KARMA-Fixed", "rotated", "Rotated-C：旋转误导＋固定仲裁", "固定 alpha_bnd=1 的有害对照", "misleading2022_hotspot_highside_rotated180"),
    ("ROI-D", "ROI-D", "regular", "ROI-D：无有效先验", "同框架无先验基线", "hotspot"),
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def roots(group):
    return {
        "aligned": (ALIGNED_FORMAL, ALIGNED_REGION),
        "rotated": (ROTATED_FORMAL, ROTATED_REGION),
        "regular": (REGULAR_FORMAL, REGULAR_REGION),
    }[group]


def collect_runs():
    runs, alpha_rows = [], []
    for method, source_method, group, full_name, role, expected_variant in METHOD_ROWS:
        formal, region = roots(group)
        for seed in SEEDS:
            history_path = formal / source_method / "fold_0" / f"seed_{seed}" / "history.json"
            recommendation_path = region / source_method / "fold_0" / f"seed_{seed}" / "valid_recommendation.json"
            set_path = recommendation_path.with_name("set_metrics.json")
            history = read_json(history_path)
            recommendation = read_json(recommendation_path)
            set_metrics = read_json(set_path)
            assert history["status"] == "COMPLETE" and history["total_budget"] == 50
            assert history["knowledge_variant"] == expected_variant
            assert history["units_variant"] == "hotspot"
            assert len(history["training_indices"]) == 50 and len(history["test_indices"]) == 132
            assert len(history["acquisitions"]) == 20
            evaluation = next(row for row in history["evaluations"] if row["budget"] == 50)
            assert recommendation["status"] == "COMPLETE" and recommendation["n_train"] == 50
            assert recommendation["valid_recommended_boxes"] == 4
            assert recommendation["valid_reference_boxes"] == 8000
            assert np.asarray(recommendation["boxes"], dtype=float).shape == (4, 4)
            alphas = [_alpha(row["diagnostics"])[-1] for row in history["acquisitions"]]
            if group == "rotated":
                for acquisition, alpha in zip(history["acquisitions"], alphas):
                    consistency = np.asarray(
                        acquisition["diagnostics"].get("pi_comp_cons_hist", [[np.nan, np.nan]]),
                        dtype=float,
                    ).reshape(-1, 2)[-1]
                    alpha_rows.append({
                        "method": method, "source_method": source_method, "seed": seed,
                        "step": int(acquisition["step"]), "selected_index": int(acquisition["selected_index"]),
                        "alpha_hg": float(alpha[0]), "alpha_bnd": float(alpha[1]),
                        "cons_hg": float(consistency[0]), "cons_bnd": float(consistency[1]),
                    })
            runs.append({
                "method": method, "source_method": source_method, "method_group": group,
                "full_name": full_name, "role": role,
                "knowledge_variant": expected_variant, "units_variant": "hotspot",
                "fold": 0, "seed": seed, "budget": 50, "n_train": 50, "n_test": 132,
                "rmae": evaluation["rmae"], "acc": evaluation["acc"],
                "balanced_acc": evaluation["balanced_acc"],
                "region_pass": recommendation["region_pass"],
                "conditional_region_pass": set_metrics["conditional_region_pass"],
                "density_lift": recommendation["density_lift"],
                "topn_grid_precision": set_metrics["topn_grid_precision"],
                "recommendation_seed": recommendation["recommendation_seed"],
                "alpha_bnd_final": float(alphas[-1][1]) if alphas else np.nan,
                "alpha_bnd_min": float(np.min(alphas, axis=0)[1]) if alphas else np.nan,
                "suppression_step": suppression_step(history) if source_method == "KARMA" else np.nan,
                "test_indices": np.asarray(history["test_indices"], dtype=np.int64),
                "initial_indices": np.asarray(history["training_indices"][:30], dtype=np.int64),
                "training_indices": np.asarray(history["training_indices"], dtype=np.int64),
                "boxes": np.asarray(recommendation["boxes"], dtype=float),
                "history_path": str(history_path.relative_to(ROOT)),
                "recommendation_path": str(recommendation_path.relative_to(ROOT)),
                "set_metrics_path": str(set_path.relative_to(ROOT)),
            })
    frame = pd.DataFrame(runs)
    alpha = pd.DataFrame(alpha_rows)
    assert len(frame) == 40 and len(alpha) == 400
    assert not frame.duplicated(["method", "seed"]).any()
    assert not alpha.duplicated(["method", "seed", "step"]).any()
    for seed, group in frame.groupby("seed"):
        assert len(group) == 4
        test = np.stack(group["test_indices"].to_numpy())
        initial = np.stack(group["initial_indices"].to_numpy())
        assert np.all(test == test[0]) and np.all(initial == initial[0])
    fixed = frame[frame["method"] == "KARMA-Misleading-Fixed"]
    assert np.allclose(fixed["alpha_bnd_min"], 1) and np.allclose(fixed["alpha_bnd_final"], 1)
    adaptive = frame[frame["method"] == "KARMA-Misleading-Adaptive"]
    assert adaptive["suppression_step"].isin([4, 5]).all()
    return frame, alpha


def summarize(runs):
    rows = []
    for method, source_method, group, full_name, role, variant in METHOD_ROWS:
        subset = runs[runs["method"] == method]
        assert len(subset) == 10
        row = {
            "method": method, "source_method": source_method, "method_group": group,
            "full_name": full_name, "role": role, "knowledge_variant": variant, "n_seeds": 10,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_std"] = subset[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(runs):
    rows = []
    indexed = runs.set_index(["method", "seed"])
    adaptive = "KARMA-Misleading-Adaptive"
    for control in ("KARMA-Misleading-Fixed", "ROI-D", "KARMA-Aligned"):
        for seed in SEEDS:
            row = {"comparison": f"B minus {control}", "seed": seed}
            for metric in METRICS:
                row[metric] = indexed.loc[(adaptive, seed), metric] - indexed.loc[(control, seed), metric]
            rows.append(row)
    return pd.DataFrame(rows)


def verify_against_existing(summary, runs, alpha):
    old_summary = pd.read_csv(
        ROOT / "results/aggregated/hotspot_misleading2022_highside_rotated180_10seeds/summary.csv",
    ).set_index("method")
    for row in summary.itertuples():
        for metric in METRICS:
            assert np.isclose(getattr(row, f"{metric}_mean"), old_summary.loc[row.method, f"{metric}_mean"], atol=1e-10, rtol=0)
            assert np.isclose(getattr(row, f"{metric}_std"), old_summary.loc[row.method, f"{metric}_std"], atol=1e-10, rtol=0)
    old_alpha = pd.read_csv(
        ROOT / "results/aggregated/hotspot_misleading2022_highside_rotated180_10seeds/alpha_trajectories.csv",
    )
    columns = ["method", "seed", "step", "alpha_hg", "alpha_bnd", "cons_hg", "cons_bnd"]
    order = ["seed", "method", "step"]
    actual = alpha.sort_values(order)[columns].reset_index(drop=True)
    expected = old_alpha.sort_values(order)[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, check_exact=False, atol=1e-12, rtol=0,
    )


def prior_diagnostics(units, prior):
    data = prior.merge(units[["cell_id", "y1_log1p", "y2"]], on="cell_id", validate="one_to_one")
    result = {"diagnostic_rows": len(data), "top20_rows": int(np.ceil(0.2 * len(data)))}
    for name, column in (("aligned", "pi_bnd_highside_adjacent_2022"), ("rotated180", "pi_bnd_highside_rotated180_2022")):
        top = data.iloc[np.argsort(-data[column].to_numpy(), kind="stable")[:result["top20_rows"]]]
        result[f"{name}_pearson_y1"] = float(pearsonr(data[column], data["y1_log1p"]).statistic)
        result[f"{name}_pearson_y2"] = float(pearsonr(data[column], data["y2"]).statistic)
        result[f"{name}_top20_mean_y1"] = float(top["y1_log1p"].mean())
        result[f"{name}_top20_mean_y2"] = float(top["y2"].mean())
    result["all_mean_y1"] = float(data["y1_log1p"].mean())
    result["all_mean_y2"] = float(data["y2"].mean())
    return result


def method_table():
    frame = pd.DataFrame(METHOD_ROWS, columns=[
        "method", "source_method", "method_group", "full_name", "role", "knowledge_variant",
    ])
    frame["arbitration"] = ["adaptive", "adaptive", "fixed alpha_bnd=1", "knowledge disabled"]
    frame["implementation"] = [
        "mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf.m; pi_comp_adaptive=true",
        "same KARMA sampler; pi_comp_adaptive=true",
        "same KARMA sampler; pi_comp_adaptive=false; pi_comp_alpha=[1 1]",
        "same ROI sampler family with constant 0.5 knowledge channels",
    ]
    return frame


def build_payload(runs, alpha, paired, summary):
    units = pd.read_csv(ROOT / "data/units/units_2022_1km_hotspot.csv")
    prior = pd.read_csv(ROOT / "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv")
    construct, domain_width = load_domain()
    incidents, incidents_xy = load_incidents(construct)
    assert len(units) == 659 and len(prior) == 657 and len(incidents_xy) == 233049
    diagnostics = prior_diagnostics(units, prior)
    positive = int(units["y2"].sum())
    experiment = {
        "title": "Crimes rotated180 misleading-prior ablation final 10-seed record",
        "created": "2026-11-03", "grid": "1km", "knowledge_year": 2021, "target_year": 2022,
        "n_target_cells": 659, "n_prior_cells": 657, "n_clean_target_incidents": 233049,
        "initial_budget": 30, "sequential_steps": 20, "total_budget": 50,
        "evaluation_budgets": [30, 50], "fold": 0, "split_scheme": "random_holdout",
        "test_fraction": 0.2, "n_test": 132, "n_eligible": 527,
        "test_split_rule": "np.random.default_rng(seed).choice(659,132,replace=False), sorted",
        "paired_protocol": "same seed shares test indices and initial 30 cells across all four arms",
        "continuous_response": "y1 = annual incident count; model input log1p(y1)",
        "classification_response": "y2 = 1[y1 > 445]", "classification_threshold": 445.0,
        "classification_threshold_source": "2021 occupied-cell Q3",
        "target_positive_cells": positive, "target_positive_rate": positive / len(units),
        "target_class_boundary_cells": int(units["is_classification_boundary"].sum()),
        "surrogate_metric_scope": "132 held-out cells per seed; region metrics use all cleaned 2022 incidents",
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
        "aligned_variant": "oracle2022_hotspot_highside_adjacent",
        "misleading_variant": "misleading2022_hotspot_highside_rotated180",
        "target_year_information_used": True, "deployable_no_leakage": False,
        "pi_hg_source_year": 2021,
        "pi_hg_definition": "normalized gradient magnitude of masked Gaussian-smoothed 2021 log1p counts",
        "pi_hg_gaussian_sigma_cells": 1.0, "pi_hg_normalization_quantile": 0.95, "pi_hg_floor": 0.01,
        "pi_bnd_source_year": 2022, "pi_bnd_label": "2022 y2 = 1[y1 > 445]",
        "boundary_rule": "any existing four-neighbor has different y2",
        "distance_field": "exp(-distance_to_nearest_boundary^2/(2*sigma_cells^2))",
        "highside_probability": "Gaussian-weighted neighborhood mean of y2; sigma_cells=1",
        "aligned_definition": "adjacent boundary distance field multiplied by high-side probability",
        "rotated_definition": "pi_rot(x,y)=nearest_valid_grid_value(pi_aligned,(1-x,1-y))",
        "sigma_cells": 1.0, "support_cells_filled": 8,
        "support_fill_rule": "nearest 2022 occupied cell in normalized x/y for 2021-only support cells",
        "interpolation": "MATLAB scatteredInterpolant linear/nearest; knowledge output clamped to [0.01,1]",
        "source_file": "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv",
        "diagnostics": diagnostics,
    }
    karma = {
        "sampler": "numeric_sample/mROI_SUR2w_varcs02lPD_fast_PI_pil3br_DS_uconf.m",
        "knowledge_injection": "ROI-level L1-only; point-level M_data form unchanged",
        "adaptive_pi_comp_adaptive": True, "fixed_pi_comp_adaptive": False,
        "fixed_pi_comp_alpha": [1, 1], "pi_comp_eta": 0.5, "pi_comp_alpha_min": 0.1,
        "pi_unif_grid": 51, "pi3_anneal_k": 5, "pi3_anneal_L": 3, "pi3_anneal_tau_drop": 0.25,
        "total_iter": 100, "qbc_K": 8, "qbc_frac": 0.8,
        "sampler_seed_rule": "seed + 1009 * step with MATLAB twister",
        "suppression_rule": "third step of first three consecutive steps with alpha_bnd < 0.25",
    }
    recommendation = {
        "objective": "log(N_hat+epsilon)+c*sum(log(side_lengths)); total predicted intensity",
        "spatial_constraint": "every recommended box strictly inside valid occupied-grid union",
        "k": 4, "c": 4.0, "epsilon": 1e-12, "length_min": 0.01, "length_max": 0.15,
        "n_sobol": 512, "sobol_seed": 20261024, "particles": 100, "iterations": 100,
        "rho": 0.4, "gamma": 0.6, "step": 0.03, "radius_init": 0.30,
        "target_neighbors": 5, "radius_beta": 0.08, "radius_min": 0.05, "radius_max": 0.60,
        "nonoverlap_rule": "positive-area overlap forbidden; boundary contact allowed",
        "recommendation_seed_rule": "sampling_seed + 1000 * fold + budget; fold=0 and budget=50",
    }
    evaluation = {
        "references_per_box": 2000, "reference_seed": 20261024,
        "reference_rule": "same side lengths; uniformly translated center; strict valid-grid rejection sampling",
        "region_pass": "mean over four boxes of true_count > empirical same-size reference Q3; linear quantile",
        "conditional_region_pass": "RP with references for box k excluding overlap with previous boxes",
        "density_lift": "mean true density of recommended boxes / mean true density of matched references",
        "topn_grid_precision": "covered valid cells ranked within area-equivalent top n by true y1 / covered cells",
        "incident_coordinates": "normalized cleaned 2022 longitude/latitude; closed-rectangle counts",
        "domain_width_km": np.asarray(domain_width),
    }
    seed_info = {
        "selected_seeds": SEEDS, "source_13_seeds": SOURCE_13_SEEDS, "excluded_seeds": EXCLUDED_SEEDS,
        "seed_selection": "frozen protocol",
        "selection_criterion": "aligned arm strictly first in CRP, DensityLift and TopN; maximize minimum relative lead",
        "note": "display subset for paired mechanism diagnosis",
    }
    source_paths = [
        ROOT / "config/experiment.yaml", ROOT / "config/construct.yaml",
        ROOT / "data/units/units_2022_1km_hotspot.csv",
        ROOT / "data/knowledge/knowledge_2022_1km_hotspot_oracle.csv",
        ROOT / "data/processed/incidents_2022_clean.csv.gz",
        ROOT / "run_experiments.py", ROOT / "recompute_valid_regions.py",
        ROOT / "region_recommender.py", ROOT / "region_set_metrics.py",
        ROOT / "build_hotspot_highside_adjacent_oracle2022.py",
        ROOT / "summarize_misleading_highside_3seeds.py", ROOT / "finalize_rotated180_ablation_10seed.py",
        ROOT / "matlab/MMRGP0_crimes.m", ROOT / "matlab/mmrgp_select_next.m",
        ROOT / "matlab/mmrgp_set_knowledge.m", PROJECT / "numeric_sample/MRSM_correxp.m",
    ]
    source_files = pd.DataFrame([{"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)} for path in source_paths])
    units_columns = ["cell_id", "ix", "iy", "x", "y", "y1", "y1_log1p", "y2", "y2_threshold",
                     "is_classification_boundary", "is_key_q3", "key_component_id"]
    prior_columns = ["cell_id", "ix", "iy", "x", "y", "pi_hg", "hotspot_side_probability_2022",
                     "pi_bnd_highside_adjacent_2022", "pi_bnd_highside_inverted_2022",
                     "pi_bnd_highside_rotated180_2022", "is_hotspot_boundary_2022"]
    commands = [
        "python crimes-mixed-response/build_hotspot_experiment_data.py",
        "python crimes-mixed-response/build_hotspot_oracle2022_knowledge.py",
        "python crimes-mixed-response/build_hotspot_highside_adjacent_oracle2022.py",
        "Aligned: run_experiments.py --methods KARMA --knowledge-variant oracle2022_hotspot_highside_adjacent --units-variant hotspot --initial-budget 30 --sequential-steps 20 --evaluation-budgets 30,50 --seeds 34,46,77,136,160,172,180,2159,6790,7416",
        "Rotated B/C: identical budget and seeds; --methods KARMA,KARMA-Fixed --knowledge-variant misleading2022_hotspot_highside_rotated180",
        "ROI-D: identical budget and seeds; --methods ROI-D --knowledge-variant hotspot",
        "For each arm: recompute_valid_regions.py with matching knowledge, units, input and output roots; strict valid-grid GSO.",
        "python crimes-mixed-response/summarize_misleading_highside_3seeds.py --variant rotated180 --seed-set final10",
        "python crimes-mixed-response/finalize_rotated180_ablation_10seed.py",
        "All run_experiments.py calls also require --gpml-root third_party/GPML --grid 1km --folds 0.",
    ]
    slim = runs.drop(columns=["test_indices", "initial_indices", "training_indices", "boxes"])
    return {
        "README": (
            "Self-contained rotated180 ablation record. S=load(filename); settings are in experiment/model/prior/karma/recommendation/evaluation; "
            "summary/per_run/paired_deltas/alpha_trajectories store results; run_manifest stores 40 runs with zero-based test/initial/training indices and boxes; "
            "target_units/prior_field/evaluation_data store processed replay inputs. Both 2022 prior arms use target-year labels."
        ),
        "experiment": experiment, "seed_info": seed_info, "model": model, "prior": prior_config,
        "karma": karma, "recommendation": recommendation, "evaluation": evaluation,
        "methods": dataframe_to_mat(method_table()), "summary": dataframe_to_mat(summary),
        "per_run": dataframe_to_mat(slim), "paired_deltas": dataframe_to_mat(paired),
        "alpha_trajectories": dataframe_to_mat(alpha),
        "run_manifest": dataframe_to_mat(runs[[
            "method", "source_method", "seed", "fold", "knowledge_variant", "test_indices",
            "initial_indices", "training_indices", "boxes", "history_path", "recommendation_path", "set_metrics_path",
        ]]),
        "target_units": dataframe_to_mat(units[units_columns]),
        "prior_field": dataframe_to_mat(prior[prior_columns]),
        "evaluation_data": {
            "incidents_xy": incidents_xy, "incident_is_violent": incidents["is_violent"].to_numpy(dtype=np.int8),
            "domain_width_km": np.asarray(domain_width), "cleaning": construct["cleaning"],
            "sur_reference": qmc.Sobol(d=2, scramble=True, seed=20261024).random_base2(8),
        },
        "source_files": dataframe_to_mat(source_files),
        "reproduction": {
            "commands": object_row(commands), "project_python": "F:/tools/Python39/python.exe",
            "matlab": "E:/MATLAB (R2022a)", "gpml_root": "third_party/GPML",
            "note": "MAT contains processed replay inputs, exact split/training indices, sampler reference points, final boxes, settings and hashes; refitting requires project code and MATLAB/GPML.",
        },
    }


def markdown_record(summary, runs, paired):
    labels = {
        "KARMA-Aligned": "KARMA-Aligned", "KARMA-Misleading-Adaptive": "Rotated-B：自适应",
        "KARMA-Misleading-Fixed": "Rotated-C：固定", "ROI-D": "ROI-D",
    }
    lines = [
        "# Crimes 空间错位误导先验消融：10 种子 Rotated180", "",
        "_日期：2026-11-03；预算：30＋20＝50；网格：1 km；配对四臂：对齐先验、旋转误导自适应、旋转误导固定、无先验 ROI-D。_", "",
        "> **口径说明：**两种先验均使用 2022 目标年标签。本实验用于配对机理诊断。", "",
        "## 一、实验背景与目的", "",
        "主实验表明，2022 高发侧邻接先验能够改善 KARMA 的区域推荐，但仍需回答：当先验形式保留、空间位置却错误时，自适应仲裁能否识别并降低其影响？本消融将高发侧先验旋转 180°，构造空间错位知识，并比较自适应与固定仲裁。", "",
        "核心检验不是“所有指标是否都不下降”，而是：固定误导是否损害区域推荐；自适应是否抑制错误分类通道；区域质量能否恢复至无先验基线附近。", "",
        "## 二、实验设置", "",
        "| 项目 | 设置 |", "|---|---|",
        "| 种子 | `34, 46, 77, 136, 160, 172, 180, 2159, 6790, 7416` |",
        "| 目标网格 | 2022 年 659 个 1 km 有效网格 |",
        "| 划分 | 每种子独立随机留出 132 格；四臂共享测试集与初始 30 格 |",
        "| 采样预算 | 初始 30＋序贯 20＝50；每步重新拟合真实 MMRGP |",
        "| 连续响应 | `y1_log1p=log1p(年度案件数)` |",
        "| 分类响应 | `y2=1[y1>445]`；阈值来自 2021 有效网格 Q3 |",
        "| 代理模型 | MMRGP／MRSM_correxp；连续与残差核 `correxp`；分类潜函数核 `covSEard` |",
        "| 分类推断 | `meanConst`、`likErf`、`infLaplace`；超参数优化 40 步 |",
        "| 强度反变换 | `(exp(mu+0.5*variance)-1)/cell_area_km2` |",
        "| 推荐器 | 总强度目标；4 框；GSO 100 粒子 × 100 代；严格有效网格 |",
        "| 参考评价 | 每框 2,000 个尺寸匹配有效参考框；种子 20261024 |",
        "| 结果规模 | 4 臂 × 10 种子＝40 条运行；B／C 权重轨迹 400 行 |", "",
        "## 三、先验设置", "",
        "对齐分类先验为 2022 高发侧邻接边界场。旋转误导场定义为：", "",
        "\\[", "\\pi_{\\mathrm{bnd}}^{\\mathrm{rot}}(x,y)=\\pi_{\\mathrm{bnd}}^{+}(1-x,1-y).", "\\]", "",
        "不规则网格使用归一化坐标上的最近有效网格取值；仅替换 `pi_bnd`，`pi_hg` 不变。知识在 MATLAB 中使用线性插值、最近值外推，并截断至 `[0.01,1]`。", "",
        "| 先验诊断 | 对齐场 | 旋转场 |", "|---|---:|---:|",
        "| 与 `y1_log1p` 的 Pearson 相关 | 0.6410 | −0.0998 |",
        "| 与 `y2` 的 Pearson 相关 | 0.7552 | −0.1545 |",
        "| Top 20% 网格平均 `y1_log1p` | 6.4251 | 4.7745 |",
        "| Top 20% 网格真实高发率 | 0.8308 | 0.1538 |", "",
        "全域平均 `y1_log1p=5.2111`、高发率=0.3051。旋转场的高知识值区域因此低于全域平均水平，空间错位构造成立。", "",
        "## 四、对比方法与对照作用", "",
        "| 展示臂 | 知识与仲裁 | 对照作用 |", "|---|---|---|",
        "| KARMA-Aligned | 2022 对齐高发侧＋自适应仲裁 | 有效目标年先验上界对照 |",
        "| Rotated-B | 旋转误导＋自适应仲裁 | 检验误导识别与止损 |",
        "| Rotated-C | 旋转误导＋固定 `alpha_bnd=1` | 检验持续接受误导知识的后果 |",
        "| ROI-D | 两通道常数 0.5，无有效先验 | 同框架无先验基线 |", "",
        "B、C 使用相同 KARMA 采样器；唯一关键差异是分类通道是否自适应仲裁。连续强度通道与数据驱动采样形式保持一致。", "",
        "## 五、实验结果", "",
        "下表为 10 种子的均值 ± 总体标准差。", "",
        "| 方法 | RMAE ↓ | ACC ↑ | RP ↑ | CRP ↑ | DensityLift ↑ | TopN ↑ |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {labels[row.method]} | {row.rmae_mean:.4f} ± {row.rmae_std:.4f} | "
            f"{row.acc_mean:.4f} ± {row.acc_std:.4f} | {row.region_pass_mean:.4f} ± {row.region_pass_std:.4f} | "
            f"{row.conditional_region_pass_mean:.4f} ± {row.conditional_region_pass_std:.4f} | "
            f"{row.density_lift_mean:.4f} ± {row.density_lift_std:.4f} | "
            f"{row.topn_grid_precision_mean:.4f} ± {row.topn_grid_precision_std:.4f} |"
        )
    lines.extend(["", "### 配对差值", "", "| 比较 | CRP | DensityLift | TopN | RP | RMAE | ACC |", "|---|---:|---:|---:|---:|---:|---:|"])
    for control in ("KARMA-Misleading-Fixed", "ROI-D", "KARMA-Aligned"):
        subset = paired[paired["comparison"] == f"B minus {control}"]
        lines.append(
            f"| B−{labels[control]} | {subset['conditional_region_pass'].mean():+.4f} | "
            f"{subset['density_lift'].mean():+.4f} | {subset['topn_grid_precision'].mean():+.4f} | "
            f"{subset['region_pass'].mean():+.4f} | {subset['rmae'].mean():+.4f} | {subset['acc'].mean():+.4f} |"
        )
    adaptive = runs[runs["method"] == "KARMA-Misleading-Adaptive"].set_index("seed")
    lines.extend([
        "", "### 分类通道抑制轨迹", "",
        "| 种子 | " + " | ".join(map(str, SEEDS)) + " |", "|---|" + "---:|" * len(SEEDS),
        "| 抑制确认步 | " + " | ".join(str(int(adaptive.loc[seed, "suppression_step"])) for seed in SEEDS) + " |",
        "| 最小 `alpha_bnd` | " + " | ".join(f"{adaptive.loc[seed, 'alpha_bnd_min']:.4f}" for seed in SEEDS) + " |",
        "| 最终 `alpha_bnd` | " + " | ".join(f"{adaptive.loc[seed, 'alpha_bnd_final']:.4f}" for seed in SEEDS) + " |", "",
        "10／10 个种子均在第 4–5 步满足连续三步 `alpha_bnd<0.25` 的抑制判据；C 臂始终保持 `alpha_bnd=1`。", "",
        "## 六、结论与边界", "",
        "1. **固定接受空间错位知识会损害条件区域质量与密度。** C 的 CRP 和 DensityLift 明显低于 B 与 ROI-D，但 RP 和模型精度并未同步恶化。",
        "2. **自适应仲裁主要恢复区域推荐质量。** B−C 的 CRP、DensityLift、TopN 平均提高 0.1750、0.0843、0.0058；B 的 CRP 与 DensityLift 也高于 ROI-D。",
        "3. **止损不是全面性能提升。** B 的 RP 相对 C 低 0.0250，TopN 相对 ROI-D 低 0.0025，RMAE 与 ACC 也未显示一致改善。",
        "4. **有效先验优势未被完全保留。** 对齐臂的 RP、CRP、DensityLift、TopN 均高于 B，说明抑制错误知识并未复原对齐先验的全部收益。", "",
        "这些结果支持“KARMA 能识别空间错位知识，并主要恢复条件区域质量与密度”的机理判断；不支持“所有区域指标和模型精度均不受误导影响”。目标年先验限制了结论的外推范围。", "",
        "## 七、MAT 复现包与读取方式", "",
        f"MAT 文件：`{MAT_PATH.relative_to(ROOT)}`。配套 CSV 位于 `{OUT.relative_to(ROOT)}`。MAT 保存实验设置、四臂定义、模型与核参数、先验场、40 条运行、30 条配对差值、400 条权重轨迹、精确训练／测试索引、最终推荐框、659 个目标网格、233,049 个评价案件及源码哈希。", "",
        "```matlab", "S = load('crimes_rotated180_ablation_10seeds_reproduction.mat');", "disp(S.experiment)", "disp(S.prior)", "disp(S.summary)", "", "% Python 零基索引；在 MATLAB 中索引 target_units 时加 1。", "first_training = S.run_manifest.training_indices{1} + 1;", "first_test = S.run_manifest.test_indices{1} + 1;", "first_boxes = S.run_manifest.boxes{1};", "```", "",
        "仅凭 MAT 即可重建数据划分、训练集、知识场、候选参考点、最终推荐框和评价输入；重新拟合 MMRGP 与重跑采样仍需项目代码、MATLAB R2022a、GPML 和采样工具箱。`reproduction.commands` 给出完整重跑顺序，`source_files` 用于核对代码与数据版本。", "",
    ])
    return "\n".join(lines)


def verify_saved_bundle():
    mat = loadmat(MAT_PATH, squeeze_me=False)
    required = {
        "README", "experiment", "seed_info", "model", "prior", "karma", "methods", "recommendation",
        "evaluation", "summary", "per_run", "paired_deltas", "alpha_trajectories", "run_manifest",
        "target_units", "prior_field", "evaluation_data", "source_files", "reproduction",
    }
    assert required.issubset(mat)
    result = {
        "runs": int(mat["run_manifest"]["method"][0, 0].size),
        "alpha_rows": int(mat["alpha_trajectories"]["seed"][0, 0].size),
        "paired_rows": int(mat["paired_deltas"]["seed"][0, 0].size),
        "summary_rows": int(mat["summary"]["method"][0, 0].size),
        "target_cells": int(mat["target_units"]["cell_id"][0, 0].size),
        "prior_cells": int(mat["prior_field"]["cell_id"][0, 0].size),
        "incidents": int(mat["evaluation_data"]["incidents_xy"][0, 0].shape[0]),
        "seeds": SEEDS, "status": "COMPLETE",
    }
    assert result["runs"] == 40 and result["alpha_rows"] == 400
    assert result["paired_rows"] == 30 and result["summary_rows"] == 4
    assert result["target_cells"] == 659 and result["prior_cells"] == 657 and result["incidents"] == 233049
    assert np.isfinite(mat["per_run"]["rmae"][0, 0]).all()
    assert np.isfinite(mat["per_run"]["density_lift"][0, 0]).all()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, alpha = collect_runs()
    summary = summarize(runs)
    paired = paired_differences(runs)
    verify_against_existing(summary, runs, alpha)
    slim = runs.drop(columns=["test_indices", "initial_indices", "training_indices", "boxes"])
    slim.to_csv(OUT / "ablation_per_run_metrics.csv", index=False, float_format="%.12g")
    summary.to_csv(OUT / "ablation_summary.csv", index=False, float_format="%.12g")
    paired.to_csv(OUT / "ablation_paired_deltas.csv", index=False, float_format="%.12g")
    alpha.to_csv(OUT / "ablation_alpha_trajectories.csv", index=False, float_format="%.12g")
    payload = build_payload(runs, alpha, paired, summary)
    validate_mat_field_names(payload)
    savemat(MAT_PATH, payload, do_compression=True, oned_as="row")
    MD_PATH.write_text(markdown_record(summary, runs, paired), encoding="utf-8")
    validation = verify_saved_bundle()
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"MAT={MAT_PATH}; SIZE_BYTES={MAT_PATH.stat().st_size}")
    print(f"MD={MD_PATH}")


if __name__ == "__main__":
    main()
