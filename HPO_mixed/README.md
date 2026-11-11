# HPO-mixed case (Bayesian-optimization facet)

LightGBM hyperparameter optimization on a Covertype subset: one continuous output (validation accuracy) and one categorical output (overfitting flag); knowledge = community consensus (learning-rate sweet-range soft band + overfitting-envelope shell).

## Reproducing the finalized archive

Each `main_compare_hpo_mixed*.m` driver runs one arm configuration against the configuration bank; `build_hpo_final_repro.py` then merges the completed runs into the finalized archive `HPO_final_repro.mat`. `main_compare_hpo_mixed.m` is the base driver, and the `main_compare_hpo_mixed_c*.m` files are the per-arm configurations of the compared entities (including the EI arms).

Supporting files: `mGlobal_SUR2w_varcs02lPD_fast_EI.m` (BO-EI), `mROI_SUR2w_varcs02lPD_fast_PI_pil3_sob*.m` (KARMA-EI region samplers), `mLHS_maximin_pool.m` (MaximinLHS), `pi_hpo_prior.m` (knowledge field), `lint_hpo.m`, `analyze_pilot_prior.py`.

1. Truth bank: `python generate_bank.py` (the bundled `data/hpo_bank.*` already contains the 12,000-configuration bank).
2. Run the per-arm drivers `main_compare_hpo_mixed*.m` (the bank lookup through `hpo_eval_online.m` is the black-box oracle).
3. Run the KARMA-EI / KARMA-EI-arb / ROI-Rec arms: `main_hpo_bo_arms.m` (single-arm entries: `main_hpo_bo_c14.m`, `main_hpo_bo_c14b.m`, `main_hpo_bo_c15.m`; samplers `mROI_SUR2w_varcs02lPD_fast_PI_pil3_sob_DS_EI.m` and `..._DS_REC.m`).
3. Merge: `python build_hpo_final_repro.py`.

- Trusted-caliber figure: `fig_tbf_paper.py`; plain best-found figure: `fig_bf_paper.py`.

Data (`data/`): Covertype subset and split (`covtype.data.gz`, `covtype_split.npz`) and the configuration bank (`hpo_bank.mat/.npz`).
