# Ship-radar case (surrogate-quality facet)

Radar anti-jamming chamber experiment: seven inputs, two continuous responses and one categorical response; knowledge = electromagnetic principles + radar engineering practice. This is the case whose region identification uses MSPSO (supplementary Sec. III).

## Reproducing the finalized archive

1. `main_ship_prior_c8_v1box3u_dim_par.m` — KARMA and the core region baselines (ten seeds).
2. `main_ship_pibo_sub10_par.m` — the two piBO arms on the evaluated ten seeds; merge them into the archive with `add_pibo_to_ship_final10.m`.
3. `main_ship_qbcg_optb.m` — the QBC arm; merge it with `add_qbc_to_ship_final10.m`.

- Finalized archive: `Ship_v1box3u_final10.mat` (twelve arms = core arms + piBO and QBC merges).
- Convergence figures: `fig_ship_paper_*.png` via `plot_ship_paper.py`; twelve-arm version via `plot_ship_final12.py`.

Dependencies: the shared library folders at the repository root (numeric_sample/, src/, sampling_Funciton/, testFunc/, grid/, MRSM_subprograms1/, MS-PSO-pro/, third_party/GPML); the driver adds them to the MATLAB path itself.
