# Branin case (surrogate-quality facet)

Two continuous outputs of the Branin function; knowledge = historical experience from related tasks (three soft-box difficulty centers + a limit-state band on log y).

## Reproducing the finalized archive

The finalized archive is the merge of several drivers. Run them in this order:

1. `main_compare_branin_box3d.m` — KARMA and the core baselines (ten seeds, box3d configuration).
2. `main_compare_branin_qbcg.m` — the QBC-global arm, then merge it into the archive with `post_branin_qbcg_merge.m`.


- Finalized archive: `Branin_box3d_final10.mat`; `Branin_box3d_repro.mat` is the one-step reproduction archive.
- Convergence figures for Branin and Ackley: `../paper_output/plot_acc_paper.py`.

Dependencies: the shared library folders at the repository root (`numeric_sample/`, `src/`, `sampling_Funciton/`, `testFunc/`, `grid/`, `MRSM_subprograms1/`, `MS-PSO-pro/`, `third_party/GPML`); the driver adds them to the MATLAB path itself.
