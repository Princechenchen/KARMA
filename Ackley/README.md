# Ackley case (surrogate-quality facet)

Two continuous outputs of the Ackley function; knowledge = historical experience with wide anchors and a radial-shell boundary band.

## Reproducing the finalized archive

1. `main_compare_ackley_box3dim.m` — KARMA and the core baselines (ten seeds).
2. `add_qbc_to_ackley_final10.m` — runs the QBC-global arm and merges it into the finalized archive (`add_qbc_to_ackley_repro.m` does the same for the repro archive).

Base builder: `make_ackley_ext6_base.m`.

- Finalized archive: `Ackley_box3_final10.mat`; `Ackley_box3_repro.mat` is the one-step reproduction archive.
- Convergence figures: `../paper_output/plot_acc_paper.py`.

Dependencies: the shared library folders at the repository root (`numeric_sample/`, `src/`, `sampling_Funciton/`, `testFunc/`, `grid/`, `MRSM_subprograms1/`, `MS-PSO-pro/`, `third_party/GPML`); the driver adds them to the MATLAB path itself.
