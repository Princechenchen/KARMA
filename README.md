# Trust, but Verify: Online-Arbitrated Domain Knowledge for Active Data Acquisition from Expensive Black-Box Sources with Mixed Outputs

Code and data for the ICDE 2027 submission. One folder per experimental case, with the shared method libraries at the repository root.

## Repository layout

```
Branin/       Branin numerical case (surrogate-quality facet)
Ackley/       Ackley numerical case (surrogate-quality facet)
Wingweight/   Wingweight engineering case (surrogate-quality facet)
Ship/         Ship-radar case (surrogate-quality facet)
HPO_mixed/    LightGBM hyperparameter optimization case (Bayesian-optimization facet)
Crime/        Chicago crime case (interesting-region recommendation facet + robustness study)
numeric_sample/  Region-guided acquisition samplers and baseline criteria shared by the MATLAB cases
testFunc/     Objective functions of the numerical cases
src/, grid/   LHS, evaluation grids, and misc. numerical utilities
sampling_Funciton/  Baseline acquisition-function implementations (AK-MCS-IRS, HHs, IMSE, EIER, ...)
MRSM_subprograms1(/_correxp)/  The mixed-output Gaussian process surrogate (MMRGP)
MS-PSO-pro/   The multi-objective PSO used for region identification (supplementary Sec. III)
ROImethod/    Python partitioned-GP region classifier used by the region samplers
gp_single/    Single-output GP utilities
third_party/GPML/  Gaussian Processes for Machine Learning library (Rasmussen & Nickisch)
paper_output/ Cross-case figure scripts
```

## Environment

- MATLAB R2022a with GPML (bundled under `third_party/GPML`). The case drivers add the required library folders (`src/`, `sampling_Funciton/`, `testFunc/`, `grid/`, `MRSM_subprograms1/`, `MS-PSO-pro/`) to the MATLAB path themselves; run them from inside their case folder.
- Python 3.9: numpy, scipy, pandas, scikit-learn, matplotlib, lightgbm (HPO case), plus the MATLAB Engine for Python (Crime case).

## Results archives

Every case folder contains a `*_final10.mat` (or `*_repro.mat`) archive with the runs of all compared methods on the evaluated ten seeds (one variable per arm, holding per-step trajectories), matching the numbers reported in the paper.

## Notes

- The Crime case is self-contained in Python; see `Crime/README.md`.
- This repository is intended to be released anonymously per the ICDE supplemental-material policy.
