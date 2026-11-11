# Chicago crime case (interesting-region recommendation facet + robustness study)

Public Chicago crime data on a 1 km grid: continuous incident count and a hotspot label; knowledge = 2021 historical intensity structure (smoothed gradient field + hotspot-boundary band). Includes the four-arm robustness study (aligned vs. 180-degree rotated knowledge).

## Pipeline

1. **Data construction**: `build_crimes_dataset.py` builds the 1 km unit grid, the two responses, and the train/test splits from the processed incident CSVs; the `build_*_knowledge.py` family constructs the knowledge fields (the finalized `pi_bnd` is the 2021 high-side hotspot-adjacency band).
2. **Region-recommendation comparison**: `benchmark_region_optimizers.py` runs KARMA and the baselines; per-run metrics are finalized by `finalize_crimes_10seed_record.py` into `results/final_10seed_karma2022/`.
3. **Robustness study**: `diagnose_hotspot_highside_rotated180.py` runs the four arms (aligned / rotated / no-arbitration / no-knowledge); records are finalized by `finalize_rotated180_ablation_10seed.py` into `results/final_10seed_rotated180_ablation/`.
4. **Metrics and figures**: `compute_region_set_metrics.py`, `compute_hotspot_metrics.py`, `plot_fig5_paper.py` (Fig. 5), `plot_rotated180_ablation_spatial.py`, `plot_sec5_bar_only.py` / `plot_sec5_bar_and_regions.py` (Fig. 7); outputs under `results/figures/`.

## Protocol (frozen)

- Initial design 30 cells + 20 sequential acquisitions (evaluation budget 50); per seed, 132 cells are held out at random (never queried) for surrogate evaluation; the remaining 527 cells form the candidate pool.
- Ten seeds: 34, 46, 77, 136, 160, 172, 180, 2159, 6790, 7416. The same seed shares the test split and the initial design across all methods.
- Recommender: K = 4 boxes, 512 local Sobol points per box, Glowworm-Swarm-style multimodal search, box side lengths 1-15% of the domain with size regularization; region quality is evaluated against all cleaned 2022 incidents with 2,000 size-matched random reference boxes per region.

## Requirements

Python 3.9 (numpy, pandas, scipy, scikit-learn, matplotlib) plus **MATLAB R2022a with the MATLAB Engine for Python and GPML** on the path: the surrogate fitting, predictions, and the feedback-driven acquisition criteria run through the MATLAB engine; Python handles the grid replay, splits, budgets, and the multimodal recommender. Run the MATLAB-side samplers with the shared library folders at the repository root (numeric_sample/, sampling_Funciton/, src/) on the MATLAB path.

`data/` ships the processed incident CSVs (gzipped), the unit grids, all knowledge fields, and the 2022 ground truth; `data/raw/` (the raw city download) is not bundled and can be rebuilt through `build_crimes_dataset.py` from the official Chicago data portal.
