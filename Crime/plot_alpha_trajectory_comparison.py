"""Compare recorded KARMA knowledge-credibility trajectories for two priors."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = (172, 136, 34)
VARIANTS = {
    "original": {
        "tag": "1km_budget50_random_holdout",
        "title": "Original boundary prior",
        "expected_knowledge_variant": None,
    },
    "adjacent": {
        "tag": "1km_budget50_random_holdout_adjacent_bnd",
        "title": "2021 adjacent-boundary prior",
        "expected_knowledge_variant": "adjacent",
    },
}


def load_trajectory(variant: str, seed: int) -> tuple[dict, pd.DataFrame]:
    spec = VARIANTS[variant]
    source = (
        ROOT / "results" / "formal" / spec["tag"] / "KARMA" / "fold_0"
        / f"seed_{seed}" / "history.json"
    )
    history = json.loads(source.read_text(encoding="utf-8"))
    assert history["status"] == "COMPLETE" and history["method"] == "KARMA"
    assert history["seed"] == seed and history["initial_budget"] == 30
    assert history["sequential_steps"] == 20 and len(history["acquisitions"]) == 20
    actual_variant = history.get("knowledge_variant")
    assert actual_variant == spec["expected_knowledge_variant"], (
        source, actual_variant, spec["expected_knowledge_variant"]
    )

    diagnostics = history["acquisitions"][-1]["diagnostics"]
    alpha = np.asarray(diagnostics["pi_comp_alpha_hist"], dtype=float)
    consistency = np.asarray(diagnostics["pi_comp_cons_hist"], dtype=float)
    gate = np.asarray(diagnostics["pi3_hist"], dtype=float)
    assert alpha.shape == consistency.shape == (20, 2) and gate.shape == (20, 4)
    assert np.isfinite(alpha).all() and np.isfinite(consistency).all()
    assert np.allclose(gate[:, :2], alpha)
    assert np.allclose(gate[:, 2], alpha.max(axis=1))
    assert np.array_equal(gate[:, 3].astype(int), (gate[:, 2] >= 0.25).astype(int))

    expected = np.maximum(0.1, 0.5 * alpha[:-1] + 0.5 * consistency[1:])
    assert np.allclose(alpha[1:], expected, atol=1e-10)

    steps = np.arange(1, 21)
    frame = pd.DataFrame({
        "variant": variant,
        "seed": seed,
        "step": steps,
        "cons_hg": consistency[:, 0],
        "cons_bnd": consistency[:, 1],
        "alpha_hg": alpha[:, 0],
        "alpha_bnd": alpha[:, 1],
        "alpha_eff": gate[:, 2],
        "knowledge_gate_active": gate[:, 3].astype(int),
        "source_history": str(source),
    })
    return history, frame


def main() -> None:
    histories: dict[tuple[str, int], dict] = {}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            history, frame = load_trajectory(variant, seed)
            histories[variant, seed] = history
            frames[variant, seed] = frame
        assert (
            histories["original", seed]["training_indices"][:30]
            == histories["adjacent", seed]["training_indices"][:30]
        )

    output = ROOT / "results" / "figures" / "prior_arbitration"
    output.mkdir(parents=True, exist_ok=True)
    all_steps = pd.concat(frames.values(), ignore_index=True)
    data_path = output / "alpha_trajectory_original_vs_adjacent_3seeds.csv"
    all_steps.to_csv(data_path, index=False)

    summary_rows = []
    for (variant, seed), frame in frames.items():
        closed = frame.loc[frame["knowledge_gate_active"] == 0, "step"].tolist()
        summary_rows.append({
            "variant": variant,
            "seed": seed,
            "gate_off_count": len(closed),
            "gate_off_steps": json.dumps(closed),
            "alpha_hg_mean": frame["alpha_hg"].mean(),
            "alpha_bnd_mean": frame["alpha_bnd"].mean(),
            "alpha_hg_final": frame["alpha_hg"].iloc[-1],
            "alpha_bnd_final": frame["alpha_bnd"].iloc[-1],
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = output / "alpha_trajectory_original_vs_adjacent_summary.csv"
    summary.to_csv(summary_path, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(
        3, 2, figsize=(12.0, 10.0), sharex=True, sharey=True,
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, top=0.925, bottom=0.15,
        hspace=0.32, wspace=0.08,
    )
    steps = np.arange(1, 21)
    for row, seed in enumerate(SEEDS):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            frame = frames[variant, seed]
            closed = frame["knowledge_gate_active"].to_numpy() == 0
            for step in steps[closed]:
                ax.axvspan(step - 0.5, step + 0.5, color="#E6E6E6", zorder=0)
            ax.plot(
                steps, frame["alpha_hg"], color="#0072B2", marker="o",
                markersize=4, linewidth=1.7,
                label=r"$\alpha_{hg}$: continuous difficulty",
            )
            ax.plot(
                steps, frame["alpha_bnd"], color="#D55E00", marker="s",
                markersize=3.8, linewidth=1.7, linestyle="--",
                label=r"$\alpha_{bnd}$: classification boundary",
            )
            ax.axhline(
                0.1, color="#666666", linewidth=1.0, linestyle=":",
                label="trust lower bound = 0.10",
            )
            ax.axhline(
                0.25, color="#009E73", linewidth=1.1, linestyle="-.",
                label="Gate threshold = 0.25",
            )
            ax.scatter(
                steps[closed], np.full(closed.sum(), 0.055), color="#CC0000",
                marker="x", s=32, linewidths=1.3, zorder=4,
                label="Gate OFF (also grey shading)",
            )
            ax.set_ylim(0.04, 0.72)
            ax.set_xlim(0.5, 20.5)
            ax.set_xticks(steps)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.6, zorder=0)
            ax.set_title(
                f"{VARIANTS[variant]['title']} — seed {seed} "
                f"(Gate OFF: {closed.sum()}/20)"
            )
            if col == 0:
                ax.set_ylabel("Knowledge credibility " + r"$\alpha$")
            if row == 2:
                ax.set_xlabel("Sequential acquisition step (after 30 initial LHS samples)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "KARMA prior arbitration: original vs. 2021 adjacent-boundary knowledge",
        fontsize=14,
    )
    figure_path = output / "alpha_trajectory_original_vs_adjacent_3seeds.png"
    fig.savefig(figure_path, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"FIGURE={figure_path}")
    print(f"STEP_DATA={data_path}")
    print(f"SUMMARY={summary_path}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
