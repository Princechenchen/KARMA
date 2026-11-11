"""Plot one formal KARMA seed's recorded knowledge-credibility trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=172)
    args = parser.parse_args()
    source = (
        ROOT / "results" / "formal" / "1km_budget50_random_holdout"
        / "KARMA" / "fold_0" / f"seed_{args.seed}" / "history.json"
    )
    history = json.loads(source.read_text(encoding="utf-8"))
    assert history["status"] == "COMPLETE" and history["method"] == "KARMA"
    diagnostics = history["acquisitions"][-1]["diagnostics"]
    alpha = np.asarray(diagnostics["pi_comp_alpha_hist"], dtype=float)
    consistency = np.asarray(diagnostics["pi_comp_cons_hist"], dtype=float)
    gate = np.asarray(diagnostics["pi3_hist"], dtype=float)
    assert alpha.shape == consistency.shape == (20, 2) and gate.shape == (20, 4)
    assert np.allclose(gate[:, :2], alpha) and np.allclose(gate[:, 2], alpha.max(axis=1))

    steps = np.arange(1, 21)
    frame = pd.DataFrame({
        "step": steps,
        "cons_hg": consistency[:, 0],
        "cons_bnd": consistency[:, 1],
        "alpha_hg": alpha[:, 0],
        "alpha_bnd": alpha[:, 1],
        "alpha_eff": gate[:, 2],
        "knowledge_gate_active": gate[:, 3].astype(int),
    })
    output = ROOT / "results" / "figures" / "prior_arbitration"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"alpha_trajectory_seed_{args.seed}.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, (ax, gate_ax) = plt.subplots(
        2, 1, figsize=(8.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        layout="constrained",
    )
    ax.plot(steps, alpha[:, 0], color="#0072B2", marker="o", markersize=4.5,
            linewidth=1.8, label=r"$\alpha_{hg}$: continuous difficulty")
    ax.plot(steps, alpha[:, 1], color="#D55E00", marker="s", markersize=4.2,
            linewidth=1.8, linestyle="--", label=r"$\alpha_{bnd}$: classification boundary")
    ax.axhline(0.1, color="#777777", linewidth=1.0, linestyle=":",
               label=r"trust lower bound = 0.10")
    ax.axhline(0.25, color="#009E73", linewidth=1.2, linestyle="-.",
               label=r"gate threshold on $\max(\alpha_{hg},\alpha_{bnd})$ = 0.25")
    ax.set_ylim(0.04, 0.72)
    ax.set_ylabel("Knowledge credibility " + r"$\alpha$")
    ax.set_title(f"KARMA knowledge-credibility trajectory — Crimes seed {args.seed}")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    closed = gate[:, 3] == 0
    gate_ax.fill_between(steps, 0, 1, where=closed, step="mid",
                         color="#DDDDDD", alpha=0.8, label="Knowledge objective OFF")
    gate_ax.step(steps, gate[:, 3], where="mid", color="#009E73", linewidth=1.8)
    gate_ax.scatter(steps[closed], np.zeros(closed.sum()), color="#CC0000",
                    marker="x", s=35, linewidths=1.4, zorder=3)
    gate_ax.set_yticks([0, 1], labels=["OFF", "ON"])
    gate_ax.set_ylim(-0.25, 1.25)
    gate_ax.set_ylabel("Gate")
    gate_ax.set_xlabel("Sequential acquisition step (30 initial LHS samples; 20 added samples)")
    gate_ax.set_xticks(steps)
    gate_ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)

    path = output / f"alpha_trajectory_seed_{args.seed}.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"FIGURE={path}")
    print(f"DATA={output / f'alpha_trajectory_seed_{args.seed}.csv'}")
    print(f"CLOSED_STEPS={steps[closed].tolist()}")


if __name__ == "__main__":
    main()
