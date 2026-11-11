"""Section V 条形图独立版 v3 (2026-11-09, 用户要求两面板合一):
单面板分组柱: 每变体内 CRP/DL 两柱(指标色), 误差棒(总体SD) + 逐种子空心散点
(固定顺序偏移, 按指标色) + 顶部均值数值; 单一线性轴 0-1.42 两指标同图可读。
逐种子数据取自结果档案, 断言均值与论文口径一致。
输出: results/figures/rotated180_ablation_spatial/sec5_bar.png/.pdf"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from finalize_rotated180_ablation_10seed import collect_runs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "figures" / "rotated180_ablation_spatial"

VARIANTS = ["KARMA-A", "KARMA-R", "KARMA-wo arbitration", "KARMA-wo knowledge"]
KEYS = ["KARMA-Aligned", "KARMA-Misleading-Adaptive",
        "KARMA-Misleading-Fixed", "ROI-D"]
METRICS = [
    ("conditional_region_pass", "CRP", "#4C72B0",
     [0.650, 0.625, 0.450, 0.525]),
    ("density_lift", "DL", "#C44E52", [1.123, 1.096, 1.012, 1.076]),
]

runs, _ = collect_runs()
stats = {}
for column, _, _, expected in METRICS:
    means, stds, values = [], [], []
    for key in KEYS:
        block = runs[runs["method"] == key].sort_values("seed")
        assert len(block) == 10, (key, len(block))
        vals = block[column].to_numpy(float)
        values.append(vals)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)))
    for got, want in zip(means, expected):
        assert abs(got - want) < 5e-4, (column, got, want)
    stats[column] = (np.array(means), np.array(stds), values)

fig, ax = plt.subplots(figsize=(4.6, 5.2), layout="constrained")
with plt.rc_context({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}):
    x = np.arange(4, dtype=float)
    upper = 1.42
    for offset, (column, name, color, _) in zip((-0.19, 0.19), METRICS):
        means, stds, values = stats[column]
        ax.bar(x + offset, means, width=0.36, color=color, alpha=0.72,
               edgecolor="#333333", linewidth=0.6, zorder=2, label=name)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor="#222222", elinewidth=1.0, capsize=4,
                    capthick=1.0, zorder=4)
        dot_offsets = np.linspace(-0.125, 0.125, 10)
        for index, vals in enumerate(values):
            ax.scatter(np.full(len(vals), x[index] + offset) + dot_offsets,
                       vals, s=17, facecolor="white", edgecolor=color,
                       linewidth=0.8, zorder=5)
            label_y = max(means[index] + stds[index], float(vals.max())) \
                + 0.035 * upper
            ax.text(x[index] + offset, label_y, f"{means[index]:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x, VARIANTS, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_ylim(0, upper)
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=12, handlelength=1.4,
              columnspacing=1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"sec5_bar.{suffix}", dpi=300, facecolor="white")
plt.close(fig)
print("saved", OUT / "sec5_bar.png")
