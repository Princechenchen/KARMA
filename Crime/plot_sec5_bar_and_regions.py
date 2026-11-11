"""Section V 论文版组合图 v3 (2026-11-09, 用户指令):
条形图按补充材料 region_metrics_10seeds 图风格重绘
(每变体独立色 + 误差棒 + 逐种子散点 + 顶部数值), 与四臂区域图上下拼接。
输出: results/figures/rotated180_ablation_spatial/sec5_bar_and_regions.png/.pdf"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from plot_rotated180_ablation_spatial import METHODS, draw_region, region_legend
from finalize_rotated180_ablation_10seed import collect_runs

ROOT = __import__("pathlib").Path(__file__).resolve().parent
OUT = ROOT / "results" / "figures" / "rotated180_ablation_spatial"
DPI = 220

SHORT = {"KARMA-Aligned": "KARMA-A", "KARMA-Misleading-Adaptive": "KARMA-R",
         "KARMA-Misleading-Fixed": "KARMA-wo arbitration", "ROI-D": "KARMA-wo knowledge"}
BAR_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#9A9A9A"]

units = pd.read_csv(ROOT / "data/units/units_2022_1km_hotspot.csv")
runs, _ = collect_runs()
selected = {}
for method, _, _ in METHODS:
    seed = 7416 if method != "ROI-D" else 34
    row = runs[(runs["method"] == method) & (runs["seed"] == seed)]
    assert len(row) == 1
    selected[method] = row.iloc[0]

order = [m for m, _, _ in METHODS]
labels = [SHORT[m] for m in order]
crp_mean = [0.650, 0.625, 0.450, 0.525]
dl_mean = [1.123, 1.096, 1.012, 1.076]
crp_std = [0.175, 0.168, 0.150, 0.236]
dl_std = [0.108, 0.091, 0.072, 0.076]

fig_bar, ax = plt.subplots(figsize=(3.0, 2.85), layout="constrained")
with plt.rc_context({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}):
    x = np.arange(4)
    w = 0.36
    b_crp = ax.bar(x - w / 2, crp_mean, w, yerr=crp_std, capsize=5,
                   color="#4C72B0", alpha=0.85, edgecolor="#333333",
                   linewidth=0.6, error_kw=dict(ecolor="#222222", lw=1.1,
                                                 capsize=5, capthick=1.0),
                   zorder=3, label="CRP")
    b_dl = ax.bar(x + w / 2, dl_mean, w, yerr=dl_std, capsize=5,
                  color="#C44E52", alpha=0.85, edgecolor="#333333",
                  linewidth=0.6, error_kw=dict(ecolor="#222222", lw=1.1,
                                                capsize=5, capthick=1.0),
                  zorder=3, label="DL")
    for bars_, errs_ in ((b_crp, crp_std), (b_dl, dl_std)):
        for rect, err in zip(bars_, errs_):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + err + 0.04,
                    f"{rect.get_height():.2f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
    ax.set_xticks(x, labels, rotation=26, ha="right")
    ax.set_ylabel("Value", fontsize=16)
    ax.set_ylim(0, 1.45)
    ax.set_title("Ten-seed means", fontsize=17, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
fig_bar.savefig(OUT / "_bar_panel.png", dpi=DPI, facecolor="white")
plt.close(fig_bar)

density_cap = float(units["y1"].quantile(0.99))
density_norm = Normalize(vmin=0, vmax=density_cap)
TITLES = {"KARMA-Aligned": "KARMA-A", "KARMA-Misleading-Adaptive": "KARMA-R",
          "KARMA-Misleading-Fixed": "KARMA-wo\narbitration",
          "ROI-D": "KARMA-wo\nknowledge"}

MAP_W_IN, MAP_H_IN = 4.3, 3.5
fig_map, axes = plt.subplots(2, 2, figsize=(MAP_W_IN, MAP_H_IN), layout="constrained")
fig_map.get_layout_engine().set(hspace=0.06, wspace=0.04)
image = None
for ax, (method, _, _) in zip(axes.flat, METHODS):
    seed = 7416 if method != "ROI-D" else 34
    run = selected[method]
    title = TITLES[method]
    image = ax.scatter(
        units["x"], units["y"], c=units["y1"], marker="s", s=44,
        cmap="viridis", norm=density_norm, linewidths=0, edgecolors="none",
    )
    training = units.iloc[list(map(int, run["training_indices"]))]
    initial = training.iloc[:30]
    acquired = training.iloc[30:]
    ax.scatter(initial["x"], initial["y"], s=28, marker="o",
               facecolors="none", edgecolors="white", linewidths=0.9)
    ax.scatter(acquired["x"], acquired["y"], s=34, marker="D",
               c="#FF9F1C", edgecolors="black", linewidths=0.45)
    for index, (cx, cy, lx, ly) in enumerate(np.asarray(run["boxes"], dtype=float), 1):
        ax.add_patch(Rectangle((cx - lx / 2, cy - ly / 2), lx, ly,
                               fill=False, edgecolor="#D62728", linewidth=2.0))
        ax.text(cx, cy, f"R{index}", color="#B2182B", fontsize=13, fontweight="bold",
                ha="center", va="center",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.0})
    ax.set_title(title, fontsize=16, fontweight="bold", pad=4)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_xticks([])
    ax.set_yticks([])

handles = [
    Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
           markeredgecolor="white", markeredgewidth=1.0, markersize=7,
           label="Initial"),
    Line2D([], [], marker="D", linestyle="none", markerfacecolor="#FF9F1C",
           markeredgecolor="black", markersize=6.5, label="Acquired"),
    Rectangle((0, 0), 1, 1, fill=False, edgecolor="#D62728", linewidth=2.0,
              label="Recommended"),
]
fig_map.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, fontsize=13.5, handlelength=1.2, columnspacing=0.8)
cb = fig_map.colorbar(image, ax=axes, location="right", shrink=0.9, pad=0.015)
cb.set_label("Observed incidents/km$^2$", fontsize=15)
cb.ax.tick_params(labelsize=13)
fig_map.savefig(OUT / "_map_panel.png", dpi=DPI, facecolor="white")
plt.close(fig_map)

bar_img = Image.open(OUT / "_bar_panel.png").convert("RGB")
map_img = Image.open(OUT / "_map_panel.png").convert("RGB")
COMMON_H = max(bar_img.size[1], map_img.size[1])
bar_s = bar_img.resize((round(bar_img.size[0] * COMMON_H / bar_img.size[1]), COMMON_H),
                       Image.LANCZOS)
map_s = map_img.resize((round(map_img.size[0] * COMMON_H / map_img.size[1]), COMMON_H),
                       Image.LANCZOS)
GAP = 50
canvas = Image.new("RGB", (bar_s.size[0] + GAP + map_s.size[0], COMMON_H), "white")
canvas.paste(bar_s, (0, 0))
canvas.paste(map_s, (bar_s.size[0] + GAP, 0))
out = OUT / "sec5_bar_and_regions.png"
canvas.save(out, dpi=(DPI, DPI))
print("saved", out, canvas.size)
