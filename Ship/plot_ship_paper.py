"""ship 论文版曲线重绘 (与 Branin/Ackley paper 版完全同规格):
1800x1350 原生画布(6x4.5in, 无 tight 裁剪), 图例两列白色半透明底框:
RMAE 右上 / ACC 右下; KRAM-EVR 深蓝压轴; 锚值断言内建。
数据 = Ship_v1box3u_final10.mat 主数组。
输出新文件 fig_ship_paper_*.png, 不覆盖任何既有图。"""
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(os.path.abspath(__file__))
LEGEND_SIZE, TICK_SIZE, LABEL_SIZE = 12, 13, 15
cn_font = FontProperties(family='SimHei', size=LABEL_SIZE, weight='bold')
C8_COLOR = [0.0000, 0.4470, 0.7410]

SPEC = [  
    ('LHS',         [0.8500, 0.3250, 0.0980], 'LHS'),
    ('MSE',         [0.9290, 0.6940, 0.1250], 'MSE'),
    ('IMSE',        [0.4940, 0.1840, 0.5560], 'IMSE'),
    ('PALC',        [0.6350, 0.0780, 0.1840], 'PALC'),
    ('AK-MCS-IRS',  [0.3010, 0.7450, 0.9330], 'AK-MCS-IRS'),
    ('KeyRegion',   [0.4660, 0.6740, 0.1880], 'KeyRegion'),
    ('Voronoi',     [0.5000, 0.5000, 0.5000], 'Voronoi'),
    ('HHs',         [0.2000, 0.2000, 0.2000], 'HHs'),
    ('piBO-global', [0.0000, 0.5500, 0.5000], 'C6-piBO-global'),
    ('piBO-ROI',    [0.6350, 0.4700, 0.2500], 'C7-piBO-ROI'),
    ('QBC-global',  [0.8000, 0.2000, 0.6000], 'C12-QBC-global'),
]
N_INIT, TOTAL_ITER = 20, 70

F = os.path.join(HERE, 'Ship_v1box3u_final10.mat')
d = sio.loadmat(F, squeeze_me=True, struct_as_record=False)
names = [str(x) for x in np.ravel(d['arm_names'])]
seeds = [int(x) for x in np.ravel(d['SUBSET_RSL'])]
assert seeds == [5, 17, 20, 43, 55, 62, 63, 74, 78, 85], seeds
acc = np.asarray(d['acc_arr'], float)        
rmae = np.asarray(d['rmae_resp'], float)     
assert acc.shape == (50, 12, 10) and np.isfinite(acc).all()

iters = np.arange(N_INIT + 1, TOTAL_ITER + 1)
rows = []
for disp, color, key in SPEC:
    i = names.index(key)
    a = np.take(acc, i, axis=1)
    r = np.take(rmae, i, axis=1)
    rows.append((disp, color, a, r))
i8 = names.index('KRAM-EVR')
a8 = np.take(acc, i8, axis=1)
r8 = np.take(rmae, i8, axis=1)
rows.append(('KRAM-EVR', C8_COLOR, a8, r8))
draw = list(range(len(rows) - 1)) + [len(rows) - 1]   

def vals(a, r, k):
    return [r[-1, :, 0].mean(), r[-1, :, 1].mean(), r[:, :, 0].mean(),
            r[:, :, 1].mean(), a[-1, :].mean(), a.mean()]
ANCH = {len(rows) - 1: [0.1336, 0.0540, 0.1572, 0.063079, 0.8790, 0.8445]}
for disp, color, key in SPEC:
    if key == 'C6-piBO-global':
        ANCH[rows.index(next(x for x in rows if x[0] == disp))] = [0.1350, 0.0545, 0.1609, 0.063571, 0.8720, 0.8487]
    elif key == 'C7-piBO-ROI':
        ANCH[rows.index(next(x for x in rows if x[0] == disp))] = [0.1347, 0.0546, 0.1623, 0.063986, 0.8530, 0.8470]
    elif key == 'QBC-global':
        ANCH[rows.index(next(x for x in rows if x[0] == disp))] = [0.1471, 0.0542, 0.1823, 0.068657, 0.8540, 0.8320]
for k, ref in ANCH.items():
    disp, color, a, r = rows[k]
    v = vals(a, r, 0)
    assert all(abs(x - y) < 5e-4 for x, y in zip(v, ref)), (disp, v)
print('锚值断言 PASS: KRAM/C6/C7/QBC 各六值')

def legend_style(ax, loc):
    return ax.legend(loc=loc, ncol=2, frameon=True, framealpha=0.85,
                     edgecolor='none', labelspacing=0.22, columnspacing=0.9,
                     handlelength=1.4,
                     prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})

for c in range(2):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in draw:
        disp, color, a, r = rows[k]
        y = r[:, :, c]
        m = np.nanmean(y, axis=1); sd = np.nanstd(y, axis=1, ddof=1)
        ax.plot(iters, m, '-', linewidth=2, color=color, label=disp)
        ax.fill_between(iters, np.clip(m - sd, 0, None), m + sd,
                        facecolor=color, alpha=0.1, edgecolor='none')
    ax.set_xlim([N_INIT + 1, TOTAL_ITER])
    ax.set_xlabel('Number of samples', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel(f'$y_{{{c + 1}}}$ RMAE', fontname='Times New Roman',
                  fontsize=LABEL_SIZE, fontweight='bold')
    legend_style(ax, 'upper right')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontname('Times New Roman')
    ax.grid(True, which='both', linestyle=':', alpha=0.7); ax.set_axisbelow(True)
    ax.set_box_aspect(0.75)
    plt.tight_layout()
    out = os.path.join(HERE, f'fig_ship_paper_y{c + 1}_RMAE.png')
    plt.savefig(out, dpi=300); plt.close(); print('Saved', out)

fig, ax = plt.subplots(figsize=(6, 4.5))
for k in draw:
    disp, color, a, r = rows[k]
    m = np.nanmean(a, axis=1); sd = np.nanstd(a, axis=1, ddof=1)
    ax.plot(iters, m, '-', linewidth=2, color=color, label=disp)
    ax.fill_between(iters, np.clip(m - sd, 0, None), np.clip(m + sd, None, 1.0),
                    facecolor=color, alpha=0.1, edgecolor='none')
ax.set_xlim([N_INIT + 1, TOTAL_ITER])
ax.set_xlabel('Number of samples', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
ax.set_ylabel('ACC', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
legend_style(ax, 'lower right')
ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
for lb in ax.get_xticklabels() + ax.get_yticklabels():
    lb.set_fontname('Times New Roman')
ax.grid(True, which='both', linestyle=':', alpha=0.7); ax.set_axisbelow(True)
ax.set_box_aspect(0.75)
plt.tight_layout()
out = os.path.join(HERE, 'fig_ship_paper_ACC.png')
plt.savefig(out, dpi=300); plt.close(); print('Saved', out)

from PIL import Image
for m in ('y1_RMAE', 'y2_RMAE', 'ACC'):
    print(m, Image.open(os.path.join(HERE, f'fig_ship_paper_{m}.png')).size)
print('ALL DONE')
