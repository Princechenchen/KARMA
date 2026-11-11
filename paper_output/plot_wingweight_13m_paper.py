"""wingweight 13 方法论文版曲线重绘 (2026-11-08, 用户指定图例口径):
与 Branin/Ackley paper 版完全一致 = 1800x1350 画布(6x4.5in, 无 tight),
图例两列 + 白色半透明底框: RMAE 右上 / ACC 右下; KRAM-EVR 深蓝压轴。
数据 = Wingweight_box3u_final10.mat (10 主臂 + qbc_*_sub10 + pibo_*_sub10)。
输出新文件 fig_wingweight_13m_paper_*.png, 不覆盖既有图。"""
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = r"F:\学习工作\搞点研究\投论文\融合物理信息的区域采样"
LEGEND_SIZE, TICK_SIZE, LABEL_SIZE = 12, 13, 15
cn_font = FontProperties(family='SimHei', size=LABEL_SIZE, weight='bold')
C8_COLOR = [0.0000, 0.4470, 0.7410]

SPEC = [
    ('LHS',         [0.8500, 0.3250, 0.0980], 'LHS'),
    ('MSE',         [0.9290, 0.6940, 0.1250], 'MSE'),
    ('IMSE',        [0.4940, 0.1840, 0.5560], 'IMSE'),
    ('PALC',        [0.6350, 0.0780, 0.1840], 'PALC'),
    ('AK-MCS-IRS',  [0.3010, 0.7450, 0.9330], 'AK-MCS-IRS'),
    ('piBO-global', [0.0000, 0.5500, 0.5000], 'C6-piBO-global'),
    ('piBO-ROI',    [0.6350, 0.4700, 0.2500], 'C7-piBO-ROI'),
    ('KeyRegion',   [0.4660, 0.6740, 0.1880], 'KeyRegion'),
    ('Voronoi',     [0.5000, 0.5000, 0.5000], 'Voronoi'),
    ('HHs',         [0.2000, 0.2000, 0.2000], 'HHs'),
    ('QBC-global',  [0.7500, 0.0000, 0.7500], 'QBC'),
]
C8_KEY = 'KRAM-EVR'
N_INIT, TOTAL_ITER = 25, 45

def load_all():
    M = sio.loadmat(os.path.join(HERE, 'wingweight', 'Wingweight_box3u_final10.mat'),
                    squeeze_me=True, struct_as_record=False)
    seeds = [int(x) for x in np.ravel(M['SUBSET_SEEDS'])]
    assert seeds == [29, 47, 66, 84, 102, 74, 110, 188, 205, 226], seeds
    names = [str(x) for x in np.ravel(M['arm_names'])]
    rmae = np.asarray(M['rmae_resp'], float)          
    acc = np.asarray(M['acc_arr'], float)             
    q_r = np.asarray(M['qbc_rmae_resp_sub10'], float)  
    q_a = np.asarray(M['qbc_acc_arr_sub10'], float)
    p_r = np.asarray(M['pibo_rmae_resp_sub10'], float)  
    p_a = np.asarray(M['pibo_acc_arr_sub10'], float)

    R, A = [], []
    for _, _, key in SPEC:
        if key == 'QBC':
            R.append(q_r); A.append(q_a)
        elif key.startswith('C6'):
            R.append(p_r[:, 0, :, :]); A.append(p_a[:, 0, :])
        elif key.startswith('C7'):
            R.append(p_r[:, 1, :, :]); A.append(p_a[:, 1, :])
        else:
            i = names.index(key)
            R.append(rmae[:, i, :, :]); A.append(acc[:, i, :])
    i8 = names.index('KRAM-EVR')
    R.append(rmae[:, i8, :, :]); A.append(acc[:, i8, :])
    R = np.stack(R, axis=1); A = np.stack(A, axis=1)
    assert R.shape == (20, 12, 10, 2) and A.shape == (20, 12, 10)

    a8 = A[:, 11, :]; r8 = R[:, 11, :, :]
    chk = {'y1': r8[-1, :, 0].mean(), 'y2': r8[-1, :, 1].mean(), 'ACC': a8[-1].mean(),
           'ACCa': a8.mean(), 'y1a': r8[:, :, 0].mean(), 'y2a': r8[:, :, 1].mean()}
    assert abs(chk['y1'] - 0.0077) < 5e-4 and abs(chk['y2'] - 0.0107) < 5e-4, chk
    assert abs(chk['ACC'] - 0.8727) < 5e-4 and abs(chk['ACCa'] - 0.8327) < 5e-4, chk
    assert abs(A[-1, 10, :].mean() - 0.8061) < 5e-4, 'QBC ACC anchor fail'
    assert abs(A[-1, 5, :].mean() - 0.8416) < 5e-4, 'C6 ACC anchor fail'
    print('锚值断言通过(13m paper):', {k: round(v, 4) for k, v in chk.items()})
    return R, A

def legend_style(ax, loc):
    return ax.legend(loc=loc, ncol=2, frameon=True, framealpha=0.85,
                     edgecolor='none', labelspacing=0.22, columnspacing=0.9,
                     handlelength=1.4,
                     prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})

R, A = load_all()
iters = np.arange(N_INIT + 1, TOTAL_ITER + 1)
outdir = os.path.join(HERE, 'wingweight')
draw = list(range(11)) + [11]   

for c in range(2):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in draw:
        disp, color, key = SPEC[k] if k < 11 else ('KRAM-EVR', C8_COLOR, C8_KEY)
        y = R[:, k, :, c]
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
    out = os.path.join(outdir, f'fig_wingweight_13m_paper_y{c + 1}_RMAE.png')
    plt.savefig(out, dpi=300); plt.close(); print('Saved', out)

fig, ax = plt.subplots(figsize=(6, 4.5))
for k in draw:
    disp, color, key = SPEC[k] if k < 11 else ('KRAM-EVR', C8_COLOR, C8_KEY)
    a = A[:, k, :]
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
out = os.path.join(outdir, 'fig_wingweight_13m_paper_ACC.png')
plt.savefig(out, dpi=300); plt.close(); print('Saved', out)

from PIL import Image
for m in ('y1_RMAE', 'y2_RMAE', 'ACC'):
    print(m, Image.open(os.path.join(outdir, f'fig_wingweight_13m_paper_{m}.png')).size)
print('ALL DONE')
