"""论文版 Branin/Ackley 六张曲线图重绘(2026-11-07, 用户指定图例口径)
统一图例样式 = 两列 + 白色半透明底框(framealpha=0.85, 无边框):
  RMAE(y1/y2) 曲线下降型 -> 图例置于图片右上 (loc='upper right');
  ACC     曲线爬升型 -> 图例置于图片右下 (loc='lower right')。
统一画布 1800x1350 (6x4.5in, 无 tight 裁剪); 数据/锚值断言与 plot_final10_all.py 一致;
输出新文件名 fig_*_paper_*.png, 不覆盖任何既有图(红线1)。"""
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(os.path.abspath(__file__))
LEGEND_SIZE, TICK_SIZE, LABEL_SIZE = 12, 13, 15
cn_font = FontProperties(family='SimHei', size=LABEL_SIZE, weight='bold')
C8_COLOR = [0.0000, 0.4470, 0.7410]

SPEC_BRANIN = [
    ('LHS',         [0.8500, 0.3250, 0.0980], 'C1-MaximinLHS'),
    ('MSE',         [0.9290, 0.6940, 0.1250], 'C2-MeanMSE'),
    ('IMSE',        [0.4940, 0.1840, 0.5560], 'C3-IMSE'),
    ('PALC',        [0.6350, 0.0780, 0.1840], 'C4-PAL-ROI-SUR'),
    ('AK-MCS-IRS',  [0.3010, 0.7450, 0.9330], 'C5-AKMCS-IRS'),
    ('piBO-global', [0.0000, 0.5500, 0.5000], 'C6-piBO-global'),
    ('piBO-ROI',    [0.6350, 0.4700, 0.2500], 'C7-piBO-ROI'),
    ('KeyRegion',   [0.4660, 0.6740, 0.1880], 'C9-KeyRegion'),
    ('Voronoi',     [0.5000, 0.5000, 0.5000], 'C10-Voronoi'),
    ('HHs',         [0.2000, 0.2000, 0.2000], 'C11-HHs'),
    ('QBC-global',  [0.7500, 0.0000, 0.7500], 'C12-QBC-global'),
]
SPEC_ACKLEY = [
    ('LHS',         [0.8500, 0.3250, 0.0980]), ('MSE',        [0.9290, 0.6940, 0.1250]),
    ('IMSE',        [0.4940, 0.1840, 0.5560]), ('PALC',       [0.6350, 0.0780, 0.1840]),
    ('AK-MCS-IRS',  [0.3010, 0.7450, 0.9330]), ('piBO-global', [0.0000, 0.5500, 0.5000]),
    ('piBO-ROI',    [0.6350, 0.4700, 0.2500]), ('KeyRegion',  [0.4660, 0.6740, 0.1880]),
    ('Voronoi',     [0.5000, 0.5000, 0.5000]), ('HHs',        [0.2000, 0.2000, 0.2000]),
]


def unwrap_arm_names(F):
    an = np.atleast_1d(F['arm_names'])
    while an.dtype == object and an.size >= 1 and isinstance(an.flat[0], np.ndarray):
        an = np.atleast_1d(an.flat[0])
    return [str(np.squeeze(x)) for x in an.ravel()]


def legend_style(ax, loc):
    """统一图例: 两列 + 白色半透明底框(无边框)"""
    return ax.legend(loc=loc, ncol=2, frameon=True, framealpha=0.85,
                     edgecolor='none', labelspacing=0.22, columnspacing=0.9,
                     handlelength=1.4,
                     prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})


def plot_panel(case, outdir, names, spec, acc, rmae, acc8, rmae8, n_init, total_iter, anchors):
    iters = np.arange(n_init + 1, total_iter + 1)
    rows = []
    for s in spec:
        key = s[2] if len(s) > 2 else s[0]
        i = names.index(key)
        rows.append((s[0], s[1], acc[:, i, :], rmae[:, i, :, :]))
    rows.append(('KRAM-EVR', C8_COLOR, acc8, rmae8))
    draw = list(range(len(rows) - 1)) + [len(rows) - 1]   

    for c in range(2):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for k in draw:
            disp, color, a, r = rows[k]
            y = r[:, :, c]
            m = np.nanmean(y, axis=1); sd = np.nanstd(y, axis=1, ddof=1)
            ax.plot(iters, m, '-', linewidth=2, color=color, label=disp)
            ax.fill_between(iters, np.clip(m - sd, 0, None), m + sd,
                            facecolor=color, alpha=0.1, edgecolor='none')
        ax.set_xlim([n_init + 1, total_iter])
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
        out = os.path.join(outdir, f'fig_{case}_paper_y{c + 1}_RMAE.png')
        plt.savefig(out, dpi=300); plt.close(); print('Saved', out)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in draw:
        disp, color, a, r = rows[k]
        m = np.nanmean(a, axis=1); sd = np.nanstd(a, axis=1, ddof=1)
        ax.plot(iters, m, '-', linewidth=2, color=color, label=disp)
        ax.fill_between(iters, np.clip(m - sd, 0, None), np.clip(m + sd, None, 1.0),
                        facecolor=color, alpha=0.1, edgecolor='none')
    ax.set_xlim([n_init + 1, total_iter])
    ax.set_xlabel('Number of samples', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('ACC', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
    legend_style(ax, 'lower right')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontname('Times New Roman')
    ax.grid(True, which='both', linestyle=':', alpha=0.7); ax.set_axisbelow(True)
    ax.set_box_aspect(0.75)
    plt.tight_layout()
    out = os.path.join(outdir, f'fig_{case}_paper_ACC.png')
    plt.savefig(out, dpi=300); plt.close(); print('Saved', out)

    got = {'ACC': acc8[-1].mean(), 'y1': rmae8[-1, :, 0].mean(), 'y2': rmae8[-1, :, 1].mean(),
           'ACCa': acc8.mean(), 'y1a': rmae8[:, :, 0].mean(), 'y2a': rmae8[:, :, 1].mean()}
    for k, v in anchors.items():
        assert abs(got[k] - v) < 5e-4, (case, k, got[k], v)
    print(f'[{case}] 锚值断言通过 {anchors}')


if __name__ == '__main__':
    F = sio.loadmat(os.path.join(HERE, 'numeric_sample', 'Branin_box3d_final10.mat'),
                    squeeze_me=True, struct_as_record=False)
    nm = unwrap_arm_names(F)
    acc8 = np.asarray(F['acc_box3d'], float)
    rmae8 = np.asarray(F['rmae_box3d'], float)
    plot_panel('branin', os.path.join(HERE, 'numeric_sample'), nm, SPEC_BRANIN,
               np.asarray(F['acc_arr'], float), np.asarray(F['rmae_resp'], float),
               acc8, rmae8, 30, 50,
               {'ACC': 0.9333, 'y1': 0.2233, 'y2': 0.4465, 'ACCa': 0.9294, 'y1a': 0.2961, 'y2a': 0.6399})
    F = sio.loadmat(os.path.join(HERE, 'numeric_sample', 'Ackley_box3_final10.mat'),
                    squeeze_me=True, struct_as_record=False)
    nm = unwrap_arm_names(F)
    i8 = nm.index('KRAM-EVR')
    acc8 = np.asarray(F['acc_arr'], float)[:, i8, :]
    rmae8 = np.asarray(F['rmae_resp'], float)[:, i8, :, :]
    plot_panel('ackley', os.path.join(HERE, 'numeric_sample'), nm, SPEC_ACKLEY,
               np.asarray(F['acc_arr'], float), np.asarray(F['rmae_resp'], float),
               acc8, rmae8, 20, 50,
               {'ACC': 0.8460, 'y1': 0.1483, 'y2': 0.1501, 'ACCa': 0.8060, 'y1a': 0.2181, 'y2a': 0.2208})
    from PIL import Image
    for c in ('branin', 'ackley'):
        for m in ('y1_RMAE', 'y2_RMAE', 'ACC'):
            p = os.path.join(HERE, 'numeric_sample', f'fig_{c}_paper_{m}.png')
            print(c, m, Image.open(p).size)
    print('ALL DONE')
