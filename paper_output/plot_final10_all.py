"""
三算例10 种子曲线图(2026-10-08) —— 数据源 = 三个 final10 一步取数文件
Branin: numeric_sample/Branin_box3d_final10.mat  seeds [95,29,89,13,30,30,41,43,55,78]
Ackley: numeric_sample/Ackley_box3_final10.mat   seeds [136,34,46,77,91,158,17,43,62,78]
ship  : Radar/Ship_v1box3u_final10.mat           seeds [63,55,74,16,85,20,17,23,62,78] (85 勘误口径)
规则: 每算例 3 图(y1/y2 RMAE + ACC), 曲线 = 外部基线 + KRAM-EVR(C8 深蓝压轴);
      同族参照(box3/box3g)不入图(用户口径"不展示 box3g"); 均值±std 带; 锚值断言内置。
输出: 各自目录下 fig_<case>_final10_{y1,y2}_RMAE.png / fig_<case>_final10_ACC.png
"""
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = os.path.dirname(os.path.abspath(__file__))
LEGEND_SIZE, TICK_SIZE, LABEL_SIZE = 14, 14, 16
cn_font = FontProperties(family='SimHei', size=LABEL_SIZE, weight='bold')

def spec_branin():
    return [  
        ('LHS',         [0.8500,0.3250,0.0980], 'C1-MaximinLHS'),
        ('MSE',         [0.9290,0.6940,0.1250], 'C2-MeanMSE'),
        ('IMSE',        [0.4940,0.1840,0.5560], 'C3-IMSE'),
        ('PALC',        [0.6350,0.0780,0.1840], 'C4-PAL-ROI-SUR'),
        ('AK-MCS-IRS',  [0.3010,0.7450,0.9330], 'C5-AKMCS-IRS'),
        ('piBO-global', [0.0000,0.5500,0.5000], 'C6-piBO-global'),
        ('piBO-ROI',    [0.6350,0.4700,0.2500], 'C7-piBO-ROI'),
        ('KeyRegion',   [0.4660,0.6740,0.1880], 'C9-KeyRegion'),
        ('Voronoi',     [0.5000,0.5000,0.5000], 'C10-Voronoi'),
        ('HHs',         [0.2000,0.2000,0.2000], 'C11-HHs'),
        ('QBC-global',  [0.7500,0.0000,0.7500], 'C12-QBC-global'),
    ]
SPEC_ACKLEY = [
    ('LHS',         [0.8500,0.3250,0.0980]), ('MSE',        [0.9290,0.6940,0.1250]),
    ('IMSE',        [0.4940,0.1840,0.5560]), ('PALC',       [0.6350,0.0780,0.1840]),
    ('AK-MCS-IRS',  [0.3010,0.7450,0.9330]), ('piBO-global',[0.0000,0.5500,0.5000]),
    ('piBO-ROI',    [0.6350,0.4700,0.2500]), ('KeyRegion',  [0.4660,0.6740,0.1880]),
    ('Voronoi',     [0.5000,0.5000,0.5000]), ('HHs',        [0.2000,0.2000,0.2000]),
]
SPEC_SHIP = [
    ('LHS',         [0.8500,0.3250,0.0980]), ('MSE',        [0.9290,0.6940,0.1250]),
    ('IMSE',        [0.4940,0.1840,0.5560]), ('PALC',       [0.6350,0.0780,0.1840]),
    ('AK-MCS-IRS',  [0.3010,0.7450,0.9330]), ('KeyRegion',  [0.4660,0.6740,0.1880]),
    ('Voronoi',     [0.5000,0.5000,0.5000]), ('HHs',        [0.2000,0.2000,0.2000]),
]
C8_COLOR = [0.0000, 0.4470, 0.7410]

def load_case(path, c8field):
    F = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [str(np.squeeze(x)) for x in np.atleast_1d(F['arm_names']).ravel()]
    acc = np.asarray(F['acc_arr'], float); rmae = np.asarray(F['rmae_resp'], float)
    acc8 = np.asarray(F[c8field[0]], float); rmae8 = np.asarray(F[c8field[1]], float)
    return names, acc, rmae, acc8, rmae8

def plot_case(case, outdir, spec, names, acc, rmae, acc8, rmae8, n_init, total_iter, anchors, acc8_name_idx=None):
    n_iter = acc.shape[0]
    iters = np.arange(n_init + 1, total_iter + 1)
    rows = []
    for disp, color, key in [(s[0], s[1], s[2]) for s in spec]:
        i = names.index(key)
        rows.append((disp, color, acc[:, i, :], rmae[:, i, :, :]))
    rows.append(('KRAM-EVR', C8_COLOR, acc8, rmae8))
    draw = list(range(len(rows)-1)) + [len(rows)-1]   
    for c in range(2):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for k in draw:
            disp, color, a, r = rows[k]
            y = r[:, :, c]
            m = np.nanmean(y, axis=1); s = np.nanstd(y, axis=1, ddof=1)
            ax.plot(iters, m, '-', linewidth=2, color=color, label=disp)
            ax.fill_between(iters, np.clip(m-s, 0, None), m+s, facecolor=color, alpha=0.1, edgecolor='none')
        ax.set_xlim([n_init + 1, total_iter])
        ax.set_xlabel('采样数量', fontproperties=cn_font)
        ax.set_ylabel(f'$y_{{{c+1}}}$ RMAE', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
        ax.legend(loc='upper right', frameon=False, labelspacing=0.25,
                  prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        for lb in ax.get_xticklabels() + ax.get_yticklabels(): lb.set_fontname('Times New Roman')
        ax.grid(True, which='both', linestyle=':', alpha=0.7); ax.set_axisbelow(True); ax.set_box_aspect(0.75)
        plt.tight_layout()
        out = os.path.join(outdir, f'fig_{case}_final10_y{c+1}_RMAE.png')
        plt.savefig(out, dpi=300); plt.close(); print('Saved', out)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in draw:
        disp, color, a, r = rows[k]
        m = np.nanmean(a, axis=1); s = np.nanstd(a, axis=1, ddof=1)
        ax.plot(iters, m, '-', linewidth=2, color=color, label=disp)
        ax.fill_between(iters, np.clip(m-s, 0, None), np.clip(m+s, None, 1.0), facecolor=color, alpha=0.1, edgecolor='none')
    ax.set_xlim([n_init + 1, total_iter])
    ax.set_xlabel('采样数量', fontproperties=cn_font)
    ax.set_ylabel('ACC', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False,
              labelspacing=0.25, prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for lb in ax.get_xticklabels() + ax.get_yticklabels(): lb.set_fontname('Times New Roman')
    ax.grid(True, which='both', linestyle=':', alpha=0.7); ax.set_axisbelow(True); ax.set_box_aspect(0.75)
    plt.tight_layout()
    out = os.path.join(outdir, f'fig_{case}_final10_ACC.png')
    plt.savefig(out, dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight'); plt.close(); print('Saved', out)
    for k, v in anchors.items():
        got = {'ACC': acc8[-1].mean(), 'y1': rmae8[-1, :, 0].mean(), 'y2': rmae8[-1, :, 1].mean(),
               'ACCa': acc8.mean(), 'y1a': rmae8[:, :, 0].mean(), 'y2a': rmae8[:, :, 1].mean()}[k]
        assert abs(got - v) < 5e-4, (case, k, got, v)
    print(f'[{case}] 锚值断言通过 {anchors}')

if __name__ == '__main__':
    p = os.path.join(HERE, 'numeric_sample', 'Branin_box3d_final10.mat')
    names, acc, rmae, acc8, rmae8 = load_case(p, ('acc_box3d', 'rmae_box3d'))
    plot_case('branin', os.path.join(HERE, 'numeric_sample'), spec_branin(),
              names, acc, rmae, acc8, rmae8, 30, 50,
              {'ACC': 0.9333, 'y1': 0.2233, 'y2': 0.4465, 'ACCa': 0.9294, 'y1a': 0.2961, 'y2a': 0.6399})
    p = os.path.join(HERE, 'numeric_sample', 'Ackley_box3_final10.mat')
    F = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    nm = [str(np.squeeze(x)) for x in np.atleast_1d(F['arm_names']).ravel()]
    i8 = nm.index('KRAM-EVR')
    acc8 = np.asarray(F['acc_arr'], float)[:, i8, :]; rmae8 = np.asarray(F['rmae_resp'], float)[:, i8, :, :]
    spec = [(d, c, k) for (d, c), k in zip(SPEC_ACKLEY,
             ['LHS','MSE','IMSE','PALC','AK-MCS-IRS','piBO-global','piBO-ROI','KeyRegion','Voronoi','HHs'])]
    plot_case('ackley', os.path.join(HERE, 'numeric_sample'), spec,
              nm, np.asarray(F['acc_arr'], float), np.asarray(F['rmae_resp'], float), acc8, rmae8, 20, 50,
              {'ACC': 0.8460, 'y1': 0.1483, 'y2': 0.1501, 'ACCa': 0.8060, 'y1a': 0.2181, 'y2a': 0.2208})
    p = os.path.join(HERE, 'Radar', 'Ship_v1box3u_final10.mat')
    F = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    nm = [str(np.squeeze(x)) for x in np.atleast_1d(F['arm_names']).ravel()]
    i8 = nm.index('KRAM-EVR')
    acc8 = np.asarray(F['acc_arr'], float)[:, i8, :]; rmae8 = np.asarray(F['rmae_resp'], float)[:, i8, :, :]
    spec = [(d, c, k) for (d, c), k in zip(SPEC_SHIP,
             ['LHS','MSE','IMSE','PALC','AK-MCS-IRS','KeyRegion','Voronoi','HHs'])]
    plot_case('ship', os.path.join(HERE, 'Radar'), spec,
              nm, np.asarray(F['acc_arr'], float), np.asarray(F['rmae_resp'], float), acc8, rmae8, 20, 70,
              {'ACC': 0.8720, 'y1': 0.1331, 'y2': 0.0537, 'ACCa': 0.8338, 'y1a': 0.1601, 'y2a': 0.0647})
    print('ALL DONE')
