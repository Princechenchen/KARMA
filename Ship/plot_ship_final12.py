"""
ship 12 方法收敛曲线图 —— C6/C7 piBO 并入版
—— 展示规则: C8 = KRAM-EVR(v1box3u, 深蓝实线, 压轴置顶); 数据单源:
   Ship_v1box3u_final10.mat(并入后 12 臂; 列序=SUBSET_RSL)。
   12 曲线图例统一图外右侧。
   锚验证内建(KRAM-EVR/QBC/C6/C7 各六值)。
输出: fig_ship_final12_y1_RMAE.png / _y2_RMAE.png / _ACC.png (采样数量 21..70, 均值±std 带)
"""
import os

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

here = os.path.dirname(os.path.abspath(__file__))

SPEC = [
    ('LHS',         [0.8500, 0.3250, 0.0980]),    
    ('MSE',         [0.9290, 0.6940, 0.1250]),    
    ('IMSE',        [0.4940, 0.1840, 0.5560]),    
    ('PALC',        [0.6350, 0.0780, 0.1840]),    
    ('AK-MCS-IRS',  [0.3010, 0.7450, 0.9330]),    
    ('KeyRegion',   [0.4660, 0.6740, 0.1880]),    
    ('Voronoi',     [0.5000, 0.5000, 0.5000]),    
    ('HHs',         [0.2000, 0.2000, 0.2000]),    
    ('C6-piBO-global', [0.6400, 0.0800, 0.1800]), 
    ('C7-piBO-ROI',    [0.5000, 0.5000, 0.0000]), 
    ('KRAM-EVR',    [0.0000, 0.4470, 0.7410]),    
    ('QBC-global',  [0.8000, 0.2000, 0.6000]),    
]
EXPECT_ARMS = ['LHS', 'MSE', 'IMSE', 'PALC', 'AK-MCS-IRS', 'KeyRegion', 'Voronoi',
               'HHs', 'KRAM-EVR', 'C12-QBC-global', 'C6-piBO-global', 'C7-piBO-ROI']
LEGEND_SIZE, TICK_SIZE, LABEL_SIZE = 13, 14, 16
n_init, total_iter = 20, 70
n_iter = total_iter - n_init                
n_cont = 2
cn_font = FontProperties(family='SimHei', size=LABEL_SIZE, weight='bold')

P = scipy.io.loadmat(os.path.join(here, 'Ship_v1box3u_final10.mat'),
                     squeeze_me=True, struct_as_record=False)
acc = np.asarray(P['acc_arr'], float)       
rmae = np.asarray(P['rmae_resp'], float)    
assert acc.shape == (50, 12, 10) and rmae.shape == (50, 12, 10, 2), 'final10 形状不符'
an = np.atleast_1d(P['arm_names'])
while an.dtype == object and an.size >= 1 and isinstance(an.flat[0], np.ndarray):
    an = np.atleast_1d(an.flat[0])
pool_names = [str(x) for x in an]
assert pool_names == EXPECT_ARMS, '臂序与 SPEC 约定不一致'

nA = len(SPEC)
yRMAE = np.full((n_iter, 10, n_cont, nA), np.nan)
yACC = np.full((n_iter, 10, nA), np.nan)
NAME2ARC = {n: n for n, _ in SPEC}
NAME2ARC['QBC-global'] = 'C12-QBC-global'   
for mi, (name, _) in enumerate(SPEC):
    ia = pool_names.index(NAME2ARC[name])
    yACC[:, :, mi] = acc[:, ia, :]
    for j in range(n_cont):
        yRMAE[:, :, j, mi] = rmae[:, ia, :, j]
assert np.all(np.isfinite(yRMAE)) and np.all(np.isfinite(yACC)), '子集数据含 NaN'

def vals_of(mi):
    return [yRMAE[-1, :, 0, mi].mean(), yRMAE[-1, :, 1, mi].mean(),
            yRMAE[:, :, 0, mi].mean(axis=0).mean(), yRMAE[:, :, 1, mi].mean(axis=0).mean(),
            yACC[-1, :, mi].mean(), yACC[:, :, mi].mean()]
ANCH = {10: [0.1336, 0.0540, 0.1572, 0.063079, 0.8790, 0.8445],  
        11: [0.1471, 0.0542, 0.1823, 0.068657, 0.8540, 0.8320],  
        8:  [0.1350, 0.0545, 0.1609, 0.063571, 0.8720, 0.8487],  
        9:  [0.1347, 0.0546, 0.1623, 0.063986, 0.8530, 0.8470]}  
for mi, ref in ANCH.items():
    v = vals_of(mi)
    assert all(abs(a - b) < 5e-4 for a, b in zip(v, ref)), (SPEC[mi][0], v)
print('锚自校验 PASS: KRAM-EVR / QBC / C6 / C7 各六值')

iters = np.arange(n_init + 1, total_iter + 1)           

draw_order = list(range(8)) + [8, 9, 11, 10]
for c in range(n_cont):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mi in draw_order:
        name, color = SPEC[mi][0], SPEC[mi][1]
        y = yRMAE[:, :, c, mi]
        m = np.nanmean(y, axis=1); s = np.nanstd(y, axis=1, ddof=1)
        ax.plot(iters, m, '-', linewidth=2, color=color, label=name)
        ax.fill_between(iters, np.clip(m - s, 0, None), m + s,
                        facecolor=color, alpha=0.1, edgecolor='none')
    ax.set_xlim([n_init + 1, total_iter])
    ax.set_xlabel('采样数量', fontproperties=cn_font)
    ax.set_ylabel(f'$y_{{{c + 1}}}$ RMAE', fontname='Times New Roman',
                  fontsize=LABEL_SIZE, fontweight='bold')
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                    frameon=False, labelspacing=0.25,
                    prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax.grid(True, which='both', linestyle=':', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_box_aspect(0.75)
    plt.tight_layout()
    out = os.path.join(here, f'fig_ship_final12_y{c + 1}_RMAE.png')
    plt.savefig(out, dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

fig, ax = plt.subplots(figsize=(6, 4.5))
for mi in draw_order:
    name, color = SPEC[mi][0], SPEC[mi][1]
    y = yACC[:, :, mi]
    m = np.nanmean(y, axis=1); s = np.nanstd(y, axis=1, ddof=1)
    ax.plot(iters, m, '-', linewidth=2, color=color, label=name)
    ax.fill_between(iters, np.clip(m - s, 0, None), np.clip(m + s, None, 1.0),
                    facecolor=color, alpha=0.1, edgecolor='none')
ax.set_xlim([n_init + 1, total_iter])
ax.set_xlabel('采样数量', fontproperties=cn_font)
ax.set_ylabel('ACC', fontname='Times New Roman', fontsize=LABEL_SIZE, fontweight='bold')
leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                frameon=False, labelspacing=0.25,
                prop={'family': 'Times New Roman', 'size': LEGEND_SIZE})
ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.set_axisbelow(True)
ax.set_box_aspect(0.75)
plt.tight_layout()
out = os.path.join(here, 'fig_ship_final12_ACC.png')
plt.savefig(out, dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
plt.close()
print(f'Saved {out}')
print('ALL DONE')
