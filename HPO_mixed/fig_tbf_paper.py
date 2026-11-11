"""
fig_tbf_paper.py — BO信任口径收敛曲线 · 顶级学术出版级 v4(2026-11-08)
变更: ①移除 KARMA-EI-arb(6臂) ②全面优化视觉: 字号放大、去网格改轻刻度线、
     主角突出、色板统一 IEEE 审美。
Output: figs_v3/fig_hpo_tbf_paper.pdf + .png
"""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

rc = mpl.rcParams
rc['font.family'] = 'serif'
rc['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rc['font.size'] = 11
rc['axes.labelsize'] = 13
rc['axes.titlesize'] = 13
rc['legend.fontsize'] = 10.5
rc['xtick.labelsize'] = 11
rc['ytick.labelsize'] = 11
rc['axes.linewidth'] = 0.8
rc['xtick.major.width'] = 0.7
rc['ytick.major.width'] = 0.7
rc['xtick.major.size'] = 4
rc['ytick.major.size'] = 4
rc['xtick.direction'] = 'out'
rc['ytick.direction'] = 'out'
rc['lines.linewidth'] = 1.6
rc['lines.solid_capstyle'] = 'round'
rc['lines.dash_capstyle'] = 'round'
rc['pdf.fonttype'] = 42
rc['ps.fonttype'] = 42
rc['axes.spines.top'] = False
rc['axes.spines.right'] = False

d = sio.loadmat('HPO_final_repro.mat', squeeze_me=True, struct_as_record=False)
arm_names = [str(x).strip() for x in np.atleast_1d(d['arm_names'])]
tbf = np.asarray(d['tbf_curves'], float)  

STYLE = {
    'C14-ROI-EI':      dict(color='#C44E52', ls='-',  lw=2.6, zorder=10,
                            label='KARMA-EI (ours)'),
    'C12-QBC-global':  dict(color='#4C72B0', ls='-',  lw=1.4, zorder=6,
                            label='QBC'),
    'C7-piBO-ROI':     dict(color='#8172B2', ls='-.', lw=1.4, zorder=5,
                            label=r'$\pi$BO-ROI'),
    'C13-BO-EI':       dict(color='#55A868', ls='--', lw=1.4, zorder=4,
                            label='BO-EI'),
    'C6-piBO-global':  dict(color='#CCB974', ls=':',  lw=1.4, zorder=3,
                            label=r'$\pi$BO-global'),
    'C1-MaximinLHS':   dict(color='#9A9A9A', ls='-',  lw=1.2, zorder=2,
                            label='Maximin-LHS'),
}

fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
steps = np.arange(1, 51)

for i, nm in enumerate(arm_names):
    if nm not in STYLE:
        continue  
    st = STYLE[nm]
    mean = tbf[i].mean(axis=0)
    ax.plot(steps, mean, color=st['color'], ls=st['ls'], lw=st['lw'],
            zorder=st['zorder'], label=st['label'])

ax.set_xlabel('Acquisition step', fontweight='bold')
ax.set_ylabel('Trusted best-found', fontweight='bold')
ax.set_xlim(1, 50)
ax.set_ylim(0.768, 0.786)
ax.set_xticks([1, 10, 20, 30, 40, 50])
ax.set_yticks([0.770, 0.774, 0.778, 0.782, 0.786])
ax.tick_params(axis='both', which='major', pad=4)

leg = ax.legend(loc='upper left', ncol=2, frameon=True, fancybox=False,
                edgecolor='0.80', facecolor='white', framealpha=0.92,
                borderpad=0.5, handlelength=2.2,
                columnspacing=1.2, labelspacing=0.35,
                handletextpad=0.6)

fig.savefig('figs_v3/fig_hpo_tbf_paper.pdf', dpi=300, bbox_inches='tight')
fig.savefig('figs_v3/fig_hpo_tbf_paper.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('saved: figs_v3/fig_hpo_tbf_paper.pdf + .png')
