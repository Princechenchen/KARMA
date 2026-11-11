"""
analyze_pilot_prior.py — 先验设计会证据包(2026-10-09, 阶段2/B8)
数据纪律: ★只读 hpo_bank.npz 的 pilot 分区(1,000 配置)——bank 12k 行是10 种子
testset/候选池, 用于先验构造即测试集泄漏(且④成本记账将失真); 200 init 同理不用。
产出:
  figs_prior/pilot_y1_axes.png    y1 沿 8 轴的分箱均值±std(找悬崖)
  figs_prior/pilot_gap_axes.png   过拟合率(gap>τ) 沿 8 轴的分箱(找边界)
  figs_prior/pilot_lr_detail.png  y1 与 gap vs lr 轴细看
  figs_prior/pilot_capacity.png   gap vs (x2 leaves, x3 min_data) 平面 + 带假设
  stdout: 悬崖定界/过拟合主导分裂(浅决策树)/好配置域(稳定盒)/v2 常数提案 vs v1 草稿
运行(须在本目录): F:/tools/Python39/python.exe analyze_pilot_prior.py | tee analyze_pilot_prior_log.txt
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text

TAU = 0.05
AX_NAMES = ['x1 lr', 'x2 leaves', 'x3 min_data', 'x4 ff',
            'x5 bf', 'x6 l1', 'x7 l2', 'x8 depth']
os.makedirs('figs_prior', exist_ok=True)

d = np.load('data/hpo_bank.npz', allow_pickle=False)
X = d['X_pilot']; y1 = d['y1_pilot']; gap = d['gap_pilot']
cls = (gap > TAU).astype(int)
n = len(y1)
print(f'pilot n={n}, y1 [{y1.min():.4f},{y1.max():.4f}] med {np.median(y1):.4f}, '
      f'overfit率 {cls.mean()*100:.1f}%')


def binned_profile(vals, mask=None, nb=10):
    """逐轴分箱统计(返回每轴每箱的均值/计数)。"""
    out = {}
    for k in range(8):
        edges = np.linspace(0, 1, nb + 1)
        idx = np.digitize(X[:, k], edges) - 1
        idx = np.clip(idx, 0, nb - 1)
        m = mask if mask is not None else np.ones(n, bool)
        mu = np.full(nb, np.nan); cnt = np.zeros(nb)
        for b in range(nb):
            s = m & (idx == b)
            cnt[b] = s.sum()
            if s.sum() >= 5:
                mu[b] = vals[s].mean()
        out[k] = (mu, cnt, edges)
    return out


prof_y1 = binned_profile(y1)
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
for k in range(8):
    mu, cnt, edges = prof_y1[k]
    c = edges[:-1] + 0.5 / 10
    ax = axes[k // 4][k % 4]
    ax.bar(c, mu, width=0.08, color='steelblue')
    ax.set_title(AX_NAMES[k]); ax.set_ylim(0.4, 0.85)
    if k % 4 == 0:
        ax.set_ylabel('y1 (val acc)')
plt.tight_layout()
plt.savefig('figs_prior/pilot_y1_axes.png', dpi=110)
plt.close(fig)

print('\n=== y1 逐轴悬崖(相邻分箱均值最大落差) ===')
for k in range(8):
    mu, cnt, edges = prof_y1[k]
    dm = np.abs(np.diff(mu))
    j = int(np.nanargmax(dm))
    print(f'{AX_NAMES[k]:14s}: 最大落差 {dm[j]:+.4f} @ x≈{(edges[j]+edges[j+1+1])/2:.2f} '
          f'(箱均值 {[f"{v:.3f}" for v in mu]})')

prof_gap = binned_profile(cls)
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
for k in range(8):
    mu, cnt, edges = prof_gap[k]
    c = edges[:-1] + 0.5 / 10
    ax = axes[k // 4][k % 4]
    ax.bar(c, mu, width=0.08, color='indianred')
    ax.set_title(AX_NAMES[k]); ax.set_ylim(0, 1)
    if k % 4 == 0:
        ax.set_ylabel('P(gap>tau)')
plt.tight_layout()
plt.savefig('figs_prior/pilot_gap_axes.png', dpi=110)
plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for vals, ax, nm, col in [(y1, a1, 'y1', 'steelblue'), (gap, a2, 'gap', 'indianred')]:
    ax.scatter(X[:, 0], vals, s=8, alpha=0.35, color=col)
    mu, cnt, edges = (prof_y1 if nm == 'y1' else binned_profile(gap))[0]
    c = edges[:-1] + 0.05
    ax.plot(c, mu, 'k.-', lw=1.5, label='bin mean')
    if nm == 'gap':
        ax.axhline(TAU, color='gray', ls='--', label='tau')
    ax.set_xlabel('x1 (lr)'); ax.set_title(nm); ax.legend()
plt.tight_layout()
plt.savefig('figs_prior/pilot_lr_detail.png', dpi=110)
plt.close(fig)

med = np.median(y1)
mu0 = prof_y1[0][0]
low_bins = np.where(mu0 < 0.75 * max(mu0[~np.isnan(mu0)]))[0]
dead_edge = low_bins.max() * 0.1 + 0.1 if len(low_bins) else 0.0
hi_bins = low_bins  
print(f'\nlr 轴低值箱(均值<75%峰值): bins {low_bins} → 死区上界≈{dead_edge:.2f}, '
      f'高端低箱 bins {[b for b in range(10) if b not in low_bins and mu0[b] < 0.75*max(mu0[~np.isnan(mu0)])]}')

fig, ax = plt.subplots(figsize=(6.5, 5))
s = ax.scatter(X[:, 2], X[:, 3], c=gap, s=14, cmap='coolwarm', vmin=-0.02, vmax=0.13)
xx = np.linspace(0, 1, 50)
for cc, lab in [(0.85, 'x2+x3=0.85 (v1 draft)'), (0.75, 'x2+x3=0.75')]:
    ax.plot(xx, cc - xx, '--', label=lab)
plt.colorbar(s, label='gap')
ax.set_xlabel('x2 (num_leaves)'); ax.set_ylabel('x3 (min_data_in_leaf)')
ax.legend(); ax.set_title('gap on capacity plane (pilot)')
plt.tight_layout()
plt.savefig('figs_prior/pilot_capacity.png', dpi=110)
plt.close(fig)
rho = np.corrcoef(X[:, 2] + X[:, 3], gap)[0, 1]
print(f'\nx2+x3 与 gap 的 Pearson r = {rho:+.3f} (v1 草稿带假设 strip_c=0.85)')

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40).fit(X, cls)
print('\n=== 过拟合类的浅决策树分裂(可读规则) ===')
print(export_text(tree, feature_names=[a.replace(' ', '_') for a in AX_NAMES],
                  decimals=2))

good = (y1 > np.quantile(y1, 0.75)) & (cls == 0)
print(f'\n=== 好配置域(y1>75%分位 且 gap<=τ, n={good.sum()}) 逐轴 [P10,P90] ===')
box = {}
for k in range(8):
    q = np.quantile(X[good, k], [0.10, 0.90])
    box[k] = [float(q[0]), float(q[1])]
    print(f'{AX_NAMES[k]:14s}: [{q[0]:.2f}, {q[1]:.2f}]')

print('\n=== v2 常数提案(数据驱动) vs v1 草稿(文档常识) ===')
proposal = {'dead_edge': f'取 lr 低值箱上界 ≈{dead_edge:.2f} (v1=0.30)',
            'high_edge': '看 pilot_y1_axes/lr_detail 高端劣化位置人工定 (v1=0.92)',
            'strip_c': f'由决策树首分裂与 capacity 图人工定 (v1=0.85; x2+x3 r={rho:+.2f})',
            'good_box': json.dumps(box)}
for kk, vv in proposal.items():
    print(f'{kk:10s}: {vv}')
print('ANALYZE ALLDONE')
