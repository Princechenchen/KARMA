"""
build_hpo_final_repro.py — HPO-mixed 复现数据包构建(2026-10-22)
将 BO 主表 7 臂 × 10 种子的全部实验数据打包为一个 mat 文件,
包含: 实验设置 / 先验几何 / 种子表 / 黑箱协议 / 7臂逐臂逐种子完整数据。
运行(须在本目录): F:/tools/Python39/python.exe build_hpo_final_repro.py
"""
import numpy as np
import scipy.io as sio
from collections import OrderedDict

TAU = 0.05

bk = sio.loadmat('data/hpo_bank.mat', squeeze_me=True, struct_as_record=False)
Xb = np.asarray(bk['X_bank'], float); yb = np.asarray(bk['y1_bank'], float).ravel()
gb = np.asarray(bk['gap_bank'], float).ravel()
Xi = np.asarray(bk['X_init'], float); yi = np.asarray(bk['y1_init'], float).ravel()
gi = np.asarray(bk['gap_init'], float).ravel()
Xp = np.asarray(bk['X_pilot'], float); yp = np.asarray(bk['y1_pilot'], float).ravel()
be = sio.loadmat('rec_hpo_base.mat', squeeze_me=True, struct_as_record=False)

def get_picks(obj):
    return np.asarray(obj[0] if getattr(obj, 'dtype', None) == object
                      and np.atleast_1d(obj).size > 1 else obj, float)

def full_stats(P, c):
    y = np.zeros(50); g = np.zeros(50)
    for r in range(50):
        m = np.where(np.all(np.abs(Xb - P[r]) < 1e-12, axis=1))[0]
        y[r] = yb[m[0]]; g[r] = gb[m[0]]
    iy, ig = yi[c*20-20:c*20], gi[c*20-20:c*20]
    bf = np.maximum.accumulate(np.concatenate([[iy.max()], y]))[1:]
    tv = np.concatenate([[iy[ig <= TAU].max() if (ig <= TAU).any() else -np.inf],
                         np.where(g <= TAU, y, -np.inf)])
    tbf = np.maximum.accumulate(tv)[1:]
    return bf, tbf, y, g

ARMS = OrderedDict([
    ('C14-ROI-EI',      ('bo',   'c14')),
    ('C14b-ROI-EI-arb', ('bo',   'c14b')),
    ('C12-QBC-global',  ('main', 'C12-QBC-global')),
    ('C7-piBO-ROI',     ('main', 'C7-piBO-ROI')),
    ('C13-BO-EI',       ('main', 'C13-BO-EI')),
    ('C6-piBO-global',  ('main', 'C6-piBO-global')),
    ('C1-MaximinLHS',   ('main', 'C1-MaximinLHS')),
])
arm_names = list(ARMS.keys())
nA = len(arm_names)

picks_all    = np.zeros((nA, 10, 50, 8))
bf_curves    = np.zeros((nA, 10, 50))
tbf_curves   = np.zeros((nA, 10, 50))
picks_y1     = np.zeros((nA, 10, 50))
picks_gap    = np.zeros((nA, 10, 50))
picks_overfit= np.zeros((nA, 10, 50))  

for ai, (disp, (src, key)) in enumerate(ARMS.items()):
    for c in range(1, 11):
        if src == 'main':
            ck = sio.loadmat('unified_hpo_mixed_ckpt_c%d.mat' % c,
                             squeeze_me=True, struct_as_record=False)
            names = [str(x) for x in np.atleast_1d(ck['arm_names'])]
            P = get_picks(ck['picks'][names.index(key)])
        else:
            d = sio.loadmat('unified_hpo_%s_ckpt.mat' % key,
                            squeeze_me=True, struct_as_record=False)
            P = get_picks(d['picks'][c-1])
        bf, tbf, y, g = full_stats(P, c)
        picks_all[ai, c-1] = P
        bf_curves[ai, c-1] = bf
        tbf_curves[ai, c-1] = tbf
        picks_y1[ai, c-1] = y
        picks_gap[ai, c-1] = g
        picks_overfit[ai, c-1] = (g > TAU).astype(float)

bf_final  = bf_curves[:, :, -1]     
tbf_final = tbf_curves[:, :, -1]
ovr_rate  = picks_overfit.mean(axis=2)  

sio.savemat('HPO_final_repro.mat', {
    'arm_names':    arm_names,
    'n_arms':       nA,
    'n_seeds':      10,
    'n_init':       20,
    'n_seq':        50,
    'TAU':          TAU,
    'rounds':       500,
    'n_threads':    1,
    'dim':          8,
    'pool_size':    2000,
    'testset_size': 10000,
    'seeds':        np.array([136, 34, 46, 77, 91, 158, 17, 43, 62, 78]),
    'dataset':      'Covertype (UCI 581012x54, 7-class, binary class2-vs-rest)',
    'train_sub':    50000,
    'val_sub':      20000,
    'split_seed':   20261009,
    'sub_seed_tr':  20261009,
    'sub_seed_va':  20261010,
    'objective':    'binary class2-vs-rest, fixed 500 rounds, seed=SHA256(config)',
    'prior_version': 'v3 (Purpose C)',
    'pi_hg_type':   'sweet-zone soft box on x1',
    'pi_hg_lo':     0.54,
    'pi_hg_hi':     0.76,
    'pi_hg_ew':     0.08,
    'pi_bnd_type':  'overfit-box shell (x1>=0.70, x2>=0.25)',
    'pi_bnd_tg':    0.08,
    'kernel':       'correxp (MMRGP0_correxp)',
    'ClassPos':     np.array([1, 1, 2]),
    'resp_layout':  'y1=val_acc(continuous), y2=gap(continuous), class=1[gap>tau]',
    'picks_all':     picks_all,       
    'picks_y1':      picks_y1,        
    'picks_gap':     picks_gap,       
    'picks_overfit': picks_overfit,   
    'bf_curves':     bf_curves,       
    'tbf_curves':    tbf_curves,      
    'bf_final':      bf_final,        
    'tbf_final':     tbf_final,       
    'ovr_rate':      ovr_rate,        
    'bf_final_mean':  bf_final.mean(axis=1),
    'bf_final_std':   bf_final.std(axis=1, ddof=1),
    'tbf_final_mean': tbf_final.mean(axis=1),
    'tbf_final_std':  tbf_final.std(axis=1, ddof=1),
    'ovr_rate_mean':  ovr_rate.mean(axis=1),
    'bank_max':       float(yb.max()),
    'trusted_max':    float(yb[gb <= TAU].max()),
    'top1pct_line':   float(np.quantile(yb, 0.99)),
    'top1pct_overfit_rate': float((gb[yb >= np.quantile(yb, 0.99)] > TAU).mean()),
}, do_compression=True)

print('HPO_final_repro.mat saved')
print('arms:', arm_names)
print('BF-final mean:', np.round(bf_final.mean(axis=1), 4))
print('tBF-final mean:', np.round(tbf_final.mean(axis=1), 4))

d = sio.loadmat('HPO_final_repro.mat', squeeze_me=True, struct_as_record=False)
assert len(d['arm_names']) == 7
assert d['tbf_curves'].shape == (7, 10, 50)
assert abs(d['tbf_final_mean'][0] - 0.7804) < 0.001  
assert abs(d['tbf_final_mean'][-1] - 0.7733) < 0.001  
print('读回验证 PASS')
print('文件大小: %.1f MB' % (os.path.getsize('HPO_final_repro.mat') / 1e6 if __import__('os').path.getsize('HPO_final_repro.mat') else 0))

import os
print('文件大小: %.1f MB' % (os.path.getsize('HPO_final_repro.mat') / 1e6))
print('BUILD REPRO ALLDONE')
