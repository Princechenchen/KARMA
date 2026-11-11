"""
generate_bank.py — HPO-mixed 真值 bank 生成(阶段1, 2026-10-09, ★待用户确认后执行)
分区(全部 LHS over [0,1]^8, 种子冻结; 评估一律走 hpo_eval.py 单一事实源):
    bank  12,000 配置 (LHS seed 20261101) — 每种子 testset/candidate0 从此分区瓜分
    pilot  1,000 配置 (LHS seed 20261102) — 先验构造/④扫描专用(已评估, 与 testset 互斥)
    init     200 配置 (10 种子 × 20, 每种子 LHS seed=rs) — 各种子初始设计
输出(全部为字面量相对路径, 不做动态拼接):
    data/hpo_bank.npz (python 侧权威) + data/hpo_bank.mat (MATLAB make_hpo_base 用)
    data/hpo_bank_partial.npz (断点续跑检查点, 完成后保留作凭证)
stdout 统计: y1 分布 / gap 分位数 / y2 正类率@τ∈{0.03..0.08}(D4 校准规则的数据)。
运行(须在本目录, 16 进程并行, 预计 ~35-70 分钟):
    cd HPO_mixed && F:/tools/Python39/python.exe generate_bank.py | tee generate_bank_log.txt
"""
import os
import time

import numpy as np
from multiprocessing import Pool
from scipy.stats import qmc
from scipy.io import savemat

import hpo_eval

SEEDS = [136, 34, 46, 77, 91, 158, 17, 43, 62, 78]   
BANK_N = 12000
PILOT_N = 1000
INIT_PER_SEED = 20
BANK_LHS_SEED = 20261101
PILOT_LHS_SEED = 20261102
N_PROC = 16
CKPT_EVERY = 500
TAU_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]      


def build_designs():
    """三分区设计矩阵(确定性: 种子冻结)。init 行序 = SEEDS 顺序, 每块 20 行。"""
    bank = qmc.LatinHypercube(d=8, seed=BANK_LHS_SEED).random(BANK_N)
    pilot = qmc.LatinHypercube(d=8, seed=PILOT_LHS_SEED).random(PILOT_N)
    init = np.concatenate(
        [qmc.LatinHypercube(d=8, seed=int(rs)).random(INIT_PER_SEED) for rs in SEEDS])
    return bank, pilot, init


_worker_subsets = None


def _init_worker():
    global _worker_subsets
    _worker_subsets = hpo_eval.load_subsets()


def _eval_one(job):
    idx, x = job
    r = hpo_eval.eval_config(x, subsets=_worker_subsets)
    return idx, r['y1'], r['train_acc'], r['gap'], r['seed']


def main():
    t0 = time.time()
    bank, pilot, init = build_designs()
    X_all = np.vstack([bank, pilot, init])
    n_all = X_all.shape[0]                     
    n_expected = BANK_N + PILOT_N + len(SEEDS) * INIT_PER_SEED
    assert n_all == n_expected, f'设计矩阵行数 {n_all} != 预期 {n_expected}'
    print(f'designs: bank {bank.shape} + pilot {pilot.shape} + init {init.shape} '
          f'= {n_all} configs, workers={N_PROC}')

    y1 = np.full(n_all, np.nan)
    tr_acc = np.full(n_all, np.nan)
    gap = np.full(n_all, np.nan)
    hseed = np.zeros(n_all, dtype=np.int64)
    done = np.zeros(n_all, dtype=bool)

    if os.path.isfile('data/hpo_bank_partial.npz'):
        d = np.load('data/hpo_bank_partial.npz', allow_pickle=False)
        di = np.asarray(d['done_idx'], dtype=int)
        y1[di] = d['y1_done']; tr_acc[di] = d['tr_done']; gap[di] = d['gap_done']
        hseed[di] = d['seed_done']
        done[di] = True
        print(f'resume: {len(di)}/{n_all} 已完成, 续跑 {n_all - len(di)}')
    jobs = [(i, X_all[i]) for i in range(n_all) if not done[i]]

    ndone0 = int(done.sum())
    ck_n = 0
    with Pool(N_PROC, initializer=_init_worker) as pool:
        for cnt, (idx, a, b, g, s) in enumerate(
                pool.imap_unordered(_eval_one, jobs, chunksize=8), 1):
            y1[idx] = a; tr_acc[idx] = b; gap[idx] = g; hseed[idx] = s
            done[idx] = True; ck_n += 1
            if cnt % 100 == 0:
                el = time.time() - t0
                eta = el / cnt * (len(jobs) - cnt)
                print(f'  {ndone0 + cnt}/{n_all} done ({el/60:.1f} min, '
                      f'ETA {eta/60:.0f} min)', flush=True)
            if ck_n >= CKPT_EVERY:
                di = np.where(done)[0]
                np.savez_compressed('data/hpo_bank_partial.npz',
                                    done_idx=di, y1_done=y1[di], tr_done=tr_acc[di],
                                    gap_done=gap[di], seed_done=hseed[di])
                ck_n = 0
    assert done.all(), '仍有未完成下标'

    di = np.where(done)[0]
    np.savez_compressed('data/hpo_bank_partial.npz',
                        done_idx=di, y1_done=y1[di], tr_done=tr_acc[di],
                        gap_done=gap[di], seed_done=hseed[di])

    rng = np.random.default_rng(0)
    for idx in rng.choice(n_all, size=3, replace=False):
        r = hpo_eval.eval_config(X_all[idx])
        assert abs(r['y1'] - y1[idx]) < 1e-15 and abs(r['gap'] - gap[idx]) < 1e-15, \
            f'确定性抽查 FAIL @ idx {idx}'
    print('确定性抽查 PASS (3/3 位级一致)')

    sl = slice(0, BANK_N)
    sp = slice(BANK_N, BANK_N + PILOT_N)
    si = slice(BANK_N + PILOT_N, n_all)
    np.savez_compressed(
        'data/hpo_bank.npz',
        X_bank=X_all[sl], y1_bank=y1[sl], train_acc_bank=tr_acc[sl],
        gap_bank=gap[sl], hseed_bank=hseed[sl],
        X_pilot=X_all[sp], y1_pilot=y1[sp], train_acc_pilot=tr_acc[sp],
        gap_pilot=gap[sp],
        X_init=X_all[si], y1_init=y1[si], train_acc_init=tr_acc[si],
        gap_init=gap[si],
        init_seed_ids=np.repeat(np.arange(len(SEEDS)), INIT_PER_SEED),
        seeds=np.asarray(SEEDS, dtype=np.int64),
        rounds=np.int64(hpo_eval.ROUNDS), tau_default=np.float64(hpo_eval.TAU_DEFAULT),
    )
    savemat('data/hpo_bank.mat',
            {'X_bank': X_all[sl], 'y1_bank': y1[sl], 'train_acc_bank': tr_acc[sl],
             'gap_bank': gap[sl],
             'X_pilot': X_all[sp], 'y1_pilot': y1[sp], 'train_acc_pilot': tr_acc[sp],
             'gap_pilot': gap[sp],
             'X_init': X_all[si], 'y1_init': y1[si], 'train_acc_init': tr_acc[si],
             'gap_init': gap[si],
             'init_seed_ids': np.repeat(np.arange(len(SEEDS)), INIT_PER_SEED) + 1,
             'SEEDS': np.asarray(SEEDS),
             'ROUNDS': float(hpo_eval.ROUNDS), 'TAU_DEFAULT': float(hpo_eval.TAU_DEFAULT)},
            do_compression=True)

    q = np.quantile(gap, [0, .05, .25, .5, .75, .95, 1])
    print('===STATS===')
    print(f'y1: min={y1.min():.4f} med={np.median(y1):.4f} mean={y1.mean():.4f} '
          f'max={y1.max():.4f}')
    print('gap 分位数 [0,5%,25%,50%,75%,95%,100%]: ' + ' '.join(f'{v:+.4f}' for v in q))
    print(f'train_acc: min={tr_acc.min():.4f} max={tr_acc.max():.4f}')
    for nm, sl2 in [('bank', sl), ('pilot', sp)]:
        g = gap[sl2]
        row = ' '.join(f'{t:.2f}:{np.mean(g > t) * 100:5.1f}%' for t in TAU_GRID)
        print(f'y2 正类率({nm}, D4 校准网格): {row}   [带内标准 15%-85%]')
    wall = (time.time() - t0) / 60
    rate = n_all / (wall * 60) if wall > 0 else float('nan')
    print(f'configs={n_all} wall={wall:.1f} min ({rate:.1f} cfg/min, {N_PROC} workers)')
    print('ALLDONE')


if __name__ == '__main__':
    main()
