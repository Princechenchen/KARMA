"""
hpo_eval.py — HPO-mixed 黑箱单一事实源(Single Source of Truth, 2026-10-09)
一切真值评估(bank 预计算 / MATLAB 在线桥 / 冒烟)必须经由本模块的同一条代码路径:
    x∈[0,1]^8 --x_to_config--> LightGBM 参数 --SHA256 哈希--> 训练种子
           --固定500轮/单线程--> (y1=验证精度, train_acc) ; y2 在消费端按冻结 τ 派生
冻结口径(D2, 用户 2026-10-09 确认): 二分类 class2-vs-rest; 50k/20k 分层子采样
(种子与 smoke_lightgbm_timing2.py 位级一致); 固定 500 轮; num_threads=1; 无早停。
搜索空间文档镜像: 搜索空间表.md(修改须两处同步)。
CLI:
    python hpo_eval.py --selfcheck            # 评估 1 个固定配置并做确定性断言
    python hpo_eval.py --x 0.75,1.0,...       # 单点评估, 打印 JSON(在线桥调试用)
运行须在 HPO_mixed 目录内。
"""
import argparse
import hashlib
import json
import os

import numpy as np

SPLIT_NPZ = 'data/covtype_split.npz'
SUBSET_SEED_TRAIN = 20261009    
SUBSET_SEED_VAL = SUBSET_SEED_TRAIN + 1
N_TRAIN_SUB = 50000
N_VAL_SUB = 20000
ROUNDS = 500
N_THREADS = 1
TAU_DEFAULT = 0.05              

HP_NAMES = ['learning_rate', 'num_leaves', 'min_data_in_leaf', 'feature_fraction',
            'bagging_fraction', 'lambda_l1', 'lambda_l2', 'max_depth']


def x_to_config(x):
    """x∈[0,1]^8 -> LightGBM 超参 dict(搜索空间表.md 的代码实现)。"""
    x = np.asarray(x, dtype=float).ravel()
    assert x.size == 8, f'期望 8 维, 收到 {x.size}'
    x = np.clip(x, 0.0, 1.0)
    lr = float(10.0 ** (-4.0 + 4.0 * x[0]))
    leaves = int(np.clip(round(2 + 254.0 * x[1]), 2, 256))
    min_data = int(np.clip(round(2.0 * (200.0 / 2.0) ** x[2]), 2, 200))
    ff = float(0.3 + 0.7 * x[3])
    bf = float(0.3 + 0.7 * x[4])
    l1 = float(10.0 ** (-8.0 + 9.0 * x[5]))
    l2 = float(10.0 ** (-8.0 + 9.0 * x[6]))
    depth = -1 if x[7] < 0.15 else int(np.clip(round(3 + (x[7] - 0.15) / 0.85 * 13), 3, 16))
    return {'learning_rate': lr, 'num_leaves': leaves, 'min_data_in_leaf': min_data,
            'feature_fraction': ff, 'bagging_fraction': bf,
            'lambda_l1': l1, 'lambda_l2': l2, 'max_depth': depth}


def config_seed(cfg):
    """配置 -> 确定性训练种子(规范化 JSON 的 SHA256, mod 2^31-1)。"""
    canon = json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)
    return int(hashlib.sha256(canon.encode('utf-8')).hexdigest(), 16) % (2 ** 31 - 1)


def stratified_take(X, y, n, rng_seed):
    """分层抽取 n 行(与 smoke_lightgbm_timing2.py 逐字一致——位级复现前提, 勿动)。"""
    rng = np.random.default_rng(rng_seed)
    idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        take = max(1, int(round(n * len(ic) / len(y))))
        idx.append(rng.choice(ic, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


_SUBSETS = None


def load_subsets():
    """冻结的 50k/20k 分层子集(模块级缓存; worker 进程各持一份)。"""
    global _SUBSETS
    if _SUBSETS is None:
        d = np.load(SPLIT_NPZ, allow_pickle=False)
        Xtr, ytr = stratified_take(d['X_train'], d['y_train'],
                                   N_TRAIN_SUB, SUBSET_SEED_TRAIN)
        Xva, yva = stratified_take(d['X_val'], d['y_val'],
                                   N_VAL_SUB, SUBSET_SEED_VAL)
        _SUBSETS = (Xtr, ytr, Xva, yva)
    return _SUBSETS


def eval_config(x, subsets=None):
    """黑箱本体: 训练一次 LightGBM -> dict(y1, train_acc, gap, seed, 配置)。
    y2 不在此算——bank/在线端按冻结 τ 从 (train_acc-y1) 派生, 换 τ 无须重训。"""
    import lightgbm as lgb
    if subsets is None:
        subsets = load_subsets()
    X_tr, y_tr, X_va, y_va = subsets
    cfg = x_to_config(x)
    seed = config_seed(cfg)
    params = dict(objective='binary', num_threads=N_THREADS, verbosity=-1,
                  force_col_wise=True, seed=seed, bagging_freq=1, **cfg)
    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=ROUNDS)
    p_tr = model.predict(X_tr, num_iteration=None)
    p_va = model.predict(X_va, num_iteration=None)
    acc_tr = float(np.mean((p_tr > 0.5) == (y_tr == 1)))
    acc_va = float(np.mean((p_va > 0.5) == (y_va == 1)))
    out = {'y1': acc_va, 'train_acc': acc_tr, 'gap': acc_tr - acc_va, 'seed': seed}
    out.update(cfg)
    return out


def eval_online(x_str, cell_id):
    """MATLAB 在线桥入口: 逗号分隔的 8 维字符串 -> JSON 字符串。
    追加式缓存(每 cell 独立文件, 避免并行 MATLAB 会话竞争):
    先查 data/hpo_cache_<cell_id>.npz, 命中直接返回; 未命中评估后立即落盘。
    返回字段: y1 / train_acc / gap / seed / cache_hit。
    调用方(MATLAB)须保证 CWD=HPO_mixed(相对路径 'data/' 的解析前提)。"""
    import json
    ci = int(str(cell_id).strip())           
    if not (1 <= ci <= 99):
        raise ValueError('cell_id 须在 1..99')
    xv = np.array([float(v) for v in str(x_str).strip().split(',')])
    key = np.round(xv, 12)
    cache_file = 'data/hpo_cache_%d.npz' % ci
    if os.path.isfile(cache_file):
        d = np.load(cache_file, allow_pickle=False)
        hit = np.all(d['X'] == key[None, :], axis=1)
        if hit.any():
            j = int(np.where(hit)[0][0])
            return json.dumps({'y1': float(d['y1'][j]), 'train_acc': float(d['tr'][j]),
                               'gap': float(d['gap'][j]), 'seed': int(d['hseed'][j]),
                               'cache_hit': True})
    r = eval_config(xv)
    if os.path.isfile(cache_file):
        d = np.load(cache_file, allow_pickle=False)
        Xc = np.vstack([d['X'], key[None, :]])
        yc = np.concatenate([d['y1'], [r['y1']]])
        tc = np.concatenate([d['tr'], [r['train_acc']]])
        gc = np.concatenate([d['gap'], [r['gap']]])
        sc = np.concatenate([d['hseed'], [np.int64(r['seed'])]])
    else:
        Xc, yc, tc, gc = key[None, :], np.array([r['y1']]), \
            np.array([r['train_acc']]), np.array([r['gap']])
        sc = np.array([np.int64(r['seed'])])
    np.savez_compressed(cache_file, X=Xc, y1=yc, tr=tc, gap=gc, hseed=sc)
    return json.dumps({'y1': r['y1'], 'train_acc': r['train_acc'], 'gap': r['gap'],
                       'seed': r['seed'], 'cache_hit': False})


def _selfcheck():
    """评估 1 个固定配置(激进角: lr=0.1/leaves=256/min_data=2/无正则/不限深)×2 次,
    断言位级一致——交付时的全链路验证, 不产生任何 bank 数据。"""
    x = np.array([0.75, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    cfg = x_to_config(x)
    print('config:', json.dumps(cfg, ensure_ascii=False))
    assert abs(cfg['learning_rate'] - 0.1) < 1e-12 and cfg['num_leaves'] == 256 \
        and cfg['min_data_in_leaf'] == 2 and cfg['max_depth'] == -1, \
        'x_to_config 激进角编码与预期不符'
    r1 = eval_config(x)
    r2 = eval_config(x)
    print('run1:', json.dumps({k: r1[k] for k in ('y1', 'train_acc', 'gap', 'seed')}, indent=1))
    assert r1 == r2, '确定性 FAIL: 同配置两次结果不一致'
    print('确定性 PASS (y1=%.4f gap=%+.4f seed=%d)' % (r1['y1'], r1['gap'], r1['seed']))
    print('SELFCHECK ALLDONE')


def main():
    ap = argparse.ArgumentParser(description='HPO-mixed 黑箱单一事实源')
    ap.add_argument('--selfcheck', action='store_true', help='全链路自检(1 配置×2 次)')
    ap.add_argument('--x', type=str, default=None, help='8 维向量, 逗号分隔')
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
    elif args.x is not None:
        xv = np.array([float(v) for v in args.x.split(',')])
        r = eval_config(xv)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
