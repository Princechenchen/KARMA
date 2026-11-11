# gp_single —— 单响应高斯过程建模与预测(MATLAB,自包含)

单输出(exact)GP 回归的最小可用代码包:极大边际似然估计超参 + 后验预测 +
留一法评估。**不依赖 GPML / 统计与优化工具箱**,只需基础 MATLAB(开发验证于 R2022a)。

## 模型

y_i = f(x_i) + ε_i,f ~ GP(m, k),ε_i ~ N(0, sn²)

nlml = ½ (y−m1)ᵀ K⁻¹ (y−m1) + ½ log|K| + (n/2) log 2π,K = k(X,X) + sn² I

梯度:dnlml/dθ = ½ tr[(K⁻¹ − ααᵀ) ∂K/∂θ],α = K⁻¹(y−m1)(对 log 超参)。

## 使用说明(建模 gp_train + 预测 gp_predict)

完整可运行示例见 `gp_example.m`(无画图,涵盖建模/预测/LOO/重复测量),
运行:`matlab -batch gp_example`(在本目录下)。以下为分步说明。

### 0) 准备

```matlab
addpath <本项目路径>/gp_single
```

数据要求:**X 为 n×d 矩阵**(每行一个样本,d 为输入维数),**y 为 n×1 列向量**(单响应)。
重复测量(同一输入多次测量)不需任何预处理,直接堆成多行(见"重复观测"一节)。

### 1) 建模:gp_train

```matlab
model = gp_train(X, y);                          % 最简用法,默认即可用
```

这一步完成:z 标准化 → 多起点 BFGS 极大化边际似然(估各维长度尺度 ℓ、
信号强度 sf、噪声 sn、均值)→ 缓存 Cholesky 因子,返回 model 结构体。

可选名称-值参数(全部可省略):

```matlab
model = gp_train(X, y, ...
    'cov',      'gpCovSEard', ...   % 核:'gpCovSEard'(默认,逐维长度尺度)|'gpCovSEiso'|'gpCovMatern52'
    'mean',     'const', ...        % 均值:'const'(默认)|'zero'
    'nrestart', 5, ...              % 随机重启次数(防局部最优,取 nlml 最小者)
    'seed',     1, ...              % 随机种子(第 r 次重启用 rng(seed+r-1),保证可复现)
    'normalize',true, ...           % X/y 标准化(默认开,建议保持)
    'jitter',   1e-10, ...          % Cholesky 抖动
    'verbose',  true);              % 打印每次重启的 nlml,排查收敛用
```

训练后常看的字段:

```matlab
model.nlml_raw                     % 原始尺度负对数边际似然(越小越好,可比较不同核)
exp(model.hyp_raw.cov(1:end-1))    % 各维长度尺度 ℓ(原始单位,已自动换算)
exp(model.hyp_raw.cov(end))        % 信号强度 sf
exp(model.hyp_raw.lik)             % 噪声标准差 sn
```

### 2) 预测:gp_predict

```matlab
[mu, s2]    = gp_predict(model, Xs);     % 基本形式
[mu, s2]    = gp_predict(model, Xs, true);   % 含噪口径
[mu, s2, ci] = gp_predict(model, Xs);    % 附加 95% 置信区间
```

- **输入**:model(gp_train 返回)、Xs 为 m×d 测试输入(任意点,可为训练点本身)
- **mu**:m×1 后验均值(**原始 y 尺度**)
- **s2**:m×1 后验方差(平方单位;标准差取 sqrt(s2))
- **ci**:m×2,95% 置信区间 [下界, 上界]

两种口径:缺省(第三参数 false)预测**潜在函数 f 本身**(SUR/主动采样算
改进量、不确定性用这个);`true` 预测**含噪观测 y = f + ε**(方差加 sn²,
评估"下次测量会看到什么"用这个)。

### 3) 模型质量评估(可选):gp_loo

```matlab
[rmse, mlpd] = gp_loo(model);       % 一行封闭式留一交叉验证
```

rmse 为原始尺度 LOO-RMSE;mlpd 为平均对数预测密度(越接近 0 越好)。

### 4) 典型完整流程

```matlab
addpath <本项目路径>/gp_single

rng(1);  X = rand(40,2);  y = f(X);          % 1) 数据 n×d / n×1
model = gp_train(X, y, 'verbose', true);      % 2) 建模
[mu, s2, ci] = gp_predict(model, Xtest);      % 3) 预测任意点
[rmse, mlpd] = gp_loo(model);                 % 4) 评估(可选)
```

## 文件一览

| 文件 | 说明 |
|---|---|
| `gp_train.m` | 建模入口:z 标准化 + 多起点 BFGS 极大边际似然 + 预缓存(L, α) |
| `gp_predict.m` | 后验预测:均值 / 方差(潜在或含噪)/ 95% CI,还原到原始尺度 |
| `gp_loo.m` | 封闭式留一法(R&W 式 5.4.2),返回 LOO-RMSE 与平均对数预测密度 |
| `gp_example.m` | 完整使用示例(无画图):建模 / 预测 / LOO / 重复测量 |
| `gp_nll.m` | 负对数边际似然 + 解析梯度(核心数学) |
| `gp_bfgs.m` | 自带 BFGS(Armijo 回溯),不依赖优化工具箱 |
| `gpCovSEard.m` | 平方指数核,ARD 长度尺度(默认) |
| `gpCovSEiso.m` | 平方指数核,单一长度尺度 |
| `gpCovMatern52.m` | Matérn 5/2 核,ARD |
| `gpCovNoise.m` | 独立高斯噪声核(自协方差 sn²·I) |
| `gp_check_gradients.m` | 自检:解析梯度 vs 中心差分 |

## 约定(改代码/扩展前必读)

- **超参一律 log 尺度**,打包顺序 `[hyp.cov; hyp.mean(仅 const); hyp.lik]`,
  核超参**末位恒为 log sf**(gp_predict 直接用 `hyp.cov(end)` 取 sf²)。
- 默认在标准化空间优化;`model.hyp_raw` 为原始尺度换算
  (ℓ_raw = ℓ·sdX,σf_raw = σf·sdy,sn_raw = sn·sdy,均值同理);
  `model.nlml_raw = model.nlml + n·log(sdy)`。
- `model.L` 为 MATLAB `chol` 的**上三角**因子(K = LᵀL),求解一律
  `K⁻¹v = L\(L'\v)`(曾在此踩坑:上下三角求解次序写反,梯度自检可查出)。
- 复数位防抖:chol 失败时抖动逐级放大(×10³、×10⁶),仍失败返回罚点 1e10。

## 重复观测(同一输入多次测量)

支持,直接把每次测量作为独立一行放进 X、y 即可,无需去重或预平均:

- 模型上 `gpCovNoise` 给每行独立加 sn²·I,相同输入的行由此解相关;
  MLE 会自动从重复值的散布估出 sn(重复即噪声的"免费样本")。
- 确定性数据 + 完全相同的重复值:sn 会塌向 ~0,K 趋奇异,靠 jitter
  逐级放大保住 Cholesky,预测退化为精确插值(已实测,误差 ~1e-11)。
- 注意:若"多组"指每个 x 对应**多个不同响应量**(多输出/多保真),
  那是多响应 GP,不在此包范围。

## 验证

```bash
cd gp_single
/e/MATLAB/bin/matlab.exe -batch gp_check_gradients   # 全部 PASS,最大相对误差 ~1e-10
/e/MATLAB/bin/matlab.exe -batch gp_example           # 端到端示例
```

gp_check_gradients 实测:三种核的解析梯度与中心差分一致到 1e-10。
gp_example 实测(log-Branin_2d3,40 训练点 / 300 测试点):test RMSE 0.28、
95% CI 覆盖率 89%、LOO-RMSE 0.50;重复测量节(15 点 × 3 次,真噪声 0.2)
估出 sn = 0.14。

## 限制

- 精确 GP:训练 O(n³)、预测 O(n·m),适合 n ≲ 2000。
- 单响应输出;同方差高斯噪声;均值仅常数/零(扩展 mean 函数需同步改
  gp_nll 的 e 与均值梯度、gp_predict 的 mu)。
