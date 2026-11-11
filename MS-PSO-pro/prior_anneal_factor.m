function f = prior_anneal_factor(Xq, Xv, k, L)
% prior_anneal_factor - box3-C 先验场退火因子(2026-08-26)
%   f(x) = min(1, ρ̂(x)^(-1/L))
%   ρ̂(x) = kNN 局部密度 / 空间平均密度(无量纲):
%          ρ̂ = kk / (n · V_d · d_k(x)^d),  V_d = d 维单位球体积, 均匀参考下 ρ̂≈1。
%   语义: ρ̂ ≤ 1(欠采/处女地) → f = 1, 先验全额生效; ρ̂ > 1(超额开采) → 按
%   对数尺度退火 —— 局部密度每翻 2^L 倍, 先验强度减半。被反复开采的区域先验
%   自动"退休", 从未被碰的区域先验全额。随 Xv(已采样点集)每步演化,
%   使先验场的【区域间排序】随数据真实改变(Pareto 保序不变性被武器化);
%   纯解析计算, 不消耗随机数, 不调代理模型。
%
%   输入: Xq [nq×d] 查询点(归一化域 [0,1]^d); Xv [n×d] 已采样点;
%         k   近邻数(默认 5);
%         L   退火尺度(默认 3, 即密度×8 → 先验减半)。
%   输出: f [nq×1] ∈ (0,1]
    if nargin < 3 || isempty(k), k = 5; end
    if nargin < 4 || isempty(L), L = 3; end
    [~, d] = size(Xq);
    n = size(Xv, 1);
    kk = max(1, min(k, n));
    D = sort(pdist2(Xq, Xv), 2);
    d_k = D(:, kk) + 1e-12;                          % 第 kk 近邻距离
    Vd = pi^(d/2) / gamma(d/2 + 1);                  % 单位球体积(d=6: π³/6≈5.17)
    rho_hat = kk ./ (n * Vd * d_k.^d);               % 局部/平均密度比(边界处低估→保守)
    f = min(1, rho_hat .^ (-1/L));
end
