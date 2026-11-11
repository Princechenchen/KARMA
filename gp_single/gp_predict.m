function [mu, s2, ci] = gp_predict(model, Xs, noisy)
% gp_predict  单响应 GP 后验预测
%
%   [mu, s2]       = gp_predict(model, Xs)      潜在函数 f(*) 的后验均值/方差
%   [mu, s2]       = gp_predict(model, Xs, true) 预测含噪观测值 y* = f + ε
%   [mu, s2, ci]   = gp_predict(model, Xs)      附加 95% 置信区间 [mu±1.96*sqrt(s2)]
%
% 输入: model(gp_train 返回)、Xs: m×d 测试输入;输出均为原始 y 尺度

if nargin < 3, noisy = false; end
Xs = Xs(:,:);
assert(size(Xs,2) == model.d, 'gp_predict: 输入维数与模型不一致');

Xsn = (Xs - model.norm.muX)./model.norm.sdX;
Ks  = feval(model.covfn, model.hyp.cov, model.Xn, Xsn);   % n×m 互协方差

mu_n = Ks' * model.alpha;                                  % 标准化空间后验均值
if strcmp(model.meanfn,'const'), mu_n = mu_n + model.hyp.mean; end

w   = model.L' \ Ks;                                       % K^{-1}Ks = R^{-T}Ks(chol 上三角 R)
sf2 = exp(2*model.hyp.cov(end));                           % 约定:核超参末位 = log sf
v   = sf2 - sum(w.^2, 1);
v   = max(v, 0);                                           % 数值保护
if noisy, v = v + exp(2*model.hyp.lik); end

mu = model.norm.muy + model.norm.sdy * mu_n;               % 还原原始尺度
s2 = model.norm.sdy^2 * v(:);
if nargout == 3
    hw = 1.96*sqrt(s2);
    ci = [mu - hw, mu + hw];
end
end
