function [rmse, mlpd, yloo, s2loo] = gp_loo(model)
% gp_loo  封闭式留一法(leave-one-out,基于含噪 K 的解析式,见 R&W 式 5.4.2)
%
%   g = K^{-1} e,  h = diag(K^{-1})
%   LOO 均值:  mu_i  = m + e_i - g_i/h_i
%   LOO 方差:  s2_i  = 1/h_i        (对含噪 y_i 的预测)
%
% 输出:
%   rmse   原始尺度 LOO-RMSE
%   mlpd   原始尺度平均对数预测密度 mean log p(y_i | y_{-i})
%   yloo, s2loo  各训练点的 LOO 预测均值/方差(原始尺度)

n = model.n;
Kinv = model.L\(model.L'\eye(n));   % chol 上三角 R(K = R'R):K^{-1} = R\(R'\I)
if strcmp(model.meanfn,'const')
    e = model.yn - model.hyp.mean;
else
    e = model.yn;
end
g = Kinv*e;
h = diag(Kinv);

mu_n = e - g./h;                       % m + e_i - g_i/h_i(m 已并入 e)
s2_n = 1./h;
yloo = model.norm.muy + model.norm.sdy * mu_n;
s2loo = model.norm.sdy^2 * s2_n;

rmse = sqrt(mean((model.y - yloo).^2));
mlpd = mean(-0.5*log(2*pi*s2loo) - (model.y - yloo).^2./(2*s2loo));
end
