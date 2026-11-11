function [nlml, dnlml] = gp_nll(hyp, covfn, meanfn, X, y, jitter)
% gp_nll  单响应 GP 的负对数边际似然(NLML)及其解析梯度(对 log 超参)
%
%   nlml = 0.5*(y-m)' K^{-1} (y-m) + 0.5*log|K| + n/2*log(2*pi)
%   K    = covfn(hyp.cov, X) + sn^2 I ;  m = hyp.mean(常数均值)或 0
%   梯度: dnlml/dθ = 0.5 * sum( (K^{-1} - alpha*alpha') ⊙ dK/dθ ),  alpha = K^{-1}(y-m)
%
% 输入:
%   hyp    结构体: hyp.cov = 核 log 超参列向量; hyp.lik = log sn;
%          hyp.mean = 常数均值(meanfn='const' 时存在)
%   covfn  核句柄(@gpCovSEard / @gpCovSEiso / @gpCovMatern52)
%   meanfn 'const' | 'zero'
%   X, y   n×d / n×1(建议传标准化后的数据;标准化由 gp_train 统一处理)
%   jitter Cholesky 抖动,缺省 1e-10
% 输出:
%   dnlml  打包顺序 [d hyp.cov; d hyp.mean(仅 const); d hyp.lik](与 gp_train 一致)

if nargin < 6 || isempty(jitter), jitter = 1e-10; end
n = size(X,1);
hasMean = strcmp(meanfn,'const');
if hasMean, m = hyp.mean; else, m = 0; end

K = feval(covfn, hyp.cov, X) + feval(@gpCovNoise, hyp.lik, X);
e = y - m;

% Cholesky(抖动逐级放大,仍失败则罚点让优化器回退)
p = 1;
for jj = [1 1e3 1e6]
    [L, p] = chol(K + jj*jitter*eye(n));
    if p == 0, break; end
end
if p > 0
    nlml = 1e10;
    dnlml = zeros(numel(hyp.cov) + hasMean + 1, 1);
    return
end

alpha = L\(L'\e);             % 注意:chol 返回上三角 R,K = R'R,故 K^{-1}e = R\(R'\e)
nlml  = 0.5*(e'*alpha) + sum(log(diag(L))) + 0.5*n*log(2*pi);

Kinv = L\(L'\eye(n));
A    = Kinv - alpha*alpha';   % dnlml/dθ = 0.5 * sum(A ⊙ dK/dθ)

ncov  = numel(hyp.cov);
dnlml = zeros(ncov + hasMean + 1, 1);
for ii = 1:ncov
    [~, dK] = feval(covfn, hyp.cov, X, [], ii);
    dnlml(ii) = 0.5*sum(A(:).*dK(:));
end
k0 = ncov;
if hasMean
    dnlml(k0+1) = -sum(alpha);               % d/d 常数均值
    k0 = k0 + 1;
end
dnlml(k0+1) = exp(2*hyp.lik) * trace(A);     % d/d log sn
end
