function model = gp_train(X, y, varargin)
% gp_train  单响应 GP 回归建模:数据标准化 + 多起点极大边际似然 + 预缓存
%
% 用法:
%   model = gp_train(X, y)                                   % 默认 gpCovSEard 核 + 常数均值
%   model = gp_train(X, y, 'cov','gpCovMatern52')            % 换核
%   model = gp_train(X, y, 'nrestart',8, 'seed',3)           % 更多随机重启
%
% 名称-值参数:
%   'cov'      核名或句柄:'gpCovSEard'(默认)|'gpCovSEiso'|'gpCovMatern52'
%   'mean'     均值函数:'const'(默认)|'zero'
%   'nrestart' 随机重启次数,默认 5
%   'seed'     随机种子,默认 1(重启 r 用 seed+r-1)
%   'normalize' 是否对 X/y 做 z 标准化,默认 true
%   'jitter'   Cholesky 抖动,默认 1e-10
%   'verbose'  打印每次重启的 nlml,默认 false
%
% 返回 model(主要字段):
%   .hyp / .hyp_raw   log 超参(标准化空间 / 原始尺度换算后)
%   .nlml / .nlml_raw 负对数边际似然(标准化空间 / 原始 y 尺度)
%   .L, .alpha        预缓存(Cholesky 因子与 K^{-1}(y-m)),供 gp_predict 复用
%   .norm             标准化参数(muX sdX muy sdy)
%   配 gp_predict / gp_loo 使用,详见 README.md

% ---- 名称-值参数 ----
opts = struct('cov','gpCovSEard','mean','const','nrestart',5,'seed',1, ...
              'normalize',true,'jitter',1e-10,'verbose',false);
for k = 1:2:numel(varargin)
    if k+1 <= numel(varargin)
        opts.(varargin{k}) = varargin{k+1};
    end
end

X = X(:,:);  y = y(:);
n = size(X,1);  d = size(X,2);
assert(numel(y) == n, 'gp_train: X 与 y 的样本数不一致');
assert(all(isfinite(X(:))) && all(isfinite(y)), 'gp_train: 输入含 NaN/Inf');

% ---- 标准化 ----
norms.muX = mean(X,1);  norms.sdX = std(X,[],1);
norms.muy = mean(y);    norms.sdy  = std(y);
norms.sdX(norms.sdX < 1e-12) = 1;                 % 常数列保护
if norms.sdy < 1e-12, norms.sdy = 1; end
if opts.normalize
    Xn = (X - norms.muX)./norms.sdX;
    yn = (y - norms.muy)/norms.sdy;
else
    Xn = X;  yn = y;
    norms.muX = zeros(1,d);  norms.sdX = ones(1,d);
    norms.muy = 0;           norms.sdy = 1;
end

% ---- 核与均值 ----
if ischar(opts.cov) || isstring(opts.cov)
    covname = char(opts.cov);  covfn = str2func(covname);
else
    covfn = opts.cov;  covname = func2str(covfn);
end
hasMean = strcmp(opts.mean, 'const');
if strcmp(covname, 'gpCovSEiso')
    nell = 1;  ncov = 2;
else
    nell = d;  ncov = d + 1;                      % ARD 核:[l_1..l_d; sf]
end

% ---- 目标函数(打包顺序 [cov; mean?; lik]) ----
nlZ = @(th) gp_nll(unpackhyp(th, ncov, hasMean), covfn, opts.mean, Xn, yn, opts.jitter);

% ---- 多起点 BFGS ----
nlml_all = nan(opts.nrestart, 1);
best.nlml = inf;  best.th = [];
for r = 1:opts.nrestart
    rng(opts.seed + r - 1);
    th0 = [log(0.3 + rand(nell,1));               % log l(标准化空间,0.3~1.3)
           log(0.7 + 0.6*rand);                   % log sf(yn 方差 ~1)
           zeros(hasMean,1);                      % 常数均值(中心化后 ~0)
           log(0.05 + 0.25*rand)];                % log sn
    [th, fl] = gp_bfgs(nlZ, th0);
    nlml_all(r) = fl;
    if fl < best.nlml, best.nlml = fl;  best.th = th; end
    if opts.verbose
        fprintf('gp_train restart %2d/%d: nlml = %.4f\n', r, opts.nrestart, fl);
    end
end

% ---- 组装模型(用最优超参重建预缓存) ----
hyp = unpackhyp(best.th, ncov, hasMean);
K = feval(covfn, hyp.cov, Xn) + feval(@gpCovNoise, hyp.lik, Xn);
if hasMean, e = yn - hyp.mean; else, e = yn; end
L = chol(K + opts.jitter*eye(n));
alpha = L\(L'\e);               % chol 上三角 R(K = R'R):K^{-1}e = R\(R'\e)

% 原始尺度超参换算:l_raw = l_n*sdX, sf_raw = sf_n*sdy, sn_raw = sn_n*sdy
hyp_raw = hyp;
if nell == 1
    lscale = exp(mean(log(norms.sdX)));           % iso 核用 sdX 几何平均
else
    lscale = norms.sdX(:);
end
hyp_raw.cov(1:nell) = hyp.cov(1:nell) + log(lscale);
hyp_raw.cov(end)    = hyp.cov(end) + log(norms.sdy);
if isfield(hyp,'mean'), hyp_raw.mean = norms.muy + norms.sdy*hyp.mean; end
hyp_raw.lik = hyp.lik + log(norms.sdy);

model = struct('covname',covname,'covfn',covfn,'meanfn',opts.mean, ...
    'hyp',hyp,'hyp_raw',hyp_raw,'norm',norms, ...
    'Xn',Xn,'yn',yn,'y',y,'n',n,'d',d, ...
    'L',L,'alpha',alpha,'nlml',best.nlml, ...
    'nlml_raw',best.nlml + n*log(norms.sdy), ...
    'nlml_restarts',nlml_all,'jitter',opts.jitter,'opts',opts);
end

function hyp = unpackhyp(th, ncov, hasMean)
if hasMean
    hyp = struct('cov',th(1:ncov),'mean',th(ncov+1),'lik',th(ncov+2));
else
    hyp = struct('cov',th(1:ncov),'lik',th(ncov+1));
end
end
