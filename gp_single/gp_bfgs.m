function [x, f, g, info] = gp_bfgs(fun, x0, maxit, gtol)
% gp_bfgs  带 Armijo 回溯线搜索的 BFGS 无约束极小化(自包含,不依赖优化工具箱)
%
%   [x, f, g, info] = gp_bfgs(fun, x0)        % maxit=300, gtol=1e-6
%   fun 接受列向量,返回 [f, g](目标值与梯度),即 gp_nll 的封装

if nargin < 3 || isempty(maxit), maxit = 300; end
if nargin < 4 || isempty(gtol),  gtol  = 1e-6; end

x = x0(:);
[f, g] = fun(x);
p = numel(x);
H = eye(p);
for it = 1:maxit
    if max(abs(g)) < gtol, break; end
    dirc = -H*g;
    if g'*dirc >= -1e-12*norm(g)*norm(dirc)      % 非下降方向 → 重置 H
        H = eye(p);  dirc = -g;
    end
    t = 1;
    while t > 1e-14
        fn = fun(x + t*dirc);
        if isfinite(fn) && fn <= f + 1e-4*t*(g'*dirc), break; end
        t = 0.5*t;
    end
    if t <= 1e-14, break; end                     % 线搜索无法再推进
    xn = x + t*dirc;
    [fn, gn] = fun(xn);
    s = xn - x;  yv = gn - g;
    sy = real(s'*yv);
    if sy > 1e-10*norm(s)*norm(yv)                % 曲率条件满足才更新 H
        rho = 1/sy;
        I = eye(p);
        H = (I - rho*s*yv')*H*(I - rho*yv*s') + rho*(s*s');
    end
    x = xn;  f = fn;  g = gn;
end
info = struct('iter', it, 'gmax', max(abs(g)), 'converged', max(abs(g)) < gtol);
end
