function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v3(X)
    if nargin < 1
        pi_hg  = @(Xq) get_hg(Xq);
        pi_bnd = @(Xq) get_bnd(Xq);
        pri    = local_params();
        return
    end
    pi_hg  = get_hg(X);
    pi_bnd = get_bnd(X);
    pri    = local_params();
end
function p = local_params()
    p = struct( ...
        'dims',   [1 7 3 9], ...
        'thr',    [0.25 0.25 0.20 0.20], ...
        'ew',     0.08, ...
        'w',      [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
        's0',     0.45, ...
        'sigma_b',0.10, ...
        'version','v3(split dual anchor, single-variable change from v1)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    for a = 1:numel(p.dims)
        d = max(X(:, p.dims(a)) - p.thr(a), 0) / p.ew;
        y = max(y, max(0.01, exp(-0.5 * d.^2)));
    end
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
