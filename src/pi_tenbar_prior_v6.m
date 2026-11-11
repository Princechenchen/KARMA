function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v6(X)
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
    p = struct('w', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
               'w_normalized', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4]/8.4, ...
               's0', 0.45, 'sigma_b', 0.10, ...
               'ridge_dims', [1 3 7 9], 'ridge_t', 0.20, ...
               'version', 'v6(四响应维1/x宽脊并集, t=0.20)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    for jj = 1:numel(p.ridge_dims)
        dj = p.ridge_dims(jj);
        gj = min(1, p.ridge_t ./ max(X(:, dj), 0.01));
        y  = max(y, gj);
    end
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
