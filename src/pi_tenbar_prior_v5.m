function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v5(X)
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
        'dims_v1',  [1 7 9], ...
        'thr_v1',   [0.25 0.25 0.20], ...
        'ew_v1',    0.08, ...
        'dims_DE',  1, ...
        'thr_DE',   0.15, ...
        'e5_dim',   3, ...
        'e5_t0',    0.15, ...
        'w',        [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
        's0',       0.45, ...
        'sigma_b',  0.10, ...
        'version','v5(v4去锚B/E, x3通道换对数衰减t0=0.15)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    d1 = max(X(:,1) - p.thr_v1(1), 0) / p.ew_v1;
    d7 = max(X(:,7) - p.thr_v1(2), 0) / p.ew_v1;
    y = max(y, max(0.01, exp(-0.5*(d1.^2 + d7.^2))));
    d = max(X(:,9) - p.thr_v1(3), 0) / p.ew_v1;
    y = max(y, max(0.01, exp(-0.5*d.^2)));
    xi1 = max(X(:,1), 0.005);
    pi_d = min(1, (p.thr_DE ./ xi1).^2);
    y = max(y, max(0.01, pi_d));
    xr = max(X(:, p.e5_dim), 1e-3);
    pi_e5 = max(0.01, min(1, 1 - log(xr / p.e5_t0) / log(1 / p.e5_t0)));
    y = max(y, pi_e5);
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
