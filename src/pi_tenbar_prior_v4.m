function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v4(X)
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
        'dims_v1',  [1 7 3 9], ...
        'thr_v1',   [0.25 0.25 0.20 0.20], ...
        'ew_v1',    0.08, ...
        'dims_DE',  [1 3], ...
        'thr_DE',   [0.15 0.15], ...
        'w',        [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
        's0',       0.45, ...
        'sigma_b',  0.10, ...
        'version','v4(v1 3锚+新增D/E反比衰减锚=5锚max并集)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    d1 = max(X(:,1) - p.thr_v1(1), 0) / p.ew_v1;
    d7 = max(X(:,7) - p.thr_v1(2), 0) / p.ew_v1;
    y = max(y, max(0.01, exp(-0.5*(d1.^2 + d7.^2))));
    d = max(X(:,3) - p.thr_v1(3), 0) / p.ew_v1;
    y = max(y, max(0.01, exp(-0.5*d.^2)));
    d = max(X(:,9) - p.thr_v1(4), 0) / p.ew_v1;
    y = max(y, max(0.01, exp(-0.5*d.^2)));
    xi1 = max(X(:,1), 0.005);
    pi_d = min(1, (p.thr_DE(1) ./ xi1).^2);
    y = max(y, max(0.01, pi_d));
    xi3 = max(X(:,3), 0.005);
    pi_e = min(1, (p.thr_DE(2) ./ xi3).^2);
    y = max(y, max(0.01, pi_e));
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
