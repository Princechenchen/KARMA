function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v12(X)
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
               'ew', 0.08, ...
               'anchorA_thr', [0.25 0.25], ...
               'anchorC_thr', 0.20, ...
               'anchorD_t', 0.15, ...
               'x7win_c', 0.10, 'x7win_s', 0.07, ...
               'version', 'v12(acd + x7 带窗 c=0.10 σ=0.07)');
end
function y = get_hg(X)
    p = local_params();
    d1 = max(X(:,1) - p.anchorA_thr(1), 0) / p.ew;
    d7 = max(X(:,7) - p.anchorA_thr(2), 0) / p.ew;
    aA = max(0.01, exp(-0.5 * (d1.^2 + d7.^2)));
    aC = max(0.01, exp(-0.5 * (max(X(:,9) - p.anchorC_thr, 0) / p.ew).^2));
    aD = min(1, (p.anchorD_t ./ max(X(:,1), 0.005)).^2);
    aW7 = max(0.01, exp(-0.5 * ((X(:,7) - p.x7win_c) / p.x7win_s).^2));
    y = max(max(aA, aC), max(max(aD, aW7), 0.01));
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
