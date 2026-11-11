function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v13(X)
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
               'win_c', 0.10, 'win_s', 0.08, ...
               'win_dims', [1 7 9], ...
               'version', 'v13(三杆带窗并集 c=0.10 σ=0.08)');
end
function y = get_hg(X)
    p = local_params();
    w1 = max(0.01, exp(-0.5 * ((X(:,1) - p.win_c) / p.win_s).^2));
    w7 = max(0.01, exp(-0.5 * ((X(:,7) - p.win_c) / p.win_s).^2));
    w9 = max(0.01, exp(-0.5 * ((X(:,9) - p.win_c) / p.win_s).^2));
    y = max(max(w1, w7), max(w9, 0.01));
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
