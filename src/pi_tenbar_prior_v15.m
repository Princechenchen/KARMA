function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v15(X)
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
               'w_normalized', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4]/8.3, ...
               's0', 0.43, 'sigma_b', 0.10, ...
               'win_c', 0.10, 'win_sL', 0.055, 'win_sR', 0.11, ...
               'win_dims', [1 7 9], ...
               'version', 'v15(左收不对称带窗 σL=0.055 σR=0.11 + s0=0.43)');
end
function y = get_hg(X)
    p = local_params();
    w1 = asym_win(X(:,1), p);
    w7 = asym_win(X(:,7), p);
    w9 = asym_win(X(:,9), p);
    y = max(max(w1, w7), max(w9, 0.01));
end
function y = asym_win(xj, p)
    s_side = p.win_sL + (p.win_sR - p.win_sL) * double(xj >= p.win_c);
    y = max(0.01, exp(-0.5 * ((xj - p.win_c) ./ s_side).^2));
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
