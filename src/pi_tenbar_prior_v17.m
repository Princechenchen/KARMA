function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v17(X)
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
               'win_c', 0.10, 'win_sL', [0.08 0.055 0.08], 'win_sR', 0.09, ...
               'win_dims', [1 7 9], ...
               'version', 'v17(分维左坡 x1/x9 σL=0.08 x7 σL=0.055 σR=0.09 + s0=0.43)');
end
function y = get_hg(X)
    p = local_params();
    y = 0.01;
    for k = 1:numel(p.win_dims)
        y = max(y, asym_win(X(:, p.win_dims(k)), p.win_sL(k), p.win_sR, p.win_c));
    end
end
function y = asym_win(xj, sL, sR, c)
    s_side = sL + (sR - sL) * double(xj >= c);
    y = max(0.01, exp(-0.5 * ((xj - c) ./ s_side).^2));
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
