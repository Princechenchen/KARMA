function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_cd(X)
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
    anchors = struct('dims', {9, 1}, 'thr', {0.20, 0.15}, ...
                     'label', {'C(杆9小截面带, 盒)', 'D(杆1反比衰减)'});
    p = struct('w', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
               'w_normalized', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4]/8.4, ...
               's0', 0.45, 'sigma_b', 0.10, 'ew', 0.08, 'anchors', {anchors}, ...
               'version', 'cd(锚C+D, 无锚A无x3通道)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    dc = max(X(:,9) - 0.20, 0) / p.ew;
    y = max(y, max(0.01, exp(-0.5 * dc.^2)));
    xi1 = max(X(:,1), 0.005);
    pd  = min(1, (0.15 ./ xi1).^2);
    y = max(y, max(0.01, pd));
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
