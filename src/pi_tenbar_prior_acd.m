function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_acd(X)
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
    anchors = struct('dims', {[1 7], 9, 1}, 'thr', {0.25, 0.20, 0.15}, ...
                     'label', {'A(主力杆对1+7, 盒)', 'C(杆9小截面带, 盒)', 'D(杆1反比衰减)'});
    p = struct('w', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
               'w_normalized', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4]/8.4, ...
               's0', 0.45, 'sigma_b', 0.10, 'ew', 0.08, 'anchors', {anchors}, ...
               'version', 'acd(锚A+C+D, 无x3通道)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    d1 = max(X(:,1) - 0.25, 0) / p.ew;
    d7 = max(X(:,7) - 0.25, 0) / p.ew;
    y = max(y, max(0.01, exp(-0.5 * (d1.^2 + d7.^2))));
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
