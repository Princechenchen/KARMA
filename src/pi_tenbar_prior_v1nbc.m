function [pi_hg, pi_bnd, pri] = pi_tenbar_prior_v1nbc(X)
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
    anchors = struct('dims', {[1 7]}, 'thr', {0.25}, ...
                     'label', {'A(主力杆对1+7)'});
    p = struct('w', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4], ...
               'w_normalized', [2.0 1.6 0.6 0.6 0.4 0.4 1.5 0.4 0.4 0.4]/8.4, ...
               's0', 0.45, 'sigma_b', 0.10, 'ew', 0.08, 'anchors', {anchors}, ...
               'version', 'v1nbc(v1nb去锚C: 仅锚A)');
end
function y = get_hg(X)
    p = local_params();
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    for a = 1:numel(p.anchors)
        dims_a = p.anchors(a).dims;
        thr_a  = p.anchors(a).thr;
        if numel(dims_a) == 2
            d1 = max(X(:, dims_a(1)) - thr_a, 0) / p.ew;
            d2 = max(X(:, dims_a(2)) - thr_a, 0) / p.ew;
            d  = sqrt(d1.^2 + d2.^2);
        else
            d = max(X(:, dims_a) - thr_a, 0) / p.ew;
        end
        y = max(y, max(0.01, exp(-0.5 * d.^2)));
    end
end
function y = get_bnd(X)
    p = local_params();
    s = X * (p.w / sum(p.w))';
    y = max(0.01, exp(-0.5 * ((s - p.s0) / p.sigma_b).^2));
end
