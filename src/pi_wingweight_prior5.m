function [pi_hg, pi_bnd, pri] = pi_wingweight_prior5(X)
    if nargin >= 1 && ~isempty(X)
        p = local_params();
        pi_hg  = get_hg(X, p);
        pi_bnd = get_bnd(X, p);
        pri = [];
        return;
    end
    p = local_params();
    pi_hg  = @(Xq) get_hg(Xq, p);
    pi_bnd = @(Xq) get_bnd(Xq, p);
    pri = p;
end
function p = local_params()
    anchors = struct('dims', {10, 5}, 'thr', {0.25, 0.75}, 'side', {+1, -1}, ...
                     'label', {'Wp低带(x<=0.25, 物理Wp<=0.0388; 加性项交互盲区)', ...
                               'q高带(x>=0.75, 物理q>=37.75; 平坦维ARD可辨识性)'});
    p = struct('ew', 0.08, 'thr1', 250, 'thr2', 50, ...
               'sigma1', 0.025, 'sigma2', 0.027, 'anchors', {anchors});
    p.margins = @(Z) get_margins(Z);
end
function y = get_hg(X, p)
    n = size(X, 1);
    y = 0.01 * ones(n, 1);
    for a = 1:numel(p.anchors)
        d = max(p.anchors(a).side * (X(:, p.anchors(a).dims) - p.anchors(a).thr), 0) / p.ew;
        y = max(y, max(0.01, exp(-0.5 * d.^2)));
    end
end
function y = get_bnd(X, p)
    E = get_margins(X);
    y = max(0.01, max(exp(-0.5 * (E(:,1) / p.sigma1).^2), ...
                      exp(-0.5 * (E(:,2) / p.sigma2).^2)));
end
function E = get_margins(X)
    [m1, m2, swp] = ww_monomials(X);
    p = local_params();
    E = [log(m1) - log(p.thr1 - swp), log(m2) - log(p.thr2 + swp)];
end
function [m1, m2, swp] = ww_monomials(X)
    Sw  = 150  + 50  * X(:,1);    Wfw = 220 + 80  * X(:,2);
    A   = 6    + 4   * X(:,3);    Lam = (20 * X(:,4) - 10) * pi/180;
    q   = 16   + 29  * X(:,5);    lam = 0.5 + 0.5 * X(:,6);
    tc  = 0.08 + 0.1 * X(:,7);    Nz  = 2.5 + 3.5 * X(:,8);
    Wdg = 1700 + 800 * X(:,9);    Wp  = 0.025 + 0.055 * X(:,10);
    c = cos(Lam);
    m1 = 0.036 * Sw.^0.758 .* Wfw.^0.0035 .* (A./c.^2).^0.6 .* q.^0.006 ...
              .* lam.^0.04 .* (100*tc./c).^(-0.3) .* (Nz.*Wdg).^0.49;
    m2 = 0.038 * Sw.^0.6 .* Wfw.^0.0035 .* (A./c.^2).^0.5 .* q.^0.006 ...
              .* lam.^0.05 .* (100*tc./c).^(-0.5) .* (Nz.*Wdg).^0.5;
    swp = Sw .* Wp;
end
