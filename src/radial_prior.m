function [pri, prof] = radial_prior(fy, g_of_y, opts)
    if nargin < 3 || isempty(opts), opts = struct(); end
    d       = getdef(opts, 'dim', 6);
    grid_n  = getdef(opts, 'grid_n', 9);
    nb      = getdef(opts, 'n_bins', 40);
    g_thr   = getdef(opts, 'g_thr', 35);
    pivot   = getdef(opts, 'pivot', zeros(1, d));
    gfrac   = getdef(opts, 'grad_frac', 0.15);
    tau_min = getdef(opts, 'tau_min', 0.05);
    tau_max = getdef(opts, 'tau_max', 0.15);
    taug_min= getdef(opts, 'taug_min', 0.03);
    taug_max= getdef(opts, 'taug_max', 0.20);
    gv = linspace(0, 1, grid_n);
    C = cell(1, d); [C{:}] = ndgrid(gv);
    n_all = grid_n^d;
    P = zeros(n_all, d);
    for dd = 1:d, P(:, dd) = C{dd}(:); end
    rho = vecnorm(P - pivot, 2, 2) / sqrt(d);
    Y = fy(P);
    nRC = size(Y, 2);
    G = g_of_y(Y);
    edges = unique(quantile(rho, linspace(0, 1, nb+1)));
    edges(1) = -inf; edges(end) = inf;
    nb2 = numel(edges) - 1;
    b  = discretize(rho, edges);
    rb  = accumarray(b, rho, [nb2 1], @mean);
    ym  = zeros(nb2, nRC);
    for j = 1:nRC
        ym(:, j) = accumarray(b, Y(:, j), [nb2 1], @mean);
    end
    pb = accumarray(b, double(G >= g_thr), [nb2 1], @mean);
    dr = diff(rb);
    Gs = zeros(nb2-1, 1);
    for j = 1:nRC
        dj = abs(diff(movmean(ym(:, j), 3)) ./ dr);
        dj = dj / max(dj);
        Gs = max(Gs, dj);
    end
    Gs = movmean(Gs, 3);
    [~, it] = max(Gs);
    r_t = rb(it);
    half = find(Gs >= 0.5*max(Gs));
    tau_t = min(max((rb(half(end)) - rb(half(1)))/2, tau_min), tau_max);
    outb = find(rb(1:end-1) > r_t & Gs < gfrac*max(Gs), 1);
    if isempty(outb), r_f = rb(end); else, r_f = rb(outb); end
    ps = movmean(pb, 3);
    ig = find(ps >= 0.5, 1);
    if isempty(ig)
        r_g = quantile(rho(G >= g_thr), 0.5);
    else
        r_g = rb(ig);
    end
    dp = max(abs(diff(ps) ./ dr));
    if dp > 0, tau_g = 1/(4*dp); else, tau_g = taug_min; end
    tau_g = min(max(tau_g, taug_min), taug_max);
    q = @(X) vecnorm(X - pivot, 2, 2) / sqrt(d);
    pri.rl = @(X) min(max(1./(1 + exp(-(q(X) - r_t)/tau_t)), 0.01), 1);
    pri.rc = @(X) min(max(exp(-(q(X) - r_g).^2 / (2*tau_g^2)), 0.01), 1);
    prof = struct('r_t',r_t, 'tau_t',tau_t, 'r_f',r_f, 'r_g',r_g, 'tau_g',tau_g, ...
                  'pivot',pivot, 'grid_n',grid_n, 'n_bins',nb2, 'g_thr',g_thr, ...
                  'rb',rb, 'ymean',ym, 'p',pb, 'grad',Gs, ...
                  'frac_cls',mean(G >= g_thr), ...
                  'cov_rl',mean(pri.rl(P) > 0.5), 'cov_rc',mean(pri.rc(P) > 0.5));
end
function v = getdef(s, f, dv)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = dv; end
end
