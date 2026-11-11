function [pi_fn, pi_info] = construct_pi_phys(type, params)
    eps_floor = 0.01;
    switch lower(type)
        case 'region'
            regions = {};
            if isfield(params,'regions')
                regions = params.regions;
                if ~iscell(regions), regions = {regions}; end
            end
            if isfield(params,'boxes')
                boxes = params.boxes;
                for b = 1:size(boxes,1)
                    regions{end+1} = [boxes(b,1),boxes(b,3); boxes(b,2),boxes(b,3); ...
                                      boxes(b,2),boxes(b,4); boxes(b,1),boxes(b,4)];
                end
            end
            if isempty(regions), error('construct_pi_phys/region: 需提供 params.regions 或 params.boxes'); end
            edge_width = 0.05; if isfield(params,'edge_width'), edge_width = params.edge_width; end
            mode      = 'soft'; if isfield(params,'mode'),      mode      = params.mode;      end
            pi_fn = @(X) region_prior(X, regions, edge_width, mode, eps_floor);
            pi_info = struct('type','region','n_regions',numel(regions), ...
                             'edge_width',edge_width,'mode',mode,'floor',eps_floor);
        case 'gaussian'
            centers = params.centers;
            k = size(centers, 1);
            if isfield(params, 'scales')
                scales = params.scales;
                if isscalar(scales), scales = scales * ones(k,1); end
            else
                scales = 0.15 * ones(k,1);
            end
            if isfield(params, 'weights')
                weights = params.weights(:);
                weights = weights / sum(weights);
            else
                weights = ones(k,1) / k;
            end
            pi_fn = @(X) local_gaussian_prior(X, centers, scales, weights, eps_floor);
            pi_info = struct('type','gaussian', 'centers',centers, ...
                             'scales',scales, 'weights',weights, 'floor',eps_floor);
        case 'boundary'
            if ~isfield(params,'g') || ~isfield(params,'thr')
                error('construct_pi_phys/boundary: 需提供 params.g(水平函数)与 params.thr(阈值)');
            end
            g_fn = params.g; thr = params.thr;
            sigma = 0.4;  if isfield(params,'sigma'),      sigma      = params.sigma;      end
            regions = {};
            if isfield(params,'regions')
                regions = params.regions;
                if ~iscell(regions), regions = {regions}; end
            end
            if isfield(params,'boxes')
                bs = params.boxes;
                for b = 1:size(bs,1)
                    regions{end+1} = [bs(b,1),bs(b,3); bs(b,2),bs(b,3); ...
                                      bs(b,2),bs(b,4); bs(b,1),bs(b,4)];
                end
            end
            if isempty(regions), error('construct_pi_phys/boundary: 需提供 params.regions 或 params.boxes'); end
            edge_width = 0.05; if isfield(params,'edge_width'), edge_width = params.edge_width; end
            mode       = 'soft'; if isfield(params,'mode'),     mode       = params.mode;     end
            pi_fn = @(X) boundary_region_prior(X, g_fn, thr, sigma, regions, edge_width, mode, eps_floor);
            pi_info = struct('type','boundary','thr',thr,'sigma',sigma, ...
                             'n_regions',numel(regions),'edge_width',edge_width, ...
                             'mode',mode,'floor',eps_floor);
        case 'multifidelity'
            f_high = params.f_high;
            f_low  = params.f_low;
            bounds = params.bounds;
            grid_n = 60;
            if isfield(params,'grid_n'), grid_n = params.grid_n; end
            max_resid = estimate_max_residual(f_high, f_low, bounds, grid_n);
            max_resid = max(max_resid, eps);
            pi_fn = @(X) multifidelity_prior(X, f_high, f_low, max_resid, eps_floor);
            pi_info = struct('type','multifidelity', 'max_resid',max_resid, ...
                             'floor',eps_floor, 'grid_n',grid_n);
        otherwise
            error('construct_pi_phys: 未知先验类型 "%s"（支持 region / gaussian / multifidelity）', type);
    end
end
function val = local_gaussian_prior(X, centers, scales, weights, eps_floor)
    X = reshape(X, [], size(centers,2));
    val = zeros(size(X,1), 1);
    for j = 1:size(centers,1)
        d2 = sum(((X - centers(j,:)) .^ 2) ./ (scales(j)^2), 2);
        val = val + weights(j) * exp(-0.5 * d2);
    end
    val = min(max(val, eps_floor), 1);
end
function val = boundary_region_prior(X, g_fn, thr, sigma, regions, edge_width, mode, eps_floor)
    X = reshape(X, [], 2);
    n = size(X,1);
    ridge = exp(-0.5 * ((g_fn(X) - thr) / sigma).^2);
    cover = zeros(n,1);
    for r = 1:numel(regions)
        poly = regions{r};
        inside = inpolygon(X(:,1), X(:,2), poly(:,1), poly(:,2));
        if strcmpi(mode,'hard')
            score = double(inside);
        else
            d = dist_to_polygon(X, poly);
            score = exp(-0.5*(d/edge_width).^2);
            score(inside) = 1;
        end
        cover = max(cover, score);
    end
    val = min(max(ridge .* cover, eps_floor), 1);
end
function val = multifidelity_prior(X, f_high, f_low, max_resid, eps_floor)
    resid = abs(f_high(X) - f_low(X));
    val = resid / max_resid;
    val = min(max(val, eps_floor), 1);
end
function m = estimate_max_residual(f_high, f_low, bounds, grid_n)
    dim = size(bounds,1);
    n = min(grid_n^dim, 20000);
    U = rand(n, dim);
    for d = 1:dim
        U(:,d) = bounds(d,1) + (bounds(d,2)-bounds(d,1)) * U(:,d);
    end
    resid = abs(f_high(U) - f_low(U));
    m = max(resid);
end
function val = region_prior(X, regions, edge_width, mode, eps_floor)
    X = reshape(X, [], 2);
    if size(X,2) ~= 2
        error('construct_pi_phys/region: 当前仅支持 2D 区域先验（inpolygon）');
    end
    n = size(X,1);
    val = zeros(n,1);
    for r = 1:numel(regions)
        poly = regions{r};
        inside = inpolygon(X(:,1), X(:,2), poly(:,1), poly(:,2));
        if strcmpi(mode,'hard')
            score = double(inside);
        else
            d = dist_to_polygon(X, poly);
            score = exp(-0.5*(d/edge_width).^2);
            score(inside) = 1;
        end
        val = max(val, score);
    end
    val = min(max(val, eps_floor), 1);
end
function d = dist_to_polygon(P, poly)
    n = size(P,1); m = size(poly,1);
    d = inf(n,1);
    for e = 1:m
        ne = e + 1;  if e == m, ne = 1; end
        a = poly(e,:);
        b = poly(ne,:);
        d = min(d, point_seg_dist(P, a, b));
    end
end
function d = point_seg_dist(P, a, b)
    ab = b - a;
    denom = ab * ab' + eps;
    t = (P - a) * ab' / denom;
    t = min(max(t, 0), 1);
    proj = a + t .* ab;
    d = sqrt(sum((P - proj).^2, 2));
end
