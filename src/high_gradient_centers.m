function centers = high_gradient_centers(f, bounds, k, grid_n, margin, metric)
    if nargin < 3 || isempty(k),      k      = 2;    end
    if nargin < 4 || isempty(grid_n), grid_n = 120;  end
    if nargin < 5 || isempty(margin), margin  = 0.10; end
    if nargin < 6 || isempty(metric), metric  = 'relcurv'; end
    xs = linspace(bounds(1,1), bounds(1,2), grid_n);
    ys = linspace(bounds(2,1), bounds(2,2), grid_n);
    dx = xs(2)-xs(1); dy = ys(2)-ys(1);
    [X, Y] = meshgrid(xs, ys);
    P = [X(:), Y(:)];
    F = reshape(f(P), grid_n, grid_n);
    switch lower(metric)
        case 'curvature'
            score = hess_norm(F, dx, dy);
        case 'oscillation'
            score = osc_score(F, dx, dy);
        otherwise
            Hn  = hess_norm(F, dx, dy);
            amp = max(abs(F(:)));
            score = Hn ./ (abs(F) + 0.1*amp + eps);
    end
    svec = score(:);
    inband = P(:,1)>=bounds(1,1)+margin & P(:,1)<=bounds(1,2)-margin & ...
             P(:,2)>=bounds(2,1)+margin & P(:,2)<=bounds(2,2)-margin;
    svec(~inband) = -inf;
    lm = local_maxima(score) & inband_grid(X, Y, bounds, margin);
    cand = lm(:);
    cand_pts = P(cand, :);
    cand_sc  = svec(cand);
    min_dist = 0.30;
    centers = pick_diverse(cand_pts, cand_sc, k, min_dist);
    if size(centers,1) < k
        [sv, si] = sort(svec, 'descend');
        si = si(isfinite(sv));
        centers = P(si(1:min(k,numel(si))), :);
    end
end
function H = hess_norm(F, dx, dy)
    [Fx,Fy]  = gradient(F,dx,dy);
    [Fxx,Fxy]= gradient(Fx,dx,dy);
    [~, Fyy] = gradient(Fy,dx,dy);
    H = sqrt(Fxx.^2 + Fyy.^2 + 2*Fxy.^2);
end
function s = osc_score(F, dx, dy)
    [Fx,Fy] = gradient(F,dx,dy);
    gmag = hypot(Fx,Fy);
    gmax = max(gmag(:));
    floorv = gmax*1e-3 + eps;
    ux = Fx./max(gmag, floorv);
    uy = Fy./max(gmag, floorv);
    flat = gmag < gmax*1e-2;
    r = 5;
    ker = ones(2*r+1,2*r+1)/((2*r+1)^2);
    R = hypot(conv2(ux,ker,'same'), conv2(uy,ker,'same'));
    s = 1 - R;
    s(flat) = 0;
end
function lm = local_maxima(F)
    Fm = F; Fm(~isfinite(Fm)) = -inf;
    lm = true(size(F));
    for ay = -1:1
        for ax = -1:1
            if ax==0 && ay==0, continue; end
            lm = lm & (Fm > circshift(Fm, [ay, ax]));
        end
    end
    lm = lm & isfinite(Fm);
end
function m = inband_grid(X, Y, bounds, margin)
    m = X>=bounds(1,1)+margin & X<=bounds(1,2)-margin & ...
        Y>=bounds(2,1)+margin & Y<=bounds(2,2)-margin;
end
function centers = pick_diverse(pts, sc, k, min_dist)
    centers = zeros(0, size(pts,2));
    if isempty(pts), return; end
    [~, ord] = sort(sc, 'descend');
    pts = pts(ord,:);
    remaining = true(size(pts,1),1);
    for ki = 1:k
        idx = find(remaining, 1, 'first');
        if isempty(idx), break; end
        centers(end+1,:) = pts(idx,:);
        d2 = sum((pts - pts(idx,:)).^2, 2);
        remaining(d2 < min_dist^2) = false;
    end
end
