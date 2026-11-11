function centers = high_gradient_centers_nd(f, bounds, k, grid_n, margin, metric)
    if nargin < 3 || isempty(k),      k      = 3;   end
    if nargin < 4 || isempty(grid_n), grid_n = 9;   end
    if nargin < 5 || isempty(margin), margin  = 0.10; end
    if nargin < 6 || isempty(metric), metric  = 'relcurv'; end
    d = size(bounds, 1);
    gv = linspace(0, 1, grid_n);
    h  = gv(2) - gv(1);
    C = cell(1, d);
    [C{:}] = ndgrid(gv);
    n_all = grid_n^d;
    P = zeros(n_all, d);
    for dd = 1:d, P(:, dd) = C{dd}(:); end
    F = reshape(f(P), repmat(grid_n, 1, d));
    Fdd2 = zeros(size(F));
    for dd = 1:d
        order = [dd, setdiff(1:d, dd)];
        inv_order(order) = 1:d;
        Fi  = permute(F, order);
        Fm  = reshape(Fi, grid_n, []);
        D2  = (Fm(3:end,:) - 2*Fm(2:end-1,:) + Fm(1:end-2,:)) / h^2;
        D2p = [D2(1,:); D2; D2(end,:)];
        Fdd2 = Fdd2 + permute(reshape(D2p, size(F)), inv_order).^2;
    end
    score = sqrt(Fdd2);
    if strcmpi(metric, 'relcurv')
        amp = max(abs(F(:)));
        score = score ./ (abs(F) + 0.1*amp + eps);
    end
    inband = true(n_all, 1);
    for dd = 1:d
        inband = inband & P(:,dd) >= bounds(dd,1)+margin & P(:,dd) <= bounds(dd,2)-margin;
    end
    svec = score(:);
    svec(~inband) = -inf;
    min_dist = 0.55;
    centers = pick_diverse(P, svec, k, min_dist);
    if size(centers,1) < k
        [~, si] = sort(svec, 'descend');
        si = si(isfinite(svec(si)));
        centers = P(si(1:min(k, numel(si))), :);
    end
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
